"""Resumable layout/bridge training. Defaults are the RTX 3070 starter runs."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import (PaleoRaster, layout_target, random_rotation, manifest_id, atomic_json,
                   node_weights, layout_geometry, VERSION)
from .layout import LayoutDenoiser, layout_loss
from .progress import TrainingConsole
from .adaptation import PROFILE, MESO_PROFILE, STRUCTURE_PROFILE, recipe_for, augment_triplet, paired_degradation
from .run_lock import training_lock
from toy_meso.common import (Normalizer, autocast, build_model, compute_stats,
                             pick_device, sample_patches)
from toy_meso.model import edm_loss, heun_sample
from toy_meso.train import augment, validation_loss, save_preview


class LayoutData:
    def __init__(self, folder, seed):
        folder = Path(folder)
        self.meta = json.loads((folder / "manifest.json").read_text())
        if self.meta.get("kind") != "layout" or self.meta.get("version") != VERSION:
            raise ValueError("Expected a prepared layout dataset")
        self.maps = np.load(folder / "maps.npy", mmap_mode="r")
        if self.maps.shape != (len(self.meta["records"]), 181, 361):
            raise ValueError("Layout maps disagree with the manifest")
        self.rng = np.random.default_rng(seed)
        self.splits = {s: np.array([i for i, r in enumerate(self.meta["records"]) if r["split"] == s])
                       for s in ("train", "val", "test")}
        if any(len(v) == 0 for v in self.splits.values()):
            raise ValueError("Layout needs nonempty train, validation and test ages")
        self.n = self.meta["face_intervals"]

    def batch(self, count, split="train", rng=None):
        rng = self.rng if rng is None else rng
        return np.stack([layout_target(PaleoRaster(self.maps[i]), random_rotation(rng),
                                       n=self.n, sub=self.meta["supersample"])
                         for i in rng.choice(self.splits[split], count)])

    @torch.no_grad()
    def cuda_batch(self, count, grid, stats, device):
        """Integrate the same rotated spherical footprints directly on CUDA.

        Age choices and rotations consume the same NumPy RNG as CPU sampling,
        so checkpoint resume and the sampling distribution remain compatible.
        Only the quadrature arithmetic switches to float32 on the GPU.
        """
        import torch.nn.functional as F
        if not hasattr(self, "_cuda_maps"):
            self._cuda_maps = torch.from_numpy(np.array(self.maps)).to(device)[:, None]
            points, weights = layout_geometry(self.n, self.meta["supersample"])
            self._cuda_points = torch.from_numpy(points.reshape(-1, 3).astype(np.float32)).to(device)
            self._cuda_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
        indices = self.rng.choice(self.splits["train"], count)
        rotations = np.stack([random_rotation(self.rng) for _ in range(count)]).astype(np.float32)
        rotation = torch.from_numpy(rotations).to(device)
        points = self._cuda_points[None] @ rotation
        latitude = torch.asin(points[..., 2].clamp(-1, 1)) * (180 / np.pi)
        longitude = torch.atan2(points[..., 1], points[..., 0]) * (180 / np.pi)
        # align_corners=True maps exactly onto the source's POINT nodes.
        # The 361st column is the duplicate dateline, so interpolation wraps.
        lookup = torch.stack((((longitude + 180) % 360) / 180 - 1,
                              -latitude / 90), dim=-1)[:, None]
        maps = self._cuda_maps[torch.from_numpy(indices).to(device)]
        sampled = F.grid_sample(maps, lookup, mode="bilinear", padding_mode="border", align_corners=True)
        sub, side = self.meta["supersample"], self.n + 1
        sampled = sampled.reshape(count, 6, side, side, sub, sub)
        metres = (sampled * self._cuda_weights).sum(dim=(-2, -1)) / self._cuda_weights.sum(dim=(-2, -1))
        padded = metres.new_zeros((count, 6, grid.side, grid.side))
        padded[..., :side, :side] = metres
        metres = grid.project(padded)
        encoded = metres.sign() * metres.abs().sqrt()
        return (encoded - stats["mean"]) / stats["std"] * .5

    def stats(self):
        # All training ages, in source orientation; test/val never enter statistics.
        w = node_weights(self.n)
        mean = second = 0.
        for i in self.splits["train"]:
            t = layout_target(PaleoRaster(self.maps[i]), np.eye(3), self.n, self.meta["supersample"])
            mean += float((t * w).sum() / w.sum())
            second += float((t*t * w).sum() / w.sum())
        mean /= len(self.splits["train"])
        second /= len(self.splits["train"])
        return {"mean": mean, "std": max(1e-6, (second - mean*mean)**.5)}

    def validation(self):
        rng = np.random.default_rng(2026)
        return np.stack([layout_target(PaleoRaster(self.maps[i]), random_rotation(rng),
                                       n=self.n, sub=self.meta["supersample"])
                         for i in self.splits["val"]])


class BridgeData:
    def __init__(self, folder, seed):
        folder = Path(folder)
        self.meta = json.loads((folder / "manifest.json").read_text())
        if self.meta.get("kind") != "bridge" or self.meta.get("version") != VERSION:
            raise ValueError("Expected a prepared intermediate dataset")
        array_folder = folder
        if self.meta.get("array_source"):
            array_folder = Path(self.meta["array_source"])
            source_meta = json.loads((array_folder/"manifest.json").read_text())
            if manifest_id(source_meta) != self.meta["array_source_dataset_id"]:
                raise ValueError("Referenced training dataset changed; prepare a new experiment")
            source_done = json.loads((array_folder/"progress.json").read_text())["done"]
            if any(source_done[s] != source_meta[f"{s}_count"] for s in ("train", "val")):
                raise ValueError("Referenced training dataset is incomplete")
            for name, info in self.meta["array_fingerprints"].items():
                stat = (array_folder/name).stat()
                if info != {"bytes":stat.st_size, "mtime_ns":stat.st_mtime_ns}:
                    raise ValueError(f"Referenced training array changed: {name}")
        done = json.loads((folder / "progress.json").read_text())["done"]
        if any(done[s] != self.meta[f"{s}_count"] for s in ("train", "val")):
            raise ValueError("Dataset preparation is incomplete. Rerun Prepare Planet Training.cmd first.")
        self.arrays = {f"{s}_{key}": np.load(array_folder / f"{s}_{key}.npy", mmap_mode="r")
                       for s in ("train", "val") for key in ("target", "cond")}
        for s in ("train", "val"):
            size, factor, count = self.meta.get("training_crop_size", self.meta["size"]), self.meta["factor"], self.meta[f"{s}_count"]
            if (self.arrays[f"{s}_target"].shape != (count, size, size)
                    or self.arrays[f"{s}_cond"].shape != (count, size//factor+2, size//factor+2)):
                raise ValueError("Intermediate arrays disagree with the manifest")
        if self.meta.get("guide_recipe"):
            expected_recipe = recipe_for(self.meta["guide_recipe"]["profile"])
            if self.meta["guide_recipe"] != expected_recipe:
                raise ValueError("Unsupported or changed guide degradation recipe")
            for s in ("train", "val"):
                self.arrays[f"{s}_context"] = np.load(array_folder/f"{s}_context.npy", mmap_mode="r")
                side = self.meta.get("training_crop_size", self.meta["size"])//self.meta["factor"]+2+2*expected_recipe["extra_coarse_halo"]
                expected = (self.meta[f"{s}_count"], side, side)
                if self.arrays[f"{s}_context"].shape != expected:
                    raise ValueError("Intermediate guide context disagrees with the manifest")
        self.rng = np.random.default_rng(seed)

    def batch(self, count, split="train", rng=None, include_context=False):
        rng = self.rng if rng is None else rng
        idx = rng.integers(0, self.meta[f"{split}_count"], count)
        return tuple(torch.from_numpy(np.array(self.arrays[f"{split}_{k}"][idx], dtype=np.float32))[:, None]
                     for k in (("target", "cond", "context") if include_context else ("target", "cond")))

    def stats(self):
        # Bound RAM usage while retaining training-only normalization.
        return compute_stats(self.arrays["train_target"], self.arrays["train_cond"],
                             self.meta["factor"], sample=512)


def padded_target(values, model, stats, device):
    core = torch.from_numpy(values).to(device)
    x = core.new_zeros((len(core), 6, model.grid.side, model.grid.side))
    x[..., :model.grid.n+1, :model.grid.n+1] = (core - stats["mean"]) / stats["std"] * .5
    return model.grid.project(x)


@torch.no_grad()
def sample_layout(model, stats, steps=24, seed=0, device="cpu", count=1):
    g = torch.Generator(device=device).manual_seed(seed)
    def denoise(x, sigma):
        return model(x.float(), sigma.float().expand(len(x))).double()
    initial = model.grid.noise(count, device, g)
    with autocast(device):
        x = heun_sample(denoise, initial.shape, steps, device=device, initial=initial)
    x = model.grid.project(x.float())[..., :model.grid.n+1, :model.grid.n+1]
    return (x / .5 * stats["std"] + stats["mean"]).cpu().numpy()


def save_layout_preview(path, generated, real=None, label="generated globe"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planet_diffusion import cube
    from toy_meso.sphere import decode
    rows = [(label, a) for a in generated]
    if real is not None:
        rows.append(("held-out reconstruction", real[0]))
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 3*len(rows)), squeeze=False)
    for ax, (label, a) in zip(axs[:, 0], rows):
        # Interpolate in metres so nonlinear encoding does not move sea level.
        image = cube.to_equirectangular(decode(a)[None], 128)[0]
        ax.imshow(image, cmap="terrain", vmin=-6500, vmax=6000, extent=(-180, 180, -90, 90))
        ax.contour(image > 0, levels=[.5], colors="black", linewidths=.3,
                   extent=(-180, 180, -90, 90), origin="upper")
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_prepared_preview(stage, dataset, path):
    if stage == "layout":
        a = dataset.batch(2, "train", np.random.default_rng(200))
        v = dataset.batch(1, "val", np.random.default_rng(201))
        save_layout_preview(path, a, v, label="training reconstruction")
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from toy_meso.model import upsample_condition
        from toy_meso.sphere import decode
        t, c = dataset.batch(4, "train", np.random.default_rng(200))
        up = upsample_condition(c, dataset.meta["factor"])
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for row, (title, arr) in enumerate((("161 km input", up), ("23 km target", t))):
            for i in range(4):
                axes[row, i].imshow(decode(arr[i, 0].numpy()), cmap="terrain", vmin=-6000, vmax=6000)
                axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
            axes[row, 0].set_ylabel(title)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("layout", "bridge"))
    p.add_argument("--data", help="prepared dataset folder")
    p.add_argument("--run", help="new or resumable training folder")
    p.add_argument("--steps", type=int, help="total optimizer updates, including resumed updates")
    p.add_argument("--batch", type=int)
    p.add_argument("--ch", type=int)
    p.add_argument("--mults", default="1,2,4")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--dropout", type=float)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=.999)
    p.add_argument("--cond-noise", type=float, default=.03, help="bridge condition noise, fraction of encoded condition std")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--preview-steps", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-only", action="store_true", help="validate data and save target previews, without training")
    p.add_argument("--init-from", help="warm-start a new experiment from checkpoint EMA weights; fresh optimizer")
    p.add_argument("--guide-profile", choices=("clean", PROFILE, MESO_PROFILE, STRUCTURE_PROFILE), default="clean")
    p.add_argument("--validation-patience", type=int, default=0, help="stop after this many evaluations without a new best; 0 disables")
    a = p.parse_args(argv)
    layout = a.stage == "layout"
    a.data = a.data or f"outputs/planet_training/data/{a.stage}"
    a.run = a.run or f"outputs/{'continents' if layout else 'intermediate'}/run1"
    a.steps = a.steps if a.steps is not None else (30000 if layout else 100000)
    a.batch = a.batch if a.batch is not None else 16
    a.ch = a.ch if a.ch is not None else (32 if layout else 64)
    a.dropout = a.dropout if a.dropout is not None else (.1 if layout else 0.)
    for name in ("steps", "batch", "ch", "blocks", "warmup", "eval_every", "save_every", "log_every", "threads"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.preview_steps < 2 or a.lr <= 0 or not 0 <= a.dropout < 1 or not 0 < a.ema < 1 or a.cond_noise < 0:
        p.error("Invalid sampler, learning rate, dropout, EMA or condition noise setting")
    if a.validation_patience < 0 or (a.stage == "layout" and a.guide_profile != "clean"):
        p.error("Invalid patience or guide adaptation option for layout training")
    return a


def main(argv=None):
    a = parse_args(argv)
    with training_lock(a.run):
        return train(a)


def train(a):
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device(a.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    dataset = (LayoutData if a.stage == "layout" else BridgeData)(a.data, a.seed)
    adapted = a.guide_profile != "clean"
    if a.stage == "bridge" and bool(dataset.meta.get("guide_recipe")) != adapted:
        raise ValueError("Adapted data requires its matching --guide-profile; clean data requires clean profile")
    recipe = recipe_for(a.guide_profile) if adapted else None
    if adapted and dataset.meta["guide_recipe"] != recipe:
        raise ValueError("Guide profile does not match the prepared dataset")
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = manifest_id(dataset.meta)
    if a.check_only:
        save_prepared_preview(a.stage, dataset, run / "training-targets.png")
        print(f"{a.stage}: data valid; saved {run / 'training-targets.png'}", flush=True)
        return
    latest = run / "latest.pt"
    ck = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if ck and (ck.get("training_version") != VERSION or ck.get("dataset_id") != identity
               or ck.get("stage") != a.stage):
        raise ValueError("Checkpoint belongs to a different dataset/stage; use a new --run folder")
    if ck and ck.get("guide_profile", "clean") != a.guide_profile:
        raise ValueError("Resume with the checkpoint's original --guide-profile")
    optimization = {key:getattr(a,key) for key in ("lr", "warmup", "ema", "cond_noise", "seed")}
    if ck and ck.get("optimization") is not None and ck["optimization"] != optimization:
        raise ValueError("Resume with the original learning rate, warmup, EMA, condition noise and seed")
    initial = None
    if not ck and a.init_from:
        initial = torch.load(a.init_from, map_location="cpu", weights_only=False)
        legacy = (a.stage=="bridge" and dataset.meta.get("legacy_parent_meta_id")
                  and not initial.get("training_version") and not initial.get("stage")
                  and manifest_id(initial["meta"])==dataset.meta["legacy_parent_meta_id"])
        scale_keys = ("face_intervals", "spacing_m", "radius_m") if a.stage == "layout" else ("size", "factor", "spacing_m")
        if ((not legacy and (initial.get("training_version") != VERSION or initial.get("stage") != a.stage))
                or any(initial["meta"].get(key) != dataset.meta.get(key) for key in scale_keys)
                or (not legacy and initial.get("dataset_id") != dataset.meta.get("parent_dataset_id", identity))):
            raise ValueError("Warm-start checkpoint does not match the source dataset and scale")
    config = {"ch": a.ch, "mults": [int(v) for v in a.mults.split(",")],
              "blocks": a.blocks, "dropout": a.dropout}
    if not config["mults"] or any(v < 1 for v in config["mults"]):
        raise ValueError("Channel multipliers must be positive")
    if ck or initial:
        reference = ck or initial
        config, stats = reference["config"], reference["stats"]
    if ck:
        if a.cond_noise != ck.get("cond_noise", .03):
            raise ValueError("Resume with the original --cond-noise setting")
    elif not initial:
        print("Computing normalization from training data only...", flush=True)
        stats = dataset.stats()
    if not all(np.isfinite(v) for v in stats.values()) or min(v for k,v in stats.items() if "std" in k) < 1e-6:
        raise ValueError("Dataset has invalid or degenerate normalization statistics")
    if a.stage == "layout":
        model = LayoutDenoiser(n=dataset.n, **config).to(device)
        norm = None
        val_raw = dataset.validation()
        vt = padded_target(val_raw, model, stats, device)
        vc = None
    else:
        if dataset.meta.get("training_crop_size", dataset.meta["size"]) % (2 ** (len(config["mults"])-1)):
            raise ValueError("Intermediate patch size is incompatible with U-Net depth")
        model = build_model(config).to(device)
        norm = Normalizer(stats, dataset.meta["factor"])
        vt, vc = (torch.from_numpy(np.array(dataset.arrays[f"val_{key}"], dtype=np.float32))[:, None]
                  for key in ("target", "cond"))
        vt, vc = vt.to(device), vc.to(device)
        clean_vt, clean_vc = vt, vc
        if adapted:
            context = torch.from_numpy(np.array(dataset.arrays["val_context"], dtype=np.float32))[:, None].to(device)
            vt, vc = paired_degradation(norm, vt, vc, context, torch.Generator(device=device).manual_seed(2026), recipe=recipe)
            del context
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(.9, .99))
    step, best = 0, float("inf")
    if ck:
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck["best_val_loss"]
        if ck.get("validation_protocol") != "entire-heldout-set-v1":
            best = float("inf")
        dataset.rng.bit_generator.state = ck["data_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if str(device).startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    elif initial:
        model.load_state_dict(initial["ema"]); ema.load_state_dict(initial["ema"])
    initialized_from = ck.get("initialized_from") if ck else (
        {"path": str(Path(a.init_from).resolve()), "step": initial["step"]} if initial else None)
    del initial
    atomic_json(run / "launch.json", dict(vars(a), dataset_id=identity, config=config, stats=stats))
    console = TrainingConsole(a.steps)
    console.start("meso" if a.guide_profile == MESO_PROFILE else a.stage, step, a.batch, device, sum(p.numel() for p in model.parameters()), run)
    if adapted:
        console.message(f"Guides: 25% clean, 75% smoothed. Targets: {recipe['target']}.\n")
    stale = ck.get("stale_evaluations", 0) if ck else 0

    def checkpoint(name):
        obj = {"training_version": VERSION, "stage": a.stage, "dataset_id": identity,
               "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
               "config": config, "stats": stats, "meta": dataset.meta, "best_val_loss": best,
               "cond_noise": a.cond_noise, "data_rng": dataset.rng.bit_generator.state,
               "guide_profile": a.guide_profile, "initialized_from": initialized_from,
               "optimization": optimization,
               "stale_evaluations": stale,
               "validation_protocol": "entire-heldout-set-v1",
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None}
        tmp = run / (name + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(run / name)

    @torch.no_grad()
    def evaluate():
        if a.stage == "bridge":
            return validation_loss(ema, norm, vt, vc, device, seeds=8, chunk=a.batch)
        g = torch.Generator(device=device).manual_seed(0)
        total = 0.
        for sigma in torch.exp(torch.linspace(-3, 1.5, 8, device=device)):
            noisy = vt + ema.grid.noise(len(vt), device, g) * sigma
            error = 0.
            for start in range(0, len(vt), a.batch):
                chunk = noisy[start:start+a.batch]
                with autocast(device):
                    d = ema(chunk, sigma.expand(len(chunk)))
                error += float(ema.grid.weighted_mean((d-vt[start:start+a.batch])**2)) * len(chunk)/len(vt)
            total += error * float((sigma**2+.25)/(sigma*.5)**2)
        return total / 8

    started, losses, logged_step = time.perf_counter(), [], step
    with (run / "log.jsonl").open("a", encoding="utf-8") as log:
        def record(obj):
            console.record(obj)
            log.write(json.dumps(obj) + "\n"); log.flush()
        try:
            if step == 0 and initialized_from:
                console.message("Checking the starting model before fine-tuning...")
                best = evaluate()
                record({"step": 0, "val_loss": best, "new_best": True})
                if adapted:
                    clean_loss = validation_loss(ema, norm, clean_vt, clean_vc, device, seeds=8, chunk=a.batch)
                    record({"step": 0, "clean_val_loss": clean_loss})
                    console.message(f"  Clean-guide validation: {clean_loss:.4f}")
                checkpoint("best.pt"); checkpoint("latest.pt")
                started = time.perf_counter()
            while step < a.steps:
                if a.stage == "layout":
                    if str(device).startswith("cuda"):
                        x = dataset.cuda_batch(a.batch, model.grid, stats, device)
                    else:
                        x = padded_target(dataset.batch(a.batch), model, stats, device)
                    with autocast(device):
                        loss = layout_loss(model, x)
                else:
                    if adapted:
                        t, c, context = augment_triplet(*(v.to(device) for v in dataset.batch(a.batch, include_context=True)))
                        t, c = paired_degradation(norm, t, c, context, recipe=recipe)
                    else:
                        t, c = dataset.batch(a.batch)
                        t, c = augment(t.to(device), c.to(device))
                    # Noisy guides mimic imperfect continent-model inputs.
                    c = c + torch.randn_like(c) * (stats["cond_std"] * a.cond_noise)
                    up, cin = norm.condition(c)
                    x = norm.residual(t, up)
                    with autocast(device):
                        loss = edm_loss(model, x, cin)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became nonfinite; last checkpoint preserved")
                for group in opt.param_groups:
                    group["lr"] = a.lr * min(1., (step+1)/a.warmup)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                step += 1
                decay = min(a.ema, (1+step)/(10+step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1-decay)
                losses.append(loss.item())
                if step % a.log_every == 0:
                    rate = (step-logged_step) / (time.perf_counter()-started)
                    record({"step": step, "loss": float(np.mean(losses)), "steps_per_s": rate,
                            "batch": a.batch, "examples_per_s": rate * a.batch,
                            "remaining_hours": (a.steps-step)/rate/3600})
                    started, losses, logged_step = time.perf_counter(), [], step
                if step % a.save_every == 0:
                    checkpoint("latest.pt")
                    if step % a.eval_every != 0 and step != a.steps:
                        console.message(f"  Checkpoint saved at update {step:,}.")
                if step % a.eval_every == 0 or step == a.steps:
                    console.message(f"\nUpdate {step:,}: checking held-out data...")
                    vl = evaluate()
                    record({"step": step, "val_loss": vl, "new_best": vl < best})
                    if vl < best:
                        best = vl
                        stale = 0
                        checkpoint("best.pt")
                    else:
                        stale += 1
                    if adapted:
                        clean_loss = validation_loss(ema, norm, clean_vt, clean_vc, device, seeds=8, chunk=a.batch)
                        record({"step": step, "clean_val_loss": clean_loss})
                        console.message(f"  Clean-guide validation: {clean_loss:.4f}")
                    checkpoint("latest.pt")
                    console.message("  Checkpoint saved. Generating preview...")
                    if a.stage == "layout":
                        gen = sample_layout(ema, stats, a.preview_steps, seed=7, device=device, count=2)
                        save_layout_preview(run / f"preview-{step:06d}.png", gen, val_raw)
                    else:
                        gen, up = sample_patches(ema, norm, vc[:4], a.preview_steps, seed=7, device=device)
                        save_preview(run / f"preview-{step:06d}.png", vc[:4], vt[:4], gen, up,
                                     real_label=("unchanged Earth" if a.guide_profile==STRUCTURE_PROFILE else
                                                 "real detail over guide" if adapted else "real"))
                    console.message(f"  Preview saved: preview-{step:06d}.png")
                    if str(device).startswith("cuda"):
                        record({"step": step, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
                    started, losses, logged_step = time.perf_counter(), [], step
                    if a.validation_patience and stale >= a.validation_patience:
                        console.message(f"\nStopping after {stale} checks without validation improvement. Best model preserved.")
                        break
        except KeyboardInterrupt:
            console.message("\nStopping and saving checkpoint...")
            checkpoint("latest.pt")
            console.message(f"Stopped at update {step:,}. Checkpoint saved; rerun to resume.")
            return 130
        checkpoint("latest.pt")
    if step == a.steps:
        console.complete(step, run)


if __name__ == "__main__":
    sys.exit(main())
