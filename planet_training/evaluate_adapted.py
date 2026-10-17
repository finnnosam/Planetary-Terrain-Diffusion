"""Compare a continent-adapted bridge with its original on identical held-out guides."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .adaptation import paired_degradation, recipe_for
from .data import atomic_json, manifest_id
from .train import BridgeData
from toy_meso.common import load_checkpoint, pick_device, sample_patches
from toy_meso.train import validation_loss
from toy_meso.sphere import decode
from toy_meso.evaluate import radial_psd, band_ratios, hillshade


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="outputs/intermediate/run2-adapted")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--reference", default="outputs/intermediate/selected-10000/best.pt")
    p.add_argument("--data", default="outputs/planet_training/data/bridge-adapted")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--clean-data", default="outputs/planet_training/data/bridge",
                   help="original complete clean holdout for regression checks")
    a = p.parse_args(argv)
    if a.batch < 1 or a.steps < 2: p.error("batch must be positive and steps must be at least two")
    torch.set_num_threads(4)
    device = pick_device(a.device)
    data = BridgeData(a.data, 0)
    if not data.meta.get("guide_recipe"): raise ValueError("Expected adapted data")
    recipe = recipe_for(data.meta["guide_recipe"]["profile"])
    legacy = bool(data.meta.get("legacy_parent_meta_id"))
    if legacy:
        archive = np.load(a.clean_data)
        clean_meta = json.loads(str(archive["meta"]))
        ids = data.meta["parent_source_indices"]["val"]
        clean_arrays = {f"val_{k}": archive[f"val_{k}"][ids] for k in ("target", "cond")}
        archive.close()
    else:
        clean_data = BridgeData(a.clean_data, 0)
        clean_meta, clean_arrays = clean_data.meta, clean_data.arrays
    parent_id = data.meta.get("legacy_parent_meta_id", data.meta.get("parent_dataset_id"))
    if manifest_id(clean_meta) != parent_id:
        raise ValueError("Clean regression data does not match the adaptation source")
    out = Path(a.out) if a.out else Path(a.run)/"evaluation"
    out.mkdir(parents=True, exist_ok=True)
    target, cond, context = (torch.from_numpy(np.array(data.arrays[f"val_{key}"], dtype=np.float32))[:, None].to(device)
                             for key in ("target", "cond", "context"))
    generated, reports = {}, []
    clean_target, clean_cond = (torch.from_numpy(np.array(clean_arrays[f"val_{key}"], dtype=np.float32))[:, None].to(device)
                                for key in ("target", "cond"))
    for label, path in (("Original", Path(a.reference)), ("Adapted", Path(a.run)/a.checkpoint)):
        model, norm, ck = load_checkpoint(path, device)
        source_matches = (manifest_id(ck["meta"]) == parent_id if label=="Original" and legacy
                          else ck.get("dataset_id") == (parent_id if label=="Original" else manifest_id(data.meta)))
        if ((ck.get("stage") != "bridge" and not (label=="Original" and legacy)) or ck["meta"]["factor"] != data.meta["factor"]
                or ck["meta"]["size"] != data.meta["size"]
                or not source_matches
                or (label=="Adapted" and ck.get("guide_profile") != recipe["profile"])):
            raise ValueError("Checkpoint does not match this adaptation experiment")
        step = ck["step"]
        del ck
        fixed = torch.Generator(device=device).manual_seed(2026)
        mixed_t, mixed_c = paired_degradation(norm, target, cond, context, fixed, recipe=recipe)
        strong_t, strong_c = paired_degradation(norm, target, cond, context,
                        torch.Generator(device=device).manual_seed(2026), sigma_cells=recipe["sigma_max_cells"], clean_fraction=0., recipe=recipe)
        model_report = {"label": label, "checkpoint": str(path), "step": step,
                        "heldout_count": len(target), "conditions": {}}
        full_clean_loss = validation_loss(model, norm, clean_target, clean_cond, device, seeds=8, chunk=a.batch)
        model_report["full_clean_holdout"] = {"patches": len(clean_target), "val_loss": full_clean_loss}
        print(f"{label} / all {len(clean_target)} clean patches: validation {full_clean_loss:.5f}", flush=True)
        for kind, t, c in (("clean", target, cond), ("mixed", mixed_t, mixed_c),
                            ("smooth", strong_t, strong_c)):
            loss = validation_loss(model, norm, t, c, device, seeds=8, chunk=a.batch)
            print(f"{label} / {kind}: validation {loss:.5f}", flush=True)
            if kind != "smooth":
                model_report["conditions"][kind] = {"val_loss": loss}
                continue
            values, guides = [], []
            for start in range(0, len(c), a.batch):
                g, up = sample_patches(model, norm, c[start:start+a.batch], a.steps, seed=700+start, device=device)
                values.append(g[:, 0].cpu().numpy()); guides.append(up[:, 0].cpu().numpy())
                if (start//a.batch+1) % 8 == 0 or start+a.batch >= len(c):
                    print(f"{label}: sampled {min(start+a.batch,len(c))}/{len(c)} held-out patches", flush=True)
            z = decode(np.concatenate(values)); up = decode(np.concatenate(guides))
            truth = decode(t[:, 0].cpu().numpy())
            if not np.isfinite(z).all(): raise FloatingPointError("Nonfinite generated terrain")
            wl, real_power = radial_psd(truth, data.meta["spacing_m"])
            _, power = radial_psd(z, data.meta["spacing_m"])
            h = data.meta["spacing_m"]/1000
            f = data.meta["factor"]
            bands = [(2*h,4*h), (4*h,f*h), (f*h,2*f*h), (2*f*h,8*f*h)]
            model_report["conditions"][kind] = {"val_loss": loss,
                 "power_generated_vs_paired_target": band_ratios(wl, real_power, power, bands),
                 "generated_minus_guide_rms_m": float(np.sqrt(np.mean((z-up)**2)))}
            generated[label] = z
            last_guide, last_truth = up, truth
        reports.append(model_report)
        del model, norm
        gc.collect()
        if str(device).startswith("cuda"): torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Six varied examples, including highland and coast cases where available.
    pick = np.linspace(0, len(target)-1, min(6, len(target)), dtype=int)
    rows = [("Smooth input guide", last_guide), ("Original refinement", generated["Original"]),
            ("Adapted refinement", generated["Adapted"]), ("Paired Earth detail target", last_truth)]
    fig, axs = plt.subplots(4, len(pick), figsize=(2.5*len(pick), 10), squeeze=False)
    for column, index in enumerate(pick):
        lo, hi = np.percentile(last_truth[index], [1, 99])
        for r, (label, values) in enumerate(rows):
            rgb = plt.cm.terrain(np.clip((values[index]-lo)/max(hi-lo, 1), 0, 1))[..., :3]
            rgb *= .3+.7*hillshade(values[index], data.meta["spacing_m"])[..., None]
            axs[r, column].imshow(rgb, interpolation="nearest")
            axs[r, column].set_xticks([]); axs[r, column].set_yticks([])
            if column == 0: axs[r, column].set_ylabel(label)
            if r == 0: axs[r, column].set_title(f"Held-out patch {index}")
    across = data.meta.get("training_crop_size", data.meta["size"])*data.meta["spacing_m"]/1000
    fig.suptitle("Same smooth guide and initial noise: original versus adapted refinement\n"
                 f"Targets keep original Earth detail over the smoother guide; ~{across:,.0f} km patches")
    fig.tight_layout(); fig.savefig(out/"comparison.png", dpi=130); plt.close(fig)
    atomic_json(out/"evaluation.json", {"dataset_id": manifest_id(data.meta), "models": reports,
                    "sampler_steps": a.steps, "strong_smoothing_sigma_cells": recipe["sigma_max_cells"],
                    "target": data.meta["guide_recipe"]["target"]})
    print(f"Saved held-out comparison and metrics to {out}", flush=True)


if __name__ == "__main__":
    main()
