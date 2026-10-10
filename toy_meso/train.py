"""Train the toy refinement model.

  python -m toy_meso.train --data outputs/toy_meso/dataset.npz --run outputs/toy_meso/run1

Rerun the same command to resume from <run>/latest.pt.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_dataset, compute_stats, Normalizer, build_model, autocast, pick_device, sample_patches
from .model import edm_loss


def augment(target, cond):
    """Same random rot90/flip for the target and its condition grid."""
    k = int(torch.randint(4, ()))
    target, cond = torch.rot90(target, k, (-2, -1)), torch.rot90(cond, k, (-2, -1))
    if torch.rand(()) < 0.5:
        target, cond = target.flip(-1), cond.flip(-1)
    return target, cond


@torch.no_grad()
def validation_loss(model, norm, target, cond, device, seeds=8, chunk=16):
    """Fixed-sigma, fixed-noise loss on validation patches (comparable across runs).

    Evaluated in chunks so validation never needs more GPU memory than training.
    """
    g = torch.Generator(device=device).manual_seed(0)
    up, cin = norm.condition(cond)
    x = norm.residual(target, up)
    total = 0.0
    sigmas = torch.exp(torch.linspace(-3, 1.5, seeds, device=device))
    for s in sigmas:
        n = torch.randn(x.shape, generator=g, device=device)
        w = (s ** 2 + 0.25) / (s * 0.5) ** 2
        err = 0.0
        for i in range(0, x.shape[0], chunk):
            xi = x[i:i + chunk]
            with autocast(device):
                d = model(xi + n[i:i + chunk] * s, s.expand(xi.shape[0]), cin[i:i + chunk])
            err += float(((d.float() - xi) ** 2).sum())
        total += float(w) * err / x.numel()
    return total / len(sigmas)


def save_preview(path, cond, real, generated, up):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sphere import decode
    rows = [("condition (bilinear)", up), ("generated", generated), ("real", real)]
    n = real.shape[0]
    fig, ax = plt.subplots(len(rows), n, figsize=(2 * n, 2 * len(rows) + 0.4))
    for r, (name, arr) in enumerate(rows):
        for i in range(n):
            ax[r, i].imshow(decode(arr[i, 0].cpu().numpy()), cmap="terrain", vmin=-6000, vmax=6000)
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
        ax[r, 0].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, default=30000, help="optimizer updates")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--mults", default="1,2,4")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=500, help="checkpoint interval (no evaluation)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-on-cpu", action="store_true",
                   help="keep the training patches in system RAM and copy each batch to the GPU (saves VRAM)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)

    d, meta = load_dataset(a.data)
    factor = meta["factor"]
    config = {"ch": a.ch, "mults": [int(m) for m in a.mults.split(",")], "blocks": a.blocks, "dropout": a.dropout}
    latest = run / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        config, stats = ck["config"], ck["stats"]
        print(f"resuming from step {ck['step']}")
    else:
        ck, stats = None, compute_stats(d["train_target"], d["train_cond"], factor)
    norm = Normalizer(stats, factor)

    data_device = "cpu" if a.data_on_cpu else device
    target = torch.from_numpy(d["train_target"]).to(data_device)[:, None]  # float16
    cond = torch.from_numpy(d["train_cond"]).to(data_device)[:, None]
    vt = torch.from_numpy(d["val_target"].astype(np.float32)).to(device)[:, None]
    vc = torch.from_numpy(d["val_cond"]).to(device)[:, None]
    vsel = torch.arange(0, len(vt), max(1, len(vt) // 128), device=device)[:128]
    print(f"device {device}; {len(target)} train / {len(vt)} val patches; stats {stats}")

    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    step = 0
    if ck is not None:
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); step = ck["step"]
    params = sum(p.numel() for p in model.parameters())
    print(f"model {params / 1e6:.2f} M parameters, config {config}")
    log = open(run / "log.jsonl", "a")

    def checkpoint(name):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "config": config, "stats": stats, "meta": meta}, run / (name + ".tmp"))
        (run / (name + ".tmp")).replace(run / name)

    started, running = time.perf_counter(), []
    while step < a.steps:
        idx = torch.randint(len(target), (a.batch,), device=data_device)
        t, c = augment(target[idx].to(device).float(), cond[idx].to(device))
        up, cin = norm.condition(c)
        x = norm.residual(t, up)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        with autocast(device):
            loss = edm_loss(model, x, cin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        decay = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        running.append(loss.item())
        if step % a.log_every == 0:
            rate = a.log_every / (time.perf_counter() - started)
            rec = {"step": step, "loss": float(np.mean(running)), "steps_per_s": rate}
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            running, started = [], time.perf_counter()
        if step % a.save_every == 0:
            checkpoint("latest.pt")
        if step % a.eval_every == 0 or step == a.steps:
            vl = validation_loss(ema, norm, vt[vsel], vc[vsel], device, chunk=a.batch)
            rec = {"step": step, "val_loss": vl}
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            checkpoint("latest.pt")
            gen, up = sample_patches(ema, norm, vc[vsel[:8]], steps=24, device=device)
            save_preview(run / f"preview-{step:06d}.png", vc[vsel[:8]], vt[vsel[:8]], gen, up)
            started = time.perf_counter()
    checkpoint("latest.pt")
    print("done")


if __name__ == "__main__":
    main()
