"""Compare the best and latest intermediate checkpoints without resuming training."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import VERSION, atomic_json, manifest_id
from .train import BridgeData
from toy_meso.common import load_checkpoint, pick_device, sample_patches
from toy_meso.sphere import decode
from toy_meso.train import validation_loss


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/intermediate/run1")
    parser.add_argument("--data", default="outputs/planet_training/data/bridge")
    parser.add_argument("--out", help="defaults to <run>/evaluation")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32, help="preview denoising steps")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch < 1 or args.steps < 2:
        parser.error("batch must be positive and steps must be at least two")
    torch.set_num_threads(4)
    device = pick_device(args.device)
    dataset = BridgeData(args.data, seed=42)
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    target, cond = (torch.from_numpy(np.array(dataset.arrays[f"val_{key}"], dtype=np.float32))[:, None].to(device)
                    for key in ("target", "cond"))
    # Keep the original four preview cases, plus four spread through the holdout.
    indices = np.unique(np.r_[np.arange(min(4, len(target))),
                              np.linspace(4, len(target)-1, min(4, max(0, len(target)-4)), dtype=int)])
    images, results = [], []
    for name in ("best.pt", "latest.pt"):
        path = run / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model, norm, ck = load_checkpoint(path, device)
        if (ck.get("stage") != "bridge" or ck.get("training_version") != VERSION
                or ck.get("dataset_id") != manifest_id(dataset.meta)):
            raise ValueError(f"{path} does not match the prepared intermediate dataset")
        step = ck["step"]
        del ck
        print(f"Checking {name}, update {step:,}, on all {len(target)} held-out patches...", flush=True)
        loss = validation_loss(model, norm, target, cond, device, seeds=8, chunk=args.batch)
        generated, up = sample_patches(model, norm, cond[indices], args.steps, seed=7, device=device)
        if not torch.isfinite(generated).all():
            raise FloatingPointError(f"Nonfinite generated elevations from {name}")
        images.append((f"{name} / update {step:,}", decode(generated[:, 0].cpu().numpy())))
        results.append({"checkpoint": name, "step": step, "sha256": digest, "val_loss": loss})
        print(f"  Held-out loss: {loss:.6f} (lower is better)", flush=True)
        if name == "best.pt":
            guide = decode(up[:, 0].cpu().numpy())
        del model, norm, generated, up
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("161 km input", guide), *images,
            ("Real held-out terrain", decode(target[indices, 0].cpu().numpy()))]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(2.25*len(indices), 8.5), squeeze=False)
    for r, (label, values) in enumerate(rows):
        for i, values_i in enumerate(values):
            axes[r, i].imshow(values_i, cmap="terrain", vmin=-6000, vmax=6000)
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
            if r == 0:
                axes[r, i].set_title(f"Held-out patch {indices[i]}")
        axes[r, 0].set_ylabel(label)
    fig.suptitle("Intermediate terrain comparison · same guide and initial noise · 32-step sampling"
                 if args.steps == 32 else f"Intermediate terrain comparison · {args.steps}-step sampling")
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=130); plt.close(fig)

    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    validation = [r for r in records if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r["step"] for r in validation], [r["val_loss"] for r in validation], label="Held-out loss")
    ax.axvline(results[0]["step"], color="green", linestyle="--", label="Saved best checkpoint")
    ax.set(xlabel="Training update", ylabel="Validation loss (lower is better)",
           title="Intermediate model: held-out validation over training")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "validation-curve.png", dpi=130); plt.close(fig)
    atomic_json(out / "evaluation.json", {"protocol": "entire-heldout-set-v1", "device": str(device),
                "validation_count": len(target), "batch": args.batch, "sampling_steps": args.steps,
                "sampling_seed": 7, "preview_indices": indices.tolist(), "checkpoints": results,
                "latest_loss_change_percent": 100*(results[1]["val_loss"]/results[0]["val_loss"]-1)})
    change = 100*(results[1]["val_loss"]/results[0]["val_loss"]-1)
    print(f"Latest checkpoint loss is {abs(change):.2f}% {'higher' if change >= 0 else 'lower'} than best.pt.", flush=True)
    print(f"Saved comparison, validation curve and results to {out}", flush=True)


if __name__ == "__main__":
    main()
