"""Scale-aware evaluation on held-out patches.

Compares generated terrain with the real held-out terrain and with plain bilinear
upsampling of the condition, by radially averaged power spectrum per wavelength.
Too much power a few pixels wide means features drawn too large or too strong.

  python -m toy_meso.evaluate --data outputs/toy_meso/dataset.npz --run outputs/toy_meso/run1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_dataset, load_checkpoint, sample_patches, pick_device
from .sphere import decode, encode, block_mean
from .build_dataset import HOLDOUT_BOXES


def radial_psd(patches, spacing_m):
    """Mean radially averaged power spectrum of (B, n, n) metre patches."""
    b, n, _ = patches.shape
    win = np.outer(np.hanning(n), np.hanning(n))
    a = (patches - patches.mean(axis=(1, 2), keepdims=True)) * win
    power = np.abs(np.fft.fft2(a)) ** 2
    f = np.fft.fftfreq(n, d=spacing_m)
    k = np.hypot(f[:, None], f[None, :])
    edges = np.linspace(0, f.max(), n // 2 + 1)
    which = np.digitize(k, edges) - 1
    valid = (which >= 1) & (which < len(edges) - 1)
    out = np.array([power[:, (which == i) & valid].mean() for i in range(1, len(edges) - 1)])
    centres = (edges[1:-1] + edges[2:]) / 2
    return 1 / centres, out  # wavelength in metres, power


def band_ratios(wl, real, other, bands_km):
    out = {}
    for lo, hi in bands_km:
        m = (wl >= lo * 1000) & (wl < hi * 1000)
        out[f"{lo}-{hi} km"] = float(other[m].sum() / real[m].sum())
    return out


def hillshade(z, spacing_m, exaggeration=8.0):
    gy, gx = np.gradient(z * exaggeration, spacing_m)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def shaded(z, spacing_m):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    rgb = plt.cm.terrain(Normalize(-6000, 6000)(z))[..., :3]
    return rgb * (0.35 + 0.65 * hillshade(z, spacing_m)[..., None])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--steps", type=int, default=32, help="sampler steps")
    p.add_argument("--batch", type=int, default=8, help="patches per model call (lower if GPU memory spills)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = pick_device(a.device)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, meta = load_dataset(a.data)
    run = Path(a.run)
    model, norm, ck = load_checkpoint(run / a.checkpoint, device)
    sel = np.linspace(0, len(d["val_target"]) - 1, min(a.count, len(d["val_target"]))).astype(int)
    vt = d["val_target"][sel].astype(np.float32)
    vc = torch.from_numpy(d["val_cond"][sel])[:, None]
    gen, ups = [], []
    for i in range(0, len(sel), a.batch):
        g, u = sample_patches(model, norm, vc[i:i + a.batch], a.steps, seed=i, device=device)
        gen.append(g[:, 0].cpu().numpy()); ups.append(u[:, 0].cpu().numpy())
    gen, ups = np.concatenate(gen), np.concatenate(ups)
    real_m, gen_m, up_m = decode(vt), decode(gen), decode(ups)
    h = meta["spacing_m"]
    wl, p_real = radial_psd(real_m, h)
    _, p_gen = radial_psd(gen_m, h)
    _, p_up = radial_psd(up_m, h)
    f = meta["factor"]
    bands = [(2 * h / 1000, 4 * h / 1000), (4 * h / 1000, 8 * h / 1000),
             (8 * h / 1000, 2 * f * h / 1000), (2 * f * h / 1000, meta["size"] * h / 1000)]
    bands = [(int(lo), int(hi)) for lo, hi in bands]
    # Consistency: the generated terrain averaged back up should reproduce the condition.
    cond_centre = d["val_cond"][sel][:, 1:-1, 1:-1]
    consistency = float(np.sqrt(np.mean((block_mean(gen_m, f) - decode(cond_centre)) ** 2)))
    real_cons = float(np.sqrt(np.mean((block_mean(real_m, f) - decode(cond_centre)) ** 2)))

    def stats(m):
        land = m > 0
        return {"land_fraction": float(land.mean()),
                "land_p50_m": float(np.percentile(m[land], 50)) if land.any() else None,
                "land_p95_m": float(np.percentile(m[land], 95)) if land.any() else None,
                "slope_rms": float(np.sqrt(np.mean(np.square(np.gradient(m, h, axis=(1, 2))))))}

    report = {"checkpoint": str(run / a.checkpoint), "step": ck["step"], "patches": len(sel),
              "sampler_steps": a.steps, "spacing_m": h, "condition_spacing_m": h * f,
              "power_ratio_generated_vs_real": band_ratios(wl, p_real, p_gen, bands),
              "power_ratio_bilinear_vs_real": band_ratios(wl, p_real, p_up, bands),
              "condition_rms_error_m": {"generated": consistency, "real": real_cons},
              "real": stats(real_m), "generated": stats(gen_m), "bilinear": stats(up_m)}
    (run / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, pw, style in (("real (held out)", p_real, "k-"), ("generated", p_gen, "C1-"), ("bilinear condition", p_up, "C0--")):
        ax[0].loglog(wl / 1000, pw, style, label=name)
        ax[1].semilogx(wl / 1000, pw / p_real, style, label=name)
    for axis in ax:
        axis.axvline(h * f / 1000, color="0.6", lw=0.8)
        axis.set_xlabel("wavelength (km)")
        axis.invert_xaxis()
    ax[0].set_ylabel("power"); ax[0].legend()
    ax[1].set_ylabel("power / real power"); ax[1].set_ylim(0, 2); ax[1].axhline(1, color="0.6", lw=0.8)
    ax[0].set_title(f"Held-out spectra (step {ck['step']}); grey line = condition spacing")
    fig.tight_layout(); fig.savefig(run / "eval-spectrum.png", dpi=90); plt.close(fig)

    n = 6
    pick = np.linspace(0, len(sel) - 1, n).astype(int)
    fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 7.2))
    for c, i in enumerate(pick):
        for r, (name, arr) in enumerate((("bilinear condition", up_m), ("generated", gen_m), ("real (held out)", real_m))):
            ax[r, c].imshow(shaded(arr[i], h)); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c == 0:
                ax[r, c].set_ylabel(name)
    fig.suptitle(f"Held-out patches, {meta['size'] * h / 1000:.0f} km across at {h / 1000:.0f} km spacing")
    fig.tight_layout(); fig.savefig(run / "eval-patches.png", dpi=90); plt.close(fig)
    print(f"wrote {run / 'eval.json'}, eval-spectrum.png, eval-patches.png")


if __name__ == "__main__":
    main()
