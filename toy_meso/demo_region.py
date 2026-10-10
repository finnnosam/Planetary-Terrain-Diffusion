"""Generate a large region by overlapping tiles that agree at every sampler step.

The condition comes from any 2:1 global elevation GeoTIFF (Earth, or one of your
generated planets), area-averaged to the model's condition spacing around a
centre point. Tiles are denoised together and their predictions blended before
each Heun step, the same consensus idea as the cube sampler, so there are no seams.

  python -m toy_meso.demo_region --run outputs/toy_meso/run1 \
      --dem upstream/data/global/etopo_10m.tif --lat 64 --lon 14 --cells 34
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_checkpoint, autocast, pick_device
from .model import sigma_schedule
from .sphere import GlobalRaster, area_average_patch, encode, decode, EARTH_RADIUS_M


def north_up_frame(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    c = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(c, east)
    return c, east, north


def tile_starts(n, size, stride, align):
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    assert all(s % align == 0 for s in starts)
    return starts


def linear_window(size):
    t = np.minimum(np.arange(size) + 0.5, size - np.arange(size) - 0.5)
    return np.outer(t, t).astype(np.float32)


@torch.no_grad()
def generate_canvas(model, norm, cond_cells, tile=64, stride=48, steps=32, seed=0, batch=16, device="cpu",
                    progress=print):
    """cond_cells: (h, w) encoded condition with a one-cell margin. Returns encoded (H, W)."""
    f = norm.factor
    c = torch.from_numpy(np.asarray(cond_cells, np.float32))[None, None].to(device)
    up, _ = norm.condition(c)
    H, W = up.shape[-2:]
    ys, xs = tile_starts(H, tile, stride, f), tile_starts(W, tile, stride, f)
    tiles = [(y, x) for y in ys for x in xs]
    cins = torch.cat([norm.condition(c[..., y // f:y // f + tile // f + 2, x // f:x // f + tile // f + 2])[1]
                      for y, x in tiles])
    w = torch.from_numpy(linear_window(tile)).to(device)
    norm_w = torch.zeros(H, W, device=device)
    for y, x in tiles:
        norm_w[y:y + tile, x:x + tile] += w

    def denoise(xc, sigma):
        out = torch.zeros(H, W, device=device, dtype=torch.float64)
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            xt = torch.stack([xc[0, 0, y:y + tile, x:x + tile] for y, x in chunk])[:, None].float()
            s = torch.full((len(chunk),), float(sigma), device=device)
            with autocast(device):
                d = model(xt, s, cins[i:i + len(chunk)]).double()
            for (y, x), dt in zip(chunk, d[:, 0]):
                out[y:y + tile, x:x + tile] += w * dt
        return (out / norm_w)[None, None]

    t = sigma_schedule(steps, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    xc = torch.randn((1, 1, H, W), generator=g, device=device, dtype=torch.float64) * t[0]
    for i in range(steps):
        d = (xc - denoise(xc, t[i])) / t[i]
        nxt = xc + (t[i + 1] - t[i]) * d
        if t[i + 1] > 0:
            d2 = (nxt - denoise(nxt, t[i + 1])) / t[i + 1]
            nxt = xc + (t[i + 1] - t[i]) * (d + d2) / 2
        xc = nxt
        progress(f"step {i + 1}/{steps} ({len(tiles)} tiles)")
    return norm.target(xc.float(), up)[0, 0].cpu().numpy(), up[0, 0].cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--dem", required=True, help="2:1 global GeoTIFF in metres, or a 2:1 grayscale PNG sketch")
    p.add_argument("--black-metres", type=float, default=-2000.0, help="PNG only: elevation of black (default -2000)")
    p.add_argument("--white-metres", type=float, default=4000.0, help="PNG only: elevation of white (default 4000)")
    p.add_argument("--ocean-metres", type=float, help="PNG only: elevation for pure-black pixels (black = ocean mask sketches)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--cells", type=int, default=34, help="condition cells across, including the 1-cell margin")
    p.add_argument("--radius-metres", type=float, default=EARTH_RADIUS_M)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output folder (default: RUN/region-LAT-LON)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = pick_device(a.device)
    if a.steps < 8:
        p.error("--steps must be at least 8 (fewer steps make the sampler unstable)")
    torch.backends.cudnn.benchmark = False  # same seed + settings -> same result
    torch.backends.cudnn.deterministic = True
    model, norm, ck = load_checkpoint(Path(a.run) / a.checkpoint, device)
    meta = ck["meta"]
    h, f = meta["spacing_m"], meta["factor"]
    scale = EARTH_RADIUS_M / a.radius_metres  # sphere geometry is in Earth-radius units
    dem = GlobalRaster.from_file(a.dem, a.black_metres, a.white_metres, a.ocean_metres)
    frame = north_up_frame(a.lat, a.lon)
    cond_m, _, _ = area_average_patch(dem, frame, a.cells, h * f * scale, supersample=8)
    n = (a.cells - 2) * f
    source_m, _, _ = area_average_patch(dem, frame, n + 2 * f, h * scale, supersample=3)
    source_m = source_m[f:-f, f:-f]
    gen, up = generate_canvas(model, norm, encode(cond_m), steps=a.steps, seed=a.seed, device=device,
                              progress=lambda s: print(s, flush=True))
    gen_m, up_m = decode(gen), decode(up)
    out = Path(a.out) if a.out else Path(a.run) / f"region-{a.lat:g}-{a.lon:g}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "generated_m.npy", gen_m)
    np.save(out / "source_m.npy", source_m)
    (out / "region.json").write_text(json.dumps({
        "dem": a.dem, "black_metres": a.black_metres, "white_metres": a.white_metres, "ocean_metres": a.ocean_metres, "lat": a.lat, "lon": a.lon, "cells": a.cells, "pixels": n, "spacing_m": h,
        "width_km": n * h / 1000, "projection": "north-up gnomonic about the centre",
        "step": ck["step"], "seed": a.seed}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import shaded
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for axis, (name, arr) in zip(ax, (("condition, bilinear", up_m), ("toy model", gen_m),
                                      ("source raster at the same spacing", source_m))):
        axis.imshow(shaded(arr, h)); axis.set_title(name); axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"{n * h / 1000:.0f} km across at {h / 1000:.0f} km spacing, centred {a.lat:g}, {a.lon:g}")
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=90); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
