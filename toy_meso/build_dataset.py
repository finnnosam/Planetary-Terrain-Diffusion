"""Build the toy training set: area-averaged sphere patches from a global DEM.

Each example is a SIZE x SIZE target at SPACING metres (encoded elevation) and
its condition: the same terrain area-averaged FACTOR x FACTOR, with one
condition cell of margin on every side so upsampling has real neighbours.

  python -m toy_meso.build_dataset --dem upstream/data/global/etopo_10m.tif \
      --out outputs/toy_meso/dataset.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .sphere import GlobalRaster, random_frames, area_average_patch, block_mean, encode

# Named held-out regions (west, south, east, north): used for validation and
# never touched by any pixel of any training patch.
HOLDOUT_BOXES = {
    "scandinavia": (0, 56, 32, 72),
    "tibet_himalaya": (75, 25, 100, 40),
    "patagonia": (-78, -56, -64, -38),
    "east_africa": (28, -12, 42, 6),
    "pacific_abyss": (-160, -30, -130, -5),
}
BLOCK_DEG = 15
BLOCK_FRACTION = 0.08


def in_boxes(lat, lon):
    held = np.zeros(np.shape(lat), bool)
    for w, s, e, n in HOLDOUT_BOXES.values():
        held |= (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    return held


def in_blocks(lat, lon, seed=1234):
    by = np.floor((np.asarray(lat) + 90) / BLOCK_DEG).astype(np.int64)
    bx = np.floor((np.asarray(lon) + 180) / BLOCK_DEG).astype(np.int64)
    h = (by * 73856093) ^ (bx * 19349663) ^ seed
    return (h % 1000) < BLOCK_FRACTION * 1000


def held_out(lat, lon):
    return in_boxes(lat, lon) | in_blocks(lat, lon)


def build(dem, count_train, count_val, size=64, factor=8, spacing_m=20000.0,
          supersample=3, seed=0, progress=print):
    rng = np.random.default_rng(seed)
    margin = factor
    full = size + 2 * margin
    out = {"train": ([], [], []), "val": ([], [], [])}
    need = {"train": count_train, "val": count_val}
    tried = 0
    started = time.perf_counter()
    while any(len(out[k][0]) < need[k] for k in out):
        frames = random_frames(rng, 64)
        for i in range(64):
            frame = tuple(f[i] for f in frames)
            tried += 1
            metres, lat, lon = area_average_patch(dem, frame, full, spacing_m, supersample)
            clat, clon = lat[lat.shape[0] // 2, lat.shape[1] // 2], lon[lon.shape[0] // 2, lon.shape[1] // 2]
            if held_out(clat, clon):
                split = "val"
            elif not held_out(lat, lon).any():
                split = "train"
            else:
                continue  # straddles a held-out region
            if len(out[split][0]) >= need[split]:
                continue
            core = metres[margin:-margin, margin:-margin]
            land = (core > 0).mean()
            if rng.random() > 0.3 + 0.7 * min(1.0, 3 * land):
                continue  # keep some open ocean, favour land and coasts
            cond = encode(block_mean(metres, factor))
            target = encode(core)
            out[split][0].append(target.astype(np.float16))
            out[split][1].append(cond.astype(np.float32))
            out[split][2].append((clat, clon))
        done = len(out["train"][0]) + len(out["val"][0])
        progress(f"patches {done}/{count_train + count_val} (tried {tried}, "
                 f"{time.perf_counter() - started:.0f} s)")
    return out, tried


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dem", required=True, help="Global 2:1 elevation GeoTIFF in metres")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=24000)
    p.add_argument("--val", type=int, default=1500)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--factor", type=int, default=8)
    p.add_argument("--spacing-km", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    dem = GlobalRaster.from_geotiff(a.dem)
    out, tried = build(dem, a.train, a.val, a.size, a.factor, a.spacing_km * 1000, seed=a.seed)
    meta = {"size": a.size, "factor": a.factor, "spacing_m": a.spacing_km * 1000,
            "condition_spacing_m": a.spacing_km * 1000 * a.factor, "margin_cells": 1,
            "encoding": "sign(z)*sqrt(|z|) of metres", "dem": str(a.dem), "seed": a.seed,
            "holdout_boxes": HOLDOUT_BOXES, "holdout_block_deg": BLOCK_DEG,
            "holdout_block_fraction": BLOCK_FRACTION, "patches_tried": tried}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        train_target=np.stack(out["train"][0]), train_cond=np.stack(out["train"][1]),
        train_centre=np.array(out["train"][2], np.float32),
        val_target=np.stack(out["val"][0]), val_cond=np.stack(out["val"][1]),
        val_centre=np.array(out["val"][2], np.float32),
        meta=json.dumps(meta))
    print(f"wrote {a.out}: {len(out['train'][0])} train, {len(out['val'][0])} val")


if __name__ == "__main__":
    main()
