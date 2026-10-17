"""Prepare both datasets, without starting GPU training. Safe to rerun/resume."""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import (VERSION, SPACING_M, FACTOR, LAYOUT_N, LAYOUT_SPACING_M,
                   VALIDATION_AGES, HOLDOUT_CAPS, PaleoRaster, age_split, atomic_json,
                   fingerprint, geographic_split, bridge_patch)
from toy_meso.sphere import GlobalRaster, patch_points, random_frames

DEFAULT_PALEO = "data/Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc"
DEFAULT_OUT = "outputs/planet_training/data"


def prepare_layout(source, folder):
    files = sorted(Path(source).rglob("*.nc"))
    if not files:
        raise ValueError(f"No PaleoDEM .nc files in {source}")
    records, values = [], []
    for f in files:
        match = re.search(r"_(\d+(?:\.\d+)?)Ma", f.name)
        if not match:
            raise ValueError(f"Cannot identify geological age: {f.name}")
        age = float(match[1])
        records.append(dict(fingerprint(f), age_ma=age, split=age_split(age)))
        values.append(PaleoRaster.from_file(f).values)
    records_values = sorted(zip(records, values), key=lambda pair: pair[0]["age_ma"])
    records, values = zip(*records_values)
    if len({r["age_ma"] for r in records}) != len(records):
        raise ValueError("Duplicate PaleoDEM ages; curate the source folder first")
    meta = {"version": VERSION, "kind": "layout", "face_intervals": LAYOUT_N,
            "spacing_m": LAYOUT_SPACING_M, "radius_m": 6371000.0,
            "supersample": 4, "encoding": "sign(z)*sqrt(abs(z)) metres",
            "validation_age_ranges_ma": VALIDATION_AGES, "buffer_ma": 20,
            "test_age_range_ma": [0, 20], "records": list(records),
            "license": "Scotese & Wright 2018; CC BY 4.0 (source License.txt)"}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(meta)):
            raise ValueError(f"Layout preparation settings changed; choose a fresh output folder: {folder}")
        if (folder / "maps.npy").exists():
            print("Layout sources already prepared.", flush=True)
            return
    tmp = folder / "maps.tmp.npy"
    np.save(tmp, np.stack(values))
    tmp.replace(folder / "maps.npy")
    atomic_json(path, meta)
    print("Layout sources ready: " + ", ".join(
        f"{sum(r['split'] == s for r in records)} {s}" for s in ("train", "val", "test", "buffer")), flush=True)


def prefilter_dem(source, folder, factor=6):
    """Stream area means to a small 0.05-degree raster; never point-decimate."""
    import rasterio
    from rasterio.windows import Window
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, info = folder / "etopo-filtered.npy", folder / "etopo-filtered.json"
    wanted = {"source": fingerprint(source), "factor": factor, "method": "cos(latitude)-weighted block mean"}
    if path.exists() and info.exists() and json.loads(info.read_text()) == wanted:
        return GlobalRaster(np.load(path, mmap_mode="r"))
    tmp = folder / "etopo-filtered.tmp.npy"
    with rasterio.open(source) as ds:
        if ds.width != 2 * ds.height or ds.width % factor or ds.height % factor:
            raise ValueError("ETOPO must be a divisible 2:1 pixel-centred grid")
        if not np.allclose(tuple(ds.bounds), (-180, -90, 180, 90), atol=1e-6):
            raise ValueError("ETOPO must cover the entire world")
        if ds.transform.a <= 0 or ds.transform.e >= 0 or ds.transform.b or ds.transform.d:
            raise ValueError("ETOPO must be a north-up lon/lat raster")
        h, w = ds.height // factor, ds.width // factor
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(h, w))
        for y in range(0, h, 60):
            rows = min(60, h - y)
            a = ds.read(1, window=Window(0, y * factor, ds.width, rows * factor), masked=True)
            if np.ma.getmaskarray(a).any() or not np.isfinite(a).all():
                raise ValueError("ETOPO contains missing elevations")
            lat = 90 - (np.arange(y*factor, (y+rows)*factor) + .5) * 180 / ds.height
            weights = np.cos(np.radians(lat)).reshape(rows, factor, 1)
            # Mean over longitude first; then use spherical row-area weights.
            means = np.asarray(a).reshape(rows, factor, w, factor).mean(axis=-1)
            out[y:y+rows] = (means * weights).sum(axis=1) / weights.sum(axis=1)
            if y % 600 == 0:
                print(f"ETOPO averaging: {y}/{h} output rows", flush=True)
        out.flush()
        del out
    tmp.replace(path)
    atomic_json(info, wanted)
    return GlobalRaster(np.load(path, mmap_mode="r"))


def prepare_bridge(source, folder, cache, train=8000, val=512, size=112, sub=6, seed=42, workers=4):
    if size % FACTOR or size % 8 or min(train, val, sub, workers) < 1:
        raise ValueError("Bridge size must be divisible by 7 and 8; counts and supersample must be positive")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"version": VERSION, "kind": "bridge", "size": size, "factor": FACTOR,
            "spacing_m": SPACING_M, "condition_spacing_m": LAYOUT_SPACING_M,
            "margin_cells": 1, "encoding": "sign(z)*sqrt(abs(z)) metres",
            "source": fingerprint(source), "prefilter_factor": 6, "supersample": sub,
            "train_count": train, "val_count": val, "seed": seed,
            "holdout_caps_lat_lon_radius_m": HOLDOUT_CAPS, "geographic_guard_m": 25000,
            "averaging": "spherical Jacobian-weighted target and condition in metres"}
    mpath, ppath = folder / "manifest.json", folder / "progress.json"
    meta = json.loads(json.dumps(meta))
    if mpath.exists() and json.loads(mpath.read_text()) != meta:
        raise ValueError(f"Bridge preparation settings changed; choose a fresh output folder: {folder}")
    arrays = {}
    cside = size // FACTOR + 2
    for split, count in (("train", train), ("val", val)):
        for key, shape, dtype in (("target", (count, size, size), np.float16),
                                  ("cond", (count, cside, cside), np.float32),
                                  ("centre", (count, 2), np.float32)):
            path = folder / f"{split}_{key}.npy"
            if ppath.exists() and not path.exists():
                raise ValueError(f"Incomplete preparation cache: missing {path}")
            arrays[f"{split}_{key}"] = np.lib.format.open_memmap(
                path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape)
            if arrays[f"{split}_{key}"].shape != shape or arrays[f"{split}_{key}"].dtype != dtype:
                raise ValueError(f"Unexpected array in {path}")
    rng = np.random.default_rng(seed)
    done, tried = {"train": 0, "val": 0}, 0
    if ppath.exists():
        progress = json.loads(ppath.read_text())
        done, tried = progress["done"], progress["tried"]
        rng.bit_generator.state = progress["rng"]
    atomic_json(mpath, meta)
    if done == {"train": train, "val": val}:
        print("Intermediate dataset already prepared.", flush=True)
        return
    raster = prefilter_dem(source, cache)
    started = time.perf_counter()
    committed_rng = rng.bit_generator.state

    def save_progress():
        for a in arrays.values():
            a.flush()
        atomic_json(ppath, {"done": done, "tried": tried, "rng": committed_rng})

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while done["train"] < train or done["val"] < val:
                if tried > (train + val) * 1000:
                    raise RuntimeError("Cannot fill geographic splits; inspect holdouts and patch size")
                jobs = []
                # Sampling and commits are ordered; only raster integration runs
                # concurrently. Save the RNG at the last committed candidate so
                # interruption cannot skip any prefetched but uncommitted jobs.
                for offset in range(1, workers*4 + 1):
                    frames = random_frames(rng, 1)
                    frame = tuple(f[0] for f in frames)
                    coin = float(rng.random())
                    points = patch_points(frame, size + 2 * FACTOR, SPACING_M)
                    split = geographic_split(points)
                    job = None
                    if split is not None and done[split] < (train if split == "train" else val):
                        job = pool.submit(bridge_patch, raster, frame, size=size, sub=sub)
                    jobs.append((tried+offset, frame, split, coin, rng.bit_generator.state, job))
                for seq, frame, split, coin, state, job in jobs:
                    result = job.result() if job is not None else None
                    if result is not None and done[split] < (train if split == "train" else val):
                        target, cond = result
                        land = float((target > 0).mean())
                        if split == "val" or coin <= .3 + .7 * min(1., 3 * land):
                            index = done[split]
                            arrays[f"{split}_target"][index] = target
                            arrays[f"{split}_cond"][index] = cond
                            from toy_meso.sphere import to_latlon
                            arrays[f"{split}_centre"][index] = to_latlon(frame[0])
                            done[split] += 1
                            if sum(done.values()) % 100 == 0:
                                tried, committed_rng = seq, state
                                save_progress()
                                print(f"Intermediate patches: {done['train']}/{train} train, {done['val']}/{val} val; "
                                      f"{time.perf_counter()-started:.0f}s this session", flush=True)
                    tried, committed_rng = seq, state
    finally:
        save_progress()
    print("Intermediate dataset ready.", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("both", "layout", "bridge"), default="both")
    p.add_argument("--paleo", default=DEFAULT_PALEO)
    p.add_argument("--dem", default="data/etopo_2022_30s_surface.tif")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=512)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--supersample", type=int, default=6)
    p.add_argument("--workers", type=int, default=4, help="CPU threads for intermediate patch integration")
    a = p.parse_args(argv)
    out = Path(a.out)
    if a.stage in ("both", "layout"):
        prepare_layout(a.paleo, out / "layout")
    if a.stage in ("both", "bridge"):
        prepare_bridge(a.dem, out / "bridge", out / "cache", a.train, a.val, a.size, a.supersample, workers=a.workers)


if __name__ == "__main__":
    main()
