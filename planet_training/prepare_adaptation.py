"""Reuse real targets and prepare leak-free coarse context for adapted training."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .adaptation import HALO, RECIPE, recovered_frames, coarse_context, frame_matches_target
from .data import (atomic_json, manifest_id, geographic_split, FACTOR, SPACING_M,
                   fingerprint)
from .prepare import prefilter_dem
from toy_meso.sphere import patch_points


def prepare(source, out, cache, workers=4, sub=12):
    source, out = Path(source), Path(out)
    parent = json.loads((source/"manifest.json").read_text())
    if parent.get("kind") != "bridge" or parent.get("guide_recipe"):
        raise ValueError("Expected the original clean bridge dataset")
    if fingerprint(parent["source"]["path"]) != parent["source"]:
        raise ValueError("Original elevation source changed since preparation")
    print("Recovering patch orientations and checking the complete filtering footprint...", flush=True)
    raster = prefilter_dem(parent["source"]["path"], cache)
    frames = recovered_frames(source, allow_partial=True)
    selected = {s: [] for s in frames}
    full = parent["size"] + 2*(HALO+1)*FACTOR
    for split, values in frames.items():
        targets = np.load(source/f"{split}_target.npy", mmap_mode="r")
        for index, frame in enumerate(values):
            if frame is None or not frame_matches_target(raster, frame, targets[index],
                                                         parent["size"], parent["supersample"]):
                continue
            # Fine grid covers the same full coarse context; 25 km guards extend
            # beyond outer centres and cover the extreme quadrature samples.
            if geographic_split(patch_points(frame, full, SPACING_M)) == split:
                selected[split].append(index)
    if any(len(v) < 1 for v in selected.values()):
        raise ValueError("Filtering footprints leave an empty geographic split")
    print("Verified reusable patches: " + ", ".join(f"{len(v):,} {s}" for s, v in selected.items()), flush=True)
    meta = dict(parent, train_count=len(selected["train"]), val_count=len(selected["val"]),
                parent_dataset_id=manifest_id(parent), guide_recipe=RECIPE,
                context_supersample=sub, context_holdout_guard_m=25000,
                parent_source_indices=selected)
    meta = json.loads(json.dumps(meta))
    out.mkdir(parents=True, exist_ok=True)
    mpath, ppath = out/"manifest.json", out/"progress.json"
    if mpath.exists() and json.loads(mpath.read_text()) != meta:
        raise ValueError("Adaptation settings changed; use a new dataset folder")
    done = json.loads(ppath.read_text())["done"] if ppath.exists() else {s: 0 for s in selected}
    arrays = {}
    side = parent["size"]//FACTOR+2
    for split, indices in selected.items():
        for key, shape, dtype in (("target", (len(indices), parent["size"], parent["size"]), np.float16),
                                  ("cond", (len(indices), side, side), np.float32),
                                  ("context", (len(indices), side+2*HALO, side+2*HALO), np.float32),
                                  ("centre", (len(indices), 2), np.float32)):
            path = out/f"{split}_{key}.npy"
            if ppath.exists() and not path.exists():
                raise ValueError(f"Missing adaptation cache: {path}")
            arrays[split, key] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                                          dtype=dtype, shape=shape)
    atomic_json(mpath, meta)
    def save_progress():
        for value in arrays.values(): value.flush()
        atomic_json(ppath, {"done": done})
    if all(done[s] == len(selected[s]) for s in selected):
        print("Adapted intermediate dataset already prepared.", flush=True)
        return meta
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for split, indices in selected.items():
                originals = {key: np.load(source/f"{split}_{key}.npy", mmap_mode="r")
                             for key in ("target", "cond", "centre")}
                for start in range(done[split], len(indices), workers*4):
                    chunk = indices[start:start+workers*4]
                    futures = [pool.submit(coarse_context, raster, frames[split][i], parent["size"], sub)
                               for i in chunk]
                    for offset, (index, future) in enumerate(zip(chunk, futures)):
                        dest = start+offset
                        context = future.result()
                        # Exact original means retain original targets, clean
                        # normalization, and phase, independent of halo sampling.
                        context[HALO:-HALO, HALO:-HALO] = originals["cond"][index]
                        arrays[split, "context"][dest] = context
                        for key in ("target", "cond", "centre"):
                            arrays[split, key][dest] = originals[key][index]
                        done[split] = dest+1
                    save_progress()
                    if start == 0 or done[split] % 256 < workers*4 or done[split] == len(indices):
                        print(f"Guide context: {split} {done[split]:,}/{len(indices):,} | "
                              f"{time.perf_counter()-started:.0f}s this session", flush=True)
    finally:
        save_progress()
    print("Adapted intermediate data ready; original data preserved.", flush=True)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/planet_training/data/bridge")
    p.add_argument("--out", default="outputs/planet_training/data/bridge-adapted")
    p.add_argument("--cache", default="outputs/planet_training/data/cache")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--supersample", type=int, default=12)
    a = p.parse_args(argv)
    if min(a.workers, a.supersample) < 1: p.error("workers and supersample must be positive")
    prepare(a.source, a.out, a.cache, a.workers, a.supersample)


if __name__ == "__main__":
    main()
