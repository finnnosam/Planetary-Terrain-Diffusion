"""Run twice with pretrained weights and save quantitative evidence, no rendering.

Run from the project root: python -m scripts.verify_learned --model PATH
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.generate import generate
from planet_diffusion.storage import save_state, load_state, export_tiff, verify_state


def check_exports(out):
    import rasterio
    with rasterio.open(out/"planet-native.tif") as globe, rasterio.open(out/"dateline-detail.tif") as tile:
        a = globe.read(1)
        expected = np.concatenate([a[100:164, -32:], a[100:164, :32]], axis=1)
        return {"native_dimensions": [globe.width, globe.height],
                "global_bounds": list(globe.bounds), "band_units": list(globe.units),
                "dateline_tile_matches_native_exactly": bool(np.array_equal(expected, tile.read(1)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/terrain-diffusion-90m")
    p.add_argument("--output", default="outputs/learned-smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--coarse-height", type=int, default=2)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    backend = TerrainBackend("upstream", args.model, MODEL_REVISION, args.device)
    reports = []
    start = time.perf_counter()
    for run in range(2):
        stages = {}
        def audit(name, state):
            stages[name] = {"sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                            "north_spread": float(np.ptp(state[..., 0, :], axis=-1).max()),
                            "south_spread": float(np.ptp(state[..., -1, :], axis=-1).max())}
        print(f"RUN {run+1}/2", flush=True)
        a, meta = generate(backend, 42, args.coarse_height, 20,
                            progress=lambda msg: print(msg, flush=True), audit=audit)
        reports.append(stages)
        if run == 0:
            meta["radius_metres"] = 6371000.
            save_state(out/"state", a, meta)
            a, meta = load_state(out/"state")
            export_tiff(out/"planet.tif", a, meta, height=128)
            export_tiff(out/"planet-native.tif", a, meta)
            export_tiff(out/"dateline-detail.tif", a, meta, window=(2*meta["native_height"]-32, 100, 64, 64))
    a, meta = load_state(out/"state")
    report = {"elapsed_seconds": time.perf_counter()-start, "runtime": backend.metadata,
              "same_seed_all_stages_bitwise_equal": reports[0] == reports[1],
              "all_stages_have_unique_poles": all(s["north_spread"] == 0 and s["south_spread"] == 0 for s in reports[0].values()),
              "export_checks": check_exports(out),
              "boundary_checks": verify_state(a, meta), "stages": reports[0],
              "elevation_range_metres": [float(a.min()), float(a.max())],
              "note": "Small out-of-distribution coarse grid; smoke test, not geographic realism validation"}
    (out/"verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not (report["same_seed_all_stages_bitwise_equal"] and report["boundary_checks"]["passed"]
            and report["all_stages_have_unique_poles"] and report["export_checks"]["dateline_tile_matches_native_exactly"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
