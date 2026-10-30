"""Capture learned intermediate states and numerical grid artifacts, without rendering."""
import argparse
import json
from pathlib import Path
import torch
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.generate import generate, decode_elevation, metadata_for
from planet_diffusion.storage import save_state, load_state, export_tiff, verify_state
from planet_diffusion.quality import grid_artifacts
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--coarse-height", type=int, default=2)
    p.add_argument("--resume-latent", action="store_true", help="Resume this seed-42 comparison from its saved latent-1.npy")
    args = p.parse_args()
    out = Path(args.output)
    if args.resume_latent:
        if (out/"state").exists() or (out/"planet-native.tif").exists():
            raise ValueError("Output already has a completed state or raster")
    else:
        out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    backend = TerrainBackend("upstream", "models/terrain-diffusion-90m", MODEL_REVISION)
    def audit(name, a):
        if name in {"coarse", "latent-1", "decoder"}:
            np.save(out/(name+".npy"), a)
    progress = lambda s: print(s, flush=True)
    if args.resume_latent:
        latent = np.load(out/"latent-1.npy", allow_pickle=False)
        expected_h = args.coarse_height*32
        if latent.shape != (5, expected_h+1, 2*expected_h):
            raise ValueError("Saved latent state does not match --coarse-height")
        a = decode_elevation(backend, 42, latent, progress=progress, audit=audit,
                             cache_dir=out/"decoder-cache")
        meta = metadata_for(backend, 42, args.coarse_height, 20)
    else:
        a, meta = generate(backend, 42, args.coarse_height, 20, audit=audit,
                           progress=progress, decoder_cache=out/"decoder-cache")
    meta["radius_metres"] = 6371000.
    save_state(out/"state", a, meta)
    a, meta = load_state(out/"state")
    export_tiff(out/"planet-native.tif", a, meta)
    report = {"artifacts": grid_artifacts(a), "boundaries": verify_state(a, meta)}
    (out/"quality.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
