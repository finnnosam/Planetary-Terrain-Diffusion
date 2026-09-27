"""Run from the project root after obtaining upstream's five reference TIFFs.

Usage: .venv/Scripts/python tools/build_conditioning_stats.py
Requires the upstream checkout, torch, rasterio, and pyfastnoiselite.
Downloads are intentionally left to the caller; runtime generation is offline.
"""
import os
from pathlib import Path
import sys
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"upstream"))
os.chdir(ROOT/"upstream")
from terrain_diffusion.inference import synthetic_map

stats = synthetic_map._compute_map_stats([1.5, 3, 3, 3, 3], .5)
synthetic_map._save_stats_cache(stats)
destination = ROOT/"planet_diffusion/data"
destination.mkdir(exist_ok=True)
(destination/"synthetic_map_stats.json").write_bytes(Path(synthetic_map.STATS_CACHE_PATH).read_bytes())
with rasterio.open("data/global/etopo_10m.tif") as ds:
    elev = ds.read(1)
crop = elev.shape[0]//6
np.savez_compressed(destination/"elevation_reference.npz", elevation=elev[crop:-crop])
(destination/"terrain_diffusion_LICENSE.txt").write_bytes(Path("LICENSE").read_bytes())
