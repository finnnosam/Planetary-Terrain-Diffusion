"""Export a trained continent model as an elevation GeoTIFF for the bridge."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from planet_diffusion import cube
from toy_meso.common import pick_device
from toy_meso.sphere import decode, encode, EARTH_RADIUS_M
from .data import VERSION, atomic_json
from .layout import LayoutDenoiser
from .train import sample_layout, save_layout_preview
from .geography import control_layout, measures, export_pixels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="outputs/continents/run1")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--height", type=int, default=256, help="export pixels; increases interpolation, not learned detail")
    p.add_argument("--device", default="auto")
    p.add_argument("--land-percent", type=float, help="post-sampling land coverage via an area-weighted sea-level shift")
    p.add_argument("--land-relief", type=float, default=1., help="multiply positive elevation after setting sea level; default 1")
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    if a.steps < 2 or a.height < 2:
        p.error("steps and height must be at least 2")
    if not np.isfinite(a.land_relief) or a.land_relief<=0 or (a.land_percent is not None and (not np.isfinite(a.land_percent) or not 0<a.land_percent<100)):
        p.error("Land relief must be positive; land coverage must be between 0 and 100 percent")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new output folder to preserve existing results")
    device = pick_device(a.device)
    payload=(Path(a.run)/a.checkpoint).read_bytes()
    checkpoint_sha=hashlib.sha256(payload).hexdigest()
    ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    del payload
    if ck.get("stage") != "layout" or ck.get("training_version") != VERSION:
        raise ValueError("Expected a planet-training layout checkpoint")
    model = LayoutDenoiser(n=ck["meta"]["face_intervals"], **ck["config"]).to(device).eval()
    model.load_state_dict(ck["ema"])
    encoded = sample_layout(model, ck["stats"], a.steps, a.seed, device)[0]
    metres, controls = control_layout(decode(encoded), a.land_percent, a.land_relief)
    image = export_pixels(metres, a.height)
    out.mkdir(parents=True, exist_ok=True)
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(out/"layout.tif", "w", driver="GTiff", height=a.height, width=2*a.height,
                       count=1, dtype="float32", crs=f"+proj=longlat +R={EARTH_RADIUS_M} +no_defs",
                       transform=from_origin(-180, 90, 180/a.height, 180/a.height), compress="deflate") as ds:
        ds.write(image.astype(np.float32), 1)
        ds.set_band_description(1, "Continent-scale elevation")
        ds.set_band_unit(1, "m")
    np.save(out/"cube-elevation.npy", metres)
    save_layout_preview(out/"layout-preview.png", encode(metres)[None])
    atomic_json(out/"layout.json", dict(vars(a), training_step=ck["step"], dataset_id=ck["dataset_id"],
                                       spacing_m=ck["meta"]["spacing_m"], face_intervals=model.grid.n,
                                       checkpoint_sha256=checkpoint_sha,
                                       initialized_from=ck.get("initialized_from"),controls=controls,
                                       export_grid="area raster pixel centres; no pole endpoint rows",
                                       geography=measures(metres),
                                       export_land_percent=float((np.cos(np.radians(90-(np.arange(a.height)+.5)*180/a.height))[:,None]*(image>0)).sum()/np.cos(np.radians(90-(np.arange(a.height)+.5)*180/a.height)).sum()/(2*a.height)*100)))
    print(f"Saved {out/'layout.tif'}", flush=True)
    if a.land_percent is not None:
        print(f"Requested land {a.land_percent:.1f}%; native result {controls['achieved_land_percent']:.2f}%; sea-level shift {controls['sea_level_offset_m']:.1f} m", flush=True)


if __name__ == "__main__":
    main()
