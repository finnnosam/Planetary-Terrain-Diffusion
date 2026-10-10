"""Run the toy refinement model over a whole planet on the cubed sphere.

The input (a 2:1 PNG sketch or elevation GeoTIFF) is area-averaged onto a coarse
cube grid at the model's condition spacing (~160 km). The model then generates
the ~20 km field on a cube grid 8x finer. Overlapping 64x64 windows read across
face edges and are blended into one shared surface before every sampler step
(planet_diffusion.cube.consensus), so there are no face seams or pole seams.

  python -m toy_meso.planet --run outputs\\toy_meso\\run1 --dem sketch.png ^
      --black-metres -6000 --white-metres 6000 --out outputs\\toy_meso\\planet1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from planet_diffusion import cube
from .common import load_checkpoint, autocast, pick_device
from .model import sigma_schedule
from .sphere import GlobalRaster, encode, decode, to_latlon, EARTH_RADIUS_M

TILE, STRIDE = 64, 48


def coarse_condition(raster, n, sub=8):
    """Area-averaged metres at every coarse cube node (sub x sub samples per cell)."""
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    out = np.empty((1, 6, n + 1, n + 1), np.float32)
    rows = np.arange(n + 1)
    for f in range(6):
        for start in range(0, n + 1, 16):
            ys = rows[start:start + 16]
            y = ys[:, None, None, None] + offsets[None, None, :, None]
            x = rows[None, :, None, None] + offsets[None, None, None, :]
            p = cube.directions(f, y, x, n)
            lat, lon = to_latlon(p)
            out[0, f, start:start + len(ys)] = raster.sample(lat, lon).mean(axis=(-2, -1))
    return cube.identify(out)


def generate_planet(model, norm, cond_metres, factor, steps=32, seed=0, batch=32, device="cpu",
                    progress=print):
    """Returns (generated encoded field, upsampled encoded condition), both (1, 6, N+1, N+1)."""
    n = cond_metres.shape[-1] - 1
    N = n * factor
    up = cube.resize(encode(cond_metres), N).astype(np.float64)  # bilinear on the shared sphere
    cin_field = ((up - norm.s["cond_mean"]) / norm.s["cond_std"]).astype(np.float32)
    positions = list(cube.tile_positions(N, TILE, STRIDE))
    locations = [(f, y, x) for f in range(6) for y in positions for x in positions]
    rng = np.arange(TILE)
    cins = {loc: cube.read(cin_field, loc[0], loc[1] + rng[:, None], loc[2] + rng[None, :])
            for loc in locations}
    window = cube.linear_weight_window(TILE)
    calls = [0]

    def denoise(x, sigma):
        def predict_batch(contexts, locs):
            xt = torch.from_numpy(np.asarray(contexts, np.float32)).to(device)
            ct = torch.from_numpy(np.stack([cins[loc] for loc in locs])).to(device)
            s = torch.full((len(locs),), float(sigma), device=device)
            with torch.no_grad(), autocast(device):
                return model(xt, s, ct).float().cpu().numpy()
        out = cube.consensus(x.astype(np.float32), None, TILE, STRIDE, batch_size=batch,
                             predict_batch=predict_batch, weight_window=window, shared_weights=True)
        calls[0] += 1
        return out.astype(np.float64)

    t = sigma_schedule(steps).numpy()
    x = cube.noise(seed, 0, 1, N).astype(np.float64) * t[0]
    started = time.perf_counter()
    for i in range(steps):
        d = (x - denoise(x, t[i])) / t[i]
        nxt = x + (t[i + 1] - t[i]) * d
        if t[i + 1] > 0:
            d2 = (nxt - denoise(nxt, t[i + 1])) / t[i + 1]
            nxt = x + (t[i + 1] - t[i]) * (d + d2) / 2
        x = cube.identify(nxt).astype(np.float64)
        progress(f"step {i + 1}/{steps}: {len(locations)} windows per pass, "
                 f"{time.perf_counter() - started:.0f} s")
    generated = up + x / norm.sd * norm.s["res_std"]
    return cube.identify(generated), up


def equirectangular(field, height):
    """Bilinear samples of a cube field at pixel centres of a 2:1 global grid."""
    out = np.empty((height, 2 * height), np.float32)
    lon = np.radians((np.arange(2 * height) + 0.5) * 180 / height - 180)
    for start in range(0, height, 128):
        rows = np.arange(start, min(height, start + 128))
        lat = np.radians(90 - (rows + 0.5) * 180 / height)[:, None]
        p = np.stack(np.broadcast_arrays(np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                                         np.sin(lat)), axis=-1)
        out[start:start + len(rows)] = cube.sample(field, p)[0]
    return out


def write_tiff(path, values, radius, tags):
    import rasterio
    from rasterio.transform import from_origin
    h, w = values.shape
    crs = rasterio.crs.CRS.from_string(f"+proj=longlat +R={radius:.12g} +no_defs")
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=crs,
                       transform=from_origin(-180, 90, 360 / w, 180 / h), compress="deflate",
                       predictor=3, tiled=True) as dst:
        dst.update_tags(AREA_OR_POINT="Area", units="metres", **{k: str(v) for k, v in tags.items()})
        dst.write(values, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="trained run folder")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--dem", required=True, help="2:1 PNG sketch or 2:1 global elevation GeoTIFF (metres)")
    p.add_argument("--black-metres", type=float, default=-2000.0, help="PNG only: elevation of black")
    p.add_argument("--white-metres", type=float, default=4000.0, help="PNG only: elevation of white")
    p.add_argument("--ocean-metres", type=float, help="PNG only: elevation for pure-black pixels")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--radius-metres", type=float, default=EARTH_RADIUS_M)
    p.add_argument("--height", type=int, help="output GeoTIFF height (default: about the model spacing)")
    p.add_argument("--steps", type=int, default=32, help="sampler steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32, help="windows per model call")
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
    quarter = np.pi * a.radius_metres / 2
    n = max(4, int(round(quarter / (h * f))))
    N = n * f
    spacing = quarter / N
    height = a.height or int(round(np.pi * a.radius_metres / spacing))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tif = out / "planet.tif"
    if tif.exists():
        raise FileExistsError(f"{tif} exists; choose a new --out folder")
    print(f"device {device}; cube {n} coarse / {N} fine intervals per face "
          f"(~{quarter / n / 1000:.0f} km condition, ~{spacing / 1000:.1f} km detail nominal); "
          f"output {2 * height} x {height}")

    raster = GlobalRaster.from_file(a.dem, a.black_metres, a.white_metres, a.ocean_metres)
    cond = coarse_condition(raster, n)
    gen, up = generate_planet(model, norm, cond, f, a.steps, a.seed, a.batch, device,
                              progress=lambda s: print(s, flush=True))
    metres = decode(gen)
    np.save(out / "cube-elevation.npy", metres.astype(np.float32))
    tags = {"source": a.dem, "black_metres": a.black_metres, "white_metres": a.white_metres,
            "ocean_metres": a.ocean_metres, "radius_metres": a.radius_metres, "seed": a.seed,
            "steps": a.steps, "checkpoint": str(Path(a.run) / a.checkpoint), "train_step": ck["step"],
            "cube_coarse_intervals": n, "cube_fine_intervals": N, "nominal_spacing_m": spacing,
            "model": "toy_meso fixed-scale refinement (20 km trained, 160 km condition)"}
    image = equirectangular(metres, height)
    write_tiff(tif, image, a.radius_metres, tags)
    cond_image = equirectangular(decode(up), height)
    write_tiff(out / "condition.tif", cond_image, a.radius_metres, dict(tags, product="bilinear condition"))
    (out / "planet.json").write_text(json.dumps(tags, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import shaded
    ps = 360 / (2 * height) * np.pi / 180 * a.radius_metres
    fig, ax = plt.subplots(2, 1, figsize=(14, 14.5))
    for axis, (name, arr) in zip(ax, (("input averaged to the condition grid (bilinear)", cond_image),
                                      ("toy model", image))):
        axis.imshow(shaded(arr, ps), extent=(-180, 180, -90, 90))
        axis.set_title(name)
    fig.tight_layout(); fig.savefig(out / "planet-preview.png", dpi=80); plt.close(fig)
    print(f"wrote {tif}, condition.tif, planet-preview.png, cube-elevation.npy")


if __name__ == "__main__":
    main()
