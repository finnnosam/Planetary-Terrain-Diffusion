"""Whole-planet (or regional) generation with bounded tile dependencies.

Every fine cube node belongs to a fixed grid of overlapping tiles. Each tile is
denoised to completion on its own, from:
  * initial noise keyed to absolute cube positions (seed, face, block), so
    overlapping tiles share noise where they overlap, and
  * the condition (input averaged to the model's coarse spacing) read at the
    tile's own position, with halos crossing face edges in 3-D.
Finished tiles are blended with linear weights (as the upstream coarse stage
does). A node therefore depends only on the few tiles covering it, so:
  * any region can be generated alone and matches the full planet exactly,
  * tiles are cached on disk: runs resume, and later regions reuse them.

  python -m toy_meso.planet_tiled --run outputs\\meso\\run2-lowlr --dem sketch.png ^
      --black-metres -6000 --white-metres 6000 --out outputs\\meso\\planetA
  python -m toy_meso.planet_tiled ... --out outputs\\meso\\planetA --bounds 0 40 20 55 --name alps.tif
"""
import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from planet_diffusion import cube
from .common import load_checkpoint, autocast, pick_device
from .model import heun_sample
from .sphere import GlobalRaster, encode, decode, EARTH_RADIUS_M
from .planet import coarse_condition, write_tiff

VERSION = "meso-tiled-v1"
NOISE_BLOCK = 64


class Grid:
    """Fixed tile grid on a cube with N fine intervals per face."""

    def __init__(self, N, size, stride):
        self.N, self.size, self.stride = N, size, stride
        self.starts = np.array(list(cube.tile_positions(N, size, stride)))
        members, ids, owners, _ = cube.edge_groups(N)
        order = np.argsort(members)
        self._members = members[order]
        self._owner = owners[ids][order]
        self._group = ids[order]
        self.members_all, self.ids_all, self.n_groups = members, ids, len(owners)

    def flat(self, f, y, x):
        return (np.asarray(f) * (self.N + 1) + y) * (self.N + 1) + x

    def unflat(self, flat):
        f, r = np.divmod(flat, (self.N + 1) ** 2)
        y, x = np.divmod(r, self.N + 1)
        return f, y, x

    def canonical(self, flat):
        """Map shared edge/corner nodes to their owning face entry."""
        flat = np.asarray(flat, dtype=np.int64)
        pos = np.clip(np.searchsorted(self._members, flat), 0, len(self._members) - 1)
        hit = self._members[pos] == flat
        return np.where(hit, self._owner[pos], flat)

    def group_members(self, flat):
        """All face entries of the physical nodes in flat (edge nodes appear on 2-3 faces)."""
        flat = np.unique(np.asarray(flat, dtype=np.int64))
        pos = np.clip(np.searchsorted(self._members, flat), 0, len(self._members) - 1)
        hit = self._members[pos] == flat
        groups = np.unique(self._group[pos[hit]])
        extra = self.members_all[np.isin(self.ids_all, groups)]
        return np.unique(np.concatenate([flat, extra]))

    def covering_tiles(self, flat):
        """Tile keys (face, ky, kx) whose in-face footprint contains any of the nodes."""
        f, y, x = self.unflat(np.asarray(flat, dtype=np.int64))
        p0, s, size = self.starts[0], self.stride, self.size
        keys = set()
        reach = int(math.ceil(size / s))
        ky_max, kx_max = (y - p0) // s, (x - p0) // s
        for dy in range(reach + 1):
            for dx in range(reach + 1):
                ky, kx = ky_max - dy, kx_max - dx
                ok = ((ky >= 0) & (kx >= 0) & (ky < len(self.starts)) & (kx < len(self.starts)))
                ky_c, kx_c = np.clip(ky, 0, len(self.starts) - 1), np.clip(kx, 0, len(self.starts) - 1)
                ys, xs = self.starts[ky_c], self.starts[kx_c]
                ok &= (ys <= y) & (y < ys + size) & (xs <= x) & (x < xs + size)
                keys.update(zip(f[ok].tolist(), ky[ok].tolist(), kx[ok].tolist()))
        return sorted(keys)

    def all_tiles(self):
        n = len(self.starts)
        return [(f, ky, kx) for f in range(6) for ky in range(n) for kx in range(n)]


@lru_cache(maxsize=512)
def _noise_block(seed, f, by, bx):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x6D65736F, f, by, bx])))
    return rng.standard_normal((NOISE_BLOCK, NOISE_BLOCK), dtype=np.float32)


def noise_at(grid, seed, flat):
    """Position-keyed unit Gaussian at fine nodes; shared nodes get one draw."""
    flat = grid.canonical(flat)
    f, y, x = grid.unflat(flat)
    out = np.empty(flat.shape, np.float32)
    by, bx = y // NOISE_BLOCK, x // NOISE_BLOCK
    keys = f * 1_000_000 + by * 1000 + bx
    for k in np.unique(keys):
        sel = keys == k
        block = _noise_block(seed, int(k // 1_000_000), int(k // 1000 % 1000), int(k % 1000))
        out[sel] = block[y[sel] % NOISE_BLOCK, x[sel] % NOISE_BLOCK]
    return out


def tile_nodes(grid, f, ky, kx):
    """Nearest fine node (flat index) for every pixel of a tile, crossing faces in 3-D."""
    N, size = grid.N, grid.size
    ys = grid.starts[ky] + np.arange(size)[:, None]
    xs = grid.starts[kx] + np.arange(size)[None, :]
    ys, xs = np.broadcast_arrays(ys, xs)
    if ys.min() >= 0 and xs.min() >= 0 and ys.max() <= N and xs.max() <= N:
        return grid.flat(f, ys, xs)
    p = cube.directions(f, ys, xs, N, normalize=False)
    face, fy, fx = cube.coordinates(p, N)
    return grid.flat(face, np.floor(fy + .5).astype(np.int64), np.floor(fx + .5).astype(np.int64))


def tile_condition(coarse_enc, factor, grid, f, ky, kx):
    """Bilinear coarse condition (encoded) at the tile's pixels; halos read across faces."""
    ys = (grid.starts[ky] + np.arange(grid.size))[:, None] / factor
    xs = (grid.starts[kx] + np.arange(grid.size))[None, :] / factor
    return cube.read(coarse_enc, f, ys, xs)[0]


def up_at(coarse_enc, factor, grid, flat):
    """Upsampled condition at fine nodes, evaluated at each node's owning face entry."""
    f, y, x = grid.unflat(grid.canonical(flat))
    out = np.empty(f.shape, np.float32)
    for face in np.unique(f):
        sel = f == face
        out[sel] = cube.read(coarse_enc, int(face), y[sel] / factor, x[sel] / factor)[0]
    return out


class TileCache:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        f, ky, kx = key
        return self.folder / f"f{f}" / f"{ky:03d}_{kx:03d}.npy"

    def has(self, key):
        return self.path(key).exists()

    def load(self, key):
        return np.load(self.path(key))

    def save(self, key, arr):
        p = self.path(key)
        p.parent.mkdir(exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, arr.astype(np.float32))
        tmp.replace(p)


def compute_tiles(keys, model, norm, coarse_enc, factor, grid, seed, cache, steps, batch, device):
    todo = [k for k in keys if not cache.has(k)]
    print(f"tiles needed {len(keys)}, cached {len(keys) - len(todo)}, to compute {len(todo)}", flush=True)
    if not todo:
        return
    sd, s = norm.sd, norm.s
    started = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        padded = chunk + [chunk[0]] * (batch - len(chunk))  # fixed batch size -> same numerics
        noise = np.stack([noise_at(grid, seed, tile_nodes(grid, *k)) for k in padded])
        cond = np.stack([tile_condition(coarse_enc, factor, grid, *k) for k in padded])
        cin = torch.from_numpy((cond - s["cond_mean"]) / s["cond_std"]).float()[:, None].to(device)
        x0 = torch.from_numpy(noise)[:, None].to(device=device, dtype=torch.float64)

        def denoise(x, sigma):
            with torch.no_grad(), autocast(device):
                sg = torch.full((x.shape[0],), float(sigma), device=device)
                return model(x.float(), sg, cin).double()

        x = heun_sample(denoise, x0.shape, steps, device=device, initial=x0)
        out = x[:, 0].cpu().numpy()
        for k, arr in zip(chunk, out):
            cache.save(k, arr)
        done = i + len(chunk)
        rate = done / (time.perf_counter() - started)
        print(f"tiles {done}/{len(todo)}  ({rate:.2f}/s, ~{(len(todo) - done) / rate / 60:.0f} min left)",
              flush=True)


def blend(keys, grid, cache):
    """Weighted residual sums over the given tiles, combined across shared edge nodes."""
    N, size = grid.N, grid.size
    acc = np.zeros((1, 6, N + 1, N + 1), np.float32)
    wsum = np.zeros((1, 6, N + 1, N + 1), np.float32)
    w = cube.linear_weight_window(size).astype(np.float32)
    for key in keys:  # sorted order -> identical float sums for any request
        f, ky, kx = key
        y0, x0 = grid.starts[ky], grid.starts[kx]
        a, b = max(0, y0), min(N + 1, y0 + size)
        c, d = max(0, x0), min(N + 1, x0 + size)
        pred = cache.load(key)
        acc[0, f, a:b, c:d] += w[a - y0:b - y0, c - x0:d - x0] * pred[a - y0:b - y0, c - x0:d - x0]
        wsum[0, f, a:b, c:d] += w[a - y0:b - y0, c - x0:d - x0]
    for values in (acc, wsum):
        flat = values.reshape(1, -1)
        sums = np.zeros((1, grid.n_groups), np.float64)
        np.add.at(sums[0], grid.ids_all, flat[0, grid.members_all])
        flat[:, grid.members_all] = sums[:, grid.ids_all]
    return acc, wsum


def file_sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="trained run folder")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--dem", required=True, help="2:1 PNG sketch or 2:1 global elevation GeoTIFF (metres)")
    p.add_argument("--black-metres", type=float, default=-2000.0)
    p.add_argument("--white-metres", type=float, default=4000.0)
    p.add_argument("--ocean-metres", type=float)
    p.add_argument("--out", required=True, help="planet folder (settings, tile cache, outputs)")
    p.add_argument("--name", help="output GeoTIFF name (default planet.tif, or region.tif with --bounds)")
    p.add_argument("--bounds", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="generate only this lon/lat box (east < west crosses the date line)")
    p.add_argument("--width", type=int, help="output width in pixels")
    p.add_argument("--height", type=int, help="output height in pixels (default: native spacing)")
    p.add_argument("--radius-metres", type=float, default=EARTH_RADIUS_M)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=32, help="sampler steps per tile")
    p.add_argument("--stride", type=int, help="tile stride in pixels (default 3/4 of the tile size)")
    p.add_argument("--batch", type=int, default=16, help="tiles per model call (fixed; lower if GPU memory spills)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    if a.steps < 8:
        p.error("--steps must be at least 8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = pick_device(a.device)
    ckpt = Path(a.run) / a.checkpoint
    model, norm, ck = load_checkpoint(ckpt, device)
    meta = ck["meta"]
    h, factor, size = meta["spacing_m"], meta["factor"], meta["size"]
    stride = a.stride or size * 3 // 4
    quarter = math.pi * a.radius_metres / 2
    n = max(4, int(round(quarter / (h * factor))))
    N = n * factor
    spacing = quarter / N
    grid = Grid(N, size, stride)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"version": VERSION, "checkpoint_sha256": file_sha(ckpt), "train_step": ck["step"],
                "dem_sha256": file_sha(a.dem), "black_metres": a.black_metres, "white_metres": a.white_metres,
                "ocean_metres": a.ocean_metres, "radius_metres": a.radius_metres, "seed": a.seed,
                "steps": a.steps, "tile": size, "stride": stride, "coarse_intervals": n,
                "fine_intervals": N, "spacing_m": h, "factor": factor, "noise_block": NOISE_BLOCK}
    sfile = out / "settings.json"
    if sfile.exists():
        old = json.loads(sfile.read_text())
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{out} was made with different settings {diff}; use a new --out folder")
    else:
        sfile.write_text(json.dumps(dict(settings, dem=a.dem, checkpoint=str(ckpt)), indent=2))

    print(f"device {device}; cube {n} coarse / {N} fine intervals per face "
          f"(~{quarter / n / 1000:.1f} km condition, ~{spacing / 1000:.2f} km nominal); "
          f"{len(grid.starts)}x{len(grid.starts)} tiles per face, tile {size}, stride {stride}", flush=True)

    cpath = out / "coarse.npy"
    if cpath.exists():
        coarse_m = np.load(cpath)
    else:
        raster = GlobalRaster.from_file(a.dem, a.black_metres, a.white_metres, a.ocean_metres)
        coarse_m = coarse_condition(raster, n)
        np.save(cpath, coarse_m)
    coarse_enc = cube.identify(encode(coarse_m))
    cache = TileCache(out / "tiles")

    if a.bounds:
        west, south, east, north = a.bounds
        if not (-90 <= south < north <= 90):
            p.error("--bounds needs south < north within -90..90")
        span = (east - west) % 360 or 360
        height = a.height or max(1, int(round((north - south) * math.pi * a.radius_metres / 180 / spacing)))
        width = a.width or max(1, int(round(height * span / (north - south))))
        lat = north - (np.arange(height) + 0.5) * (north - south) / height
        lon = west + (np.arange(width) + 0.5) * span / width
        name = a.name or "region.tif"
        transform_origin, px = (west, north), (span / width, (north - south) / height)
    else:
        height = a.height or int(round(math.pi * a.radius_metres / spacing))
        width = 2 * height
        lat = 90 - (np.arange(height) + 0.5) * 180 / height
        lon = -180 + (np.arange(width) + 0.5) * 360 / width
        name = a.name or "planet.tif"
        transform_origin, px = (-180, 90), (360 / width, 180 / height)
    target = out / name
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --name")

    def pixel_points(rows):
        la = np.radians(lat[rows])[:, None]
        lo = np.radians(lon)[None, :]
        return np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), -1)

    # Which tiles does this output need? Every node used by bilinear lookups, all faces of shared nodes.
    if a.bounds:
        needed = []
        for start in range(0, height, 256):
            rows = np.arange(start, min(height, start + 256))
            face, fy, fx = cube.coordinates(pixel_points(rows), N)
            iy, ix = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
            for dy in (0, 1):
                for dx in (0, 1):
                    needed.append(np.unique(grid.flat(face, np.minimum(iy + dy, N), np.minimum(ix + dx, N))))
        needed = grid.group_members(np.concatenate(needed))
        keys = grid.covering_tiles(needed)
    else:
        keys = grid.all_tiles()
    compute_tiles(keys, model, norm, coarse_enc, factor, grid, a.seed, cache, a.steps, a.batch, device)

    print("blending tiles", flush=True)
    acc, wsum = blend(keys, grid, cache)
    if a.bounds:
        field = np.zeros_like(acc)
        f, y, x = grid.unflat(needed)
        if np.any(wsum[0, f, y, x] <= 0):
            raise RuntimeError("incomplete tile coverage for the requested region")
        resid = acc[0, f, y, x] / wsum[0, f, y, x]
        field[0, f, y, x] = up_at(coarse_enc, factor, grid, needed) + resid / norm.sd * norm.s["res_std"]
    else:
        if np.any(wsum <= 0):
            raise RuntimeError("incomplete tile coverage")
        all_nodes = np.arange(6 * (N + 1) ** 2, dtype=np.int64)
        up = np.empty(all_nodes.shape, np.float32)
        for start in range(0, len(all_nodes), 1 << 22):
            sel = all_nodes[start:start + (1 << 22)]
            up[sel] = up_at(coarse_enc, factor, grid, sel)
        field = (up.reshape(acc.shape) + acc / wsum / norm.sd * norm.s["res_std"]).astype(np.float32)
    del acc, wsum
    metres_field = decode(field).astype(np.float32)

    print(f"writing {target} ({width} x {height})", flush=True)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    crs = rasterio.crs.CRS.from_string(f"+proj=longlat +R={a.radius_metres:.12g} +no_defs")
    tags = dict(settings, dem=a.dem, checkpoint=str(ckpt), units="metres",
                sampling="bilinear from cube nodes at pixel centres")
    preview_rows = []
    pstep = max(1, height // 1024)
    with rasterio.open(target, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs=crs, transform=from_origin(*transform_origin, *px), compress="deflate",
                       predictor=3, tiled=True, BIGTIFF="IF_SAFER") as dst:
        dst.update_tags(AREA_OR_POINT="Area", **{k: str(v) for k, v in tags.items()})
        for start in range(0, height, 128):
            rows = np.arange(start, min(height, start + 128))
            vals = cube.sample(metres_field, pixel_points(rows))[0].astype(np.float32)
            dst.write(vals, 1, window=Window(0, start, width, len(rows)))
            keep = rows % pstep == 0
            if keep.any():
                preview_rows.append(vals[keep][:, ::pstep])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import shaded
    prev = np.concatenate(preview_rows)
    plt.imsave(target.with_suffix(".preview.png"), shaded(prev, spacing * pstep))
    print(f"wrote {target} and {target.with_suffix('.preview.png').name}")


if __name__ == "__main__":
    main()
