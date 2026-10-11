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
from .sphere import GlobalRaster, encode, decode, to_latlon, EARTH_RADIUS_M
from .conditioning import (make_source, source_settings, settings_differences,
                           RANDOM_VERSIONS, RANDOM_VERSION, LEGACY_RANDOM_VERSION)
from .input_review import prepare_review, REVIEW_EXIT_CODE

VERSION = "meso-tiled-v2-cell-centres"
NOISE_BLOCK = 64


def save_array(path, values):
    """Only expose complete arrays, so terminating a run cannot poison its cache."""
    path = Path(path)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


class Grid:
    """Fixed tile grid on a cube with N fine intervals per face."""

    def __init__(self, N, size, stride, factor=8):
        if factor < 1 or N % factor or size % factor:
            raise ValueError("fine intervals and tile size must be multiples of factor")
        if not 0 < stride < size or stride % factor:
            raise ValueError("stride must be a positive multiple of factor smaller than the tile")
        self.N, self.size, self.stride, self.factor = N, size, stride, factor
        self.coarse_halo = int(math.ceil(size / factor)) + 1
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


def coarse_condition_cells(raster, n, factor, halo, sub=8):
    """Per-face cell averages at the training centres, including chart halos.

    Fine nodes 0..factor-1 have their coarse average centred at (factor-1)/2.
    Unlike a node field, these cell-centred charts have no shared edge entries:
    halo cells are sampled on the sphere in their own face's extended chart.
    """
    centres = np.arange(-halo, n + halo + 1)
    offset = (factor - 1) / (2 * factor)
    offsets = (np.arange(sub) + .5) / sub - .5
    out = np.empty((1, 6, len(centres), len(centres)), np.float32)
    for face in range(6):
        for start in range(0, len(centres), 16):
            ys = centres[start:start + 16]
            y = ys[:, None, None, None] + offset + offsets[None, None, :, None]
            x = centres[None, :, None, None] + offset + offsets[None, None, None, :]
            lat, lon = to_latlon(cube.directions(face, y, x, n))
            out[0, face, start:start + len(ys)] = raster.sample(lat, lon).mean(axis=(-2, -1))
    return out


def ensure_coarse_condition(folder, grid, keys, source_factory, sub=8):
    """Prepare only cells read by requested tiles, retaining them for later regions."""
    folder = Path(folder)
    path, valid_path = folder / "coarse.npy", folder / "coarse-valid.npy"
    if path.exists() and not valid_path.exists():
        # Existing v2 caches contain the complete globe.
        return np.load(path)
    n, halo, factor = grid.N // grid.factor, grid.coarse_halo, grid.factor
    side = n + 1 + 2 * halo
    shape = (1, 6, side, side)
    required = np.zeros(shape[1:], bool)
    count = grid.size // factor + 2
    for face, ky, kx in keys:
        y = grid.starts[ky] // factor - 1 + halo
        x = grid.starts[kx] // factor - 1 + halo
        required[face, y:y + count, x:x + count] = True
    if path.exists():
        values, valid = np.load(path), np.load(valid_path)
        if values.shape != shape or valid.shape != shape[1:] or valid.dtype != np.bool_:
            raise ValueError("The saved coarse input cache has an unexpected shape or mask")
    else:
        values, valid = np.zeros(shape, np.float32), np.zeros(shape[1:], bool)
        # Create the false mask first. Stop between publishing the array and its
        # updated mask must never make an unfinished array appear complete.
        save_array(valid_path, valid)
    missing = required & ~valid
    print(f"coarse cells needed {required.sum()}, cached {(required & valid).sum()}, "
          f"to compute {missing.sum()}", flush=True)
    if not missing.any():
        return values
    raster = source_factory()
    offsets = (np.arange(sub) + .5) / sub - .5
    centre = (factor - 1) / (2 * factor)
    for face in range(6):
        rows, cols = np.nonzero(missing[face])
        for start in range(0, len(rows), 1024):
            r, c = rows[start:start + 1024], cols[start:start + 1024]
            y = (r - halo)[:, None, None] + centre + offsets[None, :, None]
            x = (c - halo)[:, None, None] + centre + offsets[None, None, :]
            lat, lon = to_latlon(cube.directions(face, y, x, n))
            values[0, face, r, c] = raster.sample(lat, lon).mean(axis=(-2, -1))
    valid |= missing
    save_array(path, values)
    save_array(valid_path, valid)
    return values


def condition_at(coarse_enc, factor, grid, face, y, x):
    """Training's cell-centre interpolation at unshifted fine chart coordinates."""
    offset = (factor - 1) / (2 * factor)
    y, x = np.broadcast_arrays(np.asarray(y) / factor - offset + grid.coarse_halo,
                               np.asarray(x) / factor - offset + grid.coarse_halo)
    iy, ix = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    a = coarse_enc[0, face]
    if np.any(iy < 0) or np.any(ix < 0) or np.any(iy + 1 >= a.shape[0]) or np.any(ix + 1 >= a.shape[1]):
        raise ValueError("condition lookup exceeds the saved chart halo")
    fy, fx = y - iy, x - ix
    return ((1 - fy) * ((1 - fx) * a[iy, ix] + fx * a[iy, ix + 1]) +
            fy * ((1 - fx) * a[iy + 1, ix] + fx * a[iy + 1, ix + 1])).astype(np.float32)


def tile_condition(coarse_enc, factor, grid, f, ky, kx):
    """Condition on the original fine nodes, aligned exactly as in training."""
    ys = (grid.starts[ky] + np.arange(grid.size))[:, None]
    xs = (grid.starts[kx] + np.arange(grid.size))[None, :]
    return condition_at(coarse_enc, factor, grid, f, ys, xs)


def up_at(coarse_enc, factor, grid, flat):
    """Upsampled condition at fine nodes, evaluated at each node's owning face entry."""
    f, y, x = grid.unflat(grid.canonical(flat))
    out = np.empty(f.shape, np.float32)
    for face in np.unique(f):
        sel = f == face
        out[sel] = condition_at(coarse_enc, factor, grid, int(face), y[sel], x[sel])
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
        save_array(p, arr.astype(np.float32))


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
            sg = sigma.to(dtype=torch.float32).expand(x.shape[0])
            return model(x.float(), sg, cin).double()

        with torch.no_grad(), autocast(device):
            x = heun_sample(denoise, x0.shape, steps, device=device, initial=x0)
        # Cache complete encoded elevations. Face charts have different coarse
        # cell centres at an edge; blend their full predictions, not residuals
        # over a baseline belonging to some other face.
        out = cond + x[:, 0].cpu().numpy() / sd * s["res_std"]
        for k, arr in zip(chunk, out):
            cache.save(k, arr)
        done = i + len(chunk)
        rate = done / (time.perf_counter() - started)
        print(f"tiles {done}/{len(todo)}  ({rate:.2f}/s, ~{(len(todo) - done) / rate / 60:.0f} min left)",
              flush=True)


def blend(keys, grid, cache):
    """Weighted encoded elevation sums, combined across shared edge nodes."""
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


def blend_region(keys, grid, cache, nodes):
    """Blend requested nodes in the same order as the dense whole-globe arrays."""
    _, y, x = grid.unflat(nodes)
    acc, weights = np.zeros(len(nodes), np.float32), np.zeros(len(nodes), np.float32)
    window = cube.linear_weight_window(grid.size).astype(np.float32)
    for face, ky, kx in keys:
        y0, x0 = grid.starts[ky], grid.starts[kx]
        first = np.searchsorted(nodes, grid.flat(face, max(0, y0), 0))
        last = np.searchsorted(nodes, grid.flat(face, min(grid.N + 1, y0 + grid.size), 0))
        selected = np.arange(first, last)
        selected = selected[(x[selected] >= x0) & (x[selected] < x0 + grid.size)]
        iy, ix = y[selected] - y0, x[selected] - x0
        weight = window[iy, ix]
        acc[selected] += weight * cache.load((face, ky, kx))[iy, ix]
        weights[selected] += weight
    # All entries of a shared physical node are included in nodes. Retain the
    # dense implementation's member order and float64 accumulation at edges.
    positions = np.searchsorted(nodes, grid.members_all)
    present = positions < len(nodes)
    present[present] &= nodes[positions[present]] == grid.members_all[present]
    positions, groups = positions[present], grid.ids_all[present]
    for values in (acc, weights):
        sums = np.zeros(grid.n_groups, np.float64)
        np.add.at(sums, groups, values[positions])
        values[positions] = sums[groups]
    if np.any(weights <= 0):
        raise RuntimeError("incomplete tile coverage for the requested region")
    return decode(acc / weights).astype(np.float32)


def sample_region(nodes, values, grid, points):
    face, y, x = cube.coordinates(points, grid.N)
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    jy, jx = np.minimum(iy + 1, grid.N), np.minimum(ix + 1, grid.N)
    fy, fx = y - iy, x - ix

    def read(rows, cols):
        flat = grid.flat(face, rows, cols)
        positions = np.searchsorted(nodes, flat)
        if np.any(positions >= len(nodes)) or np.any(nodes[positions] != flat):
            raise RuntimeError("output sampling requested a node outside the region")
        return values[positions]

    return ((1 - fy) * ((1 - fx) * read(iy, ix) + fx * read(iy, jx))
            + fy * ((1 - fx) * read(jy, ix) + fx * read(jy, jx))).astype(np.float32)


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
    p.add_argument("--dem", help="2:1 PNG sketch or 2:1 global elevation GeoTIFF (metres)")
    p.add_argument("--source", choices=("map", "random", "mixed"),
                   help="input source (default: map with --dem, otherwise random)")
    p.add_argument("--map-weight", type=float, default=.5,
                   help="map contribution for mixed input, 0..1 (default .5); PNG alpha also masks the map")
    p.add_argument("--random-version", choices=RANDOM_VERSIONS,
                   help="random source version (existing runs retain their saved version)")
    p.add_argument("--pause-for-input", action="store_true", help="save input previews and stop until approved")
    p.add_argument("--approve-input", action="store_true", help="approve the existing unchanged input preview and generate")
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
    a.source = a.source or ("map" if a.dem else "random")
    if a.source != "random" and not a.dem:
        p.error("map and mixed inputs require --dem")
    if not math.isfinite(a.map_weight) or not 0 <= a.map_weight <= 1:
        p.error("--map-weight must be between 0 and 1")
    if a.source == "random":
        a.dem = None
        # These controls only interpret PNGs and must not affect a random run.
        a.black_metres, a.white_metres, a.ocean_metres = -2000., 4000., None
    if a.steps < 8:
        p.error("--steps must be at least 8")
    if a.batch < 1:
        p.error("--batch must be positive")
    if not 0 <= a.seed < 2**64:
        p.error("--seed must be an unsigned 64-bit integer")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = pick_device(a.device)
    ckpt = Path(a.run) / a.checkpoint
    model, norm, ck = load_checkpoint(ckpt, "cpu")
    meta = ck["meta"]
    h, factor, size = meta["spacing_m"], meta["factor"], meta["size"]
    stride = size * 3 // 4 if a.stride is None else a.stride
    if not 0 < stride < size or stride % factor:
        p.error("--stride must be a positive multiple of the model's factor, smaller than its tile")
    quarter = math.pi * a.radius_metres / 2
    n = max(4, int(round(quarter / (h * factor))))
    N = n * factor
    spacing = quarter / N
    grid = Grid(N, size, stride, factor)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"version": VERSION, "checkpoint_sha256": file_sha(ckpt), "train_step": ck["step"],
                "black_metres": a.black_metres, "white_metres": a.white_metres,
                "ocean_metres": a.ocean_metres, "radius_metres": a.radius_metres, "seed": a.seed,
                "steps": a.steps, "tile": size, "stride": stride, "coarse_intervals": n,
                "fine_intervals": N, "spacing_m": h, "factor": factor, "noise_block": NOISE_BLOCK,
                "batch": a.batch, "coarse_halo": grid.coarse_halo,
                "condition_cell_offset_fine": (factor - 1) / 2,
                "tile_values": "encoded_elevation"}
    sfile = out / "settings.json"
    old = json.loads(sfile.read_text()) if sfile.exists() else None
    random_version = a.random_version or (old or {}).get("random_source_version", RANDOM_VERSION)
    settings.update(source_settings(a.source, a.dem, a.map_weight, random_version))
    if sfile.exists():
        diff = settings_differences(old, settings)
        if diff:
            raise SystemExit(f"{out} was made with different settings {diff}; use a new --out folder")
    else:
        temporary = sfile.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(dict(settings, dem=a.dem, checkpoint=str(ckpt)), indent=2))
        temporary.replace(sfile)

    print(f"device {device}; cube {n} coarse / {N} fine intervals per face "
          f"(~{quarter / n / 1000:.1f} km condition, ~{spacing / 1000:.2f} km nominal); "
          f"{len(grid.starts)}x{len(grid.starts)} tiles per face, tile {size}, stride {stride}", flush=True)
    if a.source != "map" and random_version == LEGACY_RANDOM_VERSION:
        print("This run retains the earlier smooth random guide. Choose New run for the revised multiscale guide.", flush=True)

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
    print(f"preparing {a.source} elevation input"
          + (f" ({100 * a.map_weight:g}% map)" if a.source == "mixed" else ""), flush=True)
    coarse_m = ensure_coarse_condition(out, grid, keys,
        lambda: make_source(a.source, a.seed, a.dem, a.black_metres, a.white_metres,
                             a.ocean_metres, a.map_weight, random_version))
    coarse_enc = encode(coarse_m)
    if a.source != "map" or a.pause_for_input or a.approve_input:
        review = prepare_review(out, settings, grid, coarse_enc, lat, lon, condition_at, a.approve_input)
        if a.pause_for_input and not review["approved"]:
            print(f"input review ready: {out / 'input.preview.png'}; "
                  "paused before terrain generation. Approve this layout to continue.", flush=True)
            return REVIEW_EXIT_CODE
    cache = TileCache(out / "tiles")
    if any(not cache.has(key) for key in keys):
        model = model.to(device)
    compute_tiles(keys, model, norm, coarse_enc, factor, grid, a.seed, cache, a.steps, a.batch, device)

    print("blending tiles", flush=True)
    if a.bounds and len(needed) <= 6 * (N + 1)**2 // 8:
        metres_nodes = blend_region(keys, grid, cache, needed)
        sample_output = lambda points: sample_region(needed, metres_nodes, grid, points)
    else:
        acc, wsum = blend(keys, grid, cache)
        if a.bounds:
            field = np.zeros_like(acc)
            f, y, x = grid.unflat(needed)
            if np.any(wsum[0, f, y, x] <= 0):
                raise RuntimeError("incomplete tile coverage for the requested region")
            field[0, f, y, x] = acc[0, f, y, x] / wsum[0, f, y, x]
        else:
            if np.any(wsum <= 0):
                raise RuntimeError("incomplete tile coverage")
            field = (acc / wsum).astype(np.float32)
        del acc, wsum
        metres_field = decode(field).astype(np.float32)
        sample_output = lambda points: cube.sample(metres_field, points)[0]

    print(f"writing {target} ({width} x {height})", flush=True)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    crs = rasterio.crs.CRS.from_string(f"+proj=longlat +R={a.radius_metres:.12g} +no_defs")
    tags = dict(settings, dem=a.dem, checkpoint=str(ckpt), units="metres",
                sampling="bilinear from cube nodes at pixel centres")
    preview_rows = []
    pstep = max(1, height // 1024)
    # Publish the TIFF only after both exports are complete. Stop during export
    # leaves a temporary file that a resumed run can safely replace.
    pending = target.parent / ".incomplete"
    pending.mkdir(exist_ok=True)
    temporary_target = pending / target.name
    with rasterio.open(temporary_target, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs=crs, transform=from_origin(*transform_origin, *px), compress="deflate",
                       predictor=3, tiled=True, BIGTIFF="IF_SAFER") as dst:
        dst.update_tags(AREA_OR_POINT="Area", **{k: str(v) for k, v in tags.items()})
        for start in range(0, height, 128):
            rows = np.arange(start, min(height, start + 128))
            vals = sample_output(pixel_points(rows)).astype(np.float32)
            dst.write(vals, 1, window=Window(0, start, width, len(rows)))
            keep = rows % pstep == 0
            if keep.any():
                preview_rows.append(vals[keep][:, ::pstep])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import shaded
    prev = np.concatenate(preview_rows)
    temporary_preview = pending / target.with_suffix(".preview.png").name
    plt.imsave(temporary_preview, shaded(prev, spacing * pstep))
    temporary_preview.replace(target.with_suffix(".preview.png"))
    temporary_target.replace(target)
    print(f"wrote {target} and {target.with_suffix('.preview.png').name}")


if __name__ == "__main__":
    raise SystemExit(main())
