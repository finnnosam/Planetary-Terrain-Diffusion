"""Coordinate-aware PaleoDEM import and conservative training geometry."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from planet_diffusion import cube
from toy_meso.sphere import EARTH_RADIUS_M, encode, random_frames, to_latlon

SPACING_M = 23040.0
FACTOR = 7
LAYOUT_SPACING_M = SPACING_M * FACTOR
LAYOUT_N = round(math.pi * EARTH_RADIUS_M / 2 / LAYOUT_SPACING_M)
VERSION = "planet-training-v1"
VALIDATION_AGES = ((100.0, 140.0), (280.0, 320.0))
# Large regions accommodate the entire intermediate target AND condition halo.
HOLDOUT_CAPS = ((55.0, 20.0, 3500000.0), (45.0, -115.0, 3500000.0))


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def fingerprint(path):
    path = Path(path).resolve()
    s = path.stat()
    return {"path": str(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def manifest_id(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def age_split(age):
    if age <= 20:
        return "test"
    if any(lo <= age <= hi for lo, hi in VALIDATION_AGES):
        return "val"
    if age < 40 or any(lo - 20 <= age <= hi + 20 for lo, hi in VALIDATION_AGES):
        return "buffer"
    return "train"


class PaleoRaster:
    """North-up 1-degree point grid: pole nodes and duplicated dateline."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        if self.values.shape != (181, 361) or not np.isfinite(self.values).all():
            raise ValueError("PaleoDEM must be a finite 181 x 361 point grid")
        if not np.allclose(self.values[:, 0], self.values[:, -1], atol=1e-3):
            raise ValueError("PaleoDEM longitude endpoints disagree")

    @classmethod
    def from_file(cls, path):
        import rasterio
        with rasterio.open(path) as ds:
            first = ds.xy(0, 0)
            last = ds.xy(ds.height - 1, ds.width - 1)
            if not np.allclose((*first, *last), (-180, 90, 180, -90), atol=1e-6):
                raise ValueError(f"Unexpected PaleoDEM coordinates: {path}")
            values = ds.read(1, masked=True)
            if np.ma.getmaskarray(values).any():
                raise ValueError(f"PaleoDEM contains missing elevations: {path}")
            return cls(np.asarray(values))

    def sample(self, lat, lon):
        x = (np.asarray(lon) + 180) % 360
        y = np.clip(90 - np.asarray(lat), 0, 180)
        ix = np.floor(x).astype(np.int64)
        iy = np.minimum(np.floor(y).astype(np.int64), 179)
        fx, fy = x - ix, y - iy
        a = self.values
        return ((1 - fy) * ((1 - fx) * a[iy, ix] + fx * a[iy, ix + 1])
                + fy * ((1 - fx) * a[iy + 1, ix] + fx * a[iy + 1, ix + 1])).astype(np.float32)


def random_rotation(rng):
    c, a, b = random_frames(rng, 1)
    rotation = np.stack((a[0], b[0], c[0]))
    if rng.random() < .5:
        rotation[0] *= -1
    return rotation


@lru_cache(maxsize=4)
def layout_geometry(n=LAYOUT_N, sub=4):
    """Node-centred footprints, with half footprints at shared face boundaries."""
    centres = np.arange(n + 1)
    low = np.maximum(centres - .5, 0)
    high = np.minimum(centres + .5, n)
    samples = low[:, None] + (high - low)[:, None] * (np.arange(sub) + .5) / sub
    y = samples[:, None, :, None]
    x = samples[None, :, None, :]
    points = np.stack([cube.directions(f, y, x, n) for f in range(6)])
    weights = cube.area_weight(y, x, n)
    return points, weights


def layout_target(raster, rotation, n=LAYOUT_N, sub=4):
    points, weights = layout_geometry(n, sub)
    rotated = np.einsum("...i,ij->...j", points, rotation, optimize=False)
    lat, lon = to_latlon(rotated)
    metres = (raster.sample(lat, lon) * weights).sum(axis=(-2, -1)) / weights.sum(axis=(-2, -1))
    # Partial dual cells from adjacent faces combine at a shared node.
    metres = cube.identify(metres[None])[0]
    return encode(metres).astype(np.float32)


def node_weights(n=LAYOUT_N):
    i = np.arange(n + 1)
    w = cube.area_weight(i[:, None], i[None, :], n)
    # Half dual cells on edges; quarter cells at corners.
    w[[0, -1], :] *= .5
    w[:, [0, -1]] *= .5
    return np.broadcast_to(w, (6, n + 1, n + 1)).copy().astype(np.float32)


def geographic_split(points):
    """Entire footprint must fit one split; 25 km guards cover cell extents."""
    expanded = np.zeros(points.shape[:-1], bool)
    contracted = np.zeros_like(expanded)
    for lat, lon, radius in HOLDOUT_CAPS:
        la, lo = np.radians([lat, lon])
        centre = np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
        dot = np.sum(points * centre, axis=-1)
        expanded |= dot >= np.cos((radius + 25000) / EARTH_RADIUS_M)
        contracted |= dot >= np.cos((radius - 25000) / EARTH_RADIUS_M)
    if not expanded.any():
        return "train"
    if contracted.all():
        return "val"
    return None


def bridge_patch(raster, frame, size=112, factor=FACTOR, spacing=SPACING_M, sub=6):
    """Gnomonic, Jacobian-weighted cell means and exact block-mean condition."""
    from toy_meso.sphere import patch_points
    full = size + 2 * factor
    p = patch_points(frame, full, spacing, sub)
    lat, lon = to_latlon(p)
    # Solid angle Jacobian of the tangent-plane projection.
    jac = np.maximum(np.sum(p * frame[0], axis=-1), 0) ** 3
    a = raster.sample(lat, lon)
    values = (a * jac).reshape(full, sub, full, sub).sum(axis=(1, 3))
    weights = jac.reshape(full, sub, full, sub).sum(axis=(1, 3))
    metres = values / weights
    # Coarse blocks are means in physical metres, before nonlinear encoding.
    cside = full // factor
    coarse = (metres * weights).reshape(cside, factor, cside, factor).sum(axis=(1, 3))
    coarse /= weights.reshape(cside, factor, cside, factor).sum(axis=(1, 3))
    return encode(metres[factor:-factor, factor:-factor]).astype(np.float16), encode(coarse).astype(np.float32)
