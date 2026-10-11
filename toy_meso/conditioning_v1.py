"""Elevation-only map, random globe, and mixed input sources for meso."""
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .sphere import GlobalRaster

RANDOM_VERSION = "meso-random-elevation-v1"
# A fixed angular scale gives continents on a whole globe. It must not depend
# on output resolution, the requested region, or the model's tile grid.
NOISE_RADIUS = 32.0
NOISE_FREQUENCY = 1.5
MAP_LEGACY_DEFAULTS = {"input_source": "map", "map_weight": 1.0}
SOURCE_LABELS = {"Input map": "map", "Random terrain": "random", "Map + random": "mixed"}


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_fingerprint():
    """Pin the procedural implementation, reference distribution, and noise library."""
    from planet_diffusion import procedural
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(procedural.__file__), procedural.DATA / "synthetic_map_stats.json"):
        digest.update(path.read_bytes())
    digest.update(version("pyfastnoiselite").encode("ascii"))
    return digest.hexdigest()


def source_settings(source, dem=None, map_weight=.5):
    if source not in ("map", "random", "mixed"):
        raise ValueError("Input source must be map, random, or mixed")
    if not np.isfinite(map_weight) or not 0 <= map_weight <= 1:
        raise ValueError("Map weight must be between 0 and 1")
    if source != "random" and not dem:
        raise ValueError("Map and mixed inputs require a PNG or elevation TIFF")
    settings = dict(input_source=source, map_weight=1.0 if source == "map" else 0.0 if source == "random" else map_weight,
                    dem_sha256=file_sha(dem) if source != "random" else None)
    if source != "map":
        settings.update(random_source_version=RANDOM_VERSION, random_source_sha256=random_fingerprint())
    return settings


def settings_differences(old, settings):
    """Old map-only v2 runs retain exactly the same condition and remain reusable."""
    return {key: (old.get(key, MAP_LEGACY_DEFAULTS.get(key)), value)
            for key, value in settings.items()
            if old.get(key, MAP_LEGACY_DEFAULTS.get(key)) != value}


class RandomElevation:
    """Continuous seeded 3-D noise mapped to the project's elevation distribution."""
    def __init__(self, seed):
        from planet_diffusion import procedural
        digest = hashlib.blake2s(str(int(seed)).encode("ascii"), digest_size=4).digest()
        noise_seed = int.from_bytes(digest, "little") & 0x7fffffff
        self.noise = procedural.perlin(noise_seed, NOISE_FREQUENCY, 4)
        self.noise_quantiles = procedural.noise_quantiles(0, NOISE_FREQUENCY)
        self.elevation_quantiles = procedural.elevation_quantiles(procedural.DEFAULT_DROP_WATER)

    def sample(self, lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat), np.asarray(lon))
        latitude = np.radians(lat)
        longitude = np.radians((lon + 180) % 360 - 180)
        horizontal = np.where(np.abs(lat) == 90, 0.0, np.cos(latitude))
        points = np.stack((horizontal * np.cos(longitude), horizontal * np.sin(longitude), np.sin(latitude)), -1)
        coords = np.ascontiguousarray((points * NOISE_RADIUS).reshape(-1, 3).T, dtype=np.float32)
        noise = self.noise.gen_from_coords(coords)
        return np.interp(noise, self.noise_quantiles, self.elevation_quantiles).reshape(lat.shape).astype(np.float32)


class MixedElevation:
    def __init__(self, terrain, random, map_weight, alpha=None):
        self.terrain, self.random, self.weight, self.alpha = terrain, random, map_weight, alpha
        if alpha is not None:
            # Interpolate premultiplied elevations so hidden RGB under fully
            # transparent pixels cannot bleed into a painted edge.
            self.terrain = GlobalRaster(terrain.values * alpha.values)

    def sample(self, lat, lon):
        weight = self.weight if self.alpha is None else self.weight * self.alpha.sample(lat, lon)
        if self.alpha is None and self.weight == 1:
            return self.terrain.sample(lat, lon)
        random = self.random.sample(lat, lon)
        if self.weight == 0:
            return random
        terrain = self.terrain.sample(lat, lon)
        return ((1 - weight) * random + self.weight * terrain).astype(np.float32)


def make_source(source, seed, dem=None, black_metres=-2000., white_metres=4000., ocean_metres=None, map_weight=.5):
    if source == "random":
        return RandomElevation(seed)
    terrain = GlobalRaster.from_file(dem, black_metres, white_metres, ocean_metres)
    if source == "map":
        return terrain
    if source != "mixed":
        raise ValueError("Input source must be map, random, or mixed")
    alpha = None
    if str(dem).lower().endswith(".png"):
        from PIL import Image
        with Image.open(dem) as image:
            if "A" in image.getbands() or "transparency" in image.info:
                alpha = GlobalRaster(np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32) / 255)
    return MixedElevation(terrain, RandomElevation(seed), map_weight, alpha)
