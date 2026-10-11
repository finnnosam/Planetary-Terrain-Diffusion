"""Versioned elevation-only sources; old random caches keep their original guide."""
import hashlib
from importlib.metadata import version
from pathlib import Path

from . import conditioning_v1 as legacy
from .conditioning_v1 import (SOURCE_LABELS, MAP_LEGACY_DEFAULTS, MixedElevation,
                              file_sha, settings_differences)
from .random_terrain import RandomElevation

LEGACY_RANDOM_VERSION = legacy.RANDOM_VERSION
RANDOM_VERSION = "meso-random-elevation-v2-multiscale"
RANDOM_VERSIONS = (LEGACY_RANDOM_VERSION, RANDOM_VERSION)


def random_fingerprint(random_version=RANDOM_VERSION):
    if random_version == LEGACY_RANDOM_VERSION:
        return legacy.random_fingerprint()
    if random_version != RANDOM_VERSION:
        raise ValueError("Unsupported random terrain version")
    from planet_diffusion import procedural
    from . import random_terrain
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(legacy.__file__), Path(random_terrain.__file__),
                 Path(procedural.__file__), procedural.DATA / "synthetic_map_stats.json"):
        digest.update(path.read_bytes())
    digest.update(version("pyfastnoiselite").encode("ascii"))
    return digest.hexdigest()


def source_settings(source, dem=None, map_weight=.5, random_version=RANDOM_VERSION):
    if random_version not in RANDOM_VERSIONS:
        raise ValueError("Unsupported random terrain version")
    settings = legacy.source_settings(source, dem, map_weight)
    if source != "map":
        settings.update(random_source_version=random_version,
                        random_source_sha256=random_fingerprint(random_version))
    return settings


def make_source(source, seed, dem=None, black_metres=-2000., white_metres=4000., ocean_metres=None,
                map_weight=.5, random_version=RANDOM_VERSION):
    if random_version not in RANDOM_VERSIONS:
        raise ValueError("Unsupported random terrain version")
    if source == "map" or random_version == LEGACY_RANDOM_VERSION:
        return legacy.make_source(source, seed, dem, black_metres, white_metres, ocean_metres, map_weight)
    if source == "random":
        return RandomElevation(seed)
    # Keep the established map/alpha blend exactly; replace only its random guide.
    result = legacy.make_source(source, seed, dem, black_metres, white_metres, ocean_metres, map_weight)
    result.random = RandomElevation(seed)
    return result
