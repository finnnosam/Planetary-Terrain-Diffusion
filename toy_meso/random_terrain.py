"""Multiscale random elevations with structure down to the meso input spacing."""
from functools import lru_cache

import numpy as np

from .conditioning_v1 import RandomElevation as SmoothElevation, NOISE_FREQUENCY

OCTAVES = 9
GAIN = .7


@lru_cache(maxsize=1)
def terrain_noise_quantiles():
    from planet_diffusion import procedural
    noise = procedural.perlin(1, NOISE_FREQUENCY, OCTAVES)
    noise.fractal_gain = GAIN
    coords = np.random.default_rng(0).uniform(0, 32768, (3, 1024 * 1024)).astype(np.float32)
    return procedural.build_quantiles(noise.gen_from_coords(coords))


class RandomElevation(SmoothElevation):
    """Retain continent-scale noise while supplying finer relief to the model."""
    def __init__(self, seed):
        super().__init__(seed)
        self.noise.fractal_octaves = OCTAVES
        self.noise.fractal_gain = GAIN
        self.noise_quantiles = terrain_noise_quantiles()
