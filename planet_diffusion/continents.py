"""Planet-scale guide from a chain of coarse-model passes, small planet to real scale.

Upstream's procedural guide has no structure larger than about 1,400 km, so a
correctly scaled Earth-sized globe looks like uniform noise. The rough pass
runs the coarse model on a small guide grid, where its learned landforms span
continents. Each following pass runs the coarse model on a grid about `step`
times finer, guided by the previous pass, so the model adds its own structure
at every intermediate scale. The final pass is the ordinary coarse stage.
Nothing procedural is added between passes.
"""
import copy
import math

import numpy as np

from . import cube
from .coarse import sample_coarse

VERSION = "continent-cascade-v3"
GUIDE_CHANNELS = [0, 2, 3, 4, 5]
SEED_OFFSET = 0x9E3779B97F4A7C15
# Light smoothing (in previous-pass cells) removes bilinear facets from the
# upsampled guide without erasing that pass's smallest learned features.
SMOOTHING_PREVIOUS_CELLS = .5


def gaussian_blur(a, sigma):
    """Separable Gaussian on the shared cube field; sigma in node intervals."""
    if sigma <= 0:
        return cube.identify(a)
    radius = max(1, int(math.ceil(3*sigma)))
    offsets = np.arange(-radius, radius+1)
    weights = np.exp(-.5*(offsets/sigma)**2)
    weights /= weights.sum()
    return cube.filter_axis(cube.filter_axis(a, weights, 1), weights, 0)


def to_metres(guide):
    out = np.array(guide, dtype=np.float32, copy=True)
    out[0] = np.sign(out[0])*out[0]**2
    return out


def to_guide(metres):
    out = np.array(metres, dtype=np.float32, copy=True)
    out[0] = np.sign(out[0])*np.sqrt(np.abs(out[0]))
    return out


def upsample(a, n):
    """Smooth guide on n intervals from a coarser pass (guide units)."""
    previous = a.shape[-1]-1
    return gaussian_blur(cube.resize(a, n), SMOOTHING_PREVIOUS_CELLS*n/previous)


def cascade_heights(start, final, step):
    """Guide heights of the passes before the final coarse stage, geometric, multiples of 4."""
    if final <= start:
        return [start]
    count = max(1, math.ceil(math.log(final/start)/math.log(step)-1e-9))
    heights = [start]
    for i in range(1, count):
        height = 4*round(start*(final/start)**(i/count)/4)
        if heights[-1] < height < final:
            heights.append(height)
    return heights


class ContinentGuide:
    """Guide source: chained coarse-model passes from a small planet to the coarse stage.

    relief scales final guide land elevation (ocean depth is kept). step is
    the largest refinement ratio between passes.
    """
    def __init__(self, backend, guide_height=48, relief=1., step=3., steps=20,
                 refinement=None, progress=print):
        if isinstance(guide_height, bool) or int(guide_height) != guide_height or guide_height < 8 or guide_height % 4:
            raise ValueError("Continent guide height must be a multiple of 4, at least 8")
        if not math.isfinite(relief) or not 0 <= relief <= 4:
            raise ValueError("Continent relief must be between 0 and 4")
        if not math.isfinite(step) or not 1.5 <= step <= 16:
            raise ValueError("Continent step must be between 1.5 and 16")
        if refinement is not None and (not math.isfinite(refinement) or not .01 <= refinement <= 4):
            raise ValueError("Continent refinement must be between 0.01 and 4")
        self.backend = backend
        self.guide_height = int(guide_height)
        self.relief, self.step, self.steps = float(relief), float(step), int(steps)
        self.progress = progress
        self.cond_snr = np.array(backend.cond_snr, dtype=np.float32, copy=True)
        if refinement is not None:
            self.cond_snr[0] = refinement
        self.levels = []
        self.guide = None
        self.metadata = {"version": VERSION, "guide_height": self.guide_height,
                         "relief": self.relief, "step": self.step, "steps": self.steps,
                         "pass_seeds": "seed + (i+1) x 0x9E3779B97F4A7C15 mod 2^64",
                         "rough_conditioning_snr": np.asarray(backend.cond_snr, np.float32).tolist(),
                         "cascade_conditioning_snr": self.cond_snr.tolist(),
                         "smoothing_previous_cells": SMOOTHING_PREVIOUS_CELLS}

    @property
    def rough(self):
        return self.levels[0][1] if self.levels else None

    def _pass(self, index, backend, raw, progress_label):
        seed = (self._seed+(index+1)*SEED_OFFSET) % 2**64
        out = sample_coarse(backend, seed, raw, self.steps,
                            progress=lambda s: self.progress(f"{progress_label}: {s}"))
        return cube.identify(np.asarray(out)[GUIDE_CHANNELS])

    def conditioning(self, seed, n, **options):
        if n < self.guide_height//2:
            raise ValueError(f"Continent guide height {self.guide_height} exceeds the coarse model guide "
                             f"height {2*n}; use a smaller continent guide or coarse pooling")
        self._seed = int(seed)
        heights = cascade_heights(self.guide_height, 2*n, self.step)
        self.progress("continent passes: guide " + " -> ".join(str(h) for h in heights+[2*n]))
        self.levels = []
        refined = copy.copy(self.backend)
        refined.cond_snr = self.cond_snr.copy()
        for index, height in enumerate(heights):
            if index == 0:
                rough_seed = (self._seed+SEED_OFFSET) % 2**64
                raw = cube.conditioning(rough_seed, height//2, **options)
                out = self._pass(0, self.backend, raw, f"continent pass 1/{len(heights)} (guide {height})")
            else:
                raw = upsample(self.levels[-1][1], height//2)
                out = self._pass(index, refined, raw,
                                 f"continent pass {index+1}/{len(heights)} (guide {height})")
            self.levels.append((height, out))
        guide = to_metres(upsample(self.levels[-1][1], n))
        guide[0] = np.where(guide[0] > 0, self.relief*guide[0], guide[0])
        self.guide = cube.identify(to_guide(guide))
        return self.guide
