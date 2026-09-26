"""Upstream Laplacian reconstruction adapted to spherical node coordinates.

This reconstructs the decoder's learned low-frequency channel, not a seam blend
or an elevation-image smoothing pass. The high-frequency residual is retained.
"""
import numpy as np
from .topology import gather, resize, constrain


def downsample(a, scale):
    """Antialiased bilinear reduction on the spherical node grid."""
    h, w = a.shape[-2]-1, a.shape[-1]
    if h % scale:
        raise ValueError("Reduction scale must divide the latitude intervals")
    ys = np.arange(h//scale+1)[:, None]*scale
    xs = np.arange(w)[None, :]
    offsets = np.arange(1-scale, scale)
    weights = (1-np.abs(offsets)/scale)/scale
    vertical = sum(weight*gather(a, ys+dy, xs) for dy, weight in zip(offsets, weights))
    reduced = sum(weight*np.roll(vertical, -dx, axis=-1)[..., ::scale]
                  for dx, weight in zip(offsets, weights))
    return constrain(reduced)


def gaussian(a, sigma=5., radius=5):
    """The upstream 11-tap sigma=5 kernel, with spherical rather than planar halos."""
    offsets = np.arange(-radius, radius+1)
    weights = np.exp(-.5*(offsets/sigma)**2)
    weights /= weights.sum()
    horizontal = sum(weight*np.roll(a, -dx, axis=-1) for dx, weight in zip(offsets, weights))
    ys = np.arange(a.shape[-2])[:, None]
    xs = np.arange(a.shape[-1])[None, :]
    return constrain(sum(weight*gather(horizontal, ys+dy, xs) for dy, weight in zip(offsets, weights)))


def reconstruct(residual, lowfreq):
    """Return signed-square-root metres before the pointwise metre conversion.

    As in upstream laplacian_denoise: provisional decode -> antialiased reduction
    -> blur the low-frequency field -> add the unchanged learned residual.
    """
    h = residual.shape[-2]-1
    scale = h//(lowfreq.shape[-2]-1)
    provisional = residual + resize(lowfreq, h)
    lowfreq = gaussian(downsample(provisional, scale))
    return constrain(residual + resize(lowfreq, h))
