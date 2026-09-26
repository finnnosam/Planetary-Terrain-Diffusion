"""Node-centred sphere: H latitude intervals, 2H longitudes, unique poles.

Arrays are C x (H+1) x 2H. Exported rasters are sampled at pixel centres.
All spatial reads, including model contexts, use this topology.
"""
import numpy as np


def indices(h, y, x):
    y, x = np.broadcast_arrays(np.asarray(y), np.asarray(x))
    folded = y % (2 * h)
    south = folded > h
    return np.where(south, 2 * h - folded, folded), (x + south * h) % (2 * h)


def gather(a, y, x):
    iy, ix = indices(a.shape[-2] - 1, y, x)
    return a[..., iy.astype(np.int64), ix.astype(np.int64)]


def patch(a, y, x, size):
    return gather(a, np.arange(y, y + size)[:, None], np.arange(x, x + size)[None, :])


def sample(a, y, x):
    """Continuous spherical interpolation; arbitrary broadcastable node coordinates."""
    y, x = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    iy, ix = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    fy, fx = y - iy, x - ix
    return ((1-fy) * ((1-fx)*gather(a, iy, ix) + fx*gather(a, iy, ix+1))
            + fy * ((1-fx)*gather(a, iy+1, ix) + fx*gather(a, iy+1, ix+1)))


def resize(a, h):
    old_h = a.shape[-2] - 1
    return sample(a, np.linspace(0, old_h, h+1)[:, None],
                  np.arange(2*h)[None, :] * old_h/h).astype(np.float32)


def constrain(a):
    """Identify poles and constrain only the immediately adjacent cap rings.

    Never low-pass entire latitude rows: that changes both the learned latent
    representation and the diffusion noise distribution throughout the planet.
    This projection is for denoised states, not Gaussian innovations.
    """
    a = np.array(a, dtype=np.float32, copy=True)
    h, w = a.shape[-2]-1, a.shape[-1]
    if h < 2 or w != 2*h:
        raise ValueError("Sphere state requires C x (H+1) x 2H, H >= 2")
    for row in set((1, h-1)):
        modes = np.fft.rfft(a[..., row, :], axis=-1)
        modes[..., 2:] = 0
        a[..., row, :] = np.fft.irfft(modes, n=w, axis=-1)
    a[..., 0, :] = a[..., 0, :].mean(axis=-1, keepdims=True, dtype=np.float64)
    a[..., -1, :] = a[..., -1, :].mean(axis=-1, keepdims=True, dtype=np.float64)
    return a


def noise(seed, stream, channels, h):
    """Unit Gaussian noise per node, with one shared draw per physical pole."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    a = rng.standard_normal((channels, h+1, 2*h), dtype=np.float32)
    a[:, 0, :] = a[:, 0, :1]
    a[:, -1, :] = a[:, -1, :1]
    return a


def consensus(a, predict, size=64, stride=32, progress=None):
    """Combine denoiser predictions into ONE state before the solver advances.

    Contexts cross both boundaries. Scatter folds predictions back onto the
    sphere, including half-turn orientation at poles. No finished tiles blend.
    """
    h, w = a.shape[-2]-1, a.shape[-1]
    weight1 = np.maximum(1e-3, 1-np.abs(np.linspace(-1, 1, size)))
    weight = weight1[:, None]*weight1[None, :]
    total = None
    norm = np.zeros((h+1, w), dtype=np.float64)
    count = 0
    patches = len(range(-stride, h+1, stride))*len(range(0, w, stride))
    for y in range(-stride, h+1, stride):
        for x in range(0, w, stride):
            pred = np.asarray(predict(patch(a, y, x, size), y, x), dtype=np.float32)
            if total is None:
                total = np.zeros((pred.shape[0], h+1, w), dtype=np.float64)
            iy, ix = indices(h, np.arange(y, y+size)[:, None], np.arange(x, x+size)[None, :])
            np.add.at(norm, (iy, ix), weight)
            for c in range(pred.shape[0]):
                np.add.at(total[c], (iy, ix), pred[c]*weight)
            count += 1
            if progress is not None and (count % 16 == 0 or count == patches):
                progress(count, patches)
    if np.any(norm == 0):
        raise RuntimeError("Incomplete spherical prediction coverage")
    return constrain(total/norm)


def conditioning(seed, h):
    """Smooth 3-D Fourier sketch evaluated on the sphere, in upstream units."""
    theta = np.pi*np.arange(h+1)[:, None]/h
    lon = 2*np.pi*np.arange(2*h)[None, :]/(2*h)
    xyz = np.stack(np.broadcast_arrays(np.sin(theta)*np.cos(lon),
                                      np.sin(theta)*np.sin(lon), np.cos(theta)), -1)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 910]))
    directions = rng.normal(size=(12, 3))
    phases = rng.uniform(0, 2*np.pi, 12)
    field = np.sin(xyz @ (directions*2).T + phases).sum(-1)/np.sqrt(6)
    elev = 2200*field - 900
    temperature = np.broadcast_to(28 - 48*np.cos(theta)**2, elev.shape)
    return constrain(np.stack([np.sign(elev)*np.sqrt(np.abs(elev)), temperature,
                               np.broadcast_to(350+200*np.cos(theta)**2, elev.shape),
                               1200+400*np.tanh(field), 55+10*np.tanh(field)]))
