"""WorldPipeline's completed-tile coarse sampling on shared spherical charts."""
import numpy as np
from . import cube

TILE_SIZE = 64
TILE_STRIDE = 48
VERSION = "completed-coarse-tiles-v1"
POOL_VERSION = "cube-coarse-pooling-v1"
# Elevation and p5 reducers. "max" is upstream's more extreme option: max
# elevation and min p5. Climate channels are always averaged.
POOL_MODES = {"avg": ("avg", "avg"), "max": ("max", "min")}


def linear_weight_window(size=TILE_SIZE):
    return cube.linear_weight_window(size)


def pool_coarse(coarse, k, mode="avg"):
    """Compress a coarse field computed on a k-times finer grid (upstream coarse_pooling).

    Channels are elevation, p5 elevation, then climate, in physical units as
    returned by sample_coarse. Each output cell summarizes k x k model cells,
    so the large-scale layout keeps model-scale distances while later stages
    see one cell per k*k model cells.
    """
    if mode not in POOL_MODES:
        raise ValueError(f"Coarse pool mode must be one of {sorted(POOL_MODES)}")
    if k == 1:
        return cube.identify(coarse)
    elevation, p5 = POOL_MODES[mode]
    pooled = cube.pool(coarse, k).astype(np.float32)
    if elevation == "max":
        pooled[0] = cube.pool_extreme(coarse[:1], k, np.max)[0]
    if p5 == "min":
        pooled[1] = cube.pool_extreme(coarse[1:2], k, np.min)[0]
    return cube.identify(pooled)


def sample_coarse(backend, seed, raw_guide, steps=20, progress=print, record=None):
    """Denoise each tile independently, then combine physical channel values.

    A fresh scheduler history per tile is established by start_coarse. Initial
    and conditioning noise are shared per physical globe node and read nearest
    across face boundaries, preserving the existing spherical RNG contract.
    """
    def emit(name, value):
        return record(name,value) if record is not None else value
    n = raw_guide.shape[-1]-1
    channels = [0,2,3,4,5]
    means, stds = backend.means, backend.stds
    emit("raw-conditioning",raw_guide)
    guide = (raw_guide-means[channels,None,None,None])/stds[channels,None,None,None]
    angles = np.arctan(backend.cond_snr)[:,None,None,None]
    guide = emit("conditioning",(np.cos(angles)*guide+np.sin(angles)*cube.noise(seed,0,5,n)).astype(np.float32))
    initial = cube.noise(seed,1,6,n)
    emit("coarse-noise",initial*backend.start_coarse(steps))
    patches = 6*len(range(-TILE_STRIDE,n,TILE_STRIDE))**2
    completed = 0
    def predict(noise, face, y, x):
        nonlocal completed
        cond = cube.read(guide,face,np.arange(y,y+TILE_SIZE)[:,None],
                         np.arange(x,x+TILE_SIZE)[None,:],nearest=True)
        result = backend.sample_coarse_tile(noise,cond,steps)*stds[:,None,None]+means[:,None,None]
        result[1] = result[0]-result[1]
        completed += 1
        progress(f"cube coarse tile {completed}/{patches} ({steps} steps)")
        return result
    return cube.consensus(initial,predict,size=TILE_SIZE,stride=TILE_STRIDE,
                          weight_window=linear_weight_window(),shared_weights=True)
