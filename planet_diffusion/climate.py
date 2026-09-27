"""Upstream climate reconstruction adapted to shared-node spherical charts."""
import numpy as np
from . import cube

VERSION = "cube-climate-v1"
BANDS = (
    ("Temperature adjusted for generated elevation", "degC"),
    ("Temperature standard deviation", "degC"),
    ("Annual precipitation", "mm"),
    ("Precipitation coefficient of variation", "%"),
    ("Temperature lapse rate", "degC/m"),
)


def local_baseline(temperature, elevation, win=15):
    """Land-weighted regression matching upstream's climate call parameters.

    Valid windows only. Float64 accumulation avoids cancellation in flat areas.
    Elevation is metres above sea level, with underwater values set to zero.
    """
    t, e = np.asarray(temperature, np.float64), np.asarray(elevation, np.float64)
    if t.shape != e.shape or t.ndim != 2 or win < 3 or win % 2 != 1 or min(t.shape) < win:
        raise ValueError("Climate regression requires matching 2D arrays and an odd window >= 3")
    land = (e > 0).astype(np.float64)
    def average(a):
        integral = np.pad(a, ((1,0),(1,0))).cumsum(0).cumsum(1)
        return (integral[win:,win:]-integral[:-win,win:]
                -integral[win:,:-win]+integral[:-win,:-win])/(win*win)
    coverage = average(land)
    mt, me, me2, met = [average(a*land)/(coverage+1e-6) for a in (t,e,e*e,e*t)]
    variance = me2-me*me
    beta = (met-me*mt)/(variance+1e-6)
    beta = np.clip(np.where((variance < 1.) | (coverage < .02), -.0065, beta), -.012, 0.)
    pad = win//2
    return (t[pad:-pad,pad:-pad]-beta*e[pad:-pad,pad:-pad]).astype(np.float32), beta.astype(np.float32)


def from_coarse(coarse):
    """Return compact cube features: baseline, T std x100, P, P CV, beta."""
    if (coarse.ndim != 4 or coarse.shape[:2] != (6,6)
            or coarse.shape[-1] != coarse.shape[-2] or coarse.shape[-1] < 3
            or not np.isfinite(coarse).all()):
        raise ValueError("Invalid coarse climate source")
    n = coarse.shape[-1]-1
    coords = np.arange(-7,n+8)
    features = []
    for face in range(6):
        halo = cube.read(coarse,face,coords[:,None],coords[None,:])
        baseline, beta = local_baseline(halo[2],np.maximum(halo[0],0.)**2)
        features.append(np.stack([baseline,*coarse[3:6,face],beta]))
    return cube.identify(np.stack(features,axis=1))


def evaluate(features, longitude, latitude, elevation):
    """Sample features at degrees and adjust temperature to the exported height."""
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    p = np.stack(np.broadcast_arrays(np.cos(lat)*np.cos(lon),
                 np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
    result = cube.sample(features,p)
    result[0] += result[4]*np.maximum(elevation,0.)
    result[1] /= 100.  # Upstream stores temperature standard deviation in degC x100.
    return result.astype(np.float32)
