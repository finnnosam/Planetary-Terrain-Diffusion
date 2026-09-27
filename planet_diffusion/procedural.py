"""Source synthetic_map conditioning, evaluated in 3-D for a closed globe.

Target distributions and climate equations follow Terrain Diffusion. Noise
quantiles are calibrated in 3-D, not reused from the source's 2-D Perlin field.
See data/README.txt for reference data provenance and regeneration.
"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType

DEFAULT_FREQUENCY = (1.5, 3., 3., 3., 3.)
DEFAULT_DROP_WATER = .5
DATA = Path(__file__).with_name("data")


def validate(frequency_mult, drop_water_pct):
    frequency = np.asarray(frequency_mult, dtype=np.float64)
    if frequency.shape != (5,) or not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("frequency_mult requires five finite positive values")
    if not np.isfinite(drop_water_pct) or not 0 <= drop_water_pct <= 1:
        raise ValueError("drop_water_pct must be between 0 and 1")
    return tuple(frequency), float(drop_water_pct)


def build_quantiles(values, n_quantiles=64, eps=1e-4):
    """Same quantile grid and tie handling as upstream perlin_transform.py."""
    values = np.asarray(values).ravel()
    q = np.quantile(values[~np.isnan(values)], np.linspace(eps, 1-eps, n_quantiles))
    diff = np.diff(q)
    step = np.min(diff[diff > 0]) if np.any(diff > 0) else 1e-10
    for i in range(1, len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1]+step*.1
    return q


def perlin(seed, frequency, octaves):
    noise = FastNoiseLite(seed=int(seed))
    noise.noise_type = NoiseType.NoiseType_Perlin
    noise.frequency = .05*frequency
    noise.fractal_type = FractalType.FractalType_FBm
    noise.fractal_octaves = octaves
    noise.fractal_lacunarity = 2.
    noise.fractal_gain = .5
    return noise


@lru_cache(maxsize=1)
def stats():
    return json.loads((DATA/"synthetic_map_stats.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def elevation_quantiles(drop_water_pct):
    if drop_water_pct == DEFAULT_DROP_WATER:
        return np.asarray(stats()["data_quantile_tables"][0])
    # Retain the reference pixels so arbitrary water-drop settings reproduce
    # upstream's deterministic histogram mask exactly, without network access.
    with np.load(DATA/"elevation_reference.npz", allow_pickle=False) as archive:
        elev = archive["elevation"].astype(np.float32)
    elev[elev < -30000] = np.nan
    mask = (np.random.default_rng(0).random(elev.shape) > drop_water_pct) | (elev >= 0)
    return build_quantiles(elev[mask])


@lru_cache(maxsize=32)
def noise_quantiles(channel, frequency):
    # Fixed spatial population, independent of world seed, face, resolution or
    # requested region. Never histogram-match individual faces or small worlds.
    coords = np.random.default_rng(0).uniform(0, 32768, (3, 1024*1024)).astype(np.float32)
    noise = perlin(channel+1, frequency, 2 if channel == 1 else 4)
    return build_quantiles(noise.gen_from_coords(coords))


def sample_raw(seed, points, radius, frequency_mult=DEFAULT_FREQUENCY,
               drop_water_pct=DEFAULT_DROP_WATER):
    """Five physical channels at unit directions; temperature is sea-level here."""
    frequencies, drop = validate(frequency_mult, drop_water_pct)
    points = np.asarray(points)
    coords = np.ascontiguousarray((points*radius).reshape(-1, 3).T, dtype=np.float32)
    targets = stats()["data_quantile_tables"]
    result = []
    for i, frequency in enumerate(frequencies):
        # Source seed offsets/mask, with seed zero kept deterministic.
        noise = perlin((int(seed)+i+1) & 0x7fffffff, frequency, 2 if i == 1 else 4)
        target = elevation_quantiles(drop) if i == 0 else targets[i]
        result.append(np.interp(noise.gen_from_coords(coords), noise_quantiles(i, frequency), target))
    return np.asarray(result, dtype=np.float32).reshape((5,)+points.shape[:-1])


def finalize(raw):
    """Source climate transforms, before the model's elevation encoding."""
    elev, temp, temp_std, precip, precip_std = np.asarray(raw, dtype=np.float32)
    s = stats()
    a, b = s["a_temp_std"], s["b_temp_std"]
    p1, p99 = s["temp_std_p1"], s["temp_std_p99"]
    lapse = np.clip(-6.5+.0015*precip, -9.8, -4.)/1000
    temp = np.clip(temp+lapse*np.maximum(0, elev), -10, 40)
    temp = np.where(temp > 20, temp, (temp-20)*1.25+20)
    t = (temp_std-p1)/(p99-p1)
    baseline = np.maximum(p1, -(a*temp+b))
    temp_std = np.maximum(t*(p99-baseline)+baseline+(a*temp+b), 20)
    precip_std = precip_std*np.maximum(0, (185-.04111*precip)/185)
    return np.stack([elev, temp, temp_std, precip, precip_std])


def encode(raw):
    result = np.array(raw, dtype=np.float32, copy=True)
    result[0] = np.sign(result[0])*np.sqrt(np.abs(result[0]))
    return result
