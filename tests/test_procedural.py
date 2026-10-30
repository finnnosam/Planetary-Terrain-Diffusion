"""Source parity and spherical invariants for procedural conditioning."""
from pathlib import Path
import sys
import numpy as np
import pytest
from planet_diffusion import cube, procedural


def source_factory(monkeypatch):
    source = Path(__file__).resolve().parents[1]/"upstream"
    if not source.is_dir():
        pytest.skip("Upstream reference checkout is unavailable")
    sys.path.insert(0, str(source))
    from terrain_diffusion.inference import synthetic_map
    s = procedural.stats()
    cache = {key: s[key] for key in ("a_temp_std", "b_temp_std", "temp_std_p1", "temp_std_p99")}
    for i in range(5):
        cache[f"noise_quantiles_{i}"] = np.array(s["noise_quantile_tables"][i])
        cache[f"base_image_quantiles_{i}"] = np.array(s["data_quantile_tables"][i])
    monkeypatch.setattr(synthetic_map, "_load_stats_cache", lambda: cache)
    return synthetic_map.make_synthetic_map_factory(seed=42)


def test_finalization_matches_source_including_extremes(monkeypatch):
    factory = source_factory(monkeypatch)
    raw = np.array([
        [-6000, 0, 500, 2500, 8000, 0],
        [-30, -10, 20, 40, 10, 45],
        [-500, -100, 0, 100, 500, 1500],
        [0, 100, 500, 2000, 5000, 10000],
        [20, 50, 90, 150, 200, 300]], np.float32)[:, None, :]
    np.testing.assert_array_equal(procedural.finalize(raw), factory.finalize(raw))
    plane = factory.sample_raw(0, 0, 17, 17)
    np.testing.assert_array_equal(procedural.encode(procedural.finalize(plane)), factory(0, 0, 17, 17).numpy())


def test_shared_nodes_seed_zero_and_channel_independence():
    n = 32
    a = cube.conditioning(0, n)
    assert a.shape == (5, 6, n+1, n+1) and a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, cube.conditioning(0, n))
    assert not np.array_equal(a, cube.conditioning(1, n))
    members, ids, owners, _ = cube.edge_groups(n)
    flat = a.reshape(5, -1)
    np.testing.assert_array_equal(flat[:, members], flat[:, owners[ids]])
    raw = cube.conditioning(42, n, raw=True)
    changed = cube.conditioning(42, n, frequency_mult=(2,3,3,3,3), raw=True)
    np.testing.assert_array_equal(raw[1:], changed[1:])
    assert not np.array_equal(raw[0], changed[0])
    wetter = cube.conditioning(42, n, drop_water_pct=0., raw=True)
    np.testing.assert_array_equal(raw[1:], wetter[1:])
    assert np.mean(raw[0]) > np.mean(wetter[0])


def test_continuity_at_seam_and_poles():
    p = np.array([[-1,1e-7,0],[-1,-1e-7,0],[1e-7,0,1],[-1e-7,0,1]], float)
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    values = procedural.sample_raw(42,p,20)
    np.testing.assert_allclose(values[:,0], values[:,1], atol=.02, rtol=1e-5)
    np.testing.assert_allclose(values[:,2], values[:,3], atol=.02, rtol=1e-5)


def test_reference_water_mask_exact():
    with np.load(procedural.DATA/"elevation_reference.npz") as data:
        elev = data["elevation"].astype(np.float32)
    elev[elev < -30000] = np.nan
    for drop in (0., .5, .8, 1.):
        mask = (np.random.default_rng(0).random(elev.shape) > drop) | (elev >= 0)
        np.testing.assert_array_equal(procedural.elevation_quantiles(drop), procedural.build_quantiles(elev[mask]))


def test_three_dimensional_distribution_calibration():
    points = np.random.default_rng(200).uniform(0,32768,(100000,3))
    raw = procedural.sample_raw(999,points,1)
    targets = np.asarray(procedural.stats()["data_quantile_tables"])
    # Independent noise seed/population should retain the reference quantiles.
    for channel in range(5):
        actual = np.quantile(raw[channel], [.1,.5,.9])
        expected = np.interp([.1,.5,.9],np.linspace(1e-4,1-1e-4,64), targets[channel])
        np.testing.assert_allclose(actual,expected,atol=.035*np.ptp(targets[channel]),rtol=0)


@pytest.mark.parametrize("frequency,drop", [([1]*4,.5),([1,1,0,1,1],.5),([1]*5,1.1),([1]*5,np.nan)])
def test_invalid_configuration(frequency,drop):
    with pytest.raises(ValueError):
        procedural.validate(frequency,drop)


def test_generation_uses_model_settings_and_source_normalization(tmp_path):
    from planet_diffusion.backends import DiagnosticBackend
    from planet_diffusion.cube_generate import generate_cube
    backend = DiagnosticBackend()
    backend.frequency_mult = (1.,)*5
    backend.drop_water_pct = .8
    class Captured(Exception):
        pass
    seen = {}
    def audit(name, values):
        seen[name] = values.copy()
        if name == "conditioning":
            raise Captured
    with pytest.raises(Captured):
        generate_cube(backend,0,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                      checkpoint_dir=tmp_path,audit=audit)
    raw = cube.conditioning(0,2,frequency_mult=(1.,)*5,drop_water_pct=.8)
    np.testing.assert_array_equal(seen["raw-conditioning"],raw)
    channels = [0,2,3,4,5]
    normalized = (raw-backend.means[channels,None,None,None])/backend.stds[channels,None,None,None]
    angles = np.arctan(backend.cond_snr)[:,None,None,None]
    expected = np.cos(angles)*normalized+np.sin(angles)*cube.noise(0,0,5,2)
    np.testing.assert_array_equal(seen["conditioning"],expected)
    backend.frequency_mult = (2.,)*5
    with pytest.raises(ValueError,match="different parameters"):
        generate_cube(backend,0,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                      checkpoint_dir=tmp_path)
