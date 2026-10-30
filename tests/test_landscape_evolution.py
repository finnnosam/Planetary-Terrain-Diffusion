"""Landscape evolution and tectonic templates. CPU only; no data or checkpoints."""
import numpy as np
import pytest

from planet_training import landscape_evolution as le
from planet_training import tectonic_templates as tt


def test_landlocked_below_sea_level_is_not_ocean():
    z = np.full((20, 20), 100.); z[:, :3] = -2000.   # open sea on the west edge
    z[10:13, 10:13] = -50.                            # landlocked pit below sea level
    edge = np.zeros_like(z, bool); edge[:, 0] = True
    ocean = le.connected_ocean(z.ravel(), (edge & (z <= 0)).ravel(), 20, 20).reshape(20, 20)
    assert ocean[:, :3].all() and not ocean[10:13, 10:13].any()


def test_depressions_fill_to_spill_and_every_cell_reaches_an_outlet():
    n = 30
    y, x = np.mgrid[:n, :n]
    z = 50.+5.*x                                      # slope down to the west
    z[:, 0] = -100.
    z[12:18, 12:18] = 10.                             # closed depression
    ocean = (z <= 0).ravel()
    zf = le.priority_flood(z.ravel(), ocean, n, n, 1e-3).reshape(n, n)
    assert (zf >= z-1e-9).all()
    assert zf[12:18, 12:18].min() >= z[12:18, 11].min()-1e-6   # filled up to its rim/spill
    rec, dist, stack = le.route(zf.ravel(), ocean, n, n, 1000.)
    assert len(stack) == n*n and len(np.unique(stack)) == n*n
    outlet = le.basins(rec, stack)
    assert ocean[outlet].all()


def test_flexure_has_central_depression_and_peripheral_bulge():
    f = le.Flexure(128, 128, 10e3, 30e3)
    load = np.zeros((128, 128)); load[60:68, 60:68] = 3000.
    w = f.deflection(load)
    assert w[64, 64] > 100
    profile = w[64, 64:]
    assert profile.min() < 0, 'expected a flexural bulge (negative deflection) beyond the moat'
    assert 40 < f.alpha_m/1e3 < 100


def test_erosion_without_uplift_lowers_land_and_conserves_sediment_budget():
    n = 64; dx = 10e3
    y, x = np.mgrid[:n, :n]
    z0 = 1500.*np.exp(-((x-32)**2+(y-32)**2)/200.)+100.
    z0[:, :4] = -1000.
    model = le.Landscape(z0, dx, np.zeros_like(z0))
    for _ in range(20):
        model.step()
    z = model.z.reshape(n, n)
    assert np.isfinite(z).all()
    assert z[~(z0 <= 0)].max() < z0.max()
    t = model.totals
    assert t['deposited_m3'] <= t['eroded_m3']*1.01+1e3


@pytest.mark.parametrize('scenario', tt.SCENARIOS)
def test_templates_are_finite_seeded_and_have_land_and_sea(scenario):
    z0, uplift, erod, meta = tt.build(scenario, 128, 23040., 5)
    u = uplift(meta['young_start_yr']+1e6)
    assert np.isfinite(z0).all() and np.isfinite(u).all() and (erod > 0).all()
    assert (z0 > 0).mean() > .4 and (z0 <= 0).mean() > .05
    assert u.max() > 1e-3
    z1 = tt.build(scenario, 128, 23040., 5)[0]; z2 = tt.build(scenario, 128, 23040., 6)[0]
    assert np.array_equal(z0, z1) and not np.array_equal(z0, z2)


@pytest.mark.parametrize('scenario', tt.SCENARIOS)
def test_short_run_produces_drainage_outputs(scenario):
    z0, uplift, erod, _ = tt.build(scenario, 96, 30e3, 3)
    model = le.Landscape(z0, 30e3, uplift, None, erod).run(5e5)
    o = model.outputs()
    land = ~o['ocean']
    assert np.isfinite(o['elevation_m']).all() and land.any()
    assert (o['drainage_area_m2'] >= 30e3**2-1).all()
    assert o['river'].any()
    allowed = set(np.flatnonzero(o['ocean'].ravel())) | set(np.flatnonzero(model.edge))
    assert set(np.unique(o['outlet'][land])) <= allowed, 'every land cell drains to the sea or out of the window'


def test_meso_windows_rule_stays_in_bounds():
    from scripts.simulate_tectonic_region import meso_windows
    rng = np.random.default_rng(0)
    z = rng.normal(500, 300, (128, 128)).astype(np.float32); z[:, :20] = -2000
    w = meso_windows(z, z > 0)
    assert {'highest-relief', 'centre'} <= set(w)
    for r, c in w.values():
        assert 0 <= r <= 128-18 and 0 <= c <= 128-18


@pytest.mark.parametrize('scenario', tt.SCENARIOS)
def test_inherited_history_precedes_young_events(scenario):
    z0, uplift, erod, meta = tt.build(scenario, 128, 23040., 7)
    assert meta['inherited'] and meta['young_start_yr'] == tt.OLD_PHASE_YR and meta['years'] == tt.OLD_PHASE_YR+tt.YOUNG_PHASE_YR
    old = np.abs(uplift(5e6)).max(); young = np.abs(uplift(meta['young_start_yr']+2e6)).max()
    assert old > 0 and young > old
    flat = tt.build(scenario, 128, 23040., 7, inherited=False)[3]
    assert flat['young_start_yr'] == 0 and flat['years'] == tt.YOUNG_PHASE_YR
    assert erod.min() > 0 and erod.max()/erod.min() > 3, 'stable crust should be harder than young belts'


def test_belts_are_segmented():
    rng = np.random.default_rng(0)
    x, y = tt.grid(128, 23040.)
    c = tt.Curve((0., 0.), 0., 2.0e6, 0.)
    f = tt.belt_field(x, y, c, rng, 5e-3, 30e3, 45e3, 1)
    rows = np.arange(128)[:, None]
    centroid = (f*rows).sum(0)/np.maximum(f.sum(0), 1e-12)
    assert np.ptp(centroid[32:96]) > .5, 'segments should offset the belt laterally along strike'


def test_marine_fill_builds_graded_shelf():
    from planet_training.landscape_evolution import coast_distance, marine_sediment
    n = 40; z = np.full(n*n, -500.); ocean = np.ones(n*n, bool)
    z[:n*5] = 100.; ocean[:n*5] = False                     # land strip along one edge
    d = coast_distance(ocean, n, n, 10e3)
    sea_in = np.zeros(n*n); sea_in[5*n+20] = 1e14            # one large river mouth
    dep = marine_sediment(z, ocean, sea_in, n, n, 1e8, 10., 1.2e-3, d, 150e3)
    filled = (z+dep).reshape(n, n)
    assert filled[5, 20] <= 2.+1e-6 and filled[6:12, 20].max() < 0
    assert filled[8, 20] < filled[6, 20], 'shelf should deepen offshore'
    assert (filled[5+16:, :] == -500.).all(), 'no shelf fill beyond the shelf break (150 km)'
