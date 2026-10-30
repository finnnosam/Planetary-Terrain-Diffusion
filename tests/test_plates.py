"""Plate layer, part 1: lookup, isostasy, and invariants of a short simulation on a coarse cube."""
import numpy as np

from planet_diffusion import cube
from planet_layer import fields, plates as P
from planet_layer.sphere_graph import CubeSphere


def test_nearest_nodes_match_cube_coordinates():
    g = CubeSphere(20)
    p = np.random.default_rng(0).normal(size=(5000, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
    f, y, x = cube.coordinates(p, 20)
    entry = (f*21+np.floor(y+.5).astype(int))*21+np.floor(x+.5).astype(int)
    assert np.array_equal(fields.nearest_nodes(g, p), g.node_of_entry[entry])
    # the nearest node of a node's own position is itself
    assert np.array_equal(fields.nearest_nodes(g, g.xyz), np.arange(g.size))


def test_isostatic_elevation():
    z = P.isostatic_elevation(np.array([False, False, True, True]), np.array([7e3, 7e3, 38e3, 28e3]), np.array([0., 100., 0., 0.]))
    assert abs(z[0]+2600) < 1 and -5700 < z[1] < -5000
    assert abs(z[2]-300) < 1e-6
    assert z[3] < 0                                    # thinned continental margin under water


def test_short_run_invariants():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'erosion': False})
    cont0 = g.area[sim.world['cont']].sum()/g.area.sum()
    assert .33 < cont0 < .43
    lab = P.components(sim.world['cont'], g.nbr)
    assert lab.max() == 0                              # one supercontinent
    v0 = sim.continental_volume()
    for _ in range(30):
        n_events = len(sim.events)
        sim.step()
        ids = {pl.id for pl in sim.plates}
        assert set(np.unique(sim.owner)) <= ids        # every node owned by a live plate
        merged = any(e['kind'] == 'merge' for e in sim.events[n_events:])
        assert sim.pinholes() == 0 or merged           # a merge turns the welded plates' last ridge crust "interior"
        st = sim.state()
        oa = st['ocean_age_myr'][~np.isnan(st['ocean_age_myr'])]
        assert oa.min() >= 0
        new_ocean = sim.new_crust & ~st['continental']          # rifted gaps keep their continental age
        assert np.all(st['crust_age_myr'][new_ocean] == 0)
        # subduction erosion never strips oceanic basement (it dug -10 km pit trails before)
        assert st['crust_thickness_m'][~st['continental']].min() >= P.PARAMS['ocean_thick_m']-1
    assert abs(sim.continental_volume()/v0-1) < .01
    counts = np.bincount(sim.btype, minlength=5)
    assert counts[P.RIDGE] > 0 and counts[P.SUBDUCTION] > 0
    speeds = np.array([pl.speed for pl in sim.plates])
    assert speeds.min() > 0 and speeds.max() <= P.PARAMS['v_slab_cm_yr']+P.PARAMS['v_ridge_cm_yr']+1e-9


def test_erosion_lowers_belts_and_volume_ledger_closes():
    g = CubeSphere(48)
    # n = 48 is plumbing only (erosion is unstable this coarse; real runs need n >= 96), so the sea level is held at
    # the datum: a sea level following these unphysical elevations would change the path, not the bookkeeping
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None})
    v0 = sim.continental_volume()
    for _ in range(20):
        sim.step()
    v = sim.continental_volume()
    assert sim.erosion_totals['continental_eroded_m3'] < 0
    assert sim.erosion_totals['ocean_sediment_m3'] > 0
    assert abs((v-v0)-sum(sim.ledger.values())) < 2e-3*v0       # every volume change is accounted for
    st = sim.state()
    assert np.isfinite(st['elevation_m']).all() and (st['sediment_m'] >= 0).all()


def test_rifting_stretches_continental_crust_before_breakup():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'erosion': False})
    for _ in range(30):
        sim.step()
        rifted = sim.new_crust & sim.world['cont']
        if rifted.any():
            T = sim.world['thick'][rifted]
            assert T.min() >= P.PARAMS['breakup_thick_m']-1e-6
    assert sim.erosion_totals.get('rift_nodes', 0) > 0


def test_mantle_fields():
    from planet_layer.mantle import Mantle
    g = CubeSphere(32)
    m = Mantle(g, np.random.default_rng(0), -100., params={'hotspots_initial': 3})
    d = m.topography(-50.)
    assert 250 < np.sqrt(np.average((d-np.average(d, weights=g.area))**2, weights=g.area)) < 900
    v = m.volcanic(-50., 1.)
    live = [h for h in m.hotspots if h['start'] < -50 < h['end']]
    expected = sum(h['flux_m3_yr']*1e6 for h in live)
    assert abs((v*g.area).sum()-expected) < 1e-6*max(expected, 1)


def test_split_partitions_a_plate_without_resampling():
    g = CubeSphere(32)
    sim = P.PlateSim(g, seed=2, params={'erosion': False, 'mantle': False})
    sim.step()
    pl = max(sim.plates, key=lambda q: (sim.owner == q.id).sum())
    own = sim.owner == pl.id
    v0 = sim.continental_volume()
    half = own & (g.xyz@g.xyz[np.flatnonzero(own)[0]] > np.median(g.xyz[own]@g.xyz[np.flatnonzero(own)[0]]))
    new = sim.split(pl, half, 2., 'test')
    assert new is not None and np.allclose(new.Q, pl.Q)
    assert not (new.mask & pl.mask).any()
    assert abs(sim.continental_volume()-v0) < 1e-6*v0
    sim.step()                                      # the two plates keep working
    assert set(np.unique(sim.owner)) <= {q.id for q in sim.plates}


def test_merge_keeps_continental_volume():
    g = CubeSphere(32)
    sim = P.PlateSim(g, seed=2, params={'erosion': False, 'mantle': False})
    sim.step()
    pls = sorted(sim.plates, key=lambda q: -(sim.owner == q.id).sum())
    v0 = sim.continental_volume(); n0 = len(sim.plates)
    sim.merge(pls[0], pls[1])
    assert len(sim.plates) == n0-1 and pls[1].id not in set(np.unique(sim.owner))
    assert abs(sim.continental_volume()/v0-1) < .02          # one nearest-node move of the smaller plate
    sim.step()


def test_sea_level_holds_the_water_inventory():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'erosion': False})
    for _ in range(5):
        sim.step()
    z = sim.state()['elevation_m'].astype(np.float64)
    held = (np.maximum(-z, 0)*g.area).sum()
    assert abs(held/(P.PARAMS['water_volume_km3']*1e9)-1) < 1e-6
    assert P.sea_level_for_volume(np.array([-10., -5, 0, 5]), np.ones(4), 7.5) == -3.75


def test_wide_rift_margins_conserve_volume():
    g = CubeSphere(48)
    for width in (150., 600.):         # at n = 48 a 150 km band is narrower than a cell: falls back to the neighbours
        sim = P.PlateSim(g, seed=3, params={'erosion': False, 'rift_width_km': width})
        v0 = sim.continental_volume()
        for _ in range(10):
            sim.step()
        assert abs(sim.continental_volume()/v0-1) < .01


def test_shelf_process_keeps_the_volume_ledger():
    g = CubeSphere(48)
    # plumbing only (n = 48), sea level held at the datum as in the erosion test
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'shelf_kappa_m2_yr': 3e4})
    v0 = sim.continental_volume()
    for _ in range(15):
        sim.step()
    v = sim.continental_volume()
    assert sim.erosion_totals['shelf_off_land_m3'] > 0
    assert abs((v-v0)-sum(sim.ledger.values())) < 2e-3*v0
    st = sim.state()
    assert np.isfinite(st['elevation_m']).all() and (st['sediment_m'] >= 0).all()
    # oceanic crust thins only by erosion below its sediment, which is booked (eighteenth pass; the shelf process
    # itself never cuts below the sediment)
    oc = ~st['continental']
    deficit = float((np.clip(P.PARAMS['ocean_thick_m']-st['crust_thickness_m'][oc], 0, None)*g.area[oc]).sum())
    assert deficit <= -sim.erosion_totals.get('ocean_crust_eroded_m3', 0.)*(1+1e-6)+1e-3


def test_kept_deep_sea_sediment_builds_no_towers():
    """The first deep-sea version piled sediment in enclosed pockets into 20-250 km towers within ~10 Myr."""
    g = CubeSphere(96)
    sim = P.PlateSim(g, seed=1001)
    for _ in range(15):
        sim.step()
    assert sim.erosion_totals['deep_sea_sediment_m3'] > 0
    assert sim.world['sed'].max() < 25e3
    assert sim.elevation().max() < 9e3


def test_margin_shortening_moves_volume_into_the_belt():
    """Shortening takes crust from the foreland band and adds the same volume to the belt, per trench segment."""
    rng = np.random.default_rng(5)
    n = 4000
    d = rng.uniform(0, 900e3, n)
    near = rng.integers(0, 3, n)*7                     # three segments (labels 0, 7, 14)
    area = rng.uniform(1.5e9, 2.5e9, n)
    thick = rng.uniform(30e3, 45e3, n)
    dom = rng.random(n) > .1
    s = np.where(near == 14, 0., .01)                  # segment 14 does not shorten
    out = P.margin_shortening(near, d, dom, s, thick, area, 1e6, 150e3, 60e3, 350e3, 100e3)
    for k in (0, 7, 14):
        m = near == k
        assert abs((out*area)[m].sum()) < 1e-9*(np.abs(out)*area)[m].sum()+1e-6
    assert np.all(out[near == 14] == 0) and np.all(out[~dom] == 0)
    belt, fore = (np.abs(d-150e3) < 40e3) & dom & (near == 0), (np.abs(d-350e3) < 40e3) & dom & (near == 0)
    assert out[belt].mean() > 0 > out[fore].mean()
    # volume moved per unit trench length ~ s dt T: here 10 km x ~37.5 km of crust, spread over the belt
    assert (np.maximum(out, 0)*area)[near == 0].sum() > 0


def test_margin_shortening_run_keeps_the_ledger():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'margin_shortening': True, 'rollback_mm_yr': 0.})  # shortening everywhere
    v0 = sim.continental_volume()
    for _ in range(15):
        sim.step()
    assert sim.erosion_totals['margin_shortening_moved_m3'] > 0
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0


def test_edge_spread_moves_only_collapse_excess_across_the_coast():
    from planet_layer.sphere_graph import flat
    g = flat(1, 8, 46e3)
    th = np.array([60e3, 60e3, 60e3, 40e3, 7e3, 7e3, 7e3, 7e3])
    cont = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    owner = np.zeros(8, np.int64)
    excess = np.clip(th-45e3, 0, None)
    kap = np.where(excess > 0, 1e5, 0.)
    ok = ~cont
    out = P.edge_spread(th, cont, kap, excess, owner, ok, 1e5, g.nbr, g.dist, g.axis, g.area)
    assert np.allclose(out, th)                          # the coast node (40 km) has no excess: nothing crosses
    th2 = th.copy(); th2[3] = 60e3; ex2 = np.clip(th2-45e3, 0, None); k2 = np.where(ex2 > 0, 1e5, 0.)
    out2 = P.edge_spread(th2, cont, k2, ex2, owner, ok, 1e5, g.nbr, g.dist, g.axis, g.area)
    assert abs(((out2-th2)*g.area).sum()) < 1e-6*th2.sum()*g.area[0]
    assert out2[4] > th2[4] and out2[3] < th2[3] and out2[3] >= 45e3-1e-6
    owner2 = owner.copy(); owner2[4:] = 1                # another plate across the coast: no flux
    assert np.allclose(P.edge_spread(th2, cont, k2, ex2, owner2, ok, 1e5, g.nbr, g.dist, g.axis, g.area), th2)


def test_edge_collapse_run_keeps_the_ledger():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'collapse_edge': True, 'collapse_thick_m': 38000.})
    v0 = sim.continental_volume()
    for _ in range(15):
        sim.step()
    assert sim.erosion_totals.get('edge_collapse_m3', 0.) > 0
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0


def test_ocean_floor_eroded_below_its_sediment_loses_crust(monkeypatch):
    # eighteenth pass: the clamp dsed >= -sed dropped erosion below the sediment from the ledger while the material
    # was deposited elsewhere. With ocean_crust_erosion the crust pays for it: crust + sediment change sums to the
    # kernels' own total (erode_and_deposit's small residual, twentieth pass)
    from planet_layer import graph_landscape as gl
    g = CubeSphere(48)
    kernel = []
    ead = gl.erode_and_deposit

    def counted(*args, **kw):
        r = ead(*args, **kw)
        kernel.append(float(((r[0]-args[0])*g.area).sum()+np.sum(r[3])))
        return r
    monkeypatch.setattr(gl, 'erode_and_deposit', counted)
    out = {}
    for flag in (True, False):
        sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'ocean_crust_erosion': flag})
        for _ in range(3):
            sim.step()
        w = sim.world
        patch = np.flatnonzero(~w['cont'])[::7][:200]          # ocean floor lifted above sea level (a plume), no sediment
        w['sed'][patch] = 0.
        sim.dyn = np.zeros(g.size) if np.isscalar(sim.dyn) else sim.dyn.copy()
        sim.dyn[patch] += 8000.
        kernel.clear()
        e, dsed = sim._erode(w['thick'])
        a = g.area
        out[flag] = (float(((e+dsed)*a).sum())-sum(kernel), float((np.abs(e)*a).sum()), float((e*a)[patch].sum()))
    total, gross, patch_crust = out[True]
    assert abs(total) < 1e-6*gross
    assert patch_crust < 0                                      # the raised ocean floor lost crust
    assert out[False][0] > 1e-3*out[False][1]                   # the old clamp created volume


def _strip_setup(flag):
    """n = 217 (the flexural parameter, ~68 km at Te 30 km, needs ~46 km cells): a 100 km strip of continental crust
    thinned to 12 km along the whole coast, young ocean floor in front of it."""
    from planet_layer import graph_landscape as gl
    g = CubeSphere(217)
    sim = P.PlateSim(g, seed=1, params={'water_volume_km3': None, 'mantle': False, 'thinning_local_isostasy': flag})
    w, p = sim.world, sim.p
    d_ocean = gl.distance_to(~w['cont'], g.nbr, g.dist, 300e3)
    d_cont = gl.distance_to(w['cont'], g.nbr, g.dist, 300e3)
    strip = w['cont'] & (d_ocean <= 100e3)
    front = ~w['cont'] & (d_cont <= 100e3)
    w['thick'][strip] = 12000.
    w['thick'][front] = p['ocean_thick_m']; w['born'][front] = sim.t; w['sed'][front] = 0.
    z = sim.elevation()
    rc, rm, rw = p['rho_crust'], p['rho_mantle'], p['rho_water']
    dry = p['cont_ref_elev_m']+(12000.-p['cont_ref_thick_m'])*(rm-rc)/rm
    own_strip = dry*rm/(rm-rw)
    return g, z, strip, front, own_strip


def test_thinned_strip_stands_at_its_own_depth_and_does_not_lift_the_ocean():
    # tests fixed in the plan doc (eighteenth pass) before the change
    g, z, strip, front, own_strip = _strip_setup(True)
    a = g.area
    assert abs(np.median(z[front]+P.gdh1_depth(0.))) < 300.     # young ocean at its own (ridge) depth
    assert a[front & (z > 0)].sum() <= .01*a[front].sum()       # none of it above sea level
    assert abs(np.median(z[strip]-own_strip)) < 300.            # thinned crust at its own Airy depth
    # as a flexural load (the old elevation) the same strip lifts the ocean in front by kilometres
    g, z0, strip, front, own_strip = _strip_setup(False)
    assert np.median(z0[front]+P.gdh1_depth(0.)) > 1000.


def test_continental_sediment_layer_keeps_both_ledgers(monkeypatch):
    # twentieth pass (tests fixed in the plan doc, nineteenth pass): deposits on continental crust go to the sediment
    # layer, erosion takes sediment first; in every erosion call crust + sediment change equals the surface change
    # the landscape kernels made, and the run's crust and continental-sediment volumes change only by their causes
    from planet_layer import graph_landscape as gl
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'continental_sediment': True})
    a = g.area
    # erode_and_deposit is not exactly closed: now and then (both modes; ~2e-5 of the gross at n = 48) it sends a
    # little more to the sea than it erodes. The partition must reproduce the kernels' total, residual included
    kernel = []
    ead = gl.erode_and_deposit

    def counted(*args, **kw):
        r = ead(*args, **kw)
        kernel.append(float(((r[0]-args[0])*a).sum()+np.sum(r[3])))
        return r
    monkeypatch.setattr(gl, 'erode_and_deposit', counted)
    v0, s0 = sim.continental_volume(), sim.continental_sediment_volume()
    for _ in range(15):
        w = sim.world
        sed_before = w['sed'].copy()
        kernel.clear()
        e, dsed = sim._erode(w['thick'])
        gross = float(((np.abs(e)+np.abs(dsed))*a).sum())
        assert abs(float(((e+dsed)*a).sum())-sum(kernel)) < 1e-9*gross
        assert e[w['cont']].max() <= 1e-9                       # continental crust is never thickened by deposits
        assert np.all(sed_before+dsed >= -1e-6)                 # erosion takes no more sediment than there is
        sim.step()
    assert sim.erosion_totals['continental_sediment_m3'] > 0
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
    s1 = sim.continental_sediment_volume()
    assert s1 > s0
    assert abs((s1-s0)-sum(sim.sed_ledger.values())) < 1e-6*max(s1, 1.)+1e-3
    st = sim.state()
    assert np.isfinite(st['elevation_m']).all() and (st['sediment_m'] >= 0).all()


def test_continental_sediment_is_a_sediment_load():
    # a wide sediment cover of thickness s raises dry continental crust by s (rm - rs) / rm and submerged crust by
    # s (rm - rs) / (rm - rw): filling 1 m of water takes ~2.5 m of sediment (4.8 m when it was booked as crust).
    # The uncompacted layer (sediment_compaction False); the compacting column: test_compacted_sediment_is_a_porous_load
    from planet_layer import graph_landscape as gl
    g = CubeSphere(217)          # n = 217: SphereFlexure gives Airy for wide loads (n = 48 does not)
    for T, wet in ((38000., False), (25000., True)):
        out = {}
        for flag in (False, True):
            sim = P.PlateSim(g, seed=1, params={'water_volume_km3': None, 'mantle': False, 'continental_sediment': flag,
                                                'sediment_compaction': False})
            w, p = sim.world, sim.p
            inner = w['cont'] & (gl.distance_to(~w['cont'], g.nbr, g.dist, 1500e3) > 800e3)
            w['thick'][w['cont']] = T
            w['sed'][w['cont']] = 0.
            z0 = sim.elevation()
            w['sed'][w['cont']] = 1000.
            out[flag] = float(np.median((sim.elevation()-z0)[inner]))
        rm, rs, rw = p['rho_mantle'], p['rho_sediment'], p['rho_water']
        expect = 1000.*(rm-rs)/((rm-rw) if wet else rm)
        assert abs(out[True]-expect) < .05*expect
        assert out[False] == 0.                                 # without the layer the old elevation ignores it


# ---------------------------------------------------------------- twenty-first pass: rift rule stated in distance
def _rift_line(g, half_width_km=1500.):
    """A great-circle rift (gap nodes) through continental crust of 38 km on both sides, for the pure-shear kernel."""
    y = np.arcsin(np.clip(g.xyz[:, 2], -1, 1))*g.radius
    lon = np.arctan2(g.xyz[:, 1], g.xyz[:, 0])
    y0 = y[(np.abs(lon) < .05) & (y >= 0)].min()             # the grid row on (even n) or just north of the equator
    gap = (np.abs(y-y0*np.cos(lon)) < .3*np.sqrt(g.area)) & (np.abs(lon) < .4)   # one row, a segment away from cube edges
    zone = (~gap) & (np.abs(y-y0*np.cos(lon)) < half_width_km*1e3)
    return gap, zone, (y-y0*np.cos(lon))/g.radius


def test_pure_shear_rift_kernel_conserves_volume_and_is_resolution_independent():
    out = {}
    for n in (217, 434):
        g = CubeSphere(n)
        gap, zone, lat = _rift_line(g)
        thick = np.where(zone, 38e3, 0.)
        rows = np.flatnonzero(gap)
        Tbar, Tg, dT = P.rift_pure_shear(rows, g.area, thick, zone, g.nbr, g.dist, 100e3, 0.)
        # every gap node lies between uniform crust: the weighted mean is that crust; the gap gets less (a whole cell
        # opening at once is a finite stretch of the zone), never more, and never less than the most-stretched node
        cmax = -np.log1p(dT[zone].min()/38e3)
        assert np.allclose(Tbar, 38e3) and (Tg <= Tbar*(1+1e-9)).all() and (Tg >= 38e3*np.exp(-cmax)*(1-1e-9)).all()
        # exactly conservative: the volume the zone loses is the volume the gap nodes receive
        assert abs((dT*g.area).sum()+(Tg*g.area[rows]).sum()) < 1e-9*(Tg*g.area[rows]).sum()
        assert (dT[~zone] == 0).all() and (dT <= 0).all()
        # thinning against distance from the rift, per km of rift: where the volume comes from
        dist = np.abs(lat)*g.radius/1e3
        L = gap.sum()*np.sqrt(g.area[rows].mean())/1e3
        v = -(dT*g.area); r = .5*np.sqrt(g.area[rows].mean())/1e3
        out[n] = (float((v*(dist-r)).sum()/v.sum()), float(v[dist-r > 100.].sum()/v.sum()))
    # where the volume comes from, against distance in km: the volume-weighted mean distance from the gap cell's edge is
    # W/3 for the linear taper (33 km at W = 100 km) at 46 and 23 km cells alike, and nothing comes from beyond W
    for n, (mean_d, beyond) in out.items():
        assert abs(mean_d-100/3) < 6 and beyond < .02, (n, out)


def test_pure_shear_rift_run_conserves_volume_and_breaks_up_at_breakup_thickness():
    g = CubeSphere(48)
    p = {'erosion': False, 'rift_rule': 'pure_shear', 'rift_width_km': 600., 'breakup_thick_m': 10000.,
         'mantle': False, 'subduction_H_cont_m': 0., 'arc_H_ocean_m': 0., 'subduction_erosion_H_m': 0.}
    sim = P.PlateSim(g, seed=3, params=p)
    v0 = sim.continental_volume()
    for _ in range(30):
        sim.step()
        rifted = sim.new_crust & sim.world['cont']
        if rifted.any():
            assert sim.world['thick'][rifted].min() >= 10000.-1e-6
    assert sim.erosion_totals.get('rift_nodes', 0) > 0
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0


def test_pure_shear_is_the_default_with_earth_rifting():
    assert P.PARAMS['rift_rule'] == 'pure_shear'
    assert P.PARAMS['rift_width_km'] == 100. and P.PARAMS['breakup_thick_m'] == 10000.
    g = CubeSphere(48)
    a = P.PlateSim(g, seed=3, params={'erosion': False})
    b = P.PlateSim(g, seed=3, params={'erosion': False, 'rift_rule': 'pure_shear'})
    for _ in range(8):
        a.step(); b.step()
    for k in ('cont', 'thick', 'born', 'sed'):
        assert np.array_equal(a.world[k], b.world[k], equal_nan=True)


def test_earth_stable_crust_quantiles_match_crust1():
    """EARTH_STABLE_CRUST_KM is CRUST1.0's crystalline Precambrian / platform crust > 500 km from plate boundaries."""
    from pathlib import Path
    import pytest
    root = Path(__file__).resolve().parents[1]
    if not (root/'data'/'crust1'/'crsthk.xyz').exists():
        pytest.skip('CRUST1.0 not present')
    from planet_layer import graph_landscape as gl
    from scripts.crust_review import wpct
    from scripts.interior_probe import GROUPS
    from scripts.margin_width import C1, crust1_cell, earth_boundaries
    g = CubeSphere(217); a = g.area; cell = crust1_cell(g)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2][cell]*1e3; S = np.loadtxt(C1/'sedthk.xyz')[:, 2][cell]*1e3
    ty = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()]).ravel()[cell]
    dbnd = gl.distance_to(earth_boundaries(g), g.nbr, g.dist, 5000e3)/1e3
    m = (T >= 20e3) & np.isin(ty, GROUPS['Precambrian/platform']) & (dbnd > 500)
    got = [wpct((T-S)[m], a[m], 100*q)/1e3 for q in P.EARTH_STABLE_CRUST_Q]
    assert np.allclose(got, P.EARTH_STABLE_CRUST_KM, atol=.01)


def test_quantile_map_keeps_order_and_takes_the_target_distribution():
    rng = np.random.default_rng(0)
    v = rng.normal(40e3, 1.5e3, 5000); w = rng.uniform(.5, 1.5, 5000)
    out = P.quantile_map(v, w, target=tuple(1e3*k for k in P.EARTH_STABLE_CRUST_KM))
    assert np.all(np.diff(out[np.argsort(v)]) >= 0)                  # order kept (ties only at the clamped ends)
    o = np.argsort(out); c = np.cumsum(w[o]); c /= c[-1]
    for q, k in zip(P.EARTH_STABLE_CRUST_Q[2:-2], P.EARTH_STABLE_CRUST_KM[2:-2]):
        assert abs(out[o][np.searchsorted(c, q)]/1e3-k) < .3


def test_earth_start_keeps_the_field_and_takes_earths_thickness():
    from planet_layer import graph_landscape as gl
    assert P.PARAMS['start_crust'] == 'earth'
    g = CubeSphere(96)
    a = P.PlateSim(g, seed=3, params={'erosion': False, 'start_crust': 'assumed'})
    b = P.PlateSim(g, seed=3, params={'erosion': False, 'start_crust': 'earth'})
    assert np.array_equal(a.world['cont'], b.world['cont'])
    c = a.world['cont']
    inner = c & (gl.distance_to(~c, g.nbr, g.dist, 5000e3) > 200e3)
    ta, tb, w = a.world['thick'][inner], b.world['thick'][inner], g.area[inner]
    assert np.corrcoef(np.argsort(np.argsort(ta)), np.argsort(np.argsort(tb)))[0, 1] > .999    # the field's order kept
    o = np.argsort(tb); cw = np.cumsum(w[o]); cw /= cw[-1]
    assert abs(tb[o][np.searchsorted(cw, .5)]/1e3-38.17) < .5
    assert tb[o][np.searchsorted(cw, .9)]-tb[o][np.searchsorted(cw, .1)] > 7e3                 # Earth's spread (10.1 km)


def test_orogen_rules_mark_more_than_collisions_and_keep_the_ledger():
    assert P.PARAMS['orogen_rule'] == 'belt'
    g = CubeSphere(48)
    base = {'water_volume_km3': None, 'margin_shortening': True, 'rollback_mm_yr': 0.}
    marks = {}
    for rule in ('collision', 'belt', 'strain'):
        sim = P.PlateSim(g, seed=3, params={**base, 'orogen_rule': rule})
        v0 = sim.continental_volume()
        n = 0
        for _ in range(15):
            sim.step()
            m = sim.orogen_marked
            assert np.all(sim.world['orog'][m] == sim.t)                  # the clock is reset where marked
            assert np.all(m[sim.collided])                                # collision contacts always count
            n += int(m.sum())
        marks[rule] = n
        assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
    assert marks['belt'] > marks['collision'] and marks['strain'] > marks['collision']
    with __import__('pytest').raises(ValueError):
        sim = P.PlateSim(g, seed=3, params={**base, 'orogen_rule': 'nonsense'})
        for _ in range(3):
            sim.step()


def test_extension_is_off_by_default_and_inert_without_thick_crust():
    assert P.PARAMS['extension'] is False
    g = CubeSphere(48)
    a = P.PlateSim(g, seed=3, params={'erosion': False})
    b = P.PlateSim(g, seed=3, params={'erosion': False, 'extension': True, 'ext_onset_thick_m': 1e9})
    for _ in range(8):
        a.step(); b.step()
    for k in ('cont', 'thick', 'born', 'sed'):
        assert np.array_equal(a.world[k], b.world[k], equal_nan=True)
    assert not b.ext_events and np.isnan(b.world['ext']).all()


def test_extension_splits_unheld_thick_crust_thins_it_and_welds():
    """E (twenty-seventh pass) on a coarse cube with a low onset so events happen early: thick pieces no convergence
    holds split their plate, the halves diverge at strain rate x zone width, the gaps fill by pure shear of the zone,
    the zone thins to the end thickness and the halves weld; the volume ledger closes and the area ledger is kept.
    Run as E was built (held pieces exempt, no hot rule, no carried clock, whole-plate cut, mean of the halves' motions), so E stays reproducible."""
    g = CubeSphere(48)
    p = {'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 40000., 'ext_min_area_mkm2': .3,
         'ext_held_rule': 'exempt', 'ext_hot_myr': 0., 'ext_clock_flow': False, 'ext_cut': 'whole',
         'ext_pair_motion': 'mean'}
    sim = P.PlateSim(g, seed=3, params=p)
    v0 = sim.continental_volume()
    rel = []
    for _ in range(60):
        sim.step()
        if any(e['kind'] == 'extension end' and not e['guard'] for e in sim.events) and len(sim.events) > 3:
            break
        sim._extension_motion()                        # re-impose after this step's other events (merges, splits)
        for ev in sim.ext_events:                      # the halves diverge at the event's rate
            pa, pb = sim._pair(ev)
            if pa is not None:
                rel.append((np.linalg.norm(sim._omega_vec(pb)-sim._omega_vec(pa)), ev['v']))
    starts = [e for e in sim.events if e['kind'] == 'extension']
    ends = [e for e in sim.events if e['kind'] == 'extension end']
    assert starts and ends
    for e in starts:
        assert abs(e['v']-p.get('ext_strain_rate_per_yr', P.PARAMS['ext_strain_rate_per_yr'])*2*e['W']*100) < 1e-9
        assert P.PARAMS['rift_width_km']*1e3-1 <= e['W'] <= P.PARAMS['ext_max_half_width_km']*1e3+1
    assert rel and all(abs(r-v) < 1e-6*max(v, 1e-9) for r, v in rel)
    welded = [e for e in ends if not e['guard']]
    assert welded and all(e['zone_thick_km'] <= 30.+1e-6 for e in welded)
    st = sim.state()
    ext = np.isfinite(st['extension_age_myr'])
    assert ext.any() and st['continental'][ext].all()
    assert sim.area_ledger['extension'] > 0
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0


def _band_piece(sim, half_width_km=250., half_length_km=1500.):
    """An elongated piece inside the largest plate: nodes within half_width of a great circle through a node near the
    plate's centroid, and within half_length of that node along it. Returns the plate, the piece and the band's axis."""
    g = sim.g
    pl = max(sim.plates, key=lambda q: g.area[sim.owner == q.id].sum())
    own = sim.owner == pl.id
    c = (g.area[own, None]*g.xyz[own]).sum(0); c /= np.linalg.norm(c)
    c = g.xyz[np.flatnonzero(own)[np.argmax(g.xyz[own]@c)]]
    e = np.cross(c, [0.3, 0.5, 0.8]); e /= np.linalg.norm(e)
    m = np.cross(c, e)
    across_d = np.abs(np.arcsin(np.clip(g.xyz@m, -1, 1)))*g.radius
    along_d = np.abs(np.arctan2(g.xyz@e, g.xyz@c))*g.radius
    piece = own & (across_d < half_width_km*1e3) & (along_d < half_length_km*1e3) & (g.xyz@c > 0)
    return pl, piece, e


def test_held_piece_is_cut_across_its_long_axis():
    """E2 (twenty-ninth pass): a held piece is cut across its long axis (the cut's pole lies along the long axis, so the
    halves separate along it: orogen-parallel stretching), a free piece along it (the pole is perpendicular)."""
    assert P.PARAMS['ext_held_rule'] == 'direction' and P.PARAMS['ext_hot_myr'] == 100.
    g = CubeSphere(48)
    normals = {}
    for across in (True, False):
        sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'extension': True})
        sim.step()
        pl, piece, e = _band_piece(sim)
        X = g.xyz[piece]
        Tn = X-(X@X.mean(0))[:, None]*X.mean(0)/np.linalg.norm(X.mean(0))**2
        ev = np.linalg.eigvalsh(Tn.T@Tn)
        assert piece.sum() >= 20 and ev[-1] > 4*ev[-2]          # the piece is elongated along e
        ev_ = sim._start_extension(pl, piece, across=across)
        assert ev_ is not None and ev_['across'] is across
        normals[across] = np.asarray(ev_['cut_normal'])
    assert abs(normals[True]@e) > .9                             # across: the cut's pole along the long axis
    assert abs(normals[False]@e) < .3                            # along: the cut contains the long axis
    assert abs(normals[True]@normals[False]) < .3


def test_crust_without_a_thickening_time_never_triggers_extension():
    """E2: the start's crust has no thickening time (never thickened in the run) and counts as cold, so a thick piece of
    it starts no event; without the hot rule the same state starts events."""
    g = CubeSphere(48)
    base = {'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 38000., 'ext_min_area_mkm2': .3}
    for hot_myr, expect in ((100., False), (0., True)):
        sim = P.PlateSim(g, seed=3, params={**base, 'ext_hot_myr': hot_myr})
        sim.step()
        sim.ext_events.clear()
        sim.world['thk'][:] = np.nan                             # as if nothing had been thickened yet
        n0 = len(sim.events)
        sim._extension_events()
        started = [e for e in sim.events[n0:] if e['kind'] == 'extension']
        assert bool(started) is expect, (hot_myr, started)
        assert sim.ext_reach['gt_onset'] > 0
        assert (sim.ext_reach['eligible'] > 0) is expect
    # in a run, every event starts on a piece more than half hot
    sim = P.PlateSim(g, seed=3, params={**base, 'ext_onset_thick_m': 40000.})
    for _ in range(25):
        sim.step()
    assert all(e['hot_share'] > .5 for e in sim.events if e['kind'] == 'extension')


def test_thickening_clock_is_not_set_by_collapse_diffusion(monkeypatch):
    """E2: the thickening clock follows tectonic thickening only. With subduction thickening switched off and no crust
    consumed (no collision volume), strong collapse diffusion thickens crust faster than orogen_strain_rate, yet no
    node gets a thickening time; with subduction thickening on, the clock is set exactly where it is fast enough."""
    g = CubeSphere(48)
    p = {'water_volume_km3': None, 'mantle': False, 'erosion': False, 'collapse_thick_m': 30000., 'hidden_steps': 10**6}
    sim = P.PlateSim(g, seed=3, params=p)
    monkeypatch.setattr(sim, '_subduction_thickening', lambda: np.zeros(g.size))
    fastest = 0.
    for _ in range(5):
        before = sim.world['thick'].copy()
        sim.step()
        same = sim.world['cont'] & ~sim.new_crust
        rate = (sim.world['thick']-before)/np.maximum(before, 1e3)/(P.PARAMS['dt_myr']*1e6)
        fastest = max(fastest, float(rate[same].max()))
        assert not sim.thk_marked.any()
    assert np.isnan(sim.world['thk']).all() and all(np.isnan(pl.thk[pl.mask]).all() for pl in sim.plates)
    assert fastest > P.PARAMS['orogen_strain_rate']              # the diffusion did thicken crust fast enough
    sim = P.PlateSim(g, seed=3, params={**p, 'margin_shortening': True, 'rollback_mm_yr': 0.})
    for _ in range(5):
        sim.step()
        src = sim.thickening.copy()
        assert np.all(sim.world['thk'][sim.thk_marked] == sim.t)
    assert np.isfinite(sim.world['thk']).any()
    st = sim.state()
    assert np.isfinite(st['thickening_age_myr']).sum() == np.isfinite(sim.world['thk']).sum()


def _no_source_params():
    return {'water_volume_km3': None, 'mantle': False, 'erosion': False, 'collapse_thick_m': 30000., 'hidden_steps': 10**6,
            'extension': True, 'ext_onset_thick_m': 1e9, 'ext_clock_flow': True}


def test_flow_clock_gives_no_time_to_crust_without_a_timed_source(monkeypatch):
    """E3 (thirtieth pass), b'1: the flow only carries existing times. With no tectonic thickening, strong collapse
    diffusion thickens crust faster than orogen_strain_rate, yet the start's crust spreading into the start's crust
    stays untimed."""
    assert P.PARAMS['ext_clock_flow'] is True and P.PARAMS['ext_cut'] == 'band'
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params=_no_source_params())
    monkeypatch.setattr(sim, '_subduction_thickening', lambda: np.zeros(g.size))
    fastest = 0.
    for _ in range(5):
        before = sim.world['thick'].copy()
        sim.step()
        same = sim.world['cont'] & ~sim.new_crust
        rate = (sim.world['thick']-before)/np.maximum(before, 1e3)/(P.PARAMS['dt_myr']*1e6)
        fastest = max(fastest, float(rate[same].max()))
        assert not sim.thk_marked.any()
    assert fastest > P.PARAMS['orogen_strain_rate']
    assert np.isnan(sim.world['thk']).all() and all(np.isnan(pl.thk[pl.mask]).all() for pl in sim.plates)


def test_flow_clock_carries_the_belts_time(monkeypatch):
    """E3, b'2: crust that flows out of a belt timed at t1 carries t1: with no tectonic thickening afterwards, the timed
    area grows and every time the flow carries is t1."""
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params=_no_source_params())
    monkeypatch.setattr(sim, '_subduction_thickening', lambda: np.zeros(g.size))
    sim.step()
    t1 = sim.t
    for pl in sim.plates:                                         # a "belt": each plate's thickest crust, timed at t1
        sel = pl.mask & pl.cont
        if sel.sum() >= 10:
            pl.thk[sel & (pl.thick >= np.quantile(pl.thick[sel], .9))] = t1
    timed = lambda: sum(float(g.area[np.flatnonzero(pl.mask & np.isfinite(pl.thk))].sum()) for pl in sim.plates)
    a0 = timed(); marked_at = {t1}; carried = 0
    for _ in range(5):
        sim.step()
        if sim.thk_marked.any():
            marked_at.add(sim.t)
        c = sim.thk_carried
        f = np.isfinite(c)
        carried += int(f.sum())
        assert np.all(c[f] == t1)
        assert set(np.unique(sim.world['thk'][np.isfinite(sim.world['thk'])])) <= marked_at
    assert carried > 0 and timed() > a0


def test_flow_clock_never_sets_the_present_time():
    """E3, b'3: after every step, the nodes timed at the present are tectonically marked ones; the flow carries only
    earlier times."""
    g = CubeSphere(48)
    p = {**_no_source_params(), 'margin_shortening': True, 'rollback_mm_yr': 0.}
    sim = P.PlateSim(g, seed=3, params=p)
    carried = 0
    for _ in range(8):
        sim.step()
        now = sim.world['thk'] == sim.t
        assert not (now & ~sim.thk_marked).any()
        c = sim.thk_carried
        f = np.isfinite(c)
        carried += int(f.sum())
        assert np.all(c[f] < sim.t)
    assert carried > 0


def _band_event(cut='band', across=False, **extra):
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 1e9,
                                         'ext_cut': cut, 'split_rate_per_myr': 0., 'init_rate_per_myr': 0., **extra})
    sim.step()
    pl, piece, e = _band_piece(sim)
    own0 = sim.owner == pl.id
    ev = sim._start_extension(pl, piece, across=across)
    a = g.area[piece]; X = g.xyz[piece]
    c = (a[:, None]*X).sum(0); c /= np.linalg.norm(c)
    m = np.asarray(ev['cut_normal']); u = np.cross(m, c); u /= np.linalg.norm(u)
    su = np.abs(np.arcsin(np.clip(X@u, -1, 1)))*g.radius
    o = np.argsort(su); cw = np.cumsum(a[o]); cw /= cw[-1]
    L = max(float(su[o][np.searchsorted(cw, .9)]), P.PARAMS['rift_width_km']*1e3)
    return g, sim, pl, piece, own0, ev, c, u, L


def test_band_cut_moves_a_microplate_between_two_transforms():
    """E3, c'1 and c'2: the moving part lies inside the band on one side of the cut, the parent keeps every node outside
    the band; the relative Euler vector lies along u (the point on the cut 90 degrees from the piece's centroid), so the
    relative velocity is tangent to the band's edges (transforms) and carries the part away from the cut; the pole is
    carried in the parent's frame and re-imposed every step."""
    for across in (False, True):
        g, sim, pl, piece, own0, ev, c, u, L = _band_event(across=across)
        assert ev is not None and ev['cut'] == 'band' and abs(ev['L']-L) < 1.
        assert abs(abs(np.asarray(ev['pole'])@u)-1) < 1e-9
        child = [q for q in sim.plates if q.id == ev['child']][0]
        m = np.asarray(ev['cut_normal'])
        dist_u = np.abs(np.arcsin(np.clip(g.xyz@u, -1, 1)))*g.radius
        band = dist_u <= L+1.
        oc = sim.owner == child.id
        assert oc.sum() >= 4 and not (oc & ~own0).any()
        assert band[oc].all()
        sg = np.sign(g.xyz[oc]@m)
        assert (sg == sg[0]).all()
        assert (sim.owner[own0 & ~band] == pl.id).all()
        dw = sim._omega_vec(child)-sim._omega_vec(pl)
        assert abs(dw@u)/np.linalg.norm(dw) > .99 and abs(np.linalg.norm(dw)-ev['v']) < 1e-6*ev['v']
        assert np.cross(dw, c)@(sg[0]*m) > 0                       # the part moves away from the cut
        nb = g.nbr; okn = nb >= 0
        edge = oc & np.any(okn & (sim.owner[np.where(okn, nb, 0)] == pl.id), axis=1)
        side = edge & (dist_u > L-2*np.sqrt(g.area.mean()))
        assert side.sum() >= 2
        X = g.xyz[side]; v = np.cross(dw, X)
        n = u-(X@u)[:, None]*X; n /= np.linalg.norm(n, axis=1, keepdims=True)
        assert np.all(np.abs((v*n).sum(1)) < .05*np.linalg.norm(v, axis=1))
        for _ in range(3):
            sim.step()
            if ev not in sim.ext_events:
                break
            sim._extension_motion()
            pa, pb = sim._pair(ev)
            uw = np.asarray(ev['pole_body'])@pa.Q.T
            d = sim._omega_vec(pb)-sim._omega_vec(pa)
            assert abs(d@uw)/np.linalg.norm(d) > .99 and abs(np.linalg.norm(d)-ev['v']) < 1e-6*ev['v']


def test_band_cut_opens_no_rift_along_the_cut_beyond_the_band():
    """E3, c'3: with one event (kept open) and no other splits, no new crust opens between the two halves along the cut
    beyond the band (within 4 node spacings of the cut, more than 1.5 beyond the band's edge); the whole-plate cut (E2)
    opens it there. Nodes on the band's side edges far from the cut are the transforms' staircase at n = 48 (this test
    runs at 5x Earth's strain rate), not rifting: at 2e-8 /yr the band cut adds no rift area over a run without the
    event, the whole cut +0.6 Mkm^2 in 12 Myr (checked 25 Oct)."""
    for cut in ('band', 'whole'):
        g, sim, pl, piece, own0, ev, c, u, L = _band_event(cut, ext_strain_rate_per_yr=1e-7, ext_end_thick_m=1.)
        u_body = u@pl.Q; m_body = np.asarray(ev['cut_normal'])@pl.Q
        sp = np.sqrt(g.area.mean())
        inside = along = 0
        for _ in range(12):
            sim.step()
            assert ev in sim.ext_events
            pa, pb = sim._pair(ev)
            nb = g.nbr; okn = nb >= 0; on = sim.owner[np.where(okn, nb, 0)]
            gap = sim.new_crust & (((sim.owner == pa.id) & np.any(okn & (on == pb.id), 1))
                                   | ((sim.owner == pb.id) & np.any(okn & (on == pa.id), 1)))
            du = np.abs(np.arcsin(np.clip(g.xyz@(u_body@pa.Q.T), -1, 1)))*g.radius
            dm = np.abs(np.arcsin(np.clip(g.xyz@(m_body@pa.Q.T), -1, 1)))*g.radius
            inside += int((gap & (du <= L+1.5*sp)).sum())
            along += int((gap & (du > L+1.5*sp) & (dm < 4*sp)).sum())
        assert inside > 0, cut
        assert (along == 0) if cut == 'band' else (along > 0), (cut, inside, along)


def test_pair_moves_as_the_unsplit_plate_would():
    """E3, ext_pair_motion 'union': while an event runs, the halves' mean motion is the usual forces on the pair as one
    plate, so cutting off a microplate (which may take the plate's collision boundary with it) does not change it: one
    motion update after the cut gives the pair the motion the unsplit plate gets. The halves still diverge about the
    event's pole at its rate."""
    assert P.PARAMS['ext_pair_motion'] == 'union'
    g = CubeSphere(48)
    prm = {'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 1e9,
           'split_rate_per_myr': 0., 'init_rate_per_myr': 0.}
    for across in (False, True):
        sim = P.PlateSim(g, seed=3, params=prm); sim.step()
        ref = P.PlateSim(g, seed=3, params=prm); ref.step()
        pl, piece, e = _band_piece(sim)
        ev = sim._start_extension(pl, piece, across=across)
        assert ev is not None and 'base_speed' in ev
        sim._update_motion(); sim._extension_motion()
        ref._update_motion()
        pa, pb = sim._pair(ev)
        whole = ref._omega_vec([q for q in ref.plates if q.id == pl.id][0])
        mean = (sim._omega_vec(pa)+sim._omega_vec(pb))/2
        assert np.allclose(mean, whole, rtol=0, atol=1e-9*np.linalg.norm(whole))
        d = sim._omega_vec(pb)-sim._omega_vec(pa)
        assert abs(d@np.asarray(ev['pole']))/np.linalg.norm(d) > 1-1e-9 and abs(np.linalg.norm(d)-ev['v']) < 1e-6*ev['v']


def test_e3_switches_are_inert_without_extension():
    """d: with extension off (the default path) the E3 switches change nothing."""
    g = CubeSphere(48)
    a = P.PlateSim(g, seed=3, params={'erosion': False})
    b = P.PlateSim(g, seed=3, params={'erosion': False, 'ext_clock_flow': False, 'ext_cut': 'whole'})
    for _ in range(8):
        a.step(); b.step()
    for k in ('cont', 'thick', 'born', 'sed', 'thk'):
        assert np.array_equal(a.world[k], b.world[k], equal_nan=True)


def _sides_of_band_cut(sim, pl, piece, across=False):
    """The two candidate sides of a band cut of `piece` (as _start_extension builds them) and the plate nodes on each
    side's far edge (next to another plate)."""
    g = sim.g
    a = g.area[piece]; X = g.xyz[piece]
    c = (a[:, None]*X).sum(0); c /= np.linalg.norm(c)
    Tn = X-(X@c)[:, None]*c
    C = (a[:, None, None]*Tn[:, :, None]*Tn[:, None, :]).sum(0)
    u = np.linalg.eigh(C)[1][:, -2 if across else -1]; u = u-(u@c)*c; u /= np.linalg.norm(u)
    m = np.cross(c, u); m /= np.linalg.norm(m)
    su = np.abs(np.arcsin(np.clip(X@u, -1, 1)))*g.radius
    o = np.argsort(su); cw = np.cumsum(a[o]); cw /= cw[-1]
    L = max(float(su[o][np.searchsorted(cw, .9)]), P.PARAMS['rift_width_km']*1e3)
    own = sim.owner == pl.id
    band = own & (np.abs(np.arcsin(np.clip(g.xyz@u, -1, 1)))*g.radius <= L)
    nb = g.nbr; okn = nb >= 0; on = sim.owner[np.where(okn, nb, 0)]
    out = []
    for sg in (1., -1.):
        cand = band & (sg*(g.xyz@m) > 0)
        lab = P.components(cand, nb)
        keep = np.unique(lab[piece & cand]); keep = keep[keep >= 0]
        q = cand & np.isin(lab, keep)
        out.append((q, q & np.any(okn & (on != pl.id), axis=1)))
    return out


def test_free_side_moves_away_from_the_collision():
    """E4, g1: a thick piece between a collision on one side and a subduction or ridge boundary on the other. With
    ext_side 'free' the side facing the free boundary moves, whichever is smaller; with 'smaller' (E3) the smaller side
    moves even when it faces the collision. Two free sides (or equal shares) go to the smaller side."""
    g = CubeSphere(48)
    prm = {'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 1e9,
           'split_rate_per_myr': 0., 'init_rate_per_myr': 0.}
    seen_larger = False
    for across in (False, True):
        for coll_on_smaller in (True, False, None):          # None: both far edges free (ridge / subduction)
            got = {}
            for rule in ('free', 'smaller'):
                sim = P.PlateSim(g, seed=3, params={**prm, 'ext_side': rule}); sim.step()
                pl, piece, e = _band_piece(sim)
                (q0, e0), (q1, e1) = _sides_of_band_cut(sim, pl, piece, across)
                assert e0.any() and e1.any() and q0.sum() >= 4 and q1.sum() >= 4
                small = 0 if g.area[q0].sum() <= g.area[q1].sum() else 1
                edges = (e0, e1)
                bt = sim.btype.copy()
                bt[e0 | e1] = P.RIDGE
                if coll_on_smaller is not None:
                    bt[edges[small if coll_on_smaller else 1-small]] = P.COLLISION
                    bt[edges[1-small if coll_on_smaller else small]] = P.SUBDUCTION
                sim.btype = bt
                own0 = sim.owner == pl.id
                ev = sim._start_extension(pl, piece, across=across)
                assert ev is not None and ev['side'] == rule
                child = sim.owner == ev['child']
                moved = 0 if (child & q0).sum() > (child & q1).sum() else 1
                assert not (child & ~(q0 | q1) & own0).any()
                got[rule] = moved
                fc = ev['far_collision_sides']
                if coll_on_smaller is None:
                    assert fc[0] == fc[1] == 0.
                else:
                    assert fc[small if coll_on_smaller else 1-small] == 1. and fc[1-small if coll_on_smaller else small] == 0.
                    assert ev['far_collision'] == (1. if moved == (small if coll_on_smaller else 1-small) else 0.)
                assert ev['moved_larger'] == (moved != small)
            assert got['smaller'] == small
            if coll_on_smaller:
                assert got['free'] == 1-small
                seen_larger = True
            else:
                assert got['free'] == small
    assert seen_larger


def test_e4_side_rule_is_inert_without_extension_and_for_the_whole_cut():
    """E4, d: with extension off the side rule changes nothing; the whole-plate cut (E/E2) ignores it."""
    g = CubeSphere(48)
    a = P.PlateSim(g, seed=3, params={'erosion': False})
    b = P.PlateSim(g, seed=3, params={'erosion': False, 'ext_side': 'smaller'})
    for _ in range(6):
        a.step(); b.step()
    for k in ('cont', 'thick', 'born', 'sed', 'thk'):
        assert np.array_equal(a.world[k], b.world[k], equal_nan=True)
    assert P.PARAMS['ext_side'] == 'free'
    prm = {'water_volume_km3': None, 'mantle': False, 'extension': True, 'ext_onset_thick_m': 1e9, 'ext_cut': 'whole',
           'split_rate_per_myr': 0., 'init_rate_per_myr': 0.}
    owners = []
    for rule in ('free', 'smaller'):
        sim = P.PlateSim(g, seed=3, params={**prm, 'ext_side': rule}); sim.step()
        pl, piece, e = _band_piece(sim)
        ev = sim._start_extension(pl, piece)
        assert ev is not None and 'side' not in ev
        owners.append(sim.owner.copy())
    assert np.array_equal(owners[0], owners[1])


def _two_plates(g, m, offset_km=0.):
    """Owner 1 where x.m > sin(offset / R) (a great circle, or a small circle about m), else 0; and the boundary nodes."""
    owner = (g.xyz@m > np.sin(offset_km*1e3/g.radius)).astype(np.int64)
    nb = g.nbr; ok = nb >= 0
    return owner, np.any(ok & (owner[np.where(ok, nb, 0)] != owner[:, None]), axis=1)


def test_ring_normal_types_transforms_at_any_orientation():
    """Thirty-third pass, b1 / b2: two plates on a great circle (or a small circle 400 km off it, the edge of an E3 band)
    moving about the circle's own pole: a pure transform. With boundary_normal 'ring' at least 95% of the boundary nodes
    are typed transform at each of 12 random orientations, n = 48 and 217; with 'node' (as built) the median is ~0.3 and
    some orientations are 0 (recorded, not required)."""
    rng = np.random.default_rng(0)
    for n in (48, 217):
        g = CubeSphere(n)
        node = []
        for _ in range(12):
            m = rng.normal(size=3); m /= np.linalg.norm(m)
            for off in (0., 400.):
                owner, b = _two_plates(g, m, off)
                omega = np.stack([np.zeros(3), m*.05/g.radius])        # 5 cm/yr at 90 degrees
                args = (owner, np.array([0, 1]), omega, g.xyz, g.nbr, np.zeros(g.size), np.zeros(g.size, bool), g.radius, 5e-3)
                bt = P.classify_boundaries_ring(*args)[0]
                assert np.mean(bt[b] == P.TRANSFORM) >= .95, (n, off, np.mean(bt[b] == P.TRANSFORM))
                node.append(np.mean(P.classify_boundaries(*args)[0][b] == P.TRANSFORM))
        assert np.median(node) < .5                                    # the defect the ring normal removes


def test_ring_normal_types_opening_and_closing_as_before():
    """b3: rotation about a pole on the boundary opens it on one side of the pole and closes it on the other. Where the
    true normal rate exceeds twice the transform threshold, the ring normal types >= 98% of the nodes ridge (opening),
    subduction (oceanic) or collision (continental on both sides) (closing), as the node kernel does; trench nodes only
    on the lower-score side of subduction."""
    rng = np.random.default_rng(1)
    g = CubeSphere(96)
    for _ in range(6):
        m = rng.normal(size=3); m /= np.linalg.norm(m)
        e = np.cross(m, rng.normal(size=3)); e /= np.linalg.norm(e)
        owner, b = _two_plates(g, m)
        w = e*.05/g.radius
        omega = np.stack([np.zeros(3), w])
        sgn = np.where(owner == 0, 1., -1.)                            # normal from each node toward the other plate
        X = g.xyz
        v = np.cross(np.where(owner[:, None] == 0, -w, w), X)*g.radius    # own plate relative to the other
        true = (v@m)*sgn                                               # closing rate (+) along the true normal m
        far = b & (np.abs(true) > 2*5e-3)
        for cont in (np.zeros(g.size, bool), np.ones(g.size, bool)):
            score = owner.astype(float)
            bt, r, tr = P.classify_boundaries_ring(owner, np.array([0, 1]), omega, X, g.nbr, score, cont, g.radius, 5e-3)
            closing = P.COLLISION if cont.all() else P.SUBDUCTION
            ok = np.where(true[far] > 0, bt[far] == closing, bt[far] == P.RIDGE)
            assert ok.mean() >= .98, ok.mean()
            assert np.all(np.sign(r[far]) == np.sign(true[far]))
            if not cont.any():
                assert not tr[(owner == 1) & (bt == P.SUBDUCTION)].any() and tr[(owner == 0) & (bt == P.SUBDUCTION)].any()


def test_ring_normal_is_the_default_and_keeps_the_ledgers():
    """b4 / b5: 'ring' is the default since the thirty-fourth pass ('node' = the code before it); a short run with
    'ring' keeps the volume ledger and the run invariants (every node owned, plate masks consistent)."""
    assert P.PARAMS['boundary_normal'] == 'ring'
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'boundary_normal': 'ring'})
    v0 = sim.continental_volume()
    ref = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'boundary_normal': 'node'})
    for _ in range(10):
        sim.step(); ref.step()
    assert (sim.owner >= 0).all()
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
    nb = g.nbr; ok = nb >= 0
    for s in (sim, ref):
        b = np.any(ok & (s.owner[np.where(ok, nb, 0)] != s.owner[:, None]), axis=1)
        assert (s.btype[b] > 0).all() and (s.btype[~b] == 0).all()
    tr = lambda s: np.mean(s.btype[s.btype > 0] == P.TRANSFORM)
    assert tr(sim) > tr(ref)


# ---------------------------------------------------------------- thirty-sixth pass: sediment compaction (B-a)
def test_bulk_thickness_inverts_athys_law():
    # S = h - phi0 / c (1 - exp(-c h)); thin covers stand at S / (1 - phi0), thick columns gain ~1 m of bulk per m of
    # solid at the base (dh/dS = 1 / (1 - phi(h)))
    S = np.array([0., 1., 100., 500., 2000., 8000., 20000.])
    for phi0, c in ((.63, .51), (.56, .39)):
        h = P.bulk_thickness(S, phi0, c)
        k = c/1e3
        assert np.allclose(h-phi0/k*(1-np.exp(-k*h)), S, atol=1e-6)
        assert abs(h[1]/(1/(1-phi0))-1) < 2e-3                  # 1 m: compaction within it is ~0.1%
        assert np.all(np.diff(h) > 0) and np.all(h >= S)
        dh = (P.bulk_thickness(S[-1]+1., phi0, c)-h[-1])
        assert abs(dh-1/(1-phi0*np.exp(-k*h[-1]))) < 1e-3
    # Sclater & Christie's shale: a 0.5 km solid cover is ~1 km of bulk sediment at ~1,850 kg/m^3
    h = float(P.bulk_thickness(500., .63, .51))
    assert 950 < h < 1030 and 1800 < (2650*500+1030*(h-500))/h < 1900


def test_compaction_is_the_default_and_off_reads_the_old_layer():
    # default since the thirty-seventh pass (sweep-20261025-201623); False reproduces the runs before it
    assert P.PARAMS['sediment_compaction']
    off = {**P.PARAMS, 'sediment_compaction': False}
    sed = np.array([0., 300., 3000.]); cont = np.array([False, True, False])
    h, md, mw = P.sediment_column(sed, cont, off)
    assert np.array_equal(h, sed) and np.array_equal(md, P.PARAMS['rho_sediment']*sed)
    assert np.array_equal(mw, (P.PARAMS['rho_sediment']-P.PARAMS['rho_water'])*sed)
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'sediment_compaction': False})
    for _ in range(3):
        sim.step()
    st = sim.state()
    assert np.array_equal(st['sediment_bulk_m'], st['sediment_m'])


def test_compacted_sediment_is_a_porous_load():
    # a wide cover of solid thickness S raises the deep ocean floor by h - (rho_grain - rho_w) S / (rho_m - rho_w)
    # (bulk height, buoyant solid mass; Airy at long wavelength) and dry continental crust by h - (rho_grain S +
    # rho_w (h - S)) / rho_m
    from planet_layer import graph_landscape as gl
    g = CubeSphere(217)          # n = 217: SphereFlexure gives Airy for wide loads (n = 48 does not)
    S = 500.
    for on_cont in (False, True):
        sim = P.PlateSim(g, seed=1, params={'water_volume_km3': None, 'mantle': False, 'continental_sediment': True,
                                            'sediment_compaction': True})
        w, p = sim.world, sim.p
        where = w['cont'] if on_cont else ~w['cont']
        inner = where & (gl.distance_to(~where, g.nbr, g.dist, 1500e3) > 800e3)
        if on_cont:
            w['thick'][w['cont']] = 38000.
        w['sed'][:] = 0.
        z0 = sim.elevation()
        w['sed'][where] = S
        dz = float(np.median((sim.elevation()-z0)[inner]))
        rm, rw, rg = p['rho_mantle'], p['rho_water'], p['rho_grain']
        phi0, c = ((p['phi0_continent'], p['compaction_c_continent_per_km']) if on_cont else
                   (p['phi0_ocean'], p['compaction_c_ocean_per_km']))
        h = float(P.bulk_thickness(S, phi0, c))
        expect = h-((rg*S+rw*(h-S))/rm if on_cont else (rg-rw)*S/(rm-rw))
        assert abs(dz-expect) < .05*expect, (on_cont, dz, expect)
        assert inner.sum() > 100


def test_compaction_run_keeps_both_ledgers(monkeypatch):
    # with compaction the sediment field is solid volume: the kernels' closure, the crust and continental-sediment
    # ledgers hold as without it, and the snapshot's bulk thickness is the column's
    from planet_layer import graph_landscape as gl
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'continental_sediment': True,
                                        'sediment_compaction': True})
    a = g.area
    kernel = []
    ead = gl.erode_and_deposit

    def counted(*args, **kw):
        r = ead(*args, **kw)
        kernel.append(float(((r[0]-args[0])*a).sum()+np.sum(r[3])))
        return r
    monkeypatch.setattr(gl, 'erode_and_deposit', counted)
    v0, s0 = sim.continental_volume(), sim.continental_sediment_volume()
    for _ in range(15):
        w = sim.world
        sed_before = w['sed'].copy()
        kernel.clear()
        e, dsed = sim._erode(w['thick'])
        gross = float(((np.abs(e)+np.abs(dsed))*a).sum())
        assert abs(float(((e+dsed)*a).sum())-sum(kernel)) < 1e-9*gross
        assert np.all(sed_before+dsed >= -1e-6)
        sim.step()
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
    s1 = sim.continental_sediment_volume()
    assert abs((s1-s0)-sum(sim.sed_ledger.values())) < 1e-6*max(s1, 1.)+1e-3
    st = sim.state()
    assert np.isfinite(st['elevation_m']).all()
    h, _, _ = P.sediment_column(sim.world['sed'], sim.world['cont'], sim.p)
    assert np.allclose(st['sediment_bulk_m'], h, rtol=1e-6, atol=1e-3)
    pos = st['sediment_m'] > 1
    assert np.all(st['sediment_bulk_m'][pos] > st['sediment_m'][pos])


# ---------------------------------------------------------------- thirty-seventh pass: plate pieces (B-b)
def _plant_piece(sim, radius_km, cont=False):
    """At the start every plate frame is the world's (Q = I): give plate A a disc of B's territory (B the largest
    plate, the disc around its node farthest from its edge), i.e. a detached piece of A inside B. Returns A, B, disc."""
    from planet_layer import graph_landscape as gl
    g = sim.g
    areas = {q.id: g.area[sim.owner == q.id].sum() for q in sim.plates}
    B = max(sim.plates, key=lambda q: areas[q.id])
    A = next(q for q in sim.plates if q is not B)
    own = sim.owner == B.id
    d = gl.distance_to(~own, g.nbr, g.dist)
    c = int(np.argmax(np.where(own, d, -1)))
    assert d[c] > radius_km*1e3+400e3
    disc = own & (np.arccos(np.clip(g.xyz@g.xyz[c], -1, 1))*g.radius <= radius_km*1e3)
    for name in ('cont', 'thick', 'born', 'orog', 'sed', 'ext', 'thk', 'under'):
        getattr(A, name)[disc] = getattr(B, name)[disc]
    A.mask[disc] = True; A.cont[disc] = cont
    A.thick[disc] = 35000. if cont else 7000.
    A.born[disc] = sim.t-5.
    B.mask[disc] = False; B.cont[disc] = False
    sim.compose_from_plates()
    sim._wn = {q.id: P.fields.nearest_nodes(g, g.xyz@q.Q.T) for q in sim.plates}
    return A, B, disc


def test_piece_rule_plates_is_the_default_and_shared_keeps_pieces():
    # 'plates' is the default since the thirty-eighth pass (sweep-20261025-220434); 'shared' (each piece keeps its
    # plate's pole) is the code before the thirty-seventh pass, step by step
    assert P.PARAMS['piece_rule'] == 'plates' and P.PARAMS['piece_min_area_mkm2'] == 1.
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'piece_rule': 'shared'})
    A, B, disc = _plant_piece(sim, 700.)
    sim.step()
    assert sim.detached_area() > .5*g.area[disc].sum()       # the piece is still A's, far from A
    assert not any(e['kind'] == 'detach' for e in sim.events)


def test_large_detached_piece_becomes_a_plate_with_its_parents_motion():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'piece_rule': 'plates'})
    A, B, disc = _plant_piece(sim, 700.)
    assert g.area[disc].sum() > 1e12 and sim.detached_area() >= (1-1e-9)*g.area[disc].sum()
    ids0 = {q.id for q in sim.plates}
    axis, speed = A.axis.copy(), A.speed
    sim._separate_pieces()
    new = [q for q in sim.plates if q.id not in ids0]
    assert len(new) == 1
    q = new[0]
    assert (sim.owner[disc] == q.id).all() and not (sim.owner == q.id)[~disc].any()
    assert np.allclose(q.axis, axis) and abs(q.speed-speed) < 1e-12        # the parent's Euler vector at the split
    assert np.allclose(q.Q, A.Q) and q.mask[disc].all() and not A.mask[disc].any()
    assert sim.events[-1]['kind'] == 'detach' and sim.events[-1]['plate'] == A.id
    assert sim.detached_area() < 1e-6*g.area.sum()
    v0 = sim.continental_volume()
    for _ in range(5):                                          # then its own forces
        sim.step()
    assert (sim.owner >= 0).all()
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0


def test_small_detached_piece_joins_the_plate_around_it():
    g = CubeSphere(48)
    for cont in (False, True):
        sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'piece_rule': 'plates',
                                            'piece_min_area_mkm2': 100.})
        A, B, disc = _plant_piece(sim, 700., cont=cont)
        v0, n0 = sim.continental_volume(), len(sim.plates)
        axis, speed = B.axis.copy(), B.speed
        sim._separate_pieces()
        assert len(sim.plates) == n0 and (sim.owner[disc] == B.id).all()
        assert B.mask[disc].all() and not A.mask[disc].any()
        assert np.all(B.born[disc] == np.float32(sim.t-5.))       # B's raster reads the piece's crust
        assert (B.cont[disc] == cont).all()
        assert np.allclose(B.axis, axis) and B.speed == speed      # the capturing plate keeps its motion
        dv = sim.continental_volume()-v0
        assert abs(dv-sim.ledger['pieces']) < 1e-6*max(v0, 1.)
        assert abs(dv) < 1e-3*float((35000.*g.area[disc]).sum())   # Q = I: the piece moves node for node
        assert sim.detached_area() < 1e-6*g.area.sum()


def test_boundary_crumbs_are_left_alone():
    # a piece with no interior node (every node touches another plate) is grid staircase, not lithosphere
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'piece_rule': 'plates'})
    A, B, disc = _plant_piece(sim, 1.)
    assert disc.sum() == 1
    owner0 = sim.owner.copy()
    sim._separate_pieces()
    assert np.array_equal(sim.owner, owner0) and sim.piece_stats['crumbs_m2_steps'] > 0


def test_plates_rule_run_keeps_the_ledgers_and_connected_plates():
    # seed 3 at n = 48 makes plates from pieces and welds pieces; each weld books its crust and sediment change
    # exactly. The run goes at least 60 steps and stops once a plate is made and two pieces are welded (at most 150):
    # first welds come after 35-66 steps (seeds 3-7, Linux), so a fixed 60 steps missed them on another machine's
    # trajectory (Windows, 26 Oct: two plates, no weld). (The whole-run sediment ledger drifts by ~1e-5-1e-4 here with
    # either rule: crust read and consumed in the same step is booked at its pre-increment value in _update_rasters;
    # thirty-seventh pass.)
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'piece_rule': 'plates', 'continental_sediment': True})
    weld = sim._weld
    welds = []

    def booked(src, dst, ids):
        v, s = sim.continental_volume(), sim.continental_sediment_volume()
        lv, ls = sim.ledger.get('pieces', 0.), sim.sed_ledger.get('pieces', 0.)
        weld(src, dst, ids)
        welds.append((sim.continental_volume()-v-(sim.ledger['pieces']-lv),
                      sim.continental_sediment_volume()-s-(sim.sed_ledger['pieces']-ls), v))
    sim._weld = booked
    v0 = sim.continental_volume()
    for i in range(150):
        sim.step()
        assert (sim.owner >= 0).all()
        assert sim.detached_area() <= .01*g.area.sum()
        if i >= 59 and sim.piece_stats['plates_made'] >= 1 and len(welds) >= 2:
            break
    assert sim.piece_stats['plates_made'] >= 1 and len(welds) >= 2
    for dv, ds, v in welds:
        assert abs(dv) < 1e-9*v and abs(ds) < 1e-9*v
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
    assert sim.pinholes() == 0


# ---------------------------------------------------------------- thirty-eighth pass: the start's plate geometry
def _rim(sim):
    g = sim.g
    nb = g.nbr; ok = nb >= 0; nbc = np.where(ok, nb, 0)
    c = sim.world['cont']
    rim = (~c) & np.any(ok & c[nbc], 1)                                   # ocean nodes next to continental crust
    on_boundary = rim & np.any(ok & c[nbc] & (sim.owner[nbc] != sim.owner[:, None]), 1)
    return rim, on_boundary


def test_start_plates_voronoi_is_the_default():
    # 'voronoi' is the start before the thirty-eighth pass (same random draws, same labels)
    assert P.PARAMS['start_plates'] == 'voronoi'
    g = CubeSphere(48)
    a = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False})
    b = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'start_plates': 'voronoi'})
    assert np.array_equal(a.owner, b.owner) and np.array_equal(a.world['born'], b.world['born'])


def test_coast_start_plates_end_at_the_coast():
    # every start plate is either continental or oceanic, connected, and the supercontinent's rim is a plate boundary
    from planet_layer.plates import components
    g = CubeSphere(48)
    for seed in (3, 4):
        sim = P.PlateSim(g, seed=seed, params={'water_volume_km3': None, 'mantle': False, 'start_plates': 'coast'})
        assert (sim.owner >= 0).all()
        ncont = 0
        for pl in sim.plates:
            m = pl.mask
            assert m.any() and (pl.cont[m].all() or not pl.cont[m].any())
            ncont += int(pl.cont[m].all())
            lab = components(sim.owner == pl.id, g.nbr)
            assert lab.max() == 0                                          # one piece
        assert ncont == 5
        rim, on_boundary = _rim(sim)
        assert on_boundary.sum() == rim.sum()


def test_coast_start_rim_subducts_and_keeps_the_ledgers():
    # with the continental plates moving away from the supercontinent's centre, a large part of the rim converges and
    # is typed subduction within a few steps ('voronoi': the rim sits inside plates, under 0.2 of it subducts at n = 48)
    g = CubeSphere(48)
    share = {}
    for mode in ('voronoi', 'coast'):
        sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'mantle': False, 'start_plates': mode})
        v0 = sim.continental_volume()
        for _ in range(5):
            sim.step()
        assert (sim.owner >= 0).all()
        assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 2e-3*v0
        rim, _ = _rim(sim)
        share[mode] = (rim & (sim.btype == P.SUBDUCTION)).sum()/rim.sum()
    assert share['coast'] > .3 and share['voronoi'] < .2


def test_ledgers_close_over_a_long_run():
    # thirty-eighth pass: consumed plate nodes that are still read in the same step are booked at their value when they
    # leave, and collision volume landing on new crust is no longer booked twice; both ledgers then close to round-off
    # (they drifted by 1e-5-4e-4 of the stock over 60 steps on seeds 3 / 5 / 7 at n = 48 before)
    g = CubeSphere(48)
    for seed in (3, 5):
        sim = P.PlateSim(g, seed=seed, params={'water_volume_km3': None, 'continental_sediment': True})
        v0, s0 = sim.continental_volume(), sim.continental_sediment_volume()
        for _ in range(60):
            sim.step()
        assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 1e-9*v0
        assert abs((sim.continental_sediment_volume()-s0)-sum(sim.sed_ledger.values())) < 1e-9*v0


def test_start_poles_forces_moves_ocean_plates_toward_the_ring():
    # 'random' is the default; 'forces' sets each oceanic start plate's speed and pole from the force law fully relaxed
    # on the start's geometry (coast = trench, ocean-ocean = ridge), leaves the continental plates' motion, draws no
    # random numbers, and makes more of the rim subduct than random poles do (n = 48, seed 3: 0.83 vs 0.53 after a step)
    assert P.PARAMS['start_poles'] == 'random'
    g = CubeSphere(48)
    base = {'water_volume_km3': None, 'mantle': False, 'start_plates': 'coast'}
    a = P.PlateSim(g, seed=3, params={**base, 'start_poles': 'random'})
    b = P.PlateSim(g, seed=3, params={**base, 'start_poles': 'forces'})
    assert not hasattr(b, 'trench') and not hasattr(b, 'btype')
    assert np.array_equal(a.owner, b.owner) and a.rng.bit_generator.state == b.rng.bit_generator.state
    nb = g.nbr; ok = nb >= 0; nbc = np.where(ok, nb, 0)
    c = b.world['cont']
    for pa, pb in zip(a.plates, b.plates):
        own = b.owner == pb.id
        if (own & c).any():
            assert np.allclose(pa.axis, pb.axis) and pa.speed == pb.speed
        else:
            ctr = (g.area[own, None]*g.xyz[own]).sum(0); ctr /= np.linalg.norm(ctr)
            tr = own & np.any(ok & c[nbc], 1)
            if tr.any():                                   # its centre moves toward its trench nodes
                v = np.cross(pb.axis, ctr)
                assert (g.xyz[tr]@v).mean() > 0
    share = {}
    for k, sim in (('random', a), ('forces', b)):
        sim.step()
        c = sim.world['cont']
        rim = (~c) & np.any(ok & c[nbc], 1)
        share[k] = (rim & (sim.btype == P.SUBDUCTION)).sum()/rim.sum()
    assert share['forces'] > share['random']+.15


# ---------------------------------------------------------------- thirty-ninth pass: B-d, island arcs dock
def test_arc_accretion_is_off_by_default():
    assert P.PARAMS['arc_accretion'] is False
    assert P.PARAMS['arc_buoyant_thick_m'] == 15000. and P.PARAMS['arc_accrete_upper_m'] == 3500.
    assert P.PARAMS['arc_H_ocean_m'] == 300.


def test_docked_arcs_keep_the_ledgers():
    # with arc_accretion, consumed oceanic crust thicker than normal hands its excess (all of it from 15 km, else its
    # upper 3.5 km) to the overriding plate as volume; on continental crust that is booked as increments, so both
    # ledgers still close to round-off, and nothing docks without the switch
    g = CubeSphere(48)
    out = {}
    for acc in (False, True):
        sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'continental_sediment': True,
                                            'arc_H_ocean_m': 1000., 'arc_accretion': acc})
        v0, s0 = sim.continental_volume(), sim.continental_sediment_volume()
        for _ in range(60):
            sim.step()
        assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 1e-9*v0
        assert abs((sim.continental_sediment_volume()-s0)-sum(sim.sed_ledger.values())) < 1e-9*v0
        out[acc] = dict(sim.erosion_totals)
    assert 'arc_accreted_m3' not in out[False]
    assert out[True]['arc_accreted_m3'] > 0
    assert 0 <= out[True]['arc_accreted_on_continent_m3'] <= out[True]['arc_accreted_m3']
    assert out[True]['ocean_crust_eroded_subaerial_normal_m3'] >= out[True]['ocean_crust_eroded_subaerial_m3']   # both <= 0


def test_docked_volume_spreads_over_the_arc_band_and_is_conserved():
    g = CubeSphere(48)
    rng = np.random.default_rng(0)
    trench = np.zeros(g.size, bool); trench[rng.choice(g.size, 40, replace=False)] = True
    near, d = P.nearest_label(np.where(trench, np.arange(g.size), -1).astype(np.int64), g.nbr, g.dist)
    band = d < 600e3
    vol = np.zeros(g.size); src = rng.choice(np.flatnonzero(band), 25, replace=False); vol[src] = rng.uniform(1e14, 1e15, 25)
    out = P.spread_to_band(vol, near, d, band, 150e3, 50e3, g.area)
    assert abs((out*g.area).sum()-vol.sum()) < 1e-9*vol.sum()
    assert (out >= 0).all() and not out[~band].any()
    got = out > 0
    assert np.median(d[got]) > 80e3                 # in the band around 150 km, not on the receiving nodes


# ---------------------------------------------------------------- fortieth pass: B-d2, trench fluxes from consumed area
def test_trench_flux_band_is_the_default():
    assert P.PARAMS['trench_flux'] == 'band'


def test_consumed_trench_flux_on_a_straight_trench_is_the_band_law():
    # a great-circle trench, the overriding side's boundary row as trench nodes, a uniform consumed strip: each term
    # adds H x the consumed area, and its profile is the 'band' law's with v = consumed area / (length x dt)
    g = CubeSphere(96)
    ax = np.array([.3, -.5, .81]); ax /= np.linalg.norm(ax)
    own = (g.xyz@ax > 0).astype(np.int64)
    side = np.array([(own[g.nbr[i][g.nbr[i] >= 0]] != own[i]).any() for i in range(g.size)])
    src = side & (own == 1)
    near, d = P.nearest_label(np.where(src, np.arange(g.size), -1).astype(np.int64), g.nbr, g.dist)
    ok = (near >= 0) & (d < 1000e3) & (own == 1)
    ok[ok] &= own[near[ok]] == own[ok]
    L = 2*np.pi*g.radius
    v, dt = .05, 1e6                                  # 5 cm/yr for 1 Myr
    strip = ok & (d < 60e3)
    A = np.where(strip, g.area, 0.)
    A *= v*dt*L/A.sum()                               # consumed area = v x dt x length
    A[(own == 0) & side] = 1e9                        # consumed on the other plate's side: no trench of that plate
    cont = np.zeros(g.size, bool)
    terms = ((ok & ~cont, 1000., 150e3, 50e3),)
    out, routed, unrouted = P.trench_flux_consumed(A, near, d, ok, terms, g.area, .5*np.sqrt(g.area.mean()), own,
                                                   g.nbr, g.dist)
    assert unrouted == 1e9*((own == 0) & side).sum()
    vol = (out*g.area).sum()
    assert abs(vol-1000.*routed) < 1e-9*vol           # a full band takes all of it
    shape = np.exp(-.5*((d-150e3)/50e3)**2)/(50e3*np.sqrt(2*np.pi))
    band = np.where(ok, v*1000.*shape*dt, 0.)
    assert abs((band*g.area).sum()/vol-1) < .05       # the band law on a straight line gives the same volume
    hi = band > .5*band.max()
    assert np.median(np.abs(out[hi]/band[hi]-1)) < .25


def test_consumed_trench_flux_keeps_the_ledgers():
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'continental_sediment': True, 'arc_H_ocean_m': 1000.,
                                        'arc_accretion': True, 'trench_flux': 'consumed', 'trench_polarity': 'pair'})
    v0, s0 = sim.continental_volume(), sim.continental_sediment_volume()
    for _ in range(60):
        sim.step()
    assert abs((sim.continental_volume()-v0)-sum(sim.ledger.values())) < 1e-9*v0
    assert abs((sim.continental_sediment_volume()-s0)-sum(sim.sed_ledger.values())) < 1e-9*v0
    E = sim.erosion_totals
    assert E['trench_flux_routed_m2'] > 0
    assert E['trench_flux_routed_m2'] > E['trench_flux_unrouted_m2']
    assert 0 < E['subduction_on_ocean_arcs_m3'] <= 1000.*E['trench_flux_routed_m2']


def test_pair_polarity_follows_the_overlap_resolution():
    # trench_polarity 'pair': at ocean-ocean subduction nodes the trench flag is the polarity the overlap resolution used
    # for that pair this step (the overriding plate's nodes are not trench nodes); 'node' stays the default
    assert P.PARAMS['trench_polarity'] == 'node'
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=3, params={'water_volume_km3': None, 'trench_polarity': 'pair'})
    checked = 0
    for _ in range(40):
        sim.step()
        now = getattr(sim, 'polarity_now', {})
        own, cont, bt, tr = sim.owner, sim.world['cont'], sim.btype, sim.trench
        for i in np.flatnonzero((bt == P.SUBDUCTION) & ~cont):
            js = [j for j in g.nbr[i] if j >= 0 and own[j] != own[i] and not cont[j]]
            if not js:
                continue
            key = (min(int(own[i]), int(own[js[0]])), max(int(own[i]), int(own[js[0]])))
            if key in now:
                assert tr[i] == (now[key] != own[i])
                checked += 1
    assert checked > 50
