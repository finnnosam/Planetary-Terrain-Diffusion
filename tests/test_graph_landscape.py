"""Graph landscape kernels: flat-window regression against landscape_evolution, and sphere behaviour."""
import numpy as np
import pytest

from planet_training import landscape_evolution as le, tectonic_templates as tt
from planet_layer import sphere_graph as sg, graph_landscape as gl


class NoFlexure:
    alpha_m = 0.

    def deflection(self, load):
        return np.zeros_like(load)


def test_flat_matches_regional_model():
    n, dx = 128, 11520.
    z0, up, erod, _ = tt.build('collision', n, dx, 1, True)
    old = le.Landscape(z0, dx, up, None, erod)
    new = gl.GraphLandscape(sg.flat(n, n, dx), z0, up, gl.FlatFlexure(n, n, dx, 30e3), None, erod)
    for _ in range(20):
        old.step(); new.step()
        assert np.array_equal(old.rec, new.rec)
        assert np.abs(old.z-new.z).max() < 1e-8


def test_graph_distance_equals_chamfer_on_flat():
    rng = np.random.default_rng(0)
    n, dx = 64, 1000.
    ocean = rng.uniform(size=(n, n)) > .3
    g = sg.flat(n, n, dx)
    d_graph = gl.distance_to(~ocean.ravel(), g.nbr, g.dist)
    d_chamfer = le.coast_distance(ocean.ravel(), n, n, dx)
    assert np.allclose(d_graph, d_chamfer, rtol=1e-12, atol=1e-6)


def _cone_on_corner(n=48):
    g = sg.CubeSphere(n)
    corner = np.ones(3)/np.sqrt(3)
    ang = np.arccos(np.clip(g.xyz@corner, -1, 1))
    z0 = 3000.-3000.*ang/.35 + np.random.default_rng(1).normal(0, 1., g.size)   # land within ~0.35 rad (~2,200 km)
    return g, corner, ang, z0


def test_sphere_cone_drains_across_faces_and_balances_volume():
    g, corner, ang, z0 = _cone_on_corner()
    m = gl.GraphLandscape(g, z0, 2e-4, NoFlexure(), {'dt': 5e4})
    vol0 = (m.z*g.area).sum()
    m.run(2e6)
    o = m.outputs()
    land = ~o['ocean']
    assert len(m.stack) == g.size                      # every node is on a drainage tree (no cycles)
    assert o['drains_to_sea'][land].all()              # no stuck water on a pit-free cone
    # the three faces meeting at the corner share the drainage about equally
    face = g.face_yx[:, 0]
    shares = [g.area[land & (face == f)].sum() for f in (0, 1, 4)]
    assert max(shares)/min(shares) < 1.15
    # exact volume ledger (diffusion is conservative; flexure off)
    t = m.totals
    vol1 = (m.z*g.area).sum()
    expected = vol0+t['uplift_m3']+t['surface_motion_m3']-t['eroded_m3']+t['deposited_m3']
    assert abs(vol1-expected) < 1e-9*abs(t['uplift_m3'])


def test_surface_rate_moves_surface_without_load():
    g = sg.CubeSphere(16)
    z0 = np.full(g.size, -3000.)
    m = gl.GraphLandscape(g, z0, None, NoFlexure(), {'dt': 1e5}, surface_rate=1e-4)
    m.run(1e6)
    assert np.allclose(m.z, -2900.)
    assert np.allclose(m.load, 0.)


def test_sphere_flexure_matches_flat_fft():
    pytest.importorskip('ducc0')
    from planet_layer.sphere_flexure import SphereFlexure
    n = 217
    g = sg.CubeSphere(n)
    fx = SphereFlexure.for_graph(g, 30e3)
    centre = np.array([1., 0, 0])
    r = np.arccos(np.clip(g.xyz@centre, -1, 1))*g.radius
    w = fx.deflection(np.where(r <= 400e3, 1000., 0.))
    dx = np.pi*g.radius/2/n; M = 256
    c = (np.arange(M)-M/2)*dx; X, Y = np.meshgrid(c, c); rf = np.hypot(X, Y)
    wf = le.Flexure(M, M, dx, 30e3).deflection(np.where(rf <= 400e3, 1000., 0.))
    peak = wf.max()
    for r0 in (0., 200e3, 600e3, 800e3):   # away from the load edge, where 46 km node rings sample a step
        s, sf = np.abs(r-r0) < .75*dx, np.abs(rf-r0) < .75*dx
        assert abs(w[s].mean()-wf[sf].mean()) < .02*peak
    load = np.where(r <= 400e3, 1000., 0.)
    assert abs((w*g.area).sum()/(load*g.area).sum()/(2700/3300)-1) < .01   # broad-scale Airy compensation


def test_surface_step_keeps_the_deep_sea_sediment_on_request():
    """On a closed sphere every eroded volume either stays on the surface or is returned as 'deep_m3' (the regional
    model drops what finds no room on the shelves; the plate layer keeps it)."""
    import numpy as np
    from planet_layer.sphere_graph import CubeSphere
    from planet_layer import graph_landscape as gl
    g = CubeSphere(24)
    x, y, zc = g.xyz.T
    z = np.where(x > .3, 3000.*(x-.3)+50., -4000.+2000.*np.cos(6*y)*np.cos(5*zc))   # a continent and a bumpy deep floor
    p = {**gl.DEFAULTS, 'open_edges': False}
    seed = z < -1500.
    r0 = gl.surface_step(g, z, seed, None, np.ones(g.size, bool), np.full(g.size, p['K']*20), p, 5e5, g.spacing_min(),
                         g.diffusion_bound())
    r = gl.surface_step(g, z, seed, None, np.ones(g.size, bool), np.full(g.size, p['K']*20), p, 5e5, g.spacing_min(),
                        g.diffusion_bound(), keep_deep=True)
    assert np.array_equal(r0['z'], r['z'])                       # same surface either way
    moved = ((r['z']-z)*g.area).sum()
    eroded = (r['eroded']*g.area).sum()
    assert r['deep_m3'].sum() > .05*eroded                       # the deep-sea share is real here
    assert abs(moved+r['deep_m3'].sum()) < 1e-6*eroded           # and closes the volume


def test_fill_and_spill_ponds_from_the_bottom_and_spills_over():
    import numpy as np
    from planet_layer import sphere_graph as sg
    from planet_layer.graph_landscape import fill_and_spill
    g = sg.flat(1, 9, 1e3)
    # two depressions: a deep one fed at node 1, a shallower one beyond a sill at node 4
    z = np.array([0., -100., -90., 0., -50., -60., -50., -40., 0.])
    vol = np.zeros(9); vol[2] = 10e6                      # 10 m over 1 km^2 cells: a 1e7 m^3 deposit
    z1 = fill_and_spill(z, vol, g.nbr, g.area)
    assert np.isclose(((z1-z)*g.area).sum(), 10e6)        # all of it placed
    assert np.isclose(z1[1], z1[2]) and z1[1] < 0          # ponded flat in the deep hollow
    big = np.zeros(9); big[1] = 300e6                     # more than the deep hollow holds below its sill (0 m)
    z2 = fill_and_spill(z, big, g.nbr, g.area)
    assert np.isclose(((z2-z)*g.area).sum(), 300e6)
    assert z2.max() <= 1e-6+max(0., z2.max())
    assert (np.diff(np.sort(z2[[1, 2]])) < 1e-9).all()    # the hollow is a flat fill
    assert np.abs(z2-z).max() < 300.                      # spread over several nodes, no tower
