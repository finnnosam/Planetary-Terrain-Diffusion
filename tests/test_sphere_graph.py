import numpy as np

from planet_diffusion import cube
from planet_layer import sphere_graph as sg


def test_cube_sphere_geometry():
    for n in (4, 33):
        g = sg.CubeSphere(n)
        assert g.size == 6*n*n+2
        assert abs(g.area.sum()/(4*np.pi*g.radius**2)-1) < 1e-12
        count = (g.nbr >= 0).sum(1)
        assert set(np.unique(count)) == {6, 8} and (count == 6).sum() == 8
        assert ((g.axis & (g.nbr >= 0)).sum(1) >= 3).all()
        i = np.repeat(np.arange(g.size), 8); j = g.nbr.ravel(); ok = j >= 0
        pairs = set(zip(i[ok].tolist(), j[ok].tolist()))
        assert all((b, a) in pairs for a, b in pairs)
        d = g.dist[g.axis & (g.nbr >= 0)]/(np.pi*g.radius/2/n)
        assert d.min() > .7 and d.max() < 1+1e-9


def test_cube_round_trip():
    n = 12
    g = sg.CubeSphere(n)
    a = cube.identify(np.random.default_rng(0).normal(size=(2, 6, n+1, n+1)).astype(np.float32))
    v = g.to_nodes(a)
    assert v.shape == (2, g.size)
    assert np.array_equal(g.to_cube(v), a)
    p = np.stack([cube.directions(f, *np.meshgrid(np.arange(n+1), np.arange(n+1), indexing='ij'), n) for f in range(6)])
    assert np.allclose(g.to_cube(g.xyz.T).transpose(1, 2, 3, 0), p, atol=1e-12)


def test_flat_graph_matches_offsets():
    g = sg.flat(5, 7, 10.)
    i = 2*7+3
    assert list(g.nbr[i]) == [(2+dy)*7+3+dx for dy, dx in sg.OFFSETS]
    assert g.edge.sum() == 2*7+2*5-4
