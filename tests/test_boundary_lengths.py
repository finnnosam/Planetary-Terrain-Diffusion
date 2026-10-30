import numpy as np

from planet_layer.sphere_graph import CubeSphere
from scripts.boundary_lengths import lengths


def test_great_circle_length():
    """A plate split along a great circle measures 2 pi R within 2% at any orientation (node counting: up to +38%)."""
    g = CubeSphere(64)
    rng = np.random.default_rng(3)
    for _ in range(4):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        pid = (g.xyz@ax > 0).astype(np.int32)
        L = lengths(g, pid)['total_km']
        assert abs(L/(2*np.pi*g.radius/1e3)-1) < .02, L
