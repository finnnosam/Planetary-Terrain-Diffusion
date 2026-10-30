"""planet_layer.shelf: wave planation, shelf-edge progradation and damped coastal retreat on a flat strip."""
import numpy as np

from planet_layer import sphere_graph as sg
from planet_layer.shelf import shelf_step

DX = 46e3
BIG = 1e9


def strip(z_cols, ny=3):
    g = sg.flat(ny, len(z_cols), DX)
    z = np.tile(np.asarray(z_cols, float), ny)
    return g, z


def test_conserves_volume_and_never_erodes_ocean_basement():
    g, z = strip([300, 200, -40, -20, -80, -60, -900, -3000, -3000])
    ocean = z < 0
    avail = np.where(ocean, 0., BIG)                     # bare ocean floor, continental land
    avail[(z > -100) & ocean] = 30.                      # a little sediment on the shallow floor
    z1, a1, off = shelf_step(g, z, ocean, avail, 3e4, 50., .01, 1e6)
    assert abs((z1-z).sum()) < 1e-6*np.abs(z).sum()      # every exchange is a flux between two nodes
    assert (z1-z)[ocean].min() >= -30.-1e-6              # ocean floor loses at most its sediment
    assert (a1 >= -1e-9).all() and off > 0


def test_planes_shallow_highs_but_not_deep_ones():
    g, z = strip([-60]*4+[-10]+[-60]*4+[-650]*4+[-600]+[-650]*4)
    ocean = np.ones(g.size, bool)
    avail = np.full(g.size, 1e3)
    z1, _, _ = shelf_step(g, z, ocean, avail, 3e4, 50., 0., 1e6, max_depth=1e4)
    zc = z.reshape(3, -1)[1]; z1c = z1.reshape(3, -1)[1]
    assert z1c[4]-zc[4] < -10.                           # the shoal at -10 m is planed down
    assert abs(z1c[13]-zc[13]) < 1.                      # the same bump at -600 m (12 wave bases) barely moves


def test_shelf_edge_progrades_over_the_slope():
    # sediment arriving at the coast (a delta at 0 m) spreads into a flat topset and stops where the water deepens
    cols = [-5.]+[-80.]*3+[-2000.]*8
    g, z = strip(cols)
    ocean = np.ones(g.size, bool)
    z[::len(cols)] = 400.                                # a pile at the coast, all sediment
    avail = np.full(g.size, 0.); avail[::len(cols)] = 405.
    z1, _, _ = shelf_step(g, z, ocean, avail, 3e4, 50., 0., 2e6)
    zc = z1.reshape(3, -1)[1]
    assert zc[1:4].max() > -80.+1                        # topset filled
    assert zc[5:].max() < -2000.+5                       # nothing reaches the deep slope in 2 Myr
    assert zc[4]-(-2000.) > 10*(zc[5]-(-2000.))+1e-9     # deposition is concentrated at the shelf edge


def test_coastal_retreat_is_damped_by_a_shallow_shelf():
    g1, z1 = strip([300, -3000, -3000])                  # cliff facing deep water
    g2, z2 = strip([300, -20, -3000])                    # cliff behind a shallow shelf
    res = []
    for g, z in ((g1, z1), (g2, z2)):
        ocean = z < 0
        avail = np.where(ocean, 0., BIG)
        zn, _, off = shelf_step(g, z, ocean, avail, 3e4, 50., .01, 1e6)
        res.append(off)
    assert res[0] > 0 and res[1] < .5*res[0]
    # undamped, the volume is R x face per unit coast length: 0.01 m/yr x (300 + 50 m) x 1 Myr x 3 rows x 46 km,
    # less the lowering of the cliff top during the step
    assert np.isclose(res[0], .01*350*1e6*3*DX, rtol=.25)
