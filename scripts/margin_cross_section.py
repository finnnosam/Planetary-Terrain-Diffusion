"""Two-sided cross-section of passive margins, Earth (CRUST1.0) against plate-layer runs (twenty-second pass).

  python -m scripts.margin_cross_section outputs\\planet-calibration\\sweep-20261024-173200\\rift_rule-pure_shear

The edge-thickness band (24-29 km, 0-50 km landward of the edge) depends on where the continent-ocean label puts the
edge. CRUST1.0's label is a crust type per 1 degree cell; the planet's is its continental flag, set at breakup
(10 km). If Earth labels transitional crust oceanic, or labels the cell holding the continent-ocean boundary as a
single 'slope / transition' value, the two edges sit at different points of the same profile. This script measures
the profile on both sides of the edge, and label-free quantities:

  profile        median total thickness (crust + sediment, ocean side too) in distance bins from the passive edge,
                 negative = ocean side (oceanic nodes far from plate boundaries, distance to the nearest passive edge)
  transitional   area of crust of any label with total thickness in 10-20 / 20-30 km within 500 km of a passive
                 edge, per km of passive edge (km^2/km)
  instrument     planet edge cells read with oceanic nodes included in the 1 degree mean (the margin_width reading)
                 and with continental nodes only
Earth: CRUST1.0 crsthk (crust incl. sediment, no water) and type per cell; |lat| > 60 excluded on both sides of the
comparison is not done (ice); Antarctic and Greenland margins are few on passive-edge counts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import run_params, sediment_bulk_m
from scripts.margin_width import (BOUNDARY_KM, C1, OCEANIC, crust1_cell, earth_boundaries)

ROOT = Path(__file__).resolve().parents[1]
BINS = ((-300, -200), (-200, -150), (-150, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 150), (150, 200),
        (200, 300))


def wmed(v, w, q=.5):
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return float(v[o][np.searchsorted(c, q)])


def earth(g):
    cell = crust1_cell(g)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2]*1e3
    S = np.loadtxt(C1/'sedthk.xyz')[:, 2]*1e3
    types = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()]).ravel()
    return T[cell], S[cell], ~np.isin(types[cell], list(OCEANIC)), types[cell]


def section(g, T, cont, boundary, types=None):
    nb = g.nbr; ok = nb >= 0; a = g.area
    ocean_nb = np.any(np.where(ok, ~cont[np.where(ok, nb, 0)], False), axis=1)
    far = gl.distance_to(boundary, nb, g.dist, 5000e3) > BOUNDARY_KM*1e3
    passive = cont & ocean_nb & far
    # ocean side: oceanic nodes whose nearest continental edge is passive (distance to passive edge < distance
    # to any non-passive continental edge), far from plate boundaries
    dpass = gl.distance_to(passive, nb, g.dist, 1000e3)/1e3
    dedge = gl.distance_to(cont & ocean_nb & ~passive, nb, g.dist, 1000e3)/1e3
    spacing = float(np.sqrt(a.mean()))/1e3
    L = passive.sum()*spacing
    out = {'passive_edge_km': float(L), 'profile': {}}
    for lo, hi in BINS:
        if lo < 0:
            m = ~cont & far & (dpass < dedge) & (dpass > -hi) & (dpass <= -lo)
        else:
            m = cont & far & (dpass >= lo) & (dpass < hi)
        if m.sum() > 5:
            row = {'p25_50_75': [wmed(T[m], a[m], q)/1e3 for q in (.25, .5, .75)], 'n': int(m.sum())}
            if types is not None:
                u, c = np.unique(types[m], return_counts=True)
                o = np.argsort(-c)[:4]
                row['types'] = {str(u[i]): round(float(c[i]/c.sum()), 2) for i in o}
            out['profile'][f'{lo}..{hi}'] = row
    near = (dpass <= 500) & far & ((cont) | (dpass < dedge))
    for lo, hi in ((10e3, 20e3), (20e3, 30e3)):
        m = near & (T >= lo) & (T < hi)
        out[f'T{lo/1e3:.0f}-{hi/1e3:.0f}_per_km'] = float(a[m].sum()/1e6/L)
        out[f'T{lo/1e3:.0f}-{hi/1e3:.0f}_oceanic_label_share'] = float(a[m & ~cont].sum()/max(a[m].sum(), 1.))
    # label-free alignment: signed distance from the 20 km contour (negative where T < 20 km), near passive edges
    hi20 = T >= 20e3
    sd = np.where(hi20, gl.distance_to(~hi20, nb, g.dist, 1000e3), -gl.distance_to(hi20, nb, g.dist, 1000e3))/1e3
    zone = near & (np.abs(sd) < 400)
    out['contour20'] = {}
    for lo, hi in ((-200, -150), (-150, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 150), (150, 200),
                   (200, 300)):
        m = zone & (sd >= lo) & (sd < hi)
        if m.sum() > 5:
            out['contour20'][f'{lo}..{hi}'] = [wmed(T[m], a[m], q)/1e3 for q in (.25, .5, .75)]
    # distance from the passive edge (continent side) to where T first reaches 15 / 20 / 25 km
    for lim in (15e3, 20e3, 25e3):
        d = gl.distance_to(cont & (T >= lim), nb, g.dist, 3000e3)[passive]/1e3
        out[f'edge_to_{lim/1e3:.0f}km_p50'] = float(np.median(d))
    return out


def planet_1deg(g, T, cont, cont_only):
    cell = crust1_cell(g); a = g.area
    w = a*(cont if cont_only else 1.)
    A = np.bincount(cell, weights=a, minlength=64800); W = np.bincount(cell, weights=w, minlength=64800)
    TA = np.bincount(cell, weights=w*T, minlength=64800)
    CA = np.bincount(cell, weights=a*cont, minlength=64800)
    c1 = (CA/np.maximum(A, 1e-300)) >= .5
    Tm = TA/np.maximum(W, 1e-300)
    TO = np.bincount(cell, weights=a*T, minlength=64800)/np.maximum(A, 1e-300)     # ocean cells: all nodes
    return np.where(c1, Tm, TO)[cell], c1[cell]


def fmt(name, r):
    s = f"{name}: passive edge {r['passive_edge_km']:,.0f} km\n    profile (km, ocean < 0; T p50 km):"
    s += ''.join(f"\n      {k:>11}: {v['p25_50_75'][1]:5.1f}  ({v['p25_50_75'][0]:.1f}-{v['p25_50_75'][2]:.1f}, n {v['n']})"
                 + (f"  {v['types']}" if 'types' in v else '') for k, v in r['profile'].items())
    s += (f"\n    crust 10-20 km per km of edge {r['T10-20_per_km']:.0f} (oceanic label {r['T10-20_oceanic_label_share']:.0%}),"
          f" 20-30 km {r['T20-30_per_km']:.0f} (oceanic label {r['T20-30_oceanic_label_share']:.0%});"
          f" edge to 15 / 20 / 25 km crust p50 {r['edge_to_15km_p50']:.0f} / {r['edge_to_20km_p50']:.0f} / "
          f"{r['edge_to_25km_p50']:.0f} km")
    s += '\n    from the 20 km contour (km: T p50): ' + ', '.join(f"{k} {v[1]:.1f}" for k, v in r['contour20'].items())
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    a = ap.parse_args(argv)
    g = CubeSphere(217)
    T, S, cont, types = earth(g)
    bnd = earth_boundaries(g)
    res = {'earth_total': section(g, T, cont, bnd, types), 'earth_crystalline': section(g, T-S, cont, bnd)}
    print(fmt('Earth, CRUST1.0 total (crust + sediment)', res['earth_total']))
    print(fmt('Earth, crystalline only', res['earth_crystalline']))
    seeds = []
    for d in map(Path, a.dirs):
        seeds += [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
    for s in seeds:
        z = np.load(s/'snapshots'/'t+0000.0.npz')
        c = z['continental'].astype(bool)
        Tt = z['crust_thickness_m'].astype(float)+sediment_bulk_m(z)                   # sediment on both sides (bulk)
        Tc = z['crust_thickness_m'].astype(float)
        b = z['boundary_type'] != 0
        r = {'native_total': section(g, Tt, c, b), 'native_crystalline': section(g, Tc, c, b),
             '1deg_all_nodes': section(g, *planet_1deg(g, Tt, c, False), b),
             '1deg_continental_nodes': section(g, *planet_1deg(g, Tt, c, True), b)}
        res[str(s)] = r
        for k, v in r.items():
            print(fmt(f'{s.parent.name}/{s.name} {k}', v))
    out = ROOT/'outputs'/'planet-reference'/'margin-cross-section.json'
    out.write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    main()
