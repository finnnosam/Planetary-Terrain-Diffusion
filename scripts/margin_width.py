"""Width of passive continental margins, Earth (CRUST1.0) against plate-layer runs, on the same n = 217 graph
(nineteenth pass: the test for a rift rule stated in distance).

  python -m scripts.margin_width                                  Earth only
  python -m scripts.margin_width outputs\\planet-calibration\\sweep-20261024-114002

Passive margin edge: a continental node with an oceanic neighbour, farther than BOUNDARY_KM from any plate boundary
(Earth: Bird 2003, PB2002 boundaries; planet: boundary_type != 0). Measured there:
  width_30 / width_35   graph distance from the edge to the nearest continental crust >= 30 / 35 km (p25/50/75)
  thin_per_km           area of continental crust < 30 km within MARGIN_KM of a passive edge, per km of passive edge
                        (edge length = edge nodes x node spacing), km^2/km: an equivalent width
Earth: CRUST1.0 total thickness and type at 1 degree, each node taking the cell it lies in, so Earth widths are
resolved to ~50-110 km; read the medians, not single margins.
Planet runs are measured twice (twenty-first pass): at their own 46 km resolution, and through the same 1 degree
instrument as CRUST1.0 (`to_crust1_cells`: every node takes the area-weighted mean thickness of the planet nodes in its
1 degree cell; the cell is continental when most of its area is). The bands are judged on the 1 degree reading: a
margin with Earth's real profile (crust < 10 km at the continent-ocean boundary, seismic sections) reads ~26 km at
the edge in CRUST1.0, so a planet margin read at 46 km is not comparable with the edge-thickness band.
Which crust is continental (twenty-second pass, agreed 25 Oct): label-free on both sides, total thickness (crust +
sediment) >= 20 km (crust_review.CONTINENT_MIN_TOTAL_M). Earth: CRUST1.0 cells with crsthk >= 20 km (this reproduces
its crust types: thin per km 155 vs 140, edge T 25.2 vs 26.4); planet: every node's crust + sediment (ocean side
too), averaged into the 1 degree cells, continental where the cell mean is >= 20 km. The bands are judged on that;
Earth by crust type, the planet by its continental flag (the reading before 25 Oct) and the 46 km label-free reading
are printed beside it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import fields, graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import column_thickness_m, label_free_continent, run_params, total_thickness_m

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT/'data'/'crust1'
OCEANIC = {'A1', 'A0', 'B-', 'V1', 'Y3'}
BOUNDARY_KM = 300.
MARGIN_KM = 500.
# bands, fixed 24 Oct before any rift-rule or crust-flow change (Earth, CRUST1.0 on this grid, in brackets)
BANDS = {'thin_per_km': (100., 200.),              # km^2 of crust < 30 km per km of passive edge (140)
         'abrupt_edge_share': (0., .30),           # passive edges with no crust < 30 km (0.17)
         'edge_T_km': (24., 29.),                  # median thickness 0-50 km from the edge (26.4)
         'inner_T_km': (30., 34.)}                 # median thickness 100-150 km from the edge (32.0)
PROFILE_KM = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1000))


def earth_nodes(g):
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2].reshape(180, 360)*1e3
    types = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()])
    lat = np.degrees(np.arcsin(np.clip(g.xyz[:, 2], -1, 1))); lon = np.degrees(np.arctan2(g.xyz[:, 1], g.xyz[:, 0]))
    r = np.clip((89.5-lat+.5).astype(int), 0, 179); c = np.clip((lon+180).astype(int), 0, 359)
    return T[r, c], ~np.isin(types[r, c], list(OCEANIC))


def crust1_cell(g):
    """Index (0..64799) of the CRUST1.0 1 degree cell every node lies in (the same mapping as earth_nodes)."""
    lat = np.degrees(np.arcsin(np.clip(g.xyz[:, 2], -1, 1))); lon = np.degrees(np.arctan2(g.xyz[:, 1], g.xyz[:, 0]))
    r = np.clip((89.5-lat+.5).astype(int), 0, 179); c = np.clip((lon+180).astype(int), 0, 359)
    return r*360+c


def to_crust1_cells(g, T, cont):
    """The planet seen through CRUST1.0's 1 degree cells: area-weighted mean thickness of the nodes in each cell, and
    continental where continental nodes cover at least half the cell's node area."""
    cell = crust1_cell(g); a = g.area
    A = np.bincount(cell, weights=a, minlength=64800)
    TA = np.bincount(cell, weights=a*T, minlength=64800)
    CA = np.bincount(cell, weights=a*cont, minlength=64800)
    Am = np.maximum(A, 1e-300)
    return (TA/Am)[cell], (CA/Am)[cell] >= .5


def earth_boundaries(g, step_km=10.):
    js = json.loads((ROOT/'data'/'tectonicplates-master'/'GeoJSON'/'PB2002_boundaries.json').read_text())
    pts = []
    for f in js['features']:
        geo = f['geometry']
        lines = geo['coordinates'] if geo['type'] == 'MultiLineString' else [geo['coordinates']]
        for ln in lines:
            ll = np.radians(np.array(ln, float))
            xyz = np.stack((np.cos(ll[:, 1])*np.cos(ll[:, 0]), np.cos(ll[:, 1])*np.sin(ll[:, 0]), np.sin(ll[:, 1])), -1)
            for a, b in zip(xyz[:-1], xyz[1:]):
                n = max(2, int(np.ceil(np.arccos(np.clip(a@b, -1, 1))*g.radius/1e3/step_km))+1)
                s = np.linspace(0, 1, n)[:, None]
                p = a*(1-s)+b*s
                pts.append(p/np.linalg.norm(p, axis=1, keepdims=True))
    m = np.zeros(g.size, bool)
    m[fields.nearest_nodes(g, np.concatenate(pts))] = True
    return m


def measure(g, T, cont, boundary):
    a = g.area; nb = g.nbr; ok = nb >= 0
    ocean_nb = np.any(np.where(ok, ~cont[np.where(ok, nb, 0)], False), axis=1)
    edge = cont & ocean_nb
    far = gl.distance_to(boundary, nb, g.dist, 5000e3) > BOUNDARY_KM*1e3
    passive = edge & far
    out = {'passive_edge_nodes': int(passive.sum()), 'active_or_near_boundary_edge_nodes': int((edge & ~far).sum())}
    for lim in (30e3, 35e3):
        d = gl.distance_to(cont & (T >= lim), nb, g.dist, 3000e3)[passive]/1e3
        out[f'width_{lim/1e3:.0f}_km_p25_50_75'] = [float(np.percentile(d, q)) for q in (25, 50, 75)]
        if lim == 30e3:
            out['abrupt_edge_share'] = float((d == 0).mean())          # passive edges with no crust < 30 km
    near = gl.distance_to(passive, nb, g.dist, MARGIN_KM*1e3) <= MARGIN_KM*1e3
    spacing = float(np.sqrt(a.mean()))
    thin = cont & (T < 30e3) & near
    out['thin_per_km'] = float(a[thin].sum()/(passive.sum()*spacing)/1e3)
    out['passive_edge_length_km'] = float(passive.sum()*spacing/1e3)
    # thickness profile inland from passive edges (continental nodes far from plate boundaries), area-weighted median
    dp = gl.distance_to(passive, nb, g.dist, 1000e3)/1e3
    prof = {}
    for lo, hi in PROFILE_KM:
        m = cont & far & (dp >= lo) & (dp < hi)
        if m.any():
            o = np.argsort(T[m]); cw = np.cumsum(a[m][o]); cw /= cw[-1]
            prof[f'{lo}-{hi}'] = [float(T[m][o][np.searchsorted(cw, q)]/1e3) for q in (.25, .5, .75)]
    out['profile_T_km_p25_50_75'] = prof
    out['edge_T_km'] = prof.get('0-50', [np.nan]*3)[1]
    out['inner_T_km'] = prof.get('100-150', [np.nan]*3)[1]
    out['passes'] = bool(all(lo <= out[k] <= hi for k, (lo, hi) in BANDS.items()))
    out['continental_area_frac'] = float(a[cont].sum()/a.sum())
    out['thin_share_of_continent'] = float(a[cont & (T < 30e3)].sum()/a[cont].sum())
    out['thin_share_near_passive_edges'] = float(a[thin].sum()/max(a[cont & (T < 30e3)].sum(), 1.))
    return out


def fmt(name, m):
    w30, w35 = m['width_30_km_p25_50_75'], m['width_35_km_p25_50_75']
    return (f"{name}: passive edge {m['passive_edge_length_km']:,.0f} km ({m['passive_edge_nodes']} nodes); width to 30 km "
            f"{w30[0]:.0f} / {w30[1]:.0f} / {w30[2]:.0f} km, to 35 km {w35[0]:.0f} / {w35[1]:.0f} / {w35[2]:.0f} km; thin crust "
            f"per km of edge {m['thin_per_km']:.0f} km; continent {m['continental_area_frac']:.3f}, < 30 km "
            f"{m['thin_share_of_continent']:.3f} ({m['thin_share_near_passive_edges']:.0%} of it near passive edges); abrupt edges "
            f"{m['abrupt_edge_share']:.0%}: {'passes' if m['passes'] else 'fails'}\n    profile (km from edge: T p50): "
            +', '.join(f"{k} {v[1]:.1f}" for k, v in m['profile_T_km_p25_50_75'].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
    g = CubeSphere(217)
    T, cont = earth_nodes(g)
    eb = earth_boundaries(g)
    res = {'earth': measure(g, T, label_free_continent(T), eb), 'earth_types': measure(g, T, cont, eb)}
    print(fmt('Earth (CRUST1.0, total >= 20 km)', res['earth']))
    print('    by crust type: ' + fmt('', res['earth_types']).split(': ', 1)[1])
    seeds = []
    for d in map(Path, a.dirs):
        seeds += [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
    combos = {}
    for s in seeds:
        z = np.load(s/'snapshots'/'t+0000.0.npz')
        p = run_params(s)
        T, cont, bnd = total_thickness_m(z, p), z['continental'].astype(bool), z['boundary_type'] != 0
        Tcol = column_thickness_m(z, p)
        T1 = to_crust1_cells(g, Tcol, cont)[0]
        m = measure(g, T1, label_free_continent(T1), bnd)          # the bands: label-free, 1 degree instrument
        m['native_46km'] = measure(g, Tcol, label_free_continent(Tcol), bnd)
        m['flag_1deg'] = measure(g, *to_crust1_cells(g, T, cont), bnd)
        res[str(s)] = m
        combos.setdefault(s.parent.name, []).append(m)
        print(fmt(f'{s.parent.name[-30:]}/{s.name} (total >= 20 km, 1 degree)', m))
        print('    at 46 km: ' + fmt('', m['native_46km']).split(': ', 1)[1])
        print('    flag (1 degree): ' + fmt('', m['flag_1deg']).split(': ', 1)[1])
    for name, ms in combos.items():
        mean = {k: float(np.mean([x[k] for x in ms])) for k in BANDS}
        fl = {k: float(np.mean([x['flag_1deg'][k] for x in ms])) for k in BANDS}
        ok = all(lo <= mean[k] <= hi for k, (lo, hi) in BANDS.items())
        print(f"{name} ({len(ms)} seeds, total >= 20 km, 1 degree): thin per km {mean['thin_per_km']:.0f}, abrupt "
              f"{mean['abrupt_edge_share']:.0%}, edge T {mean['edge_T_km']:.1f}, inner T {mean['inner_T_km']:.1f} km: "
              f"{'passes' if ok else 'fails'} | flag: {fl['thin_per_km']:.0f}, {fl['abrupt_edge_share']:.0%}, "
              f"{fl['edge_T_km']:.1f}, {fl['inner_T_km']:.1f} km")
    (ROOT/'outputs'/'planet-reference').mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs'/'planet-reference'/'earth-margin-width-n217.json').write_text(
        json.dumps({**res['earth'], 'continent_rule': 'total thickness >= 20 km', 'by_crust_type': res['earth_types']}, indent=1))
    return res


if __name__ == '__main__':
    main()
