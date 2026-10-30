"""Continental crust of plate-layer runs against Earth's (CRUST1.0), plan doc, seventeenth pass.

  python -m scripts.crust_review outputs\\planet-calibration\\sweep-20261023-214324
  python -m scripts.crust_review --latest

For every combination folder (or seed folder) given: continental area and the area-weighted distribution of
continental crust thickness at present day, 3-seed means, against data/crust1/crust1-continental-summary.json
(CRUST1.0, Laske et al. 2013; continental = crust types 01-25, shelf, slope, thinned and transitional continental
crust; total thickness including sediment, as the plate layer counts it).

Bands, fixed 23 Oct before any mechanism change (Earth in brackets):
  continental area 0.35-0.42 (0.383); share thinner than 30 km 0.12-0.26 (0.19); share thicker than 45 km <= 0.13
  (0.087); median 34-40 km (37.0).

Which crust is continental (twenty-second pass, agreed 25 Oct): label-free, total thickness (crust + sediment) >=
CONTINENT_MIN_TOTAL_M = 20 km, on both sides. On Earth this rule reproduces CRUST1.0's own crust types (area 0.384 vs
0.383, < 30 km 0.189 vs 0.189, median 37.0 vs 37.0); the planet's continental flag is set at breakup (10 km) and
counts 10-20 km transitional crust that CRUST1.0 calls oceanic. The bands are judged on the label-free reading; the
flag reading (the reviews before 25 Oct) is printed beside it. Planet columns: crust + sediment on every node (ocean
side too, as CRUST1.0's crsthk), with the continental sediment rule of total_thickness_m. Earth: CRUST1.0 cells
(cos(lat) weights) with crsthk >= 20 km, computed here; the type reading comes from the summary file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer.sphere_graph import CubeSphere

ROOT = Path(__file__).resolve().parents[1]
EARTH = ROOT/'data'/'crust1'/'crust1-continental-summary.json'
BANDS = {'area': (.35, .42), 'thin_lt30': (.12, .26), 'thick_gt45': (0., .13), 'median_km': (34., 40.)}
CONTINENT_MIN_TOTAL_M = 20e3      # label-free continent: crust + sediment >= 20 km (twenty-second pass)


def wpct(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return float(v[o][np.searchsorted(c, q/100)])


# defaults that changed after runs that saved only their overrides: such a run ran with these (a review reading it
# with today's defaults would read the wrong law). sediment_compaction: default True since the thirty-seventh pass
LEGACY_DEFAULTS = {'sediment_compaction': False, 'piece_rule': 'shared'}


def run_params(seed_dir):
    """The run's parameters: its stats.json's params_effective (every parameter, saved since the thirty-seventh
    pass), else plates.PARAMS with LEGACY_DEFAULTS, updated by the overrides in its stats.json."""
    from planet_layer.plates import PARAMS
    f = Path(seed_dir)/'stats.json'
    st = json.loads(f.read_text()) if f.exists() else {}
    if 'params_effective' in st:
        return {**PARAMS, **st['params_effective']}
    return {**PARAMS, **LEGACY_DEFAULTS, **st.get('params', {})}


def total_thickness_m(z, params):
    """Thickness as CRUST1.0 counts it (crust + sediment): with continental_sediment the sediment on continental
    crust is its own layer (twentieth pass); before, deposits were booked as crust and the sediment field on
    continental nodes (ocean sediment carried onto arcs, <~100 m) was not part of the column, so it is left out."""
    T = z['crust_thickness_m'].astype(np.float64)
    if params.get('continental_sediment'):
        T = T+np.where(z['continental'].astype(bool), sediment_bulk_m(z), 0.)
    return T


def sediment_bulk_m(z):
    """Sediment thickness as CRUST1.0 and GlobSed count it (bulk, porous): the snapshot's sediment_bulk_m (written
    since the thirty-sixth pass; it differs from sediment_m only with sediment_compaction), else sediment_m."""
    return z['sediment_bulk_m'].astype(np.float64) if 'sediment_bulk_m' in getattr(z, 'files', z) else \
        z['sediment_m'].astype(np.float64)


def column_thickness_m(z, params):
    """Crust + sediment on every node, ocean side included (as CRUST1.0's crsthk); continental nodes as
    total_thickness_m (their sediment counts only with continental_sediment)."""
    c = z['continental'].astype(bool)
    return np.where(c, total_thickness_m(z, params),
                    z['crust_thickness_m'].astype(np.float64)+sediment_bulk_m(z))


def label_free_continent(T_total_m):
    """Continental crust read without labels: total thickness >= 20 km (reproduces CRUST1.0's types on Earth)."""
    return T_total_m >= CONTINENT_MIN_TOTAL_M


def distribution(T_m, c, a):
    """Area and area-weighted thickness distribution of the crust selected by c (T in m, a node or cell areas)."""
    T = T_m[c]/1e3; w = a[c]
    return {'area': float(w.sum()/a.sum()), 'thin_lt30': float(w[T < 30].sum()/w.sum()),
            'thick_gt45': float(w[T > 45].sum()/w.sum()), 'thick_gt55': float(w[T > 55].sum()/w.sum()),
            'median_km': wpct(T, w, 50), 'p10_km': wpct(T, w, 10), 'p90_km': wpct(T, w, 90)}


def earth_label_free():
    """Earth's continent read label-free: CRUST1.0 1 degree cells with crsthk >= 20 km, cos(lat) weights."""
    T = np.loadtxt(ROOT/'data'/'crust1'/'crsthk.xyz')[:, 2].reshape(180, 360)*1e3
    a = np.repeat(np.cos(np.radians(89.5-np.arange(180)))[:, None], 360, 1)
    T, a = T.ravel(), a.ravel()
    return distribution(T, label_free_continent(T), a)


def metrics(snapshot, g):
    """Label-free continental crust (the banded reading) and, as flag_*, the planet's continental flag reading."""
    z = np.load(snapshot)
    p = run_params(Path(snapshot).parent.parent)
    flag = z['continental'].astype(bool)
    Tcol = column_thickness_m(z, p)
    out = distribution(Tcol, label_free_continent(Tcol), g.area)
    out.update({'flag_'+k: v for k, v in distribution(total_thickness_m(z, p), flag, g.area).items()})
    return {**out, **sediment_metrics(z, p, g)}


# Earth's sediment on continental crust (CRUST1.0 sediment layers on the n = 217 graph, continental types; computed
# 24 Oct, twentieth pass). A read-out, not a band: CRUST1.0's cover on land includes Palaeozoic and older basins,
# while the planet starts 250 Myr ago with none; the margin sediment (thin and submerged crust, on Earth mostly
# younger than ~200 Myr) is the comparable part
EARTH_SEDIMENT_M = {'mean': 1569., 'on_thin_lt30': 2394., 'on_submerged': 2398., 'on_land': 1250.}


def sediment_metrics(z, p, g):
    """Area-weighted mean sediment on continental crust (all, total thickness < 30 km, submerged, land), m."""
    c = z['continental'].astype(bool)
    if not p.get('continental_sediment'):
        return {f'sed_{k}_m': 0. for k in EARTH_SEDIMENT_M}
    S = sediment_bulk_m(z); T = total_thickness_m(z, p); e = z['elevation_m']; a = g.area   # bulk, as CRUST1.0
    out = {}
    for k, m in (('mean', c), ('on_thin_lt30', c & (T < 30e3)), ('on_submerged', c & (e <= 0)), ('on_land', c & (e > 0))):
        out[f'sed_{k}_m'] = float(np.average(S[m], weights=a[m])) if m.any() else 0.
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='*', help='a sweep folder, combination folders, or seed folders')
    p.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    a = p.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
        print('sweep', a.dirs[0])
    g = CubeSphere(217)
    e = json.loads(EARTH.read_text(encoding='utf-8'))
    el = earth_label_free()
    print(f"Earth (CRUST1.0, total >= 20 km): area {el['area']:.3f}, <30 km {el['thin_lt30']:.3f}, >45 km {el['thick_gt45']:.3f}, "
          f">55 km {el['thick_gt55']:.3f}, median {el['median_km']:.1f} km\n"
          f"    by crust type: area {e['continental_area_fraction']:.3f}, <30 km {e['continental_share_thinner_than_30km']:.3f}, "
          f">45 km {e['continental_share_thicker_than_45km']:.3f}, >55 km {e['continental_share_thicker_than_55km']:.3f}, "
          f"median {e['continental_total_thickness_km_percentiles']['50']:.1f} km")
    combos = []
    for d in map(Path, a.dirs):
        if (d/'snapshots').is_dir():
            combos.append((d.parent.name, [d]))
        elif any(d.glob('seed*/snapshots')):
            combos.append((d.name, sorted(d.glob('seed*'))))
        else:
            combos += [(c.name, sorted(c.glob('seed*'))) for c in sorted(d.iterdir()) if c.is_dir() and any(c.glob('seed*/snapshots'))]
    out = {}
    E = EARTH_SEDIMENT_M
    for name, seeds in combos:
        ms = [metrics(s/'snapshots'/'t+0000.0.npz', g) for s in seeds]
        m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        m['passes'] = all(lo <= m[k] <= hi for k, (lo, hi) in BANDS.items())
        out[name] = m
        print(f"{name} (total >= 20 km): area {m['area']:.3f}, <30 km {m['thin_lt30']:.3f}, >45 km {m['thick_gt45']:.3f}, "
              f">55 km {m['thick_gt55']:.3f}, p10 / median / p90 {m['p10_km']:.1f} / {m['median_km']:.1f} / {m['p90_km']:.1f} km "
              f"({len(ms)} seeds): {'passes' if m['passes'] else 'fails'}\n"
              f"    flag: area {m['flag_area']:.3f}, <30 km {m['flag_thin_lt30']:.3f}, >45 km {m['flag_thick_gt45']:.3f}, "
              f">55 km {m['flag_thick_gt55']:.3f}, median {m['flag_median_km']:.1f} km"
              f"\n    sediment on continental crust (flag), mean / on < 30 km / submerged / land "
              f"{m['sed_mean_m']:.0f} / {m['sed_on_thin_lt30_m']:.0f} / {m['sed_on_submerged_m']:.0f} / {m['sed_on_land_m']:.0f} m "
              f"(Earth {E['mean']:.0f} / {E['on_thin_lt30']:.0f} / {E['on_submerged']:.0f} / {E['on_land']:.0f})")
    return out


if __name__ == '__main__':
    main()
