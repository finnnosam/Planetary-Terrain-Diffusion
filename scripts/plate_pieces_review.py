"""B-b's tests (plates are connected; plan doc, thirty-sixth and thirty-seventh passes), read from saved runs.

  python -m scripts.plate_pieces_review --latest
  python -m scripts.plate_pieces_review outputs\\planet-calibration\\sweep-...\\<combination> [...]

For each combination folder (all its seeds; or seed folders), against Earth:
  T1 detached area (plate pieces other than each plate's largest, stats.json series every 1 Myr; lasting two samples)
  <= 1% of the surface
     at every step, on every seed;
  T2 intra-oceanic trench 12-25k km at 0 Myr (Bird 2003 + CRUST1.0: 20.8k; contour length, scripts.trench_origin);
  T3 continental-side trench 20-35k km at 0 Myr (Earth 30.7k);
  T4 ridge production 2.5-4.5 km^2/yr, mean of the snapshots -40, -35, ..., 0 Myr (Bird 3.37; Rowley 2002 3.4;
     Seton et al. 2020's age grid 2.9-3.0 over the last 2-40 Myr, scripts.earth_seafloor_age);
  T5 ridge length (contour) within Bird's 94.8k km +- 15% at 0 Myr.
T2-T5 as stated are read on the 3-seed mean at 0 Myr (T4 over the last 40 Myr). Because one snapshot swings by
10-20k km of trench and the start alone carries 33-38k km of intra-oceanic trench (plan doc, thirty-seventh pass),
the sweep decides on the window means over -100 ... 0 Myr (every 10 Myr): each in its band, or no farther from it than
the control's (--control names the control combination; default: the combination whose name has 'shared').
Reported beside: plates and plates >= 1 Mkm^2 (Bird: 52 / 20, which hold 98% of the
surface), share of the surface in plates >= 1 Mkm^2, the largest plate (Bird 104.6 Mkm^2), the shape ratio
(boundary length / cap half-perimeters, Earth 1.51), the intra-oceanic share of the ocean-ocean boundary (Earth 0.14),
consumption at intra-oceanic trenches (Earth 1.28 km^2/yr), the planet's ocean-floor mean age and age distribution
(Seton et al. 2020: mean 63.9 Myr), the piece-rule counters and run time.
The start (thirty-eighth pass, `start_plates`, `start_poles`), 3-seed means at -240 Myr ("start" under the Decision
lines; -250 Myr beside): S1 continental-side trench >= 20k km (the lower edge of T3's present-day band) and S2 its
share of all subduction >= 0.60 (Earth today 30.7k of 51.5k). One-sided, because Pangea's world is not today's: it was
ringed by subduction (Collins 2003; Cawood 2005: the Terra Australis orogen alone ~18,000 km) and held more of it,
~65.7k km at ~235 Ma against ~42.4k today by the same tomographic count (van der Meer et al. 2014, PNAS 111, 4380).
No measured split exists for the start, so the present-day bands of T2 / T3 are printed but do not decide there.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer.sphere_graph import CubeSphere
from scripts import trench_origin as to

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT/'outputs'/'planet-calibration'
BIRD_RIDGE_KM = 94.8e3
BIRD_PLATES_MKM2 = None                                  # read from earth-plate-speeds-nnr.json
LATE = tuple(float(t) for t in range(-40, 1, 5))
WINDOW = tuple(float(t) for t in range(-100, 1, 10))
START = (-250., -240.)
TESTS = {'T2 intra-oceanic trench (k km)': (12e3, 25e3), 'T3 continental-side trench (k km)': (20e3, 35e3),
         'T4 production (km^2/yr)': (2.5, 4.5), 'T5 ridge (k km)': (.85*BIRD_RIDGE_KM, 1.15*BIRD_RIDGE_KM)}
AGE_EDGES = (0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 1e9)


def bird_plates():
    f = ROOT/'outputs'/'planet-reference'/'earth-plate-speeds-nnr.json'
    return np.array(sorted((q['area_mkm2'] for q in json.loads(f.read_text())['plates']), reverse=True))


def combos(dirs, latest):
    if latest:
        sw = sorted(CAL.glob('sweep-*'))[-1]
        return sorted(d for d in sw.iterdir() if d.is_dir() and d.name != 'reviews')
    return [Path(d) for d in dirs]


def seeds_of(d):
    d = Path(d)
    return [d] if (d/'snapshots').exists() else sorted(d.glob('seed*'))


def seed_numbers(g, sd):
    st = json.loads((sd/'stats.json').read_text())
    ser = st['series']
    det = [r.get('detached_area_frac', float('nan')) for r in ser]
    have = lambda t: Path(str(to.snap_path(sd, t))+'.npz').exists()
    cache = {t: to.snapshot_metrics(g, to.snap_path(sd, t)) for t in sorted(set(LATE) | set(WINDOW) | set(START)) if have(t)}
    late = [cache[t] for t in LATE if t in cache]
    win = [cache[t] for t in WINDOW if t in cache]
    now = late[-1]
    wm = lambda k: float(np.mean([r[k] for r in win]))
    z = np.load(str(to.snap_path(sd, 0.))+'.npz')
    oa = z['ocean_age_myr'].astype(np.float64); ok = np.isfinite(oa)
    a = g.area[ok]
    hist = np.histogram(oa[ok], bins=AGE_EDGES, weights=a)[0]/a.sum()
    areas = np.array(ser[-1]['plate_areas_mkm2'])
    nan = float('nan')
    return {'start_io_km': cache[-240.]['io_km'] if -240. in cache else nan,
            'start_cont_sub_km': cache[-240.]['cont_sub_km'] if -240. in cache else nan,
            'start0_io_km': cache[-250.]['io_km'] if -250. in cache else nan,
            'start0_cont_sub_km': cache[-250.]['cont_sub_km'] if -250. in cache else nan,
            # T1 reads detachment that lasts: the smaller of two consecutive 1 Myr samples (B-b makes a piece a plate on
            # the step after it appears, so a cut in the last step alone reads as detached; thirty-ninth pass)
            'detached_max': float(np.nanmax(np.minimum(det[:-1], det[1:]))) if np.isfinite(det).sum() > 1 else float('nan'),
            'detached_max_one_sample': float(np.nanmax(det)) if np.isfinite(det).any() else float('nan'),
            'detached_now': det[-1], 'io_km': now['io_km'], 'cont_sub_km': now['cont_sub_km'], 'ridge_km': now['ridge_km'],
            'production': float(np.mean([r['production_km2_yr'] for r in late])),
            'w_io_km': wm('io_km'), 'w_cont_sub_km': wm('cont_sub_km'), 'w_production': wm('production_km2_yr'),
            'w_ridge_km': wm('ridge_km'), 'w_shape_ratio': wm('shape_ratio'), 'w_detached_mkm2': wm('detached_mkm2'),
            'consumption_io': float(np.mean([r['consumption_io_km2_yr'] for r in late])),
            'io_share_oo': now['io_share_of_ocean_ocean'], 'shape_ratio': now['shape_ratio'],
            'plates': len(areas), 'plates_1mkm2': int((areas >= 1).sum()), 'share_in_1mkm2': float(areas[areas >= 1].sum()/areas.sum()),
            'largest_mkm2': float(areas.max()), 'ocean_age_mean': float(np.average(oa[ok], weights=a)),
            'age_hist': [float(x) for x in hist], 'piece_stats': ser[-1].get('piece_stats', {}),
            'events': ser[-1].get('events', {}), 'seconds': st.get('seconds', float('nan')),
            'speed_area_weighted': float(np.average(ser[-1]['speeds_cm_yr'], weights=areas)) if len(ser[-1].get('speeds_cm_yr', [])) == len(areas) else float('nan')}


def j(rows, k, fmt='{:.2f}', sc=1.):
    return ' / '.join(fmt.format(r[k]/sc) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--control', help='control combination folder name (default: the one with "shared" in its name)')
    a = ap.parse_args(argv)
    g = CubeSphere(217)
    bp = bird_plates()
    ref = ROOT/'outputs'/'planet-reference'/'earth-seafloor-age-n217.json'
    ea = json.loads(ref.read_text()) if ref.exists() else None
    print(f'Earth: plates {bp.size}, >= 1 Mkm^2 {(bp >= 1).sum()} holding {bp[bp >= 1].sum()/bp.sum():.2f} of the surface, '
          f'largest {bp[0]:.1f} Mkm^2; intra-oceanic trench 20.8k km, continental side 30.7k, ridge (Bird) 94.8k, '
          f'production 3.37 km^2/yr (Bird), intra-oceanic consumption 1.28, io / ocean-ocean boundary 0.14, shape ratio 1.51')
    if ea:
        print(f"  ocean floor (Seton et al. 2020): mean age {ea['mean_age_myr']:.1f} Myr, production "
              f"{ea['production_km2_yr']['10']:.2f} (A 10) / {ea['production_km2_yr']['40']:.2f} (A 40) km^2/yr; area by age "
              + ' '.join(f'{v:.2f}' for v in ea['area_share_by_age'].values()))
    out = {}
    for d in combos(a.dirs, a.latest):
        rows = [seed_numbers(g, sd) for sd in seeds_of(d)]
        if not rows:
            continue
        out[str(d)] = rows
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f'\n{d} ({len(rows)} seeds)')
        t1 = all(r['detached_max'] <= .01 for r in rows)
        print(f"  T1 detached area, max over the run per seed {j(rows, 'detached_max', '{:.4f}')} "
              f"(at 0 Myr {j(rows, 'detached_now', '{:.4f}')}) <= 0.01: {'pass' if t1 else 'FAIL'}")
        vals = {'T2 intra-oceanic trench (k km)': 'io_km', 'T3 continental-side trench (k km)': 'cont_sub_km',
                'T4 production (km^2/yr)': 'production', 'T5 ridge (k km)': 'ridge_km'}
        npass = int(t1)
        for name, k in vals.items():
            lo, hi = TESTS[name]
            v = m(k)
            ok = lo <= v <= hi
            npass += ok
            sc = 1e3 if 'k km' in name else 1.
            print(f"  {name}: {j(rows, k, '{:.2f}', sc)}, mean {v/sc:.2f} (band {lo/sc:.2f}-{hi/sc:.2f}): "
                  f"{'pass' if ok else 'FAIL'}")
        print(f'  B-b tests passed: {npass} of 5')
        print(f"  beside: plates {j(rows, 'plates', '{}')} (>= 1 Mkm^2: "
              f"{j(rows, 'plates_1mkm2', '{}')}, holding {m('share_in_1mkm2'):.2f}); largest "
              f"{j(rows, 'largest_mkm2', '{:.0f}')} Mkm^2; shape ratio {m('shape_ratio'):.2f}; io / ocean-ocean "
              f"{m('io_share_oo'):.2f}; intra-oceanic consumption {m('consumption_io'):.2f} km^2/yr; area-weighted speed "
              f"{m('speed_area_weighted'):.1f} cm/yr")
        print(f"  ocean floor mean age {j(rows, 'ocean_age_mean', '{:.0f}')} Myr; area by age (20 Myr bins) "
              + ' '.join(f'{x:.2f}' for x in np.mean([r['age_hist'] for r in rows], 0)))
        print(f"  piece rule: {rows[0]['piece_stats'] or '-'}; events (seed 1) {rows[0]['events']}; run time "
              f"{j(rows, 'seconds', '{:.0f}', 60.)} min")
    keys = {'T2 intra-oceanic trench (k km)': 'w_io_km', 'T3 continental-side trench (k km)': 'w_cont_sub_km',
            'T4 production (km^2/yr)': 'w_production', 'T5 ridge (k km)': 'w_ridge_km'}
    names = list(out)
    ctrl = [n for n in names if (a.control and Path(n).name == a.control) or (not a.control and 'shared' in Path(n).name)]
    dist = lambda v, lo, hi: max(lo-v, v-hi, 0.)
    print('\nDecision (window means -100 ... 0 Myr, seed mean; each in its band or no farther from it than the control):')
    for n in names:
        rows = out[n]
        mm = {k: float(np.mean([r[k] for r in rows])) for k in keys.values()}
        cm = {k: float(np.mean([r[k] for r in out[ctrl[0]]])) for k in keys.values()} if ctrl else None
        t1 = all(r['detached_max'] <= .01 for r in rows)
        res = [f"T1 {'pass' if t1 else 'FAIL'}"]
        ok_all = t1
        for name, k in keys.items():
            lo, hi = TESTS[name]
            d = dist(mm[k], lo, hi)
            ok = d == 0 or (cm is not None and d <= dist(cm[k], lo, hi))
            ok_all &= ok
            sc = 1e3 if 'k km' in name else 1.
            res.append(f"{name.split()[0]} {mm[k]/sc:.2f}{' (control %.2f)' % (cm[k]/sc) if cm else ''} {'pass' if ok else 'FAIL'}")
        sh = float(np.mean([r['w_shape_ratio'] for r in rows]))
        print(f"  {Path(n).name}: " + '; '.join(res) + f"; shape ratio {sh:.2f} (Earth 1.51); detached "
              f"{np.mean([r['w_detached_mkm2'] for r in rows]):.1f} Mkm^2 -> {'passes' if ok_all else 'fails'}")
        sm = lambda k: float(np.mean([r[k] for r in rows]))
        io, cs, io0, cs0 = sm('start_io_km'), sm('start_cont_sub_km'), sm('start0_io_km'), sm('start0_cont_sub_km')
        sh, sh0 = cs/max(io+cs, 1.), cs0/max(io0+cs0, 1.)
        s1, s2 = cs >= 20e3, sh >= .60
        print(f"     start (-240 Myr, seed mean): S1 continental-side trench {cs/1e3:.2f}k (>= 20k; -250 Myr "
              f"{cs0/1e3:.2f}k) {'pass' if s1 else 'FAIL'}; S2 its share of subduction {sh:.2f} (>= 0.60; -250 Myr "
              f"{sh0:.2f}) {'pass' if s2 else 'FAIL'} -> {'passes' if s1 and s2 else 'fails'}; beside: intra-oceanic "
              f"{io/1e3:.2f}k, total {(io+cs)/1e3:.1f}k (Triassic ~1.55x today: van der Meer et al. 2014)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
