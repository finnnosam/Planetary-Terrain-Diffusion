"""Plate-layer parameter sweep judged on texture and hypsometry together (docs/planet-layer-plan.md, step 8).

  python -m scripts.sweep_plates --grid dyn_std_m=400,600,800 --grid dyn_lmax,dyn_slope=12:2.5,30:1
  python -m scripts.sweep_plates --grid erosion_theta=8,12 --seeds 1001,1002

--grid KEY=V1,V2 sweeps one parameter; --grid K1,K2=A1:A2,B1:B2 sweeps several together (one combination per
colon-joined tuple). Separate --grid options are crossed.

Every combination x seed runs 250 Myr at n = 217 (parallel workers, 2 threads each) and is scored on its present
day against Earth on the same grid (ETOPO, no ice sheets):
  texture score    scripts.plate_texture: worst |log10| of band-power ratios (92-736 km) and 368 km relief p25/50/75;
  hypsometry score worst |log10| of land fraction and land p25 / p50 / p75 / p90 ratios;
  arrangement      worst |log10| of high-ground share, flanked-lowland share and slope p90 (planet_training.
                   arrangement_metrics on ~2,950 km windows, scripts.plate_texture.arrangement; added in step 8 after
                   round mantle swells passed the texture score without making ranges);
  budget score     |continental volume change over the run| (a fraction: -33% scores 0.33, as a factor-2 error does);
  flux score       log10 distance of the continents' net rock export to the ocean from Earth's 4-8 km^3/yr;
  score            the largest of the five, averaged over seeds.
Rule, fixed before the sweep: the smallest score wins; within 0.03 of it, the combination listed first. With
--rule shelf only combinations whose seed means pass SHELF_BANDS (whole-surface p70, continental crust under water,
land p25) compete; if none passes, nothing is chosen.
Writes outputs/planet-calibration/sweep-<timestamp>/results.json and keeps each run's snapshots
(outputs/planet-calibration/sweep-<timestamp>/<combo>/seed<s>/snapshots), so the winner can go straight to the
landscape stage. <combo> names only the parameters that differ between the combinations (e.g. rift_rule-pure_shear;
a single combination is combo-01, or defaults without parameters), or combo-01, combo-02, ... when that would exceed MAX_NAME characters; the
parameters shared by all combinations are in sweep.json, each combination's full set in <combo>/params.json and in
results.json ('chosen_dir' names the winner's folder). Before 24 Oct the folder carried every swept parameter
(paths over Windows' 260-character limit).
"""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QS = ('25', '50', '75', '90')
MAX_NAME = 80


def combo_names(combos):
    """Short folder name for every combination: only the parameters whose values differ between combinations."""
    keys = list(combos[0]) if combos else []
    varying = [k for k in keys if len({c[k] for c in combos}) > 1]
    names = ['_'.join(f'{k}-{c[k]}' for k in varying) or ('combo-01' if c else 'defaults') for c in combos]
    if any(len(n) > MAX_NAME for n in names) or len(set(names)) < len(names):
        names = [f'combo-{i+1:02d}' for i in range(len(combos))]
    return names


def one(job):
    params, seed, out, years = job
    os.environ.setdefault('NUMBA_NUM_THREADS', '2')
    from scripts import simulate_plates
    argv = ['--n', '217', '--seeds', str(seed), '--years', str(years), '--out', str(out), '--param', 'threads=2']
    for k, v in params.items():
        argv += ['--param', f'{k}={v}']
    t0 = time.time()
    simulate_plates.main(argv)
    return {'params': params, 'seed': seed, 'dir': str(Path(out)/f'seed{seed}'), 'seconds': time.time()-t0}


def score(seed_dir, earth_tex, earth_hyp, g):
    from scripts import plate_texture as pt
    z = np.load(Path(seed_dir)/'snapshots'/'t+0000.0.npz')['elevation_m'].astype(np.float64)
    tex = pt.compare(pt.texture(z, g), earth_tex)
    ar, ea = pt.arrangement(z, g), earth_tex['arrangement']
    ar_ratios = {k: ar.get(k, 0.)/ea[k] for k in ARRANGEMENT_KEYS}
    ar_score = max(abs(np.log10(max(v, 1e-2))) for v in ar_ratios.values())
    st = json.loads((Path(seed_dir)/'stats.json').read_text())
    dvol = st['continental_volume_change']
    budget_score = abs(dvol)
    e = st['present']['erosion_totals_km3']
    # km^3/yr delivered to the sea: eroded minus what stayed on land (deposits on submerged continental crust count
    # as delivered, as Earth's river-delivery figure includes shelves; runs before 22 Oct lack the split)
    export = -(e['continental_eroded_m3']+e['continental_deposited_m3']-e.get('continental_deposited_submerged_m3', 0.))/st['years']/1e6
    lo, hi = EARTH_EXPORT_KM3_YR
    # net deposition on the continents (export <= 0) scores as an export of 0.1 km^3/yr, not as NaN
    flux_score = float(np.log10(lo/max(export, .1)) if export < lo else np.log10(export/hi) if export > hi else 0.)
    land = z > 0
    hyp = {'land_fraction': float(g.area[land].sum()/g.area.sum())}
    hyp.update({q: float(np.percentile(z[land], float(q))) for q in QS})
    errs = [abs(np.log10(hyp['land_fraction']/earth_hyp['land_fraction']))]
    errs += [abs(np.log10(max(hyp[q], 1.)/earth_hyp['land_percentiles_m'][q])) for q in QS]
    sh = shelf_metrics(z, np.load(Path(seed_dir)/'snapshots'/'t+0000.0.npz')['continental'], g)
    return {'texture': tex, 'hypsometry': hyp, 'arrangement': ar, 'arrangement_ratios': ar_ratios, 'shelf': sh,
            'shelf_pass': shelf_pass(sh),
            'continental_volume_change': dvol, 'texture_score': tex['texture_score'], 'hypsometry_score': float(max(errs)),
            'arrangement_score': float(ar_score), 'budget_score': float(budget_score),
            'export_km3_yr': float(export), 'flux_score': flux_score,
            'score': float(max(tex['texture_score'], max(errs), ar_score, budget_score, flux_score))}


# continental shelves (step 8 follow-up 1), bands fixed before the shelf sweep, on the 46 km grid (Earth, ETOPO on the
# same grid without ice sheets: whole-surface p70 -43 m, 5.0% of the surface at -200..0 m, land p25 166 m; continental
# crust under water ~29%, Cogley 1984 / CRUST1.0)
SHELF_BANDS = {'whole_p70_m': (-150., 0.), 'continental_submerged': (.20, .35), 'land_p25_m': (-np.inf, 250.)}


def shelf_metrics(z, cont, g):
    a = g.area
    land = z > 0
    return {'whole_p70_m': float(np.percentile(z, 70)), 'continental_submerged': float(a[cont & ~land].sum()/a[cont].sum()),
            'land_p25_m': float(np.percentile(z[land], 25)), 'shallow_sea_share': float(a[(z > -200) & ~land].sum()/a.sum()),
            'continental_fraction': float(a[cont].sum()/a.sum())}


def shelf_pass(m):
    return bool(all(lo <= m[k] <= hi for k, (lo, hi) in SHELF_BANDS.items()))


# rock delivered from the continents to the ocean, km^3/yr: ~14 Gt/yr of pre-human suspended load plus ~4 Gt/yr
# dissolved (Milliman & Farnsworth 2011; Syvitski et al. 2005), at 2.65 t/m^3, ~6.8; band kept wide
EARTH_EXPORT_KM3_YR = (4., 8.)


ARRANGEMENT_KEYS = ('high_share', 'flanked_low_fraction', 'slope_p90_m_per_km')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', action='append', default=[], metavar='KEY=V1,V2', help='a parameter and its values')
    p.add_argument('--seeds', default='1001,1002')
    p.add_argument('--years', type=float, default=250.)
    p.add_argument('--rule', choices=('score', 'shelf'), default='score',
                   help="shelf: only combinations whose seed means pass SHELF_BANDS compete (step 8 shelf sweep)")
    p.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2)//2)))
    a = p.parse_args(argv)
    from planet_layer.plates import PARAMS
    from planet_layer.sphere_graph import CubeSphere
    from scripts import plate_texture as pt
    from scripts.earth_reference import load
    keys, values = [], []
    for spec in a.grid:
        k, v = spec.split('=', 1)
        ks = k.split(',')
        for kk in ks:
            if kk not in PARAMS:
                raise SystemExit(f'unknown parameter {kk!r}')
        tuples = [t.split(':') for t in v.split(',')]
        if any(len(t) != len(ks) for t in tuples):
            raise SystemExit(f'--grid {spec}: each value needs {len(ks)} colon-joined parts')
        keys.append(ks); values.append(tuples)
    combos = [{kk: vv for ks, t in zip(keys, c) for kk, vv in zip(ks, t)} for c in itertools.product(*values)] or [{}]
    seeds = [int(s) for s in a.seeds.split(',')]
    out = ROOT/'outputs'/'planet-calibration'/f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    dirnames = combo_names(combos)
    name = lambda c: dirnames[combos.index(c)]
    fixed = {k: v for k, v in combos[0].items() if all(c.get(k) == v for c in combos)} if len(combos) > 1 else {}
    (out/'sweep.json').write_text(json.dumps({'fixed': fixed, 'combos': dict(zip(dirnames, combos)), 'seeds': seeds,
                                              'years': a.years, 'rule': a.rule}, indent=1))
    for c in combos:
        (out/name(c)).mkdir(parents=True, exist_ok=True)
        (out/name(c)/'params.json').write_text(json.dumps(c, indent=1))
    if fixed:
        print('shared by all combinations: '+', '.join(f'{k}={v}' for k, v in fixed.items()), flush=True)
    jobs = [(c, s, out/name(c), a.years) for c in combos for s in seeds]
    print(f'{len(jobs)} runs on {a.workers} workers -> {out}', flush=True)
    g = CubeSphere(217)
    earth_tex, earth_hyp = pt.earth_texture(g), load(217)['no_ice_sheets']
    runs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            r.update(score(r['dir'], earth_tex, earth_hyp, g))
            runs.append(r)
            print(f"{name(r['params'])} seed {r['seed']}: texture {r['texture_score']:.2f}, hypsometry {r['hypsometry_score']:.2f}, "
                  f"arrangement {r['arrangement_score']:.2f}, volume {r['continental_volume_change']:+.0%}, export {r['export_km3_yr']:.1f} km3/yr, "
                  f"land {r['hypsometry']['land_fraction']:.1%}, {r['seconds']/60:.1f} min", flush=True)
            (out/'results.json').write_text(json.dumps({'runs': runs}, indent=1, default=float))
    table = []
    for c in combos:
        rs = [r for r in runs if r['params'] == c]
        table.append({'params': c, 'score': float(np.mean([r['score'] for r in rs])),
                      'texture_score': float(np.mean([r['texture_score'] for r in rs])),
                      'hypsometry_score': float(np.mean([r['hypsometry_score'] for r in rs])),
                      'arrangement_score': float(np.mean([r['arrangement_score'] for r in rs])),
                      'budget_score': float(np.mean([r['budget_score'] for r in rs])),
                      'flux_score': float(np.mean([r['flux_score'] for r in rs])),
                      'export_km3_yr': float(np.mean([r['export_km3_yr'] for r in rs])),
                      'continental_volume_change': float(np.mean([r['continental_volume_change'] for r in rs])),
                      'arrangement_ratios': {k: float(np.mean([r['arrangement_ratios'][k] for r in rs])) for k in ARRANGEMENT_KEYS},
                      'land_fraction': float(np.mean([r['hypsometry']['land_fraction'] for r in rs])),
                      'band_power_ratio': {k: float(np.mean([r['texture']['band_power_ratio'][k] for r in rs])) for k in earth_tex['band_power_m2']},
                      'relief_368km_m': {q: float(np.mean([r['texture']['relief_ratio'][q] for r in rs]))*earth_tex['relief_368km_m'][q] for q in ('25', '50', '75')},
                      'land_percentiles_m': {q: float(np.mean([r['hypsometry'][q] for r in rs])) for q in QS},
                      'shelf': {k: float(np.mean([r['shelf'][k] for r in rs])) for k in rs[0]['shelf']}})
        table[-1]['shelf_pass'] = shelf_pass(table[-1]['shelf'])
    pool = [t for t in table if t['shelf_pass']] if a.rule == 'shelf' else table
    if not pool:
        print('rule shelf: no combination passes the shelf bands (seed means); nothing chosen')
        pool = table
    best = min(t['score'] for t in pool)
    chosen = next(t for t in pool if t['score'] <= best+.03)
    chosen = chosen if (a.rule != 'shelf' or chosen['shelf_pass']) else {'params': None}
    (out/'results.json').write_text(json.dumps({'earth_texture': earth_tex, 'earth_hypsometry': earth_hyp, 'runs': runs,
                                                'table': table, 'chosen': chosen['params'],
                                                'chosen_dir': None if chosen['params'] is None else name(chosen['params'])},
                                    indent=1, default=float))
    print('\nparams | score | texture | hypsometry | arrangement | budget (volume change) | flux (export km3/yr) | land | relief p25/50/75 | power 92-147/147-245/245-368/368-736')
    for t in table:
        print(f"{name(t['params'])} | {t['score']:.2f} | {t['texture_score']:.2f} | {t['hypsometry_score']:.2f} | {t['arrangement_score']:.2f} | "
              f"{t['budget_score']:.2f} ({t['continental_volume_change']:+.0%}) | {t['flux_score']:.2f} ({t['export_km3_yr']:.1f}) | {t['land_fraction']:.1%} | "
              + '/'.join(f"{v:.0f}" for v in t['relief_368km_m'].values()) + ' | ' + '/'.join(f'{v:.2f}' for v in t['band_power_ratio'].values()))
    print('\nshelves: params | p70 (m) | continental crust under water | land p25 (m) | sea above -200 m | continental area | pass')
    for t in table:
        sh = t['shelf']
        print(f"{name(t['params'])} | {sh['whole_p70_m']:.0f} | {sh['continental_submerged']:.0%} | {sh['land_p25_m']:.0f} | "
              f"{sh['shallow_sea_share']:.1%} | {sh['continental_fraction']:.0%} | {'yes' if t['shelf_pass'] else 'no'}")
    print('Earth (46 km, no ice) | -43 | ~29% | 166 | 5.0% | ~40% | bands p70 -150..0, under water 20-35%, p25 < 250')
    print(f"Earth | | | | {earth_hyp['land_fraction']:.1%} | " + '/'.join(f"{earth_tex['relief_368km_m'][q]:.0f}" for q in ('25', '50', '75')))
    print(f"chosen: {name(chosen['params']) if chosen['params'] is not None else 'none'}; written {out}")


if __name__ == '__main__':
    main()
