"""Incision threshold of the 11.53 km landscape stage, calibrated against Earth's denudation rates
(docs/planet-layer-plan.md, step 7).

  python -m scripts.calibrate_landscape_theta --plates latest
  python -m scripts.calibrate_landscape_theta --plates latest --thetas 0,5,10,20

Builds the parcel forcing once, then runs the full 30 Myr planet landscape (scripts.run_planet_landscape.run) for
every theta, one after another (~13 min each on 16 threads). Writes outputs/planet-landscape/theta-<timestamp>/
theta<value>/ (each a normal landscape run) and summary.json.

Selection rule (revised in step 8, before the next runs): score = the larger of
  * the denudation score: denudation over the last 10 Myr on land, by present elevation (200-500 m, 0.5-1 km,
    1-2 km) and the land mean, against Earth's bands (scripts.run_planet_landscape.EARTH_DENUDATION), worst log10
    distance outside a band;
  * the texture score: the meso's 23 km input built from the result, 368 km relief p25/50/75 and 23 km curvature of
    flat / lowland-plain / plain-upland / hills windows against the meso's Earth validation split, worst |log10| ratio.
The smallest score wins; within 0.05 of it, the smallest theta. (Step 7 used denudation alone and picked theta 20,
which left plains with 1 m of 23 km curvature against Earth's 31.) Hypsometry is reported, not used: the 11.53 km
stage starts from the plate layer's surface, and fitting Earth's hypsometry here would tune it to cover the plate
layer's errors.
"""
import argparse
import json
import time
from pathlib import Path

from planet_layer import coupling
from planet_layer.sphere_graph import CubeSphere
from scripts import run_planet_landscape as rpl

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plates', required=True, help="plate run seed folder, or 'latest'")
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--thetas', default='0,5,10,20')
    p.add_argument('--window', type=float, default=30.)
    p.add_argument('--n', type=int, default=868)
    a = p.parse_args(argv)
    seed_dir = rpl.resolve_plates(a.plates, a.seed)
    thetas = [float(v) for v in a.thetas.split(',')]
    out = ROOT/'outputs'/'planet-landscape'/f"theta-{time.strftime('%Y%m%d-%H%M%S')}-{seed_dir.name}"
    out.mkdir(parents=True, exist_ok=True)
    hist = coupling.PlateHistory(seed_dir/'snapshots')
    g = CubeSphere(a.n)
    rows = []
    for th in thetas:
        st = rpl.run(seed_dir, out/f'theta{th:g}', a.n, a.window, 5e4, th, forcing_dir=out/'forcing', g=g, hist=hist)
        pl = st['planet_n868']
        tex = st['texture23']
        rows.append({'theta': th, 'score': max(st['denudation_score'], tex['score']), 'denudation_score': st['denudation_score'],
                     'texture_score': tex['score'], 'texture_ratios': tex['ratios'], 'denudation': st[f'denudation_last{min(10, int(a.window))}myr_m_per_myr'],
                     'land_fraction': pl['land_fraction'], 'land_percentiles_m': pl['land_percentiles_m'],
                     'continental_crust_submerged': st['continental_crust_submerged']})
        (out/'summary.json').write_text(json.dumps({'plates': str(seed_dir), 'rows': rows}, indent=1))
    best = min(r['score'] for r in rows)
    chosen = min(r['theta'] for r in rows if r['score'] <= best+.05)
    summary = {'plates': str(seed_dir), 'earth_denudation_m_per_myr': rpl.EARTH_DENUDATION, 'rows': rows, 'chosen_theta': chosen}
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    print('\ntheta | score (denudation, texture) | denudation 200-500 / 500-1000 / 1000-2000 / mean (m/Myr) | land | p25/p50/p75/p90 (m)')
    for r in rows:
        d, q = r['denudation'], r['land_percentiles_m']
        print(f"{r['theta']:5g} | {r['score']:.2f} ({r['denudation_score']:.2f}, {r['texture_score']:.2f}) | {d['200-500']:.0f} / {d['500-1000']:.0f} / {d['1000-2000']:.0f} / {d['mean_land']:.0f}"
              f" | {r['land_fraction']:.1%} | {q['25']:.0f}/{q['50']:.0f}/{q['75']:.0f}/{q['90']:.0f}")
    print(f'chosen theta {chosen:g}; written {out}')


if __name__ == '__main__':
    main()
