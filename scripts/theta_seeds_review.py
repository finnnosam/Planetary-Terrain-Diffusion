"""The 11.53 km stage's incision threshold decided on three seeds (docs/planet-layer-plan.md, step 8 follow-up 1).

  python -m scripts.theta_seeds_review                    # newest calibration per seed on the newest sweep's winner
  python -m scripts.theta_seeds_review --meso             # ... and the meso check on the first seed's chosen run
  python -m scripts.theta_seeds_review outputs\\planet-landscape\\theta-...-seed1001 outputs\\...-seed1002 ...

Each folder is a scripts.calibrate_landscape_theta run (summary.json with one row per theta). Rule, fixed before the
runs: score per theta = the mean over the seeds of calibrate_landscape_theta's score (the larger of the denudation and
23 km texture scores); the smallest mean wins, and within 0.05 of it the smallest theta. The chosen theta's runs then
go through scripts.shelf_review (the four final bands and the land guard). Writes theta-seeds-review.json next to the
first calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def newest_calibrations(seeds):
    from scripts.run_planet_landscape import chosen_sweep_run
    out = []
    for s in seeds:
        plates = str(chosen_sweep_run(s))
        found = []
        for f in (ROOT/'outputs'/'planet-landscape').glob(f'theta-*-seed{s}/summary.json'):
            sm = json.loads(f.read_text())
            if 'chosen_theta' in sm and Path(sm['plates']) == Path(plates):
                found.append((f.stat().st_mtime, f.parent))
        if not found:
            raise SystemExit(f'no finished theta calibration for seed {s} on {plates}')
        out.append(max(found)[1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('calibrations', nargs='*')
    ap.add_argument('--seeds', default='1001,1002,1003')
    ap.add_argument('--meso', action='store_true', help='run planet_meso_check on the first seed at the chosen theta')
    a = ap.parse_args(argv)
    cals = [Path(c) for c in a.calibrations] or newest_calibrations([int(s) for s in a.seeds.split(',')])
    sums = [json.loads((c/'summary.json').read_text()) for c in cals]
    thetas = sorted({r['theta'] for s in sums for r in s['rows']})
    table = []
    for th in thetas:
        rows = [next((r for r in s['rows'] if r['theta'] == th), None) for s in sums]
        if any(r is None for r in rows):
            continue
        table.append({'theta': th, 'score': float(np.mean([r['score'] for r in rows])),
                      'denudation_score': float(np.mean([r['denudation_score'] for r in rows])),
                      'texture_score': float(np.mean([r['texture_score'] for r in rows])),
                      'per_seed': [round(r['score'], 3) for r in rows],
                      'land_p25_m': float(np.mean([r['land_percentiles_m']['25'] for r in rows])),
                      'texture_ratios': {k: float(np.mean([r['texture_ratios'][k] for r in rows])) for k in rows[0]['texture_ratios']}})
    best = min(t['score'] for t in table)
    chosen = min(t['theta'] for t in table if t['score'] <= best+.05)
    print('theta | mean score (denudation, texture) | per seed | land p25 | curvature flat / lowlands / plain-upland / hills vs Earth')
    for t in table:
        r = t['texture_ratios']
        print(f"{t['theta']:5g} | {t['score']:.2f} ({t['denudation_score']:.2f}, {t['texture_score']:.2f}) | {t['per_seed']} | "
              f"{t['land_p25_m']:.0f} | {r.get('curvature_flat', 0):.2f} / {r.get('curvature_plains-lowlands', 0):.2f} / "
              f"{r.get('curvature_plain-upland', 0):.2f} / {r.get('curvature_hills', 0):.2f}")
    runs = [c/f'theta{chosen:g}' for c in cals]
    print(f'chosen theta {chosen:g} (three-seed rule); bands on its runs:', flush=True)
    (cals[0]/'theta-seeds-review.json').write_text(json.dumps({'calibrations': [str(c) for c in cals], 'table': table,
                                                              'chosen_theta': chosen, 'runs': [str(r) for r in runs]}, indent=1))
    from scripts import shelf_review
    shelf_review.main([str(r) for r in runs])
    if a.meso:
        from scripts import planet_meso_check
        planet_meso_check.main(['--landscape', str(runs[0])])


if __name__ == '__main__':
    main()
