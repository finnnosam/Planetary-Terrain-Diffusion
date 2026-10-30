"""Continental shelves, final check (docs/planet-layer-plan.md, step 8 follow-up 1) on the 11.53 km landscape runs.

  python -m scripts.shelf_review outputs\\planet-landscape\\shelf-1021-seed1001 outputs\\planet-landscape\\shelf-1021-seed1002 ...

Bands, fixed before testing (18 Oct handoff), judged on the mean over the seeds given (three):
  whole-surface 70th-percentile elevation   -150..0 m     (before: -316; Earth -54 at 11.53 km)
  continental crust under water             20-35%        (before: 12%; Earth ~29%)
  land p25                                  < 250 m       (before: 392; Earth 163)
  lowland-plain windows (23 km input)       >= 8% of land windows (before: 2%), the planet_meso_check definition
Guard (sect. 6 of the plan): land fraction 25-35%. Reported, not banded: share of the surface at -200..0 m, the
continental volume change and rock export of the plate run (the continental budget, the next item).
Writes shelf-review.json next to the first run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.planet_meso_check import LOWLAND, cells23, classify, core_median, window_stats, windows_of

BANDS = {'whole_p70_m': (-150., 0.), 'continental_submerged': (.20, .35), 'land_p25_m': (-np.inf, 250.),
         'lowland_plain_share': (.08, np.inf)}
GUARD = {'land_fraction': (.25, .35)}


def review(run, g):
    run = Path(run)
    st = json.loads((run/'stats.json').read_text())
    z = np.load(run/'present.npz')['elevation_m'].astype(np.float64)
    a = g.area
    land = z > 0
    z23 = cells23(g.to_cube(z.astype(np.float32)))
    _, wins = windows_of(z23)
    wl, wr, _ = window_stats(wins)
    cls = classify(wl, wr, core_median(wins))
    out = {'run': str(run), 'whole_p70_m': float(np.percentile(z, 70)), 'continental_submerged': st['continental_crust_submerged'],
           'land_p25_m': float(np.percentile(z[land], 25)), 'lowland_plain_share': float(len(cls[LOWLAND[0]])/max((wl >= .98).sum(), 1)),
           'land_fraction': float(a[land].sum()/a.sum()), 'shallow_sea_share': float(a[~land & (z > -200)].sum()/a.sum())}
    ps = Path(st['plates'])/'stats.json'
    if ps.exists():
        p = json.loads(ps.read_text())
        e = p['present']['erosion_totals_km3']
        out['plate_continental_volume_change'] = p['continental_volume_change']
        out['plate_export_km3_yr'] = -(e['continental_eroded_m3']+e['continental_deposited_m3']
                                       -e.get('continental_deposited_submerged_m3', 0.))/p['years']/1e6
    return out


def main(argv=None):
    from planet_layer.sphere_graph import CubeSphere
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args(argv)
    g = CubeSphere(868)
    rows = [review(r, g) for r in a.runs]
    keys = list(BANDS)+list(GUARD)+['shallow_sea_share', 'plate_continental_volume_change', 'plate_export_km3_yr']
    mean = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
    ok = {k: bool(lo <= mean[k] <= hi) for k, (lo, hi) in {**BANDS, **GUARD}.items()}
    res = {'bands': {k: list(v) for k, v in BANDS.items()}, 'guard': {k: list(v) for k, v in GUARD.items()}, 'runs': rows,
           'mean': mean, 'pass': ok, 'all_pass': all(ok.values())}
    (Path(a.runs[0])/'shelf-review.json').write_text(json.dumps(res, indent=1, default=float))
    fmt = {'whole_p70_m': '{:.0f} m', 'continental_submerged': '{:.0%}', 'land_p25_m': '{:.0f} m', 'lowland_plain_share': '{:.1%}',
           'land_fraction': '{:.1%}', 'shallow_sea_share': '{:.1%}', 'plate_continental_volume_change': '{:+.0%}',
           'plate_export_km3_yr': '{:.1f} km3/yr'}
    print('quantity | ' + ' | '.join(Path(r['run']).name for r in rows) + ' | mean | band | pass')
    for k in keys:
        if k not in mean:
            continue
        band = BANDS.get(k, GUARD.get(k))
        print(f"{k} | " + ' | '.join(fmt[k].format(r[k]) if k in r else '-' for r in rows) + f" | {fmt[k].format(mean[k])} | "
              + (f'{band[0]:g}..{band[1]:g}' if band else '-') + ' | ' + ({True: 'yes', False: 'NO'}[ok[k]] if k in ok else '-'))
    print('all bands pass' if res['all_pass'] else 'not all bands pass', flush=True)


if __name__ == '__main__':
    main()
