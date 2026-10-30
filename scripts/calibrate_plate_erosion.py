"""Set the plate layer's incision threshold (erosion_theta) at its own grid, n = 217 (46 km), against Earth's
hypsometry on the same grid (scripts.earth_reference, ETOPO 2022 without ice sheets).

  python -m scripts.calibrate_plate_erosion                       # theta 8,12,16,20,25 x seeds 1001,1002, parallel
  python -m scripts.calibrate_plate_erosion --thetas 10,15 --seeds 1001 --workers 4

Each run: 250 Myr from a supercontinent. Measured over the last 50 Myr (every 10 Myr): land fraction, land elevation
percentiles (p10, p25, p50, p75, p90), land denudation by elevation class. Rule, fixed before running: pick the theta
whose worst error over {land fraction, land p25, p50, p75, p90} (relative to Earth, averaged over seeds) is smallest.
Writes outputs/planet-calibration/<timestamp>/results.json and a table.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QS = ('10', '25', '50', '75', '90')


def one(args):
    theta, seed, n, years = args
    from planet_layer.sphere_graph import CubeSphere
    from planet_layer.plates import PlateSim
    g = CubeSphere(n)
    sim = PlateSim(g, seed=seed, t0=-years, params={'erosion_theta': theta, 'threads': 2})
    rates = []
    orig = sim._erode

    def erode(th):
        z0 = sim.elevation(th, sim.world['sed'])
        e, d = orig(th)
        if sim.t > -50:
            land = (z0 > 0) & sim.world['cont']
            rates.append((z0[land], -e[land], g.area[land]))
        return e, d
    sim._erode = erode
    t0 = time.time()
    snaps = []
    v0 = sim.continental_volume()
    while sim.t < -1e-6:
        sim.step()
        if sim.t > -50.5 and round(sim.t) % 10 == 0:
            st = sim.state(); z = st['elevation_m']; land = z > 0
            snaps.append({'land_fraction': float(g.area[land].sum()/g.area.sum()),
                          'land_percentiles_m': {q: float(np.percentile(z[land], float(q))) for q in QS},
                          'land_mean_m': float(np.average(z[land], weights=g.area[land]))})
    zz = np.concatenate([r[0] for r in rates]); ee = np.concatenate([r[1] for r in rates]); aa = np.concatenate([r[2] for r in rates])
    den = {}
    for lo, hi in ((0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 9000)):
        m = (zz >= lo) & (zz < hi)
        den[f'{lo}-{hi}'] = float(np.average(ee[m], weights=aa[m])) if m.any() else None
    return {'theta': theta, 'seed': seed, 'seconds': time.time()-t0, 'plates': len(sim.plates),
            'continental_volume_change': sim.continental_volume()/v0-1,
            'land_fraction': float(np.mean([s['land_fraction'] for s in snaps])),
            'land_mean_m': float(np.mean([s['land_mean_m'] for s in snaps])),
            'land_percentiles_m': {q: float(np.mean([s['land_percentiles_m'][q] for s in snaps])) for q in QS},
            'land_denudation_m_myr': den, 'mean_land_denudation_m_myr': float(np.average(ee, weights=aa))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--thetas', default='8,12,16,20,25')
    p.add_argument('--seeds', default='1001,1002')
    p.add_argument('--n', type=int, default=217)
    p.add_argument('--years', type=float, default=250.)
    p.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2)//2)))
    a = p.parse_args(argv)
    if a.n < 96:
        raise SystemExit('n < 96: the erosion parameters are for ~46 km cells and lose meaning on much coarser grids '
                         '(at n = 48 the slope cap alone allows 10 km steps between cells)')
    from scripts.earth_reference import load
    earth = load(a.n)['no_ice_sheets']
    print('Earth (no ice sheets, n=%d): land %.1f%%, land p10/25/50/75/90 = %s m' % (
        a.n, 100*earth['land_fraction'], '/'.join(f"{earth['land_percentiles_m'][q]:.0f}" for q in QS)), flush=True)
    jobs = [(float(t), int(s), a.n, a.years) for t in a.thetas.split(',') for s in a.seeds.split(',')]
    out = ROOT/'outputs'/'planet-calibration'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(jobs)} runs on {a.workers} workers -> {out}', flush=True)
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            results.append(r)
            print(f"theta {r['theta']:5.1f} seed {r['seed']}: land {r['land_fraction']:.1%}, land p25/50/75/90 "
                  f"{'/'.join(format(r['land_percentiles_m'][q], '.0f') for q in QS[1:])} m, "
                  f"denudation {r['mean_land_denudation_m_myr']:.0f} m/Myr, volume {r['continental_volume_change']:+.1%}, "
                  f"{r['seconds']/60:.1f} min", flush=True)
    table = []
    for t in sorted({r['theta'] for r in results}):
        rs = [r for r in results if r['theta'] == t]
        lf = np.mean([r['land_fraction'] for r in rs])
        pct = {q: np.mean([r['land_percentiles_m'][q] for r in rs]) for q in QS}
        errs = {'land_fraction': abs(lf/earth['land_fraction']-1),
                **{f'p{q}': abs(pct[q]/earth['land_percentiles_m'][q]-1) for q in ('25', '50', '75', '90')}}
        table.append({'theta': t, 'land_fraction': lf, 'land_percentiles_m': pct, 'errors': errs, 'worst': max(errs.values()),
                      'mean_land_denudation_m_myr': np.mean([r['mean_land_denudation_m_myr'] for r in rs]),
                      'continental_volume_change': np.mean([r['continental_volume_change'] for r in rs])})
    table.sort(key=lambda r: r['worst'])
    (out/'results.json').write_text(json.dumps({'earth_no_ice': earth, 'runs': results, 'table': table}, indent=1, default=float))
    print('\ntheta  land   p25   p50   p75   p90   worst error  denudation  volume')
    for r in sorted(table, key=lambda r: r['theta']):
        print(f"{r['theta']:5.1f} {r['land_fraction']:5.1%} " + ' '.join(f"{r['land_percentiles_m'][q]:5.0f}" for q in ('25', '50', '75', '90'))
              + f"   {r['worst']:6.1%}      {r['mean_land_denudation_m_myr']:5.0f}     {r['continental_volume_change']:+.1%}")
    print(f"Earth {earth['land_fraction']:5.1%} " + ' '.join(f"{earth['land_percentiles_m'][q]:5.0f}" for q in ('25', '50', '75', '90')))
    print(f"best theta: {table[0]['theta']} (worst error {table[0]['worst']:.1%})")


if __name__ == '__main__':
    main()
