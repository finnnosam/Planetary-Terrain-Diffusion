"""The elevation effect of sediment compaction alone, on saved states (item B-a; plan doc, thirty-sixth pass).

  python -m scripts.compaction_effect outputs\\planet-calibration\\sweep-20261025-170555\\boundary_normal-ring_extension-False_ext_onset_thick_m-45000

Runs differ in their plate histories from the first steps, so a sweep compares B-a with its control only through seed
spread. Here the same state (a snapshot's crust, ages, sediment and dynamic topography, put back into a PlateSim's
world) is given both elevation laws: the run's own (`sed` a layer of rho_sediment) and sediment_compaction (`sed`
read as solid volume, the eroded volume the run carried). Reported per snapshot: the deep floor (mean depth below the
datum of the floor deeper than 1.5 km) and its share, water below the datum (global equivalent layer), sea level on the
datum, and the label-free ocean's bulk sediment against GlobSed's 610 m. Changes nothing in the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import plates as P
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import label_free_continent, run_params

FLOOR_BELOW_M = 1500.
GLOBSED_OCEAN_M = 610.       # label-free ocean, n = 217 (scripts.ocean_sediment_probe)


def seed_dirs(dirs):
    out = []
    for d in dirs:
        d = Path(d)
        out += sorted(d.glob('seed*')) if not (d/'snapshots').exists() else [d]
    return out


def state_into(sim, z):
    w = sim.world
    w['cont'] = z['continental'].astype(bool)
    w['thick'] = z['crust_thickness_m'].astype(np.float64)
    w['born'] = float(z['time_myr'])-z['crust_age_myr'].astype(np.float64)
    w['sed'] = z['sediment_m'].astype(np.float64)
    sim.t = float(z['time_myr'])
    sim.dyn = z['dynamic_topography_m'].astype(np.float64)
    sim.sea_level = float(z['sea_level_m'])


def measure(sim, g, compaction):
    sim.p['sediment_compaction'] = compaction
    z = sim.elevation()
    sl = sim.sea_level
    zd = z+sl                                            # on the datum
    a = g.area
    deep = zd < -FLOOR_BELOW_M
    h, _, _ = P.sediment_column(sim.world['sed'], sim.world['cont'], sim.p)
    T = sim.world['thick']+h
    ocean = ~label_free_continent(T)
    return {'floor_mean_m': float(-np.average(zd[deep], weights=a[deep])), 'floor_share': float(a[deep].sum()/a.sum()),
            'water_below_datum_m': float((np.clip(-zd, 0, None)*a).sum()/a.sum()), 'sea_level_on_datum_m': sl,
            'ocean_bulk_sediment_m': float(np.average(h[ocean], weights=a[ocean]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--times', default='0', help='snapshot times (Myr), comma-separated')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    g = CubeSphere(217)
    out = {}
    print('Earth: deep floor 4205 m on 0.614 of the surface, water below the datum 2629 m (crust_budget); GlobSed on the'
          f' label-free ocean {GLOBSED_OCEAN_M:.0f} m (bulk)')
    for sd in seed_dirs(a.dirs):
        p = {**run_params(sd), 'threads': 1}
        if p.get('sediment_compaction'):
            print(f'{sd}: already a compaction run, skipped'); continue
        sim = P.PlateSim(g, seed=1, params=p)
        for t in (float(x) for x in a.times.split(',')):
            f = Path(sd)/'snapshots'/('t+0000.0.npz' if t == 0 else f't{t:+07.1f}.npz')
            z = np.load(f)
            state_into(sim, z)
            r0 = measure(sim, g, False)
            # the run's own law against the snapshot (float32 fields, sea-level flips at a few nodes): p99 of |difference|
            r0['p99_abs_vs_snapshot_m'] = float(np.percentile(np.abs(sim.elevation()-z['elevation_m']), 99))
            state_into(sim, z)
            r1 = measure(sim, g, True)
            out[f'{sd}|{t}'] = {'layer': r0, 'compaction': r1}
            print(f"{sd} t {t:.0f}: deep floor {r0['floor_mean_m']:.0f} -> {r1['floor_mean_m']:.0f} m (share {r0['floor_share']:.3f}"
                  f" -> {r1['floor_share']:.3f}); water below the datum {r0['water_below_datum_m']:.0f} -> {r1['water_below_datum_m']:.0f}"
                  f" m; sea level on the datum {r0['sea_level_on_datum_m']:.0f} -> {r1['sea_level_on_datum_m']:.0f} m; ocean bulk sediment"
                  f" {r0['ocean_bulk_sediment_m']:.0f} -> {r1['ocean_bulk_sediment_m']:.0f} m (check: the run's law vs the snapshot"
                  f" p99 {r0['p99_abs_vs_snapshot_m']:.2f} m)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
