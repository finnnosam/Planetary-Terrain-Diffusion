"""Passive-margin edges by the age of the ocean floor beside them (nineteenth pass): are abrupt edges (no crust
< 30 km) old rifted margins that were thickened later, or born abrupt?

  python -m scripts.margin_age_probe outputs\\planet-calibration\\sweep-20261024-114002
  python -m scripts.margin_age_probe --latest

Test (fixed 24 Oct, nineteenth pass): passive edges beside 100-200 Myr ocean <= 30% abrupt (Earth's ~180 Myr Atlantic
margins are not abrupt).
"""
import argparse
from pathlib import Path

import numpy as np

from planet_layer import graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import run_params, total_thickness_m
from scripts.margin_width import BOUNDARY_KM


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
    g = CubeSphere(217); nb = g.nbr; ok = nb >= 0
    for d in a.dirs:
        d = Path(d)
        seeds = [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
        for s in seeds:
            z = np.load(s/'snapshots'/'t+0000.0.npz')
            cont = z['continental'].astype(bool); T = total_thickness_m(z, run_params(s))
            age = np.nan_to_num(z['ocean_age_myr'].astype(np.float64), nan=-1.)
            j = np.where(ok, nb, 0)
            oc_age = np.where(ok & ~cont[j], age[j], -1.).max(1)
            edge = cont & (oc_age >= 0)
            far = gl.distance_to(z['boundary_type'] != 0, nb, g.dist, 5000e3) > BOUNDARY_KM*1e3
            passive = edge & far
            d30 = gl.distance_to(cont & (T >= 30e3), nb, g.dist, 3000e3)
            rows = []
            for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 200), (200, 1e4)):
                m = passive & (oc_age >= lo) & (oc_age < hi)
                if m.sum() < 10:
                    continue
                rows.append(f"{lo}-{hi:.0f} Myr: {m.sum()/passive.sum():.0%} of edges, abrupt {np.mean(d30[m] == 0):.0%}, "
                            f"edge T p50 {np.median(T[m])/1e3:.1f} km")
            print(f"{s.parent.name[-12:]}/{s.name}: " + ' | '.join(rows), flush=True)


if __name__ == '__main__':
    main()
