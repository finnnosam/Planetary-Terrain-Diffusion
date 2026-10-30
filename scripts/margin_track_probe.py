"""What thickens old passive margins (twentieth pass)? Thin continental crust (crust + sediment < 30 km) within
150 km of a passive edge at --from Myr is followed in its plate's frame (saved rotations) to every later snapshot
(10 Myr steps); its crust and sediment are read there, with what happened to it (still continental, still near
the coast, collided, near a plate boundary).

  python -m scripts.margin_track_probe outputs\\planet-calibration\\sweep-20261024-135611 --from -100
  python -m scripts.margin_track_probe --latest --from -100
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import fields, graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import run_params, total_thickness_m
from scripts.margin_width import BOUNDARY_KM

ROOT = Path(__file__).resolve().parents[1]


def load(sdir, t):
    z = np.load(sdir/'snapshots'/f't{t:+07.1f}.npz')
    Q = {r['id']: np.array(r['Q']) for r in json.loads((sdir/'snapshots'/f't{t:+07.1f}-plates.json').read_text())}
    return z, Q


def track(sdir, g, t0, step=10.):
    p = run_params(sdir)
    z0, Q0 = load(sdir, t0)
    cont = z0['continental'].astype(bool); T = total_thickness_m(z0, p)
    nb = g.nbr; ok = nb >= 0
    edge = cont & np.any(np.where(ok, ~cont[np.where(ok, nb, 0)], False), 1)
    far = gl.distance_to(z0['boundary_type'] != 0, nb, g.dist, 5000e3) > BOUNDARY_KM*1e3
    near = gl.distance_to(edge & far, nb, g.dist, 200e3) <= 150e3
    sel = np.flatnonzero(cont & (T < 30e3) & near & far)
    pid = z0['plate_id'][sel]
    frame = np.stack([g.xyz[i]@Q0[k] for i, k in zip(sel, pid)])           # plate frame: Q^T x (row vectors)
    rows = []
    t = t0
    while t <= -step+1e-6 or abs(t) < 1e-6 or t < 0:
        z, Q = load(sdir, t)
        alive = np.array([k in Q for k in pid])
        xs = np.stack([Q[k]@f if k in Q else f for k, f in zip(pid, frame)])
        j = fields.nearest_nodes(g, xs)
        same = alive & (z['plate_id'][j] == pid)
        c = z['continental'].astype(bool)
        crust = z['crust_thickness_m'][j].astype(np.float64); sed = np.where(c[j], z['sediment_m'][j], 0.)
        ed = c & np.any(np.where(ok, ~c[np.where(ok, nb, 0)], False), 1)
        dcoast = gl.distance_to(ed, nb, g.dist, 1000e3)[j]
        dbound = gl.distance_to(z['boundary_type'] != 0, nb, g.dist, 3000e3)[j]
        collided = (z['orogeny_age_myr'][j] < (t-t0)+1e-6) & np.isfinite(z['orogeny_age_myr'][j])
        m = same & c[j]
        w = g.area[sel]
        def wm(v, mm):
            return float(np.average(v[mm], weights=w[mm])) if mm.any() else np.nan
        quiet = m & (dbound > BOUNDARY_KM*1e3) & ~collided
        rows.append({'t': t, 'tracked': float(w[same].sum()/w.sum()), 'continental': float(w[m].sum()/w[same].sum()),
                     'crust_km': wm(crust, m)/1e3, 'sed_km': wm(sed, m)/1e3,
                     'thin_lt30': float(w[m & (crust+sed < 30e3)].sum()/w[m].sum()),
                     'within_150km_of_coast': float(w[m & (dcoast <= 150e3)].sum()/w[m].sum()),
                     'near_boundary': float(w[m & (dbound <= BOUNDARY_KM*1e3)].sum()/w[m].sum()),
                     'collided': float(w[m & collided].sum()/w[m].sum()),
                     'quiet_share': float(w[quiet].sum()/w[m].sum()), 'quiet_crust_km': wm(crust, quiet)/1e3,
                     'quiet_sed_km': wm(sed, quiet)/1e3})
        t += step
        if t > 1e-6:
            break
    return sel.size, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--from', dest='t0', type=float, default=-100.)
    ap.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
    g = CubeSphere(217)
    for d in map(Path, a.dirs):
        seeds = [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
        for s in seeds:
            n, rows = track(s, g, a.t0)
            print(f"{s.parent.name[-28:]}/{s.name}: {n} thin margin nodes at {a.t0:+.0f} Myr")
            print('     t  tracked  cont  crust  sed  <30km  coast<=150  near-bdy  collided | quiet: share crust sed')
            for r in rows:
                print(f"  {r['t']:+5.0f}  {r['tracked']:6.2f}  {r['continental']:5.2f}  {r['crust_km']:5.1f} {r['sed_km']:4.2f}  "
                      f"{r['thin_lt30']:5.2f}  {r['within_150km_of_coast']:9.2f}  {r['near_boundary']:8.2f}  {r['collided']:8.2f} | "
                      f"{r['quiet_share']:.2f} {r['quiet_crust_km']:5.1f} {r['quiet_sed_km']:4.2f}")


if __name__ == '__main__':
    main()
