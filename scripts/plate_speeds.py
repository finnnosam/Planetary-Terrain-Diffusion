"""Plate speeds of a plate-layer snapshot against Earth's, both in a no-net-rotation frame (plan doc, thirteenth pass).

  python -m scripts.plate_speeds outputs\\planet-calibration\\sweep-...\\seed1001\\snapshots\\t+0000.0.npz

For every plate: area, area-RMS speed after removing the net rotation of the whole plate set (the rotation for which
the area integral of r x v vanishes), and continental share. Earth: outputs/planet-reference/earth-plate-speeds-nnr.json
(Bird 2003 poles and polygons, computed once; see its 'source'). Compared on plates over 10 Mkm^2, split into mostly
continental (share > 0.45) and mostly oceanic.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer.sphere_graph import CubeSphere

ROOT = Path(__file__).resolve().parents[1]
EARTH = ROOT/'outputs'/'planet-reference'/'earth-plate-speeds-nnr.json'


def nnr_speeds(g, plate_id, omega, continental):
    """omega: {plate id: angular velocity vector, rad/yr}. Returns (net rotation rad/yr, rows of (id, area Mkm^2,
    rms speed cm/yr, continental share))."""
    a, x = g.area, g.xyz
    I = np.zeros((3, 3)); b = np.zeros(3)
    ids = np.unique(plate_id)
    for k in ids:
        m = plate_id == k
        Ik = a[m].sum()*np.eye(3)-np.einsum('i,ij,ik->jk', a[m], x[m], x[m])
        I += Ik; b += Ik@omega.get(int(k), np.zeros(3))
    w0 = np.linalg.solve(I, b)
    rows = []
    for k in ids:
        m = plate_id == k
        v = np.cross(omega.get(int(k), np.zeros(3))-w0, x[m])*g.radius*100
        rows.append((int(k), float(a[m].sum()/1e12), float(np.sqrt(np.average((v*v).sum(1), weights=a[m]))),
                     float(np.average(continental[m], weights=a[m]))))
    return w0, rows


def split(rows, min_area=10., cont=.45):
    big = [r for r in rows if r[1] > min_area]
    return sorted(r[2] for r in big if r[3] > cont), sorted(r[2] for r in big if r[3] <= cont)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshots', nargs='+', help='t<time>.npz files with t<time>-plates.json beside them')
    a = p.parse_args(argv)
    e = json.loads(EARTH.read_text())
    ec, eo = split([(r['id'], r['area_mkm2'], r['rms_speed_cm_yr'], r['continental_share']) for r in e['plates']])
    print('Earth (NNR), plates > 10 Mkm^2: continental ' + ', '.join(f'{v:.1f}' for v in ec) +
          '; oceanic ' + ', '.join(f'{v:.1f}' for v in eo) + ' cm/yr')
    gs = {}
    for f in a.snapshots:
        z = np.load(f)
        N = z['plate_id'].shape[0]; n = int(round(np.sqrt((N-2)/6)))
        g = gs.setdefault(n, CubeSphere(n))
        plates = json.loads(Path(str(f)[:-4]+'-plates.json').read_text())
        om = {int(r['id']): np.asarray(r['axis'], np.float64)*r['speed_cm_yr']*1e-2/g.radius for r in plates}
        w0, rows = nnr_speeds(g, z['plate_id'], om, z['continental'].astype(np.float64))
        pc, po = split(rows)
        print(f'{Path(f).parent.parent.name}/{Path(f).name}: net rotation {np.degrees(np.linalg.norm(w0)*1e6):.2f} deg/Myr; '
              'continental ' + ', '.join(f'{v:.1f}' for v in pc) + '; oceanic ' + ', '.join(f'{v:.1f}' for v in po) + ' cm/yr')


if __name__ == '__main__':
    main()
