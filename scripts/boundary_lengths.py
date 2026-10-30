"""True line length of plate boundaries by type, for the plate layer and Bird 2003 (docs/planet-layer-plan.md, step 8).

The plate layer's own figures count boundary nodes (a node with a neighbour on another plate) and halve them. With
8 neighbours a diagonal boundary has two nodes per side per diagonal step, so the count overstates a line by up to
sqrt(2) depending on its direction (great circles at random orientations: 1.24-1.38x). Here each plate's mask,
smoothed by a Gaussian of 1 cell on every face chart (marching squares on the raw 0/1 mask still zigzags: +5-9%), is
contoured at 0.5 (marching squares, numpy only); the contour vertices are mapped to the sphere and the great-circle segment lengths
summed. Each boundary is traced once from each side, so the sum over plates is halved. Each segment is typed from
the relative motion of the plates on its two sides (the snapshot's t<time>-plates.json), see lengths().

Check (tests/test_boundary_lengths.py): a plate split along a great circle, at several orientations, measures 2 pi R
within 2%; the node count is reported beside it.

  python -m scripts.boundary_lengths outputs\\planet-calibration\\sweep-...\\seed1001\\snapshots\\t+0000.0.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_diffusion import cube
from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere

SMOOTH_CELLS = 1.   # great circles at n = 217: within 0.0-0.8% of 2 pi R (0.7 cells: +1-2%, none: +5-9%)
TYPES = {1: 'ridge', 2: 'subduction', 3: 'collision', 4: 'transform'}
# Bird 2003 (PB2002_steps), km; computed from data/tectonicplates-master on 22 Oct (plan doc, "Data added by the user")
EARTH_KM = {'ridge': 94800., 'subduction': 51300., 'collision_and_convergent': 40500., 'transform': 73900.}


def _smooth(a, sigma):
    """Separable Gaussian filter (edge values repeated), numpy only."""
    r = int(np.ceil(4*sigma))
    k = np.exp(-.5*(np.arange(-r, r+1)/sigma)**2); k /= k.sum()
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[ax] = (r, r)
        b = np.pad(a, pad, mode='edge')
        a = sum(k[i]*np.take(b, np.arange(i, i+a.shape[ax]), axis=ax) for i in range(2*r+1))
    return a


def _march(f, level=.5):
    """Marching squares on a 2-D array: (M, 2, 2) segment end points in (row, col), unordered. Saddles are split by
    the cell-centre mean."""
    v00, v01, v10, v11 = f[:-1, :-1], f[:-1, 1:], f[1:, :-1], f[1:, 1:]
    r, c = np.mgrid[:f.shape[0]-1, :f.shape[1]-1].astype(np.float64)
    def cross(va, vb):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.clip((level-va)/(vb-va), 0, 1)
    # edges: top (00-01), right (01-11), bottom (10-11), left (00-10)
    pts = [np.stack((r, c+cross(v00, v01)), -1), np.stack((r+cross(v01, v11), c+1), -1),
           np.stack((r+1, c+cross(v10, v11)), -1), np.stack((r+cross(v00, v10), c), -1)]
    hit = [(v00 > level) != (v01 > level), (v01 > level) != (v11 > level), (v10 > level) != (v11 > level),
           (v00 > level) != (v10 > level)]
    n = sum(h.astype(int) for h in hit)
    out = []
    two = n == 2
    idx = [np.flatnonzero(h[two]) for h in hit]
    sel = np.stack([h[two] for h in hit], -1)                    # (M, 4) which edges are crossed
    P = np.stack([q[two] for q in pts], 1)                       # (M, 4, 2)
    order = np.argsort(~sel, axis=1, kind='stable')[:, :2]       # the two crossed edges
    out.append(np.take_along_axis(P, order[:, :, None], 1))
    four = n == 4
    if four.any():
        P4 = np.stack([q[four] for q in pts], 1)
        centre = (v00+v01+v10+v11)[four]/4 > level
        up = (v00[four] > level) == centre                       # corner 00 joins the centre: cut corners 01 and 10
        a = np.where(up[:, None, None], P4[:, [0, 1]], P4[:, [0, 3]])
        b = np.where(up[:, None, None], P4[:, [2, 3]], P4[:, [1, 2]])
        out += [a, b]
    return np.concatenate(out)


def segments(g, plate_id):
    """All boundary segments: (M, 2, 3) unit-vector end points, each boundary traced once from each side."""
    n = g.n
    ids = g.to_cube(np.asarray(plate_id))
    out = []
    for f in range(6):
        chart = ids[f]
        for k in np.unique(chart):
            s = _march(_smooth((chart == k).astype(np.float64), SMOOTH_CELLS))
            if len(s):
                out.append(cube.directions(f, s[..., 0], s[..., 1], n))
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def arc_len(seg, radius):
    a, b = seg[:, 0], seg[:, 1]
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))*radius


def typed_segments(g, plate_id, plates, continental=None, vmin_mm_yr=5., seg=None):
    """Every boundary segment typed by the plate layer's rule (see lengths()): per-segment arrays L (km, halved),
    typ (0 untyped, 1 ridge, 2 subduction, 3 collision, 4 transform), conv (closing rate, mm/yr), vt (shear rate),
    mid (unit vectors), na / nb (nodes 0.75 cell to each side), ka / kb (their plate ids), cont (the continental
    flags used)."""
    if seg is None:
        seg = segments(g, plate_id)
    L = arc_len(seg, g.radius)/2/1e3
    mid = seg.sum(1); mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    h = np.sqrt(g.area.mean())/1e3
    pid = np.asarray(plate_id)
    tang = seg[:, 1]-seg[:, 0]
    nrm = np.cross(mid, tang); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-15)
    off = .75*h*1e3/g.radius                                          # 0.75 cell to each side
    pa, pb = mid+off*nrm, mid-off*nrm
    pa /= np.linalg.norm(pa, axis=1, keepdims=True); pb /= np.linalg.norm(pb, axis=1, keepdims=True)
    na, nb = fields.nearest_nodes(g, pa), fields.nearest_nodes(g, pb)
    ka, kb = pid[na], pid[nb]
    om = {int(r['id']): np.asarray(r['axis'], np.float64)*r['speed_cm_yr']*1e-2/g.radius for r in plates}
    wa = np.stack([om.get(int(k), np.zeros(3)) for k in ka]); wb = np.stack([om.get(int(k), np.zeros(3)) for k in kb])
    v = np.cross(wa-wb, mid)*g.radius*1e3                             # mm/yr, plate a relative to plate b
    conv = -np.einsum('ij,ij->i', v, nrm)                             # a lies along +nrm: moving toward b is -nrm
    vt = np.sqrt(np.maximum((v*v).sum(1)-conv**2, 0.))
    same = ka == kb                                                   # a junction or a sliver: no type
    cont = np.asarray(continental, bool) if continental is not None else np.zeros(g.size, bool)
    typ = np.where(np.abs(conv) < np.maximum(vmin_mm_yr, .5*vt), 4, np.where(conv < 0, 1, np.where(cont[na] & cont[nb], 3, 2)))
    typ = np.where(same, 0, typ)
    return {'L': L, 'typ': typ, 'conv': conv, 'vt': vt, 'mid': mid, 'na': na, 'nb': nb, 'ka': ka, 'kb': kb, 'cont': cont}


def lengths(g, plate_id, boundary_type=None, convergence=None, plates=None, continental=None, vmin_mm_yr=5.):
    """Boundary length (km) in total and by type. With `plates` (the snapshot's plate table: id, axis, speed_cm_yr)
    each segment is typed from the two plates' relative motion across it, by the plate layer's own rule
    (planet_layer.plates.classify_boundaries): transform if the closing or opening rate is under max(vmin, half the
    shear rate), else ridge (opening) or convergent: collision if both sides are continental, subduction otherwise.
    Without it, a segment takes the node type nearest to its midpoint (biased: a node takes the type of its
    fastest-closing neighbour, diagonals included, so transforms are undercounted)."""
    seg = segments(g, plate_id)
    L = arc_len(seg, g.radius)/2/1e3                                      # km, halved: each line traced twice
    mid = seg.sum(1); mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    out = {'total_km': float(L.sum())}
    h = np.sqrt(g.area.mean())/1e3
    if boundary_type is not None:
        for code, name in TYPES.items():
            out[f'{name}_km_node_count'] = float((np.asarray(boundary_type) == code).sum()*h/2)
    if plates is not None:
        t = typed_segments(g, plate_id, plates, continental, vmin_mm_yr, seg=seg)
        L, typ, conv, vt, cont, na, nb = t['L'], t['typ'], t['conv'], t['vt'], t['cont'], t['na'], t['nb']
        for code, name in TYPES.items():
            out[f'{name}_km'] = float(L[typ == code].sum())
        out['untyped_km'] = float(L[typ == 0].sum())
        # subduction by the crust on its sides (item B, thirty-fifth pass): continental crust on either side (a
        # continental margin or an arc on continental crust) or oceanic on both (intra-oceanic arcs)
        ocean_ocean = ~(cont[na] | cont[nb])
        out['subduction_continental_km'] = float(L[(typ == 2) & ~ocean_ocean].sum())
        out['subduction_oceanic_km'] = float(L[(typ == 2) & ocean_ocean].sum())
        for code, name, val in ((1, 'ridge', -conv), (2, 'subduction', conv), (4, 'transform', vt)):
            m = typ == code
            if m.any():
                o = np.argsort(val[m]); c = np.cumsum(L[m][o])
                out[f'{name}_rate_mm_yr_length_median'] = float(val[m][o][np.searchsorted(c, c[-1]/2)])
    elif boundary_type is not None:
        bt = np.asarray(boundary_type)[fields.nearest_nodes(g, mid)]
        for code, name in TYPES.items():
            out[f'{name}_km'] = float(L[bt == code].sum())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshots', nargs='+', help='plate snapshot files (t<time>.npz)')
    a = p.parse_args(argv)
    res = {}
    gs = {}
    for f in a.snapshots:
        z = np.load(f)
        N = z['plate_id'].shape[0]; n = int(round(np.sqrt((N-2)/6)))
        g = gs.setdefault(n, CubeSphere(n))
        pj = Path(str(f)[:-4]+'-plates.json')
        plates = json.loads(pj.read_text()) if pj.exists() else None
        r = lengths(g, z['plate_id'], z['boundary_type'], plates=plates, continental=z['continental'])
        res[str(f)] = r
        print(Path(f).name, ', '.join(f'{k} {v:,.0f}' for k, v in r.items()), flush=True)
    print('Earth (Bird 2003):', ', '.join(f'{k} {v:,.0f} km' for k, v in EARTH_KM.items()))
    return res


if __name__ == '__main__':
    main()
