"""Earth's deforming crust on the n = 217 graph from GSRM v2.1 (Kreemer et al. 2014), and the strain-rate threshold
that reproduces Bird's (2003) orogens (twenty-fourth pass: the measured threshold for `orogen_rule` 'strain').

  python -m scripts.earth_deformation

GSRM_average_strain_v2.1.txt (data/gsrm; units 1e-9/yr): strain-rate tensor averaged over each deforming cell
(0.25 x 0.2 degrees, cell midpoints); cells not listed are rigid (zero). Every graph node takes the area-weighted mean
tensor of the cells whose midpoints are nearest to it (rigid area counts as zero). The planet's deformation proxy is
the rate of change of crustal thickness, |dT/dt| / T, which for incompressible crust equals the areal dilatation
|exx + eyy| (strike-slip registers in neither). The threshold is the one that best separates land inside Bird's
orogens from land outside them on Earth (largest Youden index, TPR - FPR), continental crust (CRUST1.0 total >= 20
km), ice excluded. Also printed: the second invariant for comparison, and land shares above the threshold.
Output: outputs/planet-reference/earth-deformation-n217.json (+ .npz with the node fields).
"""
import json
from pathlib import Path

import numpy as np

from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere
from scripts.earth_reference import ice_mask
from scripts.erosion_law_check import orogen_mask
from scripts.margin_width import C1, crust1_cell

ROOT = Path(__file__).resolve().parents[1]
GSRM = ROOT/'data'/'gsrm'/'GSRM_average_strain_v2.1.txt'
OUT = ROOT/'outputs'/'planet-reference'/'earth-deformation-n217'


def gsrm_nodes(g):
    """Node-mean exx, eyy, exy (1/yr) from GSRM's deforming cells (rigid area zero)."""
    d = np.loadtxt(GSRM, comments='#', usecols=(0, 1, 2, 3, 4))
    lat, lon = np.radians(d[:, 0]), np.radians(d[:, 1])
    xyz = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), -1)
    R = g.radius
    cell_a = (np.radians(.25)*R)*(np.radians(.2)*R)*np.cos(lat)
    nn = fields.nearest_nodes(g, xyz)
    out = [np.bincount(nn, weights=cell_a*d[:, k]*1e-9, minlength=g.size)/g.area for k in (2, 3, 4)]
    cover = np.minimum(np.bincount(nn, weights=cell_a, minlength=g.size)/g.area, 1.)
    return (*out, cover)


def main():
    g = CubeSphere(217); a = g.area
    exx, eyy, exy, cover = gsrm_nodes(g)
    dil = np.abs(exx+eyy)
    inv2 = np.sqrt(exx**2+eyy**2+2*exy**2)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2][crust1_cell(g)]*1e3
    e = np.load(ROOT/'outputs'/'planet-reference'/'earth-n217.npz')['elevation_m'].astype(float)
    oro = orogen_mask(g, ROOT/'data'/'tectonicplates-master'/'GeoJSON'/'PB2002_orogens.json')
    land = (e > 0) & (T >= 20e3) & ~ice_mask(g)
    hi = land & (e > 2000)
    res = {'source': 'GSRM v2.1 average strain (Kreemer et al. 2014) on n = 217; Bird 2003 orogens; CRUST1.0; ETOPO 2022',
           'deforming_cover_of_surface': float((a*cover).sum()/a.sum()),
           'deforming_cover_of_land': float((a*cover)[land].sum()/a[land].sum())}
    for name, f in (('dilatation', dil), ('second_invariant', inv2)):
        cs = np.logspace(-10, -6.5, 71)
        tpr = np.array([a[land & oro & (f > c)].sum()/a[land & oro].sum() for c in cs])
        fpr = np.array([a[land & ~oro & (f > c)].sum()/a[land & ~oro].sum() for c in cs])
        k = int(np.argmax(tpr-fpr)); c = float(cs[k])
        # equal-area threshold: the land flagged equals Bird's orogen land (the area the erosion law's erodibility-1
        # class covered when it was calibrated); the adopted criterion, fixed before reading the planet with it
        ce = np.logspace(-10, -6.5, 701)
        flagged = np.array([a[land & (f > x)].sum() for x in ce]); target = a[land & oro].sum()
        ke = int(np.argmin(np.abs(flagged-target))); c_eq = float(ce[ke])
        res.setdefault('equal_area', {})[name] = {
            'threshold_per_yr': c_eq, 'tpr': float(a[land & oro & (f > c_eq)].sum()/a[land & oro].sum()),
            'fpr': float(a[land & ~oro & (f > c_eq)].sum()/a[land & ~oro].sum()),
            'land_gt2km_above': float(a[hi & (f > c_eq)].sum()/a[hi].sum())}
        r = res['equal_area'][name]
        print(f"{name}: equal-area threshold {c_eq*1e9:.2f} nanostrain/yr (flags {flagged[ke]/a[land].sum():.3f} of land = Bird's "
              f"{target/a[land].sum():.3f}; TPR {r['tpr']:.2f}, FPR {r['fpr']:.2f}, land > 2 km flagged {r['land_gt2km_above']:.2f})")
        q = {p: float(np.percentile(f[land & oro], p)) for p in (10, 25, 50, 75, 90)}
        qo = {p: float(np.percentile(f[land & ~oro], p)) for p in (50, 75, 90, 95)}
        res[name] = {'threshold_per_yr': c, 'tpr': float(tpr[k]), 'fpr': float(fpr[k]),
                     'land_above': float(a[land & (f > c)].sum()/a[land].sum()),
                     'land_gt2km_above': float(a[hi & (f > c)].sum()/a[hi].sum()),
                     'in_orogens_land_p10_25_50_75_90': q, 'outside_orogens_land_p50_75_90_95': qo,
                     'curve': {f'{cc:.3g}': [float(t), float(fp)] for cc, t, fp in zip(cs[::5], tpr[::5], fpr[::5])}}
        print(f"{name}: best threshold {c*1e9:.2f} nanostrain/yr (TPR {tpr[k]:.2f}, FPR {fpr[k]:.2f}); land above "
              f"{res[name]['land_above']:.2f}, land > 2 km above {res[name]['land_gt2km_above']:.2f}\n"
              f"    inside orogens (land) p10/25/50/75/90: " + ' / '.join(f"{v*1e9:.1f}" for v in q.values())
              + "; outside p50/75/90/95: " + ' / '.join(f"{v*1e9:.1f}" for v in qo.values()))
        print('    curve (threshold: TPR / FPR): ' + ', '.join(f"{cc*1e9:.2g}: {t:.2f}/{fp:.2f}" for cc, t, fp in zip(cs[::5], tpr[::5], fpr[::5])))
    print(f"GSRM deforming cells cover {res['deforming_cover_of_surface']:.3f} of the surface, {res['deforming_cover_of_land']:.3f} of continental land")
    print(f"Bird orogens: {a[land & oro].sum()/a[land].sum():.2f} of land, {a[hi & oro].sum()/a[hi].sum():.2f} of land > 2 km (ice excluded)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix('.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT.with_suffix('.npz'), exx=exx.astype(np.float32), eyy=eyy.astype(np.float32),
                        exy=exy.astype(np.float32), cover=cover.astype(np.float32))
    return res


if __name__ == '__main__':
    main()
