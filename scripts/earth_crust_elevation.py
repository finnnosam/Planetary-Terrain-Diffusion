"""Earth: elevation of continental crust against its thickness (CRUST1.0 x ETOPO 2022 on the n = 217 graph), and
the planet's local Airy relation for the same thickness (eighteenth pass, test for the elevation model).

Each graph node takes the CRUST1.0 1-degree cell it lies in (total thickness incl. sediment, crust type); elevation is
the node mean of ETOPO 2022 (outputs/planet-reference/earth-n217.npz). Ice sheets excluded (earth_reference.ice_mask).
"""
import json
import re
from pathlib import Path

import numpy as np

from planet_layer.plates import PARAMS
from planet_layer.sphere_graph import CubeSphere
from scripts.earth_reference import ice_mask

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT/'data'/'crust1'
OCEANIC = {'A1', 'A0', 'B-', 'V1', 'Y3'}


def airy(T, S=0., p=PARAMS):
    """Planet's local Airy elevation for total thickness T (m) of which S (m) is sediment of density rho_sediment
    (S = 0: all of it crust of density rho_crust, as the plate layer books it now)."""
    rc, rm, rw, rs = p['rho_crust'], p['rho_mantle'], p['rho_water'], p['rho_sediment']
    dry = p['cont_ref_elev_m']+(T-p['cont_ref_thick_m'])*(rm-rc)/rm+S*(rc-rs)/rm
    return np.where(dry < 0, dry*rm/(rm-rw), dry)


def main():
    g = CubeSphere(217)
    z = np.load(ROOT/'outputs'/'planet-reference'/'earth-n217.npz')['elevation_m'].astype(np.float64)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2].reshape(180, 360)*1e3          # rows lat 89.5 -> -89.5, lon -179.5 -> 179.5
    types = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()])
    lat = np.degrees(np.arcsin(np.clip(g.xyz[:, 2], -1, 1))); lon = np.degrees(np.arctan2(g.xyz[:, 1], g.xyz[:, 0]))
    r = np.clip((89.5-lat+.5).astype(int), 0, 179); c = np.clip((lon+180).astype(int), 0, 359)
    S = np.loadtxt(C1/'sedthk.xyz')[:, 2].reshape(180, 360)*1e3
    Tn, tn, Sn = T[r, c], types[r, c], S[r, c]
    cont = ~np.isin(tn, list(OCEANIC)) & ~ice_mask(g)
    a = g.area
    bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 55), (55, 80)]
    out = {'source': 'CRUST1.0 total thickness and type x ETOPO 2022 node means, n = 217, ice excluded', 'bins': []}
    print(f"{'T (km)':>8} {'area':>6} {'Earth z p25/p50/p75 (m)':>24} {'sed km':>6} {'Airy all crust':>14} {'Airy + sed layer':>16} (node medians)")
    for lo, hi in bins:
        m = cont & (Tn >= lo*1e3) & (Tn < hi*1e3)
        if not m.any():
            continue
        o = np.argsort(z[m]); cw = np.cumsum(a[m][o]); cw /= cw[-1]
        q = [float(z[m][o][np.searchsorted(cw, k)]) for k in (.25, .5, .75)]
        Tm = float(np.average(Tn[m], weights=a[m]))
        wm = lambda v: float(v[m][np.argsort(v[m])][np.searchsorted(np.cumsum(a[m][np.argsort(v[m])])/a[m].sum(), .5)])
        rec = {'T_km': [lo, hi], 'area_frac': float(a[m].sum()/a.sum()), 'earth_z_p25_p50_p75_m': q,
               'mean_T_km': Tm/1e3, 'sediment_km_p50': wm(Sn)/1e3,
               'planet_airy_all_crust_m_p50': wm(airy(Tn)), 'planet_airy_with_sediment_layer_m_p50': wm(airy(Tn, Sn))}
        out['bins'].append(rec)
        print(f"{lo:>3}-{hi:<4} {rec['area_frac']:6.3f} {q[0]:7.0f} {q[1]:7.0f} {q[2]:7.0f} {rec['sediment_km_p50']:6.1f} "
              f"{rec['planet_airy_all_crust_m_p50']:14.0f} {rec['planet_airy_with_sediment_layer_m_p50']:16.0f}")
    (ROOT/'outputs'/'planet-reference').mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs'/'planet-reference'/'earth-crust-elevation-n217.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
