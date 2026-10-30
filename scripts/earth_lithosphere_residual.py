"""Earth: how much of continental elevation the crust explains, and what the mantle lithosphere adds (thirty-fourth
pass, Earth reference for item C, thickness-elevation from the lithosphere's thermal state).

Each n = 217 graph node takes its CRUST1.0 1-degree cell (total thickness, sediment thickness, crust type) and its
ETOPO 2022 node mean elevation (ice excluded). The crystalline crust is split into upper / middle / lower layers by the
type's thickness shares, each with the type's density (CNtype1_key.txt, Laske et al. 2013); sediment at
rho_sediment. The crust's Airy height above the asthenosphere level is

    e_c = sum_i h_i (rho_a - rho_i) / rho_a        (rho_a = PARAMS rho_mantle)

and the node's elevation, unloaded of water where it is below sea level, z_dry = z (z >= 0), z (rho_a - rho_w) / rho_a
(z < 0). The residual r = z_dry - e_c is what the column below the crust adds (mantle lithosphere: thermal and
compositional density; plus dynamic topography and flexure). Ocean crust 3 Myr and younger (type A0) has almost no
mantle lithosphere, so r(A0) is the asthenosphere level and dL = r - r(A0) is the mantle lithosphere's own
contribution (negative = it pulls the surface down).
"""
import json
from pathlib import Path

import numpy as np

from planet_layer.plates import PARAMS
from planet_layer.sphere_graph import CubeSphere
from scripts.earth_reference import ice_mask
from scripts.erosion_law_check import orogen_mask

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT/'data'/'crust1'
OCEANIC = {'A1', 'A0', 'B-', 'V1', 'Y3'}
GROUPS = {
    'Archean (G1 G2 F)': {'G1', 'G2', 'F-'},
    'Proterozoic (H1 H2 I1 I2)': {'H1', 'H2', 'I1', 'I2'},
    'platform (D E)': {'D-', 'E-'},
    'Phanerozoic (Z1 Z2)': {'Z1', 'Z2'},
    'orogen (O P Q R1 R2)': {'O-', 'P-', 'Q-', 'R1', 'R2'},
    'arc / forearc (J K L1 L2)': {'J-', 'K-', 'L1', 'L2'},
    'extended / rift (M N X V2)': {'M-', 'N-', 'X-', 'V2'},
    'margin / shelf (T U C S)': {'T-', 'U-', 'C-', 'S-'},
}


def type_key():
    """{code: (densities of upper, middle, lower crust in kg/m^3, thickness shares)} from CNtype1_key.txt."""
    lines = [l.split() for l in (C1/'CNtype1_key.txt').read_text().splitlines()]
    out = {}
    for i, l in enumerate(lines):
        if len(l) >= 2 and l[1].rstrip(':').isdigit() and len(l[0]) == 2:
            d = [float(v)*1e3 for v in lines[i+3][:3]]
            t = np.array([float(v.rstrip('.')) for v in lines[i+4][:3]])
            out[l[0]] = (np.array(d), t/t.sum())
    return out


def main():
    p = PARAMS
    ra, rw, rs = p['rho_mantle'], p['rho_water'], p['rho_sediment']
    g = CubeSphere(217)
    z = np.load(ROOT/'outputs'/'planet-reference'/'earth-n217.npz')['elevation_m'].astype(np.float64)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2].reshape(180, 360)*1e3
    S = np.loadtxt(C1/'sedthk.xyz')[:, 2].reshape(180, 360)*1e3
    types = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()])
    lat = np.degrees(np.arcsin(np.clip(g.xyz[:, 2], -1, 1))); lon = np.degrees(np.arctan2(g.xyz[:, 1], g.xyz[:, 0]))
    r = np.clip((89.5-lat+.5).astype(int), 0, 179); c = np.clip((lon+180).astype(int), 0, 359)
    Tn, Sn, tn = T[r, c], S[r, c], types[r, c]
    key = type_key()
    dens = np.array([key[t][0] for t in tn]); share = np.array([key[t][1] for t in tn])
    h = (Tn-Sn)[:, None]*share
    ec = (Sn*(ra-rs)+(h*(ra-dens)).sum(1))/ra
    rho_cryst = (h*dens).sum(1)/np.maximum(h.sum(1), 1.)
    zd = np.where(z < 0, z*(ra-rw)/ra, z)
    res = zd-ec
    ice = ice_mask(g); a = g.area
    oro = orogen_mask(g, ROOT/'data'/'tectonicplates-master'/'GeoJSON'/'PB2002_orogens.json')

    def wq(v, m, q=(.25, .5, .75)):
        o = np.argsort(v[m]); cw = np.cumsum(a[m][o]); cw /= cw[-1]
        return [float(v[m][o][min(np.searchsorted(cw, k), len(cw)-1)]) for k in q]

    a0 = (tn == 'A0') & ~ice
    r0 = wq(res, a0)[1]
    out = {'source': 'CRUST1.0 (thickness, sediment, type; type layer densities) x ETOPO 2022, n = 217, ice excluded',
           'rho_a': ra, 'rho_sediment': rs, 'r_A0_m': r0, 'groups': {}, 'bins': [], 'bins_outside_orogens': [],
           'bins_inside_orogens': []}
    print(f"asthenosphere level: residual of ocean crust <= 3 Myr (A0) p50 {r0:.0f} m; normal ocean (A1) "
          f"p50 {wq(res, (tn == 'A1') & ~ice)[1]-r0:+.0f} m relative to it")
    cont = ~np.isin(tn, list(OCEANIC)) & ~ice & (Tn >= 20e3)
    print(f"\ncontinental crust (total >= 20 km), by CRUST1.0 type group: dL = residual - r(A0), the mantle lithosphere's"
          f" part (m; p25 / p50 / p75)")
    print(f"{'group':28} {'area':>6} {'T km':>5} {'rho_cr':>6} {'z p50':>6} {'e_c p50':>7} {'dL p25 / p50 / p75':>20}")
    for name, codes in GROUPS.items():
        m = cont & np.isin(tn, list(codes))
        if not m.any():
            continue
        q = [v-r0 for v in wq(res, m)]
        rec = {'area_frac_of_continent': float(a[m].sum()/a[cont].sum()), 'T_km_p50': wq(Tn, m)[1]/1e3,
               'rho_crystalline_p50': wq(rho_cryst, m)[1], 'z_p50_m': wq(z, m)[1], 'e_c_p50_m': wq(ec, m)[1],
               'dL_p25_p50_p75_m': q}
        out['groups'][name] = rec
        print(f"{name:28} {rec['area_frac_of_continent']:6.3f} {rec['T_km_p50']:5.1f} {rec['rho_crystalline_p50']:6.0f} "
              f"{rec['z_p50_m']:6.0f} {rec['e_c_p50_m']:7.0f} {q[0]:6.0f} / {q[1]:5.0f} / {q[2]:5.0f}")
    bins = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 80)]
    for tag, mm in (('bins', cont), ('bins_outside_orogens', cont & ~oro), ('bins_inside_orogens', cont & oro)):
        print(f"\n{tag}: by total thickness (m; node p50): Earth z, crust-only Airy e_c + r(A0) (CRUST1.0 densities), "
              f"planet Airy (rho_crust {p['rho_crust']:.0f}), dL")
        for lo, hi in bins:
            m = mm & (Tn >= lo*1e3) & (Tn < hi*1e3)
            if m.sum() < 20:
                continue
            zz = wq(z, m)[1]; e = wq(ec+r0, m)[1]
            pa = p['cont_ref_elev_m']+(wq(Tn, m)[1]-p['cont_ref_thick_m'])*(ra-p['rho_crust'])/ra
            dl = wq(res, m); dl = [v-r0 for v in dl]
            rec = {'T_km': [lo, hi], 'area_frac_of_continent': float(a[m].sum()/a[cont].sum()), 'z_p50_m': zz,
                   'crust_airy_p50_m': e, 'planet_airy_m': pa, 'dL_p25_p50_p75_m': dl,
                   'rho_crystalline_p50': wq(rho_cryst, m)[1]}
            out[tag].append(rec)
            print(f"  {lo:>2}-{hi:<3} area {rec['area_frac_of_continent']:.3f}  z {zz:6.0f}  crust Airy {e:6.0f} (dry)  "
                  f"planet Airy {pa:6.0f}  rho_cr {rec['rho_crystalline_p50']:.0f}  dL {dl[0]:6.0f} / {dl[1]:6.0f} / {dl[2]:6.0f}")
    (ROOT/'outputs'/'planet-reference'/'earth-lithosphere-residual-n217.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
