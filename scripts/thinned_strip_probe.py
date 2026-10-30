"""Thinned continental crust vs its own Airy depth, and the young ocean beside it (eighteenth pass)."""
import json
import sys

import numpy as np

from planet_layer import graph_landscape as gl, plates as P
from planet_layer.sphere_graph import CubeSphere


def main():
    seed, years, every, rift = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    params = dict(erosion_K_factor=1.0, erosion_theta=200., erosion_Sc=0.036, margin_shortening=True,
                  sediment_accretion=0.13)
    params.update(dict(breakup_thick_m=10000., rift_width_km=100.) if rift == 'earth' else
                  dict(breakup_thick_m=17500., rift_width_km=0.))
    g = CubeSphere(217); a = g.area
    sim = P.PlateSim(g, seed=seed, t0=-250., params=params)
    p = sim.p
    rc, rm, rw = p['rho_crust'], p['rho_mantle'], p['rho_water']
    out = []
    for i in range(years):
        sim.step()
        if (i+1) % every:
            continue
        w = sim.world; z = sim.elevation(); cont = w['cont']; T = w['thick']
        # local Airy elevation of a continental column (water-loaded below sea level), on the same datum as z
        dry = p['cont_ref_elev_m']+(T-p['cont_ref_thick_m'])*(rm-rc)/rm
        airy = np.where(dry < 0, dry*rm/(rm-rw), dry)+sim.dyn-sim.sea_level
        age = sim.t-w['born']
        ridge = -P.gdh1_depth(age)+(T-p['ocean_thick_m'])+w['sed']+sim.dyn-sim.sea_level
        rec = {'t': sim.t}
        for lo, hi in ((0, 20e3), (20e3, 30e3), (30e3, 36e3)):
            m = cont & (T >= lo) & (T < hi)
            if m.any():
                rec[f'cont_{lo/1e3:.0f}-{hi/1e3:.0f}km'] = {
                    'area_frac': float(a[m].sum()/a.sum()),
                    'z_minus_own_airy_m_p10_50_90': [float(np.percentile((z-airy)[m], q)) for q in (10, 50, 90)],
                    'z_m_p50': float(np.median(z[m])), 'own_airy_m_p50': float(np.median(airy[m]))}
        d = gl.distance_to(cont, g.nbr, g.dist, 300e3)
        for lo, hi in ((0, 100e3), (100e3, 200e3), (200e3, 300e3)):
            m = (~cont) & (d >= lo) & (d < hi) & (age < 20)
            if m.any():
                rec[f'young_ocean_{lo/1e3:.0f}-{hi/1e3:.0f}km_from_coast'] = {
                    'area_frac': float(a[m].sum()/a.sum()),
                    'z_minus_own_m_p10_50_90': [float(np.percentile((z-ridge)[m], q)) for q in (10, 50, 90)],
                    'share_above_sea': float(a[m & (z > 0)].sum()/a[m].sum())}
        out.append(rec); print(json.dumps(rec), flush=True)
    json.dump(out, open(f'thinned_strip_{rift}_{seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
