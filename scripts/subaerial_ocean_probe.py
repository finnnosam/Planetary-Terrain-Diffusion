"""Which oceanic nodes stand above sea level, and why (eighteenth pass ledger check).

For each sample time: oceanic nodes with z > 0, their area, age, own crust anomaly and sediment, their elevation
without the flexural share of other nodes' loads, and the nearby continental crust.
"""
import json
import sys

import numpy as np

from planet_layer import graph_landscape as gl, plates as P
from planet_layer.sphere_graph import CubeSphere


def parts(sim):
    p, w = sim.p, sim.world
    thick, sed, cont = w['thick'], w['sed'], w['cont']
    rc, rm, rw, rs = p['rho_crust'], p['rho_mantle'], p['rho_water'], p['rho_sediment']
    anomaly = np.where(cont, thick-p['cont_ref_thick_m'], thick-p['ocean_thick_m'])
    age = np.where(cont, 0., sim.t-w['born'])
    s0 = getattr(sim, 'sea_level', 0.)
    above = (-P.gdh1_depth(age)+anomaly+sed) > s0
    dry = np.where(cont, rc*anomaly, np.where(above, rc*anomaly+rs*sed, 0.))
    wet = np.where(cont | above, 0., (rc-rw)*anomaly+(rs-rw)*sed)
    d_cont = sim.flex_dry.deflection(np.where(cont, dry, 0.))+sim.flex_wet.deflection(np.where(cont, wet, 0.))
    d_ocean = sim.flex_dry.deflection(np.where(cont, 0., dry))+sim.flex_wet.deflection(np.where(cont, 0., wet))
    return anomaly, age, d_cont, d_ocean


def main():
    seed, years, every = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    rift = sys.argv[4] if len(sys.argv) > 4 else 'earth'
    params = dict(erosion_K_factor=1.0, erosion_theta=200., erosion_Sc=0.036, margin_shortening=True,
                  sediment_accretion=0.13)
    params.update(dict(breakup_thick_m=10000., rift_width_km=100.) if rift == 'earth' else
                  dict(breakup_thick_m=17500., rift_width_km=0.))
    g = CubeSphere(217)
    a = g.area
    sim = P.PlateSim(g, seed=seed, t0=-250., params=params)
    out = []
    for i in range(int(years)):
        sim.step()
        if (i+1) % every:
            continue
        w = sim.world
        z = sim.elevation()
        cont = w['cont']
        anomaly, age, d_cont, d_ocean = parts(sim)
        sub = (~cont) & (z > 0)
        d = gl.distance_to(cont, g.nbr, g.dist, 400e3)
        near = (d < 200e3)
        # continental crust within 150 km of each subaerial ocean node: mean thickness
        rec = {'t': sim.t, 'sea_level_m': float(sim.sea_level),
               'ocean_subaerial_area_frac': float(a[sub].sum()/a.sum()),
               'ocean_area_frac': float(a[~cont].sum()/a.sum())}
        if sub.any():
            wt = a[sub]
            q = lambda v: [float(np.percentile(v[sub], k)) for k in (10, 50, 90)]
            rec.update({
                'age_myr_p10_50_90': q(age), 'own_anomaly_m': q(anomaly), 'sed_m': q(w['sed']),
                'z_m': q(z), 'uplift_from_continent_loads_m': q(-d_cont), 'own_and_ocean_flexure_m': q(-d_ocean),
                'z_without_continent_flexure_m': q(z+d_cont),
                'share_within_200km_of_continent': float(wt[near[sub]].sum()/wt.sum()),
                'share_still_subaerial_without_continent_flexure': float(wt[(z+d_cont)[sub] > 0].sum()/wt.sum()),
                'share_young_lt10myr': float(wt[age[sub] < 10].sum()/wt.sum()),
                'share_anomaly_gt3km': float(wt[anomaly[sub] > 3000].sum()/wt.sum()),
            })
        # the continental side: area of thinned continental crust (< 30 km)
        rec['cont_lt30_area_frac'] = float(a[cont & (w['thick'] < 30e3)].sum()/a.sum())
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(f'subaerial_ocean_{rift}_{seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
