"""python -m scripts.ledger_probe earth 1001 250   (or: fitted 1001 250); writes ledger_probe_<label>_<seed>.json

Ledger probe (eighteenth pass): where does the sediment on submerged continental crust come from?

Runs the plate layer with an instrumented copy of PlateSim._erode that also books, per plate step:
  total   sum(dz * a) over all nodes (surface_step is closed: should be ~0)
  cont    continental net change (crust)
  ocean_raw / ocean_booked   ocean-floor net change before / after the clamp dz >= -sed
  unbooked = ocean_booked - ocean_raw  (ocean floor cut below its sediment: removed from z, never charged)
split by the node's start depth (z0 > 0, -1000 < z0 <= 0, z0 <= -1000).
"""
import json
import sys
import time

import numpy as np

from planet_layer import graph_landscape as gl, plates as P, shelf as shelf_mod
from planet_layer.sphere_graph import CubeSphere

BINS = (('subaerial', lambda z: z > 0), ('shallow_sea', lambda z: (z <= 0) & (z > -1000)), ('deep', lambda z: z <= -1000))


def probe_erode(self, thick):
    g, p, w = self.g, self.p, self.world
    z0 = self.elevation(thick, w['sed'])
    lp = {**gl.DEFAULTS, 'K': gl.DEFAULTS['K']*p['erosion_K_factor'], 'Sc': p['erosion_Sc'],
          'Kd': gl.DEFAULTS['Kd']*p['erosion_Kd_factor'], 'open_edges': False, 'theta': p['erosion_theta']}
    since = np.where(np.isnan(w['orog']), np.inf, self.t-w['orog'])
    erod = p['stable_erodibility']+(1-p['stable_erodibility'])*np.exp(-np.clip(since, 0, None)/p['orogen_erodibility_tau_myr'])
    K = lp['K']*np.where(w['cont'], erod, 1.)
    seed = z0 < -1500.
    dt = p['dt_myr']*1e6/p['erosion_substeps']
    z = z0.copy()
    fallback = np.ones(g.size, bool)
    shelf_on = p['shelf_kappa_m2_yr'] > 0
    keep = p['keep_deep_sediment']
    deep = np.zeros(g.size)
    proc = {'surface_step': np.zeros(g.size), 'fill_spill': np.zeros(g.size), 'shelf': np.zeros(g.size)}
    for _ in range(p['erosion_substeps']):
        za = z.copy()
        r = gl.surface_step(g, z, seed, None, fallback, K, lp, dt, self.h_min, self.bound, keep_deep=keep)
        z = r['z']
        proc['surface_step'] += z-za          # fluvial, marine, hillslope and marine diffusion together
        if keep and r['deep_m3'].any():
            deep += r['deep_m3']
            zb = z.copy()
            z = gl.fill_and_spill(z, r['deep_m3'], g.nbr, g.area)
            proc['fill_spill'] += z-zb
        if shelf_on:
            zb = z.copy()
            ocean = gl.connected_ocean(z, seed, g.nbr)
            avail = np.where(w['cont'], 1e9, np.clip(w['sed']+z-z0, 0, None))
            z, _, off = shelf_mod.shelf_step(g, z, ocean, avail, p['shelf_kappa_m2_yr'], p['shelf_wave_base_m'],
                                             p['cliff_retreat_m_yr'], dt)
            proc['shelf'] += z-zb
            self.erosion_totals['shelf_off_land_m3'] = self.erosion_totals.get('shelf_off_land_m3', 0.)+off
    self.erosion_totals['deep_sea_sediment_m3'] = self.erosion_totals.get('deep_sea_sediment_m3', 0.)+float(deep.sum())
    dz = z-z0
    cont = w['cont']
    e_cont = np.where(cont, dz, 0.)
    dsed = np.where(cont, 0., np.maximum(dz, -w['sed']))
    a = g.area
    self.erosion_totals['continental_eroded_m3'] += float((np.minimum(e_cont, 0)*a).sum())
    self.erosion_totals['continental_deposited_m3'] += float((np.maximum(e_cont, 0)*a).sum())
    self.erosion_totals['continental_deposited_submerged_m3'] = self.erosion_totals.get(
        'continental_deposited_submerged_m3', 0.)+float((np.maximum(e_cont, 0)*a)[z0 <= 0].sum())
    self.erosion_totals['ocean_sediment_m3'] += float((dsed*a).sum())
    # --- probe
    oc = ~cont
    unb = np.where(oc, dsed-dz, 0.)*a
    rec = {'t': self.t, 'total': float((dz*a).sum()), 'cont': float((e_cont*a).sum()),
           'cont_dep_sub': float((np.maximum(e_cont, 0)*a)[z0 <= 0].sum()),
           'cont_ero': float((np.minimum(e_cont, 0)*a).sum()),
           'ocean_raw': float((np.where(oc, dz, 0)*a).sum()), 'ocean_booked': float((dsed*a).sum()),
           'unbooked': float(unb.sum()), 'deep_m3': float(deep.sum())}
    for name, f in BINS:
        m = f(z0)
        rec['unbooked_'+name] = float(unb[m].sum())
        rec['cont_dep_sub_'+name] = float((np.maximum(e_cont, 0)*a)[(z0 <= 0) & m].sum())
    # which process lowered the ocean floor below its sediment: each process's ocean-node erosion where the node ends
    # cut below its sediment
    cut = oc & (dz < -w['sed'])
    for k, v in proc.items():
        rec['ocean_cut_by_'+k] = float((np.minimum(v, 0)*a)[cut].sum())
        rec['cont_gain_by_'+k] = float((v*a)[cont].sum())
        rec['ocean_net_by_'+k] = float((v*a)[oc].sum())
    rec['ocean_cut_nodes'] = int(cut.sum())
    rec['ocean_cut_area_frac'] = float(a[cut].sum()/a.sum())
    self.probe.append(rec)
    return e_cont, dsed


def main():
    label, seed, years = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    params = dict(erosion_K_factor=1.0, erosion_theta=200., erosion_Sc=0.036, margin_shortening=True,
                  sediment_accretion=0.13)
    if label == 'earth':
        params.update(breakup_thick_m=10000., rift_width_km=100.)
    else:
        params.update(breakup_thick_m=17500., rift_width_km=0.)
    P.PlateSim._erode = probe_erode
    g = CubeSphere(217)
    sim = P.PlateSim(g, seed=seed, t0=-years, params=params)
    sim.probe = []
    c = time.time()
    n = int(round(years/sim.p['dt_myr']))
    for i in range(n):
        sim.step()
        if (i+1) % 25 == 0:
            print(label, f'{sim.t:+.0f} Myr', f'{time.time()-c:.0f} s', flush=True)
            json.dump({'params': params, 'seed': seed, 'probe': sim.probe, 'totals': sim.erosion_totals},
                      open(f'ledger_probe_{label}_{seed}.json', 'w'))
    json.dump({'params': params, 'seed': seed, 'probe': sim.probe, 'totals': sim.erosion_totals},
              open(f'ledger_probe_{label}_{seed}.json', 'w'))
    print('done', label, time.time()-c)


if __name__ == '__main__':
    main()
