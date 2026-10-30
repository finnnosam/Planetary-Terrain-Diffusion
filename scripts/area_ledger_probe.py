"""Continental area ledger by mechanism (eighteenth pass): why thinning_local_isostasy loses continental area.

  python -m scripts.area_ledger_probe <seed> <True|False>     writes area_ledger_<seed>_<flag>.json

Per plate step, as fractions of the planet's surface: area lost to collision (continental crust consumed under
continental crust), to subduction (continental crust consumed under ocean floor), gained by rifting (new stretched
continental nodes) and by arcs (oceanic crust grown past arc_to_continent_thick_m); and the thickness of the
continental crust consumed by subduction.
"""
import json
import sys

import numpy as np

from planet_layer import fields, plates as P
from planet_layer.sphere_graph import CubeSphere


def main():
    seed, flag = int(sys.argv[1]), sys.argv[2] == 'True'
    params = dict(erosion_K_factor=1.0, erosion_theta=200., erosion_Sc=0.036, margin_shortening=True,
                  sediment_accretion=0.13, breakup_thick_m=10000., rift_width_km=100., thinning_local_isostasy=flag,
                  threads=1)
    g = CubeSphere(217); A = g.area.sum()
    sim = P.PlateSim(g, seed=seed, t0=-250., params=params)
    orig = P.PlateSim._update_rasters
    rec = []

    def wrapped(self, dthick, idx, dsed=None):
        a = g.area; w = self.world; p = self.p
        col = sub = 0.; subT = []
        for k, pl in enumerate(self.plates):
            wn = fields.nearest_nodes(g, g.xyz@pl.Q.T)
            own = self.owner_slot[wn] == k
            gone = pl.mask & ~own & (pl.under+1 > p['hidden_steps'])
            cc = gone & pl.cont & w['cont'][wn]; lost = gone & pl.cont & ~w['cont'][wn]
            col += a[cc].sum(); sub += a[lost].sum(); subT += list(pl.thick[lost])
        self._probe = {'collision': col/A, 'subduction': sub/A,
                       'subducted_T_km_median': float(np.median(subT)/1e3) if subT else None}
        return orig(self, dthick, idx, dsed)

    P.PlateSim._update_rasters = wrapped
    for i in range(250):
        r0 = sim.erosion_totals.get('rift_nodes', 0)
        sim.step()
        w = sim.world
        mean_a = g.area.mean()
        r = dict(sim._probe)
        r['t'] = sim.t
        r['rift'] = (sim.erosion_totals.get('rift_nodes', 0)-r0)*mean_a/A
        r['arc'] = float(g.area[sim.new_continent].sum()/A)
        c = w['cont']; T = w['thick']
        r['area'] = float(g.area[c].sum()/A)
        r['thin_lt30'] = float(g.area[c & (T < 30e3)].sum()/g.area[c].sum())
        rec.append(r)
        if (i+1) % 25 == 0:
            print(seed, flag, sim.t, {k: round(sum(x[k] for x in rec[-25:]), 4) for k in ('collision', 'subduction', 'rift', 'arc')},
                  'area', round(r['area'], 3), 'thin', round(r['thin_lt30'], 3), flush=True)
            json.dump(rec, open(f'area_ledger_{seed}_{flag}.json', 'w'))
    json.dump(rec, open(f'area_ledger_{seed}_{flag}.json', 'w'))


if __name__ == '__main__':
    main()
