"""Calibrate the landscape model at the plate layer's 46 km spacing against the 11.52 km model it was tuned at
(docs/planet-layer-plan.md, sect. 6).

Identical test-bed forcing (built at 11.52 km, area-averaged to 46 km), 30 Myr, both run with the graph landscape
kernels on flat windows. Compared, with the fine run area-averaged to 46 km blocks:
  * mean land elevation of 368 km blocks;
  * total denudation;
  * denudation of *stable* land (no young uplift) over the last 10 Myr, median of 46 km blocks.
Rule: all three within 15%. The third was added after the plate layer showed coarse erosion wearing continents down
(46 km stable lowlands eroding several times faster than the 11.52 km model's ~7 m/Myr).

  python -m scripts.calibrate_coarse_erosion
"""
import itertools
import json
import time

import numpy as np

from planet_training import tectonic_templates as tt
from planet_layer import graph_landscape as gl, sphere_graph as sg

FINE = (256, 11520., 5e4)
COARSE = (64, 46080., 2.5e5)
CASES = [('collision', 1), ('collision', 2), ('margin', 1)]
LATE_MYR = 10.


def block(a, f):
    n = a.shape[0]
    return a.reshape(n//f, f, n//f, f).mean((1, 3))


def forcing(scenario, seed, n):
    z0, up, erod, meta = tt.build(scenario, FINE[0], FINE[1], seed, True)
    young = sum(np.asarray(f)*np.ones((FINE[0], FINE[0])) for a, b, f in up.events if a >= meta['young_start_yr'])
    stable = np.abs(young) < 1e-5
    if n == FINE[0]:
        return z0, up, erod, meta, stable
    f = FINE[0]//n
    events = [(a, b, block(np.asarray(fld)*np.ones((FINE[0], FINE[0])), f)) for a, b, fld in up.events]
    return block(z0, f), tt.Timeline(events, up.ramp), block(erod, f), meta, block(stable.astype(float), f) > .99


def run(scenario, seed, n, dx, dt, params):
    z0, up, erod, meta, stable = forcing(scenario, seed, n)
    g = sg.flat(n, n, dx)
    m = gl.GraphLandscape(g, z0, up, gl.FlatFlexure(n, n, dx, 30e3), {**params, 'dt': dt}, erod)
    years = meta['years']
    late = np.zeros(n*n)
    while m.t < years-1:
        if m.t >= years-LATE_MYR*1e6-1:
            z_before = m.z.copy(); load_before = m.load.copy()
            m.step()
            # rock removed = drop of the rock column independent of flexure and uplift: change of the load term
            late += np.maximum(-(m.load-load_before-(m._rate(m.uplift)*dt)), 0)
        else:
            m.step()
    m._route()
    z = m.z.reshape(n, n); land = ~m.ocean.reshape(n, n)
    f = n//64
    late_b = block(late.reshape(n, n), f)/LATE_MYR             # m/Myr on 46 km blocks
    land_b = block(land.astype(float), f) > .99
    st = (stable if n == 64 else block(stable.astype(float), f) > .99) & land_b
    b = n//8
    zb = z.reshape(8, b, 8, b).mean((1, 3)); lb = land.reshape(8, b, 8, b).mean((1, 3)) > .9
    return {'block_land_mean_m': float(zb[lb].mean()), 'eroded_km3': m.totals['eroded_m3']/1e9, 'uplift_km3': m.totals['uplift_m3']/1e9,
            'stable_denudation_m_myr': float(np.median(late_b[st])) if st.any() else float('nan'), 'max_m': float(z.max())}


def main():
    t0 = time.time()
    ref = {c: run(*c, *FINE, {}) for c in CASES}
    print('reference done', round(time.time()-t0), 's', json.dumps({f'{a}-{b}': r for (a, b), r in ref.items()}), flush=True)
    results = []
    for kf, kdf, sf in itertools.product([.5, .75, 1., 1.25], [1., 2., 4.], [1., .5, .25]):
        p = {'K': gl.DEFAULTS['K']*kf, 'Sc': .05, 'Kd': gl.DEFAULTS['Kd']*kdf}
        out = {}
        for c in CASES:
            z0, up, erod, meta, stable = forcing(*c, 64)
            out[c] = run_coarse(c, p, sf)
        e = [np.mean([abs(out[c][k]/ref[c][k]-1) for c in CASES]) for k in ('block_land_mean_m', 'eroded_km3', 'stable_denudation_m_myr')]
        results.append({'K_factor': kf, 'Kd_factor': kdf, 'stable_factor': sf, 'errors': e, 'worst': max(e), 'cases': {f'{a}-{b}': out[(a, b)] for a, b in CASES}})
        print(f"K x{kf:<5} Kd x{kdf:<3} stable x{sf:<4}: block mean {e[0]:6.1%}  denudation {e[1]:6.1%}  stable denudation {e[2]:6.1%}", flush=True)
    results.sort(key=lambda r: r['worst'])
    best = results[0]
    print('best', {k: best[k] for k in ('K_factor', 'Kd_factor', 'stable_factor', 'errors')}, 'PASS' if best['worst'] <= .15 else 'FAIL (rule: all within 15%)')
    print('best cases', json.dumps(best['cases']))


def run_coarse(case, p, stable_factor):
    """Coarse run; erodibility of stable crust (relative erodibility < 0.5) scaled by stable_factor."""
    orig = tt.build

    def build(*a, **k):
        z0, up, erod, meta = orig(*a, **k)
        return z0, up, erod, meta
    z0, up, erod, meta, stable = forcing(*case, 64)
    erod = np.where(erod < .5, erod*stable_factor, erod)
    n, dx, dt = COARSE
    g = sg.flat(n, n, dx)
    m = gl.GraphLandscape(g, z0, up, gl.FlatFlexure(n, n, dx, 30e3), {**p, 'dt': dt}, erod)
    years = meta['years']
    late = np.zeros(n*n)
    while m.t < years-1:
        if m.t >= years-LATE_MYR*1e6-1:
            load_before = m.load.copy()
            m.step()
            late += np.maximum(-(m.load-load_before-(m._rate(m.uplift)*dt)), 0)
        else:
            m.step()
    m._route()
    z = m.z.reshape(n, n); land = ~m.ocean.reshape(n, n)
    st = stable & land
    zb = z.reshape(8, 8, 8, 8).mean((1, 3)); lb = land.reshape(8, 8, 8, 8).mean((1, 3)) > .9
    return {'block_land_mean_m': float(zb[lb].mean()), 'eroded_km3': m.totals['eroded_m3']/1e9, 'uplift_km3': m.totals['uplift_m3']/1e9,
            'stable_denudation_m_myr': float(np.median((late/LATE_MYR).reshape(n, n)[st])) if st.any() else float('nan'), 'max_m': float(z.max())}


if __name__ == '__main__':
    main()
