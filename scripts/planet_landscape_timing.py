"""Timing and smoke test of the landscape model on the whole cube sphere (docs/planet-layer-plan.md, step 4).

A synthetic planet stands in for the plate layer (not built yet): ~30% continents from a random low-degree field,
graded shelves and abyss, three uplift belts on land (6 mm/yr peak, rock load), and slow mantle swells as non-load
surface motion (+-30 m/Myr). It runs a few 50 kyr steps with sphere flexure, reports seconds per step by process
and the projected time for a 30 Myr run, and writes a preview map. The terrain itself is not meant to be judged.

  python -m scripts.planet_landscape_timing                 # n = 868 (11.53 km), 20 steps = 1 Myr
  python -m scripts.planet_landscape_timing --n 434 --steps 10
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from planet_diffusion import cube
from planet_layer import sphere_graph as sg, graph_landscape as gl
from planet_layer.sphere_flexure import SphereFlexure

ROOT = Path(__file__).resolve().parents[1]


def random_field(loc, lmax, slope, rng):
    """Unit-variance random field from spherical harmonics with power ~ l^-slope (l >= 1), at node locations."""
    import ducc0
    alm = np.zeros((lmax+1)*(lmax+2)//2, np.complex128)
    for m in range(lmax+1):
        for l in range(max(m, 1), lmax+1):
            i = m*(2*lmax+1-m)//2+l
            a = rng.normal()*l**(-slope/2)
            alm[i] = a if m == 0 else (a+1j*rng.normal()*l**(-slope/2))/np.sqrt(2)
    f = ducc0.sht.experimental.synthesis_general(alm=alm[None], spin=0, lmax=lmax, loc=loc, epsilon=1e-8)[0]
    return (f-f.mean())/f.std()


def synthetic_planet(g, seed):
    rng = np.random.default_rng(seed)
    x, y, z = g.xyz.T
    loc = np.ascontiguousarray(np.stack((np.arccos(np.clip(z, -1, 1)), np.mod(np.arctan2(y, x), 2*np.pi)), 1))
    cont = random_field(loc, 16, 2., rng)
    land = cont > np.percentile(cont, 70)
    off = gl.distance_to(land, g.nbr, g.dist)
    z0 = np.where(land, 400.+150.*np.clip(cont-cont[land].min(), 0, 3), -120.-3900.*(1-np.exp(-np.clip(off-80e3, 0, None)/150e3))-80.*np.minimum(off/80e3, 1))
    z0 = z0+rng.normal(0, 2., g.size)
    belts = np.zeros(g.size)
    land_ids = np.flatnonzero(land)
    for _ in range(3):
        centre = g.xyz[rng.choice(land_ids)]
        axis = np.cross(centre, rng.normal(size=3)); axis /= np.linalg.norm(axis)
        across = np.arcsin(np.clip(g.xyz@axis, -1, 1))*g.radius
        along = np.arccos(np.clip(g.xyz@centre, -1, 1))
        belts += 6e-3*np.exp(-.5*(across/40e3)**2)*np.clip(1-along/.35, 0, 1)
    belts *= land
    swells = 3e-5*random_field(loc, 8, 2., rng)
    erod = .15+.85*np.clip(belts/.5e-3, 0, 1)
    return z0, belts, swells, erod, land


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=868, help='cube intervals per face (868 = 11.53 km, 2x the meso grid)')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--dt', type=float, default=5e4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    out = Path(a.out) if a.out else ROOT/'outputs'/'planet-landscape-timing'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    g = sg.CubeSphere(a.n)
    t_graph = time.perf_counter()-t
    print(f'graph n={a.n}: {g.size:,} nodes, {np.pi*g.radius/2/a.n/1e3:.2f} km nominal, built in {t_graph:.1f} s', flush=True)
    z0, belts, swells, erod, land = synthetic_planet(g, a.seed)
    flex = SphereFlexure.for_graph(g, gl.DEFAULTS['Te'])
    print(f'flexure lmax {flex.lmax}; land {land.mean():.0%}', flush=True)
    t = time.perf_counter()
    model = gl.GraphLandscape(g, z0, belts, flex, {'dt': a.dt}, erod, surface_rate=swells)
    model.step()                      # includes numba compilation
    t_first = time.perf_counter()-t
    for k in model.timing:
        model.timing[k] = 0.
    times = []
    for s in range(a.steps-1):
        t = time.perf_counter()
        model.step()
        times.append(time.perf_counter()-t)
        print(f'step {s+2}/{a.steps}: {times[-1]:.2f} s', flush=True)
    per_step = float(np.median(times)) if times else t_first
    parts = {k: v/max(len(times), 1) for k, v in model.timing.items()}
    steps_30 = 30e6/a.dt
    o = model.outputs()
    report = {'n': a.n, 'nodes': g.size, 'nominal_km': np.pi*g.radius/2/a.n/1e3, 'dt_yr': a.dt, 'steps': a.steps,
              'graph_build_s': t_graph, 'first_step_s_incl_compile': t_first, 'median_step_s': per_step,
              'seconds_per_step_by_process': parts, 'projected_30Myr_minutes': per_step*steps_30/60,
              'cpu_count': os.cpu_count(), 'flexure_lmax': flex.lmax, 'land_fraction': float((~o['ocean']).mean()),
              'land_draining_to_sea': float(o['drains_to_sea'][~o['ocean']].mean()), 'totals_m3': model.totals}
    (out/'timing.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    np.savez_compressed(out/'state.npz', elevation_m=o['elevation_m'], drainage_area_m2=o['drainage_area_m2'], n=a.n)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        h = 720
        zc = cube.to_equirectangular(g.to_cube(o['elevation_m'].astype(np.float32))[None], h)[0]
        ac = cube.to_equirectangular(g.to_cube(np.log10(o['drainage_area_m2']).astype(np.float32))[None], h)[0]
        fig, ax = plt.subplots(2, 1, figsize=(14, 14))
        ax[0].imshow(zc, cmap='terrain', vmin=-5000, vmax=5000); ax[0].set_title(f'elevation after {model.t/1e6:.2f} Myr (synthetic forcing)')
        ax[1].imshow(np.where(zc > 0, ac, np.nan), cmap='Blues', vmin=10, vmax=13); ax[1].set_title('log10 drainage area on land (m^2)')
        for x in ax:
            x.axis('off')
        fig.tight_layout(); fig.savefig(out/'preview.png', dpi=80); plt.close(fig)
    except ImportError:
        pass
    print(f'written {out}')


if __name__ == '__main__':
    main()
