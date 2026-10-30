"""Whole-planet landscape run driven by the plate layer (docs/planet-layer-plan.md, step 7).

  python -m scripts.run_planet_landscape --plates latest
  python -m scripts.run_planet_landscape --plates outputs\\planet-plates\\20261017-141621\\seed1001 --window 40

--plates latest picks the newest complete plate run (outputs/planet-plates/<timestamp>/seed*/stats.json).
The plate run must have snapshots every 1 Myr over at least the window (simulate_plates saves the last 40 Myr).
Builds the parcel forcing (planet_layer.coupling), runs the landscape model at n = 868 (11.53 km, the spacing it was
tuned at) for the window with sphere flexure, and writes outputs/planet-landscape/<timestamp>-seed<s>/:
present.npz (cube node fields), preview.png, stats.json (hypsometry against Earth on the same grids, and denudation
rates by elevation over the last 10 Myr against Earth's).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from planet_layer import coupling, fields, graph_landscape as gl
from planet_layer.sphere_flexure import SphereFlexure
from planet_layer.sphere_graph import CubeSphere

ROOT = Path(__file__).resolve().parents[1]
QS = (10, 25, 50, 75, 90, 99)
# incision threshold (A^m S) at 11.53 km (scripts.calibrate_landscape_theta). Step 7 picked 20 on denudation alone;
# with the step 8 rule (denudation and 23 km texture) theta-20261017-230010 picked 10 (scores 1.01 / 0.84 / 0.67 / 0.87
# for 0 / 5 / 10 / 20). On the 22 Oct plate baseline, three seeds (scripts.theta_seeds_review, theta-20261022-214812/
# 224344/234017): mean scores 0.92 / 0.65 / 0.67 / 1.06, so 5 (adopted as the working default on 23 Oct)
THETA = 5.
# Earth's long-term denudation by elevation (m/Myr; the step 6 table in the plan doc) and the classes it is read on
DENUDATION_CLASSES = ((200, 500), (500, 1000), (1000, 2000))
EARTH_DENUDATION = {'200-500': (1., 10.), '500-1000': (50., 100.), '1000-2000': (200., 500.), 'mean_land': (40., 70.)}


def latest_plate_run(seed=None):
    runs = sorted(d for d in (ROOT/'outputs'/'planet-plates').glob('2*') if d.is_dir())
    for run in reversed(runs):
        seeds = sorted(s for s in run.glob('seed*') if (s/'stats.json').exists() and (s/'snapshots').is_dir())
        if seed is not None:
            seeds = [s for s in seeds if s.name == f'seed{seed}']
        if seeds:
            return seeds[0]
    raise FileNotFoundError('no complete plate run in outputs/planet-plates')


def chosen_sweep_run(seed=None):
    """The winning combination of the newest finished scripts.sweep_plates run."""
    for d in sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'), reverse=True):
        r = d/'results.json'
        if r.exists() and 'chosen' in (res := json.loads(r.read_text())):
            if res['chosen'] is None:                   # the newest finished sweep's rule chose nothing: stop here
                raise SystemExit(f'{d.name}: the sweep rule chose no combination (see its results.json)')
            # sweeps since 24 Oct name the winner's folder; older ones used every swept parameter in the name
            name = res.get('chosen_dir') or '_'.join(f'{k}-{v}' for k, v in res['chosen'].items()) or 'defaults'
            seeds = sorted((d/name).glob('seed*')) if seed is None else [d/name/f'seed{seed}']
            if seeds and (seeds[0]/'snapshots').is_dir():
                return seeds[0]
    raise FileNotFoundError('no finished plate sweep in outputs/planet-calibration')


def resolve_plates(arg, seed=None):
    """'latest' = newest complete simulate_plates run; 'sweep' = winner of the newest sweep_plates run."""
    if arg == 'latest':
        return latest_plate_run(seed)
    if arg == 'sweep':
        return chosen_sweep_run(seed)
    return Path(arg)


def hypsometry(z, area):
    land = z > 0
    return {'land_fraction': float(area[land].sum()/area.sum()),
            'land_mean_m': float(np.average(z[land], weights=area[land])) if land.any() else 0.,
            'land_percentiles_m': {str(q): float(np.percentile(z[land], q)) for q in QS} if land.any() else {}}


def to_coarse(g_fine, z, g_coarse):
    """Area mean of fine node values onto the nearest coarse nodes."""
    k = fields.nearest_nodes(g_coarse, g_fine.xyz)
    s = np.bincount(k, weights=z*g_fine.area, minlength=g_coarse.size)
    w = np.bincount(k, weights=g_fine.area, minlength=g_coarse.size)
    return s/np.maximum(w, 1e-12)


def denudation_table(z, rate, area):
    """Area-mean denudation rate (m/Myr, rock removed net of deposition) on land by present elevation."""
    land = z > 0
    out = {'mean_land': float(np.average(rate[land], weights=area[land]))}
    for lo, hi in DENUDATION_CLASSES:
        m = land & (z > lo) & (z <= hi)
        out[f'{lo}-{hi}'] = float(np.average(rate[m], weights=area[m])) if m.any() else float('nan')
    return out


def denudation_score(table):
    """Worst distance (log10) of the planet's rates outside Earth's bands; 0 = every class inside its band."""
    worst = 0.
    for k, (lo, hi) in EARTH_DENUDATION.items():
        v = max(table[k], 1e-3)
        worst = max(worst, np.log10(lo/v) if v < lo else np.log10(v/hi) if v > hi else 0.)
    return float(worst)


def earth_texture23():
    """Earth's 23 km input texture from the meso's held-out split (outputs/meso/dataset.npz, val), cached."""
    cache = ROOT/'outputs'/'planet-reference'/'earth-meso-val-texture.json'
    if cache.exists():
        t = json.loads(cache.read_text())
        if all(k in t.get('curvature_m', {}) for k in TEXTURE_CLASSES):   # a cache from fewer classes is rebuilt
            return t
    from scripts.planet_meso_check import DATA, window_stats, classify, core_median
    cond = np.asarray(np.load(DATA, mmap_mode='r')['val_cond'], np.float64)
    w = np.sign(cond)*cond**2
    t = texture23_of(*window_stats(w), core_median(w), classify)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(t, indent=1))
    return t


def texture23_of(land, rel, curv, elev, classify):
    lw = land >= .98
    cls = classify(land, rel, elev)
    return {'relief_368km_m': {str(q): float(np.percentile(rel[lw], q)) for q in (10, 25, 50, 75, 90)},
            'relief_below_300_share': float((rel[lw] < 300).mean()),
            'curvature_m': {k: float(np.median(curv[v])) if len(v) else 0. for k, v in cls.items() if k in TEXTURE_CLASSES}}


TEXTURE_CLASSES = ('flat', 'plains-lowlands', 'plain-upland', 'hills')


def texture23(g, z):
    """The meso's 23 km input built from this landscape (scripts.planet_meso_check), and its texture against Earth:
    368 km land-window relief p25 / p50 / p75 and median 23 km curvature for flat, lowland-plain, plain/upland and
    hills windows. Score = the worst |log10| of those seven ratios (0 = Earth)."""
    from scripts.planet_meso_check import cells23, windows_of, window_stats, classify, core_median
    _, w = windows_of(cells23(g.to_cube(np.asarray(z, np.float32))))
    t = texture23_of(*window_stats(w), core_median(w), classify)
    e = earth_texture23()
    ratios = {f'relief_p{q}': t['relief_368km_m'][q]/e['relief_368km_m'][q] for q in ('25', '50', '75')}
    ratios.update({f'curvature_{k}': t['curvature_m'][k]/e['curvature_m'][k] for k in TEXTURE_CLASSES})
    t['ratios'] = ratios
    t['score'] = float(max(abs(np.log10(max(v, 1e-3))) for v in ratios.values()))
    return t


def preview(path, g, o, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eq = fields.equirect_nodes(g, 900)
    fig, ax = plt.subplots(2, 1, figsize=(18, 18))
    ax[0].imshow(o['elevation_m'][eq], cmap='terrain', vmin=-6000, vmax=6000, interpolation='nearest')
    ax[0].set_title(title, fontsize=10)
    land = ~o['ocean'][eq]
    A = o['drainage_area_m2'][eq]
    ax[1].imshow(np.where(land, 0., np.nan), cmap='Greys', vmin=0, vmax=4, interpolation='nearest')
    ax[1].imshow(np.where(land & (A > 1e9), np.log10(np.maximum(A, 1)), np.nan), cmap='Blues', vmin=9, vmax=12.5,
                 interpolation='nearest')
    ax[1].set_title('rivers: log10 drainage area > 9 (m^2)', fontsize=10)
    for a in ax:
        a.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def get_forcing(hist, g, window, forcing_dir):
    meta = Path(forcing_dir)/'forcing.json'
    if meta.exists():
        m = json.loads(meta.read_text())
        if m['window_myr'] == int(round(window)) and m['fine_n'] == g.n and m.get('plates') == str(hist.dir):
            f = Path(forcing_dir)
            print(f'reusing forcing {f}', flush=True)
            return {'load_m': np.load(f/'load_m.npy', mmap_mode='r'), 'surface_m': np.load(f/'surface_m.npy', mmap_mode='r'),
                    'z0_m': np.load(f/'z0_m.npy'), 'orogeny_age_myr': np.load(f/'orogeny_age_myr.npy'),
                    'continental_now': np.load(f/'continental_now.npy'), 'meta': m}
    F = coupling.build_forcing(hist, g, window, forcing_dir, progress=lambda s: print(s, flush=True))
    F['meta']['plates'] = str(hist.dir)
    meta.write_text(json.dumps(F['meta'], indent=1))
    return F


def run(seed_dir, out, n=868, window=30., dt=5e4, theta=THETA, forcing_dir=None, g=None, hist=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    hist = hist or coupling.PlateHistory(Path(seed_dir)/'snapshots')
    g = g or CubeSphere(n)
    print(f'plate history {seed_dir} (n={hist.g.n}); landscape grid n={n} ({g.size:,} nodes, '
          f'{np.pi*g.radius/2/n/1e3:.2f} km); theta {theta:g}', flush=True)
    F = get_forcing(hist, g, window, forcing_dir or out/'forcing')
    W = F['meta']['window_myr']
    load, surf = F['load_m'], F['surface_m']
    print(f'forcing ready in {time.perf_counter()-t_start:.0f} s', flush=True)
    rng = np.random.default_rng(0)
    z0 = np.asarray(F['z0_m'], np.float64)+rng.normal(0, 1., g.size)
    erod = coupling.erodibility(np.asarray(F['orogeny_age_myr']))
    flex = SphereFlexure.for_graph(g, gl.DEFAULTS['Te'])
    k_of = lambda t: min(int(t//1e6), W-1)
    model = gl.GraphLandscape(g, z0, lambda t: load[k_of(t)]/1e6, flex, {'dt': dt, 'theta': float(theta)}, erod,
                              surface_rate=lambda t: surf[k_of(t)]/1e6)
    t1 = time.perf_counter()
    every = int(round(5e6/dt))
    late = min(10, W)                                 # denudation is measured over the last `late` Myr
    mark = {}

    def report(m):
        z = m.z; land = ~m.ocean
        print(f'  {m.t/1e6-W:+5.1f} Myr: land {g.area[land].sum()/g.area.sum():.1%}, land mean {np.average(z[land], weights=g.area[land]):.0f} m, '
              f'max {z.max():.0f} m, {(time.perf_counter()-t1)/(m.t/dt):.2f} s/step', flush=True)

    model.run((W-late)*1e6, report, every)
    mark['load'] = model.load.copy()
    model.run(late*1e6, report, every)
    o = model.outputs()
    uplift_late = np.asarray(load[W-late:], np.float64).sum(0)
    rate = (uplift_late-(model.load-mark['load']))/late    # rock removed, m/Myr (net of deposition)
    np.savez_compressed(out/'present.npz', n=n, **{k: v for k, v in o.items() if k in ('elevation_m', 'drainage_area_m2', 'outlet', 'lake_depth_m', 'river', 'ocean', 'drains_to_sea', 'rock_load_m')},
                        erodibility=erod.astype(np.float32), initial_m=F['z0_m'], denudation_late_m_myr=rate.astype(np.float32))
    z = o['elevation_m'].astype(np.float64)
    from scripts.earth_reference import load as earth_load
    gc = CubeSphere(217)
    cont = np.asarray(F['continental_now']) > .5
    den = denudation_table(z, rate, g.area)
    stats = {'plates': str(seed_dir), 'n': n, 'window_myr': W, 'theta': float(theta), 'seconds': time.perf_counter()-t_start,
             'landscape_seconds_per_step': (time.perf_counter()-t1)/(W*1e6/dt), 'totals_m3': model.totals,
             'timing_s': model.timing,
             'planet_n868': hypsometry(z, g.area), 'planet_n217': hypsometry(to_coarse(g, z, gc), gc.area),
             'continental_crust_submerged': float(g.area[cont & (z <= 0)].sum()/g.area[cont].sum()),
             f'denudation_last{late}myr_m_per_myr': den, 'earth_denudation_m_per_myr': EARTH_DENUDATION,
             'denudation_score': denudation_score(den),
             'texture23': texture23(g, z) if n == 868 else None,
             'land_draining_to_sea': float(o['drains_to_sea'][~o['ocean']].mean())}
    for n_ref in (n, 217):
        try:
            stats[f'earth_n{n_ref}_no_ice'] = earth_load(n_ref)['no_ice_sheets']
        except Exception as e:                       # ETOPO missing: report the planet only
            stats[f'earth_n{n_ref}_no_ice'] = f'unavailable: {e}'
    (out/'stats.json').write_text(json.dumps(stats, indent=1))
    preview(out/'preview.png', g, o, f'{Path(seed_dir).parent.name}/{Path(seed_dir).name}: landscape after {W} Myr at '
            f'{np.pi*g.radius/2/n/1e3:.2f} km, theta {theta:g}')
    pl, pe = stats['planet_n868'], stats.get(f'earth_n{n}_no_ice')
    print(f"land {pl['land_fraction']:.1%}; land p10/25/50/75/90/99 = " + '/'.join(f"{pl['land_percentiles_m'][str(q)]:.0f}" for q in QS) + ' m')
    if isinstance(pe, dict):
        print(f"Earth {pe['land_fraction']:.1%}; land p10/25/50/75/90/99 = " + '/'.join(f"{pe['land_percentiles_m'][str(q)]:.0f}" for q in QS) + ' m')
    if stats['texture23']:
        t = stats['texture23']
        print('23 km texture / Earth: ' + ', '.join(f'{k} {v:.2f}' for k, v in t['ratios'].items()) + f"; score {t['score']:.2f}")
    print('denudation m/Myr (last %d Myr): ' % late + ', '.join(f'{k} {v:.0f}' for k, v in den.items()) +
          f'; score {stats["denudation_score"]:.2f}')
    print(f'written {out}', flush=True)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plates', required=True, help="plate run seed folder (contains snapshots/), 'latest' or 'sweep'")
    p.add_argument('--seed', type=int, default=None, help="with --plates latest: which seed folder")
    p.add_argument('--n', type=int, default=868)
    p.add_argument('--window', type=float, default=30.)
    p.add_argument('--dt', type=float, default=5e4)
    p.add_argument('--theta', type=float, default=THETA)
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    seed_dir = resolve_plates(a.plates, a.seed)
    out = Path(a.out) if a.out else ROOT/'outputs'/'planet-landscape'/f"{time.strftime('%Y%m%d-%H%M%S')}-{seed_dir.name}"
    run(seed_dir, out, a.n, a.window, a.dt, a.theta)


if __name__ == '__main__':
    main()
