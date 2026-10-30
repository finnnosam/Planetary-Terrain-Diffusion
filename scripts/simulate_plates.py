"""Plate layer: simulate plates, crust and coarse erosion from a supercontinent and draw maps (docs/planet-layer-plan.md, steps 5-6).

  python -m scripts.simulate_plates                    # n = 217 (46 km), 250 Myr, seed 1
  python -m scripts.simulate_plates --seeds 1,2,3 --years 250

Writes outputs/planet-plates/<timestamp>/: index.html with maps every 25 Myr and present-day fields, stats.json
(time series and present-day statistics against Earth), and snapshots/ (world fields every --save-every Myr over the
last --save-last Myr, every 10 Myr before; plate rotations at every save).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere
from planet_layer.plates import PlateSim, RIDGE, SUBDUCTION, COLLISION, TRANSFORM

ROOT = Path(__file__).resolve().parents[1]
EARTH = {'land_fraction': .29, 'land_mean_elevation_m': 840., 'ocean_age_mean_myr': 64., 'ocean_age_max_myr': 180., 'ridge_km': 65000.,
         'subduction_km': 55000., 'plates_bird2003': 52, 'largest_plate_mkm2': 104.}


def boundary_lengths(g, st):
    h = np.sqrt(g.area.mean())/1e3
    bt = st['boundary_type']
    # boundary nodes appear on both sides; halve to get line length
    return {name: float((bt == code).sum()*h/2) for name, code in (('ridge_km', RIDGE), ('subduction_km', SUBDUCTION),
                                                                      ('collision_km', COLLISION), ('transform_km', TRANSFORM))}


def summary(sim, st):
    g = sim.g
    A = g.area.sum()
    land = st['elevation_m'] > 0
    oa = st['ocean_age_myr']; ok = ~np.isnan(oa)
    sizes = np.sort([r['area_km2'] for r in sim.plate_table()])[::-1]/1e6
    w = g.area[ok]
    kinds = {}
    for e in sim.events:
        kinds[e['kind']] = kinds.get(e['kind'], 0)+1
    z = st['elevation_m']
    return {'time_myr': sim.t, 'plates': len(sim.plates), 'plates_over_1mkm2': int((sizes > 1).sum()),
            'plate_areas_mkm2': [round(float(s), 2) for s in sizes], 'events': kinds,
            'land_fraction': float(g.area[land].sum()/A), 'continental_fraction': float(g.area[st['continental']].sum()/A),
            'land_mean_elevation_m': float(np.average(z[land], weights=g.area[land])) if land.any() else 0.,
            'land_elevation_percentiles_m': {str(q): float(np.percentile(z[land], q)) for q in (10, 25, 50, 75, 90, 99)} if land.any() else {},
            'sediment_max_m': float(st['sediment_m'].max()), 'erosion_totals_km3': {k: v/1e9 for k, v in sim.erosion_totals.items()},
            'continental_volume_km3': sim.continental_volume()/1e9,
            'continental_sediment_km3': sim.continental_sediment_volume()/1e9,
            'sediment_ledger_km3': {k: v/1e9 for k, v in sim.sed_ledger.items()},
            'ocean_age_mean_myr': float((oa[ok]*w).sum()/w.sum()), 'ocean_age_max_myr': float(oa[ok].max()),
            'speeds_cm_yr': [round(r['speed_cm_yr'], 2) for r in sim.plate_table()],
            'pinholes': sim.pinholes(), 'new_crust_nodes': int(st['new_crust'].sum()), **boundary_lengths(g, st),
            # plate pieces other than each plate's largest (B-b, thirty-seventh pass) and piece_rule 'plates' counters
            'detached_area_frac': sim.detached_area()/A, 'piece_stats': dict(getattr(sim, 'piece_stats', {})),
            'elevation_percentiles_m': {str(q): float(np.percentile(st['elevation_m'], q)) for q in (5, 25, 50, 75, 95)},
            'sea_level_on_datum_m': float(st.get('sea_level_m', 0.)),
            # continental area (flag) change over the run and by mechanism, as fractions of the surface; the
            # residual is nearest-node bookkeeping (twenty-seventh pass)
            'continental_area_change_frac': float((g.area[st['continental']].sum()-getattr(sim, 'area0', np.nan))/A),
            'area_ledger_frac': {k: v/A for k, v in getattr(sim, 'area_ledger', {}).items()},
            'extension_lost': sum(e['kind'] == 'extension lost' for e in sim.events),   # a half merged away or consumed
            'extension_ends': [{k: e[k] for k in ('t', 'myr', 'zone_thick_km', 'zone_area_mkm2', 'guard')}
                               for e in sim.events if e['kind'] == 'extension end'],
            'extension_starts': [{k: e.get(k) for k in ('t', 'W', 'v', 'piece_mkm2', 'piece_thick_km', 'across', 'hot_share',
                                                        'cut', 'L', 'far_nodes', 'far_collision', 'side',
                                                        'far_collision_sides', 'moved_larger')}
                                 for e in sim.events if e['kind'] == 'extension'],
            # crust thicker than the onset, summed over the steps so far, as fractions of that sum: in pieces eligible
            # to start an event (E2's trigger reach), hot pieces, held pieces, hot nodes (twenty-ninth pass)
            'extension_trigger': {**{k: v/max(sim.ext_reach['gt_onset'], 1e-9) for k, v in getattr(sim, 'ext_reach', {}).items()
                                     if k != 'gt_onset'}, 'gt_onset_steps_mkm2': getattr(sim, 'ext_reach', {}).get('gt_onset', 0.)/1e12}}


def save_snapshot(folder, sim, st):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f't{sim.t:+07.1f}.npz', **{k: v for k, v in st.items() if k != 'time_myr'}, time_myr=sim.t)
    (folder/f't{sim.t:+07.1f}-plates.json').write_text(json.dumps(sim.plate_table()))


# Elevation colours for the maps every 25 Myr (relative to sea level, m): blues below 0 m, a hypsometric tint above
# (green lowland, olive and sand uplands, browns, grey-brown, snow). matplotlib's 'terrain' put 0 m in the middle of
# its ramp, so shallow sea (-1500..0 m) came out green like lowland (changed 25 Oct 2026). N = 65,536 levels, as in
# planet_training/elevation_shading.py, so gentle plains draw no false contour bands.
MAP_STOPS_M = [(-6000, '#0b2447'), (-4500, '#13386b'), (-3000, '#1f5791'), (-1500, '#3478b3'), (-500, '#5c9fd0'),
               (-150, '#8cc3e3'), (-1, '#b8ddef'),
               (0, '#5f9654'), (150, '#7fac63'), (400, '#a9bf78'), (800, '#cfcd92'), (1300, '#dcc394'),
               (2000, '#c49c74'), (3000, '#9c765e'), (4200, '#8d827b'), (5200, '#cbc6c2'), (6000, '#f8f8f8')]


def map_colours():
    """(cmap, norm) for the elevation maps: MAP_STOPS_M over -6000..6000 m."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    lo, hi = MAP_STOPS_M[0][0], MAP_STOPS_M[-1][0]
    cmap = LinearSegmentedColormap.from_list('planet_map', [((z-lo)/(hi-lo), c) for z, c in MAP_STOPS_M], N=65536)
    return cmap, Normalize(lo, hi)


def draw(out, maps, present, eq):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap, norm = map_colours()
    rows = (len(maps)+1)//2
    fig, axs = plt.subplots(rows, 2, figsize=(16, 4.3*rows))
    for ax, (t, elev, bt) in zip(axs.ravel(), maps):
        ax.imshow(elev[eq], cmap=cmap, norm=norm, interpolation='nearest')
        b = bt[eq]
        for code, col in ((RIDGE, 'red'), (SUBDUCTION, 'black'), (COLLISION, 'magenta')):
            yy, xx = np.nonzero(b == code)
            ax.scatter(xx, yy, s=.15, c=col, linewidths=0)
        ax.set_title(f'{t:.0f} Myr: elevation; ridges red, trenches black, collisions magenta', fontsize=9); ax.axis('off')
    for ax in axs.ravel()[len(maps):]:
        ax.axis('off')
    fig.tight_layout(rect=(0, .7/(4.3*rows), 1, 1))
    cax = fig.add_axes([.25, .25/(4.3*rows), .5, .12/(4.3*rows)])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_ticks([-6000, -4000, -2000, -500, 0, 500, 1000, 2000, 3000, 4000, 6000])
    cb.ax.tick_params(labelsize=8); cb.set_label('elevation relative to sea level (m)', fontsize=9)
    fig.savefig(out/'maps.png', dpi=70); plt.close(fig)
    fig, axs = plt.subplots(3, 2, figsize=(16, 13.5))
    st = present
    panels = [('plate id', st['plate_id'] % 20, 'tab20', None), ('ocean-floor age (Myr)', st['ocean_age_myr'], 'viridis_r', (0, 200)),
              ('crust thickness (km)', st['crust_thickness_m']/1e3, 'magma', (5, 70)),
              ('time since last collision (Myr)', st['orogeny_age_myr'], 'cividis', (0, 1500)),
              ('ocean-floor sediment (m)', st['sediment_m'], 'YlOrBr', (0, 8000)),
              ('dynamic topography (m)', st['dynamic_topography_m'], 'RdBu_r', (-1200, 1200))]
    for ax, (title, v, cm, lim) in zip(axs.ravel(), panels):
        im = ax.imshow(v[eq].astype(float), cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_title(title, fontsize=10); ax.axis('off'); fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout(); fig.savefig(out/'present.png', dpi=70); plt.close(fig)


def run_seed(seed, a, out):
    g = CubeSphere(a.n)
    t0 = time.perf_counter()
    sim = PlateSim(g, seed=seed, t0=-a.years, params=a.params)
    eq = fields.equirect_nodes(g, 360)
    series, maps = [], []
    sdir = out/f'seed{seed}'
    sdir.mkdir(parents=True, exist_ok=True)

    def record(s):
        st = s.state()
        series.append(summary(s, st))
        late = s.t > -a.save_last-1e-6
        if (late and round(s.t) % a.save_every == 0) or round(s.t) % 10 == 0:
            save_snapshot(sdir/'snapshots', s, st)
        if round(s.t) % 25 == 0:
            maps.append((s.t, st['elevation_m'], st['boundary_type']))
        if round(s.t) % 10 == 0:
            r = series[-1]
            print(f"seed {seed} t={s.t:6.0f}: plates {r['plates']}, land {r['land_fraction']:.0%} (mean {r['land_mean_elevation_m']:.0f} m), ocean age mean {r['ocean_age_mean_myr']:.0f} "
                  f"max {r['ocean_age_max_myr']:.0f} Myr, ridge {r['ridge_km']:.0f} km, trench {r['subduction_km']:.0f} km, pinholes {r['pinholes']}, "
                  f"sea level {r['sea_level_on_datum_m']:+.0f} m", flush=True)
    record(sim)
    sim.run(a.years, record, 1)
    st = sim.state()
    draw(sdir, maps, st, eq)
    v = [r['continental_volume_km3'] for r in series]
    # with continental_sediment the deposits on continental crust are sediment, not crust: the volume compared with
    # earlier runs (where they were crust) and with CRUST1.0's total thickness is crust + continental sediment
    layer = sim.p['continental_sediment']
    vt = [r['continental_volume_km3']+(r['continental_sediment_km3'] if layer else 0.) for r in series]
    eff = {k: v for k, v in sim.p.items() if isinstance(v, (bool, int, float, str, list, tuple, type(None)))}
    result = {'seed': seed, 'n': a.n, 'years': a.years, 'params': a.params, 'params_effective': eff,
              'seconds': time.perf_counter()-t0, 'present': series[-1],
              'continental_volume_change': vt[-1]/vt[0]-1, 'continental_crust_volume_change': v[-1]/v[0]-1, 'max_pinholes': max(r['pinholes'] for r in series), 'earth': EARTH,
              'series': series,
              # every plate event with its time (thirty-ninth pass; output only): what cuts and joins plates when
              'event_list': [{k: (float(e[k]) if isinstance(e[k], (float, np.floating)) else e[k])
                              for k in ('t', 'kind', 'plate', 'new_plate', 'merged', 'area_mkm2') if k in e}
                             for e in sim.events]}
    (sdir/'stats.json').write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=217)
    p.add_argument('--seeds', default='1')
    p.add_argument('--years', type=float, default=250.)
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--save-last', type=float, default=40.)
    p.add_argument('--out', default=None)
    p.add_argument('--allow-coarse', action='store_true', help='allow n < 96 (plumbing tests only)')
    p.add_argument('--param', action='append', default=[], metavar='KEY=VALUE',
                   help='override a planet_layer.plates.PARAMS entry, e.g. --param erosion_theta=16')
    a = p.parse_args(argv)
    if a.n < 96 and not a.allow_coarse:
        raise SystemExit('n < 96: the erosion parameters are for ~46 km cells (n = 217) and lose meaning on much coarser '
                         'grids; use --allow-coarse for plumbing tests only')
    from planet_layer.plates import PARAMS
    a.params = {}
    for kv in a.param:
        k, v = kv.split('=', 1)
        if k not in PARAMS:
            raise SystemExit(f'unknown parameter {k!r}')
        a.params[k] = type(PARAMS[k])(v) if PARAMS[k] is not None and not isinstance(PARAMS[k], bool) else (v.lower() in ('1', 'true', 'yes') if isinstance(PARAMS[k], bool) else float(v))
    out = Path(a.out) if a.out else ROOT/'outputs'/'planet-plates'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    results = [run_seed(int(s), a, out) for s in a.seeds.split(',')]
    rows = ''.join(f"<tr><td>{r['seed']}</td><td>{r['present']['plates']} ({r['present']['plates_over_1mkm2']}), largest {r['present']['plate_areas_mkm2'][0]:.0f}</td>"
                   f"<td>{r['present']['land_fraction']:.0%}</td><td>{r['present']['land_mean_elevation_m']:.0f}</td>"
                   f"<td>{r['present']['ocean_age_mean_myr']:.0f} / {r['present']['ocean_age_max_myr']:.0f}</td>"
                   f"<td>{r['present']['ridge_km']:.0f} / {r['present']['subduction_km']:.0f}</td><td>{r['continental_volume_change']:+.1%}</td>"
                   f"<td>{r['max_pinholes']}</td><td>{r['seconds']:.0f}</td></tr>" for r in results)
    imgs = ''.join(f"<h2>Seed {r['seed']}</h2><img src='seed{r['seed']}/maps.png'><img src='seed{r['seed']}/present.png'>" for r in results)
    (out/'index.html').write_text(f"<html><body style='font-family:sans-serif'><h1>Plate layer</h1><table border=1 cellpadding=4>"
                                  f"<tr><th>seed</th><th>plates (over 1 Mkm²), largest Mkm²</th><th>land</th><th>mean land elevation (m)</th><th>ocean age mean / max (Myr)</th><th>ridge / trench (km)</th>"
                                  f"<th>continental volume change</th><th>max pinholes</th><th>seconds</th></tr>{rows}"
                                  f"<tr><td>Earth</td><td>52 (Bird 2003), largest 104</td><td>29%</td><td>~840</td><td>~64 / ~180</td><td>~65,000 / ~55,000</td><td>-</td><td>-</td><td>-</td></tr>"
                                  f"</table>{imgs}</body></html>")
    print(f'written {out}')


if __name__ == '__main__':
    main()
