"""Regional tectonic + landscape-evolution prototype, optionally refined by the selected meso.

For each scenario and seed: build the tectonic template, evolve it (uplift, rivers, sediment,
flexure), then save
  - 11.52 km fields: elevation, drainage area, rivers, drainage-basin labels, closed basins,
  - the 23.04 km elevation (2x2 area means) that the meso model is conditioned on,
  - the 40.32 km elevation used for the arrangement metrics,
and draw terrain / drainage / uplift / evolution figures. With --meso (needs the selected meso
checkpoint, ideally a GPU), four 368.64 km windows chosen by a fixed rule are refined to 2.88 km.

No training, no learned large-scale stage, no constraints: sea level is the 0 m datum only.

  python -m scripts.simulate_tectonic_region                    # both scenarios, seeds 1-3
  python -m scripts.simulate_tectonic_region --meso             # plus meso windows
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from planet_training import arrangement_metrics as am
from planet_training import landscape_evolution as le
from planet_training import tectonic_templates as tt
from planet_training.elevation_shading import image, legend
from planet_training.terrain_structure import conservative_resample, decode, encode

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT/'outputs/tectonic-prototype'
# The original Earth-trained meso (true area-averaged 23 km input, as the simulation produces). The later
# 'adapted' meso was fine-tuned for smoothed learned guides and loses plain texture: on held-out Earth plains it
# keeps 34-57% of real channel incision vs 100-105% for this one (outputs/meso-plains-diagnosis/20261015-193134).
MESO = ROOT/'outputs/meso/detail-ft/step10000-passed.pt'   # detail-aware, softening-robust fine-tune (docs/tectonic-prototype.md)
ROUND_TRIPS = ROOT/'outputs/terrain-round-trips'
N, DX = 256, 11520.            # 2,949 km window; 2x2 cells = the meso's 23.04 km condition grid
MESO_SEEDS = [113, 997]


def to40(z11):
    """Central 2,580 km area-averaged to 64 x 64 at 40.32 km."""
    c = int(round(2580480/DX)); a = (N-c)//2
    return conservative_resample(torch.from_numpy(z11[a:a+c, a:a+c].astype(np.float32))[None], 64, spacing=DX)[0].numpy()


def to23(z11):
    f = int(round(23040/DX))
    return z11.reshape(N//f, f, N//f, f).mean((1, 3)).astype(np.float32)


def earth_reference_metrics():
    """Arrangement metrics of unconstrained modern Earth at 40.32 km, when the round-trip files exist."""
    out = {}
    for name in ('alps', 'california', 'plains'):
        path = ROUND_TRIPS/f'{name}.npz'
        if path.exists():
            raw = conservative_resample(torch.from_numpy(np.load(path)['reference23_metres'])[:, 0], 64)[0].numpy()
            out[name] = am.arrangement(raw, raw)
    return out


def meso_windows(z23, land):
    """Four 18x18-cell (368.64 km + margin) windows by fixed rule: highest relief; the lowest
    land interior relative to high ground around it ('lowland between ranges'); a coastal window;
    the window centre."""
    size, n = 18, z23.shape[0]
    best = {}
    for i in range(1, n-size, 2):
        for j in range(1, n-size, 2):
            w = z23[i:i+size, j:j+size]; lw = land[i:i+size, j:j+size]
            if lw.mean() < .5:
                continue
            relief = float(np.percentile(w[lw], 95)-np.percentile(w[lw], 5))
            core = w[6:12, 6:12]; core_land = lw[6:12, 6:12]
            lowland = -1.
            if core_land.mean() > .8:
                rim = np.concatenate((w[:4].ravel(), w[-4:].ravel(), w[:, :4].ravel(), w[:, -4:].ravel()))
                # high ground on opposite sides of a low, land core
                sides = [np.percentile(w[:5][lw[:5]], 90) if lw[:5].any() else 0., np.percentile(w[-5:][lw[-5:]], 90) if lw[-5:].any() else 0.,
                         np.percentile(w[:, :5][lw[:, :5]], 90) if lw[:, :5].any() else 0., np.percentile(w[:, -5:][lw[:, -5:]], 90) if lw[:, -5:].any() else 0.]
                pair = max(min(sides[0], sides[1]), min(sides[2], sides[3]))
                lowland = float(pair-np.median(core)) if np.median(core) < np.percentile(rim, 40) else -1.
            coast = float(min(lw.mean(), 1-lw.mean()))
            for key, score in (('highest-relief', relief), ('lowland-between-ranges', lowland), ('coast', coast)):
                if key not in best or score > best[key][0]:
                    best[key] = (score, i, j)
    c = (n-size)//2
    windows = {k: (v[1], v[2]) for k, v in best.items() if v[0] > 0}
    windows['centre'] = (c, c)
    return windows


def figure(path, title, o, uplift, snaps, z40, metrics):
    z = o['elevation_m']; land = ~o['ocean']
    half = N*DX/2e3; ext = [-half, half, -half, half]
    fig, axs = plt.subplots(2, 3, figsize=(19, 13), constrained_layout=True)
    ax = axs[0, 0]; ax.imshow(image(z, DX), origin='lower', extent=ext)
    lakes = land & (o['lake_depth_m'] > 5)
    ax.imshow(np.ma.masked_where(~lakes, np.ones_like(z)), origin='lower', extent=ext, cmap=ListedColormap(['#5d9ccf']), alpha=.85)
    river = o['river'] & ~lakes
    ax.imshow(np.ma.masked_where(~river, np.log10(o['drainage_area_m2'])), origin='lower', extent=ext, cmap='Blues', vmin=9.5, vmax=12.5)
    ax.set_title('Elevation, rivers (≥20,000 km² upstream) and lakes / closed basins')
    ax = axs[0, 1]
    u, inv = np.unique(o['outlet'], return_inverse=True)
    colours = np.random.default_rng(0).permutation(len(u))[inv].reshape(z.shape) % 20
    ax.imshow(np.ma.masked_where(~land, colours), origin='lower', extent=ext, cmap='tab20', interpolation='nearest')
    closed = land & ~o['drains_to_sea'] & ~o['drains_off_window']
    ax.imshow(np.ma.masked_where(~closed, np.ones_like(z)), origin='lower', extent=ext, cmap=ListedColormap(['#222222']), alpha=.55)
    ax.set_title('Drainage basins (colours); dark = closed (endorheic) basins')
    ax = axs[0, 2]
    im = ax.imshow(uplift*1e3, origin='lower', extent=ext, cmap='RdBu_r', vmin=-6, vmax=6)
    fig.colorbar(im, ax=ax, fraction=.046, label='rock uplift, mm/yr (final)'); ax.set_title('Tectonic forcing (mechanism, not landform)')
    for k, (label, s) in enumerate(snaps[:2]):
        axs[1, k].imshow(image(s, DX), origin='lower', extent=ext); axs[1, k].set_title(label)
    ax = axs[1, 2]; h = 32*40.32/2
    ax.imshow(image(z40[16:48, 16:48], 40320.), origin='lower', extent=[-h, h, -h, h], interpolation='nearest')
    ax.set_title('Central 1,290 km at 40.32 km (metrics grid)\n' +
                 f"long ridges {metrics['long_ridge_fraction']:.2f}, flanked lowland {metrics['flanked_low_fraction']:.2f}, "
                 f"connected lows {metrics['connected_low_fraction']:.2f}", fontsize=9)
    for a in axs.flat:
        a.set_xlabel('km'); a.set_ylabel('km')
    fig.suptitle(title, fontsize=12); legend(fig, axs[1]); fig.savefig(path, dpi=70); plt.close(fig)


def run_one(scenario, seed, years, out, params, inherited=True):
    z0, uplift, erod, meta = tt.build(scenario, N, DX, seed, inherited)
    years = years or meta['years']
    t0 = time.perf_counter()
    model = le.Landscape(z0, DX, uplift, params, erod)
    snaps = []
    young = meta['young_start_yr']
    if young > 0:
        model.run(young)
        snaps.append((f'{young/1e6:.0f} Myr: inherited landscape before the young events', model.z.reshape(N, N).copy()))
        model.run((years-young)/2)
    else:
        model.run(years/2)
    snaps.append((f'{model.t/1e6:.0f} Myr', model.z.reshape(N, N).copy()))
    model.run(years-model.t)
    o = model.outputs()
    o['drains_off_window'] = model.edge.reshape(N, N)[np.unravel_index(o['outlet'], (N, N))] & ~o['ocean'] & ~o['drains_to_sea']
    z = o['elevation_m']; land = ~o['ocean']
    z23 = to23(z); z40 = to40(z)
    m40 = am.arrangement(z40, z40)
    final_u = uplift(years) if callable(uplift) else uplift
    stem = f'{scenario}-s{seed}'
    np.savez_compressed(out/f'{stem}.npz', **o, elevation23_m=z23, elevation40_m=z40, uplift_mm_yr=(final_u*1e3).astype(np.float32),
                        initial_m=z0.astype(np.float32))
    areas = np.bincount(o['outlet'][land].ravel())*DX**2/1e6
    stats = {'scenario': scenario, 'seed': seed, 'years': years, 'dx_m': DX, 'seconds': time.perf_counter()-t0, 'template': meta,
             'params': model.p, 'numba': le.HAVE_NUMBA,
             'land_fraction': float(land.mean()), 'land_elevation_p50_p95_max_m': [float(np.percentile(z[land], 50)), float(np.percentile(z[land], 95)), float(z.max())],
             'land_draining_to_sea': float(o['drains_to_sea'][land].mean()), 'land_draining_off_window': float(o['drains_off_window'][land].mean()),
             'land_in_closed_basins': float((~o['drains_to_sea'] & ~o['drains_off_window'])[land].mean()),
             'basins_over_10000_km2': int((areas > 1e4).sum()), 'largest_basin_km2': float(areas.max()),
             'flexure_alpha_km': model.flex.alpha_m/1e3, 'volumes_m3': model.totals, 'metrics_40km': m40}
    snaps = [(f'{tt_:s}', s) for tt_, s in snaps]
    figure(out/f'{stem}.png', f'{scenario}, seed {seed}: {years/1e6:.0f} Myr ({f"{young/1e6:.0f} Myr inherited history + " if young else ""}young events); '
           f'uplift, rivers, sediment, flexure; window 2,949 km at {DX/1e3:.2f} km',
           o, final_u, snaps, z40, m40)
    return stats, z23, land


@torch.no_grad()
def run_meso(out, stem, z23, land, device, steps, checkpoint=MESO):
    from toy_meso.common import load_checkpoint, sample_patches
    meso, mnorm, _ = load_checkpoint(checkpoint, device)
    wins = meso_windows(z23, to23(land.astype(np.float32)) > .5)
    rows = []
    fig, axs = plt.subplots(len(MESO_SEEDS)+1, len(wins), figsize=(4.2*len(wins), 4.3*(len(MESO_SEEDS)+1)), squeeze=False, constrained_layout=True)
    for j, (name, (r, c)) in enumerate(wins.items()):
        cond = encode(torch.from_numpy(z23[r:r+18, c:c+18].copy())[None, None]).float()
        h = 16*23.04/2
        axs[0, j].imshow(image(z23[r+1:r+17, c+1:c+17], 23040.), origin='lower', extent=[-h, h, -h, h], interpolation='nearest')
        axs[0, j].set_title(f'{name}: 23 km input', fontsize=9)
        for i, s in enumerate(MESO_SEEDS, 1):
            field = decode(sample_patches(meso, mnorm, cond, steps, s, device)[0])[0, 0].cpu().numpy()
            np.savez_compressed(out/f'{stem}-meso-{name}-{s}.npz', raw_meso_metres=field, meso_input_encoded=cond.numpy(), window_rc=[r, c])
            axs[i, j].imshow(image(field, 2880.), origin='lower', extent=[-h, h, -h, h])
            axs[i, j].set_title(f'{name}: meso seed {s}, 2.88 km', fontsize=9)
            rows.append({'window': name, 'row_col_23km': [int(r), int(c)], 'meso_seed': s})
    fig.suptitle(f'{stem}: meso ({Path(checkpoint).parent.name}) on simulated 23 km terrain; each panel 368.64 km', fontsize=11)
    legend(fig, axs); fig.savefig(out/f'{stem}-meso.png', dpi=80); plt.close(fig)
    return rows


def write_html(out, runs, earth):
    keys = ['long_ridge_fraction', 'flanked_low_fraction', 'connected_low_fraction', 'contrast_p50_m', 'slope_p90_m_per_km']
    head = ''.join(f'<th>{k}</th>' for k in keys)
    rows = ''.join(f'<tr><td>Earth: {k}</td>'+''.join(f'<td>{v[c]:.3g}</td>' for c in keys)+'</tr>' for k, v in earth.items())
    rows += ''.join(f'<tr><td>{r["scenario"]} s{r["seed"]}</td>'+''.join(f'<td>{r["metrics_40km"][c]:.3g}</td>' for c in keys)+'</tr>' for r in runs)
    drain = ''.join(f'<tr><td>{r["scenario"]} s{r["seed"]}</td><td>{r["land_fraction"]:.0%}</td><td>{r["land_elevation_p50_p95_max_m"][0]:.0f} / {r["land_elevation_p50_p95_max_m"][1]:.0f} / {r["land_elevation_p50_p95_max_m"][2]:.0f}</td>'
                    f'<td>{r["land_draining_to_sea"]:.0%}</td><td>{r["land_draining_off_window"]:.0%}</td><td>{r["land_in_closed_basins"]:.0%}</td><td>{r["basins_over_10000_km2"]}</td><td>{r["seconds"]:.0f}</td></tr>' for r in runs)
    imgs = ''.join(f'<h3>{r["scenario"]}, seed {r["seed"]}</h3><img src="{r["scenario"]}-s{r["seed"]}.png">' +
                   (f'<img src="{r["scenario"]}-s{r["seed"]}-meso.png">' if (out/f'{r["scenario"]}-s{r["seed"]}-meso.png').exists() else '') for r in runs)
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Tectonic prototype</title><style>
body{{font:15px system-ui;margin:24px;background:#f6f6f4;color:#222}}img{{width:100%;max-width:1500px;display:block;margin:8px 0 24px}}
table{{border-collapse:collapse;margin:10px 0 24px}}td,th{{border:1px solid #bbb;padding:4px 8px}}p{{max-width:1100px;line-height:1.5}}</style></head><body>
<h1>Regional tectonic + landscape-evolution prototype</h1>
<p>Mechanism-driven: tectonic uplift fields, stream-power erosion with sediment deposition, basin filling and flexural isostasy, at {DX/1e3:.2f} km.
No training and no constraints; sea level is only the 0 m datum. Rivers, drainage basins and closed basins come from the same flow routing that shapes the terrain.</p>
<h2>Arrangement at 40.32 km (central 1,290 km for the map, 1,935 km for metrics)</h2><table><tr><th></th>{head}</tr>{rows}</table>
<h2>Elevation and drainage</h2><table><tr><th></th><th>land</th><th>land p50 / p95 / max (m)</th><th>drains to sea</th><th>drains off window</th><th>closed basins</th><th>basins &gt;10⁴ km²</th><th>run s</th></tr>{drain}</table>
{imgs}</body></html>'''
    (out/'index.html').write_text(html, encoding='utf-8')


def main(argv=None):
    global N, DX
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarios', default='collision,margin')
    p.add_argument('--seeds', default='1,2,3')
    p.add_argument('--years', type=float, help='total simulated years (default: the template\'s, 40 Myr with inherited history)')
    p.add_argument('--no-inherited', action='store_true', help='start the young events on a flat plain (the first prototype)')
    p.add_argument('--param', action='append', default=[], help='override a landscape parameter, e.g. --param Sc=0.04')
    p.add_argument('--meso', action='store_true', help='refine four 368 km windows per run with the meso model')
    p.add_argument('--meso-checkpoint', default=str(MESO), help='meso checkpoint (default: detail-aware fine-tune; original: outputs/meso/run2-lowlr/latest.pt)')
    p.add_argument('--meso-steps', type=int, default=32)
    p.add_argument('--device', default='auto')
    p.add_argument('--dx', type=float, default=DX, help='simulation cell size in m: 11520 (default) or 5760; the window stays 2,949 km')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if int(round(23040/a.dx))*a.dx != 23040:
        raise ValueError('--dx must divide 23,040 m (e.g. 11520 or 5760)')
    N, DX = int(round(256*11520/a.dx)), float(a.dx)
    out = Path(a.out) if a.out else OUT_ROOT/datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    params = {}
    for item in a.param:
        k, v = item.split('='); params[k] = type(le.DEFAULTS[k])(v) if not isinstance(le.DEFAULTS[k], bool) else v.lower() in ('1', 'true', 'yes')
    device = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if not le.HAVE_NUMBA:
        print('numba not installed: the simulation runs in plain Python and is much slower. Install with: pip install numba', flush=True)
    if a.meso and not Path(a.meso_checkpoint).exists():
        raise FileNotFoundError(f'Meso checkpoint not found: {a.meso_checkpoint}')
    earth = earth_reference_metrics(); runs = []
    for scenario in a.scenarios.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            stats, z23, land = run_one(scenario, seed, a.years, out, params, not a.no_inherited)
            if a.meso:
                stats['meso'] = run_meso(out, f'{scenario}-s{seed}', z23, land, device, a.meso_steps, a.meso_checkpoint)
            runs.append(stats)
            print(f"{scenario} s{seed}: {stats['seconds']:.0f}s, land {stats['land_fraction']:.0%}, p95 {stats['land_elevation_p50_p95_max_m'][1]:.0f} m, "
                  f"to sea {stats['land_draining_to_sea']:.0%}, closed {stats['land_in_closed_basins']:.0%}", flush=True)
            (out/'summary.json').write_text(json.dumps({'runs': runs, 'earth_40km': earth, 'metrics_spec': am.SPEC}, indent=1, default=float), encoding='utf-8')
    write_html(out, runs, earth)
    print(f'Done: {out/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
