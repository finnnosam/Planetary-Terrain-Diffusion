"""Step 8, first pass: hand the planet's 11.53 km landscape to the meso model (23.04 km -> 2.88 km) on windows and
compare everything with Earth (docs/planet-layer-plan.md, step 8).

  python -m scripts.planet_meso_check                         # newest landscape run made with the default theta
  python -m scripts.planet_meso_check --landscape outputs\\planet-landscape\\theta-...\\theta20 --per-class 16

Hand-off. The 23.04 km input is built exactly as toy_meso.planet_tiled builds it for the model: per face of the
n = 434 cube, each cell is the mean of 8 x 8 sample points of the landscape surface over the fine nodes 8c..8c+7
(cell centres at c + 7/16). The landscape grid (n = 868) is exactly twice as fine, so no resampling to another grid
is involved; the surface is read bilinearly from its own cube nodes. Windows are 18 x 18 cells (16 x 16 core =
368.64 km), cut from the face charts as the tiled generator does, with stride 9.

Earth reference: the meso's held-out validation split of outputs/meso/dataset.npz (val_cond, val_target), the data
the model was judged on. Windows are classed the same way on both sides (land >= 98% of the core; core p95-p5 relief):
flat < 300 m, plain/upland 300-800, hills 800-2000, mountains >= 2000; coast = land 30-70% of the core; and
plains-lowlands = flat windows with a core median elevation below 200 m (a subset of flat, reported alongside).

Measured, with definitions fixed before any planet window is seen:
  input (23 km, no model): share of land windows per class; relief p10/25/50/75 and shares < 300 m / < 100 m;
    cell-to-cell curvature = median |5-point Laplacian| (m) over the core, median over windows per class.
  output (2.88 km, meso step10000-passed, detail unknown, seeds 113 and 997, 32 steps): the diagnose_meso_plains
    metrics (band power 5.8-11.5 / 11.5-23 / 23-46 km, river incision, roughness), meso on planet windows and meso on
    Earth windows, both against real Earth 2.88 km in the same class.
  rule (same as diagnose_meso_plains, applied to the planet): a class passes if meso-on-planet incision and
    5.8-11.5 km power are each >= 0.7 of real Earth; above 2.0 is flagged as over-textured.

Writes outputs/planet-meso-check/<timestamp>/: index.html (tables and pictures), results.json, windows.npz.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'outputs/meso/dataset.npz'
MESO = ROOT/'outputs/meso/detail-ft/step10000-passed.pt'
OUT_ROOT = ROOT/'outputs/planet-meso-check'
CLASSES = [('flat', 0, 300), ('plain-upland', 300, 800), ('hills', 800, 2000), ('mountains', 2000, 1e9)]
COAST = ('coast', .3, .7)
# lowland plains (suggested by the user, 18 Oct): flat windows whose core median elevation is below 200 m (the
# Amazon / West Siberian / Pampas kind), reported beside the classes above; they are a subset of 'flat'
LOWLAND = ('plains-lowlands', 300., 200.)
N_COARSE, FACTOR, WIN, STRIDE = 434, 8, 18, 9
RULE = {'incision_ratio_min': .7, 'band_5_11_ratio_min': .7, 'over_textured_above': 2.}


def latest_landscape(theta=None):
    """The newest theta calibration's chosen run if there is one, else the newest run made with `theta`."""
    cal = sorted((ROOT/'outputs'/'planet-landscape').glob('theta-*/summary.json'), key=lambda f: f.stat().st_mtime)
    for f in reversed(cal):
        ch = json.loads(f.read_text()).get('chosen_theta')
        if ch is not None and (f.parent/f'theta{ch:g}'/'present.npz').exists():
            return f.parent/f'theta{ch:g}'
    return latest_run_with_theta(theta)


def latest_run_with_theta(theta=None):
    runs = []
    for f in (ROOT/'outputs'/'planet-landscape').rglob('present.npz'):
        st = f.parent/'stats.json'
        if not st.exists():
            continue
        th = json.loads(st.read_text()).get('theta', 0.)
        if theta is None or abs(th-theta) < 1e-9:
            runs.append((f.stat().st_mtime, f.parent))
    if not runs:
        raise FileNotFoundError('no finished landscape run (present.npz + stats.json) under outputs/planet-landscape')
    return max(runs)[1]


def cells23(face_field, n=N_COARSE, factor=FACTOR, sub=8):
    """(6, N+1, N+1) node field of the fine cube -> (6, n, n) cell means, exactly as toy_meso.planet_tiled."""
    from planet_diffusion import cube
    centre = (factor-1)/(2*factor)
    offsets = (np.arange(sub)+.5)/sub-.5
    out = np.empty((6, n, n), np.float32)
    src = face_field[None].astype(np.float32)
    c = np.arange(n)
    for f in range(6):
        for r0 in range(0, n, 16):
            ys = c[r0:r0+16]
            y = ys[:, None, None, None]+centre+offsets[None, None, :, None]
            x = c[None, :, None, None]+centre+offsets[None, None, None, :]
            y, x = np.broadcast_arrays(y, x)
            p = cube.directions(f, y, x, n)
            out[f, r0:r0+len(ys)] = cube.sample(src, p)[0].mean(axis=(-2, -1))
    return out


def windows_of(z23):
    """All 18 x 18 windows of the face charts, stride 9: (face, i, j) and the metre arrays."""
    keys, wins = [], []
    for f in range(6):
        for i in range(0, z23.shape[1]-WIN+1, STRIDE):
            for j in range(0, z23.shape[2]-WIN+1, STRIDE):
                keys.append((f, i, j)); wins.append(z23[f, i:i+WIN, j:j+WIN])
    return keys, np.stack(wins)


def window_stats(w):
    """w (M, 18, 18) metres -> land fraction, p95-p5 relief, median |Laplacian| of the 16 x 16 core."""
    core = w[:, 1:17, 1:17]
    land = (core > 0).mean((1, 2))
    rel = np.percentile(core, 95, axis=(1, 2))-np.percentile(core, 5, axis=(1, 2))
    lap = w[:, :-2, 1:-1]+w[:, 2:, 1:-1]+w[:, 1:-1, :-2]+w[:, 1:-1, 2:]-4*w[:, 1:-1, 1:-1]
    return land, rel, np.median(np.abs(lap), axis=(1, 2))


def core_median(w):
    return np.median(w[:, 1:17, 1:17], axis=(1, 2))


def classify(land, rel, elev=None):
    out = {}
    for name, lo, hi in CLASSES:
        out[name] = np.flatnonzero((land >= .98) & (rel >= lo) & (rel < hi))
    out[COAST[0]] = np.flatnonzero((land >= COAST[1]) & (land <= COAST[2]))
    if elev is not None:
        out[LOWLAND[0]] = np.flatnonzero((land >= .98) & (rel < LOWLAND[1]) & (elev < LOWLAND[2]))
    return out


def input_table(land, rel, curv, cls):
    lw = land >= .98
    r = rel[lw]
    t = {'land_windows': int(lw.sum()),
         'relief_percentiles_m': {str(q): float(np.percentile(r, q)) for q in (10, 25, 50, 75, 90)} if lw.any() else {},
         'relief_below_300_share': float((r < 300).mean()) if lw.any() else None,
         'relief_below_100_share': float((r < 100).mean()) if lw.any() else None,
         'class_share_of_land_windows': {k: float(len(v)/max(lw.sum(), 1)) for k, v in cls.items() if k != 'coast'},  # lowlands are a subset of flat
         'curvature_median_m': {k: (float(np.median(curv[v])) if len(v) else None) for k, v in cls.items()}}
    return t


def main(argv=None):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from planet_layer.sphere_graph import CubeSphere
    from planet_training.elevation_shading import image, legend
    from planet_training.terrain_structure import decode, encode
    from scripts.diagnose_meso_plains import metrics, pick
    from scripts.run_planet_landscape import THETA
    from toy_meso.common import load_checkpoint, sample_patches

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--landscape', default='latest', help="landscape run folder with present.npz, or 'latest' (the newest "
                   "theta calibration's chosen run, else the newest run at the default theta)")
    p.add_argument('--meso', default=str(MESO))
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--per-class', type=int, default=16)
    p.add_argument('--seeds', default='113,997')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='auto')
    p.add_argument('--out')
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    run = latest_landscape(THETA) if a.landscape == 'latest' else Path(a.landscape)
    out = Path(a.out) if a.out else OUT_ROOT/datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    device = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'landscape {run}; device {device}', flush=True)

    # hand-off: 11.53 km cube nodes -> 23.04 km cells of the meso's own cube charts
    pres = np.load(run/'present.npz')
    n_fine = int(pres['n'])
    if n_fine != N_COARSE*2:
        raise SystemExit(f'landscape grid n={n_fine}; the exact hand-off needs n={2*N_COARSE}')
    g = CubeSphere(n_fine)
    fine = g.to_cube(pres['elevation_m'].astype(np.float32))                # (6, 869, 869)
    z23 = cells23(fine)
    keys, wins = windows_of(z23)
    pl_land, pl_rel, pl_curv = window_stats(wins)
    pl_cls = classify(pl_land, pl_rel, core_median(wins))
    print(f'hand-off: {len(keys)} planet windows ({time.perf_counter()-t0:.0f} s)', flush=True)

    d = np.load(a.data, mmap_mode='r')
    e_cond = np.asarray(d['val_cond'], np.float32)
    e_m = decode(torch.from_numpy(e_cond).double()).numpy()
    e_land, e_rel, e_curv = window_stats(e_m)
    e_cls = classify(e_land, e_rel, core_median(e_m))
    inputs = {'planet': input_table(pl_land, pl_rel, pl_curv, pl_cls), 'earth_meso_val': input_table(e_land, e_rel, e_curv, e_cls),
              'earth_doc_reference': {'relief_percentiles_m': {'10': 133, '25': 237, '50': 456, '75': 872},
                                      'relief_below_300_share': .34, 'relief_below_100_share': .06,
                                      'curvature_median_m': {'flat': 31, 'plain-upland': 68, 'hills': 179},
                                      'source': 'docs/tectonic-prototype.md (intermediate dataset |lat| <= 60; meso validation input)'}}

    model, norm, ck = load_checkpoint(a.meso, device)
    rows, examples, chosen = [], {}, {}
    names = [c[0] for c in CLASSES]+[LOWLAND[0], COAST[0]]

    def run_meso(cond_m):
        c = encode(torch.from_numpy(np.asarray(cond_m, np.float32)).double()).float()[:, None]
        res = {s: [] for s in seeds}
        for s in seeds:
            for b0 in range(0, len(c), a.batch):
                gen = sample_patches(model, norm, c[b0:b0+a.batch], a.steps, s+b0, device)[0]
                res[s].append(decode(gen.double().cpu())[:, 0].numpy())
        return {s: np.concatenate(v) for s, v in res.items()}

    for name in names:
        pi = pick(list(pl_cls[name]), a.per_class)
        ei = pick(list(e_cls[name]), a.per_class)
        chosen[name] = {'planet': [list(map(int, keys[k])) for k in pi], 'earth_val_index': [int(i) for i in ei]}
        if pi:
            gen = run_meso(wins[pi])
            for s in seeds:
                rows += [{'source': 'planet-meso', 'cls': name, 'm': metrics(z)} for z in gen[s]]
        if ei:
            real = decode(torch.from_numpy(np.asarray(d['val_target'][ei], np.float32)).double()).numpy()
            rows += [{'source': 'earth-real', 'cls': name, 'm': metrics(z)} for z in real]
            egen = run_meso(e_m[ei])
            for s in seeds:
                rows += [{'source': 'earth-meso', 'cls': name, 'm': metrics(z)} for z in egen[s]]
        ex = []
        for k in range(min(3, len(pi))):
            f, i, j = keys[pi[k]]
            ex.append({'planet23': wins[pi[k]], 'planet11': fine[f, 2*i:2*i+2*WIN+1, 2*j:2*j+2*WIN+1],
                       'planet_meso': gen[seeds[0]][k],
                       'earth23': e_m[ei[k]] if k < len(ei) else None, 'earth_real': real[k] if k < len(ei) else None,
                       'earth_meso': egen[seeds[0]][k] if k < len(ei) else None})
        examples[name] = ex
        print(f'{name}: {len(pi)} planet windows (of {len(pl_cls[name])}), {len(ei)} Earth windows (of {len(e_cls[name])}), '
              f'{time.perf_counter()-t0:.0f} s', flush=True)

    summary = {}
    for src in ('planet-meso', 'earth-meso', 'earth-real'):
        for name in names:
            rs = [r for r in rows if r['source'] == src and r['cls'] == name]
            if rs:
                summary[f'{src}/{name}'] = {'n': len(rs), 'bands': np.mean([r['m']['bands'] for r in rs], 0).tolist(),
                                            'incision_m': float(np.mean([r['m']['incision_m'] for r in rs])),
                                            'roughness_m': float(np.mean([r['m']['roughness_m'] for r in rs]))}
    verdict = {}
    for name in names:
        r, gp, ge = summary.get(f'earth-real/{name}'), summary.get(f'planet-meso/{name}'), summary.get(f'earth-meso/{name}')
        if not (r and gp):
            continue
        v = {'incision_ratio': gp['incision_m']/max(r['incision_m'], 1e-6), 'band_5_11_ratio': gp['bands'][0]/max(r['bands'][0], 1e-12),
             'band_11_23_ratio': gp['bands'][1]/max(r['bands'][1], 1e-12), 'roughness_ratio': gp['roughness_m']/max(r['roughness_m'], 1e-6)}
        if ge:
            v['earth_meso_incision_ratio'] = ge['incision_m']/max(r['incision_m'], 1e-6)
            v['earth_meso_band_5_11_ratio'] = ge['bands'][0]/max(r['bands'][0], 1e-12)
        v['passes'] = v['incision_ratio'] >= RULE['incision_ratio_min'] and v['band_5_11_ratio'] >= RULE['band_5_11_ratio_min']
        v['over_textured'] = v['incision_ratio'] > RULE['over_textured_above'] or v['band_5_11_ratio'] > RULE['over_textured_above']
        verdict[name] = v
    result = {'landscape': str(run), 'meso': a.meso, 'rule': RULE, 'inputs': inputs, 'summary': summary, 'verdict': verdict,
              'windows': chosen, 'seconds': time.perf_counter()-t0}
    (out/'results.json').write_text(json.dumps(result, indent=1, default=float), encoding='utf-8')
    np.savez_compressed(out/'windows.npz', z23=z23, keys=np.array(keys), relief=pl_rel, land=pl_land, curvature=pl_curv)

    # pictures
    h = 368.64/2
    for name, exs in examples.items():
        if not exs:
            continue
        cols = [('planet23', 'planet 23 km input', 23040.), ('planet11', 'planet 11.53 km landscape', 11520.),
                ('planet_meso', 'meso on planet, 2.88 km', 2880.), ('earth23', 'Earth 23 km input', 23040.),
                ('earth_real', 'real Earth 2.88 km', 2880.), ('earth_meso', 'meso on Earth input', 2880.)]
        fig, axs = plt.subplots(len(exs), len(cols), figsize=(3.4*len(cols), 3.5*len(exs)), squeeze=False, constrained_layout=True)
        for r_, ex in enumerate(exs):
            for k, (key, title, sp) in enumerate(cols):
                ax = axs[r_, k]; ax.set_xticks([]); ax.set_yticks([])
                z = ex[key]
                if z is None:
                    ax.axis('off'); continue
                if key.endswith('23'):
                    z = z[1:17, 1:17]
                elif key == 'planet11':
                    z = z[2:34, 2:34]
                ax.imshow(image(np.asarray(z, np.float64), sp), origin='lower', extent=[-h, h, -h, h],
                          interpolation='nearest' if sp > 3000 else 'bilinear')
                ax.set_title(title, fontsize=9)
        fig.suptitle(f'{name}: first three planet windows and Earth validation windows of the same class; each 368.64 km', fontsize=11)
        legend(fig, axs); fig.savefig(out/f'{name}.png', dpi=75); plt.close(fig)

    # overview map of the windows used
    from planet_diffusion import cube
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    H = 540
    lat = np.radians(90-(np.arange(H)+.5)*180/H); lon = np.radians(-180+(np.arange(2*H)+.5)*180/H)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    pts = np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), -1)
    ax.imshow(image(cube.sample(fine[None], pts)[0].astype(np.float64), 2e7/H*1.0), extent=[-180, 180, -90, 90])
    colors = dict(zip(names, ('#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#8ab17d', '#264653')))
    for name in names:
        for f, i, j in chosen[name]['planet']:
            c = cube.directions(f, np.array(i+9.)/1., np.array(j+9.)/1., N_COARSE)
            ax.plot(np.degrees(np.arctan2(c[1], c[0])), np.degrees(np.arcsin(c[2]/np.linalg.norm(c))), 's', mfc='none',
                    mec=colors[name], ms=6, mew=1.5, label=name)
    hd, lb = ax.get_legend_handles_labels(); uniq = dict(zip(lb, hd))
    ax.legend(uniq.values(), uniq.keys(), loc='lower left', fontsize=9)
    ax.set_title('planet at 11.53 km with the meso windows (squares)'); fig.savefig(out/'windows.png', dpi=80); plt.close(fig)

    # report
    def fmt(v, nd=0):
        return '—' if v is None else f'{v:.{nd}f}'
    pin, ein = inputs['planet'], inputs['earth_meso_val']
    rows_in = ''.join(f"<tr><td>{c}</td><td>{fmt(pin['class_share_of_land_windows'].get(c), 2)}</td><td>{fmt(ein['class_share_of_land_windows'].get(c), 2)}</td>"
                      f"<td>{fmt(pin['curvature_median_m'].get(c))}</td><td>{fmt(ein['curvature_median_m'].get(c))}</td></tr>" for c in names)
    rel = lambda t: ' / '.join(fmt(t['relief_percentiles_m'].get(q)) for q in ('10', '25', '50', '75', '90'))
    rows_out = ''.join(f"<tr><td>{c}</td><td>{fmt(v['incision_ratio'], 2)}</td><td>{fmt(v['band_5_11_ratio'], 2)}</td><td>{fmt(v['band_11_23_ratio'], 2)}</td>"
                       f"<td>{fmt(v['roughness_ratio'], 2)}</td><td>{fmt(v.get('earth_meso_incision_ratio'), 2)} / {fmt(v.get('earth_meso_band_5_11_ratio'), 2)}</td>"
                       f"<td>{'pass' if v['passes'] else 'FAIL'}{' (over-textured)' if v['over_textured'] else ''}</td></tr>" for c, v in verdict.items())
    raw = ''.join(f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['incision_m']:.1f}</td><td>{v['roughness_m']:.1f}</td>" + ''.join(f'<td>{b:.3g}</td>' for b in v['bands']) + '</tr>'
                  for k, v in summary.items())
    imgs = ''.join(f'<h3>{c}</h3><img src="{c}.png">' for c in names if examples.get(c))
    (out/'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Planet meso check</title><style>
body{{font:15px system-ui;margin:24px;background:#f6f6f4;color:#222}}img{{max-width:1400px;width:100%}}td,th{{border:1px solid #bbb;padding:4px 8px;text-align:right}}
td:first-child{{text-align:left}}table{{border-collapse:collapse;margin:8px 0 18px}}</style></head><body>
<h1>Planet → meso (23.04 → 2.88 km)</h1><p>Landscape <code>{run}</code>; meso <code>{a.meso}</code>, detail unknown, seeds {seeds}.</p>
<h2>23 km input (no model)</h2>
<p>368 km land-window relief p10/25/50/75/90: planet {rel(pin)} m; Earth (meso validation) {rel(ein)} m; Earth (doc reference) 133 / 237 / 456 / 872 m.
Share of land windows with relief &lt; 300 m: planet {fmt(pin['relief_below_300_share'], 2)}, Earth val {fmt(ein['relief_below_300_share'], 2)}, doc 0.34;
&lt; 100 m: {fmt(pin['relief_below_100_share'], 2)} / {fmt(ein['relief_below_100_share'], 2)} / 0.06.</p>
<table><tr><th>class</th><th>planet share</th><th>Earth share</th><th>planet curvature m</th><th>Earth curvature m</th></tr>{rows_in}</table>
<p>The Earth validation split comes from held-out boxes (Scandinavia, Tibet/Himalaya, Patagonia, ...), so its class shares are
not the whole Earth's; the curvature per class is the comparable number.</p>
<h2>2.88 km output, generated ÷ real Earth in the same class</h2>
<table><tr><th>class</th><th>incision</th><th>5.8–11.5 km power</th><th>11.5–23 km power</th><th>roughness</th><th>meso on Earth input: incision / 5.8–11.5</th><th>rule</th></tr>{rows_out}</table>
<p>Rule: incision and 5.8–11.5 km power ≥ 0.7 of real Earth; above 2 is over-textured.</p>
<details><summary>raw numbers</summary><table><tr><th>source/class</th><th>n</th><th>incision m</th><th>roughness m</th><th>5.8–11.5</th><th>11.5–23</th><th>23–46</th></tr>{raw}</table></details>
<h2>Where the windows are</h2><img src="windows.png">
<h2>Examples</h2>{imgs}</body></html>''', encoding='utf-8')
    print(json.dumps({'inputs_planet': {k: pin[k] for k in ('relief_percentiles_m', 'relief_below_300_share', 'curvature_median_m')},
                      'verdict': verdict}, indent=1, default=float))
    print(f'Done in {time.perf_counter()-t0:.0f} s: {out/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
