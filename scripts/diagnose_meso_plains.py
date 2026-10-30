"""Is the meso model the reason simulated plains look smooth? A held-out Earth test, no training.

Runs the meso model on held-out Earth patches (where the true 2.88 km terrain is known) and on
windows of a simulated region, grouped by 23 km relief class, and measures:

  band power     radially averaged power of the detrended field in 5.8-11.5, 11.5-23 and 23-46 km
                 wavelength bands, generated / real
  incision       how far river channels (>= 300 km^2 upstream, D8 on the filled surface) sit below
                 the surrounding terrain (7-cell Gaussian mean minus channel height), metres
  roughness      std of the field minus its 2-cell Gaussian smoothing, metres

Decision rule, fixed before any result is seen:
  * MODEL IS ADEQUATE for plains if, on Earth flat and plain/upland patches, meso/real >= 0.7 for both
    incision and 5.8-11.5 km band power. Then smooth simulated plains are an input problem: fix the
    simulation (finer-scale erosion), not the model.
  * Otherwise there is room for a new model; it must beat this one on the same held-out test first.

  python -m scripts.diagnose_meso_plains                       # the simulation's meso, newest simulation run
  python -m scripts.diagnose_meso_plains --compare <another meso checkpoint>
  python -m scripts.diagnose_meso_plains --blur-earth 0.7     # also: meso on Earth 23 km input softened like the simulation

Softened-input test (--blur-earth). The simulated 23 km fields have about half (plains) to a third (hills) of Earth's
cell-to-cell curvature at the same relief, about what a 0.7-cell Gaussian blur does to Earth's input. Running the
meso on blurred Earth input shows how much of its texture depends on that fine cue. Rule for a meso that is robust
to softer input, fixed before testing one: on blurred (0.7) Earth flat and plain/upland input, incision and
5.8-11.5 km power >= 0.7 of real, while the plain test above still passes. For a detail-aware meso
(toy_meso.finetune_detail) the softened test also runs with the true sub-cell detail supplied ('-detail' rows).

First result (20261015-193134): the adapted meso fails (flat / plain incision 0.34 / 0.57 of real), the
original Earth-trained run2-lowlr passes (1.00 / 1.05; 5.8-11.5 km power 1.06 / 1.73).
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

from planet_training import landscape_evolution as le
from planet_training.elevation_shading import image, legend
from planet_training.terrain_structure import blur, decode, encode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'outputs/meso/dataset.npz'
MESO = ROOT/'outputs/meso/detail-ft/step10000-passed.pt'   # the meso the simulation uses (detail-aware fine-tune)
ADAPTED = ROOT/'outputs/structure/joint-structure-v1/review-20261014-143545-003794/meso-checkpoint.pt'
SIM_ROOT = ROOT/'outputs/tectonic-prototype'
OUT_ROOT = ROOT/'outputs/meso-plains-diagnosis'
CLASSES = [('flat', 0, 300), ('plain-upland', 300, 800), ('hills', 800, 2000)]
BANDS_KM = [(5.76, 11.52), (11.52, 23.04), (23.04, 46.08)]
SPACING = 2880.
RULE = {'classes': ['flat', 'plain-upland'], 'incision_ratio_min': .7, 'band_5_11_ratio_min': .7}


def relief23(cond_encoded):
    core = decode(torch.as_tensor(cond_encoded, dtype=torch.float64))[..., 1:17, 1:17].numpy()
    land = (core > 0).mean((-2, -1))
    rel = np.percentile(core, 95, axis=(-2, -1))-np.percentile(core, 5, axis=(-2, -1))
    return land, rel


def pick(indices, count):
    if len(indices) <= count:
        return list(indices)
    return [indices[int(round(q))] for q in np.linspace(0, len(indices)-1, count)]


def band_power(z):
    n = z.shape[0]
    y, x = np.mgrid[:n, :n]
    a = np.c_[x.ravel(), y.ravel(), np.ones(n*n)]
    plane = a@np.linalg.lstsq(a, z.ravel(), rcond=None)[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fft2((z.ravel()-plane).reshape(n, n)*w))**2
    f = np.sqrt(np.fft.fftfreq(n)[:, None]**2+np.fft.fftfreq(n)[None, :]**2)
    wl = np.where(f > 0, SPACING/1e3/np.maximum(f, 1e-9), np.inf)
    return [float(p[(wl >= lo) & (wl < hi)].sum()) for lo, hi in BANDS_KM]


def incision_and_roughness(z):
    n = z.shape[0]
    zz = z.astype(np.float64).ravel()
    edge = np.zeros((n, n), bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
    zf = le.priority_flood(zz, edge.ravel(), n, n, 1e-3)
    rec, dist, stack = le.route(zf, edge.ravel(), n, n, SPACING)
    area = le.accumulate(rec, stack, SPACING**2).reshape(n, n)
    t = torch.from_numpy(z.astype(np.float32))[None, None]
    smooth7 = blur(t, 7.)[0, 0].numpy(); smooth2 = blur(t, 2.)[0, 0].numpy()
    inner = np.zeros((n, n), bool); inner[8:-8, 8:-8] = True
    channel = (area >= 300e6) & inner
    incision = float((smooth7-z)[channel].mean()) if channel.any() else 0.
    return incision, float((z-smooth2)[inner].std())


def metrics(z):
    inc, rough = incision_and_roughness(z)
    return {'bands': band_power(z), 'incision_m': inc, 'roughness_m': rough}


def summarize(rows):
    out = {}
    for key in sorted({(r['source'], r['cls']) for r in rows}):
        rs = [r for r in rows if (r['source'], r['cls']) == key]
        out['/'.join(key)] = {'n': len(rs), 'bands': np.mean([r['m']['bands'] for r in rs], 0).tolist(),
                              'incision_m': float(np.mean([r['m']['incision_m'] for r in rs])),
                              'roughness_m': float(np.mean([r['m']['roughness_m'] for r in rs]))}
    return out


@torch.no_grad()
def main(argv=None):
    from toy_meso.common import load_checkpoint, sample_patches
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--meso', default=str(MESO)); p.add_argument('--compare', help='second meso checkpoint to test the same way')
    p.add_argument('--sim', help='tectonic-prototype run folder (default: newest)')
    p.add_argument('--per-class', type=int, default=24); p.add_argument('--seeds', default='113,997')
    p.add_argument('--steps', type=int, default=32); p.add_argument('--device', default='auto'); p.add_argument('--out')
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--blur-earth', default='', help='comma list of Gaussian sigmas (23 km cells) for the softened-input test, e.g. 0.7')
    a = p.parse_args(argv)
    blurs = [float(x) for x in a.blur_earth.split(',') if x.strip()]
    device = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out) if a.out else OUT_ROOT/datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.perf_counter()
    d = np.load(a.data, mmap_mode='r')
    cond = np.asarray(d['val_cond'], np.float32)
    land, rel = relief23(cond)
    earth = {}
    for name, lo, hi in CLASSES:
        idx = np.flatnonzero((land >= .98) & (rel >= lo) & (rel < hi))
        earth[name] = [int(i) for i in pick(list(idx), a.per_class)]
    sim_dir = Path(a.sim) if a.sim else max((q for q in SIM_ROOT.iterdir() if q.is_dir() and q.name[:2] == '20'), default=None)
    sims = {name: [] for name, _, _ in CLASSES}
    if sim_dir is not None:
        windows = []
        for f in sorted(sim_dir.glob('*-s[0-9].npz')):
            z23 = np.load(f)['elevation23_m']
            for i in range(0, z23.shape[0]-18, 9):
                for j in range(0, z23.shape[1]-18, 9):
                    windows.append((f.stem, i, j, z23[i:i+18, j:j+18].copy()))
        if windows:
            wl, wr = relief23(encode(torch.from_numpy(np.stack([w[3] for w in windows]))).numpy())
            for name, lo, hi in CLASSES:
                idx = [k for k in range(len(windows)) if wl[k] >= .98 and lo <= wr[k] < hi]
                sims[name] = [windows[k] for k in pick(idx, a.per_class)]
    models = {'meso': load_checkpoint(a.meso, device)[:2]}
    if a.compare:
        models['compare'] = load_checkpoint(a.compare, device)[:2]
    settings = {'meso': a.meso, 'compare': a.compare, 'sim_run': str(sim_dir), 'per_class': a.per_class, 'seeds': seeds,
                'steps': a.steps, 'rule': RULE, 'bands_km': BANDS_KM, 'earth_indices': earth,
                'sim_windows': {k: [(w[0], int(w[1]), int(w[2])) for w in v] for k, v in sims.items()}}
    (out/'settings.json').write_text(json.dumps(settings, indent=1), encoding='utf-8')
    rows, examples = [], {}
    for name, _, _ in CLASSES:
        for k, i in enumerate(earth[name]):
            c = torch.from_numpy(cond[i:i+1, None])
            real = decode(torch.from_numpy(np.asarray(d['val_target'][i], np.float32)))[None].numpy()[0]
            rows.append({'source': 'earth-real', 'cls': name, 'm': metrics(real)})
            ex = {'cond': decode(c)[0, 0].numpy(), 'real': real}
            for mname, (model, norm) in models.items():
                for s in seeds:
                    g = decode(sample_patches(model, norm, c, a.steps, s, device)[0])[0, 0].cpu().numpy()
                    rows.append({'source': f'earth-{mname}', 'cls': name, 'm': metrics(g)})
                    if s == seeds[0]:
                        ex[mname] = g
                for sig in blurs:
                    cb = encode(blur(decode(c.double()), sig)).float()
                    variants = [('', None)]
                    if getattr(model, 'cond_channels', 1) > 1:   # detail-aware meso: also give it the true detail
                        from toy_meso.detail import cell_detail
                        tgt = torch.from_numpy(np.asarray(d['val_target'][i:i+1], np.float32))[:, None]
                        variants.append(('-detail', cell_detail(tgt, c)))
                    for suffix, det in variants:
                        for s in seeds:
                            g = decode(sample_patches(model, norm, cb, a.steps, s, device, detail_cells=det)[0])[0, 0].cpu().numpy()
                            rows.append({'source': f'earth-{mname}-blur{sig:g}{suffix}', 'cls': name, 'm': metrics(g)})
                            if s == seeds[0] and mname == 'meso' and not suffix:
                                ex[f'blur{sig:g}'] = g
            if k < 3:
                examples.setdefault(('earth', name), []).append(ex)
        for k, (stem, i, j, z23) in enumerate(sims[name]):
            c = encode(torch.from_numpy(z23)[None, None]).float()
            ex = {'cond': z23}
            for s in seeds:
                g = decode(sample_patches(models['meso'][0], models['meso'][1], c, a.steps, s, device)[0])[0, 0].cpu().numpy()
                rows.append({'source': 'sim-meso', 'cls': name, 'm': metrics(g)})
                if s == seeds[0]:
                    ex['meso'] = g
            if k < 3:
                examples.setdefault(('sim', name), []).append(ex)
        print(f'{name}: {len(earth[name])} Earth patches, {len(sims[name])} simulated windows, {time.perf_counter()-t0:.0f}s', flush=True)
    summary = summarize(rows)
    verdict = {}
    for name in RULE['classes']:
        r, g = summary.get(f'earth-real/{name}'), summary.get(f'earth-meso/{name}')
        if r and g:
            verdict[name] = {'incision_ratio': g['incision_m']/max(r['incision_m'], 1e-6), 'band_5_11_ratio': g['bands'][0]/max(r['bands'][0], 1e-12)}
    softened = {}
    for src in sorted({r['source'] for r in rows if '-blur' in r['source']}):
        for name, _, _ in CLASSES:
            r, g = summary.get(f'earth-real/{name}'), summary.get(f'{src}/{name}')
            if r and g:
                softened.setdefault(src, {})[name] = {'incision_ratio': g['incision_m']/max(r['incision_m'], 1e-6),
                                                      'band_5_11_ratio': g['bands'][0]/max(r['bands'][0], 1e-12)}
    robust = None
    if 'earth-meso-blur0.7' in softened:   # rule fixed before any detail-aware model was trained
        sv = softened['earth-meso-blur0.7']
        robust = all(n in sv and sv[n]['incision_ratio'] >= .7 and sv[n]['band_5_11_ratio'] >= .7 for n in RULE['classes'])
    adequate = set(verdict) == set(RULE['classes']) and all(v['incision_ratio'] >= RULE['incision_ratio_min'] and v['band_5_11_ratio'] >= RULE['band_5_11_ratio_min'] for v in verdict.values())
    result = {'summary': summary, 'verdict_by_class': verdict, 'model_adequate_for_plains': adequate, 'softened_input': softened,
              'robust_to_soft_input': robust, 'adopt_if_both_true': [adequate, robust],
              'meaning': ('Meso reproduces Earth plain texture from real 23 km input: smooth simulated plains are an input problem; fix the simulation.'
                          if adequate else 'Meso loses Earth plain texture even from real 23 km input: a new model could help, if it beats this one on this test.'),
              'seconds': time.perf_counter()-t0}
    (out/'results.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for (src, name), exs in examples.items():
        cols = ['cond', 'real', 'meso']+[f'blur{x:g}' for x in blurs]+['compare'] if src == 'earth' else ['cond', 'meso']
        cols = [c for c in cols if c in exs[0]]
        fig, axs = plt.subplots(len(exs), len(cols), figsize=(3.6*len(cols), 3.7*len(exs)), squeeze=False, constrained_layout=True)
        for r, ex in enumerate(exs):
            for k, col in enumerate(cols):
                z = ex[col]; sp = 23040. if col == 'cond' else SPACING
                if col == 'cond':
                    z = z[1:17, 1:17]
                h = 368.64/2
                axs[r, k].imshow(image(z, sp), origin='lower', extent=[-h, h, -h, h], interpolation='nearest' if col == 'cond' else 'bilinear')
                axs[r, k].set_title({'cond': '23 km input', 'real': 'real Earth 2.88 km', 'meso': 'meso', 'compare': 'compare model'}.get(col, f'meso, input {col}'), fontsize=9)
        fig.suptitle(f'{src} {name} windows (first three of the declared set); each 368.64 km', fontsize=11)
        legend(fig, axs); fig.savefig(out/f'{src}-{name}.png', dpi=80); plt.close(fig)
    lines = ['<tr><th>source / class</th><th>n</th><th>5.8-11.5 km</th><th>11.5-23 km</th><th>23-46 km</th><th>incision m</th><th>roughness m</th></tr>']
    for key, v in summary.items():
        lines.append(f'<tr><td>{key}</td><td>{v["n"]}</td>'+''.join(f'<td>{b:.3g}</td>' for b in v['bands'])+f'<td>{v["incision_m"]:.1f}</td><td>{v["roughness_m"]:.1f}</td></tr>')
    imgs = ''.join(f'<h3>{src} {name}</h3><img src="{src}-{name}.png">' for (src, name) in examples)
    (out/'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Meso plains diagnosis</title><style>
body{{font:15px system-ui;margin:24px;background:#f6f6f4}}img{{max-width:1300px;width:100%}}td,th{{border:1px solid #bbb;padding:4px 8px}}table{{border-collapse:collapse}}</style></head><body>
<h1>Does the meso lose plain texture?</h1><p><b>{result["meaning"]}</b></p><p>Rule: on Earth flat and plain/upland patches, meso/real must be ≥ 0.7 for incision and
5.8-11.5 km band power. Ratios: {json.dumps({k: {m: round(x, 2) for m, x in v.items()} for k, v in verdict.items()})}</p>
<table>{"".join(lines)}</table>{imgs}</body></html>''', encoding='utf-8')
    print(json.dumps({'verdict_by_class': verdict, 'model_adequate_for_plains': adequate}, indent=1))
    if robust is not None:
        print(f'Robust to softened input (blur 0.7: flat and plain/upland incision and 5.8-11.5 km power >= 0.7): {robust}; '
              f'plain test passes: {adequate}')
    for src, v in softened.items():
        print(f'{src} (generated / real, Earth input softened):')
        for name, x in v.items():
            print(f"  {name:13s} incision {x['incision_ratio']:.2f}   5.8-11.5 km power {x['band_5_11_ratio']:.2f}")
    print(f'Done in {time.perf_counter()-t0:.0f}s: {out/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
