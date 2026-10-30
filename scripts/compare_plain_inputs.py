"""What do Earth's plains have at 23 km that the simulated plains lack? (no model, a minute on CPU)

First result (2026-10-15): within a relief class the simulated plains match Earth (undulation, slope and
23 km channel incision all within ~15%). The difference is how much of the land is plain: 67% of simulated
368 km land windows had < 300 m relief against 34% on Earth (median relief 186 m vs 456 m). The section
"How much of the land is plain" below measures that; it led to stronger mantle swells and harder stable crust.

The meso model only sees 23.04 km input. On Earth plains it reproduces real 2.88 km valley texture from that input
(outputs/meso-plains-diagnosis/20261015-193134); on simulated plains it has little to work with. This compares the
23 km inputs directly: modern Earth (the intermediate-stage dataset, 112 x 112 cells per window, ice sheets
excluded by |latitude| <= 60) against a tectonic-prototype run, on 32 x 32-cell (737 km) land subwindows grouped
by relief class (flat: p95-p5 < 300 m; plain/upland: 300-800 m).

Per subwindow:
  band power    detrended radial power in 46-92, 92-184 and 184-368 km wavelength bands
  undulation    std of z minus its 1-cell and 2-cell Gaussian smoothing (broad valleys, terraces, scarps)
  slope         median and 90th percentile, m/km
  incision      mean depth of 23 km channels (>= 20,000 km^2 upstream) below a 3-cell Gaussian mean

  python -m scripts.compare_plain_inputs --sim outputs/tectonic-prototype/<run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planet_training import landscape_evolution as le
from planet_training.terrain_structure import blur

ROOT = Path(__file__).resolve().parents[1]
EARTH = ROOT/'outputs/planet_training/data/bridge-adapted'
SIM_ROOT = ROOT/'outputs/tectonic-prototype'
SP = 23040.
W = 32
CLASSES = [('flat', 0, 300), ('plain-upland', 300, 800)]
BANDS = [(46.08, 92.16), (92.16, 184.32), (184.32, 368.64)]


def band_power(z):
    n = z.shape[0]
    y, x = np.mgrid[:n, :n]
    a = np.c_[x.ravel(), y.ravel(), np.ones(n*n)]
    r = z.ravel()-a@np.linalg.lstsq(a, z.ravel(), rcond=None)[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fft2(r.reshape(n, n)*w))**2
    f = np.sqrt(np.fft.fftfreq(n)[:, None]**2+np.fft.fftfreq(n)[None, :]**2)
    wl = np.where(f > 0, SP/1e3/np.maximum(f, 1e-9), np.inf)
    return [float(p[(wl >= lo) & (wl < hi)].sum()/(n*n)) for lo, hi in BANDS]


def metrics(z):
    n = z.shape[0]
    t = torch.from_numpy(z.astype(np.float32))[None, None]
    g1, g2, g3 = [blur(t, s)[0, 0].numpy() for s in (1., 2., 3.)]
    gy, gx = np.gradient(z, SP/1e3)
    slope = np.hypot(gx, gy)[2:-2, 2:-2]
    edge = np.zeros((n, n), bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
    zf = le.priority_flood(z.astype(np.float64).ravel(), edge.ravel(), n, n, 1e-3)
    rec, dist, stack = le.route(zf, edge.ravel(), n, n, SP)
    area = le.accumulate(rec, stack, SP**2).reshape(n, n)
    inner = np.zeros((n, n), bool); inner[3:-3, 3:-3] = True
    ch = (area >= 2e10) & inner
    return {'bands': band_power(z), 'undulation1_m': float((z-g1)[2:-2, 2:-2].std()), 'undulation2_m': float((z-g2)[2:-2, 2:-2].std()),
            'slope_p50': float(np.median(slope)), 'slope_p90': float(np.percentile(slope, 90)),
            'incision_m': float((g3-z)[ch].mean()) if ch.any() else 0., 'channel_cells': int(ch.sum())}


def subwindows(fields, stride=16):
    for f in fields:
        for i in range(0, f.shape[0]-W+1, stride):
            for j in range(0, f.shape[1]-W+1, stride):
                s = f[i:i+W, j:j+W]
                if (s > 0).mean() < .98:
                    continue
                rel = float(np.percentile(s, 95)-np.percentile(s, 5))
                for name, lo, hi in CLASSES:
                    if lo <= rel < hi:
                        yield name, s.astype(np.float64)


def collect(fields, limit):
    out = {name: [] for name, _, _ in CLASSES}
    for name, s in subwindows(fields):
        if len(out[name]) < limit:
            out[name].append(metrics(s))
    return out


def land_relief(fields, step=4):
    """p95-p5 relief of every >= 98% land 16 x 16-cell (368.64 km, one meso window) window."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = []
    for f in fields:
        v = sliding_window_view(np.asarray(f, np.float64), (16, 16))[::step, ::step].reshape(-1, 256)
        v = v[(v > 0).mean(1) >= .98]
        out.append(np.percentile(v, 95, 1)-np.percentile(v, 5, 1))
    r = np.concatenate(out)
    return {'windows': int(len(r)), 'relief_p10_25_50_75_m': np.percentile(r, [10, 25, 50, 75]).round(0).tolist(),
            'fraction_under_300m': float((r < 300).mean()), 'fraction_under_100m': float((r < 100).mean())}


def summary(rows):
    keys = ['undulation1_m', 'undulation2_m', 'slope_p50', 'slope_p90', 'incision_m']
    s = {'n': len(rows)}
    if rows:
        s.update({k: float(np.median([r[k] for r in rows])) for k in keys})
        s['bands'] = np.median([r['bands'] for r in rows], 0).tolist()
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sim', help='tectonic-prototype run folder (default: newest)')
    p.add_argument('--earth', default=str(EARTH)); p.add_argument('--limit', type=int, default=400)
    p.add_argument('--out')
    a = p.parse_args(argv)
    sim = Path(a.sim) if a.sim else max(q for q in SIM_ROOT.iterdir() if q.is_dir() and q.name[:2] == '20')
    enc = np.load(Path(a.earth)/'train_target.npy', mmap_mode='r'); centre = np.load(Path(a.earth)/'train_centre.npy')
    keep = np.flatnonzero(np.abs(centre[:, 0]) <= 60)
    rng = np.random.default_rng(0); keep = rng.permutation(keep)
    decode = lambda i: np.sign(x := np.asarray(enc[i], np.float32))*np.square(x)
    earth = collect((decode(i) for i in keep), a.limit)
    sim_fields = [np.load(f)['elevation23_m'] for f in sorted(sim.glob('*-s[0-9].npz'))]
    simr = collect(sim_fields, a.limit)
    result = {'sim_run': str(sim), 'earth': {k: summary(v) for k, v in earth.items()}, 'sim': {k: summary(v) for k, v in simr.items()},
              'land_relief_368km': {'earth': land_relief(decode(i) for i in keep[:1500]), 'sim': land_relief(sim_fields)}}
    ratio = {}
    for name, _, _ in CLASSES:
        e, s = result['earth'][name], result['sim'][name]
        if e['n'] and s['n']:
            ratio[name] = {k: s[k]/max(e[k], 1e-9) for k in ('undulation1_m', 'undulation2_m', 'slope_p50', 'slope_p90', 'incision_m')}
            ratio[name]['bands'] = [sb/max(eb, 1e-9) for sb, eb in zip(s['bands'], e['bands'])]
    result['sim_over_earth'] = ratio
    text = json.dumps(result, indent=1)
    out = Path(a.out) if a.out else sim/'plain-inputs-vs-earth.json'
    out.write_text(text, encoding='utf-8')
    for name in ratio:
        e, s, r = result['earth'][name], result['sim'][name], ratio[name]
        print(f"{name}: Earth n={e['n']} sim n={s['n']}")
        print(f"  undulation 1/2-cell  Earth {e['undulation1_m']:.1f}/{e['undulation2_m']:.1f} m  sim {s['undulation1_m']:.1f}/{s['undulation2_m']:.1f} m  ratio {r['undulation1_m']:.2f}/{r['undulation2_m']:.2f}")
        print(f"  slope p50/p90        Earth {e['slope_p50']:.2f}/{e['slope_p90']:.2f}  sim {s['slope_p50']:.2f}/{s['slope_p90']:.2f} m/km")
        print(f"  incision             Earth {e['incision_m']:.1f} m  sim {s['incision_m']:.1f} m  ratio {r['incision_m']:.2f}")
        print('  band power ratio 46-92 / 92-184 / 184-368 km: '+' / '.join(f'{x:.2f}' for x in r['bands']))
    lr = result['land_relief_368km']
    print('How much of the land is plain (368 km land windows, p95-p5 relief):')
    for k in ('earth', 'sim'):
        print(f"  {k:5s} p10/25/50/75 {lr[k]['relief_p10_25_50_75_m']} m, under 300 m: {lr[k]['fraction_under_300m']:.0%}, under 100 m: {lr[k]['fraction_under_100m']:.0%}")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
