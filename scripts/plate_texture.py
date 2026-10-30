"""Texture of the plate layer's continents against Earth on the same 46 km grid (docs/planet-layer-plan.md, step 8).

  python -m scripts.plate_texture outputs\\planet-plates\\<run>\\seed1001            # present-day snapshot
  python -m scripts.plate_texture <seed folder> --time -30

Earth: ETOPO 2022 area-averaged onto CubeSphere(217) (scripts.earth_reference), ice sheets excluded.
Windows are cut from the six face charts of the node field:
  * 368 km windows (8 x 8 nodes, stride 4), land >= 98%: p95-p5 relief percentiles and the share below 300 / 100 m
    (the 368 km land-relief target);
  * 736 km windows (16 x 16 nodes, stride 8), land >= 90%: mean detrended, Hann-windowed power in the wavelength bands
    92-147, 147-245, 245-368 and 368-736 km, planet / Earth.
Score (fixed before any plate-layer change is tested): the worst |log10| over the band-power ratios at 245-368 and
368-736 km and the 368 km relief p25 / p50 / p75 ratios. 0 = Earth; 0.3 = a factor of two off. The 92-245 km bands
are reported but not scored: at 46 km they are 2-5 cells, below what the plate grid can hold; that relief is the
11.53 km landscape stage's job (scored on the meso's 23 km input, scripts.run_planet_landscape.texture23).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPACING_KM = 46.1
BANDS_KM = ((92, 147), (147, 245), (245, 368), (368, 736))


def windows(face, size, stride):
    out = []
    for f in range(6):
        for i in range(0, face.shape[1]-size+1, stride):
            for j in range(0, face.shape[2]-size+1, stride):
                out.append(face[f, i:i+size, j:j+size])
    return np.stack(out)


def band_power(w):
    n = w.shape[-1]
    y, x = np.mgrid[:n, :n]
    A = np.c_[x.ravel(), y.ravel(), np.ones(n*n)]
    proj = A@np.linalg.pinv(A)
    r = w.reshape(len(w), -1)
    r = (r-r@proj.T).reshape(w.shape)*np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fft2(r))**2
    f = np.sqrt(np.fft.fftfreq(n)[:, None]**2+np.fft.fftfreq(n)[None, :]**2)
    wl = np.where(f > 0, SPACING_KM/np.maximum(f, 1e-9), np.inf)
    return np.stack([p[:, (wl >= lo) & (wl < hi)].mean(1) for lo, hi in BANDS_KM], 1)


def texture(z, g, mask=None):
    """Statistics of one elevation field on CubeSphere(217) nodes (mask: nodes to count, e.g. no ice)."""
    face = g.to_cube(np.asarray(z, np.float32))
    valid = g.to_cube(np.ones(g.size, np.float32) if mask is None else mask.astype(np.float32))
    w8, v8 = windows(face, 8, 4), windows(valid, 8, 4)
    keep8 = (v8 > .5).all((1, 2)) & ((w8 > 0).mean((1, 2)) >= .98)
    rel = np.percentile(w8[keep8], 95, axis=(1, 2))-np.percentile(w8[keep8], 5, axis=(1, 2))
    w16, v16 = windows(face, 16, 8), windows(valid, 16, 8)
    keep16 = (v16 > .5).all((1, 2)) & ((w16 > 0).mean((1, 2)) >= .9)
    bp = band_power(w16[keep16].astype(np.float64)).mean(0)
    return {'land_windows_368km': int(keep8.sum()), 'land_windows_736km': int(keep16.sum()),
            'relief_368km_m': {str(q): float(np.percentile(rel, q)) for q in (10, 25, 50, 75, 90)},
            'relief_below_300_share': float((rel < 300).mean()), 'relief_below_100_share': float((rel < 100).mean()),
            'band_power_m2': {f'{lo}-{hi}': float(v) for (lo, hi), v in zip(BANDS_KM, bp)}}


def arrangement(z, g, mask=None):
    """planet_training.arrangement_metrics on 64 x 64-node windows of the face charts (3 x 3 per face, ~2,950 km),
    windows with >= 50% land in the central 48 x 48 crop and (Earth) <= 5% ice. Means weighted by each metric's own
    cell count. Same code on Earth and planet, so the 46 vs 40 km spacing difference cancels."""
    from planet_training import arrangement_metrics as am
    face = g.to_cube(np.asarray(z, np.float32))
    valid = g.to_cube(np.ones(g.size, np.float32) if mask is None else mask.astype(np.float32))
    rows = []
    starts = np.linspace(0, face.shape[1]-64, 3).round().astype(int)
    for f in range(6):
        for i in starts:
            for j in starts:
                w, v = face[f, i:i+64, j:j+64], valid[f, i:i+64, j:j+64]
                if (v[8:56, 8:56] < .5).mean() > .05 or (w[8:56, 8:56] > 0).mean() < .5:
                    continue
                rows.append(am.arrangement(w, w))
    if not rows:
        return {}
    wsum = lambda key, wkey: float(sum(r[key]*r[wkey] for r in rows)/max(sum(r[wkey] for r in rows), 1))
    return {'windows': len(rows), 'long_ridge_fraction': wsum('long_ridge_fraction', 'high_cells'),
            'flanked_low_fraction': wsum('flanked_low_fraction', 'land_cells'),
            'connected_low_fraction': wsum('connected_low_fraction', 'low_cells'),
            'high_share': float(sum(r['high_cells'] for r in rows)/max(sum(r['land_cells'] for r in rows), 1)),
            'contrast_p50_m': float(np.median([r['contrast_p50_m'] for r in rows])),
            'slope_p90_m_per_km': float(np.median([r['slope_p90_m_per_km'] for r in rows]))}


def compare(planet, earth):
    ratios = {k: planet['band_power_m2'][k]/earth['band_power_m2'][k] for k in earth['band_power_m2']}
    rel = {q: planet['relief_368km_m'][q]/earth['relief_368km_m'][q] for q in ('25', '50', '75')}
    scored = [ratios['245-368'], ratios['368-736']]+list(rel.values())
    worst = max(abs(np.log10(max(v, 1e-6))) for v in scored)
    return {'band_power_ratio': ratios, 'relief_ratio': rel, 'texture_score': float(worst)}


def earth_texture(g):
    from scripts.earth_reference import ice_mask, load
    cache = ROOT/'outputs'/'planet-reference'/'earth-n217-texture.json'
    if cache.exists():
        t = json.loads(cache.read_text())
        if 'arrangement' in t:
            return t
    load(217)
    z = np.load(ROOT/'outputs'/'planet-reference'/'earth-n217.npz')['elevation_m']
    t = texture(z, g, ~ice_mask(g))
    t['arrangement'] = arrangement(z, g, ~ice_mask(g))
    cache.write_text(json.dumps(t, indent=1))
    return t


def main(argv=None):
    from planet_layer.sphere_graph import CubeSphere
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('seed_dirs', nargs='+')
    p.add_argument('--time', type=float, default=0.)
    a = p.parse_args(argv)
    g = CubeSphere(217)
    earth = earth_texture(g)
    print('Earth 368 km relief p10/25/50/75/90: ' + '/'.join(f"{earth['relief_368km_m'][q]:.0f}" for q in ('10', '25', '50', '75', '90'))
          + f" m; <300 m {earth['relief_below_300_share']:.0%}")
    for d in a.seed_dirs:
        z = np.load(Path(d)/'snapshots'/f't{a.time:+07.1f}.npz')['elevation_m']
        t = texture(z, g); c = compare(t, earth); ar = arrangement(z, g)
        print(f"{d}: relief p10/25/50/75/90 " + '/'.join(f"{t['relief_368km_m'][q]:.0f}" for q in ('10', '25', '50', '75', '90'))
              + f" m; <300 m {t['relief_below_300_share']:.0%}; band power ratio "
              + ' '.join(f'{k} {v:.3f}' for k, v in c['band_power_ratio'].items()) + f"; texture score {c['texture_score']:.2f}")
        print('   arrangement (planet / Earth): ' + ', '.join(f"{k} {ar[k]:.2f}/{earth['arrangement'][k]:.2f}" for k in ar if k != 'windows'))


if __name__ == '__main__':
    main()
