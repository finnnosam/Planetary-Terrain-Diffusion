"""Arrangement metrics for 40.32 km structural fields: do ranges and lowlands form systems?

These measure *arrangement*, which power spectra and relief statistics cannot:
a smooth ramp plus correctly scaled noise matches Earth's spectrum but has
no elongated belts and no lowland held between ranges.

All metrics use physical metres on the 64x64 / 40.32 km grid and are evaluated
inside a central crop (default 48x48 = 1,935 km) so patch edges do not dominate.
They compare a generated field with a reference on the same guide; they are not
standalone realism scores, and visual review is still required.

Definitions (fixed before any new model is reviewed):
  high cell   land cell >= +150 m above its land-only 80.64 km Gaussian mean
  low cell    land cell <= -150 m below its land-only 161.28 km Gaussian mean
  ridge       8-connected component of high cells; length = sqrt(12*lambda1) cells,
              elongation = sqrt(lambda1/lambda2) from the component's coordinate covariance
  long_ridge_fraction      share of high cells in ridges >= 300 km long and elongation >= 2
  flanked_low_fraction     share of land cells that are low AND have high cells within
                           242 km in two directions >= 135 degrees apart (between/beside ranges)
  connected_low_fraction   share of low cells in components of >= 8 cells (~13,000 km2)
  contrast_p50_m           median (max-min) land height in 8x8-cell (322 km) windows, stride 4,
                           windows >= 50% land
  contrast_ge_1500_fraction  share of those windows with contrast >= 1,500 m
  slope_p90_m_per_km       90th percentile land slope
"""
import numpy as np
import torch

from planet_training.terrain_structure import blur

SPACING_KM = 40.32
SPEC = {'version': 'arrangement-metrics-v2', 'high_rel_m': 150., 'high_sigma_cells': 2.,
        'low_rel_m': -150., 'low_sigma_cells': 4., 'ridge_min_length_km': 300., 'ridge_min_elongation': 2.,
        'flank_radius_cells': 6, 'flank_min_angle_deg': 135.,
        'low_component_min_cells': 8, 'window_cells': 8, 'window_stride': 4, 'window_min_land': .5,
        'contrast_threshold_m': 1500., 'crop': [8, 56]}
METRICS = ['long_ridge_fraction', 'flanked_low_fraction', 'connected_low_fraction',
           'contrast_p50_m', 'contrast_ge_1500_fraction', 'slope_p90_m_per_km']

# Pass/fail gates, declared before the long planner run or the rule-based plans are reviewed.
GATES = {
    'version': 'arrangement-gates-v2',
    'structured_reference': {'long_ridge_fraction': .25, 'flanked_low_fraction': .02,
                             'meaning': 'only cases whose reference meets both thresholds are gated'},
    # Calibration on 4 Earth guides (alps, california, plains, holdout-rugged), frozen v1 renderer:
    #   supplied reference plans pass 4/4 (flanked ratio .98-1.11); 5,000-update planner passes 1/8
    #   (flanked ratio .14-.43 on alps/california/rugged; its one pass is on plains).
    'sample_ratio_minimums': {'long_ridge_fraction': .5, 'flanked_low_fraction': .6, 'contrast_p50_m': .6},
    'sample_pass': 'every listed ratio (source / reference on the same guide) >= its minimum',
    'set_pass_fraction': .6,
    'set_pass': '>= 60% of (case, plan seed) samples on structured-reference cases pass',
    'memorization': {'nn_corr_margin': .1,
                     'meaning': 'flag if median nearest-training-plan correlation of generated holdout plans exceeds '
                                'that of the holdout reference plans by more than this margin'},
    'note': 'Gates decide whether to continue, switch plan source or stop; visual review still required.',
}


def _land_relative(z, land, sigma):
    zt = torch.as_tensor(z, dtype=torch.float32)[None, None]
    m = torch.as_tensor(land, dtype=torch.float32)[None, None]
    den = blur(m, sigma).clamp_min(1e-6)
    return (zt-blur(zt*m, sigma)/den)[0, 0].numpy()


def components(mask):
    """8-connected labels with plain numpy BFS (grids are 64x64; no scipy dependency)."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32); count = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        count += 1; stack = [(y0, x0)]; labels[y0, x0] = count
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = count; stack.append((yy, xx))
    return labels, count


def _ridge_shape(ys, xs):
    if len(ys) < 2:
        return SPACING_KM, 1.
    cov = np.cov(np.stack((ys, xs)).astype(float)) + np.eye(2)/12
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return float(np.sqrt(12*lam[0])*SPACING_KM), float(np.sqrt(lam[0]/lam[1]))


def _flanked(low, high, radius, min_angle):
    """Low cells with high cells within `radius` in two directions >= min_angle apart (exact, symmetric)."""
    h, w = low.shape
    offsets = np.array([(dy, dx) for dy in range(-radius, radius+1) for dx in range(-radius, radius+1)
                        if 0 < dy*dy+dx*dx <= radius*radius])
    unit = offsets/np.linalg.norm(offsets, axis=1, keepdims=True)
    opposite = (unit@unit.T) <= np.cos(np.radians(min_angle))+1e-9
    padded = np.pad(high, radius)
    present = np.stack([padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w] for dy, dx in offsets], -1).reshape(h*w, -1)
    pairs = ((present.astype(np.float32)@opposite.astype(np.float32)) > 0) & present
    return pairs.any(1).reshape(h, w) & low


def arrangement(z, base, crop=None):
    """Metrics for one 64x64 physical field (metres) on its baseline grid."""
    s = SPEC
    z = np.asarray(z, np.float32).squeeze(); base = np.asarray(base, np.float32).squeeze()
    if z.shape != base.shape or z.ndim != 2:
        raise ValueError('Expected matching 2-D fields')
    land = (base > 0) & (z > 0)
    rel_hi = _land_relative(z, land, s['high_sigma_cells']); rel_lo = _land_relative(z, land, s['low_sigma_cells'])
    high = land & (rel_hi >= s['high_rel_m']); low = land & (rel_lo <= s['low_rel_m'])
    a, b = crop or s['crop']; win = (slice(a, b), slice(a, b))
    landc, highc, lowc = land[win], high[win], low[win]
    labels, n = components(high)
    long_cells = 0
    for k in range(1, n+1):
        ys, xs = np.nonzero(labels == k)
        length, elong = _ridge_shape(ys, xs)
        if length >= s['ridge_min_length_km'] and elong >= s['ridge_min_elongation']:
            long_cells += int((labels[win] == k).sum())
    flank = _flanked(low, high, s['flank_radius_cells'], s['flank_min_angle_deg'])[win]
    lab_lo, n_lo = components(low)
    big = [k for k in range(1, n_lo+1) if (lab_lo == k).sum() >= s['low_component_min_cells']]
    connected = np.isin(lab_lo[win], big) & lowc if big else np.zeros_like(lowc)
    contrasts = []
    zc = z[win]; size = s['window_cells']
    for y in range(0, zc.shape[0]-size+1, s['window_stride']):
        for x in range(0, zc.shape[1]-size+1, s['window_stride']):
            m = landc[y:y+size, x:x+size]
            if m.mean() >= s['window_min_land']:
                v = zc[y:y+size, x:x+size][m]; contrasts.append(float(v.max()-v.min()))
    gy, gx = np.gradient(z, SPACING_KM)
    slope = np.hypot(gx, gy)[win][landc]
    nland, nhigh, nlow = int(landc.sum()), int(highc.sum()), int(lowc.sum())
    return {'land_cells': nland, 'high_cells': nhigh, 'low_cells': nlow,
            'long_ridge_fraction': long_cells/nhigh if nhigh else 0.,
            'flanked_low_fraction': float(flank.sum())/nland if nland else 0.,
            'connected_low_fraction': float(connected.sum())/nlow if nlow else 0.,
            'contrast_p50_m': float(np.median(contrasts)) if contrasts else 0.,
            'contrast_ge_1500_fraction': float(np.mean(np.asarray(contrasts) >= s['contrast_threshold_m'])) if contrasts else 0.,
            'slope_p90_m_per_km': float(np.percentile(slope, 90)) if len(slope) else 0.,
            'windows': len(contrasts)}


def structured(reference_metrics):
    g = GATES['structured_reference']
    return (reference_metrics['long_ridge_fraction'] >= g['long_ridge_fraction']
            and reference_metrics['flanked_low_fraction'] >= g['flanked_low_fraction'])


def compare(sample, reference):
    ratios = {k: (sample[k]/reference[k] if reference[k] > 1e-9 else float('nan')) for k in GATES['sample_ratio_minimums']}
    passed = all(np.isfinite(ratios[k]) and ratios[k] >= v for k, v in GATES['sample_ratio_minimums'].items())
    return {'ratios': ratios, 'pass': bool(passed)}


def summarize(rows):
    """rows: dicts with keys source, case, structured(bool), pass(bool) -> per-source gate verdicts."""
    out = {}
    for source in sorted({r['source'] for r in rows}):
        gated = [r for r in rows if r['source'] == source and r.get('structured')]
        frac = float(np.mean([r['pass'] for r in gated])) if gated else float('nan')
        out[source] = {'gated_samples': len(gated), 'pass_fraction': frac,
                       'set_pass': bool(gated) and frac >= GATES['set_pass_fraction']}
    return out


def plan_signature(plan_channel, crop=(16, 48)):
    """Zero-mean unit-norm central crop of one plan channel, for nearest-neighbour search."""
    a, b = crop
    v = np.asarray(plan_channel, np.float32)[..., a:b, a:b].reshape(*np.shape(plan_channel)[:-2], -1)
    v = v-v.mean(-1, keepdims=True)
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def dihedral(x):
    """The 8 rotations/reflections of a (..., H, W) array (training uses the same augmentation)."""
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(-2, -1)); out += [r, r[..., ::-1]]
    return out


def nearest_training_correlation(plan_channel, library):
    """Max correlation of one generated plan channel against a training-plan library, over 8 symmetries."""
    best = -1.
    for v in dihedral(np.asarray(plan_channel, np.float32)):
        sig = plan_signature(np.ascontiguousarray(v))
        best = max(best, float((library @ sig).max()))
    return best
