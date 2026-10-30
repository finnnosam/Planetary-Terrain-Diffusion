"""Target feasibility and paired stage-role probes. No training or planet run."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, manifest_id
from planet_training.terrain_structure import (
    DEFAULT_DATA, StructureData, area_weights, conservative_resample, decode,
    encode, handoff, load as load_structure, sample as sample_structure,
    target_for, constraint_metrics, HANDOFF_FACTOR, CONSERVE_FACTOR,
)
from planet_training.smooth_conservation import project_smooth
from scripts.diagnose_terrain_structure import image, legend, gauss, CMAP, NORM
from scripts.structure_job import console_log
from toy_meso.common import load_checkpoint, sample_patches, pick_device

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT/'outputs/terrain-stage-roles'
MODELS = {
    'intermediate': ('outputs/intermediate/selected-adapted-1750/best.pt',
        'fd04c930551ddebc809accf75942a37503403bb3f2f0b3b93957b0ddc9585ac9'),
    'meso': ('outputs/meso/selected-adapted-1000/best.pt',
        '717e1c718fefe51fc0f5066de39618da9cc5a6b034c9df0fb1af09ebcbd7fcf6'),
    'structure': ('outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544/structure-checkpoint.pt',
        '65562cb3e4e4eeb1339ad8a1a7007f17305f6b14d351af22b69b5f0f56783bc6'),
}
BRANCH_TITLES = {
    'selected': 'Selected intermediate',
    'coarse-direct': 'Coarse guide; no intermediate',
    'oracle-direct': 'Oracle structure; no intermediate detail',
    'oracle-target-detail': 'Oracle structure + target detail',
    'oracle-sampled-detail': 'Oracle structure + sampled detail',
    'learned-direct': 'Learned structure; no intermediate detail',
    'learned-sampled-detail': 'Learned structure + sampled detail',
    'modern23-oracle': 'Modern 23 km oracle guide',
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array(value):
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor(value, device='cpu'):
    z = torch.as_tensor(np.array(value, dtype=np.float32), device=device)
    return z[None, None] if z.ndim == 2 else z


def component_summary(mask, spacing_m):
    """8-connected local objects; no wraparound across independent patch edges."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or spacing_m <= 0:
        raise ValueError('Expected a two-dimensional mask and positive physical spacing')
    seen = np.zeros_like(mask)
    rows = []
    for sy, sx in np.argwhere(mask):
        if seen[sy, sx]:
            continue
        queue = deque([(int(sy), int(sx))]); seen[sy, sx] = True; cells = []
        while queue:
            y, x = queue.popleft(); cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; queue.append((yy, xx))
        points = np.asarray(cells, dtype=float)
        # Rectangle-equivalent extent from covariance, not a classified belt width.
        covariance = np.cov(points.T, bias=True) if len(points) > 1 else np.zeros((2, 2))
        eigen = np.linalg.eigvalsh(covariance).clip(0)
        length = float(np.sqrt(12*eigen[-1]+1)*spacing_m/1000)
        area = len(points)*(spacing_m/1000)**2
        rows.append({'cells': len(points), 'area_km2_tangent_plane': area,
            'equivalent_length_km': length, 'equivalent_width_km': area/length,
            'elongation': float(np.sqrt((eigen[-1]+1/12)/(eigen[0]+1/12))),
            'edge_censored': bool((points[:, 0] == 0).any() or (points[:, 1] == 0).any()
                or (points[:, 0] == mask.shape[0]-1).any() or (points[:, 1] == mask.shape[1]-1).any())})
    rows.sort(key=lambda r: r['cells'], reverse=True)
    return {'count': len(rows), 'largest_area_fraction_of_patch': rows[0]['cells']/mask.size if rows else 0.,
        'largest_fraction_of_mask': rows[0]['cells']/max(1, mask.sum()) if rows else 0., 'largest_objects': rows[:5]}


def describe(z, spacing_m):
    """Physical-scale descriptors, deliberately not an automatic realism score."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all() or not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('Expected finite two-dimensional metre terrain and positive spacing')
    land = z > 0
    smooth = gauss(z, max(1., 23040./spacing_m))
    dy, dx = np.gradient(smooth, spacing_m/1000)
    slope = np.hypot(dy, dx)
    neighbourhood = gauss(z, 80640./spacing_m)
    relief = np.sqrt(np.maximum(gauss(z*z, 80640./spacing_m)-neighbourhood**2, 0))
    gentle = land & (slope < 3.) & (relief < 100.)
    prominence = z-neighbourhood
    high = land & (prominence > 200.)
    low = land & (prominence < -150.)
    return {'land_fraction': float(land.mean()),
        'land_median_m': float(np.median(z[land])) if land.any() else None,
        'land_slope_p50_p90_m_per_km': np.percentile(slope[land], [50, 90]).tolist() if land.any() else [],
        'land_local_relief_p50_p90_m': np.percentile(relief[land], [50, 90]).tolist() if land.any() else [],
        'gentle_fraction_of_land': float(gentle.sum()/max(1, land.sum())),
        'gentle_objects': component_summary(gentle, spacing_m),
        'positive_prominence_objects': component_summary(high, spacing_m),
        'negative_prominence_objects': component_summary(low, spacing_m)}


def build_branches(coarse, guide_up, sampled_encoded, learned40, base40,
                   oracle40=None, adapted_target=None, modern23=None):
    """Change only short-scale intermediate detail within each structure pair.

    Both routes use the existing physical 40->23 handoff and identical local
    constraints. guide_up supplies exactly zero detail to the direct route.
    Selected/coarse/modern references remain unprojected diagnostic controls.
    """
    learned_direct, _, base23 = handoff(learned40, base40, guide_up, coarse)
    learned_detail, _, _ = handoff(learned40, base40, sampled_encoded, coarse)
    result = {'selected': decode(sampled_encoded), 'coarse-direct': base23,
        'learned-direct': learned_direct, 'learned-sampled-detail': learned_detail}
    if oracle40 is not None:
        if adapted_target is None or modern23 is None:
            raise ValueError('Oracle branches need a paired target and modern reference')
        direct, _, _ = handoff(oracle40, base40, guide_up, coarse)
        ideal_detail, _, _ = handoff(oracle40, base40, adapted_target, coarse)
        sampled_detail, _, _ = handoff(oracle40, base40, sampled_encoded, coarse)
        result.update({'oracle-direct': direct, 'oracle-target-detail': ideal_detail,
            'oracle-sampled-detail': sampled_detail, 'modern23-oracle': modern23})
    order = [k for k in BRANCH_TITLES if k in result]
    return {k: result[k] for k in order}, base23


def assert_constraints(z, base, factor, spacing):
    result = constraint_metrics(z, base, factor, spacing)
    if not torch.isfinite(z).all() or result['added_footprint_mean_drift_max_m'] > .05 or result['protected_height_change_max_m'] > .005 or result['sign_disagreement_fraction']:
        raise ValueError(f'Experimental constraint failed: {result}')
    return result


def render_fields(path, title, fields, spacing, note='', columns=4):
    rows = int(np.ceil(len(fields)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns*3.25, rows*3.35+.7), squeeze=False, constrained_layout=True)
    for ax, (name, z) in zip(axes.flat, fields):
        half = len(z)*spacing/2000
        ax.imshow(image(z, spacing), extent=(-half, half, -half, half), interpolation='nearest')
        ax.set_title(name, fontsize=9); ax.set_xlabel('km'); ax.set_ylabel('km')
    for ax in axes.flat[len(fields):]:
        ax.axis('off')
    fig.suptitle(title+'\n'+note, fontsize=11)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        ax=axes.ravel().tolist(), orientation='horizontal', fraction=.05, pad=.025, aspect=45)
    bar.set_ticks([-6000, -3000, 0, 1000, 2000, 4000, 6000])
    bar.ax.tick_params(labelsize=8); bar.set_label('Elevation (m); fixed colours; display clipping only', fontsize=9)
    fig.savefig(path, dpi=130); plt.close(fig)


def target_audit(data, norm, out):
    count = data.meta['val_count']
    raw40, original, context = data.batch(0, 'val', np.arange(count))
    modern_encoded = torch.from_numpy(np.array(data.source.arrays['val_target'], dtype=np.float32))[:, None]
    degraded = degrade_guides(context, original, torch.Generator().manual_seed(2026), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    original_up, _ = norm.condition(original); new_up, _ = norm.condition(degraded)
    adapted = modern_encoded+new_up-original_up
    constrained_encoded, base40 = target_for(raw40, degraded)
    constrained40 = decode(constrained_encoded)
    modern = decode(modern_encoded)
    recomputed = conservative_resample(modern[:, 0], 64)[:, None]
    torch.testing.assert_close(recomputed, raw40, atol=.02, rtol=1e-5)
    centres = np.load(Path(data.meta['array_source'])/'val_centre.npy')
    records = []
    for i in range(count):
        values = {'modern23': modern[i:i+1], 'translated23': decode(adapted[i:i+1]),
            'unchanged23': modern[i:i+1],
            'unconstrained40_display23': F.interpolate(raw40[i:i+1], size=(112, 112), mode='bilinear', align_corners=False),
            'constrained40_display23': F.interpolate(constrained40[i:i+1], size=(112, 112), mode='bilinear', align_corners=False)}
        row = {'index': i, 'latitude': float(centres[i, 0]), 'longitude': float(centres[i, 1]), 'fields': {}}
        for name, field in values.items():
            z = array(field)[0, 0]
            row['fields'][name] = describe(z[30:82, 30:82], 23040.)
            row['fields'][name]['full_patch_difference_from_modern_rms_m'] = float((field-modern[i:i+1]).square().mean().sqrt())
            row['fields'][name]['modern_land_becoming_nonpositive_fraction'] = float(((field <= 0) & (modern[i:i+1] > 0)).sum()/max(1, int((modern[i:i+1] > 0).sum())))
        row['constraints40'] = assert_constraints(constrained40[i:i+1], base40[i:i+1], CONSERVE_FACTOR, 40320.)
        records.append(row)
    # Deterministic target-only selection, before any sampled output is seen.
    eligible = [r for r in records if r['fields']['modern23']['land_fraction'] >= .85]
    if len(eligible) < 2:
        raise ValueError('Insufficient inland holdouts for the declared selection')
    quiet = min(eligible, key=lambda r: (r['fields']['modern23']['land_local_relief_p50_p90_m'][0], r['index']))
    rugged = max([r for r in eligible if r['index'] != quiet['index']],
        key=lambda r: (r['fields']['modern23']['land_local_relief_p50_p90_m'][1], -r['index']))
    coastal = [r for r in records if .1 < r['fields']['modern23']['land_fraction'] < .9 and r['index'] not in [quiet['index'], rugged['index']]]
    if not coastal:
        raise ValueError('No coastal holdout for the declared selection')
    coast = min(coastal, key=lambda r: (abs(r['fields']['modern23']['land_fraction']-.5), r['index']))
    selected = [('holdout-quiet', quiet), ('holdout-rugged', rugged), ('holdout-coast', coast)]
    np.savez_compressed(out/'target-audit-arrays.npz', modern23_encoded=array(modern_encoded), original_coarse_encoded=array(original),
        degraded_coarse_encoded=array(degraded), translated23_encoded=array(adapted), modern40_metres=array(raw40),
        constrained40_metres=array(constrained40), baseline40_metres=array(base40), centres=centres)
    atomic_json(out/'target-audit.json', {'records': records, 'selected_holdouts': [{'label': label, 'index': r['index']} for label, r in selected],
        'geographic_scope': 'All 55 formal expanded-footprint holdouts, drawn from two geographic caps; not a global prevalence survey.',
        'selection': 'Target-only extremes of inland local relief and coastal land fraction; sampled outputs never enter selection.',
        'metric_limits': 'Descriptive local thresholds, not classified mountain belts or terrain acceptance. 40-km fields resampled only for matched display/diagnostics.'})
    for label, record in selected:
        i = record['index']; fields = [('Modern / unchanged target', array(modern)[i, 0, 30:82, 30:82]),
            ('Selected translated target', array(decode(adapted))[i, 0, 30:82, 30:82]),
            ('Modern 40 km; no constraints', array(F.interpolate(raw40[i:i+1], size=(112, 112), mode='bilinear', align_corners=False))[0, 0, 30:82, 30:82]),
            ('Constrained 40 km target', array(F.interpolate(constrained40[i:i+1], size=(112, 112), mode='bilinear', align_corners=False))[0, 0, 30:82, 30:82]),
            ('Degraded guide', array(decode(new_up))[i, 0, 30:82, 30:82])]
        render_fields(out/f'{label}-targets.png', f'{label}, saved holdout {i}: 1,198.08 km per panel', fields, 23040.,
            'Same saved local orientation, not north-up; fixed metre colours and NW 5x shading.', columns=5)
    return selected, (modern_encoded, degraded, adapted, constrained40, base40)


def build_cases(selected, audit_arrays):
    cases = []
    for label in ('alps', 'california', 'plains'):
        path = ROOT/f'outputs/terrain-round-trips/{label}.npz'
        with np.load(path) as d:
            cases.append({'label': label, 'kind': 'Earth diagnostic', 'orientation': 'north-up',
                'path': str(path), 'sha256': digest(path), 'coarse': tensor(d['shared_coarse_encoded']),
                'modern_encoded': encode(tensor(d['reference23_metres'])),
                'adapted': tensor(d['ideal_intermediate_encoded']), 'reference_fine': d['reference_fine_metres'].copy()})
    modern, degraded, adapted, oracle40, base40 = audit_arrays
    for label, record in selected:
        i = record['index']
        cases.append({'label': label, 'kind': 'formal geographic holdout', 'orientation': 'stored random local frame',
            'index': i, 'latitude': record['latitude'], 'longitude': record['longitude'],
            'coarse': degraded[i:i+1], 'modern_encoded': modern[i:i+1], 'adapted': adapted[i:i+1],
            'oracle40': oracle40[i:i+1], 'base40': base40[i:i+1]})
    for age in (60, 200):
        path = ROOT/f'outputs/terrain-round-trips/complete-{age}.npz'
        with np.load(path) as d:
            cases.append({'label': f'generated-{age}', 'kind': 'generated coast/relief prefix', 'orientation': 'north-up',
                'path': str(path), 'sha256': digest(path), 'latitude': float(d['latitude']), 'longitude': float(d['longitude']),
                'coarse': tensor(d['actual_intermediate_coarse_encoded']), 'modern_encoded': None,
                'adapted': None, 'prefix_note': 'Age names identify illustrative source records/land fractions, not geographic pairing or model age conditioning. Prefixes share coast/relief seed 42.'})
    return cases


def compare_meso(raw, guide23):
    # Compare actual fine footprint means with the exact 16-cell guide centre.
    weight = area_weights(128, 2880.).to(raw)
    means = F.avg_pool2d(raw*weight, 8)/F.avg_pool2d(weight, 8)
    guide_centre = guide23[..., 48:64, 48:64]
    return {'input_footprint_mean_drift_rms_m': float((means-guide_centre).square().mean().sqrt()),
        'input_footprint_mean_drift_max_m': float((means-guide_centre).abs().max()),
        'footprint_mean_field': describe(array(means)[0, 0], 23040.)}


def render_case(out, case, seed, branches, meso_seeds):
    rows = len(branches); cols = 1+len(meso_seeds)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.1+.8), squeeze=False, constrained_layout=True)
    for row, name in enumerate(branches):
        with np.load(out/f'{case["label"]}-i{seed}-{name}-m{meso_seeds[0]}.npz') as d:
            stage = d['guide23_metres'][0, 0, 30:82, 30:82]
        axes[row, 0].imshow(image(stage, 23040.), extent=(-599.04, 599.04, -599.04, 599.04), interpolation='nearest')
        axes[row, 0].set_title(BRANCH_TITLES[name]+'\nGuide: 1,198 km', fontsize=9)
        for col, meso_seed in enumerate(meso_seeds, 1):
            with np.load(out/f'{case["label"]}-i{seed}-{name}-m{meso_seed}.npz') as d:
                raw = d['meso_raw_metres'][0, 0]
            axes[row, col].imshow(image(raw, 2880.), extent=(-184.32, 184.32, -184.32, 184.32), interpolation='nearest')
            axes[row, col].set_title(f'Raw meso: 369 km; seed {meso_seed}', fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel('km'); ax.set_ylabel('km'); ax.tick_params(labelsize=8)
    fig.suptitle(f'{case["label"]}, intermediate/structure seed {seed}; {case["orientation"]}\nSame colours and NW 5x shading. Raw meso is primary; corrected fields saved separately.', fontsize=11)
    legend(fig, axes); fig.savefig(out/f'{case["label"]}-i{seed}.png', dpi=115); plt.close(fig)


@torch.no_grad()
def probes(cases, models, norms, args, out, settings):
    total = sum((8 if c['modern_encoded'] is not None else 4)*len(args.intermediate_seeds)*len(args.meso_seeds) for c in cases)
    done = 0; start = time.perf_counter(); records = []
    for case in cases:
        coarse = case['coarse'].to(args.device)
        guide_up, _ = norms['intermediate'].condition(coarse)
        modern23 = decode(case['modern_encoded'].to(args.device)) if case['modern_encoded'] is not None else None
        adapted = case['adapted'].to(args.device) if case['adapted'] is not None else None
        oracle = None
        if modern23 is not None:
            modern40 = conservative_resample(modern23[:, 0], 64)[:, None]
            expected, _ = target_for(modern40, coarse); oracle = decode(expected)
            if 'oracle40' in case:
                torch.testing.assert_close(oracle, case['oracle40'].to(args.device), atol=.02, rtol=1e-5)
        for seed in args.intermediate_seeds:
            learned40, base40, raw40 = sample_structure(models['structure'], norms['structure'], coarse, args.steps, seed, args.device)
            assert_constraints(learned40, base40, CONSERVE_FACTOR, 40320.)
            sampled, sampled_up = sample_patches(models['intermediate'], norms['intermediate'], coarse, args.steps, seed, args.device)
            torch.testing.assert_close(sampled_up, guide_up, atol=0, rtol=0)
            branches, base23 = build_branches(coarse, guide_up, sampled.float(), learned40, base40, oracle, adapted, modern23)
            stage_arrays = dict(coarse_encoded=array(coarse), base40_metres=array(base40), raw40_metres=array(raw40),
                learned40_metres=array(learned40), sampled_intermediate_encoded=array(sampled.float()), guide_up_encoded=array(guide_up))
            if oracle is not None:
                stage_arrays.update(oracle40_metres=array(oracle), modern23_metres=array(modern23), adapted_target_encoded=array(adapted))
            np.savez_compressed(out/f'{case["label"]}-i{seed}-stages.npz', **stage_arrays)
            for name, guide23 in branches.items():
                if name.startswith(('oracle-', 'learned-')):
                    constraints23 = assert_constraints(guide23, base23, HANDOFF_FACTOR, 23040.)
                else:
                    constraints23 = constraint_metrics(guide23, base23, HANDOFF_FACTOR, 23040.)
                condition = encode(guide23[..., 47:65, 47:65]).float()
                for meso_seed in args.meso_seeds:
                    encoded, up = sample_patches(models['meso'], norms['meso'], condition, args.steps, meso_seed, args.device)
                    raw = decode(encoded.float()); base = decode(up)
                    corrected, correction = project_smooth(raw, base, 8, 2880., return_info=True)
                    constraints_meso = assert_constraints(corrected, base, 8, 2880.)
                    saved = dict(guide23_metres=array(guide23), meso_input_encoded=array(condition), meso_baseline_metres=array(base),
                        meso_raw_metres=array(raw), meso_smooth_corrected_metres=array(corrected))
                    np.testing.assert_array_equal(saved['meso_input_encoded'], array(encode(guide23[..., 47:65, 47:65]).float()))
                    if not all(np.isfinite(v).all() for v in saved.values()):
                        raise FloatingPointError('Nonfinite branch output')
                    np.savez_compressed(out/f'{case["label"]}-i{seed}-{name}-m{meso_seed}.npz', **saved)
                    row = {'case': case['label'], 'intermediate_seed': seed, 'meso_seed': meso_seed, 'branch': name,
                        'guide': describe(array(guide23)[0, 0, 30:82, 30:82], 23040.),
                        'meso_raw': describe(array(raw)[0, 0], 2880.), 'meso_corrected': describe(array(corrected)[0, 0], 2880.),
                        'raw_preservation': compare_meso(raw, guide23), 'corrected_preservation': compare_meso(corrected, guide23),
                        'handoff_constraints': constraints23, 'meso_constraints': constraints_meso, 'correction_info': correction}
                    records.append(row); done += 1
                    atomic_json(out/'results.json', {'status': 'in progress', 'settings': settings, 'records': records})
                    if done % 8 == 0 or done == total:
                        elapsed = time.perf_counter()-start
                        print(f'Completed {done}/{total} meso probes | {case["label"]}, structure/intermediate seed {seed}, meso seed {meso_seed} | ETA {(total-done)*elapsed/max(1,done)/60:.1f} min', flush=True)
            render_case(out, case, seed, branches, args.meso_seeds)
    atomic_json(out/'results.json', {'status': 'complete', 'settings': settings, 'records': records,
        'elapsed_seconds': time.perf_counter()-start, 'finite_outputs': True, 'training_launched': False, 'planet_generated': False, 'selection_changed': False})
    return records


def gallery(out, settings, cases):
    cards = []
    for case in cases:
        for seed in settings['intermediate_seeds']:
            name = f'{case["label"]}-i{seed}.png'
            if (out/name).exists():
                cards.append(f'<section><h2>{html.escape(case["label"])} · intermediate seed {seed}</h2><a href="{name}"><img src="{name}"></a></section>')
    targets = ''.join(f'<section><h2>{p.stem}</h2><img src="{p.name}"></section>' for p in sorted(out.glob('*-targets.png')))
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Terrain stage roles</title><style>'
        'body{font:16px system-ui;background:#f5f5f5;color:#26343b;margin:24px}p{max-width:1100px;line-height:1.5}img{width:100%;max-width:1600px}section{margin:35px 0}</style></head><body>'
        '<h1>Terrain targets and stage roles</h1><p>Oracle structure is supplied Earth information, not learned success. Within each structure pair only intermediate short-scale detail changes. '
        'Guide columns cover 1,198 km; raw meso columns cover the central 369 km. All use fixed metre colours and NW 5× shading. '
        'Smooth conservation is saved separately and is not applied in these primary meso panels. Independent local patches; no production blending.</p>'
        '<p>Formal holdouts use stored random orientations. Famous regions are development diagnostics. Generated cases use saved generated coast/relief prefixes; '
        'no Earth reset. PlanetC is not a round trip and is not included as a sampled branch.</p><p><a href="target-audit.json">Target audit</a> · <a href="settings.json">Provenance</a> · '
        '<a href="results.json">All measurements</a></p>'+targets+''.join(cards)+'</body></html>', encoding='utf-8')
    (out/'README.md').write_text('''# Terrain targets and stage roles

This is a bounded diagnosis, not a model selection or world-frequency evaluation.

Compare **oracle-direct** with **oracle-sampled-detail** to test whether the selected intermediate model's short-scale detail helps when structure is known. **Oracle-target-detail** supplies the corresponding training-target detail and is a feasibility reference. Compare **learned-direct** with **learned-sampled-detail** to test the same choice under the completed, unselected 40 km pilot. **Coarse-direct** supplies no intermediate or structural synthesis. **Modern23-oracle** supplies the original modern intermediate target directly. **Selected** is the current selected intermediate→meso chain.

Direct structural routes still use the existing deterministic 40→23 km handoff, permission mask and regional conservation. They remove the learned intermediate detail contribution; they do not supply the exact unmodified 40 km Earth field. The direct and detail routes share constraints. Reference/selected/coarse branches are not projected at intermediate scale, so their differences include target/baseline/constraint effects.

All branches feed their exact saved 18×18 encoded guide crop to the same frozen meso. Independent intermediate/structure seeds are crossed with independent meso seeds. Same seeds intentionally reuse noise across branches and regions. This is a diagnostic of conditional response, not evidence of production tile cloning. Primary figures show raw meso for every branch, avoiding selective postprocessing. Continuous corrected outputs and their measurements are saved separately.

The target audit covers all 55 formal geographically guarded holdouts from two caps. Three holdout examples are selected using only modern targets before generation: quiet inland, rugged inland and coast. These are stress tests, not a representative prevalence sample. Saved local orientations are retained. Alps, California and Plains reuse exact saved Earth-derived round-trip inputs. Generated examples reuse exact coarse inputs from two genuinely generated coast/relief prefixes, rather than Earth guides. Their names 60/200 identify illustrative Paleo source records used for land fractions, not age conditioning or paired modern targets. PlanetC's stepped Earth-derived B2W.png guide is not part of a complete generative round trip and is not sampled here.

Every regional guide panel is 1,198.08 km; every meso panel is 368.64 km. Fixed elevation colours −6,000 to +6,000 m; NW light, 45° altitude and labelled 5× shading. Fields originating at 40 km are resampled for aligned display only. Axes are local tangent-plane distances, not equirectangular pixel distances. Local component areas are tangent-plane descriptors; target conversion and conservation use spherical Jacobian weights.

Descriptors include physical local relief, slopes, connected gentle areas and relative-prominence objects. They are not a validated morphology score; positive-prominence components are not automatically mountain belts, and isolated negative components are not automatically plausible basins. Cropped objects are flagged. Visual arrangement remains decisive. Footprint-mean drift measures preservation of the supplied guide, not exact Earth reconstruction.

Sea level remains 0 m. No global shifts, height scaling, histogram remapping, forced coverage, training or planet generation occur. Selected checkpoints are only read. See index.html for all target examples and all seed panels; settings.json records hashes and arguments, results.json records every comparison. Console/errors have a durable log under outputs/terrain-stage-roles/job-logs.
''', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out'); p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--device', default='cuda'); p.add_argument('--steps', type=int, default=32)
    p.add_argument('--intermediate-seeds', type=int, nargs='+', default=[42, 314, 2718, 2026])
    p.add_argument('--meso-seeds', type=int, nargs='+', default=[113, 997])
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if args.steps < 8:
        p.error('Use at least eight sampler steps')
    for seeds in (args.intermediate_seeds, args.meso_seeds):
        if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds):
            p.error('Seeds must be distinct within each stage and in 0..2^63-1')
    if args.smoke:
        args.steps = 8; args.intermediate_seeds = [42]; args.meso_seeds = [113]
    out = Path(args.out).resolve() if args.out else OUT_ROOT/('probe-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a fresh output directory')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    args.device = pick_device(args.device)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    out.mkdir(parents=True, exist_ok=True)
    with console_log(OUT_ROOT/'job-logs', 'stage-roles') as log:
        print(f'Output: {out}\nTarget audit; bounded stage probes. No training or planet run.', flush=True)
        paths = {}; identities = {}
        for name, (source, expected) in MODELS.items():
            source = ROOT/source
            if digest(source) != expected:
                raise ValueError(f'Frozen checkpoint changed: {source}')
            paths[name] = source
            identities[name] = {'checkpoint': str(source), 'sha256': expected}
        models = {}; norms = {}
        models['intermediate'], norms['intermediate'], ck = load_checkpoint(paths['intermediate'], args.device)
        identities['intermediate']['step'] = ck['step']; del ck
        data = StructureData(args.data)
        selected, audit_arrays = target_audit(data, norms['intermediate'], out)
        cases = build_cases(selected, audit_arrays)
        if args.smoke:
            cases = [cases[1], cases[-2]]
        settings = {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': identities,
            'data_manifest_id': manifest_id(data.meta), 'data': str(Path(args.data).resolve()),
            'source_array_fingerprints': data.meta['source_array_fingerprints'],
            'case_selection': [{k: v for k, v in case.items() if not isinstance(v, (torch.Tensor, np.ndarray))} for case in cases],
            'intermediate_seeds': args.intermediate_seeds, 'meso_seeds': args.meso_seeds, 'steps': args.steps,
            'audit_only': args.audit_only, 'smoke': args.smoke, 'console_log': str(log),
            'limits': 'Independent bounded patches, no production tiling/blending. Target-selected holdouts are stress tests, not population estimates. Same local seeds deliberately shared across conditions.',
            'training_launched': False, 'planet_generated': False, 'selection_changed': False, 'sea_level_m': 0.}
        atomic_json(out/'settings.json', settings)
        print(f'Target audit complete: {data.meta["val_count"]} holdouts. Selected: {[(a,b["index"]) for a,b in selected]}', flush=True)
        if args.audit_only:
            atomic_json(out/'results.json', {'status': 'audit only', 'settings': settings, 'records': []})
        else:
            models['structure'], norms['structure'], ck = load_structure(paths['structure'], args.device)
            identities['structure']['step'] = ck['step']; del ck
            models['meso'], norms['meso'], ck = load_checkpoint(paths['meso'], args.device)
            identities['meso']['step'] = ck['step']; del ck
            atomic_json(out/'settings.json', settings)
            probes(cases, models, norms, args, out, settings)
        gallery(out, settings, cases)
        print(f'Finished. Review {out / "index.html"}. No model selected; no follow-on job starts.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
