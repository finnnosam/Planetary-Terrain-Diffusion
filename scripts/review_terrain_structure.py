"""Paired regional review of the 40-km structure stage, its control and oracle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from planet_training.adaptation import coarse_context, degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, bridge_patch, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import (DEFAULT_RUN, DEFAULT_DATA, OBJECTIVE, CONSERVE_FACTOR,
    HANDOFF_FACTOR, StructureData, load, sample, encode, decode, target_for,
    conservative_resample, handoff, project, constraint_metrics, eligibility)
from planet_training.train_terrain_structure import evaluate
from planet_training.smooth_conservation import project_smooth, grid_jumps
from toy_meso.common import load_checkpoint, sample_patches, pick_device
from toy_meso.sphere import GlobalRaster
from scripts.review_intermediate_structure import CASES, MESO, MESO_SHA, LAYOUT, LAYOUT_SHA, frame, sha, measurements
from scripts.train_intermediate_structure import BASELINE, BASELINE_SHA


def assert_constraints(z, base, factor, spacing):
    result = constraint_metrics(z, base, factor, spacing)
    if not torch.isfinite(z).all(): raise FloatingPointError('Nonfinite structure field')
    if result['added_footprint_mean_drift_max_m'] > .05 or result['protected_height_change_max_m'] > .005 or result['sign_disagreement_fraction']:
        raise ValueError(f'Coastal/conservation constraint failed: {result}')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default=DEFAULT_RUN); p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', default='latest.pt', choices=['latest.pt', 'best.pt'])
    p.add_argument('--out'); p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 314, 2718, 2026])
    p.add_argument('--meso-constraint', default='smooth-v1', choices=['block-v1', 'smooth-v1'],
                   help='Explicit meso postprocessing version; structure checkpoint/targets remain unchanged')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    if a.smoke:
        if a.run == DEFAULT_RUN: a.run = 'outputs/structure/checks/train-smoke'
        if a.data == DEFAULT_DATA: a.data = 'outputs/structure/checks/data-smoke'
        a.seeds = [42]
    if len(a.seeds) != len(set(a.seeds)) or any(s < 0 or s >= 2**63 for s in a.seeds): p.error('Provide distinct seeds in 0..2^63-1')
    cases = [CASES[1], CASES[3]] if a.smoke else CASES
    steps = 8 if a.smoke else 32
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = pick_device(a.device); data = StructureData(a.data)
    for file, expected in [(BASELINE, BASELINE_SHA), (MESO, MESO_SHA), (LAYOUT, LAYOUT_SHA)]:
        if sha(file) != expected: raise ValueError(f'Frozen input changed: {file}')
    out = Path(a.out) if a.out else Path(a.run)/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a fresh review folder')
    with training_lock(a.run):
        path = Path(a.run)/a.checkpoint
        if not path.exists(): raise ValueError('Run the new structure training pilot first')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('dataset_id') != manifest_id(data.meta) or ck.get('objective') != OBJECTIVE: raise ValueError('Candidate data/objective changed')
        step = ck['step']; del ck
        out.mkdir(parents=True)
        files = {}
        for name, source in [('structure', path), ('intermediate', BASELINE), ('meso', MESO)]:
            dest = out/f'{name}-checkpoint.pt'; shutil.copyfile(source, dest); files[name] = dest
    settings = {'created_utc': datetime.now(timezone.utc).isoformat(), 'step': step, 'dataset_id': manifest_id(data.meta),
                'objective': OBJECTIVE, 'models': {k: {'path': str(v.resolve()), 'sha256': sha(v)} for k, v in files.items()},
                'layout': LAYOUT, 'layout_sha256': LAYOUT_SHA, 'cases': cases, 'seeds': a.seeds, 'sampler_steps': steps,
                'smoke_test': a.smoke, 'full_planet_generation': False, 'selection_changed': False,
                'meso_constraint': a.meso_constraint,
                'branches': ['original frozen pipeline', 'constrained control without learned structure', 'new stage plus same control detail', 'Earth-derived oracle where available'],
                'limits': 'Independent north-up gnomonic patches, no production sphere tiling/blending. 645-km conservation protects regional means, not each 161-km value. Ocean/shelf and <=200-m input terrain locked. Meso conservation applied to control/new/oracle, not original. Oracle targets are constrained modern 40-km averages, not generated terrain.'}
    atomic_json(out/'review-settings.json', settings)
    print(f'\n40-KM STRUCTURE REGIONAL REVIEW\nUpdate {step}; {len(cases)} regions x {len(a.seeds)} seeds. Output: {out}', flush=True)
    if a.check_only: print('Review inputs frozen. No model sampling or validation.', flush=True); return 0
    structure, norm, _ = load(files['structure'], device)
    intermediate, inorm, _ = load_checkpoint(files['intermediate'], device)
    meso, mnorm, _ = load_checkpoint(files['meso'], device)
    print('Checking complete structure holdout...', flush=True)
    validation = evaluate(structure, norm, data, device)
    print(f'Structure validation (different objective from legacy loss): {validation}', flush=True)
    earth = GlobalRaster(np.load('outputs/planet_training/data/cache/etopo-filtered.npy'))
    layout = GlobalRaster.from_geotiff(LAYOUT)
    from scripts.diagnose_terrain_structure import image, legend
    import matplotlib.pyplot as plt
    records = []; started = time.perf_counter()
    try:
        for label, lat, lon, kind in cases:
            target, coarse = bridge_patch(earth if kind == 'earth' else layout, frame(lat, lon))
            coarse = torch.from_numpy(coarse)[None, None].to(device)
            true23 = decode(torch.from_numpy(target.astype(np.float32))[None, None].to(device)) if kind == 'earth' else None
            if kind == 'earth':
                ctx = torch.from_numpy(coarse_context(earth, frame(lat, lon), 112))[None, None].to(device)
                coarse = degrade_guides(ctx, coarse, None, sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
                modern40 = conservative_resample(true23[:, 0], 64)[:, None]
                oracle_encoded, _ = target_for(modern40, coarse); oracle = decode(oracle_encoded)
            else: oracle = None
            for seed in a.seeds:
                stage40, base40, raw40 = sample(structure, norm, coarse, steps, seed, device)
                constraints40 = assert_constraints(stage40, base40, CONSERVE_FACTOR, 40320.)
                frozen, original_up = sample_patches(intermediate, inorm, coarse, steps, seed, device)
                new23, control23, base23 = handoff(stage40, base40, frozen, coarse)
                branches = [('original', decode(frozen), base23), ('control', control23, base23),
                            ('new-stage', new23, F.interpolate(stage40, size=(112, 112), mode='bilinear', align_corners=False))]
                if oracle is not None:
                    oracle23, _, _ = handoff(oracle, base40, frozen, coarse)
                    branches.append(('Earth-oracle', oracle23, F.interpolate(oracle, size=(112, 112), mode='bilinear', align_corners=False)))
                entry = {'region': label, 'latitude': lat, 'longitude': lon, 'seed': seed, 'structure_constraints': constraints40, 'branches': {}}
                for name, field in [('structure40', stage40), ('unconstrained-structure40', raw40), ('baseline40', base40)]+([('Earth-oracle40', oracle)] if oracle is not None else []):
                    np.save(out/f'{label}-seed{seed}-{name}.npy', field[0, 0].float().cpu().numpy())
                fig, axs = plt.subplots(len(branches), 3, figsize=(12, 3.7*len(branches)), squeeze=False, constrained_layout=True)
                profiles = []
                for row, (name, z23, display40) in enumerate(branches):
                    constraints23 = assert_constraints(z23, base23, HANDOFF_FACTOR, 23040.) if name != 'original' else constraint_metrics(z23, base23, HANDOFF_FACTOR, 23040.)
                    condition = encode(z23[..., 47:65, 47:65]).float()
                    fine_encoded, fine_up = sample_patches(meso, mnorm, condition, steps, seed, device)
                    fine_base = decode(fine_up); fine_raw = decode(fine_encoded)
                    smooth_info = None
                    if name == 'original': fine = fine_raw
                    elif a.meso_constraint == 'smooth-v1':
                        fine, smooth_info = project_smooth(fine_raw, fine_base, 8, 2880., return_info=True)
                    else: fine = project(fine_raw, fine_base, 8, 2880.)
                    fine_constraints = constraint_metrics(fine, fine_base, 8, 2880.) if name == 'original' else assert_constraints(fine, fine_base, 8, 2880.)
                    z = z23[0, 0].float().cpu().numpy(); f = fine[0, 0].float().cpu().numpy(); g = base23[0, 0].float().cpu().numpy()
                    entry['branches'][name] = {'intermediate': measurements(z, g, 23040., coarse[0, 0].cpu().numpy(), 7),
                        'meso': measurements(f, fine_base[0, 0].float().cpu().numpy(), 2880., condition[0, 0].cpu().numpy(), 8),
                        'intermediate_constraints': constraints23, 'meso_constraints': fine_constraints,
                        'meso_grid_jumps': grid_jumps(fine), 'raw_meso_grid_jumps': grid_jumps(fine_raw),
                        'smooth_correction_info': smooth_info}
                    if true23 is not None: entry['branches'][name]['central_Earth_rms_m'] = float((z23[..., 30:-30, 30:-30]-true23[..., 30:-30, 30:-30]).square().mean().sqrt())
                    np.save(out/f'{label}-seed{seed}-{name}-intermediate.npy', z); np.save(out/f'{label}-seed{seed}-{name}-meso.npy', f)
                    for col, (title, field, spacing) in enumerate([('Structure / guide', display40[0, 0, 30:-30, 30:-30].float().cpu().numpy(), 23040.), ('Intermediate', z[30:-30, 30:-30], 23040.), ('Meso', f, 2880.)]):
                        half = len(field)*spacing/2000
                        axs[row, col].imshow(image(field, spacing), extent=(-half, half, -half, half), interpolation='nearest')
                        axs[row, col].set_title(f'{name}: {title}\n{2*half:.0f} km', fontsize=10); axs[row, col].set_xlabel('km'); axs[row, col].set_ylabel('km')
                    profiles.append((name, z[30:-30, 30:-30]))
                prefix = 'SMOKE ONLY: ' if a.smoke else ''
                fig.suptitle(f'{prefix}{label}, seed {seed}, {steps} steps; meso correction {a.meso_constraint}\nOriginal / control / new stage / Earth oracle. Fixed colours and NW 5x shading; meso central 369 km.')
                legend(fig, axs); fig.savefig(out/f'{label}-seed{seed}.png', dpi=115); plt.close(fig)
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                x = (np.arange(52)+.5-26)*23.04
                for i, ax in enumerate(axes):
                    for name, z in profiles: ax.plot(x, z[26] if i == 0 else z[::-1, 26], label=name)
                    if true23 is not None:
                        z = true23[0, 0, 30:-30, 30:-30].cpu().numpy(); ax.plot(x, z[26] if i == 0 else z[::-1, 26], 'k--', label='Original modern Earth')
                    ax.set_title(['West-east', 'South-north'][i]); ax.set_ylim(-6000, 6000); ax.set_xlabel('km'); ax.set_ylabel('metres'); ax.grid(alpha=.2); ax.legend(fontsize=8)
                fig.suptitle(f'{label}, seed {seed}: raw centre profiles, fixed elevation scale')
                fig.savefig(out/f'{label}-seed{seed}-profiles.png', dpi=115); plt.close(fig)
                records.append(entry)
                atomic_json(out/'regional-review.json', {'status': 'in progress', 'settings': settings, 'validation': validation, 'regions': records})
                total = len(cases)*len(a.seeds); done = len(records); elapsed = time.perf_counter()-started
                print(f'Completed {done}/{total}: {label}, seed {seed}; ETA {(total-done)*elapsed/done/60:.1f} min', flush=True)
    except KeyboardInterrupt:
        print('Review stopped. Completed cases remain saved; rerun into a fresh output directory.', flush=True); return 130
    atomic_json(out/'regional-review.json', {'status': 'complete', 'settings': settings, 'validation': validation, 'regions': records,
        'visual_review_required': True, 'selection_changed': False, 'full_planet_generation': False})
    (out/'README.txt').write_text(f'Meso correction: {a.meso_constraint}. Compare new-stage to control to isolate learned structure. Original is the unmodified selected pipeline; control shares filtering/coastal constraints with the candidate. Earth-oracle uses known modern structure and is not a learned success. First column displays 40-km fields resampled on the same 23-km display grid; original/control show the coarse guide. Intermediate/structure panels cover 1198 km, meso covers central 369 km. Review compact belts, connected valleys, secondary ranges, plains, shelves and coarse/block-boundary drift across all seeds. Constraint passing is not morphology approval. '+('Smoke execution only; not quality evidence.' if a.smoke else 'No model promoted or globe generated.')+'\n', encoding='utf-8')
    print(f'Regional review complete: {out}. Inspect images before any selection or planet generation.', flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
