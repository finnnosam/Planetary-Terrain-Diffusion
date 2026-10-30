"""Paired oracle-plan interventions; no training, promotion or planet run."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import (
    StructureData, DEFAULT_DATA, conservative_resample, decode, project, target_for, handoff,
)
from planet_training.terrain_plan_oracle import load, sample, plan_for, plan_agreement
from scripts.probe_structure_stage_roles import (
    build_cases, array, tensor, digest, assert_constraints,
)
from scripts.diagnose_terrain_structure import image, legend
from scripts.structure_job import console_log
from scripts.train_terrain_plan_oracle import OUT_ROOT
from toy_meso.model import upsample_condition
from toy_meso.common import pick_device


def paired_inputs(source_folder):
    audit = json.loads((source_folder/'target-audit.json').read_text())
    selected = [(s['label'], audit['records'][s['index']]) for s in audit['selected_holdouts']]
    with np.load(source_folder/'target-audit-arrays.npz') as a:
        arrays = tuple(tensor(a[k]) for k in ('modern23_encoded', 'degraded_coarse_encoded', 'translated23_encoded', 'constrained40_metres', 'baseline40_metres'))
    cases = [case for case in build_cases(selected, arrays) if case['modern_encoded'] is not None]
    settings = json.loads((source_folder/'settings.json').read_text())
    previous = {c['label']: c for c in settings['case_selection']}
    for case in cases:
        if 'path' in case and case['sha256'] != previous[case['label']]['sha256']:
            raise ValueError('Previously audited regional input changed')
    return cases, settings


@torch.no_grad()
def review(args, out):
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = pick_device(args.device)
    control, cnorm, cc = load(args.control, device)
    oracle, onorm, oc = load(args.oracle, device)
    if cc['mode'] != 'control' or oc['mode'] != 'oracle':
        raise ValueError('Specify the matched control and oracle checkpoints in that order')
    for key in ('step', 'dataset_id', 'source_sha256', 'config', 'optimization', 'stats', 'objective', 'plan_spec'):
        if cc[key] != oc[key]: raise ValueError(f'Unmatched pilot histories: {key}')
    if not args.smoke and cc['step'] <= 2:
        raise ValueError('These are execution-smoke checkpoints; use --smoke or train the pilots')
    data = StructureData(DEFAULT_DATA)
    if manifest_id(data.meta) != cc['dataset_id']: raise ValueError('Pilot source data identity changed')
    cases, previous = paired_inputs(Path(args.inputs).resolve())
    if previous['data_manifest_id'] != cc['dataset_id']: raise ValueError('Audited input dataset changed')
    # Verify formal holdout sources rather than trusting derived input archives.
    for case in cases:
        if 'index' in case:
            raw, _, _ = data.batch(0, 'val', np.array([case['index']]))
            recomputed = conservative_resample(decode(case['modern_encoded'])[:, 0], 64)[:, None]
            torch.testing.assert_close(raw, recomputed, atol=.02, rtol=1e-5)
    seeds, steps = ([42], 8) if args.smoke else (args.seeds, 32)
    if args.smoke: cases = [cases[1], cases[4]]
    settings = {'created_utc': datetime.now(timezone.utc).isoformat(), 'step': cc['step'],
        'control': {'path': str(Path(args.control).resolve()), 'sha256': digest(args.control)},
        'oracle': {'path': str(Path(args.oracle).resolve()), 'sha256': digest(args.oracle)},
        'checkpoint_sources': args.checkpoint_sources,
        'paired_dataset_id': cc['dataset_id'], 'source_ema_sha256': cc['source_sha256'], 'plan_spec': cc['plan_spec'],
        'inputs': str(Path(args.inputs).resolve()), 'seeds': seeds, 'steps': steps, 'smoke_only': args.smoke,
        'cases': [{k: v for k, v in c.items() if not isinstance(v, (torch.Tensor, np.ndarray))} for c in cases],
        'limits': 'Oracle plan test, not a learned global planner. Feasible counterfactual is a rotated/reprojected target residual; it can be out of the natural conditional distribution. No meso or tiled sampling.',
        'selection_changed': False, 'training_launched': False, 'planet_generated': False, 'sea_level_m': 0.}
    atomic_json(out/'settings.json', settings)
    records = []; cards = []
    for case in cases:
        coarse = case['coarse'].to(device)
        raw40 = conservative_resample(decode(case['modern_encoded'].to(device))[:, 0], 64)[:, None]
        encoded_target, base = target_for(raw40, coarse); target = decode(encoded_target)
        counter = project(base+torch.rot90(target-base, 1, (-2, -1)), base)
        assert_constraints(target, base, 16, 40320.); assert_constraints(counter, base, 16, 40320.)
        correct_plan, counter_plan = plan_for(target, base), plan_for(counter, base)
        absent = torch.zeros_like(correct_plan)
        for seed in seeds:
            rows = [('feasible-target', target, target, correct_plan), ('feasible-counterfactual', counter, counter, counter_plan)]
            for name, model, norm, plan in [('control', control, cnorm, absent), ('oracle-correct', oracle, onorm, correct_plan),
                                          ('oracle-absent', oracle, onorm, absent), ('oracle-counterfactual', oracle, onorm, counter_plan)]:
                z, _, raw = sample(model, norm, coarse, plan, steps, seed, device)
                rows.append((name, z, raw, plan))
            fig, axes = plt.subplots(len(rows), 3, figsize=(11, len(rows)*3.1+.7), squeeze=False, constrained_layout=True)
            for row, (name, projected, raw, plan) in enumerate(rows):
                direct23, _, base23 = handoff(projected, base, upsample_condition(coarse, 7), coarse)
                checks = assert_constraints(projected, base, 16, 40320.)
                assert_constraints(direct23, base23, 28, 23040.)
                saved = {'coarse_encoded': array(coarse), 'baseline40_metres': array(base), 'plan': array(plan),
                    'raw40_metres': array(raw), 'projected40_metres': array(projected), 'direct23_metres': array(direct23),
                    'correct_plan': array(correct_plan), 'counterfactual_plan': array(counter_plan)}
                if not all(np.isfinite(a).all() for a in saved.values()): raise ValueError('Nonfinite review field')
                np.savez_compressed(out/f'{case["label"]}-seed{seed}-{name}.npz', **saved)
                records.append({'case': case['label'], 'seed': seed, 'branch': name, 'constraints': checks,
                    'correct_plan_agreement': plan_agreement(projected, base, correct_plan),
                    'counterfactual_plan_agreement': plan_agreement(projected, base, counter_plan)})
                for col, (title, field) in enumerate([('Raw 40 km', raw), ('Constrained 40 km', projected), ('Direct 23 km handoff', direct23)]):
                    field23 = F.interpolate(field, size=(112, 112), mode='bilinear', align_corners=False) if col < 2 else field
                    z = array(field23)[0, 0, 30:82, 30:82]
                    axes[row, col].imshow(image(z, 23040.), extent=(-599.04, 599.04, -599.04, 599.04), interpolation='nearest')
                    axes[row, col].set_title(name+'\n'+title, fontsize=9); axes[row, col].set_xlabel('km'); axes[row, col].set_ylabel('km')
            fig.suptitle(f'{"SMOKE ONLY: " if args.smoke else ""}{case["label"]}, paired seed {seed}; same 1,198.08 km square\nTargets/plans are supplied information. Counterfactual rotates residual under fixed heights; no global elevation shift.', fontsize=11)
            legend(fig, axes); name = f'{case["label"]}-seed{seed}.png'; fig.savefig(out/name, dpi=115); plt.close(fig)
            cards.append(name)
            atomic_json(out/'results.json', {'status': 'in progress', 'settings': settings, 'records': records})
            print(f'Completed {len(cards)}/{len(cases)*len(seeds)}: {case["label"]}, seed {seed}', flush=True)
    atomic_json(out/'results.json', {'status': 'complete', 'settings': settings, 'records': records, 'visual_review_required': True})
    text = '<!doctype html><html><head><meta charset="utf-8"><title>Oracle plan review</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5}img{width:100%;max-width:1400px}p{max-width:1000px;line-height:1.5}</style></head><body>'
    text += '<h1>Oracle plan intervention</h1><p>Matched control/oracle training histories. All rows use fixed metre colours and NW 5× shading; all panels cover 1,198 km. '
    text += 'Correct/absent/counterfactual plans hold coarse heights and noise fixed. Target reference rows are not learned successes. '
    text += 'This test does not generate a global plan or use meso. A passing constraint check or plan agreement is not terrain-quality approval.</p>'
    text += ''.join(f'<section><h2>{html.escape(name[:-4])}</h2><a href="{name}"><img src="{name}"></a></section>' for name in cards)+'</body></html>'
    (out/'index.html').write_text(text, encoding='utf-8')
    print(f'Finished: {out / "index.html"}. No checkpoint selected or follow-on job started.', flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', default=str(OUT_ROOT/'control/latest.pt'))
    p.add_argument('--oracle', default=str(OUT_ROOT/'oracle/latest.pt'))
    p.add_argument('--inputs', default='outputs/terrain-stage-roles/paired-v1')
    p.add_argument('--out'); p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 314, 2718, 2026]); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**63 for s in args.seeds): p.error('Invalid paired seeds')
    out = Path(args.out).resolve() if args.out else OUT_ROOT/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a fresh review directory')
    out.mkdir(parents=True, exist_ok=True)
    with console_log(OUT_ROOT/'job-logs', 'review'):
        args.checkpoint_sources = {'control': str(Path(args.control).resolve()), 'oracle': str(Path(args.oracle).resolve())}
        with ExitStack() as locks:
            for directory in sorted({str(Path(path).parent) for path in args.checkpoint_sources.values()}):
                locks.enter_context(training_lock(directory))
            for name, path in args.checkpoint_sources.items():
                shutil.copyfile(path, out/f'{name}-checkpoint.pt')
        args.control, args.oracle = str(out/'control-checkpoint.pt'), str(out/'oracle-checkpoint.pt')
        return review(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
