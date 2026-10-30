"""Score structural plan sources on identical guides with the predeclared arrangement gates.

Plan sources, all rendered by the same frozen continuous renderer and seed:
  supplied   the feasible Earth reference's own plan (renderer capability; not generation)
  zero       an all-zero plan (lower bound)
  learned    a planner checkpoint (v1 paired or v2), several plan seeds        [--planner]
  procedural rule-based layouts (planet_training.procedural_plan), 2 seeds    [--procedural]

Case groups:
  train      training patches with degraded guides (the learnability test: can the planner
             produce structure on data it was trained on?)
  holdout    monitored holdout patches with degraded guides
  named      Alps / California / Plains, formal holdouts and the two generated coast/relief prefixes

Cases are selected by a fixed rule before any output is generated. Each output is compared
with the feasible reference on the same guide (arrangement_metrics.GATES). Generated prefixes
have no Earth reference; their metrics are reported but not gated.

No training. Sea level stays 0 m. Nothing is selected or promoted.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training import arrangement_metrics as am
from planet_training import procedural_plan as pp
from planet_training import structure_plan_v2 as v2
from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json
from planet_training.elevation_shading import show, legend
from planet_training.terrain_spatial_plan import continuous_fields
from planet_training.terrain_structure import StructureData, DEFAULT_DATA, decode, encode, conservative_resample

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT/'outputs/structure/plan-sources-v2'
JOINT = ROOT/'outputs/structure/joint-structure-v1/review-20261014-143545-003794'
ROUND_TRIPS = ROOT/'outputs/terrain-round-trips'
DEFAULT_RENDERER = ROOT/'outputs/structure/spatial-supervision-v1/autonomous-review-5000/renderer-pair.pt'
MESO = JOINT/'meso-checkpoint.pt'
MESO_SHA = '717e1c718fefe51fc0f5066de39618da9cc5a6b034c9df0fb1af09ebcbd7fcf6'
NAMED_EARTH = ['alps', 'california', 'plains']
NAMED_HOLDOUTS = {'holdout-rugged': 36, 'holdout-quiet': 16, 'holdout-coast': 6, 'extra-holdout-q50': 13, 'extra-holdout-q90': 21}
NAMED_GENERATED = ['generated-60', 'generated-200']
PLAN_SEEDS = [42, 314, 2718, 2026]
LAYOUT_SEEDS = [1, 2]
RENDER_SEED = 7919
MESO_SEEDS = [113, 997]
MESO_CASES = ['alps', 'california', 'generated-60']
SELECTION = {'candidates_per_split': 400, 'candidate_seed': 20261015, 'guide_seed_base': 2026, 'sigma_cells': 1.,
             'clean_fraction': 0., 'central_permitted_min': .5, 'require_structured_reference': True,
             'exclude': 'named formal holdout indices are not reselected as holdout data cases',
             'pick': 'evenly spaced quantiles of reference contrast_p50 among eligible candidates'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def data_cases(data, split, count, policy, device):
    """Rule-based selection of degraded-guide cases whose feasible reference is structured."""
    n = data.meta[f'{split}_count']
    rng = np.random.default_rng(SELECTION['candidate_seed']+(split == 'val'))
    candidates = np.sort(rng.choice(n, min(SELECTION['candidates_per_split'], n), replace=False))
    if split == 'val':  # the named formal holdouts are reviewed separately with their stored guides
        candidates = np.setdiff1d(candidates, list(NAMED_HOLDOUTS.values()))
    eligible = []
    for start in range(0, len(candidates), 32):
        idx = candidates[start:start+32]
        raw, c, ctx = data.batch(0, split, idx, device)
        degraded = torch.cat([degrade_guides(ctx[k:k+1], c[k:k+1], torch.Generator(device=device).manual_seed(SELECTION['guide_seed_base']+int(i)),
                                             sigma_cells=SELECTION['sigma_cells'], clean_fraction=SELECTION['clean_fraction'], recipe=STRUCTURE_RECIPE)
                              for k, i in enumerate(idx)])
        target, base = v2.target_with(raw, degraded, policy)
        ref = decode(target)
        for k, i in enumerate(idx):
            central = base[k, 0, 16:48, 16:48]
            if float((central > 200).float().mean()) < SELECTION['central_permitted_min']:
                continue
            m = am.arrangement(ref[k].cpu().numpy(), base[k].cpu().numpy())
            if SELECTION['require_structured_reference'] and not am.structured(m):
                continue
            eligible.append((m['contrast_p50_m'], int(i), degraded[k:k+1].cpu(), raw[k:k+1].cpu()))
    eligible.sort(key=lambda e: (e[0], e[1]))
    if not eligible:
        return []
    picks = sorted({round(q*(len(eligible)-1)) for q in np.linspace(0, 1, min(count, len(eligible)))})
    return [{'label': f'{split}-{eligible[p][1]}', 'group': 'train' if split == 'train' else 'holdout', 'split': split,
             'index': eligible[p][1], 'coarse': eligible[p][2], 'raw40': eligible[p][3], 'eligible_count': len(eligible)}
            for p in picks]


def named_cases(data):
    cases = []
    for name in NAMED_EARTH:
        t = np.load(ROUND_TRIPS/f'{name}.npz')
        raw = conservative_resample(torch.from_numpy(t['reference23_metres'])[:, 0], 64)[:, None]
        cases.append({'label': name, 'group': 'named', 'coarse': torch.from_numpy(t['shared_coarse_encoded']), 'raw40': raw,
                      'source': str(ROUND_TRIPS/f'{name}.npz')})
    for name, index in NAMED_HOLDOUTS.items():
        j = np.load(JOINT/f'{name}-s42-joint.npz')
        raw = torch.from_numpy(np.array(data.arrays['val'][index]))[None, None]
        cases.append({'label': name, 'group': 'named', 'coarse': torch.from_numpy(j['coarse_encoded']), 'raw40': raw,
                      'split': 'val', 'index': index, 'source': str(JOINT/f'{name}-s42-joint.npz')})
    for name in NAMED_GENERATED:
        path = JOINT/f'{name}-s42-joint.npz'
        if path.exists():
            cases.append({'label': name, 'group': 'named', 'coarse': torch.from_numpy(np.load(path)['coarse_encoded']), 'raw40': None,
                          'source': str(path)})
    return cases


@torch.no_grad()
def memorization_library(data, policy, device, cache):
    """Rel-161 km plan channel of every training patch (clean guide), as unit signatures."""
    if cache.exists():
        d = np.load(cache)
        if (str(d['dataset']) == data.meta.get('source_dataset_id', '') and str(d['policy']) == policy
                and str(d.get('version', '')) == LIBRARY_VERSION):
            return d['signatures'].astype(np.float32)
    sigs = []
    for start in range(0, data.meta['train_count'], 64):
        idx = np.arange(start, min(start+64, data.meta['train_count']))
        raw, c, _ = data.batch(0, 'train', idx, device)
        target, base = v2.target_with(raw, c, policy)
        rel = masked_rel161(decode(target), base, policy)
        sigs.append(am.plan_signature(rel).astype(np.float16))
    sigs = np.concatenate(sigs)
    np.savez_compressed(cache, signatures=sigs, dataset=data.meta.get('source_dataset_id', ''), policy=policy, version=LIBRARY_VERSION)
    return sigs.astype(np.float32)


LIBRARY_VERSION = 'rel161-policy-masked-v2'


def masked_rel161(z, base, policy):
    """161 km relative-height plan channel with the policy's frozen cells zeroed, as generated plans are."""
    return (continuous_fields(z, base)[:, 1]*v2.active_for(base, policy)[:, 0]).cpu().numpy()


def case_figure(out, case, rows, render_files):
    cols = 6
    groups = [['reference', 'supplied', 'zero'], [r for r in render_files if r.startswith('learned')],
              [r for r in render_files if r.startswith('procedural') and r.endswith('s1')],
              [r for r in render_files if r.startswith('procedural') and r.endswith('s2')]]
    groups = [g for g in groups if g]
    fig, axs = plt.subplots(len(groups), cols, figsize=(3.3*cols, 3.45*len(groups)), squeeze=False, constrained_layout=True)
    for ax in axs.flat:
        ax.set_axis_off()
    for i, names in enumerate(groups):
        for j, name in enumerate(names[:cols]):
            if name not in render_files:
                continue
            z = np.load(out/render_files[name])['projected40_metres'][0, 0, 16:48, 16:48]
            row = rows.get(name, {})
            verdict = '' if row.get('pass') is None else (' PASS' if row['pass'] else ' fail')
            ax = axs[i, j]; ax.set_axis_on()
            show(ax, z, f'{name}{verdict}', 40320.)
    fig.suptitle(f'{case}: plan sources through the same frozen renderer (seed {RENDER_SEED}); each panel 1,290 km\n'
                 '"reference" is the feasible Earth field (not rendered); "supplied" renders its plan; PASS/fail = predeclared arrangement gate',
                 fontsize=10)
    legend(fig, axs); fig.savefig(out/f'case-{case}.png', dpi=90); plt.close(fig)


@torch.no_grad()
def run(args, out):
    started = time.perf_counter()
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    renderer, norm, codec, rck = v2.load_any_renderer(args.renderer, device)
    policy = rck['policy']
    planner = None
    if args.planner:
        planner, pnorm, pcodec, pck = v2.load_any_planner(args.planner, device)
        if pck['policy'] != policy or pcodec.stats != codec.stats or pnorm.s != norm.s:
            raise ValueError('Planner and renderer disagree on policy, plan normalization or height normalization')
    data = StructureData(DEFAULT_DATA)
    steps = 4 if args.smoke else 32
    n_cases = 2 if args.smoke else args.cases_per_split
    cases = []
    if not args.skip_data_cases:
        cases += data_cases(data, 'train', n_cases, policy, device)+data_cases(data, 'val', n_cases, policy, device)
    named = named_cases(data)
    cases += named[:3] if args.smoke else named
    layouts = pp.LAYOUTS[:2] if args.smoke else pp.LAYOUTS
    plan_seeds = PLAN_SEEDS[:1] if args.smoke else PLAN_SEEDS[:args.plan_seeds]
    layout_seeds = LAYOUT_SEEDS[:1] if args.smoke else LAYOUT_SEEDS
    settings = {'version': 'plan-sources-v2', 'created': datetime.now().isoformat(timespec='seconds'), 'policy': policy,
                'renderer': str(Path(args.renderer).resolve()), 'renderer_sha256': digest(args.renderer), 'renderer_updates': rck.get('step'),
                'planner': str(Path(args.planner).resolve()) if args.planner else None,
                'planner_sha256': digest(args.planner) if args.planner else None, 'planner_updates': pck.get('step') if planner else None,
                'planner_width': pck['config']['ch'] if planner else None,
                'procedural': args.procedural, 'procedural_spec': pp.SPEC if args.procedural else None,
                'metrics_spec': am.SPEC, 'gates': am.GATES, 'selection': SELECTION, 'sampling_steps': steps,
                'render_seed': RENDER_SEED, 'plan_seeds': plan_seeds, 'layout_seeds': layout_seeds, 'layouts': layouts,
                'cases': [{k: v for k, v in c.items() if k not in ('coarse', 'raw40')} for c in cases],
                'smoke_only': args.smoke, 'sea_level_m': 0., 'training': False, 'selection_changed': False}
    atomic_json(out/'settings.json', settings)
    print(f'{len(cases)} cases, policy {policy}, device {device}. Settings: {out/"settings.json"}', flush=True)
    records, memo = [], {'generated': [], 'reference': []}
    library = None
    if planner and args.memorization:
        library = memorization_library(data, policy, device, OUT_ROOT/f'memorization-library-{policy}.npz')
    budget_hit = False
    for case in cases:
        if time.perf_counter()-started > args.max_seconds-20:
            budget_hit = True; break
        label = case['label']; coarse = case['coarse'].to(device)
        base = v2.condition_with(norm, coarse, policy)[0]
        base_np = base.cpu().numpy()
        reference = ref_metrics = None
        if case['raw40'] is not None:
            target, _ = v2.target_with(case['raw40'].to(device), coarse, policy)
            reference = decode(target)
            ref_metrics = am.arrangement(reference.cpu().numpy(), base_np)
        structured = bool(ref_metrics and am.structured(ref_metrics))
        plans = {'zero': torch.zeros((1, 4, 64, 64), device=device)}
        if reference is not None:
            plans['supplied'] = codec.encode(reference, base)
        if planner is not None:
            for s in plan_seeds:
                plans[f'learned-s{s}'] = v2.sample_planner(planner, pnorm, coarse, policy, s, steps, device)
        if args.procedural:
            for lay in layouts:
                for s in layout_seeds:
                    plans[f'procedural-{lay}-s{s}'] = pp.plan(lay, base, codec, s)[0]
        files, rows = {}, {}
        if reference is not None:
            files['reference'] = f'{label}-reference.npz'
            np.savez_compressed(out/files['reference'], projected40_metres=reference.cpu().numpy(), baseline40_metres=base_np,
                                coarse_encoded=coarse.cpu().numpy())
            rows['reference'] = {'pass': None}
        for name, plan in plans.items():
            z, _, raw = v2.sample_renderer(renderer, norm, coarse, plan, policy, RENDER_SEED, steps, device)
            check = v2.constraint_check(z, base, policy, 16, 40320.)
            m = am.arrangement(z.cpu().numpy(), base_np)
            verdict = am.compare(m, ref_metrics) if ref_metrics else {'ratios': None, 'pass': None}
            source = name.split('-s')[0] if name.startswith('learned') else ('procedural' if name.startswith('procedural') else name)
            layout = name[len('procedural-'):].rsplit('-s', 1)[0] if name.startswith('procedural') else None
            row = {'case': label, 'group': case['group'], 'source': source, 'layout': layout, 'name': name,
                   'structured': structured, 'metrics': m, 'reference_metrics': ref_metrics,
                   'ratios': verdict['ratios'], 'pass': verdict['pass'], 'constraints': check}
            if library is not None and name.startswith('learned') and case.get('split') == 'val':
                rel = codec.decode(plan, base)[:, 1].cpu().numpy()[0]
                row['nn_training_corr'] = am.nearest_training_correlation(rel, library); memo['generated'].append(row['nn_training_corr'])
            files[name] = f'{label}-{name}.npz'
            np.savez_compressed(out/files[name], projected40_metres=z.cpu().numpy(), raw40_metres=raw.cpu().numpy(),
                                encoded_plan=plan.cpu().numpy(), baseline40_metres=base_np)
            records.append(row); rows[name] = row
        if library is not None and reference is not None and case.get('split') == 'val':
            memo['reference'].append(am.nearest_training_correlation(masked_rel161(reference, base, policy)[0], library))
        case_figure(out, label, rows, files)
        atomic_json(out/'results.json', {'status': 'in progress', 'records': records})
        print(f'{label} ({case["group"]}, structured reference: {structured}) done, {time.perf_counter()-started:.0f}s', flush=True)
    meso_rows = []
    if args.meso and not budget_hit:
        meso_rows = run_meso(args, out, device, steps, norm, policy, records)
    verdicts = gate_verdicts(records, memo)
    atomic_json(out/'results.json', {'status': 'budget limited' if budget_hit else 'complete', 'records': records, 'meso': meso_rows,
                                     'elapsed_seconds': time.perf_counter()-started})
    atomic_json(out/'gates.json', verdicts)
    write_html(out, settings, verdicts, cases, meso_rows)
    print(json.dumps(verdicts['summary'], indent=1), flush=True)
    print(f'Review {"STOPPED AT BUDGET" if budget_hit else "complete"} in {time.perf_counter()-started:.0f}s: {out/"index.html"}', flush=True)
    return 2 if budget_hit else 0


def gate_verdicts(records, memo):
    summary = {}
    for group in ['train', 'holdout', 'named']:
        rows = [r for r in records if r['group'] == group and r['pass'] is not None]
        summary[group] = am.summarize(rows)
        layouts = sorted({r['layout'] for r in rows if r['layout']})
        for lay in layouts:
            summary[group][f'procedural:{lay}'] = am.summarize([{**r, 'source': 'x'} for r in rows if r['layout'] == lay])['x']
    memorization = None
    if memo['generated'] and memo['reference']:
        g, r = float(np.median(memo['generated'])), float(np.median(memo['reference']))
        memorization = {'generated_median_nn_corr': g, 'reference_median_nn_corr': r,
                        'flagged': g > r+am.GATES['memorization']['nn_corr_margin'], 'generated_count': len(memo['generated'])}
    learned = {g: summary[g].get('learned', {}) for g in summary}
    decisions = {
        'learnability: learned planner on training cases': learned['train'].get('set_pass'),
        'generalization: learned planner on holdout + named cases, no memorization flag': (
            learned['holdout'].get('set_pass') and learned['named'].get('set_pass')
            and not (memorization or {}).get('flagged', False)) if learned['holdout'] else None,
        'rule-based plans, all groups pooled': None,
    }
    pooled = [r for r in records if r['source'] == 'procedural' and r['pass'] is not None]
    if pooled:
        decisions['rule-based plans, all groups pooled'] = am.summarize(pooled)['procedural']['set_pass']
    return {'gates': am.GATES, 'summary': summary, 'memorization': memorization, 'decisions': decisions}


@torch.no_grad()
def run_meso(args, out, device, steps, norm, policy, records):
    from toy_meso.common import load_checkpoint, sample_patches
    if not MESO.exists():
        print('Meso checkpoint not found; skipping fine-stage check.', flush=True); return []
    if not args.smoke and digest(MESO) != MESO_SHA:
        raise ValueError('Selected meso checkpoint changed')
    meso, mnorm, _ = load_checkpoint(MESO, device)
    wanted = []
    for case in MESO_CASES:
        names = [r['name'] for r in records if r['case'] == case]
        for prefer in ['supplied', 'learned-s42', 'procedural-composite-s1', 'procedural-paired-ranges-s1', 'zero']:
            if prefer in names:
                wanted.append((case, prefer))
    rows = []
    for case, name in wanted:
        d = np.load(out/f'{case}-{name}.npz')
        coarse = named_case_coarse(case).to(device)
        z = torch.from_numpy(d['projected40_metres']).to(device); base = torch.from_numpy(d['baseline40_metres']).to(device)
        guide, base23 = v2.handoff_with(z, base, coarse, policy)
        v2.constraint_check(guide, base23, policy, 28, 23040., tolerance_m=.1)
        cond = encode(guide[..., 47:65, 47:65]).float()
        for ms in MESO_SEEDS:
            field = decode(sample_patches(meso, mnorm, cond, steps, ms, device)[0]).cpu().numpy()
            fname = f'{case}-{name}-meso{ms}.npz'
            np.savez_compressed(out/fname, raw_meso_metres=field, meso_input_encoded=cond.cpu().numpy(), direct23_metres=guide.cpu().numpy())
            rows.append({'case': case, 'name': name, 'meso_seed': ms, 'file': fname})
    for case in MESO_CASES:
        mine = [r for r in rows if r['case'] == case]
        if not mine:
            continue
        names = list(dict.fromkeys(r['name'] for r in mine))
        fig, axs = plt.subplots(len(MESO_SEEDS), len(names), figsize=(3.4*len(names), 3.5*len(MESO_SEEDS)), squeeze=False, constrained_layout=True)
        for j, name in enumerate(names):
            for i, ms in enumerate(MESO_SEEDS):
                r = next(r for r in mine if r['name'] == name and r['meso_seed'] == ms)
                show(axs[i, j], np.load(out/r['file'])['raw_meso_metres'][0, 0], f'{name}\nmeso seed {ms}', 2880.)
        fig.suptitle(f'{case}: raw meso (selected checkpoint) from each plan source; each panel 368.64 km', fontsize=10)
        legend(fig, axs); fig.savefig(out/f'meso-{case}.png', dpi=90); plt.close(fig)
    return rows


_NAMED_COARSE = {}


def named_case_coarse(case):
    if case not in _NAMED_COARSE:
        if case in NAMED_EARTH:
            _NAMED_COARSE[case] = torch.from_numpy(np.load(ROUND_TRIPS/f'{case}.npz')['shared_coarse_encoded'])
        else:
            _NAMED_COARSE[case] = torch.from_numpy(np.load(JOINT/f'{case}-s42-joint.npz')['coarse_encoded'])
    return _NAMED_COARSE[case]


def write_html(out, settings, verdicts, cases, meso_rows):
    def fmt(v):
        return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else (f'{v:.0%}' if isinstance(v, float) else str(v))
    table = '<table><tr><th>Group</th><th>Source</th><th>Gated samples</th><th>Pass fraction</th><th>Set passes (≥60%)</th></tr>'
    for group, sources in verdicts['summary'].items():
        for source, v in sources.items():
            table += f'<tr><td>{group}</td><td>{source}</td><td>{v["gated_samples"]}</td><td>{fmt(v["pass_fraction"])}</td><td>{fmt(v["set_pass"])}</td></tr>'
    table += '</table>'
    decisions = ''.join(f'<li><b>{k}</b>: {fmt(v)}</li>' for k, v in verdicts['decisions'].items())
    memo = verdicts['memorization']
    memo_html = (f'<p>Memorization: median nearest-training correlation of generated holdout plans {memo["generated_median_nn_corr"]:.3f} '
                 f'vs holdout references {memo["reference_median_nn_corr"]:.3f}; flagged: {memo["flagged"]}.</p>') if memo else ''
    imgs = ''.join(f'<details{" open" if c["group"] == "named" and c["label"] in ("alps", "california") else ""}><summary>{c["label"]} ({c["group"]})</summary>'
                   f'<img loading="lazy" src="case-{c["label"]}.png"></details>' for c in cases if (out/f'case-{c["label"]}.png').exists())
    meso = ''.join(f'<h3>{c}</h3><img loading="lazy" src="meso-{c}.png">' for c in MESO_CASES if (out/f'meso-{c}.png').exists())
    smoke = '<p><strong>SMOKE RUN: 4 sampling steps, few cases. Not a terrain result.</strong></p>' if settings['smoke_only'] else ''
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Plan sources review</title>
<style>body{{font:15px system-ui;margin:24px;background:#f6f6f4;color:#222}}img{{width:100%;max-width:1700px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:4px 9px;text-align:left}}details{{margin:14px 0}}p,li{{max-width:1100px;line-height:1.5}}</style></head><body>
<h1>Structural plan sources, policy {settings["policy"]}</h1>{smoke}
<p>Every source is rendered by the same frozen renderer and noise seed and compared with the feasible reference on the same guide.
A sample passes when long-ridge share ≥50%, flanked-lowland share ≥60% and median 322-km window contrast ≥60% of the reference's.
Only cases whose reference itself has structure are gated. Generated prefixes have no reference and are shown ungated.
"supplied" renders the Earth reference's own plan: it measures the renderer, not generation.</p>
<h2>Decisions</h2><ul>{decisions}</ul>{memo_html}<h2>Gate summary</h2>{table}
<h2>Cases</h2>{imgs}<h2>Raw meso</h2>{meso or "<p>Not run (use --meso).</p>"}
<p>Settings: <a href="settings.json">settings.json</a>, all metrics: <a href="results.json">results.json</a>, verdicts: <a href="gates.json">gates.json</a>.</p></body></html>'''
    (out/'index.html').write_text(html, encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--renderer', default=str(DEFAULT_RENDERER), help='v2 renderer or v1 paired renderer checkpoint')
    p.add_argument('--planner', help='v2 planner or v1 paired planner checkpoint (learned plan source)')
    p.add_argument('--procedural', action='store_true', help='include rule-based layouts')
    p.add_argument('--meso', action='store_true', help='also run the selected meso on Alps/California/generated-60')
    p.add_argument('--memorization', action='store_true', help='nearest-training-plan check for learned holdout plans')
    p.add_argument('--cases-per-split', type=int, default=8)
    p.add_argument('--plan-seeds', type=int, default=4)
    p.add_argument('--skip-data-cases', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--max-seconds', type=float, default=1200.)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out')
    args = p.parse_args(argv)
    if not (args.planner or args.procedural):
        p.error('Give --planner and/or --procedural')
    out = Path(args.out).resolve() if args.out else OUT_ROOT/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a fresh review folder')
    out.mkdir(parents=True, exist_ok=True)
    from scripts.structure_job import console_log
    with console_log(OUT_ROOT/'job-logs', 'review-plan-sources'):
        return run(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
