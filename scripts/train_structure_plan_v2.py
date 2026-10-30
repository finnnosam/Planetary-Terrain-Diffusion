"""Long-budget planner (plan source A) or renderer training for structure-plan v2.

Planner: learns to generate the four continuous plan channels from the coarse guide
alone (no target height, no plan supplied). Renderer: learns 40 km heights from a
supplied plan; needed only for a new constraint policy (coast-only), because the
existing v1 continuous renderer already serves locked-200.

Default A1 run: continue the existing 5,000-update continuous planner (exact weight
copy, same plan encoding as the frozen renderer) to 100,000 updates, saving EMA
snapshots for a duration curve. Nothing else changes, so the result isolates training
duration. A2 (only if A1 fails learnability): --init scratch --width 64.

Resumable (Ctrl+C saves), run-locked, durable console log. No automatic follow-on job,
no model selection, sea level stays 0 m.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from planet_training import structure_plan_v2 as v2
from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import StructureData, StructureNorm, DEFAULT_DATA, decode
from toy_meso.common import autocast

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT/'outputs/structure/plan-v2'
V1_REVIEW = ROOT/'outputs/structure/spatial-supervision-v1/autonomous-review-5000'
V1_PLANNER = V1_REVIEW/'planner-pair.pt'
V1_RENDERER = V1_REVIEW/'renderer-pair.pt'
VAL_SIGMAS = [.05, .2, .5, 1., 2., 5.]


def load_v1(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'ema' not in ck or 'continuous' not in ck['ema']:
        raise ValueError(f'{path} is not a v1 paired checkpoint')
    return ck


def stats_from(path):
    """(height-normalization stats, continuous plan stats, policy) from a v1 pair or v2 checkpoint."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') == v2.VERSION:
        return ck['stats'], ck['plan_stats'], ck['policy'], ck['config']
    return ck['stats'], ck['plan_stats']['continuous'], 'locked-200', ck['config']


def batch_fields(data, args, norm, codec, device, indices=None, split='train', generator=None, eval_mode=False):
    raw, c, ctx = data.batch(args.batch if indices is None else 0, split, indices, device)
    if not eval_mode:
        k = int(torch.randint(4, ()))
        raw, c, ctx = [torch.rot90(x, k, (-2, -1)) for x in (raw, c, ctx)]
        if torch.rand(()) < .5:
            raw, c, ctx = [x.flip(-1) for x in (raw, c, ctx)]
        c = degrade_guides(ctx, c, None, recipe=STRUCTURE_RECIPE)
    else:
        c = degrade_guides(ctx, c, generator, sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    target, base = v2.target_with(raw, c, args.policy)
    _, up, cin = v2.condition_with(norm, c, args.policy)
    plan = codec.encode(decode(target), base)
    return target, base, up, cin, plan


def loss(model, args, norm, fields, sigma, noise):
    target, base, up, cin, plan = fields
    if args.task == 'planner':
        active = v2.active_for(base, args.policy).float()
        return v2.planner_loss(model, plan, cin, active, sigma, noise[:, :4]), None
    total, main, shape = v2.renderer_loss(model, norm, target, base, up, cin, plan, sigma, noise[:, :1], args.policy)
    return total, {'main': float(main.detach()), 'shape': float(shape.detach())}


@torch.no_grad()
def validate(model, args, norm, codec, data, device):
    count = data.meta['val_count']; g = torch.Generator(device=device).manual_seed(2026); total = 0.
    for start in range(0, count, 16):
        fields = batch_fields(data, args, norm, codec, device, np.arange(start, min(count, start+16)), 'val', g, eval_mode=True)
        n = len(fields[0])
        for s in VAL_SIGMAS:
            sigma = torch.full((n,), s, device=device); noise = torch.randn((n, 4, 64, 64), generator=g, device=device)
            with autocast(device):
                value, _ = loss(model, args, norm, fields, sigma, noise)
            total += float(value)*n/(len(VAL_SIGMAS)*count)
    return total


def train(args, out):
    started = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and not args.allow_cpu:
        raise ValueError('Training expects CUDA (use --allow-cpu only for tiny execution checks)')
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    data = StructureData(DEFAULT_DATA, args.seed)
    stats_path = Path(args.stats_from) if args.stats_from else (V1_RENDERER if args.policy == 'locked-200' else None)
    init_state = None
    if args.init == 'v1':
        source = V1_PLANNER if args.task == 'planner' else V1_RENDERER
        v1 = load_v1(source); base_config = v2.unet_config(v1['config'])
        init_state = v1['ema']['continuous']; height_stats = v1['stats']
        init = {'kind': 'v1', 'path': str(source), 'v1_step': v1['step']}
        if args.width and args.width != base_config['ch']:
            raise ValueError('--init v1 keeps the v1 width (32); use --init scratch for another width')
        del v1
    else:
        src = load_v1(V1_RENDERER); base_config = v2.unet_config(src['config'], args.width); height_stats = src['stats']; del src
        init = {'kind': 'scratch', 'width': base_config['ch']}
    if stats_path:
        h, plan_stats, stats_policy, _ = stats_from(stats_path)
        if stats_policy != args.policy:
            raise ValueError(f'--stats-from checkpoint uses policy {stats_policy}, not {args.policy}')
        if h != height_stats:
            raise ValueError('Height normalization differs from the v1 source')
        normalization = {'from': str(stats_path)}
    else:
        if args.task == 'planner':
            raise ValueError('A coast-only planner must share plan normalization with its renderer: pass --stats-from <coast-only renderer>')
        plan_stats, normalization = v2.fit_plan_stats(data, args.policy, STRUCTURE_RECIPE, device)
    norm = StructureNorm(height_stats); codec = v2.PlanCodec(plan_stats, args.policy)
    config = base_config
    optimization = {'batch': args.batch, 'seed': args.seed, 'lr': args.lr, 'warmup': 200, 'betas': [.9, .99], 'clip': 1., 'ema': args.ema}
    signature = {'version': v2.VERSION, 'plan_spec': v2.PLAN_SPEC, 'task': args.task, 'policy': args.policy, 'config': config,
                 'stats': height_stats, 'plan_stats': plan_stats, 'normalization_source': normalization, 'init': init,
                 'optimization': optimization, 'dataset_id': manifest_id(data.meta)}
    checkpoint = out/'latest.pt'; launch = out/'launch.json'
    resume = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if resume and any(resume.get(k) != v for k, v in signature.items()):
        bad = [k for k, v in signature.items() if resume.get(k) != v]
        raise ValueError(f'Resume identity mismatch in {bad}; choose a fresh --run folder')
    if launch.exists() and not resume:
        raise ValueError('Run folder has a launch record but no checkpoint; choose a fresh --run folder')
    model = (v2.build_planner if args.task == 'planner' else v2.build_renderer)(config).to(device)
    if init_state is not None:
        if args.task == 'planner':
            v2.planner_from_v1(init_state, model)
        else:
            model.load_state_dict(init_state)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99))
    step = 0
    if resume:
        model.load_state_dict(resume['model']); ema.load_state_dict(resume['ema']); opt.load_state_dict(resume['opt'])
        step = resume['step']; data.rng.bit_generator.state = resume['data_rng']; torch.set_rng_state(resume['torch_rng'])
        if device == 'cuda' and resume.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    elif init_state is not None:
        ema.load_state_dict(model.state_dict())
    atomic_json(launch, {**{k: v for k, v in signature.items() if k not in ('stats', 'plan_stats')}, 'arguments': vars(args),
                         'parameters': sum(p.numel() for p in model.parameters()), 'device': device,
                         'sea_level_m': 0., 'selection_changed': False, 'planet_generated': False})
    snapshots = sorted({int(s) for s in args.snapshots.split(',') if s}) if args.snapshots else []

    def state(full=True):
        ck = {**signature, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict()}
        if full:
            ck.update(opt=opt.state_dict(), data_rng=data.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None)
        return ck

    def save(path, full=True):
        tmp = path.with_suffix('.tmp'); torch.save(state(full), tmp); tmp.replace(path)

    print(f'{args.task} ({args.policy}, width {config["ch"]}, {sum(p.numel() for p in model.parameters()):,} parameters, init {init["kind"]}): '
          f'update {step}/{args.steps}, batch {args.batch}, wall-time limit {args.max_seconds:.0f}s.', flush=True)
    stop = [False]; previous = signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    status = 'complete'; log = (out/'log.jsonl').open('a', encoding='utf-8')

    def record(row):
        log.write(json.dumps(row)+'\n'); log.flush()
    try:
        if step == 0 or args.validate_at_start:
            v = validate(ema, args, norm, codec, data, device); record({'step': step, 'val_loss': v}); print(f'Validation at {step}: {v:.4f}', flush=True)
        window = []; last = time.perf_counter(); last_step = step
        while step < args.steps:
            if stop[0]:
                status = 'interrupted'; break
            if time.perf_counter()-started >= args.max_seconds-30:
                status = 'wall-time limit'; break
            fields = batch_fields(data, args, norm, codec, device)
            sigma = (torch.randn(args.batch, device=device)*1.2-1.2).exp(); noise = torch.randn((args.batch, 4, 64, 64), device=device)
            with autocast(device):
                value, extra = loss(model, args, norm, fields, sigma, noise)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite loss')
            for group in opt.param_groups:
                group['lr'] = args.lr*min(1., (step+1)/200)
            opt.zero_grad(set_to_none=True); value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
            with torch.no_grad():
                decay = min(args.ema, (1+step)/(10+step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-decay)
            step += 1; window.append(float(value.detach()))
            if step % 100 == 0 or step == args.steps:
                now = time.perf_counter(); rate = (step-last_step)/max(1e-3, now-last)
                eta = (args.steps-step)/max(rate, 1e-6)
                record({'step': step, 'loss': float(np.mean(window)), 'updates_per_second': rate, 'elapsed_seconds': now-started})
                print(f'Update {step}/{args.steps} | loss {np.mean(window):.4f} | {rate:.1f}/s | elapsed {(now-started)/60:.1f} min | ETA {eta/60:.0f} min', flush=True)
                window = []; last = now; last_step = step
            if step % args.save_every == 0:
                save(checkpoint)
            if step in snapshots:
                save(out/f'ema-{step:06d}.pt', full=False); print(f'Snapshot saved: ema-{step:06d}.pt', flush=True)
            if step % args.val_every == 0:
                v = validate(ema, args, norm, codec, data, device); record({'step': step, 'val_loss': v}); print(f'Validation at {step}: {v:.4f}', flush=True)
        save(checkpoint)
        if status == 'complete':
            save(out/f'ema-{step:06d}.pt', full=False)
    finally:
        log.close(); signal.signal(signal.SIGINT, previous)
    result = {'status': status, 'task': args.task, 'policy': args.policy, 'width': config['ch'], 'updates': step, 'target_updates': args.steps,
              'elapsed_seconds': time.perf_counter()-started, 'checkpoint': str(checkpoint), 'selection_changed': False, 'sea_level_m': 0.}
    atomic_json(out/'training-result.json', result)
    print(f'Finished ({status}) at update {step}. Checkpoint {checkpoint}. Rerun the same command to resume. No follow-on job starts.', flush=True)
    return 130 if status == 'interrupted' else (2 if status != 'complete' else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['planner', 'renderer'], default='planner')
    p.add_argument('--policy', choices=sorted(v2.POLICIES), default='locked-200')
    p.add_argument('--init', choices=['v1', 'scratch'], default='v1', help='v1: exact copy of the existing 5,000-update planner / 1,000-update renderer')
    p.add_argument('--width', type=int, help='U-Net base channels (v1 is 32; only with --init scratch)')
    p.add_argument('--stats-from', help='checkpoint whose plan normalization to share (required: coast-only planner -> its renderer)')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--max-seconds', type=float, default=110.)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--snapshots', default='10000,25000,50000')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--val-every', type=int, default=5000)
    p.add_argument('--validate-at-start', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--run')
    args = p.parse_args(argv)
    if args.width and args.init == 'v1':
        p.error('--width needs --init scratch')
    name = args.run or f'{args.task}-{args.policy}-{"v1cont" if args.init == "v1" else "w"+str(args.width or 32)}'
    out = Path(name) if Path(name).is_absolute() else OUT_ROOT/name
    out.mkdir(parents=True, exist_ok=True)
    from scripts.structure_job import console_log
    with console_log(OUT_ROOT/'job-logs', f'train-{out.name}'):
        with training_lock(out):
            return train(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
