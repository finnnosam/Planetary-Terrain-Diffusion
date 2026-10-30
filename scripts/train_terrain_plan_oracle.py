"""Matched control/oracle training jobs, separate from selected terrain models."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import StructureData, DEFAULT_DATA, StructureNorm, OBJECTIVE, target_for, decode
from planet_training.terrain_plan_oracle import VERSION, PLAN_SPEC, build, initialize_from_source, loss, plan_for, condition
from scripts.probe_structure_stage_roles import MODELS, ROOT, digest, render_fields
from scripts.diagnose_terrain_structure import image
from scripts.structure_job import console_log
from toy_meso.common import pick_device, autocast

OUT_ROOT = ROOT/'outputs/structure/plan-oracle-v1'


@torch.no_grad()
def evaluate(model, norm, data, device, mode):
    results = {}
    for label, smoothing in [('clean', False), ('smooth', True)]:
        total = 0.; count = data.meta['val_count']
        g = torch.Generator(device=device).manual_seed(2026)
        for start in range(0, count, 16):
            raw, coarse, context = data.batch(0, 'val', np.arange(start, min(start+16, count)), device)
            if smoothing:
                coarse = degrade_guides(context, coarse, g, sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
            target, base = target_for(raw, coarse)
            plan = plan_for(decode(target), base)
            if mode == 'control': plan.zero_()
            _, up, cin = condition(norm, coarse, plan)
            residual = norm.residual(target, up)
            for sigma in torch.exp(torch.linspace(-3, 1.5, 8, device=device)):
                noisy = residual+torch.randn(residual.shape, generator=g, device=device)*sigma
                with autocast(device): predicted = model(noisy, sigma.expand(len(raw)), cin)
                weight = (sigma*sigma+.25)/(sigma*.5)**2
                total += float(weight*(predicted.float()-residual).square().mean())*len(raw)/(8*count)
        results[label] = total
    return results


def preflight(out, norm, data, device):
    indices = [i for i in (16, 36, 6) if i < data.meta['val_count']]
    if not indices: indices = [0]
    raw, coarse, context = data.batch(0, 'val', np.asarray(indices), device)
    coarse = degrade_guides(context, coarse, torch.Generator(device=device).manual_seed(2026), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    target, base = target_for(raw, coarse); target = decode(target); plan = plan_for(target, base)
    condition(norm, coarse, plan)
    np.savez_compressed(out/'plan-preflight.npz', indices=np.asarray(indices), coarse_encoded=coarse.cpu().numpy(),
        target_metres=target.cpu().numpy(), baseline_metres=base.cpu().numpy(), oracle_plan=plan.cpu().numpy())
    fields = []
    for index, z, b in zip(indices, target[:, 0], base[:, 0]):
        fields += [(f'Holdout {index}: input baseline', b.cpu().numpy()), (f'Holdout {index}: feasible target', z.cpu().numpy())]
    render_fields(out/'target-preflight.png', 'Plan pilot: unchanged feasible targets; entire 2,580.48 km patches', fields, 40320.,
        'Oracle channels supply position/type/direction, not exact elevations. Stored local orientations.', columns=2)
    fig, axes = plt.subplots(len(indices), 4, figsize=(13, len(indices)*3.2+.6), squeeze=False, constrained_layout=True)
    q = (np.arange(64)+.5-32)*40.32
    xx, yy = np.meshgrid(q, -q)
    for row, index in enumerate(indices):
        planes = plan[row].cpu().numpy()
        for col, z in enumerate((base[row, 0].cpu().numpy(), target[row, 0].cpu().numpy())):
            axes[row, col].imshow(image(z, 40320.), extent=(-1290.24, 1290.24, -1290.24, 1290.24), interpolation='nearest')
        axes[row, 2].imshow(planes[0], cmap=ListedColormap(['#4a8bb1', '#e5e5df', '#a65f3d']), vmin=-1, vmax=1,
            extent=(-1290.24, 1290.24, -1290.24, 1290.24), interpolation='nearest')
        angle = .5*np.arctan2(planes[3], planes[2])
        selected = (np.indices((64, 64))[0] % 4 == 0) & (np.indices((64, 64))[1] % 4 == 0) & (np.abs(planes[0]) > .5)
        axes[row, 2].quiver(xx[selected], yy[selected], 80*np.cos(angle[selected]), 80*np.sin(angle[selected]),
            angles='xy', scale_units='xy', scale=1, pivot='middle', headlength=0, headaxislength=0, headwidth=0, color='#222222', width=.005)
        axes[row, 3].imshow(planes[1], cmap=ListedColormap(['#e5e5df', '#56824e']), vmin=0, vmax=1,
            extent=(-1290.24, 1290.24, -1290.24, 1290.24), interpolation='nearest')
        for col, title in enumerate(('Baseline elevation', 'Feasible target elevation', 'Blue low / brown high; corridor axes', 'Green: quiet permitted terrain')):
            axes[row, col].set_title(f'Holdout {index}\n{title}', fontsize=9)
            axes[row, col].set_xlabel('km'); axes[row, col].set_ylabel('km'); axes[row, col].tick_params(labelsize=8)
    fig.suptitle('Oracle plan inputs: 2,580.48 km per panel; stored local orientations\nElevation panels use fixed metre colours/NW 5x shading. Plan panels use categorical colours; inactive cells have zero plan.', fontsize=11)
    fig.savefig(out/'plan-inputs.png', dpi=125); plt.close(fig)
    atomic_json(out/'preflight.json', {'status': 'passed', 'indices': indices, 'plan_spec': PLAN_SPEC,
        'plan_finite': bool(torch.isfinite(plan).all()), 'plan_min': float(plan.min()), 'plan_max': float(plan.max()),
        'training_launched': False, 'planet_generation': False})


def run(args, out):
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = pick_device(args.device)
    if device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    source, expected = MODELS['structure']; source = ROOT/source
    if digest(source) != expected: raise ValueError('Frozen structure initialization changed')
    data = StructureData(args.data, args.seed); identity = manifest_id(data.meta)
    initial = torch.load(source, map_location='cpu', weights_only=False)
    stats, config = initial['stats'], initial['config']
    optimization = {'lr': args.lr, 'warmup': args.warmup, 'seed': args.seed, 'ema': .999, 'batch': args.batch}
    signature = {'version': VERSION, 'mode': args.mode, 'plan_spec': PLAN_SPEC, 'dataset_id': identity,
        'objective': OBJECTIVE, 'source_sha256': expected, 'config': config, 'optimization': optimization, 'stats': stats}
    if (out/'launch.json').exists():
        old_launch = json.loads((out/'launch.json').read_text())
        if any(old_launch.get(k) != v for k, v in signature.items()):
            raise ValueError('Run identity changed; use a fresh experiment folder')
    checkpoint = out/'latest.pt'
    resume = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if resume and any(resume.get(k) != v for k, v in signature.items()):
        raise ValueError('Resume mode/data/architecture/optimizer identity changed')
    model = build(config).to(device)
    initialize_from_source(model, initial['ema']); del initial
    norm = StructureNorm(stats); ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99))
    step, best = 0, float('inf')
    if resume:
        model.load_state_dict(resume['model']); ema.load_state_dict(resume['ema']); opt.load_state_dict(resume['opt'])
        step, best = resume['step'], resume['best_val_loss']; data.rng.bit_generator.state = resume['data_rng']
        torch.set_rng_state(resume['torch_rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    atomic_json(out/'launch.json', {**signature, 'arguments': vars(args), 'output': str(out), 'source_checkpoint': str(source),
        'fresh_initialization': 'Both modes use the same frozen structure EMA; four extra input weights are zero. Fresh optimizer and EMA schedule.',
        'training_step_is_additional': True, 'selection_changed': False, 'planet_generated': False})
    preflight(out, norm, data, device)
    print(f'{args.mode}: update {step}/{args.steps}; {sum(p.numel() for p in model.parameters()):,} parameters; batch {args.batch}\nOutput: {out}', flush=True)
    if args.check_only:
        print('Preflight passed; no optimizer updates or model sampling.', flush=True); return 0

    def save(name):
        ck = {**signature, 'step': step, 'best_val_loss': best, 'model': model.state_dict(), 'ema': ema.state_dict(),
            'opt': opt.state_dict(), 'data_rng': data.rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
        temp = out/(name+'.tmp'); torch.save(ck, temp); temp.replace(out/name)

    start = time.perf_counter(); last_time = start; last_step = step; losses = []
    with (out/'log.jsonl').open('a', encoding='utf-8') as log:
        def record(row): log.write(json.dumps(row)+'\n'); log.flush()
        try:
            while step < args.steps:
                raw, coarse, context = data.batch(args.batch, device=device)
                k = int(torch.randint(4, ()))
                raw, coarse, context = [torch.rot90(v, k, (-2, -1)) for v in (raw, coarse, context)]
                if torch.rand(()) < .5: raw, coarse, context = [v.flip(-1) for v in (raw, coarse, context)]
                coarse = degrade_guides(context, coarse, None, recipe=STRUCTURE_RECIPE)
                with autocast(device): value = loss(model, norm, raw, coarse, args.mode)
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite oracle-plan loss')
                for group in opt.param_groups: group['lr'] = args.lr*min(1., (step+1)/args.warmup)
                opt.zero_grad(set_to_none=True); value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step(); step += 1
                decay = min(.999, (1+step)/(10+step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1-decay)
                losses.append(float(value.detach()))
                if step % 25 == 0 or step == args.steps:
                    now = time.perf_counter(); rate = (step-last_step)/max(.001, now-last_time)
                    row = {'step': step, 'loss': float(np.mean(losses)), 'updates_per_second': rate, 'eta_minutes': (args.steps-step)/rate/60}
                    record(row); print(f'Update {step}/{args.steps} | loss {row["loss"]:.4f} | {rate:.2f} updates/s | ETA {row["eta_minutes"]:.1f} min', flush=True)
                    last_step, last_time, losses = step, now, []
                if step % 100 == 0: save('latest.pt')
                if step % 250 == 0 or step == args.steps:
                    val = evaluate(ema, norm, data, device, args.mode)
                    improved = val['smooth'] < best
                    if improved: best = val['smooth']; save('best.pt')
                    save('latest.pt'); record({'step': step, 'validation': val, 'best_loss_changed': improved})
                    print(f'Validation {val}; checkpoints saved. Loss is not morphology approval.', flush=True)
                    last_time, last_step, losses = time.perf_counter(), step, []
        except KeyboardInterrupt:
            save('latest.pt'); print(f'Stopped at {step}; resumable checkpoint saved.', flush=True); return 130
    save('latest.pt')
    atomic_json(out/'training-result.json', {'status': 'complete', 'mode': args.mode, 'additional_steps': step,
        'elapsed_seconds': time.perf_counter()-start, 'smoke_only': args.smoke, 'selection_changed': False, 'planet_generated': False})
    print('Finished. Regional plan review is a separate command; no model selected or planet started.', flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['control', 'oracle'], required=True)
    p.add_argument('--run'); p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--steps', type=int, default=2000); p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=.0002); p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--check-only', action='store_true'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if min(args.steps, args.batch, args.warmup) < 1 or args.lr <= 0 or not 0 <= args.seed < 2**63:
        p.error('Invalid training counts, learning rate or seed')
    out = Path(args.run).resolve() if args.run else OUT_ROOT/args.mode
    if args.smoke:
        args.steps = 2; args.batch = 2
        if not args.run: out = OUT_ROOT/('smoke-'+args.mode+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        if (out/'latest.pt').exists(): raise ValueError('Choose a fresh smoke directory')
    out.mkdir(parents=True, exist_ok=True)
    with console_log(OUT_ROOT/'job-logs', 'train-'+args.mode):
        with training_lock(out): return run(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
