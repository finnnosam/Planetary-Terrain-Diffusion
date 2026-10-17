"""Isolated, resumable training of a new regional 40-km structure model."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .adaptation import STRUCTURE_RECIPE, degrade_guides
from .data import atomic_json, manifest_id
from .run_lock import training_lock
from .terrain_structure import (VERSION, OBJECTIVE, DEFAULT_DATA, DEFAULT_RUN,
                                StructureData, StructureNorm, build, loss, target_for,
                                decode, sample)
from toy_meso.common import autocast, pick_device


@torch.no_grad()
def evaluate(model, norm, dataset, device, batch=16):
    """Complete geographic holdout, deterministic noise; clean and smooth guides."""
    result = {}
    for label, sigma in [('clean', None), ('smooth', 1.), ('strong_smooth', 1.5)]:
        total = 0.; n = dataset.meta['val_count']
        g = torch.Generator(device=device).manual_seed(2026)
        for start in range(0, n, batch):
            raw, c, ctx = dataset.batch(0, 'val', np.arange(start, min(start+batch, n)), device)
            if sigma is not None: c = degrade_guides(ctx, c, torch.Generator(device=device).manual_seed(2026), sigma_cells=sigma, clean_fraction=0., recipe=STRUCTURE_RECIPE)
            target, _ = target_for(raw, c)
            _, up, cin = norm.condition(c); x = norm.residual(target, up)
            for s in torch.exp(torch.linspace(-3, 1.5, 8, device=device)):
                noisy = x+torch.randn(x.shape, generator=g, device=device)*s
                with autocast(device): p = model(noisy, s.expand(len(x)), cin)
                w = (s*s+.25)/(s*.5)**2
                total += float(w*(p.float()-x).square().mean())*len(x)/(8*n)
        result[label] = total
    return result


@torch.no_grad()
def preview(path, model, norm, dataset, device, steps):
    from scripts.diagnose_terrain_structure import image, legend
    import matplotlib.pyplot as plt
    raw, c, ctx = dataset.batch(0, 'val', np.arange(min(4, dataset.meta['val_count'])), device)
    c = degrade_guides(ctx, c, torch.Generator(device=device).manual_seed(7), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    target, base = target_for(raw, c); generated, _, _ = sample(model, norm, c, steps, seed=7, device=device)
    rows = [('Input guide', base), ('Original modern averages', raw), ('Constrained training target', decode(target)), ('New structure', generated)]
    fig, axs = plt.subplots(4, len(raw), figsize=(3*len(raw), 11), squeeze=False, constrained_layout=True)
    for row, (label, fields) in enumerate(rows):
        for col in range(len(raw)):
            z = fields[col, 0].float().cpu().numpy()
            axs[row, col].imshow(image(z, 40320.), extent=(-1290.24, 1290.24, -1290.24, 1290.24))
            axs[row, col].set_title(label, fontsize=9); axs[row, col].set_xticks([-1000, 0, 1000]); axs[row, col].set_yticks([-1000, 0, 1000])
    fig.suptitle('40.32-km stage / entire 2580-km patches / seed 7\nTargets protect shallow input terrain and preserve interpolated-baseline footprint means. Same colours and NW 5x shading.')
    legend(fig, axs); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=DEFAULT_DATA); p.add_argument('--run', default=DEFAULT_RUN)
    p.add_argument('--steps', type=int, default=2000); p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cuda'); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--ch', type=int, default=32); p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--lr', type=float, default=.0002); p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=100); p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--preview-steps', type=int, default=16)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    if any(getattr(a, k) < 1 for k in ['steps', 'batch', 'threads', 'ch', 'blocks', 'warmup', 'eval_every', 'save_every', 'log_every']) or a.preview_steps < 2 or a.lr <= 0:
        p.error('Training counts/rate must be positive; preview needs at least two steps')
    if a.ch % 8 or a.seed < 0 or a.seed >= 2**63: p.error('Channels must divide by eight; seed must be in 0..2^63-1')
    if a.smoke:
        if a.data == DEFAULT_DATA: a.data = 'outputs/structure/checks/data-smoke'
        if a.run == DEFAULT_RUN: a.run = 'outputs/structure/checks/train-smoke'
        a.steps, a.eval_every, a.save_every, a.log_every, a.preview_steps = 2, 2, 1, 1, 2
        if (Path(a.run)/'latest.pt').exists(): raise ValueError('Choose a fresh smoke run directory')
    with training_lock(a.run): return train(a)


def train(a):
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = pick_device(a.device)
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
    data = StructureData(a.data, a.seed); run = Path(a.run); identity = manifest_id(data.meta)
    config = {'ch': a.ch, 'mults': [1, 2, 4], 'blocks': a.blocks, 'dropout': 0.}
    optimization = {'lr': a.lr, 'warmup': a.warmup, 'seed': a.seed, 'ema': .999}
    path = run/'latest.pt'; ck = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    if ck and (ck.get('version') != VERSION or ck.get('dataset_id') != identity or ck.get('objective') != OBJECTIVE or ck.get('config') != config or ck.get('optimization') != optimization):
        raise ValueError('Resume identity/objective/config changed; use the original arguments or a fresh run')
    stats = ck['stats'] if ck else data.stats()
    if not all(np.isfinite(v) for v in stats.values()) or min(stats['cond_std'], stats['res_std']) <= 0: raise ValueError('Invalid normalization')
    model = build(config).to(device); norm = StructureNorm(stats)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(.9, .99))
    step, best = 0, float('inf')
    if ck:
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema']); opt.load_state_dict(ck['opt'])
        step, best = ck['step'], ck['best_val_loss']; data.rng.bit_generator.state = ck['data_rng']
        torch.set_rng_state(ck['torch_rng'])
        if str(device).startswith('cuda') and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    atomic_json(run/'launch.json', dict(vars(a), dataset_id=identity, config=config, stats=stats,
                                      objective=OBJECTIVE, initialization='new model; no selected weights changed', full_planet_generation=False))
    print(f'\n40-KM TERRAIN STRUCTURE PILOT\nUpdate {step:,} / ceiling {a.steps:,}; batch {a.batch}; {sum(p.numel() for p in model.parameters()):,} parameters.\nOutput: {run}\nTraining only; regional review is a separate command.', flush=True)
    if a.check_only:
        preview(run/'target-preflight.png', ema, norm, data, device, 2)
        print('Data/model preflight passed; no optimizer updates.', flush=True); return 0
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()

    def checkpoint(name):
        obj = {'version': VERSION, 'stage': 'terrain-structure-40', 'objective': OBJECTIVE,
               'dataset_id': identity, 'meta': data.meta, 'config': config, 'stats': stats,
               'optimization': optimization, 'step': step, 'best_val_loss': best,
               'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
               'data_rng': data.rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
        tmp = run/(name+'.tmp'); torch.save(obj, tmp); tmp.replace(run/name)

    started = time.perf_counter(); session_start = started; last = step; losses = []
    with (run/'log.jsonl').open('a', encoding='utf-8') as f:
        def record(obj):
            f.write(json.dumps(obj)+'\n'); f.flush()
        try:
            while step < a.steps:
                raw, c, ctx = data.batch(a.batch, device=device)
                k = int(torch.randint(4, ()))
                raw, c, ctx = [torch.rot90(v, k, (-2, -1)) for v in (raw, c, ctx)]
                if torch.rand(()) < .5: raw, c, ctx = [v.flip(-1) for v in (raw, c, ctx)]
                c = degrade_guides(ctx, c, None, recipe=STRUCTURE_RECIPE)
                with autocast(device): value = loss(model, norm, raw, c)
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite structure loss; prior checkpoints preserved')
                for group in opt.param_groups: group['lr'] = a.lr*min(1., (step+1)/a.warmup)
                opt.zero_grad(set_to_none=True); value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step(); step += 1
                decay = min(.999, (1+step)/(10+step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1-decay)
                losses.append(value.item())
                if step % a.log_every == 0 or step == a.steps:
                    elapsed = time.perf_counter()-started; rate = (step-last)/max(elapsed,.001)
                    row = {'step': step, 'loss': float(np.mean(losses)), 'steps_per_s': rate, 'remaining_minutes': (a.steps-step)/rate/60}
                    record(row); print(f"Update {step:,}/{a.steps:,} | loss {row['loss']:.4f} | {rate:.2f} updates/s | ETA {row['remaining_minutes']:.1f} min", flush=True)
                    started, last, losses = time.perf_counter(), step, []
                if step % a.save_every == 0: checkpoint('latest.pt')
                if step % a.eval_every == 0 or step == a.steps:
                    print('Checking the complete held-out set...', flush=True)
                    vals = evaluate(ema, norm, data, device, a.batch); improved = vals['smooth'] < best
                    if improved: best = vals['smooth']; checkpoint('best.pt')
                    checkpoint('latest.pt'); record({'step': step, 'validation': vals, 'new_best': improved})
                    print(f'Validation: {vals}. Checkpoints saved; rendering bounded preview.', flush=True)
                    preview(run/f'preview-{step:06d}.png', ema, norm, data, device, a.preview_steps)
                    if str(device).startswith('cuda'): record({'step': step, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
                    started, last, losses = time.perf_counter(), step, []
        except KeyboardInterrupt:
            checkpoint('latest.pt'); print(f'Stopped at update {step}; latest checkpoint saved. Rerun to resume.', flush=True); return 130
    checkpoint('latest.pt')
    atomic_json(run/'training-result.json', {'status': 'complete', 'step': step, 'seconds_this_session': time.perf_counter()-session_start,
                                           'smoke_test': a.smoke, 'full_planet_generation': False, 'selection_changed': False,
                                           'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if str(device).startswith('cuda') else None})
    print('Smoke training passed; no terrain-quality claim.' if a.smoke else 'Pilot finished. Run the separate regional review; nothing else starts automatically.', flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
