"""Fine-tune the meso to make Earth-like detail from softer 23 km input, with an optional detail input.

Why: the meso adds detail as a residual on its upsampled 23 km input and takes its cue for how much from the
input's cell-to-cell curvature. Simulated fields carry about half of Earth's; on Earth input blurred the same way
(0.7 cells) the original meso keeps only ~0.3 of real river incision (diagnosis 20261015-213919).

What changes (starting from the original Earth-trained run2-lowlr weights; nothing else is altered):
  * softened inputs: with probability --soften-prob the 23 km heights are Gaussian-blurred (sigma drawn from
    --soften-sigma, in 23 km cells); the target stays the real 2.88 km terrain.
  * detail input: two new input channels, the per-cell sub-cell detail (toy_meso.detail) and a known mask. They are
    hidden with probability --detail-drop, so the model also works when no detail is supplied. New input weights
    start at zero, so step 0 behaves exactly like run2-lowlr.

  python -m toy_meso.finetune_detail          # defaults: outputs/meso/dataset.npz -> outputs/meso/detail-ft

Rerun the same command to resume. Adopt the result only if scripts.diagnose_meso_plains --blur-earth 0.7 passes
the rule fixed in docs/tectonic-prototype.md.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_dataset, Normalizer, build_model, autocast, pick_device, sample_patches
from .detail import cell_detail, soften, detail_inputs
from .model import edm_loss

ROOT = Path(__file__).resolve().parents[1]


def augment(target, cond, detail):
    k = int(torch.randint(4, ()))
    target, cond, detail = (torch.rot90(x, k, (-2, -1)) for x in (target, cond, detail))
    if torch.rand(()) < 0.5:
        target, cond, detail = target.flip(-1), cond.flip(-1), detail.flip(-1)
    return target, cond, detail


def expand_input(state, new_in):
    """Old checkpoint (height only) -> state dict with extra zero-initialized input channels."""
    state = dict(state)
    w = state['net.inp.weight']
    pad = torch.zeros(w.shape[0], new_in-w.shape[1], *w.shape[2:], dtype=w.dtype)
    state['net.inp.weight'] = torch.cat([w, pad], 1)
    return state


@torch.no_grad()
def detail_all(target, cond, device, chunk=1024):
    out = []
    for i in range(0, len(target), chunk):
        t = torch.from_numpy(np.asarray(target[i:i+chunk], np.float32)).to(device)[:, None]
        c = torch.from_numpy(np.asarray(cond[i:i+chunk], np.float32)).to(device)[:, None]
        out.append(cell_detail(t, c).half().cpu())
    return torch.cat(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=str(ROOT/'outputs/meso/dataset.npz'))
    p.add_argument('--init', default=str(ROOT/'outputs/meso/run2-lowlr/latest.pt'))
    p.add_argument('--run', default=str(ROOT/'outputs/meso/detail-ft'))
    p.add_argument('--steps', type=int, default=24000, help='fine-tuning updates (~3.5 per second at batch 32); validation\n'
                   'loss bottomed out at ~10-24k of 48k batch-16 steps in the first run, so best.pt is also kept')
    p.add_argument('--batch', type=int, default=32, help='use 16 (with --steps 48000) on an 8 GB GPU')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--soften-prob', type=float, default=0.6)
    p.add_argument('--soften-sigma', default='0.3,1.1', help='range of blur sigmas, 23 km cells')
    p.add_argument('--detail-drop', type=float, default=0.4, help='probability the detail input is hidden')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--device', default='auto')
    p.add_argument('--data-on-cpu', action='store_true', help='keep training patches in system RAM (saves VRAM)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    lo, hi = (float(x) for x in a.soften_sigma.split(','))
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    (run/'args.json').write_text(json.dumps(vars(a), indent=1))

    d, meta = load_dataset(a.data)
    print('loading patches...', flush=True)
    t_np, c_np = d['train_target'], d['train_cond']
    latest = run/'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        config, stats, step = ck['config'], ck['stats'], ck['step']
        print(f'resuming from fine-tune step {step}', flush=True)
    else:
        base = torch.load(a.init, map_location='cpu', weights_only=False)
        config = dict(base['config'], cond_channels=3, inputs=['height', 'detail', 'detail_known'])
        stats, step = dict(base['stats']), 0
        print(f"starting from {a.init} (step {base['step']})", flush=True)
    data_device = 'cpu' if a.data_on_cpu else device
    target = torch.from_numpy(t_np).to(data_device)[:, None]
    cond = torch.from_numpy(c_np).to(data_device)[:, None]
    detail = detail_all(t_np, c_np, 'cpu').to(data_device)   # on the CPU: keeps the GPU free for training
    if 'detail_mean' not in stats:
        stats['detail_mean'] = float(detail.float().mean()); stats['detail_std'] = float(detail.float().std())
    norm = Normalizer(stats, meta['factor'])
    sel = np.arange(0, len(d['val_cond']), max(1, len(d['val_cond'])//128))[:128]   # only the validation patches used
    vt = torch.from_numpy(d['val_target'][sel].astype(np.float32)).to(device)[:, None]
    vc = torch.from_numpy(d['val_cond'][sel]).to(device)[:, None]
    vsel = torch.arange(len(sel), device=device)
    vd = cell_detail(vt, vc)
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    print(f'device {device}; {len(target)} train patches; detail mean {stats["detail_mean"]:.3f} std {stats["detail_std"]:.3f} (encoded)', flush=True)

    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    if step:
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema']); opt.load_state_dict(ck['opt'])
    else:
        w = expand_input(base['ema'], 1+config['cond_channels'])
        model.load_state_dict(w); ema.load_state_dict(w)
    log = open(run/'log.jsonl', 'a')

    def checkpoint(name):
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'config': config, 'stats': stats, 'meta': meta, 'init': a.init, 'finetune': vars(a)}, run/(name+'.tmp'))
        (run/(name+'.tmp')).replace(run/name)

    @torch.no_grad()
    def val_loss(soft, known):
        torch.manual_seed(1234); total = 0.
        for i in range(0, len(vsel), a.batch):
            t, c, dd = vt[vsel[i:i+a.batch]], vc[vsel[i:i+a.batch]], vd[i:i+a.batch]
            cs = soften(c, soft) if soft else c
            up, cin = norm.condition(cs)
            cin = torch.cat([cin, detail_inputs(norm, dd if known else None, cs.shape, device)], 1)
            with autocast(device):
                total += float(edm_loss(ema, norm.residual(t, up), cin))*len(t)
        return total/len(vsel)

    started, running = time.perf_counter(), []
    while step < a.steps:
        idx = torch.randint(len(target), (a.batch,), device=data_device)
        t, c, dd = augment(target[idx].to(device).float(), cond[idx].to(device), detail[idx].to(device).float())
        sig = torch.empty(a.batch, device=device).uniform_(lo, hi)
        soft = torch.rand(a.batch, device=device) < a.soften_prob
        if soft.any():
            c = c.clone(); c[soft] = soften(c[soft], sig[soft])
        known = (torch.rand(a.batch, device=device) >= a.detail_drop).float()
        up, cin = norm.condition(c)
        cin = torch.cat([cin, detail_inputs(norm, dd, c.shape, device, known)], 1)
        x = norm.residual(t, up)
        for g in opt.param_groups:
            g['lr'] = a.lr*min(1., (step+1)/a.warmup)
        with autocast(device):
            loss = edm_loss(model, x, cin)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); step += 1
        decay = min(a.ema, (1+step)/(10+step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1-decay)
        running.append(loss.item())
        if step % a.log_every == 0:
            rec = {'step': step, 'loss': float(np.mean(running)), 'steps_per_s': a.log_every/(time.perf_counter()-started),
                   'eta_min': (a.steps-step)/max(a.log_every/(time.perf_counter()-started), 1e-6)/60}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
            running, started = [], time.perf_counter()
        if step % a.save_every == 0:
            checkpoint('latest.pt')
        if step % a.eval_every == 0 or step == a.steps:
            rec = {'step': step, 'val_loss_plain': val_loss(0, False), 'val_loss_soft0.7': val_loss(.7, False),
                   'val_loss_soft0.7_detail': val_loss(.7, True)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
            checkpoint('latest.pt')
            score = (rec['val_loss_plain']+rec['val_loss_soft0.7']+rec['val_loss_soft0.7_detail'])/3
            best_file = run/'best.json'
            best = json.loads(best_file.read_text())['score'] if best_file.exists() else float('inf')
            if score < best:   # held-out validation stops improving once the fine-tune starts to overfit
                checkpoint('best.pt'); best_file.write_text(json.dumps({'step': step, 'score': score}))
                print(f'best so far (mean validation loss {score:.4f}): {run/"best.pt"}', flush=True)
            started = time.perf_counter()
    checkpoint('latest.pt')
    print(f'done: {run/"latest.pt"}\nNext: python -m scripts.diagnose_meso_plains --meso {run/"latest.pt"} --compare outputs/meso/run2-lowlr/latest.pt --blur-earth 0.7', flush=True)


if __name__ == '__main__':
    main()
