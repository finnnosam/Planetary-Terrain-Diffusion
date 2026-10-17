"""Train the coastline or fixed-datum relief model; old runs remain independent."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json,manifest_id
from .progress import TrainingConsole
from .run_lock import training_lock,RunAlreadyActive
from .spherical_net import CoastReliefDenoiser
from .structured_data import (StructuredData,STRUCTURED_VERSION,decode_relief,OBJECTIVE,
                              DEFAULT_COAST_RUN,DEFAULT_RELIEF_RUN)
from toy_meso.common import autocast,pick_device
from toy_meso.model import heun_sample


def targets(model,coast,magnitude):
    return coast-.5 if model.stage=='coast' else magnitude


def spatial_weights(model,coast,magnitude,stats):
    if model.stage=='coast':
        # The clean label is unknown to this conditional denoiser. Weighting
        # land and water separately changes its posterior mean and biases the
        # sampling score. Spherical area belongs in grid.mean, not class odds.
        return torch.ones_like(coast)
    # In relief, the coastline is a KNOWN condition. Balancing its two domains
    # therefore does not reweight unknown values within either conditional
    # posterior. Depth-dependent shallow/deep weights would, so remove those.
    partitions = [coast,1-coast]
    populations = [model.grid.mean(p) for p in partitions]
    count = sum((v>1e-6).float() for v in populations).clamp_min(1)
    return sum(p/v.clamp_min(1e-6)[:, :,None,None,None] for p,v in zip(partitions,populations))/count[:,:,None,None,None]


def loss_for(model,coast,magnitude,stats,generator=None,sigma=None):
    x = targets(model,coast,magnitude)
    if sigma is None:
        sigma = (torch.randn(len(x),device=x.device,generator=generator)*1.2-1.2).exp()
    noise = model.grid.noise(len(x),1,x.device,generator)
    coverage = model.grid.mean(coast)[:,0]
    with autocast(x.device):
        d = model(x+noise*sigma[:,None,None,None,None],sigma,coverage,coast)
    edm_weight = (sigma.square()+.25)/(sigma*.5).square()
    error = (d.float()-x).square()*spatial_weights(model,coast,magnitude,stats)
    return (model.grid.mean(error)[:,0]*edm_weight).mean()


@torch.no_grad()
def evaluate(model,coast,magnitude,stats,batch=16):
    generator = torch.Generator(device=coast.device).manual_seed(2026)
    total = 0.
    for s in torch.exp(torch.linspace(-3,1.5,8,device=coast.device)):
        for start in range(0,len(coast),batch):
            c,m = coast[start:start+batch],magnitude[start:start+batch]
            value = loss_for(model,c,m,stats,generator,s.expand(len(c)))
            total += float(value)*len(c)/len(coast)/8
    return total


@torch.no_grad()
def sample(model,stats,coverage,steps=32,seed=42,coast=None):
    device = next(model.parameters()).device
    coverage = torch.as_tensor(coverage,device=device,dtype=torch.float32).reshape(-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = model.grid.noise(len(coverage),1,device,generator)
    def denoise(x,s):
        with autocast(device):
            return model(x.float(),s.float().expand(len(x)),coverage,coast).double()
    x = heun_sample(denoise,initial.shape,steps,device=device,initial=initial)
    x = model.grid.project(x.float())
    if model.stage=='coast': return (x+.5).clamp(0,1)
    return x/.5*stats['std']+stats['mean']


def save_preview(path,model,stats,coast,steps,seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .geography import export_pixels
    coverage = model.grid.mean(coast[:2])[:,0]
    measurements = None
    if model.stage=='coast':
        generated = sample(model,stats,coverage,steps,seed)
        field = generated[:,0].cpu().numpy()>.5
        real = coast[:2,0].cpu().numpy()
        achieved = model.grid.mean((generated>.5).float())[:,0]
        measurements = dict(requested_land_percent=(coverage*100).cpu().tolist(),
                            actual_land_percent=(achieved*100).cpu().tolist(),
                            mean_absolute_error_pp=float((achieved-coverage).abs().mean()*100),
                            sampling_steps=steps,seed=seed)
    else:
        # These are held-out true masks. A sampled-mask end-to-end check comes
        # later, after the coastline model itself passes the preview gate.
        mag = sample(model,stats,coverage,steps,seed,coast[:2])[:,0].cpu().numpy()
        field = decode_relief(coast[:2,0].cpu().numpy(),mag)
        real = None
    fig,axes = plt.subplots(1,len(field),figsize=(6*len(field),4),constrained_layout=True,squeeze=False)
    axes = axes.ravel()
    for i,ax in enumerate(axes):
        image = export_pixels(field[i].astype(np.float32),128)
        from matplotlib.colors import ListedColormap
        ax.imshow(image,cmap=ListedColormap(['#16465e','#a6af70']) if model.stage=='coast' else 'terrain',
                  vmin=0 if model.stage=='coast' else -6000,vmax=1 if model.stage=='coast' else 6000)
        if real is not None:
            actual = measurements['actual_land_percent'][i]
            ax.set_title(f'Land: requested {float(coverage[i])*100:.1f}% / actual {actual:.1f}%\nGreen land / blue ocean')
        else: ax.set_title('Relief generated around a held-out true coastline')
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('New architecture preview — quality must be checked before downstream generation')
    fig.savefig(path,dpi=120);plt.close(fig)
    if measurements is not None: atomic_json(Path(path).with_suffix('.json'),measurements)
    return measurements


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('coast','relief'))
    p.add_argument('--data',default='outputs/planet_training/data/layout')
    p.add_argument('--run')
    p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--ch',type=int,default=24)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--ema',type=float,default=.999)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--preview-steps',type=int,default=24)
    p.add_argument('--validation-patience',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--init-from',help='warm start a fresh run from EMA; reset optimizer and validation baseline')
    a = p.parse_args(argv)
    if any(getattr(a,k)<1 for k in ('steps','batch','ch','warmup','eval_every','save_every','log_every','threads')):
        p.error('Update counts, channels and batch size must be positive')
    if a.ch%4 or a.ch<8 or a.preview_steps<2 or a.validation_patience<0 or not np.isfinite(a.lr) or a.lr<=0 or not 0<=a.dropout<1 or not 0<a.ema<1:
        p.error('Invalid channels, preview steps or training settings')
    a.run = a.run or (DEFAULT_COAST_RUN if a.stage=='coast' else DEFAULT_RELIEF_RUN)
    return a


def main(argv=None):
    a = parse_args(argv)
    with training_lock(a.run):
        return train(a)


def train(a):
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    device = pick_device(a.device)
    data = StructuredData(a.data,a.seed)
    identity = manifest_id(dict(source=data.meta,representation=STRUCTURED_VERSION))
    run = Path(a.run);run.mkdir(parents=True,exist_ok=True)
    checkpoint_path = run/'latest.pt'
    ck = torch.load(checkpoint_path,map_location='cpu',weights_only=False) if checkpoint_path.exists() else None
    if ck and (ck.get('version')!=STRUCTURED_VERSION or ck.get('stage')!=a.stage or ck.get('dataset_id')!=identity):
        raise ValueError('Different architecture/stage/dataset; choose a fresh run folder')
    if ck and ck.get('objective')!=OBJECTIVE:
        raise ValueError('Old biased objective cannot be resumed; use a fresh run with --init-from')
    initial = None
    if not ck and a.init_from:
        initial = torch.load(a.init_from,map_location='cpu',weights_only=False)
        if (initial.get('version')!=STRUCTURED_VERSION or initial.get('stage')!=a.stage
                or initial.get('dataset_id')!=identity or initial['config'].get('n')!=data.n):
            raise ValueError('Warm-start checkpoint must match the stage, dataset and native scale')
    reference = ck or initial
    config = reference['config'] if reference else dict(n=data.n,ch=a.ch,dropout=a.dropout,coverage_prior=a.stage=='coast')
    stats = reference['stats'] if reference else data.stats()
    if ck and any(ck['training'][key]!=getattr(a,key) for key in ('lr','warmup','ema','seed')):
        raise ValueError('Resume with the original lr, warmup, EMA and seed')
    model = CoastReliefDenoiser(a.stage,**config).to(device)
    if ck: model.load_state_dict(ck['model'])
    elif initial: model.load_state_dict(initial['ema'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if ck: ema.load_state_dict(ck['ema'])
    opt = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    if ck:
        opt.load_state_dict(ck['opt']);data.rng.bit_generator.state=ck['data_rng']
        torch.set_rng_state(ck['torch_rng'])
        if str(device).startswith('cuda') and ck.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    coast,magnitude = data.validation(stats,device)
    step,best,stale = (ck['step'],ck['best_val_loss'],ck['stale']) if ck else (0,float('inf'),0)
    initialized_from = ck.get('initialized_from') if ck else (
        dict(path=str(Path(a.init_from).resolve()),step=initial['step'],objective=initial.get('objective','legacy-balanced-v1')) if initial else None)
    del initial
    console = TrainingConsole(a.steps)
    console.start('layout',step,a.batch,device,sum(p.numel() for p in model.parameters()),run)
    console.message(f'New {a.stage} stage: neighbour-face convolutions; sea level fixed at zero before relief.')
    atomic_json(run/'launch.json',dict(vars(a),config=config,stats=stats,dataset_id=identity,version=STRUCTURED_VERSION,
                                     objective=OBJECTIVE,initialized_from=initialized_from))
    if a.check_only:
        values = targets(model,coast,magnitude)
        console.message(f'Data valid: {len(data.source.splits["train"])} training ages, {len(coast)} validation ages, shape {tuple(values.shape)}')
        return 0

    def save(name):
        obj = dict(version=STRUCTURED_VERSION,stage=a.stage,dataset_id=identity,meta=data.meta,
                   objective=OBJECTIVE,initialized_from=initialized_from,
                   config=config,stats=stats,step=step,best_val_loss=best,stale=stale,
                   training={k:getattr(a,k) for k in ('lr','warmup','ema','seed')},
                   model=model.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),
                   data_rng=data.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None)
        tmp = run/(name+'.tmp');torch.save(obj,tmp);tmp.replace(run/name)

    started,last,losses = time.perf_counter(),step,[]
    with (run/'log.jsonl').open('a',encoding='utf-8') as log:
        def record(obj):
            log.write(json.dumps(obj)+'\n');log.flush();console.record(obj)
        try:
            if step==0 and initialized_from:
                console.message('Checking the warm-start model under the corrected objective...')
                best = evaluate(ema,coast,magnitude,stats,a.batch)
                record(dict(step=0,val_loss=best,new_best=True))
                save('best.pt');save('latest.pt')
                coverage_check = save_preview(run/'preview-000000.png',ema,stats,coast,a.preview_steps)
                if coverage_check: record(dict(step=0,coverage_preview=coverage_check))
                started,last,losses=time.perf_counter(),step,[]
            while step<a.steps:
                c,m = data.batch(a.batch,stats,device)
                loss = loss_for(model,c,m,stats)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss; previous checkpoint retained')
                for group in opt.param_groups: group['lr']=a.lr*min(1,(step+1)/a.warmup)
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
                step += 1
                with torch.no_grad():
                    for pe,pm in zip(ema.parameters(),model.parameters()): pe.lerp_(pm,1-min(a.ema,(1+step)/(10+step)))
                losses.append(float(loss.detach()))
                if step%a.log_every==0:
                    rate = (step-last)/(time.perf_counter()-started)
                    record(dict(step=step,loss=float(np.mean(losses)),batch=a.batch,steps_per_s=rate,
                                examples_per_s=rate*a.batch,remaining_hours=(a.steps-step)/rate/3600))
                    started,last,losses = time.perf_counter(),step,[]
                if step%a.save_every==0: save('latest.pt')
                if step%a.eval_every==0 or step==a.steps:
                    console.message('Checking all held-out ages...')
                    value = evaluate(ema,coast,magnitude,stats,a.batch)
                    improved = value<best
                    if improved: best,stale = value,0
                    else: stale += 1
                    record(dict(step=step,val_loss=value,new_best=improved))
                    if improved: save('best.pt')
                    save('latest.pt')
                    coverage_check = save_preview(run/f'preview-{step:06d}.png',ema,stats,coast,a.preview_steps)
                    if coverage_check: record(dict(step=step,coverage_preview=coverage_check))
                    if str(device).startswith('cuda'): record(dict(step=step,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
                    started,last,losses = time.perf_counter(),step,[]
                    if a.validation_patience and stale>=a.validation_patience:
                        console.message('Stopped after repeated validation checks without improvement.');break
        except KeyboardInterrupt:
            save('latest.pt');console.message(f'Stopped at update {step:,}; checkpoint saved.');return 130
    save('latest.pt');console.complete(step,run)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except RunAlreadyActive as error:
        print(str(error),flush=True);raise SystemExit(1)
