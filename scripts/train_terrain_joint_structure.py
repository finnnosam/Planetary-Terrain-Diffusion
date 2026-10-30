"""Matched autonomous joint-structure/control pilots; no automatic sampling."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import StructureData, StructureNorm, DEFAULT_DATA, OBJECTIVE, decode
from planet_training.terrain_joint_structure import VERSION, JOINT_SPEC, build, initialize_from_control, targets, loss_parts
from scripts.probe_structure_stage_roles import ROOT, digest, array, render_fields
from scripts.structure_job import console_log
from toy_meso.common import pick_device, autocast

OUT_ROOT = ROOT/'outputs/structure/joint-structure-v1'
SOURCE = ROOT/'outputs/structure/plan-oracle-v1/review-20261013-220148-733529/control-checkpoint.pt'
SOURCE_SHA = '84127921814f560379bc805a7011d6d51ae439d37ffb8df6b1f6228485a3eb8a'


@torch.no_grad()
def evaluate(model, norm, data, device, mode):
    result = {}
    for label, smooth in [('clean', False), ('smooth', True)]:
        total = {k: 0. for k in ('total', 'height', 'plan', 'shape')}
        g = torch.Generator(device=device).manual_seed(2026)
        count = data.meta['val_count']
        for start in range(0, count, 16):
            raw, coarse, context = data.batch(0, 'val', np.arange(start, min(start+16, count)), device)
            if smooth:
                coarse = degrade_guides(context, coarse, g, sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
            for sigma in torch.exp(torch.linspace(-3, 1.5, 8, device=device)):
                with autocast(device):
                    parts = loss_parts(model, norm, raw, coarse, mode, g, sigma.expand(len(raw)))
                for key in total: total[key] += float(parts[key])*len(raw)/(8*count)
        result[label] = total
    return result


@torch.no_grad()
def preflight(out, norm, data, device, mode):
    indices = np.array([i for i in (16,36,6) if i < data.meta['val_count']] or [0])
    raw, coarse, ctx = data.batch(0, 'val', indices, device)
    coarse = degrade_guides(ctx, coarse, torch.Generator(device=device).manual_seed(2026),
                           sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    x, cin, target, base, _ = targets(norm, raw, coarse, mode)
    if not torch.isfinite(x).all() or cin.shape[1] != 2: raise ValueError('Invalid joint targets or conditions')
    np.savez_compressed(out/'preflight.npz', indices=indices, coarse_encoded=array(coarse),
                        target_metres=array(decode(target)), baseline_metres=array(base),
                        model_condition=array(cin), five_channel_target=array(x))
    fields = []
    for i,b,t in zip(indices,array(base),array(decode(target))):
        fields.extend([(f'Holdout {i}: coarse baseline',b[0]),(f'Holdout {i}: feasible target',t[0])])
    render_fields(out/'target-preflight.png', 'Autonomous intermediate: labels are outputs, never conditioning',
                  fields, 40320., '2,580.48 km; stored local frames. Same target/protection as the prior pilot.',columns=2)
    atomic_json(out/'preflight.json', {'status':'passed','mode':mode,'condition_channels':2,
        'generated_channels':5,'oracle_plan_conditioning':False,'indices':indices.tolist(),
        'sea_level_m':0.,'training_launched':False})


def run(args,out):
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=pick_device(args.device)
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if digest(SOURCE)!=SOURCE_SHA: raise ValueError('Frozen source control changed')
    initial=torch.load(SOURCE,map_location='cpu',weights_only=False)
    if initial['mode']!='control' or initial['objective']!=OBJECTIVE: raise ValueError('Unexpected source task')
    data=StructureData(args.data,args.seed)
    config,stats=initial['config'],initial['stats']
    optimization={'lr':args.lr,'warmup':args.warmup,'seed':args.seed,'ema':.999,'batch':args.batch,
                  'optimizer':'Adam','betas':[.9,.99],'clip_grad_norm':1.}
    signature={'version':VERSION,'mode':args.mode,'joint_spec':JOINT_SPEC,'objective':OBJECTIVE,
               'source_sha256':SOURCE_SHA,'dataset_id':manifest_id(data.meta),
               'config':config,'stats':stats,'optimization':optimization}
    if (out/'launch.json').exists():
        old=json.loads((out/'launch.json').read_text())
        if any(old.get(k)!=v for k,v in signature.items()): raise ValueError('Changed run identity; use a fresh --run')
    resume=torch.load(out/'latest.pt',map_location='cpu',weights_only=False) if (out/'latest.pt').exists() else None
    if resume and any(resume.get(k)!=v for k,v in signature.items()): raise ValueError('Resume identity changed')
    model=build(config).to(device); initialize_from_control(model,initial['ema']); del initial
    norm=StructureNorm(stats); ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.99))
    step,best=0,float('inf')
    if resume:
        model.load_state_dict(resume['model']);ema.load_state_dict(resume['ema']);opt.load_state_dict(resume['opt'])
        step,best=resume['step'],resume['best_val_loss'];data.rng.bit_generator.state=resume['data_rng']
        torch.set_rng_state(resume['torch_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    atomic_json(out/'launch.json',{**signature,'arguments':vars(args),'output':str(out),
        'source_checkpoint':str(SOURCE),'source_additional_updates':2000,'source_inherited_structure_updates':2000,
        'fresh_initialization':'Identical frozen height-only EMA; new input/output paths zero; fresh optimizer/EMA schedule.',
        'step_is_additional':True,'selection_changed':False,'planet_generated':False})
    preflight(out,norm,data,device,args.mode)
    print(f'{args.mode}: {sum(p.numel() for p in model.parameters()):,} parameters; update {step}/{args.steps}; batch {args.batch}\nOutput: {out}',flush=True)
    if args.check_only:print('Preflight passed. No optimizer updates or model sampling.',flush=True);return 0
    def save(name):
        ck={**signature,'step':step,'best_val_loss':best,'model':model.state_dict(),'ema':ema.state_dict(),
            'opt':opt.state_dict(),'data_rng':data.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
        temp=out/(name+'.tmp');torch.save(ck,temp);temp.replace(out/name)
    start=time.perf_counter();last=start;last_step=step;losses=[]
    atomic_json(out/'training-result.json',{'status':'running','mode':args.mode,'additional_steps':step,'ceiling':args.steps})
    with (out/'log.jsonl').open('a',encoding='utf-8') as log:
        def record(row):log.write(json.dumps(row)+'\n');log.flush()
        try:
            while step<args.steps:
                raw,coarse,ctx=data.batch(args.batch,device=device)
                k=int(torch.randint(4,()))
                raw,coarse,ctx=[torch.rot90(v,k,(-2,-1)) for v in (raw,coarse,ctx)]
                if torch.rand(())<.5:raw,coarse,ctx=[v.flip(-1) for v in (raw,coarse,ctx)]
                coarse=degrade_guides(ctx,coarse,None,recipe=STRUCTURE_RECIPE)
                with autocast(device):parts=loss_parts(model,norm,raw,coarse,args.mode)
                if not all(torch.isfinite(v) for v in parts.values()):raise FloatingPointError('Nonfinite joint loss')
                for group in opt.param_groups:group['lr']=args.lr*min(1.,(step+1)/args.warmup)
                opt.zero_grad(set_to_none=True);parts['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
                with torch.no_grad():
                    for pe,pm in zip(ema.parameters(),model.parameters()):pe.lerp_(pm,1-min(.999,(1+step)/(10+step)))
                losses.append({k:float(v.detach()) for k,v in parts.items()})
                if step%25==0 or step==args.steps:
                    now=time.perf_counter();rate=(step-last_step)/max(.001,now-last)
                    row={'step':step,'loss':{k:float(np.mean([v[k] for v in losses])) for k in parts},
                         'updates_per_second':rate,'eta_minutes':(args.steps-step)/rate/60}
                    record(row);print(f'Update {step}/{args.steps} | {row["loss"]} | {rate:.2f} updates/s | ETA {row["eta_minutes"]:.1f} min',flush=True)
                    last,last_step,losses=now,step,[]
                if step%100==0:save('latest.pt')
                if step%250==0 or step==args.steps:
                    val=evaluate(ema,norm,data,device,args.mode)
                    improved=val['smooth']['height']<best
                    if improved:best=val['smooth']['height'];save('best.pt')
                    save('latest.pt');record({'step':step,'validation':val,'best_height_loss_changed':improved})
                    print(f'Validation {val}; saved. Loss is not terrain-quality approval.',flush=True)
                    last,last_step,losses=time.perf_counter(),step,[]
        except KeyboardInterrupt:
            save('latest.pt');atomic_json(out/'training-result.json',{'status':'interrupted','mode':args.mode,'additional_steps':step})
            print(f'Stopped at {step}; resumable checkpoint saved.',flush=True);return 130
    save('latest.pt')
    atomic_json(out/'training-result.json',{'status':'complete','mode':args.mode,'additional_steps':step,
        'elapsed_seconds':time.perf_counter()-start,'smoke_only':args.smoke,'selection_changed':False,'planet_generated':False})
    print('Finished. Launch review separately; selected pipeline unchanged.',flush=True);return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['control','joint']);p.add_argument('--run')
    p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--warmup',type=int,default=200);p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if min(args.steps,args.batch,args.warmup)<1 or not 0<=args.seed<2**63 or args.lr<=0:p.error('Invalid counts, seed or learning rate')
    out=Path(args.run).resolve() if args.run else OUT_ROOT/args.mode
    if args.smoke:
        if not args.run:p.error('--smoke requires a separate --run directory')
        if (out/'latest.pt').exists():raise ValueError('Choose a fresh smoke directory')
        args.steps,args.batch=2,2
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','train-'+args.mode):
        with training_lock(out):return run(args,out)


if __name__=='__main__':raise SystemExit(main())
