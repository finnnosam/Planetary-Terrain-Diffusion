"""Paired categorical/continuous renderers or planners, with durable bounded runs."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from planet_training.adaptation import degrade_guides,STRUCTURE_RECIPE
from planet_training.data import atomic_json,manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import StructureData,StructureNorm,DEFAULT_DATA,OBJECTIVE,target_for,decode
from planet_training.terrain_plan_oracle import plan_for
from planet_training.terrain_spatial_plan import VERSION,SPEC,MODES,PlanCodec,continuous_fields,compress,build,initialize,loss_parts
from scripts.probe_joint_plan_influence import digest
from scripts.structure_job import console_log
from toy_meso.common import autocast

ROOT=Path(__file__).resolve().parents[1]
OUT_ROOT=ROOT/'outputs/structure/spatial-supervision-v1'
SOURCE=ROOT/'outputs/structure/plan-oracle-v1/review-20261013-220148-733529/control-checkpoint.pt'
SOURCE_SHA='84127921814f560379bc805a7011d6d51ae439d37ffb8df6b1f6228485a3eb8a'


@torch.no_grad()
def fit_stats(data):
    indices=np.random.default_rng(1099).choice(data.meta['train_count'],min(256,data.meta['train_count']),replace=False)
    accum={m:[] for m in MODES};g=torch.Generator(device='cuda').manual_seed(1099)
    for start in range(0,len(indices),16):
        raw,c,ctx=data.batch(0,'train',indices[start:start+16],'cuda')
        c=degrade_guides(ctx,c,g,recipe=STRUCTURE_RECIPE);target,base=target_for(raw,c);z=decode(target)
        for mode in MODES:
            p=plan_for(z,base) if mode=='categorical' else continuous_fields(z,base)
            values=compress(p,mode).permute(0,2,3,1)[(base[:,0]>200)]
            accum[mode].append(values.cpu())
    stats={}
    for mode in MODES:
        v=torch.cat(accum[mode]);mean=v.mean(0);std=v.std(0,unbiased=False).clamp_min(.05)
        stats[mode]={'mean':mean.tolist(),'std':std.tolist(),'active_cells':len(v),
            'compressed_p01_p50_p99':torch.quantile(v,torch.tensor([.01,.5,.99]),dim=0).tolist(),
            'compressed_correlation':torch.corrcoef(v.T).nan_to_num().tolist()}
    return stats,{'split':'train','seed':1099,'indices':indices.tolist(),'degradation_recipe':STRUCTURE_RECIPE}


def batch_fields(norm,codec,raw,c):
    target,base=target_for(raw,c);_,up,cin=norm.condition(c)
    return target,base,up,cin,{m:codec.encode(decode(target),base,m) for m in MODES}


@torch.no_grad()
def evaluate(models,task,norm,codec,data):
    values={m:{k:0. for k in ['total','main','shape']} for m in MODES};count=data.meta['val_count']
    g=torch.Generator(device='cuda').manual_seed(2026)
    for start in range(0,count,16):
        raw,c,ctx=data.batch(0,'val',np.arange(start,min(count,start+16)),'cuda')
        c=degrade_guides(ctx,c,g,sigma_cells=1.,clean_fraction=0.,recipe=STRUCTURE_RECIPE)
        target,base,up,cin,plans=batch_fields(norm,codec,raw,c)
        for s in [.05,.2,.5,1.,2.,5.]:
            sigma=torch.full((len(raw),),s,device='cuda');noise=torch.randn((len(raw),5,64,64),generator=g,device='cuda')
            for m in MODES:
                with autocast('cuda'):v=loss_parts(models[m],task,norm,target,base,up,cin,plans[m],sigma,noise)
                for k in v:values[m][k]+=float(v[k])*len(raw)/(6*count)
    return values


def train(args,out):
    started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise ValueError('Bounded paired training requires CUDA')
    if digest(SOURCE)!=SOURCE_SHA:raise ValueError('Frozen height-only source changed')
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    if source['config']['dropout']!=0:raise ValueError('Matched-noise protocol assumes zero dropout')
    data=StructureData(DEFAULT_DATA,args.seed);norm=StructureNorm(source['stats'])
    optimization={'batch':args.batch,'seed':args.seed,'lr':.0002,'warmup':200,'betas':[.9,.99],'clip':1.,'ema':.999}
    signature={'version':VERSION,'spec':SPEC,'task':args.task,'objective':OBJECTIVE,'dataset_id':manifest_id(data.meta),
        'source_sha256':SOURCE_SHA,'config':source['config'],'stats':source['stats'],'optimization':optimization}
    launch=out/'launch.json';checkpoint=out/'latest.pt'
    if launch.exists():
        old=json.loads(launch.read_text())
        if any(old.get(k)!=v for k,v in signature.items()):raise ValueError('Changed run identity; choose a fresh folder')
        if not checkpoint.exists():raise ValueError('Incomplete run has no paired checkpoint; choose a fresh folder')
    resume=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() else None
    if resume and any(resume.get(k)!=v for k,v in signature.items()):raise ValueError('Resume identity mismatch')
    if resume:stats,normalization=resume['plan_stats'],resume['normalization_source']
    else:stats,normalization=fit_stats(data)
    codec=PlanCodec(stats);signature.update(plan_stats=stats,normalization_source=normalization)
    models={};emas={};opts={}
    for mode in MODES:
        model=build(args.task,source['config']).cuda();initialize(model,source['ema'],args.task)
        models[mode]=model;emas[mode]=copy.deepcopy(model).eval().requires_grad_(False)
        opts[mode]=torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.99))
    step=0
    if resume:
        for mode in MODES:
            models[mode].load_state_dict(resume['models'][mode]);emas[mode].load_state_dict(resume['ema'][mode]);opts[mode].load_state_dict(resume['opts'][mode])
        step=resume['step'];data.rng.bit_generator.state=resume['data_rng'];torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    del source
    initial_step=step
    atomic_json(launch,{**signature,'source':str(SOURCE),'arguments':vars(args),
        'matched':'Identical source, capacity, optimizer schedule, batches/augmentation/guide corruption/sigma/noise; new plan weights/heads zero.',
        'selection_changed':False,'sea_level_m':0.,'planet_generated':False})
    def save():
        ck={**signature,'step':step,'models':{m:v.state_dict() for m,v in models.items()},'ema':{m:v.state_dict() for m,v in emas.items()},
            'opts':{m:v.state_dict() for m,v in opts.items()},'data_rng':data.rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        tmp=out/'latest.pt.tmp';torch.save(ck,tmp);tmp.replace(checkpoint)
    print(f'{args.task}: paired update {step}/{args.steps}; batch {args.batch}; wall-time budget {args.max_seconds}s including setup and validation.',flush=True)
    stop=[False];previous_handler=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    status='complete';log=(out/'log.jsonl').open('a',encoding='utf-8')
    def record(row):log.write(json.dumps(row)+'\n');log.flush()
    try:
        before=evaluate(emas,args.task,norm,codec,data);record({'step':step,'validation_before':before})
        window=[];last=time.perf_counter();last_step=step
        while step<args.steps:
            if stop[0]:status='interrupted';break
            if time.perf_counter()-started>=args.max_seconds-8:status='wall-time limit';break
            raw,c,ctx=data.batch(args.batch,device='cuda');k=int(torch.randint(4,()))
            raw,c,ctx=[torch.rot90(x,k,(-2,-1)) for x in [raw,c,ctx]]
            if torch.rand(())<.5:raw,c,ctx=[x.flip(-1) for x in [raw,c,ctx]]
            c=degrade_guides(ctx,c,None,recipe=STRUCTURE_RECIPE)
            target,base,up,cin,plans=batch_fields(norm,codec,raw,c)
            sigma=(torch.randn(len(raw),device='cuda')*1.2-1.2).exp();noise=torch.randn((len(raw),5,64,64),device='cuda')
            row={}
            for mode in MODES:
                with autocast('cuda'):parts=loss_parts(models[mode],args.task,norm,target,base,up,cin,plans[mode],sigma,noise)
                if not all(torch.isfinite(v) for v in parts.values()):raise FloatingPointError('Nonfinite paired loss')
                for group in opts[mode].param_groups:group['lr']=.0002*min(1.,(step+1)/200)
                opts[mode].zero_grad(set_to_none=True);parts['total'].backward()
                torch.nn.utils.clip_grad_norm_(models[mode].parameters(),1.,error_if_nonfinite=True);opts[mode].step()
                with torch.no_grad():
                    decay=min(.999,(2+step)/(11+step))
                    for pe,pm in zip(emas[mode].parameters(),models[mode].parameters()):pe.lerp_(pm,1-decay)
                row[mode]={k:float(v.detach()) for k,v in parts.items()}
            step+=1;window.append(row)
            if step%50==0 or step==args.steps:
                now=time.perf_counter();rate=(step-last_step)/max(.001,now-last)
                means={m:float(np.mean([v[m]['total'] for v in window])) for m in MODES}
                record({'step':step,'loss':means,'pairs_per_second':rate,'elapsed_seconds':now-started})
                print(f'Pair {step}/{args.steps} | {rate:.1f} pairs/s | {now-started:.1f}s | losses {means}',flush=True)
                window=[];last=now;last_step=step
            if step%250==0:save()
        save();after=evaluate(emas,args.task,norm,codec,data);record({'step':step,'validation_after':after})
    finally:
        log.close();signal.signal(signal.SIGINT,previous_handler)
    result={'status':status,'task':args.task,'paired_updates_each':step,'session_updates_each':step-initial_step,'target_updates_each':args.steps,
        'elapsed_seconds':time.perf_counter()-started,'validation_before':before,'validation_after':after,
        'source_unchanged':digest(SOURCE)==SOURCE_SHA,'selected_changed':False,'sea_level_m':0.,'planet_generated':False}
    atomic_json(out/'training-result.json',result)
    print(f'Finished {status}: {step} updates EACH, {result["elapsed_seconds"]:.1f}s; snapshot {checkpoint}. No automatic next job.',flush=True)
    return 130 if status=='interrupted' else 2 if status!='complete' else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['renderer','planner'],required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--max-seconds',type=float,default=110.)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=42);p.add_argument('--run')
    args=p.parse_args(argv)
    if min(args.steps,args.batch)<1 or args.max_seconds<10 or not 0<=args.seed<2**63:p.error('Invalid counts or budget')
    out=Path(args.run).resolve() if args.run else OUT_ROOT/(args.task+'-paired-1000')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','train-'+args.task):
        with training_lock(out):return train(args,out)


if __name__=='__main__':raise SystemExit(main())
