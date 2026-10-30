"""One paired, wall-time-bounded planner/coupled experiment; never selects models."""
import argparse
import copy
from datetime import datetime
import json
import signal
from pathlib import Path
import time

import numpy as np
import torch

from planet_training.data import atomic_json,manifest_id
from planet_training.adaptation import degrade_guides,STRUCTURE_RECIPE
from planet_training.terrain_structure import StructureData,StructureNorm,DEFAULT_DATA,OBJECTIVE
from planet_training.terrain_joint_structure import build,targets
from planet_training.terrain_plan_first import VERSION,SPEC,parts
from planet_training.run_lock import training_lock
from scripts.probe_joint_plan_influence import digest
from scripts.structure_job import console_log
from toy_meso.common import autocast

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/structure/joint-structure-v1/review-20261014-143545-003794/joint-checkpoint.pt'
SOURCE_SHA='42b0fa2ed2c4f472021cfeeba417a1c09f1994a6e842cbdfebaed1fada4558d0'
OUT_ROOT=ROOT/'outputs/structure/plan-first-v1'


@torch.no_grad()
def evaluate(models,norm,data):
    total={mode:{k:0. for k in ['total','plan','height','shape']} for mode in models}
    g=torch.Generator(device='cuda').manual_seed(2026);count=data.meta['val_count']
    for start in range(0,count,16):
        raw,c,ctx=data.batch(0,'val',np.arange(start,min(start+16,count)),'cuda')
        c=degrade_guides(ctx,c,g,sigma_cells=1.,clean_fraction=0.,recipe=STRUCTURE_RECIPE)
        x,cin,target,base,up=targets(norm,raw,c,'joint')
        for sigma_value in [.05,.2,.5,1.,2.,5.]:
            sigma=torch.full((len(raw),),sigma_value,device='cuda')
            noise=torch.randn(x.shape,device='cuda',generator=g)
            for mode,model in models.items():
                with autocast('cuda'):values=parts(model,mode,x,cin,target,base,up,norm,sigma,noise)
                for k,v in values.items():total[mode][k]+=float(v)*len(raw)/(6*count)
    return total


def train(args,out):
    started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise ValueError('This bounded pilot requires CUDA')
    if digest(SOURCE)!=SOURCE_SHA:raise ValueError('Frozen source changed')
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    data=StructureData(DEFAULT_DATA,args.seed);norm=StructureNorm(source['stats'])
    optimization={'lr':.0002,'warmup':200,'seed':args.seed,'batch':args.batch,'ema':.999,'betas':[.9,.99],'clip':1.}
    identity={'version':VERSION,'spec':SPEC,'objective':OBJECTIVE,'source_sha256':SOURCE_SHA,
              'dataset_id':manifest_id(data.meta),'config':source['config'],'stats':source['stats'],'optimization':optimization}
    previous=json.loads((out/'launch.json').read_text()) if (out/'launch.json').exists() else None
    if previous and any(previous.get(k)!=v for k,v in identity.items()):raise ValueError('Changed run identity; use a fresh --run')
    resume=torch.load(out/'latest.pt',map_location='cpu',weights_only=False) if (out/'latest.pt').exists() else None
    if resume and any(resume.get(k)!=v for k,v in identity.items()):raise ValueError('Resume identity mismatch')
    models={};emas={};opts={}
    for mode in ['coupled','planner']:
        model=build(source['config']).cuda();model.load_state_dict(source['ema'])
        models[mode]=model;emas[mode]=copy.deepcopy(model).eval().requires_grad_(False)
        opts[mode]=torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.99))
    step=0
    if resume:
        for mode in models:
            models[mode].load_state_dict(resume['models'][mode]);emas[mode].load_state_dict(resume['ema'][mode]);opts[mode].load_state_dict(resume['opts'][mode])
        step=resume['step'];data.rng.bit_generator.state=resume['data_rng'];torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    del source
    atomic_json(out/'launch.json',{**identity,'arguments':vars(args),'source':str(SOURCE),
        'initialization':'Identical frozen joint EMA; both optimizers/EMA schedules fresh; single shared batch/noise per paired update.',
        'selection_changed':False,'sea_level_m':0.})
    print(f'Paired update {step}/{args.steps}; {args.batch} examples/mode; wall-time ceiling {args.max_seconds}s including setup/validation before final save.',flush=True)
    def save():
        ck={**identity,'step':step,'models':{m:v.state_dict() for m,v in models.items()},
            'ema':{m:v.state_dict() for m,v in emas.items()},'opts':{m:v.state_dict() for m,v in opts.items()},
            'data_rng':data.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        temp=out/'latest.pt.tmp';torch.save(ck,temp);temp.replace(out/'latest.pt')
    log=(out/'log.jsonl').open('a',encoding='utf-8')
    def record(row):log.write(json.dumps(row)+'\n');log.flush()
    validation_before=evaluate(emas,norm,data);record({'step':step,'initial_validation':validation_before})
    window=[];last=time.perf_counter();last_step=step;status='complete';stop_requested=[False]
    old_handler=signal.signal(signal.SIGINT,lambda signum,frame:stop_requested.__setitem__(0,True))
    try:
        while step<args.steps:
            if stop_requested[0]:status='interrupted';break
            if time.perf_counter()-started>=args.max_seconds-6:
                status='wall-time limit';break
            raw,c,ctx=data.batch(args.batch,device='cuda')
            k=int(torch.randint(4,()))
            raw,c,ctx=[torch.rot90(v,k,(-2,-1)) for v in (raw,c,ctx)]
            if torch.rand(())<.5:raw,c,ctx=[v.flip(-1) for v in (raw,c,ctx)]
            c=degrade_guides(ctx,c,None,recipe=STRUCTURE_RECIPE)
            x,cin,target,base,up=targets(norm,raw,c,'joint')
            sigma=(torch.randn(len(x),device='cuda')*1.2-1.2).exp();noise=torch.randn_like(x)
            values={}
            for mode,model in models.items():
                with autocast('cuda'):row=parts(model,mode,x,cin,target,base,up,norm,sigma,noise)
                if not all(torch.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite plan-first loss')
                for group in opts[mode].param_groups:group['lr']=.0002*min(1.,(step+1)/200)
                opts[mode].zero_grad(set_to_none=True);row['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opts[mode].step()
                with torch.no_grad():
                    for pe,pm in zip(emas[mode].parameters(),model.parameters()):pe.lerp_(pm,1-min(.999,(2+step)/(11+step)))
                values[mode]={k:float(v.detach()) for k,v in row.items()}
            step+=1;window.append(values)
            if step%50==0 or step==args.steps:
                now=time.perf_counter();rates=(step-last_step)/max(.001,now-last)
                summary={m:{k:float(np.mean([v[m][k] for v in window])) for k in ['total','plan','height','shape']} for m in models}
                record({'step':step,'loss':summary,'paired_updates_per_second':rates,'elapsed_seconds':now-started})
                print(f'Paired update {step}/{args.steps} | {rates:.1f} pairs/s | elapsed {now-started:.1f}s | plan losses '+str({m:round(summary[m]['plan'],4) for m in models}),flush=True)
                window=[];last=now;last_step=step
            if step%250==0:save()
        save();validation_after=evaluate(emas,norm,data);record({'step':step,'validation':validation_after})
    finally:log.close();signal.signal(signal.SIGINT,old_handler)
    result={'status':status,'additional_steps_each':step,'target_steps_each':args.steps,
        'elapsed_seconds':time.perf_counter()-started,'validation_before':validation_before,'validation_after':validation_after,
        'source_hash_unchanged':digest(SOURCE)==SOURCE_SHA,'selection_changed':False,'planet_generated':False,'sea_level_m':0.}
    atomic_json(out/'training-result.json',result)
    print(f'Finished paired training: {status}; {step} updates each; {result["elapsed_seconds"]:.1f}s. Review is separate.',flush=True)
    return 130 if status=='interrupted' else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--max-seconds',type=float,default=100.);p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run',default=str(OUT_ROOT/'paired-1000'))
    args=p.parse_args(argv)
    if min(args.steps,args.batch)<1 or args.max_seconds<10 or not 0<=args.seed<2**63:p.error('Invalid budget or seed')
    out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','train-pair'):
        with training_lock(out):return train(args,out)


if __name__=='__main__':raise SystemExit(main())
