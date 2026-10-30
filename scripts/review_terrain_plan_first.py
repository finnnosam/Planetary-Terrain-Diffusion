"""Bounded shared-renderer review of a dedicated planner; compact gallery."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from planet_training.run_lock import training_lock
from planet_training.terrain_plan_first import load_pair,sample_plan
from planet_training.terrain_joint_structure import interpret_plan
from planet_training.terrain_plan_oracle import load as load_renderer,sample as render_plan
from planet_training.terrain_structure import handoff,encode
from toy_meso.model import upsample_condition
from toy_meso.common import load_checkpoint,sample_patches
from scripts.train_terrain_plan_first import OUT_ROOT,SOURCE_SHA
from scripts.probe_joint_plan_influence import digest,DEFAULT_REVIEW
from scripts.probe_structure_stage_roles import array,tensor,assert_constraints,describe,compare_meso
from scripts.review_terrain_joint_structure import plan_metrics
from scripts.diagnose_terrain_structure import image,legend
from scripts.structure_job import console_log

CASES=['alps','california','plains','holdout-rugged','holdout-quiet','generated-60']
SEEDS=[42,314,2718,2026]
RENDER_SEEDS=[7919,104729]
MODES=['source','coupled','planner']
LABELS={'source':'Original joint plan','coupled':'Continued joint plan','planner':'Dedicated planner','reference':'SUPPLIED reference plan'}


def panel(ax,field,title,spacing,seen):
    key=field.tobytes()
    if key in seen:
        ax.text(.5,.5,'Identical to\n'+seen[key],ha='center',va='center',transform=ax.transAxes,fontsize=9);ax.set_axis_off()
    else:
        seen[key]=title
        half=len(field)*spacing/2000
        ax.imshow(image(field,spacing),extent=(-half,half,-half,half),interpolation='nearest')
        ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
    ax.set_title(title,fontsize=9)


def visualise(out,rendered,fine,settings):
    for fine_seed in [None,113,997]:
        fig,axs=plt.subplots(len(CASES),4,figsize=(13,3.1*len(CASES)+.8),squeeze=False,constrained_layout=True)
        seen={}
        for i,case in enumerate(CASES):
            for j,mode in enumerate(MODES+['reference']):
                key=(case,42,mode,7919)
                if key not in rendered:axs[i,j].set_axis_off();continue
                if fine_seed is None:
                    field=rendered[key]['direct23_metres'][0,0,30:82,30:82];spacing=23040.
                else:field=fine[key+(fine_seed,)][0,0];spacing=2880.
                panel(axs[i,j],field,case+'\n'+LABELS[mode],spacing,seen)
        extent='1,198.08 km guides' if fine_seed is None else f'368.64 km raw meso; seed {fine_seed}'
        fig.suptitle(('EXECUTION SMOKE ONLY: ' if settings['smoke_only'] else '')+'Shared frozen plan renderer: '+extent+'\nPlan seed 42, renderer seed 7919. Fixed metre colours, NW 5x shading. Reference supplied, never generated.',fontsize=11)
        legend(fig,axs);name='primary-guides.png' if fine_seed is None else f'primary-meso-{fine_seed}.png'
        fig.savefig(out/name,dpi=115);plt.close(fig)
    for case in CASES:
        for rs in RENDER_SEEDS:
            fig,axs=plt.subplots(3,4,figsize=(12,9.5),squeeze=False,constrained_layout=True);seen={}
            for i,mode in enumerate(MODES):
                for j,seed in enumerate(SEEDS):
                    field=rendered[case,seed,mode,rs]['direct23_metres'][0,0,30:82,30:82]
                    panel(axs[i,j],field,LABELS[mode]+f'\nplan seed {seed}',23040.,seen)
            fig.suptitle(f'{case}; renderer seed {rs}; each 1,198.08 km\nSame frozen renderer/coarse guide/noise across methods. All declared plan seeds; references shown once in primary view.',fontsize=11)
            legend(fig,axs);fig.savefig(out/f'{case}-all-seeds-r{rs}.png',dpi=110);plt.close(fig)
    page='<!doctype html><html><head><meta charset="utf-8"><title>Dedicated terrain planner</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5}p{max-width:1100px;line-height:1.5}img{width:100%;max-width:1600px}details{margin:25px 0}summary{cursor:pointer;font-weight:600}</style></head><body><h1>Dedicated planner versus continued joint task</h1><p>Original joint plans, continued joint plans and dedicated plans share exactly the same frozen renderer. All plans are generated from coarse guides. The final reference column receives supplied Earth-derived plans; it is not a learned planner success. Primary view uses the first predeclared seeds, not selected best samples. Fixed metre colours, zero sea level and NW 5x shading. Exact duplicate fields are labelled instead of redrawn.</p>'
    if settings['smoke_only']:page+='<p><strong>Execution smoke only; two training updates per branch.</strong></p>'
    page+='<h2>Raw meso, 369 km</h2><img src="primary-meso-113.png"><details><summary>Second meso seed, same comparisons</summary><img src="primary-meso-997.png"></details>'
    page+='<details><summary>Wider structural guides, 1,198 km</summary><img src="primary-guides.png"></details>'
    for case in CASES:
        page+=f'<details><summary>{html.escape(case)}: every plan seed and both renderer seeds</summary>'
        for rs in RENDER_SEEDS:page+=f'<img loading="lazy" src="{case}-all-seeds-r{rs}.png">'
        page+='</details>'
    (out/'index.html').write_text(page+'</body></html>',encoding='utf-8')


@torch.no_grad()
def review(args,out):
    started=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source_review=Path(args.source_review).resolve();previous=json.loads((source_review/'settings.json').read_text())
    models,norm,ck=load_pair(out/'paired-checkpoint.pt','cuda')
    if ck['source_sha256']!=SOURCE_SHA or previous['checkpoints']['joint']['sha256']!=SOURCE_SHA:raise ValueError('Wrong source joint checkpoint')
    if ck['step']<10 and not args.smoke:raise ValueError('Smoke checkpoint requires --smoke')
    for name in ['renderer','meso']:
        if digest(source_review/f'{name}-checkpoint.pt')!=previous['checkpoints'][name]['sha256']:raise ValueError('Frozen downstream checkpoint changed')
    teacher,tnorm,_=load_renderer(source_review/'renderer-checkpoint.pt','cuda')
    meso,mnorm,_=load_checkpoint(source_review/'meso-checkpoint.pt','cuda')
    global CASES,SEEDS
    if args.smoke:CASES=['california','generated-60'];SEEDS=[42]
    settings={'created_local':datetime.now().isoformat(),'paired_step':ck['step'],'checkpoint_sha256':digest(out/'paired-checkpoint.pt'),
        'source_review':str(source_review),'source_joint_sha256':SOURCE_SHA,'dataset_id':ck['dataset_id'],'spec':ck['spec'],
        'cases':[c for c in previous['cases'] if c['label'] in CASES],'plan_seeds':SEEDS,'renderer_seeds':RENDER_SEEDS,
        'meso_seeds':[113,997],'fine_subset':'All declared cases at first plan seed 42 and first renderer seed 7919; selected before inspection.',
        'downstream_hashes':{k:previous['checkpoints'][k]['sha256'] for k in ['renderer','meso']},
        'limits':'Regional pilot, task/focus/input change together. Same renderer isolates differences in generated plans; extra renderer weights are not credited as planner capacity. No world survey or production tiling.',
        'smoke_only':args.smoke,'maximum_seconds':args.max_seconds,'selection_changed':False,'planet_generated':False,'sea_level_m':0.}
    records=[];rendered={};fine={};done=0;cache_count=0;completed=set()
    if args.resume:
        old=json.loads((out/'results.json').read_text())
        for key in ['paired_step','checkpoint_sha256','source_review','source_joint_sha256','dataset_id','spec','cases','plan_seeds','renderer_seeds','meso_seeds','smoke_only']:
            if old['settings'][key]!=settings[key]:raise ValueError(f'Resume review identity changed: {key}')
        settings=old['settings'];settings.setdefault('resume_sessions',[]).append({'started_local':datetime.now().isoformat(),'maximum_seconds':args.max_seconds})
        records=old['records']
        for row in records:
            case,seed,mode,rs=row['case'],row['plan_seed'],row['mode'],row['renderer_seed']
            key=(case,seed,mode,rs)
            rendered[key]=dict(np.load(out/f'{case}-p{seed}-{mode}-r{rs}.npz'))
            for m in row['meso']:
                with np.load(out/f'{case}-p{seed}-{mode}-r{rs}-m{m["seed"]}.npz') as a:fine[key+(m['seed'],)]=a['raw_meso_metres'].copy()
            cache_count+=int(row['source_cached'])+(len(row['meso']) if mode=='source' else 0)
        for case in CASES:
            for seed in SEEDS:
                if all((case,seed,m,rs) in rendered for m in MODES for rs in RENDER_SEEDS):completed.add((case,seed))
        print(f'Resuming {len(completed)}/{len(CASES)*len(SEEDS)} completed groups; saved fields reused exactly.',flush=True)
    atomic_json(out/'settings.json',settings)
    for case in CASES:
        d=dict(np.load(source_review/f'{case}-s42-joint.npz'));coarse=tensor(d['coarse_encoded']).cuda();base=norm.condition(coarse)[0]
        for seed in SEEDS:
            if (case,seed) in completed:done+=1;continue
            if time.perf_counter()-started>args.max_seconds-15:
                atomic_json(out/'results.json',{'status':'budget limited','settings':settings,'records':records,
                    'cache_reuses':cache_count,'session_inference_seconds':time.perf_counter()-started})
                print(f'Wall-time budget reached; {done}/{len(CASES)*len(SEEDS)} groups saved. Resume this exact --out with --resume. No training or follow-on job started.',flush=True)
                return 2
            old=dict(np.load(source_review/f'{case}-s{seed}-joint.npz'))
            plans={'source':tensor(old['generated_plan_raw']).cuda()}
            for mode in ['coupled','planner']:plans[mode]=sample_plan(models[mode],norm,coarse,mode,32,seed,'cuda')
            if seed==42 and (source_review/f'{case}-s42-feasible-target-reference.npz').exists():
                ref=dict(np.load(source_review/f'{case}-s42-feasible-target-reference.npz'));plans['reference']=tensor(ref['interpreted_plan']).cuda()
            for mode,raw_plan in plans.items():
                plan=interpret_plan(raw_plan,base)
                np.savez_compressed(out/f'{case}-p{seed}-{mode}-plan.npz',coarse_encoded=array(coarse),raw_plan=array(raw_plan),interpreted_plan=array(plan))
                for rs in RENDER_SEEDS:
                    reused=mode=='source' and rs==7919
                    if reused:
                        saved=dict(np.load(source_review/f'{case}-s{seed}-joint-plan-rendered.npz'))
                        raw=tensor(saved['raw40_metres']).cuda();z=tensor(saved['projected40_metres']).cuda();cache_count+=1
                    else:z,_,raw=render_plan(teacher,tnorm,coarse,plan,32,rs,'cuda')
                    assert_constraints(z,base,16,40320.)
                    guide,_,base23=handoff(z,base,upsample_condition(coarse,7),coarse);assert_constraints(guide,base23,28,23040.)
                    if reused:np.testing.assert_array_equal(array(guide),saved['direct23_metres'])
                    saved={'raw40_metres':array(raw),'projected40_metres':array(z),'direct23_metres':array(guide),'plan':array(plan)}
                    if not all(np.isfinite(v).all() for v in saved.values()):raise ValueError('Nonfinite structure')
                    key=(case,seed,mode,rs);rendered[key]=saved
                    np.savez_compressed(out/f'{case}-p{seed}-{mode}-r{rs}.npz',**saved)
                    row={'case':case,'plan_seed':seed,'mode':mode,'renderer_seed':rs,'source_cached':reused,
                         'plan_following':plan_metrics(z,base,plan),'guide_description':describe(array(guide)[0,0,30:82,30:82],23040.),'meso':[]}
                    if seed==42 and rs==7919:
                        cond=encode(guide[...,47:65,47:65]).float()
                        for ms in [113,997]:
                            if mode=='source':
                                existing=dict(np.load(source_review/f'{case}-s{seed}-joint-plan-rendered-m{ms}.npz'))
                                np.testing.assert_array_equal(existing['meso_input_encoded'],array(cond));field=tensor(existing['raw_meso_metres']).cuda();cache_count+=1
                            else:
                                encoded,_=sample_patches(meso,mnorm,cond,32,ms,'cuda');field=encoded.sign()*encoded.square()
                            if not torch.isfinite(field).all():raise ValueError('Nonfinite meso')
                            fine[key+(ms,)]=array(field)
                            np.savez_compressed(out/f'{case}-p{seed}-{mode}-r{rs}-m{ms}.npz',meso_input_encoded=array(cond),raw_meso_metres=array(field))
                            row['meso'].append({'seed':ms,'description':describe(array(field)[0,0],2880.),'preservation':compare_meso(field,guide)})
                    records.append(row)
            done+=1;print(f'Review {done}/{len(CASES)*len(SEEDS)} plan groups; elapsed {time.perf_counter()-started:.1f}s',flush=True)
            atomic_json(out/'results.json',{'status':'in progress','settings':settings,'records':records,'cache_reuses':cache_count,'session_inference_seconds':time.perf_counter()-started})
    atomic_json(out/'results.json',{'status':'complete','settings':settings,'records':records,'cache_reuses':cache_count,'inference_seconds':time.perf_counter()-started,'visual_review_required':True})
    visualise(out,rendered,fine,settings)
    print(f'Finished review in {time.perf_counter()-started:.1f}s: {out/"index.html"}. No selection or planet.',flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default=str(OUT_ROOT/'paired-1000/latest.pt'))
    p.add_argument('--source-review',default=DEFAULT_REVIEW);p.add_argument('--out');p.add_argument('--max-seconds',type=float,default=120.);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);out=Path(args.out).resolve() if args.out else OUT_ROOT/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if args.resume:
        if not args.out or not (out/'results.json').exists():raise ValueError('--resume requires an existing review --out')
    elif out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output folder')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','review'):
        with training_lock(Path(args.checkpoint).resolve().parent):
            if args.resume:
                if digest(args.checkpoint)!=digest(out/'paired-checkpoint.pt'):raise ValueError('Checkpoint changed since this review')
            else:shutil.copyfile(args.checkpoint,out/'paired-checkpoint.pt')
        return review(args,out)


if __name__=='__main__':raise SystemExit(main())
