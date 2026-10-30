"""Review autonomous intermediate structure, optionally through frozen meso."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from planet_training.adaptation import degrade_guides, STRUCTURE_RECIPE
from planet_training.data import atomic_json, manifest_id
from planet_training.run_lock import training_lock
from planet_training.terrain_structure import StructureData, DEFAULT_DATA, decode, encode, conservative_resample, target_for, handoff
from planet_training.terrain_plan_oracle import load as load_renderer, sample as render_plan, plan_for
from planet_training.terrain_joint_structure import load, sample, interpret_plan
from scripts.train_terrain_joint_structure import OUT_ROOT
from scripts.probe_structure_stage_roles import ROOT, MODELS, build_cases, array, tensor, digest, assert_constraints, describe, compare_meso
from scripts.diagnose_terrain_structure import image, legend
from scripts.structure_job import console_log
from toy_meso.model import upsample_condition
from toy_meso.common import pick_device, load_checkpoint, sample_patches

RENDERER = ROOT/'outputs/structure/plan-oracle-v1/review-20261013-220148-733529/oracle-checkpoint.pt'
RENDERER_SHA = 'bd5e1fc86bbe835d5b7f96d73736e780bb32d9456f26e506a285845ade0ceb6f'
RENDERER_SEED = 7919


def inputs(folder):
    audit=json.loads((folder/'target-audit.json').read_text())
    selected=[(s['label'],audit['records'][s['index']]) for s in audit['selected_holdouts']]
    used={r['index'] for _,r in selected}
    candidates=sorted([r for r in audit['records'] if r['index'] not in used and r['fields']['modern23']['land_fraction']>=.85],
                      key=lambda r:(r['fields']['modern23']['land_local_relief_p50_p90_m'][0],r['index']))
    for quantile in (.1,.5,.9):
        if not candidates:break
        record=candidates[round(quantile*(len(candidates)-1))]
        selected.append((f'extra-holdout-q{int(100*quantile)}',record));candidates.remove(record)
    with np.load(folder/'target-audit-arrays.npz') as a:
        arrays=tuple(tensor(a[k]) for k in ('modern23_encoded','degraded_coarse_encoded','translated23_encoded','constrained40_metres','baseline40_metres'))
    cases=build_cases(selected,arrays)
    prior=json.loads((folder/'settings.json').read_text())
    expected={r['label']:r for r in prior['case_selection']}
    for case in cases:
        if 'path' in case and case['sha256']!=expected[case['label']]['sha256']:raise ValueError('Audited regional input changed')
    return cases,prior


def plan_metrics(output,base,desired):
    """Self-consistency only, with rare-class denominators made explicit."""
    actual=plan_for(output,base); active=base>200; result={}
    for key,truth,pred in [('high',desired[:,:1]>.5,actual[:,:1]>.5),
                           ('low',desired[:,:1]<-.5,actual[:,:1]<-.5),
                           ('quiet',desired[:,1:2]>.5,actual[:,1:2]>.5)]:
        truth,pred=truth&active,pred&active;tp=int((truth&pred).sum());n=int(truth.sum());m=int(pred.sum())
        result[key]={'desired_cells':n,'predicted_cells':m,'precision':tp/m if m else None,'recall':tp/n if n else None}
    result['limit']='Generated-plan / own-height consistency, not Earth reconstruction or terrain realism.'
    return result


def render_case(out,case,seed,rows,base,meso_seeds,with_meso,smoke=False):
    cols=3+(len(meso_seeds) if with_meso else 0)
    fig,axes=plt.subplots(len(rows),cols,figsize=(3.4*cols,3.15*len(rows)+.8),squeeze=False,constrained_layout=True)
    for row,(name,raw,z,guide,plan) in enumerate(rows):
        for col,field in enumerate((raw,z,guide)):
            field=F.interpolate(field,size=(112,112),mode='bilinear',align_corners=False) if col<2 else field
            axes[row,col].imshow(image(array(field)[0,0,30:82,30:82],23040.),extent=(-599.04,599.04,-599.04,599.04),interpolation='nearest')
            axes[row,col].set_title(name+'\n'+['Raw 40 km; 1,198 km','Constrained 40 km; 1,198 km','23 km handoff; 1,198 km'][col],fontsize=9)
        if with_meso:
            for col,ms in enumerate(meso_seeds,3):
                with np.load(out/f'{case["label"]}-s{seed}-{name}-m{ms}.npz') as a:field=a['raw_meso_metres'][0,0]
                axes[row,col].imshow(image(field,2880.),extent=(-184.32,184.32,-184.32,184.32),interpolation='nearest')
                axes[row,col].set_title(f'Raw meso; 369 km\nseed {ms}',fontsize=9)
        for ax in axes[row]:ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=8)
    fig.suptitle(f'{"EXECUTION SMOKE ONLY: " if smoke else ""}{case["label"]}; structure seed {seed}; {case["orientation"]}\nCoarse-only generation. Supplied reference rows are not learned successes. Fixed colours / NW 5x shading.',fontsize=11)
    legend(fig,axes);name=f'{case["label"]}-s{seed}.png';fig.savefig(out/name,dpi=115);plt.close(fig)
    # Show generated plan and the plan implied by its own terrain separately.
    learned=next(r for r in rows if r[0]=='joint')
    fig,axs=plt.subplots(2,2,figsize=(9,9),constrained_layout=True)
    q=(np.arange(64)+.5-32)*40.32;xx,yy=np.meshgrid(q,-q)
    for col,(p,title) in enumerate(zip([learned[4],plan_for(learned[2],base)],['Generated joint plan','Plan measured from generated terrain'])):
        field=F.interpolate(p[:,:1],size=(112,112),mode='nearest')
        axs[0,col].imshow(array(field)[0,0,30:82,30:82],vmin=-1,vmax=1,cmap=ListedColormap(['#4a8bb1','#e5e5df','#a65f3d']),extent=(-599.04,599.04,-599.04,599.04),interpolation='nearest')
        planes=array(p)[0];theta=.5*np.arctan2(planes[3],planes[2])
        sel=(np.indices((64,64))[0]%3==0)&(np.indices((64,64))[1]%3==0)&(np.abs(planes[0])>.5)&(np.hypot(planes[2],planes[3])>.1)&(np.abs(xx)<599.04)&(np.abs(yy)<599.04)
        axs[0,col].quiver(xx[sel],yy[sel],70*np.cos(theta[sel]),70*np.sin(theta[sel]),angles='xy',scale_units='xy',scale=1,pivot='middle',headlength=0,headaxislength=0,headwidth=0,color='#222222',width=.004)
        field=F.interpolate(p[:,1:2],size=(112,112),mode='nearest')
        axs[1,col].imshow(array(field)[0,0,30:82,30:82],vmin=0,vmax=1,cmap=ListedColormap(['#e5e5df','#56824e']),extent=(-599.04,599.04,-599.04,599.04),interpolation='nearest')
        axs[0,col].set_title(title+'\nHigh/low corridors and axes');axs[1,col].set_title('Green: quiet permitted cells')
    for ax in axs.flat:ax.set_xlabel('km');ax.set_ylabel('km')
    fig.suptitle(('EXECUTION SMOKE ONLY\n' if smoke else '')+'Categorical colours: blue low / brown high; neutral includes protected cells\nGenerated plan is a model output, not an Earth input; 1,198.08 km',fontsize=11)
    fig.savefig(out/f'{case["label"]}-s{seed}-plans.png',dpi=115);plt.close(fig)
    return name


@torch.no_grad()
def review(args,out):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=pick_device(args.device)
    control,cnorm,cc=load(args.control,device);joint,jnorm,jc=load(args.joint,device)
    if cc['mode']!='control' or jc['mode']!='joint':raise ValueError('Wrong control/joint checkpoint modes')
    for key in ('step','dataset_id','source_sha256','config','optimization','stats','objective','joint_spec'):
        if cc[key]!=jc[key]:raise ValueError(f'Unmatched pilot identity: {key}')
    if not args.smoke and cc['step']<=2:raise ValueError('Execution-smoke checkpoints require --smoke')
    data=StructureData(DEFAULT_DATA)
    cases,prior=inputs(Path(args.inputs).resolve())
    if manifest_id(data.meta)!=cc['dataset_id'] or prior['data_manifest_id']!=cc['dataset_id']:raise ValueError('Dataset identity changed')
    _,original,context=data.batch(0,'val',np.arange(data.meta['val_count']))
    verified_coarse=degrade_guides(context,original,torch.Generator().manual_seed(2026),sigma_cells=1.,clean_fraction=0.,recipe=STRUCTURE_RECIPE)
    for case in cases:
        if 'index' in case:
            raw,_,_=data.batch(0,'val',np.array([case['index']]))
            torch.testing.assert_close(raw,conservative_resample(decode(case['modern_encoded'])[:,0],64)[:,None],atol=.02,rtol=1e-5)
            torch.testing.assert_close(case['coarse'],verified_coarse[case['index']:case['index']+1],atol=1e-5,rtol=1e-5)
    if digest(args.renderer)!=RENDERER_SHA:raise ValueError('Frozen renderer changed')
    renderer,rnorm,_=load_renderer(args.renderer,device)
    meso=mnorm=None
    if args.with_meso:
        meso,mnorm,_=load_checkpoint(args.meso,device)
    seeds,steps=([42],8) if args.smoke else (args.seeds,32)
    if args.smoke:cases=[next(c for c in cases if c['label']==v) for v in ['california','generated-60']]
    settings={'created_utc':datetime.now(timezone.utc).isoformat(),'step':cc['step'],'seeds':seeds,
        'steps':steps,'renderer_seed':RENDERER_SEED,'meso_seeds':args.meso_seeds,'with_meso':args.with_meso,'smoke_only':args.smoke,
        'checkpoints':{k:{'path':str(Path(v).resolve()),'sha256':digest(v)} for k,v in args.snapshots.items()},
        'checkpoint_sources':args.checkpoint_sources,'dataset_id':cc['dataset_id'],'joint_spec':cc['joint_spec'],
        'inputs':str(Path(args.inputs).resolve()),'input_settings_sha256':digest(Path(args.inputs)/'settings.json'),
        'input_audit_sha256':digest(Path(args.inputs)/'target-audit.json'),
        'input_arrays_sha256':digest(Path(args.inputs)/'target-audit-arrays.npz'),
        'cases':[{k:v for k,v in c.items() if not isinstance(v,(np.ndarray,torch.Tensor))} for c in cases],
        'extra_holdout_selection':'Remaining inland holdouts sorted by target relief; choose q10/q50/q90 sequentially without replacement before generation. All were validation-monitored, not fresh confirmation.',
        'limits':'Regional diagnostics, no world prevalence or production tile blending. Generated prefixes share upstream seed 42. Plan rendering is an intervention through the frozen oracle, not another production stage.',
        'oracle_plan_conditioning':False,'sea_level_m':0.,'selection_changed':False,'planet_generated':False}
    atomic_json(out/'settings.json',settings);records=[];cards=[]
    for case in cases:
        coarse=case['coarse'].to(device);base=jnorm.condition(coarse)[0]
        oracle=None
        if case['modern_encoded'] is not None:
            raw40=conservative_resample(decode(case['modern_encoded'].to(device))[:,0],64)[:,None]
            oracle=decode(target_for(raw40,coarse)[0])
        for seed in seeds:
            cz,cb,cr,cp=sample(control,cnorm,coarse,steps,seed,device)
            jz,jb,jr,jp=sample(joint,jnorm,coarse,steps,seed,device)
            torch.testing.assert_close(cb,jb,atol=0,rtol=0)
            interpreted=interpret_plan(jp,jb)
            rz,rb,rr=render_plan(renderer,rnorm,coarse,interpreted,steps,RENDERER_SEED,device)
            branches=[('control',cr,cz,interpret_plan(cp,cb),cp),('joint',jr,jz,interpreted,jp),
                      ('joint-plan-rendered',rr,rz,interpreted,jp)]
            if oracle is not None:branches.append(('feasible-target-reference',oracle,oracle,plan_for(oracle,base),plan_for(oracle,base)))
            rows=[]
            for name,raw,z,plan,raw_plan in branches:
                checks=assert_constraints(z,base,16,40320.)
                guide,_,base23=handoff(z,base,upsample_condition(coarse,7),coarse)
                checks23=assert_constraints(guide,base23,28,23040.)
                saved={'coarse_encoded':array(coarse),'baseline40_metres':array(base),'raw40_metres':array(raw),
                    'projected40_metres':array(z),'generated_plan_raw':array(raw_plan),'interpreted_plan':array(plan),
                    'direct23_metres':array(guide)}
                if not all(np.isfinite(v).all() for v in saved.values()):raise ValueError('Nonfinite saved field')
                np.savez_compressed(out/f'{case["label"]}-s{seed}-{name}.npz',**saved)
                row={'case':case['label'],'seed':seed,'branch':name,'constraints40':checks,'constraints23':checks23,
                     'plan_self_consistency':plan_metrics(z,base,plan),'structure':describe(array(guide)[0,0,30:82,30:82],23040.),'meso':[]}
                if args.with_meso:
                    condition=encode(guide[...,47:65,47:65]).float()
                    for ms in args.meso_seeds:
                        encoded,up=sample_patches(meso,mnorm,condition,steps,ms,device)
                        fine=decode(encoded.float());baseline=decode(up)
                        np.savez_compressed(out/f'{case["label"]}-s{seed}-{name}-m{ms}.npz',
                            meso_input_encoded=array(condition),meso_baseline_metres=array(baseline),raw_meso_metres=array(fine))
                        np.testing.assert_array_equal(array(condition),array(encode(guide[...,47:65,47:65]).float()))
                        if not torch.isfinite(fine).all():raise ValueError('Nonfinite meso field')
                        row['meso'].append({'seed':ms,'raw_preservation':compare_meso(fine,guide),
                                            'description':describe(array(fine)[0,0],2880.)})
                records.append(row);rows.append((name,raw,z,guide,plan))
            cards.append(render_case(out,case,seed,rows,base,args.meso_seeds,args.with_meso,args.smoke))
            atomic_json(out/'results.json',{'status':'in progress','settings':settings,'records':records})
            print(f'Completed {len(cards)}/{len(cases)*len(seeds)} regional seed groups: {case["label"]}, {seed}',flush=True)
    atomic_json(out/'results.json',{'status':'complete','settings':settings,'records':records,'visual_review_required':True})
    content='<!doctype html><html><head><meta charset="utf-8"><title>Autonomous structure review</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5}p{max-width:1100px;line-height:1.5}img{width:100%;max-width:1900px}</style></head><body><h1>Autonomous intermediate structure</h1>'
    if args.smoke:content+='<p><strong>EXECUTION SMOKE ONLY: two training updates and eight sampling steps. These images are not a terrain-quality result.</strong></p>'
    content+='<p>Joint/control generation uses coarse heights only. Plans are generated outputs. The frozen renderer receives the generated plan with independent fixed seed 7919. Supplied feasible-target rows are references, never learned successes. Raw meso is primary and uncorrected. Structure panels cover 1,198 km; meso panels the central 369 km. Fixed elevation colours and NW 5x shading, zero datum. Regional diagnostics do not establish global frequency or production blending.</p>'
    for name in cards:
        content+=f'<section><h2>{html.escape(name[:-4])}</h2><a href="{name}"><img src="{name}"></a><img src="{name[:-4]}-plans.png"></section>'
    (out/'index.html').write_text(content+'</body></html>',encoding='utf-8')
    print(f'Finished: {out/"index.html"}. No model selected or planet started.',flush=True);return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',default=str(OUT_ROOT/'control/latest.pt'));p.add_argument('--joint',default=str(OUT_ROOT/'joint/latest.pt'))
    p.add_argument('--inputs',default='outputs/terrain-stage-roles/paired-v1');p.add_argument('--out');p.add_argument('--device',default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,314,2718,2026]);p.add_argument('--meso-seeds',type=int,nargs='+',default=[113,997])
    p.add_argument('--with-meso',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    for values in (args.seeds,args.meso_seeds):
        if len(set(values))!=len(values) or any(s<0 or s>=2**63 for s in values):p.error('Invalid seeds')
    out=Path(args.out).resolve() if args.out else OUT_ROOT/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh review folder')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','review'):
        sources={'control':args.control,'joint':args.joint,'renderer':str(RENDERER)}
        if args.with_meso:
            path,expected=MODELS['meso'];path=ROOT/path
            if digest(path)!=expected:raise ValueError('Selected meso changed')
            sources['meso']=str(path)
        args.checkpoint_sources={k:str(Path(v).resolve()) for k,v in sources.items()};args.snapshots={}
        with ExitStack() as locks:
            for folder in sorted({str(Path(v).parent) for v in sources.values()}):locks.enter_context(training_lock(folder))
            for key,path in sources.items():
                target=out/f'{key}-checkpoint.pt';shutil.copyfile(path,target);args.snapshots[key]=str(target)
        for key,path in args.snapshots.items():setattr(args,key,path)
        return review(args,out)


if __name__=='__main__':raise SystemExit(main())
