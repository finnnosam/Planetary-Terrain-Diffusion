"""Bounded matched spatial-plan review; supplied plans and autonomous plans labelled."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from planet_training.run_lock import training_lock
from planet_training.terrain_spatial_plan import MODES,load_pair,sample_renderer,sample_planner
from planet_training.terrain_structure import decode,encode,handoff
from toy_meso.model import upsample_condition
from toy_meso.common import load_checkpoint,sample_patches
from scripts.train_terrain_spatial_plan import OUT_ROOT,SOURCE_SHA
from scripts.probe_joint_plan_influence import digest
from scripts.probe_structure_stage_roles import assert_constraints
from scripts.diagnose_terrain_structure import image,legend
from scripts.structure_job import console_log

PREVIOUS=Path(__file__).resolve().parents[1]/'outputs/structure/plan-first-v1/review-20261014-bounded-v1'
CASES=['alps','california','plains','holdout-rugged','holdout-quiet']
RENDER_SEEDS=[7919,104729]
PLAN_SEEDS=[42,314,2718,2026]
MESO_SEEDS=[113,997]


def show(ax,z,title,spacing):
    half=z.shape[-1]*spacing/2000
    ax.imshow(image(z,spacing),extent=(-half,half,-half,half),interpolation='nearest')
    ax.set_title(title,fontsize=9);ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)


def figures(out,records,autonomous,smoke=False):
    bykey={(r['case'],r['kind'],r['mode'],r['plan_seed'],r['renderer_seed']):r for r in records}
    kind='autonomous' if autonomous else 'supplied';ps=42 if autonomous else None
    cases=CASES+(['generated-60'] if autonomous else [])
    for rs in RENDER_SEEDS:
        fig,axs=plt.subplots(len(cases),3,figsize=(11,3.2*len(cases)),squeeze=False,constrained_layout=True)
        for i,c in enumerate(cases):
            ref=out/f'{c}-reference.npz'
            if ref.exists() and rs==RENDER_SEEDS[0]:show(axs[i,0],np.load(ref)['projected40_metres'][0,0,16:48,16:48],c+'\nSUPPLIED feasible height',40320.)
            elif ref.exists():
                axs[i,0].text(.5,.5,c+'\nSame supplied reference\nshown in first-seed guides',ha='center',va='center',transform=axs[i,0].transAxes)
                axs[i,0].set_axis_off()
            else:axs[i,0].set_axis_off()
            for j,m in enumerate(MODES,1):
                row=bykey[c,kind,m,ps,rs];z=np.load(out/row['file'])['projected40_metres'][0,0,16:48,16:48]
                show(axs[i,j],z,c+'\n'+m+' '+kind+' plan',40320.)
        fig.suptitle(f'Spatial supervision: {kind} plans; renderer seed {rs}; each 1290.24 km\nSame feasible height target, same model size and paired training; fixed metre colours / NW 5x shading.')
        legend(fig,axs);fig.savefig(out/f'guides-r{rs}.png',dpi=112);plt.close(fig)
    fine_cases=['alps','california']+(['generated-60'] if autonomous else [])
    fig,axs=plt.subplots(len(fine_cases)*2,3,figsize=(11,3.15*len(fine_cases)*2),squeeze=False,constrained_layout=True)
    for i,(c,ms) in enumerate((c,s) for c in fine_cases for s in MESO_SEEDS):
        ref=out/f'{c}-reference-m{ms}.npz'
        if ref.exists():
            z=np.load(ref)['raw_meso_metres'][0,0]
            show(axs[i,0],z,c+f', meso {ms}\nSUPPLIED feasible height → meso',2880.)
        else:
            axs[i,0].text(.5,.5,'Generated coast/relief prefix\nNo paired Earth reference',ha='center',va='center',transform=axs[i,0].transAxes)
            axs[i,0].set_axis_off()
        for j,m in enumerate(MODES,1):
            row=bykey[c,kind,m,ps,7919];entry=next(q for q in row['meso'] if q['seed']==ms)
            z=np.load(out/entry['file'])['raw_meso_metres'][0,0]
            show(axs[i,j],z,c+f', meso {ms}\n'+m+' '+kind+' plan → renderer → meso',2880.)
    fig.suptitle(f'Spatial supervision: {kind} planning, fixed first seeds\nEach 368.64 km; same selected meso; no old-intermediate detail; fixed elevations/shading.\nThe reference receives feasible Earth heights and is not learned generation.',fontsize=11)
    legend(fig,axs);fig.savefig(out/'fine-comparison.png',dpi=112);plt.close(fig)
    if autonomous:
        for c in cases:
            fig,axs=plt.subplots(2,4,figsize=(13,6.8),constrained_layout=True)
            for i,m in enumerate(MODES):
                for j,seed in enumerate(PLAN_SEEDS):
                    r=bykey[c,'autonomous',m,seed,7919]
                    show(axs[i,j],np.load(out/r['file'])['projected40_metres'][0,0,16:48,16:48],m+f'; plan seed {seed}',40320.)
            fig.suptitle(c+': all declared plan seeds, renderer 7919, each 1290.24 km');legend(fig,axs)
            fig.savefig(out/f'{c}-all-plan-seeds.png',dpi=110);plt.close(fig)
    else:
        fig,axs=plt.subplots(2,4,figsize=(13,7),constrained_layout=True)
        for i,c in enumerate(['alps','california']):
            for j,(m,k) in enumerate((m,k) for m in MODES for k in ['supplied','zero']):
                r=bykey[c,k,m,None,7919]
                show(axs[i,j],np.load(out/r['file'])['projected40_metres'][0,0,16:48,16:48],c+'\n'+m+' / '+k+' plan',40320.)
        fig.suptitle('Does each renderer use its plan? Matching noise; each 1290.24 km\nZero encoded plan removes information; it is not a neutral Earth target.');legend(fig,axs)
        fig.savefig(out/'conditioning-check.png',dpi=110);plt.close(fig)
    text='<!doctype html><html><head><meta charset="utf-8"><title>Spatial supervision comparison</title><style>body{font:16px system-ui;background:#f5f5f5;margin:24px}img{width:100%;max-width:1650px}p{max-width:1150px;line-height:1.5}details{margin:25px 0}</style></head><body><h1>Matched categorical / continuous terrain supervision</h1>'
    if smoke:text+='<p><strong>EXECUTION SMOKE ONLY: 2–4 training updates. These are not terrain-quality results.</strong></p>'
    text+=f'<p>This review uses <strong>{kind} plans</strong>. Supplied plans and feasible heights are reference information, not autonomous successes. Same initialization, paired update budgets and model sizes; fixed physical scales, metre colours and NW 5x shading. Primary seeds were declared before inspection.</p><h2>Raw meso, both fine seeds</h2><img src="fine-comparison.png">'
    for rs in RENDER_SEEDS:text+=f'<details><summary>Wider guides, renderer seed {rs}</summary><img loading="lazy" src="guides-r{rs}.png"></details>'
    if autonomous:
        for c in cases:text+=f'<details><summary>{c}: all plan seeds</summary><img loading="lazy" src="{c}-all-plan-seeds.png"></details>'
    else:text+='<details><summary>Supplied versus zero plan: conditioning check</summary><img loading="lazy" src="conditioning-check.png"></details>'
    (out/'index.html').write_text(text+'</body></html>',encoding='utf-8')


@torch.no_grad()
def run(args,out):
    started=time.perf_counter();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    models,norm,codec,ck=load_pair(out/'renderer-pair.pt','renderer','cuda')
    if ck['source_sha256']!=SOURCE_SHA or (ck['step']<10 and not args.smoke):raise ValueError('Wrong or smoke-only renderer')
    planners=None
    if args.planner:
        planners,pnorm,pcodec,pc=load_pair(out/'planner-pair.pt','planner','cuda')
        for k in ['source_sha256','dataset_id','stats','plan_stats','normalization_source','config','spec']:
            if ck[k]!=pc[k]:raise ValueError('Unmatched planner/renderer identity: '+k)
        if pc['step']<10 and not args.smoke:raise ValueError('Smoke planner needs --smoke')
    previous=json.loads((PREVIOUS/'settings.json').read_text());source=Path(previous['source_review'])
    if digest(source/'meso-checkpoint.pt')!=previous['downstream_hashes']['meso']:raise ValueError('Frozen meso changed')
    meso,mnorm,_=load_checkpoint(source/'meso-checkpoint.pt','cuda')
    settings={'renderer_sha256':digest(out/'renderer-pair.pt'),'renderer_updates':ck['step'],'planner_sha256':digest(out/'planner-pair.pt') if planners else None,
        'planner_updates':pc['step'] if planners else None,'source_sha256':SOURCE_SHA,'dataset_id':ck['dataset_id'],
        'meso_sha256':previous['downstream_hashes']['meso'],'cases':CASES+(['generated-60'] if planners else []),'renderer_seeds':RENDER_SEEDS,
        'plan_seeds':PLAN_SEEDS if planners else [],'meso_seeds':MESO_SEEDS,
        'fine_subset':'Alps/California and (when autonomous) generated-60, renderer 7919, first plan seed if autonomous; both meso seeds, declared before substantive inspection.',
        'smoke_only':args.smoke,'max_seconds':args.max_seconds,'sea_level_m':0.,'selection_changed':False,'planet_generated':False}
    atomic_json(out/'settings.json',settings);rows=[]
    for c in settings['cases']:
        source_input=source/f'{c}-s42-joint.npz'
        d=dict(np.load(source_input));coarse=torch.from_numpy(d['coarse_encoded']).cuda();base=norm.condition(coarse)[0]
        settings.setdefault('input_sha256',{})[c]={'guide_source':digest(source_input)}
        ref_path=source/f'{c}-s42-feasible-target-reference.npz';teacher={}
        if ref_path.exists():
            ref=dict(np.load(ref_path));np.testing.assert_array_equal(ref['coarse_encoded'],d['coarse_encoded'])
            settings['input_sha256'][c]['reference_source']=digest(ref_path)
            np.savez_compressed(out/f'{c}-reference.npz',**ref)
            z=torch.from_numpy(ref['projected40_metres']).cuda();teacher={m:codec.encode(z,base,m) for m in MODES}
            if c in ['alps','california']:
                for ms in MESO_SEEDS:
                    old=dict(np.load(source/f'{c}-s42-feasible-target-reference-m{ms}.npz'))
                    np.savez_compressed(out/f'{c}-reference-m{ms}.npz',raw_meso_metres=old['raw_meso_metres'])
        jobs=[('supplied',None,teacher)] if teacher else [];generated_raw={}
        if planners:
            for ps in PLAN_SEEDS:
                raw_plans={m:sample_planner(planners[m],pnorm,coarse,ps,32,'cuda') for m in MODES}
                for m in MODES:generated_raw[ps,m]=raw_plans[m]
                plans={m:codec.canonical_generated(raw_plans[m],base,m) for m in MODES}
                jobs.append(('autonomous',ps,plans))
        elif c in ['alps','california']:jobs.append(('zero',None,{m:torch.zeros_like(teacher[m]) for m in MODES}))
        atomic_json(out/'settings.json',settings)
        for kind,ps,plans in jobs:
            for rs in RENDER_SEEDS:
                if time.perf_counter()-started>args.max_seconds-15:
                    atomic_json(out/'results.json',{'status':'budget limited','records':rows,'elapsed_seconds':time.perf_counter()-started})
                    print('Budget reached; partial arrays saved. No follow-on job.',flush=True);return 2
                for m in MODES:
                    z,_,raw=sample_renderer(models[m],norm,coarse,plans[m],rs,32,'cuda');check=assert_constraints(z,base,16,40320.)
                    guide,_,base23=handoff(z,base,upsample_condition(coarse,7),coarse);assert_constraints(guide,base23,28,23040.)
                    stem=f'{c}-{kind}-p{ps}-{m}-r{rs}';filename=stem+'.npz'
                    saved={'coarse_encoded':coarse.cpu().numpy(),'baseline40_metres':base.cpu().numpy(),'encoded_plan':plans[m].cpu().numpy(),
                        'raw_generated_encoded_plan':generated_raw[ps,m].cpu().numpy() if kind=='autonomous' else plans[m].cpu().numpy(),
                        'raw40_metres':raw.cpu().numpy(),'projected40_metres':z.cpu().numpy(),'direct23_metres':guide.cpu().numpy()}
                    if not all(np.isfinite(v).all() for v in saved.values()):raise ValueError('Nonfinite review field')
                    np.savez_compressed(out/filename,**saved)
                    row={'case':c,'kind':kind,'mode':m,'plan_seed':ps,'renderer_seed':rs,'file':filename,'constraints':check,'meso':[]}
                    if c in ['alps','california','generated-60'] and rs==7919 and kind!='zero' and ps in [None,42]:
                        cond=encode(guide[...,47:65,47:65]).float()
                        for ms in MESO_SEEDS:
                            field=decode(sample_patches(meso,mnorm,cond,32,ms,'cuda')[0]).cpu().numpy()
                            if not np.isfinite(field).all():raise ValueError('Nonfinite meso')
                            name=stem+f'-m{ms}.npz';np.savez_compressed(out/name,raw_meso_metres=field,meso_input_encoded=cond.cpu().numpy())
                            row['meso'].append({'seed':ms,'file':name})
                    rows.append(row)
            atomic_json(out/'results.json',{'status':'in progress','records':rows})
        print(f'{c} reviewed, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    atomic_json(out/'results.json',{'status':'complete','records':rows,'sampling_seconds':time.perf_counter()-started,'visual_review_required':True})
    figures(out,rows,bool(planners),args.smoke);print(f'Review complete in {time.perf_counter()-started:.1f}s: {out/"index.html"}',flush=True);return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--renderer',default=str(OUT_ROOT/'renderer-paired-1000/latest.pt'))
    p.add_argument('--planner');p.add_argument('--out');p.add_argument('--max-seconds',type=float,default=120.);p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv);out=Path(args.out).resolve() if args.out else OUT_ROOT/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh review folder')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','review'):
        for task,path in [('renderer',args.renderer),('planner',args.planner)]:
            if path:
                with training_lock(Path(path).resolve().parent):shutil.copyfile(path,out/(task+'-pair.pt'))
        return run(args,out)


if __name__=='__main__':raise SystemExit(main())
