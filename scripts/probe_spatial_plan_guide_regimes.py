"""Short training/holdout and clean/degraded guide diagnostic; no learning."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.terrain_structure import StructureData,DEFAULT_DATA,decode,target_for
from planet_training.adaptation import degrade_guides,STRUCTURE_RECIPE
from planet_training.terrain_spatial_plan import MODES,load_pair,sample_planner,sample_renderer,continuous_fields
from planet_training.data import atomic_json
from scripts.probe_joint_plan_influence import digest
from scripts.probe_structure_stage_roles import assert_constraints
from scripts.diagnose_terrain_structure import image,legend
from scripts.structure_job import console_log

ROOT=Path(__file__).resolve().parents[1]/'outputs/structure/spatial-supervision-v1'


@torch.no_grad()
def run(out,review,max_seconds):
    started=time.perf_counter();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    models,norm,codec,ck=load_pair(review/'planner-pair.pt','planner','cuda')
    renderers,rnorm,rc,renderer_ck=load_pair(review/'renderer-pair.pt','renderer','cuda')
    if ck['plan_stats']!=renderer_ck['plan_stats']:raise ValueError('Unmatched codec')
    data=StructureData(DEFAULT_DATA)
    selected=[]
    for split in ['train','val']:
        indices=np.asarray(ck['normalization_source']['indices']) if split=='train' else np.arange(data.meta['val_count'])
        raw,c,_=data.batch(0,split,indices,'cuda');base=norm.condition(c)[0]
        fields=continuous_fields(raw,base)
        scores=[]
        for n,i in enumerate(indices):
            active=base[n,0,16:48,16:48]>200
            if float(active.float().mean())<.5:continue
            score=float(fields[n,3,16:48,16:48][active].mean())
            scores.append((score,int(i)))
        scores.sort()
        if len(scores)<2:raise ValueError('Not enough eligible diagnostic inputs')
        for q in [.5,.9]:
            score,index=scores[round(q*(len(scores)-1))]
            selected.append({'split':split,'index':index,'quantile':q,'selection_relief_mean_m':score,'eligible_count':len(scores)})
    settings={'planner_updates':ck['step'],'renderer_updates':renderer_ck['step'],'planner_sha256':digest(review/'planner-pair.pt'),
        'renderer_sha256':digest(review/'renderer-pair.pt'),'cases':selected,'seeds':[42,314],'renderer_seed':7919,
        'selection':'Before generation: original-guide central permitted fraction >=.5; rank mean 80.64-km land-only RMS relief in raw modern target; choose median and q90 separately per split. Train pool is 256 normalization indices, val pool all 55 monitored holdouts.',
        'limits':'Four target-stratified diagnostics, not a train/val generalization estimate or world survey. Training indices were used for fitting; val indices monitored. Guide changes also change permission and feasible targets; not a pure conditioning-channel ablation. No meso or full chain in this probe.',
        'sea_level_m':0.,'training':False,'selected_changed':False,'max_seconds':max_seconds}
    atomic_json(out/'settings.json',settings);rows=[];saved={}
    for case in selected:
        label=f'{case["split"]}-{case["index"]}-q{int(case["quantile"]*100)}'
        raw,c,ctx=data.batch(0,case['split'],np.asarray([case['index']]),'cuda')
        degraded=degrade_guides(ctx,c,torch.Generator(device='cuda').manual_seed(2026),sigma_cells=1.,clean_fraction=0.,recipe=STRUCTURE_RECIPE)
        for regime,coarse in [('clean',c),('degraded',degraded)]:
            target,base=target_for(raw,coarse);target=decode(target)
            saved[label,regime,'target',None]=target.cpu().numpy()
            np.savez_compressed(out/f'{label}-{regime}-inputs.npz',coarse_encoded=coarse.cpu().numpy(),raw_modern40_metres=raw.cpu().numpy(),feasible40_metres=target.cpu().numpy(),baseline40_metres=base.cpu().numpy())
            for mode in MODES:
                p=codec.encode(target,base,mode)
                z,_,raw_height=sample_renderer(renderers[mode],rnorm,coarse,p,7919,32,'cuda')
                saved[label,regime,mode+'-supplied',None]=z.cpu().numpy()
                np.savez_compressed(out/f'{label}-{regime}-{mode}-supplied.npz',projected40_metres=z.cpu().numpy(),encoded_plan=p.cpu().numpy())
                rows.append({'case':label,'split':case['split'],'regime':regime,'mode':mode,'kind':'supplied','seed':None})
            for seed in [42,314]:
                for mode in MODES:
                    if time.perf_counter()-started>max_seconds-12:
                        atomic_json(out/'results.json',{'status':'budget limited','records':rows});return 2
                    p=codec.canonical_generated(sample_planner(models[mode],norm,coarse,seed,32,'cuda'),base,mode)
                    z,_,raw_height=sample_renderer(renderers[mode],rnorm,coarse,p,7919,32,'cuda')
                    checks=assert_constraints(z,base,16,40320.)
                    if not torch.isfinite(z).all():raise ValueError('Nonfinite diagnostic')
                    saved[label,regime,mode,seed]=z.cpu().numpy()
                    np.savez_compressed(out/f'{label}-{regime}-{mode}-s{seed}.npz',projected40_metres=z.cpu().numpy(),raw40_metres=raw_height.cpu().numpy(),encoded_plan=p.cpu().numpy())
                    rows.append({'case':label,'split':case['split'],'regime':regime,'mode':mode,'kind':'autonomous','seed':seed,'constraints':checks})
        print(label+f' completed in {time.perf_counter()-started:.1f}s',flush=True)
        atomic_json(out/'results.json',{'status':'in progress','records':rows})
    for seed in [42,314]:
        fig,axs=plt.subplots(4,6,figsize=(17,12.8),constrained_layout=True)
        for i,case in enumerate(selected):
            label=f'{case["split"]}-{case["index"]}-q{int(case["quantile"]*100)}'
            for j,(regime,mode) in enumerate((r,m) for r in ['clean','degraded'] for m in ['target',*MODES]):
                z=saved[label,regime,mode,None if mode=='target' else seed][0,0,16:48,16:48]
                ax=axs[i,j]
                if mode=='target' and seed!=42:
                    ax.text(.5,.5,'Same feasible reference\nshown with first seed',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
                else:ax.imshow(image(z,40320.),extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest');ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
                ax.set_title(label+'\n'+regime+' / '+('SUPPLIED feasible height' if mode=='target' else mode),fontsize=9)
        fig.suptitle(f'Guide-regime diagnostic; planner {ck["step"]:,}, seed {seed}; same frozen renderer seed 7919\nEach 1290.24 km, stored local orientations, fixed elevation colours and NW 5x shade. Targets are supplied references, not generation.',fontsize=11)
        legend(fig,axs);fig.savefig(out/f'guide-regimes-s{seed}.png',dpi=110);plt.close(fig)
    # Supplied-plan capability is separate from autonomous guide-regime comparisons.
    fig,axs=plt.subplots(4,4,figsize=(13,12.8),constrained_layout=True)
    for i,case in enumerate(selected):
        label=f'{case["split"]}-{case["index"]}-q{int(case["quantile"]*100)}'
        for j,(regime,mode) in enumerate((r,m) for r in ['clean','degraded'] for m in MODES):
            z=saved[label,regime,mode+'-supplied',None][0,0,16:48,16:48];ax=axs[i,j]
            ax.imshow(image(z,40320.),extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest');ax.set_title(label+'\n'+regime+' / supplied '+mode+' plan',fontsize=9)
            ax.set_xlabel('km');ax.set_ylabel('km')
    fig.suptitle('SUPPLIED-plan renderer controls on these same inputs; each 1290.24 km\nThese are not autonomous planner successes; fixed elevation colours and shading.',fontsize=11)
    legend(fig,axs);fig.savefig(out/'supplied-controls.png',dpi=110);plt.close(fig)
    atomic_json(out/'results.json',{'status':'complete','records':rows,'elapsed_seconds':time.perf_counter()-started})
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px;background:#f5f5f5}img{width:100%;max-width:1800px}details{margin:25px 0}p{max-width:1100px;line-height:1.5}</style></head><body><h1>Guide regime and training / monitored holdout diagnostics</h1><p>Four target-stratified examples selected before generation: median/high relief in training and monitored holdout pools. Not a world sample or a statistical generalization estimate. Same model and noise per comparison. Changing guide also changes its permission and feasible reference. Each map covers 1,290.24 km; stored local orientations.</p><img src="guide-regimes-s42.png"><details><summary>Second planner seed</summary><img src="guide-regimes-s314.png"></details><details><summary>Supplied-plan renderer capability on these same inputs</summary><img src="supplied-controls.png"></details></body></html>',encoding='utf-8')
    print(f'Probe complete in {time.perf_counter()-started:.1f}s: {out}',flush=True);return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',default=str(ROOT/'autonomous-review-5000'));p.add_argument('--out',default=str(ROOT/'guide-regimes-5000'));p.add_argument('--max-seconds',type=float,default=110.)
    a=p.parse_args();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output folder')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(ROOT/'job-logs','guide-regime-probe'):return run(out,Path(a.review).resolve(),a.max_seconds)


if __name__=='__main__':raise SystemExit(main())
