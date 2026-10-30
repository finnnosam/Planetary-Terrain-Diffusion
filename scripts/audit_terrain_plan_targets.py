"""Audit plan supervision and bounded frozen-renderer interventions; no learning."""
import argparse
from collections import deque
from datetime import datetime
import html
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from planet_training.data import atomic_json
from planet_training.terrain_structure import blur, decode, handoff, encode
from planet_training.terrain_plan_oracle import plan_for, load, sample
from planet_training.terrain_joint_structure import interpret_plan
from toy_meso.model import upsample_condition
from toy_meso.common import load_checkpoint, sample_patches
from scripts.probe_joint_plan_influence import digest
from scripts.probe_structure_stage_roles import assert_constraints
from scripts.diagnose_terrain_structure import image, legend
from scripts.structure_job import console_log

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'outputs/structure/plan-first-v1/review-20261014-bounded-v1'
OUT_ROOT=ROOT/'outputs/structure/target-design-v1'
CASES=['alps','california','plains','holdout-rugged','holdout-quiet']
METHODS=['reference','no-axis','coarse80','coarse160','shuffle','no-plan']
RENDER_SEEDS=[7919,104729]
MESO_SEEDS=[113,997]
LABELS={'reference':'SUPPLIED reference plan','no-axis':'Same classes; no directions',
        'coarse80':'80.64 km majority plan','coarse160':'161.28 km majority plan',
        'shuffle':'Shuffled reference cells','no-plan':'No supplied plan'}


def intervene(plan,base,method):
    """Transform labels only; never edit elevation or the permission baseline."""
    p=plan.clone();active=base>200
    if method=='reference':return p
    if method=='no-plan':return torch.zeros_like(p)
    if method=='no-axis':p[:,2:]=0;return p
    if method=='shuffle':
        # Preserve joint four-channel tuples AND their permission band histogram.
        g=torch.Generator(device=p.device).manual_seed(5501)
        bands=[(base>200)&(base<=400),(base>400)&(base<=600),base>600]
        for b in range(len(p)):
            flat=p[b].reshape(4,-1)
            for band in bands:
                idx=torch.where(band[b,0].flatten())[0]
                flat[:,idx]=flat[:,idx[torch.randperm(len(idx),generator=g,device=p.device)]]
        return p
    factor={'coarse80':2,'coarse160':4}.get(method)
    if factor is None:raise ValueError(method)
    c=p[:,:1];counts=[F.avg_pool2d((c==v).float(),factor) for v in [-1,0,1]]
    cat=(counts[2]>torch.maximum(counts[0],counts[1])).float()-(counts[0]>torch.maximum(counts[1],counts[2])).float()
    quiet=(F.avg_pool2d(p[:,1:2],factor)>.5).float()
    axes=torch.zeros((len(p),2,*cat.shape[-2:]),device=p.device)
    for v in [-1,1]:
        a=F.avg_pool2d(p[:,2:]*(c==v),factor)
        a=a/a.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
        axes+=a*(cat==v)
    return F.interpolate(torch.cat((cat,quiet,axes),1),size=base.shape[-2:],mode='nearest')*active


def objects(mask):
    """Four-connected components, fixed grid; descriptives, not a realism score."""
    mask=np.asarray(mask,dtype=bool);seen=np.zeros_like(mask);sizes=[];elong=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        queue=deque([(y,x)]);seen[y,x]=True;coords=[]
        while queue:
            a,b=queue.popleft();coords.append((a,b))
            for c,d in [(a-1,b),(a+1,b),(a,b-1),(a,b+1)]:
                if 0<=c<mask.shape[0] and 0<=d<mask.shape[1] and mask[c,d] and not seen[c,d]:
                    seen[c,d]=True;queue.append((c,d))
        sizes.append(len(coords))
        if len(coords)>=4:
            eig=np.linalg.eigvalsh(np.cov(np.asarray(coords).T))
            elong.append(float(np.sqrt((eig[1]+1/12)/(eig[0]+1/12))))
    n=int(mask.sum())
    return {'cells':n,'components':len(sizes),'largest_fraction':max(sizes)/n if n else None,
            'single_cell_fraction':sum(v==1 for v in sizes)/n if n else None,
            'elongation_median_for_4plus':float(np.median(elong)) if elong else None}


def plan_stats(plan,base,raw=None):
    # Interior crop excludes label/filter boundary effects; 32 cells = 1290.24 km.
    p=plan[0,:,16:48,16:48].cpu().numpy();b=base[0,0,16:48,16:48].cpu().numpy();a=b>200
    result={'permitted_fraction':float(a.mean()),'protected_land_fraction':float(((b>0)&~a).sum()/max(1,(b>0).sum()))}
    for name,mask in [('high',(p[0]>.5)&a),('low',(p[0]<-.5)&a),('quiet',(p[1]>.5)&a)]:
        result[name]=objects(mask);result[name]['fraction_of_permitted']=float(mask.sum()/max(1,a.sum()))
    if raw is not None:
        r=raw[0,:,16:48,16:48].cpu().numpy()
        result['raw_category_near_threshold_fraction']=float(((np.abs(np.abs(r[0])-.5)<.15)&a).sum()/max(1,a.sum()))
        corridor=(np.abs(p[0])>.5)&a
        result['raw_axis_length_median_on_corridors']=float(np.median(np.hypot(r[2],r[3])[corridor])) if corridor.any() else None
        result['raw_outside_label_range_fraction']=float((np.abs(r[:,a])>1.00001).mean()) if a.any() else None
    return result


def physical_labels(z,base):
    """Diagnostic alternatives, not a new selected target definition."""
    mean=blur(z,2);sm=blur(z,.75)
    dy,dx=torch.gradient(sm,spacing=(40.32,40.32),dim=(-2,-1))
    xx,yy,xy=[blur(v,1) for v in [dx.square(),dy.square(),dx*dy]]
    coherence=((xx-yy).square()+4*xy.square()).sqrt()/(xx+yy).clamp_min(1e-8)
    return {'position80_m':z-mean,'position160_m':z-blur(z,4),
        'position320_m':z-blur(z,8),'roughness80_m':(blur(z.square(),2)-mean.square()).clamp_min(0).sqrt(),
        'axis_coherence':coherence,'baseline_correction80_m':blur(z-base,2)}


def continuous_candidate(z,base,padding='reflect'):
    """Untrained candidate labels: fixed guide-land support, no elevation edits."""
    with torch.autocast(device_type=z.device.type,enabled=False):
        z=z.float();mask=(base>0).float();fields=[]
        for sigma in [2.,4.,8.]:
            def filt(value):
                if padding=='reflect':return blur(value,sigma)
                if padding!='replicate':raise ValueError(padding)
                radius=int(np.ceil(3*sigma))
                return blur(F.pad(value,(radius,radius,radius,radius),mode='replicate'),sigma)[...,radius:-radius,radius:-radius]
            denom=filt(mask).clamp_min(1e-6);mean=filt(z*mask)/denom
            fields.append((z-mean)*mask)
            if sigma==2.:relief=((filt(z.square()*mask)/denom-mean.square()).clamp_min(0).sqrt())*mask
        return torch.cat((*fields,relief),1)


def audit(previous,out):
    settings=json.loads((previous/'settings.json').read_text());source=Path(settings['source_review'])
    audit_path=ROOT/'outputs/terrain-stage-roles/paired-v1/target-audit-arrays.npz'
    a=dict(np.load(audit_path));z=torch.from_numpy(a['constrained40_metres']);base=torch.from_numpy(a['baseline40_metres'])
    p=plan_for(z,base);fields=physical_labels(z,base);records=[]
    for i in range(len(z)):
        row={'index':i,**plan_stats(p[i:i+1],base[i:i+1])}
        region=(slice(i,i+1),slice(None),slice(16,48),slice(16,48));active=base[region]>200
        corridors=(p[i:i+1,:1,16:48,16:48].abs()>.5)&active
        conf=fields['axis_coherence'][region]
        row['axis_low_coherence_fraction_below_point2']=float(((conf<.2)&corridors).sum()/corridors.sum().clamp_min(1))
        row['corridor_cells_for_coherence']=int(corridors.sum())
        row['quiet_protected_land_cells']=int(((base[region]>0)&~active).sum())
        row['physical_fields']={k:{'rms':float(v[region].square().mean().sqrt()),'p10_p50_p90':[float(q) for q in torch.quantile(v[region].flatten(),torch.tensor([.1,.5,.9]))]} for k,v in fields.items()}
        records.append(row)
    generated=[]
    for case in [c['label'] for c in settings['cases']]:
        d=dict(np.load(source/f'{case}-s42-joint.npz'));b=torch.from_numpy(d['baseline40_metres'])
        for seed in settings['plan_seeds']:
            for mode in ['source','coupled','planner']:
                d=dict(np.load(previous/f'{case}-p{seed}-{mode}-plan.npz'))
                generated.append({'case':case,'seed':seed,'mode':mode,**plan_stats(torch.from_numpy(d['interpreted_plan']),b,torch.from_numpy(d['raw_plan']))})
    atomic_json(out/'target-audit.json',{'target_source':str(audit_path),'target_source_sha256':digest(audit_path),
        'crop':'Central 32x32 at 40.32 km = 1290.24 km; all 55 monitored geographic holdouts, not an unbiased world sample.',
        'components':'4-connected at fixed 40.32 km; label connectivity, not mountain-system quality. Local low category does not describe every lowland.',
        'holdouts':records,'generated':generated})
    return settings,source


def figures(out,source,rendered,plans,fine):
    cm=ListedColormap(['#4a8bb1','#e5e5df','#a65f3d'])
    for case in CASES:
        fig,axs=plt.subplots(3,6,figsize=(17,9.8),constrained_layout=True)
        for j,method in enumerate(METHODS):
            p=plans[case,method][0].cpu().numpy();plane=p[0,16:48,16:48]
            axs[0,j].imshow(plane,cmap=cm,vmin=-1,vmax=1,extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest')
            yy,xx=np.indices((32,32));theta=.5*np.arctan2(p[3,16:48,16:48],p[2,16:48,16:48]);sel=(yy%3==0)&(xx%3==0)&(np.abs(plane)>.5)&(np.hypot(p[2,16:48,16:48],p[3,16:48,16:48])>.1)
            axs[0,j].quiver((xx[sel]+.5-16)*40.32,(16-yy[sel]-.5)*40.32,65*np.cos(theta[sel]),65*np.sin(theta[sel]),angles='xy',scale_units='xy',scale=1,pivot='middle',headlength=0,headaxislength=0,headwidth=0,color='#222',width=.005)
            axs[0,j].set_title(LABELS[method],fontsize=9)
            for row,rs in enumerate(RENDER_SEEDS,1):
                field=rendered[case,method,rs]['projected40_metres'][0,0,16:48,16:48]
                axs[row,j].imshow(image(field,40320.),extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest')
                axs[row,j].set_title(f'Renderer seed {rs}',fontsize=8)
        for ax in axs.flat:ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
        fig.suptitle(f'{case}: supplied-plan interventions, same frozen renderer / guide\nEach map 1290.24 km. Top: blue local low / brown high; axes black. Below: two noise seeds, fixed elevation colours / NW 5x shade.')
        legend(fig,axs[1:]);fig.savefig(out/f'{case}-interventions.png',dpi=110);plt.close(fig)
    fig,axs=plt.subplots(4,6,figsize=(17,11.8),constrained_layout=True)
    for i,(case,ms) in enumerate((c,s) for c in ['alps','california'] for s in MESO_SEEDS):
        seen={}
        for j,m in enumerate(METHODS):
            z=fine[case,m,ms];key=z.tobytes()
            if key in seen:axs[i,j].text(.5,.5,'Identical to '+seen[key],ha='center',va='center',transform=axs[i,j].transAxes)
            else:axs[i,j].imshow(image(z[0,0],2880.),extent=(-184.32,184.32,-184.32,184.32),interpolation='nearest');seen[key]=m
            axs[i,j].set_title(f'{case}, meso {ms}\n'+LABELS[m],fontsize=9)
    for ax in axs.flat:ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
    fig.suptitle('Frozen meso after the plan intervention; each 368.64 km\nSame renderer noise 7919, no sampled old-intermediate detail. Supplied references are not autonomous generation.')
    legend(fig,axs);fig.savefig(out/'fine-interventions.png',dpi=110);plt.close(fig)
    # Plans the planner actually generated, separate from supplied-plan experiments.
    fig,axs=plt.subplots(2,4,figsize=(12,6.8),constrained_layout=True)
    for i,case in enumerate(['alps','california']):
        ref=dict(np.load(source/f'{case}-s42-feasible-target-reference.npz'))
        b=torch.from_numpy(ref['baseline40_metres'])
        for j,(name,path) in enumerate([('Supplied feasible reference',None),('Original joint','source'),('Continued joint','coupled'),('Dedicated planner','planner')]):
            p=ref['interpreted_plan'] if path is None else np.load(PREVIOUS/f'{case}-p42-{path}-plan.npz')['interpreted_plan']
            axs[i,j].imshow(p[0,0,16:48,16:48],cmap=cm,vmin=-1,vmax=1,extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest')
            # Protected land / water are marked explicitly, not counted as neutral predicted plains.
            lock=(b.numpy()[0,0,16:48,16:48]<=200)
            overlay=np.zeros((32,32,4));overlay[lock]=[.15,.15,.15,.45]
            axs[i,j].imshow(overlay,extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest')
            axs[i,j].set_title(case+'\n'+name,fontsize=10);axs[i,j].set_xlabel('km');axs[i,j].set_ylabel('km')
    fig.suptitle('What the learned plan contains, seed 42, each 1290.24 km\nBlue local low, brown high; grey overlay = protected (including sea), not predicted plains.')
    fig.savefig(out/'generated-plans.png',dpi=115);plt.close(fig)
    page='<!doctype html><html><head><meta charset="utf-8"><title>Terrain learning target audit</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5}img{width:100%;max-width:1900px}p{max-width:1150px;line-height:1.5}details{margin:25px 0}</style></head><body><h1>What should terrain planning learn?</h1><p>No training. These tests modify supplied Earth-derived plans while keeping the frozen height model, guide and noise fixed. This measures which plan information matters; it does not demonstrate autonomous generation. The shuffled plan preserves joint label counts within permission bands but destroys spatial organization. Coarsening can change label counts as well as detail. All maps use fixed physical scales; elevation colours and shading are identical.</p><h2>Generated plan comparison</h2><img src="generated-plans.png"><h2>Effect through frozen meso, two seeds</h2><img src="fine-interventions.png">'
    for c in CASES:page+=f'<details><summary>{html.escape(c)}: wider plans and rendered terrain</summary><img loading="lazy" src="{c}-interventions.png"></details>'
    (out/'index.html').write_text(page+'</body></html>',encoding='utf-8')


def summarize(out):
    """Reproducible saved-array checks; no model sampling."""
    a=dict(np.load(ROOT/'outputs/terrain-stage-roles/paired-v1/target-audit-arrays.npz'))
    z=torch.from_numpy(a['constrained40_metres']);b=torch.from_numpy(a['baseline40_metres'])
    fields=physical_labels(z,b);p=plan_for(z,b)
    region=(slice(None),slice(None),slice(16,48),slice(16,48))
    active=b[region]>200;cat=p[:,:1,16:48,16:48];pos=fields['position80_m'][region]
    neutral=(cat==0)&active
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'target-audit.json').read_text())
    h=audit['holdouts']
    summary={'structural_outputs':len(r['records']),'fine_outputs':sum(len(x['meso']) for x in r['records']),
        'monitored_holdouts':len(h),'generated_plan_audits':len(audit['generated']),'interventions':{},
        'neutral_permitted_cells_with_abs_160km_position_over150m_fraction':float(((fields['position160_m'][region].abs()>150)&neutral).sum()/neutral.sum()),
        'protected_land_fraction_pooled_holdout_crop':float(((b[region]>0)&~active).sum()/(b[region]>0).sum()),
        'protected_land_fraction_median_holdout':float(np.median([x['protected_land_fraction'] for x in h])),
        'low_axis_coherence_fraction_corridor_cell_weighted':sum(x['axis_low_coherence_fraction_below_point2']*x['corridor_cells_for_coherence'] for x in h)/sum(x['corridor_cells_for_coherence'] for x in h),
        'amplitude80_p10_p50_p90_by_category':{str(c):np.percentile(pos[(cat==c)&active].numpy(),[10,50,90]).tolist() for c in [-1,0,1]},
        'checks':{'all_saved_npz_finite':True,'full_patch_shuffle_joint_tuple_identity':True,'frozen_hashes_unchanged':{}},
        'limits':'Regional diagnostics on monitored inputs, not world-frequency or realism scores.'}
    candidate=continuous_candidate(z,b);alternative=continuous_candidate(z,b,'replicate')
    edge=(candidate-alternative)[:,:,16:48,16:48]
    summary['candidate_land_support_padding_sensitivity']={
        'comparison':'Reflect versus replicate outer padding, same physical fields/guide-land mask. Sensitivity only, neither is a true-context target.',
        'rms_by_channel_m':[float(v) for v in edge.square().mean((0,2,3)).sqrt()],
        'p90_abs_by_channel_m':[float(torch.quantile(edge[:,i].abs().flatten(),.9)) for i in range(4)]}
    for case in CASES:
        summary['interventions'][case]={}
        for method in METHODS:
            rows=[x for x in r['records'] if x['case']==case and x['method']==method]
            fs=[q['change_from_supplied_reference_rms_m'] for x in rows for q in x['meso']]
            summary['interventions'][case][method]={
                'projected40_change_rms_m_mean_over_2_renderer_seeds':float(np.mean([x['change_from_supplied_reference_rms_m'] for x in rows])),
                'raw_meso_change_rms_m_mean_over_2_meso_seeds':float(np.mean(fs)) if fs else None,'plan_central_crop':rows[0]['plan']}
        ref=np.load(out/f'{case}-reference-plan.npz')['plan'];sh=np.load(out/f'{case}-shuffle-plan.npz')['plan']
        base=np.load(out/f'{case}-reference.npz')['baseline40_metres']
        for mask in [(base>200)&(base<=400),(base>400)&(base<=600),base>600]:
            v=ref[0][:,mask[0,0]].T;w=sh[0][:,mask[0,0]].T
            np.testing.assert_array_equal(v[np.lexsort(v.T)],w[np.lexsort(w.T)])
    for path in out.glob('*.npz'):
        with np.load(path) as d:
            if not all(np.isfinite(d[k]).all() for k in d.files):raise ValueError(path)
    settings=json.loads((out/'settings.json').read_text())
    source=Path(json.loads((Path(settings['previous'])/'settings.json').read_text())['source_review'])
    for name in ['renderer','meso']:
        summary['checks']['frozen_hashes_unchanged'][name]=digest(source/f'{name}-checkpoint.pt')==settings['checkpoints'][name]
        if not summary['checks']['frozen_hashes_unchanged'][name]:raise ValueError('Frozen model changed')
    atomic_json(out/'measurements.json',summary)
    # Physical fields for the proposed four-channel comparison. Not model outputs.
    fig,axs=plt.subplots(2,4,figsize=(13,7.4),constrained_layout=True)
    for i,case in enumerate(['alps','california']):
        ref=dict(np.load(out/f'{case}-reference.npz'))
        proposed=continuous_candidate(torch.from_numpy(ref['projected40_metres']),torch.from_numpy(ref['baseline40_metres'])).numpy()
        np.savez_compressed(out/f'{case}-candidate-fields.npz',candidate_physical_channels=proposed)
        for j,(key,title) in enumerate([('position80_m','Relative height, sigma 80.64 km'),('position160_m','Relative height, sigma 161.28 km'),('position320_m','Relative height, sigma 322.56 km'),('roughness80_m','Local relief, sigma 80.64 km')]):
            field=proposed[0,j,16:48,16:48]
            im=axs[i,j].imshow(field,cmap='RdBu_r' if j<3 else 'viridis',vmin=-1000 if j<3 else 0,vmax=1000,extent=(-645.12,645.12,-645.12,645.12),interpolation='nearest')
            axs[i,j].set_title(case+'\n'+title,fontsize=9);axs[i,j].set_xlabel('km');axs[i,j].set_ylabel('km')
            fig.colorbar(im,ax=axs[i,j],orientation='horizontal',fraction=.04,pad=.12).set_label('metres (field values)',fontsize=8)
    fig.suptitle('Candidate supervision: averages restricted to guide land, same feasible targets\nEach 1290.24 km. Physical label fields, not elevations or generated samples. Display clipping only; broad-scale edges need context.')
    fig.savefig(out/'candidate-target-fields.png',dpi=115);plt.close(fig)
    page=(out/'index.html').read_text(encoding='utf-8')
    insert='<details><summary>Candidate continuous target fields (not trained)</summary><img loading="lazy" src="candidate-target-fields.png"></details>'
    if 'candidate-target-fields.png' not in page:(out/'index.html').write_text(page.replace('</body>',insert+'</body>'),encoding='utf-8')
    return summary


@torch.no_grad()
def run(args,out):
    started=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    settings,source=audit(PREVIOUS,out)
    renderer_path=source/'renderer-checkpoint.pt';meso_path=source/'meso-checkpoint.pt'
    for name,path in [('renderer',renderer_path),('meso',meso_path)]:
        if digest(path)!=settings['downstream_hashes'][name]:raise ValueError('Changed frozen model')
    renderer,norm,_=load(renderer_path,'cuda');meso,mnorm,_=load_checkpoint(meso_path,'cuda')
    provenance={'previous':str(PREVIOUS),'previous_settings_sha256':digest(PREVIOUS/'settings.json'),
        'checkpoints':settings['downstream_hashes'],'cases':CASES,'methods':METHODS,'renderer_seeds':RENDER_SEEDS,
        'fine_subset':'Alps and California, first renderer seed 7919, both meso seeds 113/997; declared before inspection.',
        'steps':32,'sea_level_m':0.,'training':False,'selected_changed':False,'max_seconds':args.max_seconds,
        'limits':'No-plan/coarsened/shuffled conditions may be out of the oracle renderer training distribution. A failed intervention is not proof a retrained representation cannot work.'}
    atomic_json(out/'settings.json',provenance);rows=[];rendered={};plans={};fine={}
    for case in CASES:
        ref=dict(np.load(source/f'{case}-s42-feasible-target-reference.npz'))
        coarse=torch.from_numpy(ref['coarse_encoded']).cuda();base=torch.from_numpy(ref['baseline40_metres']).cuda()
        reference=torch.from_numpy(ref['interpreted_plan']).cuda()
        np.savez_compressed(out/f'{case}-reference.npz',**ref,**{k:v.cpu().numpy() for k,v in physical_labels(torch.from_numpy(ref['projected40_metres']),torch.from_numpy(ref['baseline40_metres'])).items()})
        for method in METHODS:
            p=intervene(reference,base,method);plans[case,method]=p.cpu()
            np.savez_compressed(out/f'{case}-{method}-plan.npz',plan=p.cpu().numpy())
            for rs in RENDER_SEEDS:
                if time.perf_counter()-started>args.max_seconds-15:
                    atomic_json(out/'results.json',{'status':'budget limited','records':rows,'seconds':time.perf_counter()-started})
                    print('Budget reached. Saved completed interventions; no follow-on work.',flush=True);return 2
                if method=='reference':
                    d=dict(np.load(PREVIOUS/f'{case}-p42-reference-r{rs}.npz'));z=torch.from_numpy(d['projected40_metres']).cuda();raw=torch.from_numpy(d['raw40_metres']).cuda();guide=torch.from_numpy(d['direct23_metres']).cuda()
                else:
                    z,_,raw=sample(renderer,norm,coarse,p,32,rs,'cuda')
                    guide,_,_=handoff(z,base,upsample_condition(coarse,7),coarse)
                checks=assert_constraints(z,base,16,40320.)
                d={'projected40_metres':z.cpu().numpy(),'raw40_metres':raw.cpu().numpy(),'direct23_metres':guide.cpu().numpy()}
                if not all(np.isfinite(v).all() for v in d.values()):raise ValueError('Nonfinite output')
                rendered[case,method,rs]=d;np.savez_compressed(out/f'{case}-{method}-r{rs}.npz',**d)
                desired=rendered[case,'reference',rs]['projected40_metres'][0,0,16:48,16:48]
                delta=d['projected40_metres'][0,0,16:48,16:48]-desired
                row={'case':case,'method':method,'renderer_seed':rs,'cached':method=='reference','constraints':checks,
                    'plan':plan_stats(p,base),'terrain_plan':plan_stats(plan_for(z,base),base),
                    'change_from_supplied_reference_rms_m':float(np.sqrt(np.mean(delta**2))),'meso':[]}
                if case in ['alps','california'] and rs==7919:
                    cond=encode(guide[...,47:65,47:65]).float()
                    for ms in MESO_SEEDS:
                        if method=='reference':
                            cached=dict(np.load(PREVIOUS/f'{case}-p42-reference-r{rs}-m{ms}.npz'))
                            np.testing.assert_array_equal(cached['meso_input_encoded'],cond.cpu().numpy());field=cached['raw_meso_metres']
                        else:field=decode(sample_patches(meso,mnorm,cond,32,ms,'cuda')[0]).cpu().numpy()
                        if not np.isfinite(field).all():raise ValueError('Nonfinite meso')
                        fine[case,method,ms]=field;np.savez_compressed(out/f'{case}-{method}-m{ms}.npz',raw_meso_metres=field,meso_input_encoded=cond.cpu().numpy())
                        delta=field-fine[case,'reference',ms]
                        row['meso'].append({'seed':ms,'change_from_supplied_reference_rms_m':float(np.sqrt(np.mean(delta**2)))})
                rows.append(row)
                atomic_json(out/'results.json',{'status':'in progress','records':rows,'seconds':time.perf_counter()-started})
        print(f'{case}: all six interventions / both renderer seeds saved, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    atomic_json(out/'results.json',{'status':'complete','records':rows,'inference_and_audit_seconds':time.perf_counter()-started,'visual_review_required':True})
    figures(out,source,rendered,plans,fine)
    summarize(out)
    print(f'Completed without training in {time.perf_counter()-started:.1f}s: {out}',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out');p.add_argument('--max-seconds',type=float,default=120.)
    args=p.parse_args();out=Path(args.out).resolve() if args.out else OUT_ROOT/('audit-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output folder')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(OUT_ROOT/'job-logs','audit'):return run(args,out)


if __name__=='__main__':raise SystemExit(main())
