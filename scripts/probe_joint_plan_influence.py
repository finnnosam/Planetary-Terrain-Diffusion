"""Bounded frozen-model causal probes; no optimizer or checkpoint writes."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from planet_training.terrain_joint_structure import load, mask_plan, JOINT_SPEC
from planet_training.terrain_structure import decode, encode, project, blur
from toy_meso.model import sigma_schedule
from toy_meso.common import autocast
from scripts.structure_job import console_log
from scripts.diagnose_terrain_structure import image, legend

DEFAULT_REVIEW='outputs/structure/joint-structure-v1/review-20261014-143545-003794'
SIGMAS=[.05,.2,.5,1.,2.,5.,10.]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rms(x,mask):
    return float(x.masked_select(mask.expand_as(x)).float().square().mean().sqrt()) if mask.any() else None


def rotate_plan(p):
    r=torch.rot90(p,1,(-2,-1)).clone()
    r[:,2:4]*=-1  # axial cos(2 theta), sin(2 theta) under a 90-degree rotation
    return r


def states_from_anchor(clean,donor,noise,sigma,base):
    """Signal-only interventions retain the SAME height and plan noise fields."""
    original=mask_plan(clean+sigma*noise,base)
    zero=clean.clone();zero[:,1:]=0
    swapped=clean.clone();swapped[:,1:]=donor[:,1:]
    rotated=clean.clone();rotated[:,1:]=rotate_plan(clean[:,1:]);rotated=mask_plan(rotated,base)
    axes=clean.clone();axes[:,3:5]*=-1
    positive=original.clone();positive[:,:1]=clean[:,:1]+sigma*noise[:,:1].roll(7,-1)
    result={'same':original,'repeat':original.clone(),'remove_signal':mask_plan(zero+sigma*noise,base),
            'swap_signal':mask_plan(swapped+sigma*noise,base),'rotate_signal':mask_plan(rotated+sigma*noise,base),
            'rotate_axis_signal':mask_plan(axes+sigma*noise,base),'height_noise_control':positive}
    for name,x in result.items():
        if name!='height_noise_control':torch.testing.assert_close(x[:,:1],original[:,:1],atol=0,rtol=0)
        if name in ('remove_signal','swap_signal','rotate_signal','rotate_axis_signal'):
            altered={'remove_signal':zero,'swap_signal':swapped,'rotate_signal':rotated,'rotate_axis_signal':axes}[name]
            torch.testing.assert_close(x-altered,original-clean,atol=2e-6,rtol=1e-5)
    return result


@torch.no_grad()
def predict_many(model,states,sigma,cin):
    names=list(states)
    x=torch.cat([states[k].float() for k in names])
    # Float32 evaluation distinguishes weak coupling from AMP rounding.
    pred=model(x,torch.full((len(names),),float(sigma),device=x.device),cin.expand(len(names),-1,-1,-1)).float()
    return dict(zip(names,pred.split(1)))


def response_record(pred,x,reference,reference_x,base,up,norm):
    raw=decode(norm.target(pred[:,:1],up));raw0=decode(norm.target(reference[:,:1],up))
    z,z0=project(raw,base),project(raw0,base)
    active=base>200; full=base>600
    q=(torch.arange(64,device=base.device)+.5-32)*40.32
    centre=(q.abs()<599.04)[None,None,:,None]&(q.abs()<599.04)[None,None,None,:]
    residual_delta=pred[:,:1]-reference[:,:1]
    input_delta=x-reference_x
    in_plan=rms(input_delta[:,1:],active)
    return {'raw_height_response_active_rms_m':rms(raw-raw0,active),
            'projected_height_response_active_rms_m':rms(z-z0,active),
            'raw_height_response_over600_rms_m':rms(raw-raw0,full),
            'projected_height_response_central_active_rms_m':rms(z-z0,active&centre),
            'normalised_height_response_rms':rms(residual_delta,active),
            'normalised_plan_input_change_rms':in_plan,
            'normalised_height_input_change_rms':rms(input_delta[:,:1],active),
            'height_to_plan_response_gain':rms(residual_delta,active)/in_plan if in_plan else None}


def anchor(d,norm,device):
    coarse=torch.as_tensor(d['coarse_encoded'],device=device)
    base,up,cin=norm.condition(coarse)
    height=norm.residual(encode(torch.as_tensor(d['raw40_metres'],device=device)),up)
    plan=torch.as_tensor(d['generated_plan_raw'],device=device)*JOINT_SPEC['plan_scale']
    return mask_plan(torch.cat((height,plan),1),base),base,up,cin,coarse


def aggregate(records,group_keys,value):
    groups={}
    for row in records:
        v=row[value]
        if v is not None:groups.setdefault(tuple(row[k] for k in group_keys),[]).append(v)
    return [{**dict(zip(group_keys,k)),'count':len(v),'median':float(np.median(v)),
             'minimum':float(np.min(v)),'maximum':float(np.max(v))} for k,v in groups.items()]


def figures(out,records,visuals):
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True,sharey=True)
    for mode,ax in zip(['joint','control'],axs):
        for variant,label in [('remove_signal','Remove plan signal'),('swap_signal','Swap same-guide plan'),('rotate_signal','Rotate plan signal'),('height_noise_control','Change height noise')]:
            vals=[]
            for sigma in SIGMAS:
                data=[r['raw_height_response_active_rms_m'] for r in records if r['model']==mode and r['intervention']==variant and r['sigma']==sigma]
                vals.append(np.median(data))
            ax.plot(SIGMAS,vals,'o-',label=label)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(.1,200);ax.set_xlabel('Noise sigma');ax.set_ylabel('Median raw height response (m RMS)')
        ax.set_title(mode+' frozen denoiser');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Local response to interventions on re-noised saved joint endpoints\nIdentical height/noise/coarse guide except height-noise control; response is not terrain quality',fontsize=11)
    fig.savefig(out/'response-curves.png',dpi=140);plt.close(fig)
    if visuals:
        fig,axs=plt.subplots(1,len(visuals),figsize=(4.2*len(visuals),4.5),constrained_layout=True)
        for ax,(label,delta) in zip(np.atleast_1d(axs),visuals.items()):
            a=F.interpolate(torch.from_numpy(delta),size=(112,112),mode='bilinear',align_corners=False).numpy()[0,0,30:82,30:82]
            im=ax.imshow(a,cmap='RdBu_r',vmin=-100,vmax=100,extent=(-599.04,599.04,-599.04,599.04),interpolation='nearest')
            ax.set_title(label,fontsize=10);ax.set_xlabel('km');ax.set_ylabel('km')
        fig.colorbar(im,ax=np.atleast_1d(axs).tolist(),orientation='horizontal',fraction=.06,pad=.02,label='One-step predicted height CHANGE (m); fixed ±100 m display limits')
        fig.suptitle('California seed 42, sigma 0.5: float32 local denoiser response\nDifference maps, not new completed terrain; same height state and noise; 1,198.08 km each',fontsize=11)
        fig.savefig(out/'california-response-differences.png',dpi=140);plt.close(fig)


@torch.no_grad()
def run(args,out):
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    review=Path(args.review).resolve();settings=json.loads((review/'settings.json').read_text())
    paths={k:review/f'{k}-checkpoint.pt' for k in ['joint','control']}
    hashes={k:digest(p) for k,p in paths.items()}
    if any(hashes[k]!=settings['checkpoints'][k]['sha256'] for k in paths):raise ValueError('Frozen checkpoint identity changed')
    models={};norms={}
    for k,p in paths.items():models[k],norms[k],_=load(p,args.device)
    case_names=[r['label'] for r in settings['cases']];seeds=settings['seeds']
    archives={(c,s):dict(np.load(review/f'{c}-s{s}-joint.npz')) for c in case_names for s in seeds}
    records=[];trajectory=[];visuals={};done=0;timed_out=False
    metadata={'review':str(review),'checkpoint_hashes':hashes,'sigmas':SIGMAS,'seeds':seeds,'cases':case_names,
              'precision':'float32 diagnostic forwards; original AMP retained for trajectory integration',
              'noise_seed':8675309,'maximum_seconds':args.max_seconds,
              'anchor':'Saved joint raw height and continuous plan outputs; same inputs applied to joint and control. Nonzero plan signals are out of training distribution for the zero-target control.',
              'noise':'Gaussian noise added at prescribed sigmas. Donor is the next saved seed under the same coarse guide. Height-noise control uses a seven-cell circular shift of the same Gaussian height field.',
              'limits':'Local sensitivity, not terrain realism, conditional reconstruction, or a long-run training comparison. Incompatible height/plan combinations can be OOD.',
              'sea_level_m':0.,'training_updates':0,'selection_changed':False}
    atomic_json(out/'settings.json',metadata)
    for case in case_names:
        for si,seed in enumerate(seeds):
            if time.perf_counter()-start>args.max_seconds:timed_out=True;break
            clean,base,up,cin,_=anchor(archives[case,seed],norms['joint'],args.device)
            donor,*_=anchor(archives[case,seeds[(si+1)%len(seeds)]],norms['joint'],args.device)
            noise=torch.randn(clean.shape,device=args.device,generator=torch.Generator(device=args.device).manual_seed(8675309+seed))
            for sigma in SIGMAS:
                states=states_from_anchor(clean,donor,noise,sigma,base)
                for mode in ['joint','control']:
                    predictions=predict_many(models[mode],states,sigma,cin)
                    torch.testing.assert_close(predictions['same'],predictions['repeat'],atol=0,rtol=0)
                    for name in states:
                        if name=='same':continue
                        row={'phase':'renoised_endpoint','model':mode,'case':case,'seed':seed,'sigma':sigma,'intervention':name,
                             **response_record(predictions[name],states[name],predictions['same'],states['same'],base,up,norms[mode])}
                        records.append(row)
                        if case=='california' and seed==42 and sigma==.5 and mode=='joint' and name in ['remove_signal','swap_signal','rotate_signal']:
                            visuals[name]=(decode(norms[mode].target(predictions[name][:,:1],up))-decode(norms[mode].target(predictions['same'][:,:1],up))).cpu().numpy()
            done+=1
        atomic_json(out/'results.json',{'status':'in progress','settings':metadata,'endpoint_records':records,'trajectory_records':trajectory})
        print(f'Endpoint probes: {done}/{len(archives)} inputs; elapsed {time.perf_counter()-start:.1f}s',flush=True)
        if timed_out:break
    # Reconstruct real production-style Heun paths, and intervene locally without
    # feeding counterfactual predictions back into the baseline trajectory.
    for case in ['california','alps','holdout-rugged','generated-60']:
        if timed_out:break
        for seed in [42,314]:
            if time.perf_counter()-start>args.max_seconds:timed_out=True;break
            clean,base,up,cin,coarse=anchor(archives[case,seed],norms['joint'],args.device)
            model=models['joint'];norm=norms['joint'];g=torch.Generator(device=args.device).manual_seed(seed)
            ts=sigma_schedule(32,device=args.device)
            x=mask_plan(torch.randn(clean.shape,generator=g,device=args.device,dtype=torch.float64),base)*ts[0]
            def denoise(state,sigma):
                with autocast(args.device):return mask_plan(model(state.float(),sigma.float().expand(1),cin),base).double()
            for i in range(32):
                original_d=denoise(x,ts[i])
                if i in [0,8,16,24,30]:
                    same=x.float();clamped=same.clone();clamped[:,1:]=0
                    rotated=same.clone();rotated[:,1:]=rotate_plan(same[:,1:])
                    # Remove the current clean-plan estimate but retain x-D.
                    erased=same.clone();erased[:,1:]-=original_d[:,1:].float()
                    states={k:mask_plan(v,base) for k,v in [('same',same),('clamp_plan_state',clamped),('rotate_plan_state',rotated),('remove_estimated_signal',erased)]}
                    predictions=predict_many(model,states,float(ts[i]),cin)
                    for name in list(states)[1:]:
                        torch.testing.assert_close(states[name][:,:1],states['same'][:,:1],atol=0,rtol=0)
                        trajectory.append({'phase':'actual_trajectory','model':'joint','case':case,'seed':seed,'step_index':i,'sigma':float(ts[i]),'intervention':name,
                            **response_record(predictions[name],states[name],predictions['same'],states['same'],base,up,norm)})
                d=(x-original_d)/ts[i];xn=x+(ts[i+1]-ts[i])*d
                if i+1<32:
                    d2=(xn-denoise(xn,ts[i+1]))/ts[i+1];xn=x+(ts[i+1]-ts[i])*(d+d2)/2
                x=xn
            raw=decode(norm.target(x[:,:1].float(),up));saved=torch.as_tensor(archives[case,seed]['raw40_metres'],device=args.device)
            err=float((raw-saved).abs().max())
            print(f'Trajectory {case}/{seed}: exact saved-output maximum difference {err:.6g}m; elapsed {time.perf_counter()-start:.1f}s',flush=True)
            if err>.02:raise AssertionError('Reconstructed trajectory differs from reviewed generation')
    elapsed=time.perf_counter()-start
    result={'status':'budget limited' if timed_out else 'complete','settings':metadata,'endpoint_records':records,'trajectory_records':trajectory,
            'endpoint_summary':aggregate(records,['model','sigma','intervention'],'raw_height_response_active_rms_m'),
            'trajectory_summary':aggregate(trajectory,['step_index','intervention'],'raw_height_response_active_rms_m'),
            'elapsed_seconds_before_figures':elapsed,'identical_input_repeat_exact':True,
            'checkpoint_hashes_unchanged':{k:digest(p)==hashes[k] for k,p in paths.items()}}
    if not all(result['checkpoint_hashes_unchanged'].values()):raise AssertionError('Checkpoint changed during read-only probe')
    if not all(np.isfinite(r[k]) for r in records+trajectory for k,v in r.items() if isinstance(v,float)):raise ValueError('Nonfinite sensitivity metric')
    atomic_json(out/'results.json',result)
    figures(out,records,visuals)
    print(f'Finished {result["status"]}: {len(records)} endpoint / {len(trajectory)} trajectory intervention records, {time.perf_counter()-start:.1f}s total. No training.',flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',default=DEFAULT_REVIEW)
    p.add_argument('--out');p.add_argument('--device',default='cuda');p.add_argument('--max-seconds',type=float,default=110.)
    args=p.parse_args(argv)
    if not 10<=args.max_seconds<=120:p.error('This bounded probe requires 10..120 seconds')
    out=Path(args.out).resolve() if args.out else Path(args.review)/('plan-influence-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(out,'probe'):run(args,out)


if __name__=='__main__':main()
