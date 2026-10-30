"""Short plan-signal clamping intervention on the frozen joint model."""
import argparse
from pathlib import Path
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from planet_training.terrain_joint_structure import load, mask_plan, interpret_plan, JOINT_SPEC
from planet_training.terrain_structure import decode, project
from toy_meso.model import sigma_schedule
from toy_meso.common import autocast
from scripts.review_terrain_joint_structure import plan_metrics
from scripts.probe_joint_plan_influence import DEFAULT_REVIEW, digest, rotate_plan, rms
from scripts.structure_job import console_log


@torch.no_grad()
def run(args,out):
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    review=Path(args.review).resolve();settings=json.loads((review/'settings.json').read_text())
    checkpoint=review/'joint-checkpoint.pt';before=digest(checkpoint)
    if before!=settings['checkpoints']['joint']['sha256']:raise ValueError('Frozen checkpoint identity changed')
    model,norm,_=load(checkpoint,'cuda');records=[];differences={}
    meta={'review':str(review),'checkpoint_sha256':before,'cases':['california','alps','generated-60'],
          'height_noise_seeds':[42,314],'steps':32,'variants':['own','repeat','swap','zero_signal','rotate'],
          'method':'Prescribe clean generated plan plus sigma times SAME fixed plan noise at each Heun evaluation. Same initial height noise/coarse guide; no Earth plan. Donor is next saved seed. Baseline own plan is also clamped, not free sampling.',
          'limit':'Intervention, not a production planner. Constraints can make a changed plan infeasible. Marginalising or clamping fields in a joint denoiser is not guaranteed exact conditional sampling.',
          'training_updates':0,'selection_changed':False,'sea_level_m':0.,'maximum_seconds':100.,
          'additional_supplied_reference_plan_benchmark':args.with_reference_plan}
    if args.with_reference_plan:
        meta['method']=meta['method'].replace('no Earth plan.', 'no Earth plan in generated-plan interventions. Additional labelled supplied_reference_plan rows use the Earth-derived feasible reference plan on Earth cases only.')
    atomic_json(out/'settings.json',meta)
    for case in meta['cases']:
        for seed in meta['height_noise_seeds']:
            if time.perf_counter()-start>90:raise TimeoutError('Bounded clamp probe stopped before next group')
            donor_seed=settings['seeds'][(settings['seeds'].index(seed)+1)%len(settings['seeds'])]
            a=dict(np.load(review/f'{case}-s{seed}-joint.npz'));donor=dict(np.load(review/f'{case}-s{donor_seed}-joint.npz'))
            coarse=torch.as_tensor(a['coarse_encoded'],device='cuda');base,up,cin=norm.condition(coarse)
            own=torch.as_tensor(a['generated_plan_raw'],device='cuda')*JOINT_SPEC['plan_scale']
            other=torch.as_tensor(donor['generated_plan_raw'],device='cuda')*JOINT_SPEC['plan_scale']
            names=meta['variants'].copy()
            signals=torch.cat((own,own.clone(),other,torch.zeros_like(own),rotate_plan(own)))
            reference=review/f'{case}-s{seed}-feasible-target-reference.npz'
            if args.with_reference_plan and reference.exists():
                with np.load(reference) as ref:reference_plan=torch.as_tensor(ref['interpreted_plan'],device='cuda')*JOINT_SPEC['plan_scale']
                names.append('supplied_reference_plan');signals=torch.cat((signals,reference_plan))
            count=len(names)
            signals=signals*(base>200)
            initial=mask_plan(torch.randn((1,5,64,64),generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.float64),base)
            eps=initial[:,1:].expand(count,-1,-1,-1)
            ts=sigma_schedule(32,device='cuda')
            x=initial.expand(count,-1,-1,-1).clone()*ts[0]
            def constrain(state,sigma):return torch.cat((state[:,:1],signals+sigma*eps),1)
            def denoise(state,sigma):
                with autocast('cuda'):return mask_plan(model(state.float(),sigma.float().expand(count),cin.expand(count,-1,-1,-1)),base).double()
            for i in range(32):
                x=constrain(x,ts[i]);d=(x-denoise(x,ts[i]))/ts[i]
                xn=constrain(x+(ts[i+1]-ts[i])*d,ts[i+1])
                if i+1<32:
                    d2=(xn-denoise(xn,ts[i+1]))/ts[i+1]
                    xn=constrain(x+(ts[i+1]-ts[i])*(d+d2)/2,ts[i+1])
                x=xn
            raw=decode(norm.target(x[:,:1].float(),up));z=project(raw,base.expand_as(raw))
            torch.testing.assert_close(raw[:1],raw[1:2],atol=0,rtol=0)
            q=(torch.arange(64,device='cuda')+.5-32)*40.32
            central=(q.abs()<599.04)[None,None,:,None]&(q.abs()<599.04)[None,None,None,:]
            for i,name in enumerate(names):
                plan=interpret_plan(signals[i:i+1]/JOINT_SPEC['plan_scale'],base)
                row={'case':case,'seed':seed,'donor_seed':donor_seed,'intervention':name,
                     'raw_change_from_own_rms_m':rms(raw[i:i+1]-raw[:1],base>200),
                     'projected_change_from_own_rms_m':rms(z[i:i+1]-z[:1],base>200),
                     'central_projected_change_from_own_rms_m':rms(z[i:i+1]-z[:1],(base>200)&central),
                     'own_height_agreement_with_desired_plan':plan_metrics(z[:1],base,plan),
                     'changed_height_agreement_with_desired_plan':plan_metrics(z[i:i+1],base,plan)}
                records.append(row)
                np.savez_compressed(out/f'{case}-s{seed}-{name}.npz',coarse_encoded=a['coarse_encoded'],baseline40_metres=base.cpu().numpy(),
                    raw40_metres=raw[i:i+1].cpu().numpy(),projected40_metres=z[i:i+1].cpu().numpy(),
                    prescribed_plan=plan.cpu().numpy(),prescribed_plan_signal=signals[i:i+1].cpu().numpy())
                if case=='california' and seed==42 and name in ['swap','zero_signal','rotate']:
                    differences[name]=(z[i:i+1]-z[:1]).cpu()
            print(f'Clamp group {case}/{seed}: elapsed {time.perf_counter()-start:.1f}s',flush=True)
    result={'status':'complete','settings':meta,'records':records,'identical_repeat_exact':True,
            'checkpoint_hash_unchanged':digest(checkpoint)==before,'elapsed_seconds':time.perf_counter()-start}
    atomic_json(out/'results.json',result)
    fig,axs=plt.subplots(1,3,figsize=(12,4.5),constrained_layout=True)
    for ax,(name,delta) in zip(axs,differences.items()):
        z=F.interpolate(delta,size=(112,112),mode='bilinear',align_corners=False).numpy()[0,0,30:82,30:82]
        im=ax.imshow(z,cmap='RdBu_r',vmin=-100,vmax=100,extent=(-599.04,599.04,-599.04,599.04),interpolation='nearest')
        ax.set_title(name+' plan versus own clamped plan',fontsize=10);ax.set_xlabel('km');ax.set_ylabel('km')
    fig.colorbar(im,ax=axs.tolist(),orientation='horizontal',fraction=.06,pad=.03,label='Final projected height CHANGE (m); fixed ±100 m colour limits')
    fig.suptitle('California, height seed 42: same height noise AND plan noise; change plan signal only\n32-step diagnostic intervention, no Earth plan; each panel 1,198.08 km',fontsize=11)
    fig.savefig(out/'california-clamp-differences.png',dpi=140);plt.close(fig)
    print(f'Finished: {time.perf_counter()-start:.1f}s; {len(records)-6} intervention samples plus six exact repeat controls. No training.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',default=DEFAULT_REVIEW);p.add_argument('--out',required=True);p.add_argument('--with-reference-plan',action='store_true')
    args=p.parse_args(argv);out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    with console_log(out,'clamp'):run(args,out)


if __name__=='__main__':main()
