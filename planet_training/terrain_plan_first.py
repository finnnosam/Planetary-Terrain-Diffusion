"""Dedicated morphology generator versus coupled continuation, unselected."""
import torch

from planet_training.terrain_joint_structure import build, mask_plan, JOINT_SPEC, sample as sample_joint
from planet_training.terrain_structure import StructureNorm, decode, project, shape_loss
from toy_meso.common import autocast
from toy_meso.model import heun_sample

VERSION='terrain-plan-first-v1'
SPEC={'joint_spec':JOINT_SPEC,'planner':'Zero noisy height channel; plan-only EDM loss weight 1; height output unused.',
      'coupled':'Continue joint objective unchanged: height + .25 plan + physical shape.',
      'conditioning':'Coarse height and permission only; no supplied plan or target height.',
      'comparison_limit':'Task/input/focus change together; not an isolated loss-weight or capacity ablation.',
      'renderer':'Frozen separately trained oracle; not optimized by either branch.'}


def planner_state(x,base):
    return mask_plan(torch.cat((torch.zeros_like(x[:,:1]),x[:,1:]),1),base)


def parts(model,mode,x,cin,target,base,up,norm,sigma,noise):
    noisy=mask_plan(x+noise*sigma[:,None,None,None],base)
    if mode=='planner':noisy=planner_state(noisy,base)
    elif mode!='coupled':raise ValueError('Unknown plan-first mode')
    predicted=model(noisy,sigma,cin).float()
    weight=((sigma.square()+.25)/(sigma*.5).square())[:,None,None,None]
    error=weight*(predicted-x).square();active=(base>200).float()
    plan=(error[:,1:]*active).sum()/(4*active.sum()).clamp_min(1)
    if mode=='coupled':
        height=error[:,:1].mean()
        shape=shape_loss(project(decode(norm.target(predicted[:,:1],up)),base),decode(target),base)
        total=height+.25*plan+shape
    else:
        height=shape=torch.zeros((),device=x.device);total=plan
    return {'total':total,'plan':plan,'height':height,'shape':shape}


@torch.no_grad()
def sample_plan(model,norm,coarse,mode,steps=32,seed=42,device='cuda'):
    if mode=='coupled':return sample_joint(model,norm,coarse,steps,seed,device)[3]
    if mode!='planner':raise ValueError('Unknown plan-first mode')
    base,up,cin=norm.condition(coarse.to(device))
    g=torch.Generator(device=device).manual_seed(seed)
    initial=planner_state(torch.randn((len(up),5,*up.shape[-2:]),generator=g,device=device,dtype=torch.float64),base)
    def denoise(x,sigma):
        return planner_state(model(x.float(),sigma.float().expand(len(x)),cin),base).double()
    with autocast(device):x=heun_sample(denoise,initial.shape,steps,device=device,initial=initial)
    return x[:,1:]/JOINT_SPEC['plan_scale']


def load_pair(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version')!=VERSION or ck.get('spec')!=SPEC:raise ValueError('Wrong plan-first checkpoint')
    models={}
    for mode in ['coupled','planner']:
        models[mode]=build(ck['config']);models[mode].load_state_dict(ck['ema'][mode])
        models[mode]=models[mode].to(device).eval()
    return models,StructureNorm(ck['stats']),ck
