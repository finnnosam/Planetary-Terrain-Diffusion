"""Unselected matched categorical/continuous spatial-supervision experiment."""
import numpy as np
import torch

from planet_training.terrain_structure import (
    StructureNorm, blur, decode, project, shape_loss, target_for,
)
from planet_training.terrain_plan_oracle import build as build_renderer, plan_for
from planet_training.terrain_joint_structure import build as build_planner, initialize_from_control, interpret_plan
from toy_meso.common import autocast
from toy_meso.model import heun_sample

VERSION='terrain-spatial-plan-v1'
MODES=['categorical','continuous']
SPEC={'grid_m':40320.,'channels':4,'continuous_sigmas_m':[80640.,161280.,322560.],
    'continuous_support':'baseline >0; normalized land-only Gaussian moments; reflect padding',
    'continuous_compression':'asinh(relative/100m), log1p(local_rms_relief/100m)',
    'categorical':'unchanged terrain-plan-oracle-v1 four physical channels',
    'normalization':'per-channel active-cell train-only mean/std, floor .05; encoded scale .5',
    'permission':'all encoded channels zero where baseline <=200m, same in both modes',
    'planner':'zero noisy height, four-channel plan-only EDM objective, no target-height conditioning',
    'renderer':'unchanged feasible height EDM plus physical shape objective; supplied plan only',
    'limits':'Representation bundle: continuity, multiscale context, land statistics; not a single-factor ablation.'}


@torch.no_grad()
def continuous_fields(z,base):
    if z.shape!=base.shape or z.ndim!=4 or z.shape[1]!=1:raise ValueError('Expected matching Bx1 physical fields')
    with torch.autocast(device_type=z.device.type,enabled=False):
        z=z.float();mask=(base>0).float();fields=[]
        for sigma in [2.,4.,8.]:
            den=blur(mask,sigma).clamp_min(1e-6);mean=blur(z*mask,sigma)/den
            fields.append((z-mean)*mask)
            if sigma==2.:relief=(blur(z.square()*mask,sigma)/den-mean.square()).clamp_min(0).sqrt()*mask
        p=torch.cat((*fields,relief),1)
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite physical plan')
        return p


def compress(p,mode):
    if mode=='categorical':return p.float()
    if mode!='continuous':raise ValueError(mode)
    return torch.cat((torch.asinh(p[:,:3].float()/100),torch.log1p(p[:,3:4].float().clamp_min(0)/100)),1)


class PlanCodec:
    def __init__(self,stats):self.stats=stats

    def encode_fields(self,physical,base,mode):
        s=self.stats[mode];mean=physical.new_tensor(s['mean'])[None,:,None,None];std=physical.new_tensor(s['std'])[None,:,None,None]
        return .5*(compress(physical,mode)-mean)/std*(base>200)

    def encode(self,z,base,mode):
        physical=plan_for(z,base) if mode=='categorical' else continuous_fields(z,base)
        return self.encode_fields(physical,base,mode)

    def decode(self,encoded,base,mode):
        s=self.stats[mode];mean=encoded.new_tensor(s['mean'])[None,:,None,None];std=encoded.new_tensor(s['std'])[None,:,None,None]
        v=encoded.float()/.5*std+mean
        if mode=='continuous':v=torch.cat((100*torch.sinh(v[:,:3]),100*torch.expm1(v[:,3:4])),1)
        return v*(base>200)

    def canonical_generated(self,p,base,mode):
        # Same hard interpretation as earlier category pilots, in physical label units.
        if mode=='categorical':return self.encode_fields(interpret_plan(self.decode(p,base,mode),base),base,mode)
        return p.float()*(base>200)


def build(task,config):
    if task=='renderer':return build_renderer(config)
    if task=='planner':return build_planner(config)
    raise ValueError(task)


def initialize(model,source_state,task):
    if source_state['net.inp.weight'][:,3:].abs().max()!=0:raise ValueError('Source must be height-only control')
    if task=='renderer':model.load_state_dict(source_state)
    elif task=='planner':initialize_from_control(model,source_state)
    else:raise ValueError(task)


def loss_parts(model,task,norm,target,base,up,cin,plan,sigma,noise):
    weight=((sigma.square()+.25)/(sigma*.5).square())[:,None,None,None]
    if task=='renderer':
        x=norm.residual(target,up);pred=model(x+noise[:,:1]*sigma[:,None,None,None],sigma,torch.cat((cin,plan),1)).float()
        main=(weight*(pred-x).square()).mean()
        physical=project(decode(norm.target(pred,up)),base)
        shape=shape_loss(physical,decode(target),base)
        return {'total':main+shape,'main':main,'shape':shape}
    if task!='planner':raise ValueError(task)
    x=torch.cat((torch.zeros_like(up),plan),1)
    noisy=torch.cat((torch.zeros_like(up),(plan+noise[:,1:]*sigma[:,None,None,None])*(base>200)),1)
    pred=model(noisy,sigma,cin).float();active=(base>200).float()
    main=(weight*(pred[:,1:]-x[:,1:]).square()*active).sum()/(4*active.sum()).clamp_min(1)
    return {'total':main,'main':main,'shape':main.new_zeros(())}


@torch.no_grad()
def sample_renderer(model,norm,coarse,plan,seed=7919,steps=32,device='cuda'):
    base,up,cin=norm.condition(coarse.to(device))
    if plan.shape!=(len(up),4,*up.shape[-2:]) or not torch.isfinite(plan).all():raise ValueError('Bad encoded plan')
    cin=torch.cat((cin,plan.to(cin)),1)
    def denoise(x,sigma):return model(x.float(),sigma.float().expand(len(x)),cin).double()
    with autocast(device):x=heun_sample(denoise,up.shape,steps,torch.Generator(device=device).manual_seed(seed),device)
    raw=decode(norm.target(x.float(),up));return project(raw,base),base,raw


@torch.no_grad()
def sample_planner(model,norm,coarse,seed=42,steps=32,device='cuda'):
    base,up,cin=norm.condition(coarse.to(device))
    def mask(x):return torch.cat((torch.zeros_like(x[:,:1]),x[:,1:]*(base>200)),1)
    initial=mask(torch.randn((len(up),5,*up.shape[-2:]),generator=torch.Generator(device=device).manual_seed(seed),device=device,dtype=torch.float64))
    def denoise(x,sigma):return mask(model(x.float(),sigma.float().expand(len(x)),cin)).double()
    with autocast(device):x=heun_sample(denoise,initial.shape,steps,device=device,initial=initial)
    return x[:,1:].float()


def load_pair(path,task=None,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version')!=VERSION or ck.get('spec')!=SPEC or (task and ck['task']!=task):raise ValueError('Wrong spatial-plan checkpoint')
    models={}
    for mode in MODES:
        model=build(ck['task'],ck['config']);model.load_state_dict(ck['ema'][mode]);models[mode]=model.to(device).eval()
    return models,StructureNorm(ck['stats']),PlanCodec(ck['plan_stats']),ck
