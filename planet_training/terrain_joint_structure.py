"""Unselected intermediate: jointly generate elevation and morphology from noise."""
import torch
from torch import nn

from toy_meso.model import EDMDenoiser, UNet, heun_sample
from toy_meso.common import autocast
from planet_training.terrain_structure import StructureNorm, target_for, decode, project, shape_loss, eligibility
from planet_training.terrain_plan_oracle import PLAN_SPEC, plan_for

VERSION = 'terrain-joint-structure-v1'
JOINT_SPEC = {
    'channels': ['encoded_height_residual', *PLAN_SPEC['channels']],
    'conditioning': 'coarse height and baseline permission only; no target plan',
    'plan_scale': .5, 'plan_loss_weight': .25,
    'plan_loss_mask': 'baseline >200 m, same mask in both modes',
    'plan_spec': PLAN_SPEC,
    'control': 'same five-channel model; four clean morphology targets zero',
    'sampling': 'all five channels start as independent noise; inactive plan cells zero',
    'height_objective': 'unchanged feasible 40-km target and physical shape loss',
}


class JointDenoiser(EDMDenoiser):
    def __init__(self, **config):
        nn.Module.__init__(self)
        self.sigma_data = .5
        self.net = UNet(in_ch=7, out_ch=5, **config)


def build(config): return JointDenoiser(**config)


def initialize_from_control(model, source_state):
    """Copy scalar height response; all new noisy-plan paths/outputs start zero."""
    state = model.state_dict()
    if set(state) != set(source_state): raise ValueError('Source architecture keys changed')
    for name, old in source_state.items():
        if name == 'net.inp.weight':
            if old.shape[1] != 7 or old[:, 3:].abs().max() != 0:
                raise ValueError('Expected the frozen zero-plan oracle-experiment control')
            state[name].zero_()
            state[name][:, 0] = old[:, 0]
            state[name][:, 5:7] = old[:, 1:3]
        elif name in ('net.out.2.weight', 'net.out.2.bias'):
            if old.shape[0] != 1: raise ValueError('Expected scalar source output')
            state[name].zero_(); state[name][:1] = old
        else:
            if state[name].shape != old.shape: raise ValueError(f'Changed architecture: {name}')
            state[name].copy_(old)
    model.load_state_dict(state)


def targets(norm, raw_metres, coarse, mode):
    target, base = target_for(raw_metres, coarse)
    _, up, cin = norm.condition(coarse)
    plan = plan_for(decode(target), base)
    if mode == 'control': plan = torch.zeros_like(plan)
    elif mode != 'joint': raise ValueError('Mode must be control or joint')
    x = torch.cat((norm.residual(target, up), JOINT_SPEC['plan_scale']*plan), 1)
    return x, cin, target, base, up


def mask_plan(x, base):
    return torch.cat((x[:, :1], x[:, 1:]*(eligibility(base) > 0)), 1)


def loss_parts(model, norm, raw, coarse, mode, generator=None, sigma=None):
    x, cin, target, base, up = targets(norm, raw, coarse, mode)
    if sigma is None:
        sigma = (torch.randn(len(x), device=x.device, generator=generator)*1.2-1.2).exp()
    noise = torch.randn(x.shape, device=x.device, generator=generator)
    noisy = mask_plan(x+noise*sigma[:, None, None, None], base)
    predicted = model(noisy, sigma, cin).float()
    weight = ((sigma.square()+.25)/(sigma*.5).square())[:, None, None, None]
    error = weight*(predicted-x).square()
    height = error[:, :1].mean()
    active = (eligibility(base) > 0).float()
    plan = (error[:, 1:]*active).sum()/(4*active.sum()).clamp_min(1)
    physical = project(decode(norm.target(predicted[:, :1], up)), base)
    shape = shape_loss(physical, decode(target), base)
    total = height+JOINT_SPEC['plan_loss_weight']*plan+shape
    return {'total': total, 'height': height, 'plan': plan, 'shape': shape}


def loss(model, norm, raw, coarse, mode): return loss_parts(model, norm, raw, coarse, mode)['total']


def interpret_plan(raw_plan, base):
    """For frozen-renderer diagnostics only; preserve raw generated channels too."""
    p = raw_plan.float()
    category = (p[:, :1] > .5).float()-(p[:, :1] < -.5).float()
    quiet = (p[:, 1:2] > .5).float()
    axes = p[:, 2:4]
    length = axes.square().sum(1, keepdim=True).sqrt()
    axes = torch.where(length > 1e-6, axes/length.clamp_min(1e-6), torch.zeros_like(axes))
    return torch.cat((category, quiet, axes*category.abs()), 1)*(eligibility(base) > 0)


@torch.no_grad()
def sample(model, norm, coarse, steps=32, seed=42, device='cpu'):
    """No modern target, oracle plan, or reference region is accepted here."""
    base, up, cin = norm.condition(coarse.to(device))
    g = torch.Generator(device=device).manual_seed(seed)
    initial = mask_plan(torch.randn((len(up), 5, *up.shape[-2:]), generator=g,
                                   device=device, dtype=torch.float64), base)
    def denoise(x, sigma):
        return mask_plan(model(x.float(), sigma.float().expand(len(x)), cin), base).double()
    with autocast(device):
        x = heun_sample(denoise, initial.shape, steps, device=device, initial=initial)
    raw = decode(norm.target(x[:, :1], up))
    plan = x[:, 1:]/JOINT_SPEC['plan_scale']
    return project(raw, base), base, raw, plan


def load(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') != VERSION or ck.get('joint_spec') != JOINT_SPEC or ck.get('mode') not in ('control', 'joint'):
        raise ValueError('Not a matching autonomous joint-structure checkpoint')
    model = build(ck['config']); model.load_state_dict(ck['ema'])
    return model.to(device).eval(), StructureNorm(ck['stats']), ck
