"""Structural plan v2: continuous plans, explicit constraint policy, long-run planner.

This module keeps the v1 continuous plan definition (three land-restricted
relative-height fields at 80.64/161.28/322.56 km and 80.64 km local relief)
and the v1 renderer architecture. It adds:

* an explicit constraint *policy*. ``locked-200`` reproduces the existing
  rule exactly (every cell whose interpolated baseline is <=200 m is frozen,
  feathered to 600 m). ``coast-only`` freezes only ocean cells, so low coastal
  and inland land may form ranges and basins. Both keep sea level at 0 m and
  never let a land cell fall below 1 m (the existing projection already does
  that);
* a plan-only planner network (4 noisy plan channels + 2 conditioning
  channels in, 4 plan channels out). It can be initialised *exactly* from a
  v1 continuous planner, whose height channel was always zero;
* a variable network width for the long-budget test.

Nothing here selects a model, shifts sea level or forces land coverage.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from planet_training.terrain_structure import (
    OBJECTIVE, SPACING, CONSERVE_FACTOR, HANDOFF_FACTOR, FACTOR,
    StructureNorm, area_weights, blocks, decode, encode,
)
from planet_training.terrain_spatial_plan import continuous_fields, compress
from toy_meso.common import autocast
from toy_meso.model import EDMDenoiser, UNet, heun_sample, upsample_condition

VERSION = 'structure-plan-v2'
POLICIES = {
    'locked-200': {'protected_max_m': 200., 'feather_m': 400.,
                   'meaning': 'existing rule: baseline <=200 m frozen, feathered to 600 m'},
    'coast-only': {'protected_max_m': 0., 'feather_m': 50.,
                   'meaning': 'only ocean frozen; land feathered 0-50 m; land never projected below 1 m'},
}
PLAN_SPEC = {'channels': ['rel_80km', 'rel_161km', 'rel_323km', 'relief_80km'],
             'definition': 'terrain_spatial_plan.continuous_fields (v1 continuous, unchanged)',
             'compression': 'asinh(rel/100 m), log1p(relief/100 m)',
             'normalization': 'per-channel train-only mean/std, floor .05, scale .5',
             'mask': 'policy eligibility > 0'}
SIGMA_DATA = .5


def _policy(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown constraint policy {policy!r}; choose {sorted(POLICIES)}')
    return POLICIES[policy]


def eligibility_for(base, policy):
    p = _policy(policy)
    return ((base-p['protected_max_m'])/p['feather_m']).clamp(0, 1)


def active_for(base, policy):
    return eligibility_for(base, policy) > 0


def project_with(raw, base, policy, factor=CONSERVE_FACTOR, spacing=SPACING, feather=False):
    """terrain_structure.project with an explicit eligibility policy.

    For ``locked-200`` this is numerically identical to the existing project().
    Regional (factor-cell) means of the baseline are preserved, eligible land
    never goes below 1 m and ineligible cells keep their baseline height.
    """
    if raw.shape != base.shape:
        raise ValueError('Raw and baseline grids differ')
    w = eligibility_for(base, policy)
    r = (raw-base)*(w if feather else (w > 0))
    area = area_weights(base.shape[-1], spacing).to(base).expand_as(base)
    denominator = blocks(w*area, factor).sum((3, 5), keepdim=True).clamp_min(1e-12)
    correction = blocks(r*area, factor).sum((3, 5), keepdim=True)/denominator
    r = r-w*(correction.expand_as(blocks(r, factor)).reshape_as(r))
    negative = torch.maximum(r.clamp_max(0), (1.-base).clamp_max(0))
    negative = torch.where(w > 0, negative, torch.zeros_like(negative))
    positive = r.clamp_min(0)
    amount = -blocks(negative*area, factor).sum((3, 5), keepdim=True)
    total = blocks(positive*area, factor).sum((3, 5), keepdim=True).clamp_min(1e-12)
    scale = (amount/total).clamp(0, 1)
    r = negative+positive*scale.expand_as(blocks(r, factor)).reshape_as(r)
    return torch.where(w > 0, base+r, base)


def target_with(raw_metres, coarse, policy):
    base = upsample_condition(decode(coarse), FACTOR)
    return encode(project_with(raw_metres, base, policy, feather=True)), base


def condition_with(norm, coarse, policy):
    """Same channels as StructureNorm.condition, with the policy's permission map."""
    base = upsample_condition(decode(coarse), FACTOR)
    up = encode(base)
    s = norm.s
    return base, up, torch.cat(((up-s['cond_mean'])/s['cond_std'], 2*eligibility_for(base, policy)-1), 1)


def shape_loss_with(predicted, real, base, policy):
    losses = []
    for factor in (1, 2):
        p, t = [F.avg_pool2d(v/OBJECTIVE['shape_height_scale_m'], factor) for v in (predicted, real)]
        mask = F.avg_pool2d(active_for(base, policy).float(), factor) > .99
        for dim in (-1, -2):
            dp, dt = [torch.diff(v, dim=dim)/factor for v in (p, t)]
            m = (mask[..., 1:] & mask[..., :-1]) if dim == -1 else (mask[..., 1:, :] & mask[..., :-1, :])
            slope = ((dp-dt).square()*m).sum()/m.sum().clamp_min(1)
            cp, ct = [torch.diff(v, dim=dim)/factor for v in (dp, dt)]
            m2 = (m[..., 1:] & m[..., :-1]) if dim == -1 else (m[..., 1:, :] & m[..., :-1, :])
            curve = ((cp-ct).square()*m2).sum()/m2.sum().clamp_min(1)
            losses.append(OBJECTIVE['slope_weight']*slope+OBJECTIVE['curvature_weight']*curve)
    return sum(losses)/len(losses)


class PlanCodec:
    """Continuous plan codec; identical to v1 'continuous' for locked-200."""

    def __init__(self, stats, policy):
        _policy(policy)
        self.stats, self.policy = stats, policy

    def _ms(self, like):
        mean = like.new_tensor(self.stats['mean'])[None, :, None, None]
        std = like.new_tensor(self.stats['std'])[None, :, None, None]
        return mean, std

    def encode_fields(self, physical, base):
        mean, std = self._ms(physical)
        return .5*(compress(physical, 'continuous')-mean)/std*active_for(base, self.policy)

    def encode(self, z, base):
        return self.encode_fields(continuous_fields(z, base), base)

    def decode(self, encoded, base):
        mean, std = self._ms(encoded)
        v = encoded.float()/.5*std+mean
        v = torch.cat((100*torch.sinh(v[:, :3]), 100*torch.expm1(v[:, 3:4])), 1)
        return v*active_for(base, self.policy)


@torch.no_grad()
def fit_plan_stats(data, policy, recipe, device, count=256, seed=1099):
    """Train-split-only normalization, same protocol as v1."""
    from planet_training.adaptation import degrade_guides
    indices = np.random.default_rng(seed).choice(data.meta['train_count'], min(count, data.meta['train_count']), replace=False)
    g = torch.Generator(device=device).manual_seed(seed); values = []
    for start in range(0, len(indices), 16):
        raw, c, ctx = data.batch(0, 'train', indices[start:start+16], device)
        c = degrade_guides(ctx, c, g, recipe=recipe)
        target, base = target_with(raw, c, policy)
        p = compress(continuous_fields(decode(target), base), 'continuous')
        values.append(p.permute(0, 2, 3, 1)[active_for(base, policy)[:, 0]].cpu())
    v = torch.cat(values)
    return ({'mean': v.mean(0).tolist(), 'std': v.std(0, unbiased=False).clamp_min(.05).tolist(), 'active_cells': len(v)},
            {'split': 'train', 'seed': seed, 'indices': indices.tolist(), 'policy': policy})


class PlanDenoiser(nn.Module):
    """EDM denoiser over the 4 encoded plan channels, conditioned on coarse height + permission."""

    def __init__(self, **unet):
        super().__init__()
        self.sigma_data = SIGMA_DATA
        self.net = UNet(in_ch=6, out_ch=4, **unet)

    def forward(self, x, sigma, cond):
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd**2/(sigma**2+sd**2)
        c_out = sigma*sd/(sigma**2+sd**2).sqrt()
        c_in = 1/(sigma**2+sd**2).sqrt()
        f = self.net(torch.cat([c_in*x, cond], 1), sigma.flatten().log()/4)
        return c_skip*x+c_out*f.to(x.dtype)


def unet_config(base_config, width=None):
    config = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in base_config.items()}
    if width:
        config['ch'] = int(width)
    if config.get('dropout', 0) != 0:
        raise ValueError('Matched-noise protocol assumes zero dropout')
    return config


def build_planner(config):
    return PlanDenoiser(**{**config, 'mults': tuple(config['mults'])})


def build_renderer(config):
    return EDMDenoiser(cond_channels=6, **{**config, 'mults': tuple(config['mults'])})


def planner_from_v1(v1_state, planner):
    """Exact functional copy of a v1 continuous planner (JointDenoiser, height channel unused).

    v1 input channels: [noisy height, 4 noisy plan, coarse height, permission]; its height
    input was always zero, so dropping that column changes nothing. v1 outputs:
    [height, 4 plan]; only the plan outputs are kept.
    """
    state = planner.state_dict()
    for name, old in v1_state.items():
        if name not in state:
            raise ValueError(f'Unexpected v1 planner key {name}')
        if name == 'net.inp.weight':
            state[name].copy_(old[:, 1:7])
        elif name in ('net.out.2.weight', 'net.out.2.bias'):
            state[name].copy_(old[1:5])
        else:
            if state[name].shape != old.shape:
                raise ValueError(f'Width/architecture differs from v1 planner at {name}')
            state[name].copy_(old)
    planner.load_state_dict(state)


def edm_weight(sigma):
    return ((sigma.square()+SIGMA_DATA**2)/(sigma*SIGMA_DATA).square())[:, None, None, None]


def planner_loss(model, plan, cin, active, sigma, noise):
    noisy = (plan+noise*sigma[:, None, None, None])*active
    pred = model(noisy, sigma, cin).float()
    a = active.float()
    return (edm_weight(sigma)*(pred-plan).square()*a).sum()/(4*a.sum()).clamp_min(1)


def renderer_loss(model, norm, target, base, up, cin, plan, sigma, noise, policy):
    x = norm.residual(target, up)
    pred = model(x+noise*sigma[:, None, None, None], sigma, torch.cat((cin, plan), 1)).float()
    main = (edm_weight(sigma)*(pred-x).square()).mean()
    physical = project_with(decode(norm.target(pred, up)), base, policy)
    shape = shape_loss_with(physical, decode(target), base, policy)
    return main+shape, main, shape


@torch.no_grad()
def sample_planner(model, norm, coarse, policy, seed=42, steps=32, device='cuda'):
    base, up, cin = condition_with(norm, coarse.to(device), policy)
    active = active_for(base, policy).to(torch.float64)
    g = torch.Generator(device=device).manual_seed(seed)
    # Draw five channels and drop the first, so a seed gives exactly the v1 planner's noise.
    initial = torch.randn((len(up), 5, *up.shape[-2:]), generator=g, device=device, dtype=torch.float64)[:, 1:]*active
    def denoise(x, sigma):
        return (model(x.float(), sigma.float().expand(len(x)), cin).double())*active
    with autocast(device):
        x = heun_sample(denoise, initial.shape, steps, device=device, initial=initial)
    return x.float()*active.float()


@torch.no_grad()
def sample_renderer(model, norm, coarse, plan, policy, seed=7919, steps=32, device='cuda'):
    base, up, cin = condition_with(norm, coarse.to(device), policy)
    if plan.shape != (len(up), 4, *up.shape[-2:]) or not torch.isfinite(plan).all():
        raise ValueError('Bad encoded plan')
    cin = torch.cat((cin, plan.to(cin)), 1)
    def denoise(x, sigma):
        return model(x.float(), sigma.float().expand(len(x)), cin).double()
    with autocast(device):
        x = heun_sample(denoise, up.shape, steps, torch.Generator(device=device).manual_seed(seed), device)
    raw = decode(norm.target(x.float(), up))
    return project_with(raw, base, policy), base, raw


def handoff_with(structure, structure_base, coarse, policy):
    """Direct 40 -> 23 km handoff with no old-intermediate detail (as in the v1 reviews)."""
    base23 = decode(upsample_condition(coarse, 7))
    delta = F.interpolate(structure-structure_base, size=base23.shape[-2:], mode='bilinear', align_corners=False)
    return project_with(base23+delta, base23, policy, factor=HANDOFF_FACTOR, spacing=23040.), base23


@torch.no_grad()
def constraint_check(z, base, policy, factor, spacing, tolerance_m=.05):
    area = area_weights(z.shape[-1], spacing).to(z).expand_as(z)
    drift = blocks((z-base)*area, factor).sum((3, 5))/blocks(area, factor).sum((3, 5))
    frozen = eligibility_for(base, policy) == 0
    result = {'footprint_mean_drift_max_m': float(drift.abs().max()),
              'frozen_change_max_m': float((z-base)[frozen].abs().max()) if frozen.any() else 0.,
              'sign_disagreement_fraction': float(((z > 0) != (base > 0)).float().mean())}
    if (not torch.isfinite(z).all() or result['footprint_mean_drift_max_m'] > tolerance_m
            or result['frozen_change_max_m'] > .005 or result['sign_disagreement_fraction']):
        raise ValueError(f'Constraint check failed: {result}')
    return result


def load_v1_continuous_renderer(path, device='cpu'):
    """The frozen v1 continuous renderer + its codec stats (valid for policy locked-200)."""
    from planet_training.terrain_spatial_plan import load_pair
    models, norm, codec, ck = load_pair(path, 'renderer', device)
    return models['continuous'], norm, PlanCodec(ck['plan_stats']['continuous'], 'locked-200'), ck


def load_checkpoint(path, device='cpu', use_ema=True):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') != VERSION or ck.get('plan_spec') != PLAN_SPEC:
        raise ValueError('Not a structure-plan-v2 checkpoint')
    model = (build_planner if ck['task'] == 'planner' else build_renderer)(ck['config'])
    model.load_state_dict(ck['ema'] if use_ema else ck['model'])
    return model.to(device).eval(), StructureNorm(ck['stats']), PlanCodec(ck['plan_stats'], ck['policy']), ck


def load_any_renderer(path, device='cpu'):
    """Accept a v2 renderer checkpoint or a v1 paired renderer (its continuous member)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') == VERSION:
        del ck
        return load_checkpoint(path, device)
    del ck
    model, norm, codec, ck = load_v1_continuous_renderer(path, device)
    return model, norm, codec, {**ck, 'policy': 'locked-200', 'task': 'renderer', 'source_format': 'v1-paired-continuous'}


def load_any_planner(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') == VERSION:
        del ck
        return load_checkpoint(path, device)
    from planet_training.terrain_spatial_plan import VERSION as V1, SPEC as V1SPEC
    if ck.get('version') != V1 or ck.get('spec') != V1SPEC or ck.get('task') != 'planner':
        raise ValueError('Not a v1 paired planner or v2 planner checkpoint')
    model = build_planner(unet_config(ck['config'])); planner_from_v1(ck['ema']['continuous'], model)
    meta = {**{k: ck[k] for k in ('config', 'stats', 'step', 'dataset_id')}, 'policy': 'locked-200', 'task': 'planner',
            'source_format': 'v1-paired-continuous'}
    return model.to(device).eval(), StructureNorm(ck['stats']), PlanCodec(ck['plan_stats']['continuous'], 'locked-200'), meta
