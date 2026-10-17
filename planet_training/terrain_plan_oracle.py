"""Regional oracle-plan conditioning experiment; no production selection."""
import torch

from planet_training.terrain_structure import (
    StructureNorm, blur, decode, encode, eligibility, project, target_for, shape_loss,
)
from toy_meso.common import autocast
from toy_meso.model import EDMDenoiser, heun_sample

VERSION = 'terrain-plan-oracle-v1'
PLAN_SPEC = {'channels': ['relative_position', 'quiet_region', 'corridor_cos2axis', 'corridor_sin2axis'],
    'relative_position_threshold_m': 150., 'neighbourhood_sigma_m': 80640.,
    'quiet_neighbourhood_relief_max_m': 100., 'quiet_slope_max_m_per_km': 3.,
    'direction_tensor_sigma_m': 40320., 'spacing_m': 40320.,
    'calculation_dtype': 'float32; mixed precision disabled for physical geometry',
    'source': 'feasible constrained target, not exact elevations; no global plan generator',
    'inactive': 'all plan channels zero where the baseline permission is zero'}


@torch.no_grad()
def plan_for(target_metres, baseline):
    """Four coarse morphological channels. These are oracle information."""
    if target_metres.shape != baseline.shape or target_metres.ndim != 4 or target_metres.shape[1] != 1:
        raise ValueError('Expected matching Bx1xHxW physical metre fields')
    if not torch.isfinite(target_metres).all() or not torch.isfinite(baseline).all():
        raise ValueError('Plan fields must be finite')
    # Squared metre heights overflow float16 well below mountain elevations.
    # Training's model autocast must not change these physical plan definitions.
    with torch.autocast(device_type=target_metres.device.type, enabled=False):
        return _plan_for_float32(target_metres.float(), baseline.float())


def _plan_for_float32(target_metres, baseline):
    sigma = PLAN_SPEC['neighbourhood_sigma_m']/PLAN_SPEC['spacing_m']
    mean = blur(target_metres, sigma)
    position = target_metres-mean
    category = (position > PLAN_SPEC['relative_position_threshold_m']).float()-(position < -PLAN_SPEC['relative_position_threshold_m']).float()
    relief = (blur(target_metres.square(), sigma)-mean.square()).clamp_min(0).sqrt()
    smoothed = blur(target_metres, .75)
    dy, dx = torch.gradient(smoothed, spacing=(40.32, 40.32), dim=(-2, -1))
    # Local axes: x right, y up. Direction is axial: theta and theta+pi agree.
    gy, gx = -dy, dx
    quiet = ((relief < PLAN_SPEC['quiet_neighbourhood_relief_max_m']) &
        (torch.hypot(gx, gy) < PLAN_SPEC['quiet_slope_max_m_per_km'])).float()
    xx, yy, xy = [blur(v, PLAN_SPEC['direction_tensor_sigma_m']/PLAN_SPEC['spacing_m']) for v in (gx.square(), gy.square(), gx*gy)]
    anisotropy = torch.sqrt((xx-yy).square()+4*xy.square())
    stable = anisotropy > 1e-6
    cosine = torch.where(stable, (yy-xx)/anisotropy.clamp_min(1e-6), torch.zeros_like(xx))
    sine = torch.where(stable, -2*xy/anisotropy.clamp_min(1e-6), torch.zeros_like(xx))
    corridors = category.abs()
    active = (eligibility(baseline) > 0).float()
    return torch.cat((category, quiet, corridors*cosine, corridors*sine), 1)*active


def build(config):
    # Both control and oracle use exactly the same architecture/capacity.
    return EDMDenoiser(cond_channels=6, **config)


def initialize_from_source(model, source_state):
    """Preserve the old noisy/height/mask response; new plan weights start zero."""
    state = model.state_dict()
    if set(state) != set(source_state):
        raise ValueError('Source architecture keys changed')
    for name, old in source_state.items():
        if name == 'net.inp.weight':
            if old.shape[1] != 3 or state[name].shape[1] != 7:
                raise ValueError('Expected noisy input + height/mask + four plan channels')
            state[name].zero_(); state[name][:, :3].copy_(old)
        else:
            if old.shape != state[name].shape:
                raise ValueError(f'Source architecture changed: {name}')
            state[name].copy_(old)
    model.load_state_dict(state)


def condition(norm, coarse, plan):
    base, up, height_mask = norm.condition(coarse)
    expected = (len(coarse), 4, *up.shape[-2:])
    if tuple(plan.shape) != expected or not torch.isfinite(plan).all():
        raise ValueError(f'Expected finite plan of shape {expected}')
    if plan.abs().max() > 1.00001:
        raise ValueError('Plan channels must lie in [-1,1]')
    return base, up, torch.cat((height_mask, plan.to(height_mask)), 1)


def loss(model, norm, raw_metres, coarse, mode):
    target, base = target_for(raw_metres, coarse)
    plan = plan_for(decode(target), base)
    if mode == 'control':
        plan = torch.zeros_like(plan)
    elif mode != 'oracle':
        raise ValueError('Mode must be control or oracle')
    _, up, cin = condition(norm, coarse, plan)
    x = norm.residual(target, up)
    sigma = (torch.randn(len(x), device=x.device)*1.2-1.2).exp()
    noisy = x+torch.randn_like(x)*sigma[:, None, None, None]
    predicted = model(noisy, sigma, cin)
    weight = (sigma.square()+.25)/(sigma*.5).square()
    main = (weight[:, None, None, None]*(predicted.float()-x).square()).mean()
    terrain = project(decode(norm.target(predicted.float(), up)), base)
    return main+shape_loss(terrain, decode(target), base)


@torch.no_grad()
def sample(model, norm, coarse, plan, steps=32, seed=42, device='cpu'):
    base, up, cin = condition(norm, coarse.to(device), plan.to(device))
    def denoise(x, sigma):
        return model(x.float(), sigma.float().expand(len(x)), cin).double()
    with autocast(device):
        x = heun_sample(denoise, up.shape, steps, torch.Generator(device=device).manual_seed(seed), device)
    raw = decode(norm.target(x.float(), up))
    return project(raw, base), base, raw


def load(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') != VERSION or ck.get('plan_spec') != PLAN_SPEC or ck.get('mode') not in ('control', 'oracle'):
        raise ValueError('Not an oracle-plan experiment checkpoint')
    model = build(ck['config']); model.load_state_dict(ck['ema'])
    return model.to(device).eval(), StructureNorm(ck['stats']), ck


@torch.no_grad()
def plan_agreement(output, baseline, desired):
    actual = plan_for(output, baseline)
    active = eligibility(baseline) > 0
    corridors = active & (desired[:, :1].abs() > .5)
    dot = (actual[:, 2:4]*desired[:, 2:4]).sum(1, keepdim=True)
    return {'corridor_cells': int(corridors.sum()),
        'corridor_type_match_fraction': float(((actual[:, :1] == desired[:, :1]) & corridors).sum()/corridors.sum().clamp_min(1)),
        'corridor_axis_dot_mean': float((dot*corridors).sum()/corridors.sum().clamp_min(1)),
        'quiet_region_match_fraction': float(((actual[:, 1:2] == desired[:, 1:2]) & active).sum()/active.sum().clamp_min(1)),
        'limit': 'Agreement with explicit coarse oracle plan, not an automatic realism score.'}
