"""Regional 40-km structure representation and conservative downstream handoff."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adaptation import STRUCTURE_RECIPE, degrade_guides
from .data import manifest_id
from toy_meso.common import autocast
from toy_meso.model import EDMDenoiser, heun_sample, upsample_condition

VERSION = 'terrain-structure-40-v1'
SPACING = 40320.
FACTOR = 4
SIZE = 64
CONSERVE_FACTOR = 16
HANDOFF_FACTOR = 28
COARSE_SPACING = 161280.
DEFAULT_DATA = 'outputs/planet_training/data/terrain-structure-40-v1'
DEFAULT_RUN = 'outputs/structure/terrain-40-pilot-v1'
OBJECTIVE = {'version': VERSION, 'guide_recipe': STRUCTURE_RECIPE,
             'target': 'modern 40.32-km metre averages, feathered conservative interior redistribution',
             'protected_max_m': 200., 'feather_range_m': 400.,
             'coarse_constraint': 'preserve Jacobian-weighted 645.12-km regional means of interpolated baseline; not original coarse nodes',
             'conservation_spacing_m': 645120.,
             'slope_weight': .15, 'curvature_weight': .05, 'shape_height_scale_m': 1000.,
             'handoff_detail_sigma_m': 80640.}


def decode(x):
    return x.sign()*x.square()


def encode(x):
    return x.sign()*x.abs().sqrt()


@lru_cache(maxsize=8)
def area_weights(size, spacing):
    """Tangent-plane cell-centre solid-angle Jacobian; rotation invariant."""
    q = (np.arange(size)+.5-size/2)*spacing/6371000.
    return torch.from_numpy((1+q[:, None]**2+q[None, :]**2)**-1.5).float()[None, None]


def overlap_matrix(old, new):
    """Exact cell interval overlaps, including noninteger 7:4 grid conversion."""
    lo = torch.arange(new, dtype=torch.float64)[:, None]*old/new
    hi = (torch.arange(new, dtype=torch.float64)[:, None]+1)*old/new
    j = torch.arange(old, dtype=torch.float64)[None]
    return (torch.minimum(hi, j+1)-torch.maximum(lo, j)).clamp_min(0).float()


def conservative_resample(z, new_size, spacing=23040.):
    """Area-average existing metre cell means; never interpolate encoded heights."""
    old = z.shape[-1]
    if z.shape[-2] != old: raise ValueError('Expected square source fields')
    a = overlap_matrix(old, new_size).to(z)
    w = area_weights(old, spacing).to(z)[0, 0]
    return (a @ (z*w) @ a.T)/(a @ w @ a.T)


def eligibility(base):
    # Marine terrain, shelves and already-low plains stay at their input height.
    return ((base-OBJECTIVE['protected_max_m'])/OBJECTIVE['feather_range_m']).clamp(0, 1)


def blocks(x, factor):
    b, c, h, w = x.shape
    if h % factor or w % factor: raise ValueError('Footprint does not divide the field')
    return x.reshape(b, c, h//factor, factor, w//factor, factor)


def project(raw, base, factor=CONSERVE_FACTOR, spacing=SPACING, feather=False):
    """Redistribute elevation locally; lock shallow cells and preserve sign/means.

    The mean being preserved is the input *interpolated baseline's* footprint
    mean. Interpolation itself can differ from the original coarse value. There
    is no planet-wide offset, sea-level change, coverage forcing or histogram fit.
    Projection is idempotent (apart from rounding) when feather=False.
    """
    if raw.shape != base.shape: raise ValueError('Raw and baseline grids differ')
    w = eligibility(base)
    r = (raw-base)*(w if feather else (w > 0))
    area = area_weights(base.shape[-1], spacing).to(base).expand_as(base)
    denominator = blocks(w*area, factor).sum((3, 5), keepdim=True).clamp_min(1e-12)
    correction = blocks(r*area, factor).sum((3, 5), keepdim=True)/denominator
    r = r-w*(correction.expand_as(blocks(r, factor)).reshape_as(r))
    # Limit negative corrections per cell, then balance positive mass. A single
    # shallow valley must not attenuate all other valleys in its footprint.
    negative = torch.maximum(r.clamp_max(0), (1.-base).clamp_max(0))
    negative = torch.where(w > 0, negative, torch.zeros_like(negative))
    positive = r.clamp_min(0)
    amount = -blocks(negative*area, factor).sum((3, 5), keepdim=True)
    total = blocks(positive*area, factor).sum((3, 5), keepdim=True).clamp_min(1e-12)
    scale = (amount/total).clamp(0, 1)
    r = negative+positive*scale.expand_as(blocks(r, factor)).reshape_as(r)
    return torch.where(w > 0, base+r, base)


def target_for(raw_metres, coarse):
    base = upsample_condition(decode(coarse), FACTOR)
    return encode(project(raw_metres, base, feather=True)), base


class StructureNorm:
    def __init__(self, stats): self.s = stats

    def condition(self, coarse):
        base = upsample_condition(decode(coarse), FACTOR)
        up = encode(base)
        return base, up, torch.cat(((up-self.s['cond_mean'])/self.s['cond_std'], 2*eligibility(base)-1), 1)

    def residual(self, target, up): return (target-up)/self.s['res_std']*.5

    def target(self, x, up): return up+x/.5*self.s['res_std']


class StructureData:
    def __init__(self, folder, seed=42):
        from .train import BridgeData
        self.folder = Path(folder)
        self.meta = json.loads((self.folder/'manifest.json').read_text())
        if self.meta.get('version') != VERSION or self.meta.get('objective') != OBJECTIVE:
            raise ValueError('Wrong structure representation/objective')
        self.source = BridgeData(self.meta['array_source'], seed)
        if manifest_id(self.source.meta) != self.meta['source_dataset_id']:
            raise ValueError('Structure source manifest changed')
        for name, expected in self.meta['source_array_fingerprints'].items():
            stat = (Path(self.meta['array_source'])/name).stat()
            if expected != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                raise ValueError(f'Structure source array changed: {name}')
        progress = json.loads((self.folder/'progress.json').read_text())
        self.arrays = {}
        for split in ('train', 'val'):
            count = self.meta[f'{split}_count']
            if progress['done'][split] != count: raise ValueError('Run structure data preparation to completion first')
            a = np.load(self.folder/f'{split}_metres.npy', mmap_mode='r')
            if a.shape != (count, SIZE, SIZE) or a.dtype != np.float32:
                raise ValueError('Structure target array shape/dtype changed')
            stat = (self.folder/f'{split}_metres.npy').stat()
            if progress.get('array_fingerprints', {}).get(split) != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                raise ValueError('Prepared structure targets changed')
            self.arrays[split] = a
        self.rng = np.random.default_rng(seed)

    def batch(self, count, split='train', indices=None, device='cpu'):
        idx = self.rng.integers(self.meta[f'{split}_count'], size=count) if indices is None else indices
        raw = torch.from_numpy(np.array(self.arrays[split][idx]))[:, None].to(device)
        c, context = [torch.from_numpy(np.array(self.source.arrays[f'{split}_{key}'][idx], dtype=np.float32))[:, None].to(device) for key in ('cond', 'context')]
        return raw, c, context

    def stats(self):
        idx = np.random.default_rng(0).choice(self.meta['train_count'], min(512, self.meta['train_count']), replace=False)
        raw, c, ctx = self.batch(len(idx), indices=idx)
        c = degrade_guides(ctx, c, torch.Generator().manual_seed(42), recipe=STRUCTURE_RECIPE)
        target, base = target_for(raw, c); up = encode(base)
        return {'cond_mean': float(up.mean()), 'cond_std': max(float(up.std()), 1e-3),
                'res_std': max(float((target-up).std()), 1.)}


def build(config):
    return EDMDenoiser(cond_channels=2, **config)


def load(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') != VERSION or ck.get('objective') != OBJECTIVE:
        raise ValueError('Not a matching 40-km structure checkpoint')
    model = build(ck['config']); model.load_state_dict(ck['ema'])
    return model.to(device).eval(), StructureNorm(ck['stats']), ck


def shape_loss(predicted, real, base):
    """Physical 40/80-km slope and curvature supervision, not spectral matching."""
    losses = []
    for factor in (1, 2):
        p, t = [F.avg_pool2d(v/OBJECTIVE['shape_height_scale_m'], factor) for v in (predicted, real)]
        mask = F.avg_pool2d((eligibility(base) > 0).float(), factor) > .99
        for dim in (-1, -2):
            dp, dt = [torch.diff(v, dim=dim)/factor for v in (p, t)]
            m = (mask[..., 1:] & mask[..., :-1]) if dim == -1 else (mask[..., 1:, :] & mask[..., :-1, :])
            slope = ((dp-dt).square()*m).sum()/m.sum().clamp_min(1)
            cp, ct = [torch.diff(v, dim=dim)/factor for v in (dp, dt)]
            m2 = (m[..., 1:] & m[..., :-1]) if dim == -1 else (m[..., 1:, :] & m[..., :-1, :])
            curve = ((cp-ct).square()*m2).sum()/m2.sum().clamp_min(1)
            losses.append(OBJECTIVE['slope_weight']*slope+OBJECTIVE['curvature_weight']*curve)
    return sum(losses)/len(losses)


def loss(model, norm, raw, coarse):
    target, base = target_for(raw, coarse)
    _, up, cin = norm.condition(coarse); x = norm.residual(target, up)
    sigma = (torch.randn(len(x), device=x.device)*1.2-1.2).exp()
    noisy = x+torch.randn_like(x)*sigma[:, None, None, None]
    denoised = model(noisy, sigma, cin)
    weight = (sigma.square()+.25)/(sigma*.5).square()
    main = (weight[:, None, None, None]*(denoised.float()-x).square()).mean()
    # Shape losses use metre predictions. EDM noise weighting belongs to main.
    predicted = project(decode(norm.target(denoised.float(), up)), base)
    return main+shape_loss(predicted, decode(target), base)


@torch.no_grad()
def sample(model, norm, coarse, steps=32, seed=42, device='cpu'):
    base, up, cin = norm.condition(coarse.to(device))
    def denoise(x, sigma):
        return model(x.float(), sigma.float().expand(len(x)), cin).double()
    with autocast(device):
        x = heun_sample(denoise, up.shape, steps, torch.Generator(device=device).manual_seed(seed), device)
    raw = decode(norm.target(x.float(), up))
    return project(raw, base), base, raw


def blur(x, sigma_cells):
    r = int(np.ceil(3*sigma_cells))
    q = torch.arange(-r, r+1, device=x.device, dtype=x.dtype)
    k = torch.exp(-.5*(q/sigma_cells).square()); k /= k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k[None, None, None, :])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k[None, None, :, None])


def handoff(structure, structure_base, frozen_encoded, coarse):
    """Retain 40-km structure, adding only frozen intermediate's shorter detail."""
    base23 = decode(upsample_condition(coarse, 7))
    detail = decode(frozen_encoded)-base23
    detail = detail-blur(detail, OBJECTIVE['handoff_detail_sigma_m']/23040.)
    delta = F.interpolate(structure-structure_base, size=base23.shape[-2:], mode='bilinear', align_corners=False)
    control = project(base23+detail, base23, factor=HANDOFF_FACTOR, spacing=23040.)
    candidate = project(base23+detail+delta, base23, factor=HANDOFF_FACTOR, spacing=23040.)
    return candidate, control, base23


@torch.no_grad()
def constraint_metrics(z, base, factor, spacing):
    area = area_weights(z.shape[-1], spacing).to(z).expand_as(z)
    drift = blocks((z-base)*area, factor).sum((3, 5))/blocks(area, factor).sum((3, 5))
    protected = eligibility(base) == 0
    return {'added_footprint_mean_drift_max_m': float(drift.abs().max()),
            'added_footprint_mean_drift_rms_m': float(drift.square().mean().sqrt()),
            'protected_height_change_max_m': float((z-base)[protected].abs().max()) if protected.any() else 0.,
            'sign_disagreement_fraction': float(((z > 0) != (base > 0)).float().mean())}
