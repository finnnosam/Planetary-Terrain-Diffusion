"""Continuous local mean correction for regional meso review; no model changes."""
import torch
import torch.nn.functional as F

from .terrain_structure import area_weights, blocks, eligibility

VERSION = 'meso-smooth-conservation-v1'


@torch.no_grad()
def project_smooth(raw, base, factor=8, spacing=2880., return_info=False):
    """Match input footprint means using a continuous bilinear correction field.

    Unlike a constant offset independently applied inside each footprint, the
    correction is interpolated between coarse coefficients. A small linear
    system chooses those coefficients so actual area-weighted means match.
    Shallow elevations are locked. If positivity needs a limit, one global
    convex scale reduces the correction without introducing block boundaries.
    This is generation-time meso review only; structure v1 targets are unchanged.
    """
    if raw.shape != base.shape or raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError('Expected matching Bx1xHxW fields')
    if raw.shape[-1] != raw.shape[-2] or raw.shape[-1] % factor:
        raise ValueError('Expected a divisible square footprint grid')
    if not torch.isfinite(raw).all() or not torch.isfinite(base).all():
        raise FloatingPointError('Nonfinite conservation input')
    outputs, info = [], []
    side = raw.shape[-1]//factor
    for source, baseline in zip(raw, base):
        baseline = baseline[None].double(); source = source[None].double()
        w = eligibility(baseline); area = area_weights(raw.shape[-1], spacing).to(baseline)
        r = (source-baseline)*w
        support = blocks(w*area, factor).sum((3, 5)).flatten()
        active = torch.nonzero(support > 0, as_tuple=False).flatten()
        if not len(active):
            outputs.append(baseline.to(raw.dtype)); info.append({'positivity_scale': 1., 'active_footprints': 0}); continue
        # Columns are smooth coefficient basis fields; rows are footprint means.
        basis = torch.zeros((len(active), side*side), device=raw.device, dtype=torch.float64)
        basis[torch.arange(len(active), device=raw.device), active] = 1.
        basis = F.interpolate(basis.reshape(-1, 1, side, side), size=raw.shape[-2:], mode='bilinear', align_corners=False)
        matrix = blocks(basis*w*area, factor).sum((3, 5))[:, 0].flatten(1).T[active]
        rhs = -blocks(r*area, factor).sum((3, 5)).flatten()[active]
        # Row normalization keeps small coastal supports numerically well scaled.
        matrix = matrix/support[active, None]; rhs = rhs/support[active]
        coefficients = torch.linalg.solve(matrix, rhs)
        grid = torch.zeros(side*side, device=raw.device, dtype=torch.float64)
        grid[active] = coefficients
        correction = F.interpolate(grid.reshape(1, 1, side, side), size=raw.shape[-2:], mode='bilinear', align_corners=False)
        r = r+w*correction
        ratio = torch.where(r < 0, (baseline-1.).clamp_min(0)/(-r).clamp_min(1e-12), torch.ones_like(r))
        scale = ratio.amin().clamp(0, 1)
        z = torch.where(w > 0, baseline+scale*r, baseline)
        outputs.append(z.to(raw.dtype))
        info.append({'positivity_scale': float(scale), 'active_footprints': len(active),
                     'correction_rms_m': float((w*correction).square().mean().sqrt())})
    result = torch.cat(outputs)
    return (result, info) if return_info else result


def grid_jumps(field, factor=8):
    """Boundary versus interior neighbour jumps; descriptive, not a terrain score."""
    z = field.detach().double()
    boundary, inside = [], []
    for dim in (-1, -2):
        delta = torch.diff(z, dim=dim).abs()
        index = torch.arange(delta.shape[dim], device=z.device)
        selected = (index+1) % factor == 0
        shape = [1]*z.ndim; shape[dim] = len(selected)
        mask = selected.reshape(shape).expand_as(delta)
        if mask.any(): boundary.append(delta[mask].mean())
        if (~mask).any(): inside.append(delta[~mask].mean())
    b, i = torch.stack(boundary).mean(), torch.stack(inside).mean()
    return {'footprint_boundary_mean_jump_m': float(b), 'interior_mean_jump_m': float(i),
            'boundary_to_interior_ratio': float(b/i.clamp_min(1e-12))}
