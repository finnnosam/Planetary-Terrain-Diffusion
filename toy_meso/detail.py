"""Sub-cell detail as an optional meso input, and softened heights for training.

Detail of a 23.04 km cell = standard deviation (m) of the 2.88 km terrain about the bilinear 23 km surface inside
that cell: how much valley/ridge relief the meso should add there. Encoded as log1p(detail / 10 m). A detail-aware
meso takes two extra input channels: the normalized detail map and a 'known' mask (0 = unknown: the model picks a
plausible amount itself).

Softening: during training the 23 km heights are sometimes Gaussian-blurred, so the model learns to make Earth-like
detail even when its input does not carry the fine 23 km cue (simulated generators produce such softer fields;
see docs/tectonic-prototype.md).
"""
import torch
import torch.nn.functional as F

from .model import upsample_condition


def decode(e):
    return torch.sign(e)*e.square()


def encode(z):
    return torch.sign(z)*z.abs().sqrt()


def encode_detail(metres):
    return torch.log1p(metres.clamp(min=0)/10.)


def decode_detail(encoded):
    return 10.*torch.expm1(encoded)


@torch.no_grad()
def cell_detail(target_enc, cond_enc, factor=8):
    """(B,1,128,128) target and (B,1,18,18) condition, both encoded -> (B,1,18,18) encoded detail
    (margin cells copied from their neighbours)."""
    res = decode(target_enc.float())-decode(upsample_condition(cond_enc.float(), factor))
    b, _, h, w = res.shape
    blocks = res.reshape(b, 1, h//factor, factor, w//factor, factor)
    std = blocks.std(dim=(3, 5))
    return F.pad(encode_detail(std), (1, 1, 1, 1), mode='replicate')


def soften(cond_enc, sigma):
    """Gaussian-blur each (1,h,w) encoded height grid in metres with its own sigma (cells); reflect edges."""
    b, _, h, w = cond_enc.shape
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=cond_enc.device).reshape(-1).expand(b)
    r = int(torch.ceil(3*sigma.max()).item()) if b else 1
    q = torch.arange(-r, r+1, device=cond_enc.device, dtype=torch.float32)
    k = torch.exp(-.5*(q[None]/sigma[:, None].clamp(min=1e-3)).square()); k = k/k.sum(1, keepdim=True)
    z = decode(cond_enc.float()).reshape(1, b, h, w)
    z = F.conv2d(F.pad(z, (r, r, 0, 0), mode='reflect'), k[:, None, None, :], groups=b)
    z = F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), k[:, None, :, None], groups=b)
    return encode(z.reshape(b, 1, h, w)).to(cond_enc.dtype)


def detail_inputs(norm, detail_cells, shape, device, known=None):
    """Two model-input channels at patch resolution: normalized detail (0 where unknown) and the known mask."""
    b, _, h, w = shape
    if detail_cells is None:
        detail_cells = torch.zeros(b, 1, h, w, device=device)
        known = torch.zeros(b, device=device)
    elif known is None:
        known = torch.ones(b, device=device)
    d = (detail_cells.to(device).float()-norm.s['detail_mean'])/norm.s['detail_std']
    kn = known.to(device).float().reshape(b, 1, 1, 1)
    d = upsample_condition(d*kn, norm.factor)
    return torch.cat([d, kn.expand_as(d)], 1)
