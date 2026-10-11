"""Compact EDM-preconditioned U-Net (Karras et al. 2022) for conditional refinement."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_condition(cond_cells, factor, margin=None):
    """Cell-centre bilinear upsampling, then drop the margin cells.

    cond_cells: (B, C, h, w) with one margin cell on each side (default).
    Returns (B, C, (h-2)*factor, (w-2)*factor).
    """
    margin = factor if margin is None else margin
    up = F.interpolate(cond_cells, scale_factor=factor, mode="bilinear", align_corners=False)
    return up[..., margin:up.shape[-2] - margin, margin:up.shape[-1] - margin]


class FourierEmbedding(nn.Module):
    def __init__(self, channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(channels // 2) * scale)

    def forward(self, x):
        x = x.float()[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, cin // 4), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.affine = nn.Linear(emb, 2 * cout)
        self.norm2 = nn.GroupNorm(min(32, cout // 4), cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.affine(emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        h = self.conv2(self.drop(h))
        return (h + self.skip(x)) / math.sqrt(2)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch // 4), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.proj(out)) / math.sqrt(2)


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, ch=64, mults=(1, 2, 4), blocks=2, attn_last=True, dropout=0.0):
        super().__init__()
        emb = 4 * ch
        self.fourier = FourierEmbedding(ch)
        self.embed = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans, c = [ch], ch
        for level, m in enumerate(mults):
            for _ in range(blocks):
                self.down.append(ResBlock(c, ch * m, emb, dropout))
                c = ch * m
                chans.append(c)
            if level < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb, dropout),
                                  Attention(c) if attn_last else nn.Identity(),
                                  ResBlock(c, c, emb, dropout)])
        self.up = nn.ModuleList()
        for level, m in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(c + chans.pop(), ch * m, emb, dropout))
                c = ch * m
            if level > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(min(32, c // 4), c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, noise_label):
        emb = self.embed(self.fourier(noise_label))
        h = self.inp(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.out(h)


class EDMDenoiser(nn.Module):
    """D(x; sigma, cond) with EDM preconditioning. x is the normalized residual."""
    def __init__(self, sigma_data=0.5, cond_channels=1, **unet):
        super().__init__()
        self.sigma_data = sigma_data
        self.net = UNet(in_ch=1 + cond_channels, **unet)

    def forward(self, x, sigma, cond):
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        f = self.net(torch.cat([c_in * x, cond], dim=1), sigma.flatten().log() / 4)
        return c_skip * x + c_out * f.to(x.dtype)


def edm_loss(model, x, cond, p_mean=-1.2, p_std=1.2):
    sigma = (torch.randn(x.shape[0], device=x.device) * p_std + p_mean).exp()
    noisy = x + torch.randn_like(x) * sigma.reshape(-1, 1, 1, 1)
    denoised = model(noisy, sigma, cond)
    sd = model.sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
    return (weight.reshape(-1, 1, 1, 1) * (denoised.float() - x) ** 2).mean()


def sigma_schedule(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    i = torch.arange(steps, device=device, dtype=torch.float64)
    s = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)])


@torch.no_grad()
def heun_sample(denoise, shape, steps=32, generator=None, device="cpu", initial=None):
    """EDM deterministic Heun sampler. denoise(x, sigma_scalar) -> D(x)."""
    t = sigma_schedule(steps, device=device)
    if initial is None:
        initial = torch.randn(shape, generator=generator, device=device, dtype=torch.float64)
    x = initial.to(device=device, dtype=torch.float64) * t[0]
    for i in range(steps):
        d = (x - denoise(x, t[i])) / t[i]
        x_next = x + (t[i + 1] - t[i]) * d
        # The schedule ends in zero; testing a CUDA scalar here would force a
        # CPU/GPU synchronization on every step.
        if i + 1 < steps:
            d2 = (x_next - denoise(x_next, t[i + 1])) / t[i + 1]
            x_next = x + (t[i + 1] - t[i]) * (d + d2) / 2
        x = x_next
    return x.float()
