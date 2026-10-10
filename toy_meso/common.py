"""Shared data normalization, checkpoint loading and patch sampling."""
import json
from pathlib import Path

import numpy as np
import torch

from .model import EDMDenoiser, upsample_condition, heun_sample


def load_dataset(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    return d, meta


def compute_stats(train_target, train_cond, factor, sample=4096):
    """Normalization constants from the training split only."""
    idx = np.random.default_rng(0).choice(len(train_target), min(sample, len(train_target)), replace=False)
    t = torch.from_numpy(train_target[idx].astype(np.float32))[:, None]
    c = torch.from_numpy(train_cond[idx])[:, None]
    up = upsample_condition(c, factor)
    res = t - up
    return {"cond_mean": float(up.mean()), "cond_std": float(up.std()), "res_std": float(res.std())}


class Normalizer:
    def __init__(self, stats, factor, sigma_data=0.5):
        self.s, self.factor, self.sd = stats, factor, sigma_data

    def condition(self, cond_cells):
        """cond_cells (B,1,h,w) encoded elevation -> (upsampled encoded, model input)."""
        up = upsample_condition(cond_cells, self.factor)
        return up, (up - self.s["cond_mean"]) / self.s["cond_std"]

    def residual(self, target, up):
        return (target - up) / self.s["res_std"] * self.sd

    def target(self, x, up):
        return up + x / self.sd * self.s["res_std"]


def build_model(config):
    return EDMDenoiser(sigma_data=0.5, ch=config["ch"], mults=tuple(config["mults"]),
                       blocks=config["blocks"], dropout=config.get("dropout", 0.0))


def load_checkpoint(path, device="cpu", use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    model.eval()
    return model, Normalizer(ck["stats"], ck["meta"]["factor"]), ck


def autocast(device):
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def sample_patches(model, norm, cond_cells, steps=32, seed=0, device="cpu"):
    """Generate encoded-elevation patches for (B,1,h,w) encoded condition cells."""
    cond_cells = cond_cells.to(device)
    up, cin = norm.condition(cond_cells)
    g = torch.Generator(device=device).manual_seed(seed)
    def denoise(x, sigma):
        with autocast(device):
            s = torch.full((x.shape[0],), float(sigma), device=device)
            return model(x.float(), s, cin).double()
    x = heun_sample(denoise, (cond_cells.shape[0], 1) + up.shape[-2:], steps, g, device)
    return norm.target(x, up), up


def pick_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
