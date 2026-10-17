"""A joint six-face EDM model, with differentiable shared-edge identification."""
import numpy as np
import torch
from torch import nn

from planet_diffusion import cube
from toy_meso.model import UNet
from .data import LAYOUT_N, node_weights


class SphereGrid(nn.Module):
    def __init__(self, n=LAYOUT_N):
        super().__init__()
        self.n = n
        self.side = (n + 1 + 7) // 8 * 8
        total = 6 * (n + 1) ** 2
        owner = np.arange(total)
        members, ids, owners, counts = cube.edge_groups(n)
        owner[members] = owners[ids]
        _, groups = np.unique(owner, return_inverse=True)
        sizes = np.bincount(groups)
        lookups = []
        i = np.arange(self.side)
        for face in range(6):
            f, y, x = cube.coordinates(cube.directions(face, i[:, None], i[None, :], n), n)
            index = (f * (n + 1) + np.floor(y+.5).astype(int)) * (n + 1) + np.floor(x+.5).astype(int)
            lookups.append(groups[index])
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.long), persistent=False)
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.float32), persistent=False)
        self.register_buffer("lookup", torch.tensor(np.stack(lookups).ravel(), dtype=torch.long), persistent=False)
        # One independent Gaussian per physical node, including identical edge copies.
        self.register_buffer("weights", torch.from_numpy(node_weights(n))[None], persistent=False)

    def project(self, x):
        core = x[..., :self.n+1, :self.n+1].reshape(x.shape[0], -1)
        sums = core.new_zeros((x.shape[0], len(self.sizes)))
        sums.scatter_add_(1, self.groups[None].expand(x.shape[0], -1), core)
        shared = sums / self.sizes
        return shared[:, self.lookup].reshape(x.shape[0], 6, self.side, self.side)

    def noise(self, batch, device, generator=None):
        unique = torch.randn(batch, len(self.sizes), device=device, generator=generator)
        return unique[:, self.lookup].reshape(batch, 6, self.side, self.side)

    def weighted_mean(self, values):
        core = values[..., :self.n+1, :self.n+1]
        return (core * self.weights).sum() / (self.weights.sum() * core.shape[0])


class LayoutDenoiser(nn.Module):
    def __init__(self, n=LAYOUT_N, ch=32, mults=(1, 2, 4), blocks=2, dropout=.1):
        super().__init__()
        self.sigma_data = .5
        self.grid = SphereGrid(n)
        self.net = UNet(in_ch=6, out_ch=6, ch=ch, mults=mults, blocks=blocks, dropout=dropout)

    def forward(self, x, sigma):
        x = self.grid.project(x)
        s = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        f = self.net(x / (s*s + sd*sd).sqrt(), sigma.log() / 4)
        d = sd*sd / (s*s + sd*sd) * x + s*sd / (s*s + sd*sd).sqrt() * f.float()
        return self.grid.project(d)


def layout_loss(model, x):
    sigma = (torch.randn(x.shape[0], device=x.device) * 1.2 - 1.2).exp()
    noisy = x + model.grid.noise(x.shape[0], x.device) * sigma[:, None, None, None]
    d = model(noisy, sigma)
    weight = (sigma**2 + .25) / (sigma * .5)**2
    return model.grid.weighted_mean((d - x)**2 * weight[:, None, None, None])
