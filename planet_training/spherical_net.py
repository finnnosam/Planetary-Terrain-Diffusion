"""A shared-weight cube-face U-Net with geometric halos at every convolution.

Faces are spatial charts, not input channels. Halo values are bilinear samples
on the neighbouring chart; feature seams are identified before halo reads.
This is a cubed-sphere CNN, not an exactly rotation-equivariant network.
"""
import math
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from planet_diffusion import cube
from .data import node_weights
from toy_meso.model import FourierEmbedding


@lru_cache(maxsize=24)
def chart_tables(n):
    side = n + 1
    owner = np.arange(6 * side * side)
    members, ids, owners, _ = cube.edge_groups(n)
    owner[members] = owners[ids]
    _, groups = np.unique(owner, return_inverse=True)
    q = np.arange(-1, side + 1)
    p = np.stack([cube.directions(f, q[:,None], q[None,:], n) for f in range(6)])
    face, y, x = cube.coordinates(p, n)
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    jy, jx = np.minimum(iy+1,n), np.minimum(ix+1,n)
    fy, fx = y-iy, x-ix
    index = np.stack([(face*side+a)*side+b for a,b in ((iy,ix),(iy,jx),(jy,ix),(jy,jx))])
    weight = np.stack(((1-fy)*(1-fx),(1-fy)*fx,fy*(1-fx),fy*fx))
    # Interior nodes should be copied exactly, rather than recomputed through
    # roundoff in spherical coordinates.
    interior = np.arange(6*side*side).reshape(6,side,side)
    index[:, :, 1:-1, 1:-1] = interior[None]
    weight[:, :, 1:-1, 1:-1] = 0
    weight[0, :, 1:-1, 1:-1] = 1
    return groups, np.bincount(groups), index.reshape(4,-1), weight.reshape(4,-1)


class Chart(nn.Module):
    def __init__(self, n):
        super().__init__()
        if n < 2: raise ValueError('Cube charts require at least two intervals')
        self.n = n
        groups, counts, index, weight = chart_tables(n)
        members,edge_ids,_,edge_counts = cube.edge_groups(n)
        halo = np.ones((6,n+3,n+3),bool)
        halo[:,1:-1,1:-1] = False
        halo_positions = np.flatnonzero(halo.ravel())
        for name, value in [('groups',groups),('counts',counts.astype(np.float32)),
                            ('members',members),('edge_ids',edge_ids),('edge_counts',edge_counts.astype(np.float32)),
                            ('halo_positions',halo_positions),('index',index[:,halo_positions]),
                            ('weight',weight[:,halo_positions].astype(np.float32))]:
            self.register_buffer(name,torch.from_numpy(value),persistent=False)
        self.register_buffer('area',torch.from_numpy(node_weights(n))[None,None],persistent=False)

    def project(self,x):
        b,c,faces,h,w = x.shape
        if (faces,h,w) != (6,self.n+1,self.n+1): raise ValueError('Feature shape does not match cube chart')
        # Keep shared-node accumulation and halo interpolation in float32
        # even when neighbouring convolution outputs are bfloat16 under AMP.
        core = x.float().reshape(b,c,-1)
        sums = core.new_zeros((b,c,len(self.edge_counts)))
        sums.scatter_add_(2,self.edge_ids[None,None].expand(b,c,-1),core[...,self.members])
        result = core.clone()
        result[...,self.members] = (sums/self.edge_counts)[...,self.edge_ids]
        return result.reshape_as(x)

    def halo(self,x):
        x = self.project(x)
        core = x.reshape(x.shape[0],x.shape[1],-1)
        ring = sum(core[...,self.index[i]] * self.weight[i] for i in range(4))
        result = x.new_zeros((x.shape[0],x.shape[1],6,self.n+3,self.n+3))
        result[...,1:-1,1:-1] = x
        result.reshape(x.shape[0],x.shape[1],-1)[...,self.halo_positions] = ring
        return result

    def noise(self,batch,channels,device,generator=None):
        x = torch.randn(batch,channels,len(self.counts),device=device,generator=generator)
        return x[...,self.groups].reshape(batch,channels,6,self.n+1,self.n+1)

    def mean(self,x):
        return (x*self.area).sum((-3,-2,-1))/self.area.sum()


def faces_to_batch(x):
    b,c,f,h,w = x.shape
    return x.permute(0,2,1,3,4).reshape(b*f,c,h,w)


def batch_to_faces(x,batch):
    _,c,h,w = x.shape
    return x.reshape(batch,6,c,h,w).permute(0,2,1,3,4)


def norm_layer(channels):
    groups = min(16,channels//4)
    while channels % groups: groups -= 1
    return nn.GroupNorm(groups,channels)


class CubeConv(nn.Module):
    def __init__(self,cin,cout,n,kernel=3):
        super().__init__()
        self.chart = Chart(n) if kernel==3 else None
        self.conv = nn.Conv2d(cin,cout,kernel)

    def forward(self,x):
        batch = x.shape[0]
        if self.chart is not None: x = self.chart.halo(x)
        return batch_to_faces(self.conv(faces_to_batch(x)),batch)


class CubeBlock(nn.Module):
    def __init__(self,cin,cout,n,embedding,dropout):
        super().__init__()
        self.norm1,self.norm2 = norm_layer(cin),norm_layer(cout)
        self.conv1,self.conv2 = CubeConv(cin,cout,n),CubeConv(cout,cout,n)
        self.affine = nn.Linear(embedding,2*cout)
        self.skip = CubeConv(cin,cout,n,kernel=1) if cin!=cout else nn.Identity()
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.conv2.conv.weight); nn.init.zeros_(self.conv2.conv.bias)

    def normalized(self,x,norm):
        return batch_to_faces(norm(faces_to_batch(x)),x.shape[0])

    def forward(self,x,embedding):
        h = self.conv1(F.silu(self.normalized(x,self.norm1)))
        scale,shift = self.affine(embedding).reshape(x.shape[0],-1,1,1,1).chunk(2,dim=1)
        h = F.silu(self.normalized(h,self.norm2)*(1+scale)+shift)
        return (self.skip(x)+self.conv2(self.drop(h)))/math.sqrt(2)


class GlobeAttention(nn.Module):
    """One attention population across all six coarse face charts."""
    def __init__(self,ch):
        super().__init__()
        self.norm = norm_layer(ch)
        self.qkv,self.proj = nn.Conv1d(ch,3*ch,1),nn.Conv1d(ch,ch,1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self,x):
        b,c = x.shape[:2]
        flat = x.reshape(b,c,-1)
        q,k,v = self.qkv(self.norm(flat)).reshape(b,3,4,c//4,-1).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1,-2),k.transpose(-1,-2),v.transpose(-1,-2))
        h = h.transpose(-1,-2).reshape(b,c,-1)
        return (flat+self.proj(h)).reshape_as(x)/math.sqrt(2)


def resize_faces(x,n):
    # All resolutions share the same chart endpoints. align_corners=True
    # keeps features attached to those physical endpoints in the pyramid.
    return batch_to_faces(F.interpolate(faces_to_batch(x),size=(n+1,n+1),
                                       mode='bilinear',align_corners=True),x.shape[0])


class SphericalUNet(nn.Module):
    def __init__(self,n=62,in_ch=2,out_ch=1,ch=24,dropout=.1):
        super().__init__()
        self.ns = [n,n//2,n//4]
        if self.ns[-1]<2 or ch<8 or ch%4: raise ValueError('Use n>=8 and channels divisible by four')
        self.charts = nn.ModuleList([Chart(i) for i in self.ns])
        emb = 4*ch
        self.fourier = FourierEmbedding(ch)
        self.embed = nn.Sequential(nn.Linear(ch+1,emb),nn.SiLU(),nn.Linear(emb,emb))
        widths = [ch,2*ch,4*ch]
        self.inp = CubeConv(in_ch,ch,n)
        self.down = nn.ModuleList([CubeBlock(ch if i==0 else widths[i-1],widths[i],self.ns[i],emb,dropout) for i in range(3)])
        self.mid = nn.ModuleList([CubeBlock(widths[-1],widths[-1],self.ns[-1],emb,dropout),
                                 GlobeAttention(widths[-1]),
                                 CubeBlock(widths[-1],widths[-1],self.ns[-1],emb,dropout)])
        self.up = nn.ModuleList([CubeBlock(widths[i]+(widths[i+1] if i<2 else widths[i]),widths[i],self.ns[i],emb,dropout) for i in range(3)])
        self.outnorm,self.out = norm_layer(ch),CubeConv(ch,out_ch,n)
        nn.init.zeros_(self.out.conv.weight); nn.init.zeros_(self.out.conv.bias)

    def forward(self,x,noise_label,coverage):
        emb = self.embed(torch.cat((self.fourier(noise_label),(coverage-.3)[:,None]*5),dim=1))
        h = self.inp(x); skips=[]
        for i,block in enumerate(self.down):
            if i: h = self.charts[i].project(resize_faces(h,self.ns[i]))
            h = block(h,emb); skips.append(h)
        for layer in self.mid: h = layer(h,emb) if isinstance(layer,CubeBlock) else layer(h)
        for i in reversed(range(3)):
            if i<2: h = self.charts[i].project(resize_faces(h,self.ns[i]))
            h = self.up[i](torch.cat((h,skips[i]),dim=1),emb)
        h = batch_to_faces(F.silu(self.outnorm(faces_to_batch(h))),x.shape[0])
        return self.charts[0].project(self.out(h))


class CoastReliefDenoiser(nn.Module):
    def __init__(self,stage,n=62,ch=24,dropout=.1,coverage_prior=False):
        super().__init__()
        if stage not in ('coast','relief'): raise ValueError('Stage must be coast or relief')
        self.stage,self.sigma_data = stage,.5
        # False preserves old checkpoints for exact review. New coast runs
        # explicitly opt in through their saved configuration.
        self.coverage_prior = coverage_prior
        self.grid = Chart(n)
        self.net = SphericalUNet(n,in_ch=1 if stage=='coast' else 2,ch=ch,dropout=dropout)

    def forward(self,x,sigma,coverage,coast=None):
        x = self.grid.project(x)
        s = sigma[:,None,None,None,None].to(x.dtype)
        sd = self.sigma_data
        # Under uniform whole-sphere rotation the clean marginal land
        # probability is the known area coverage. Center EDM around that
        # conditional mean, rather than around an implicit 50/50 prior.
        prior = (coverage[:,None,None,None,None]-.5).to(x.dtype) if self.stage=='coast' and self.coverage_prior else 0.
        inputs = (x-prior)/(s*s+sd*sd).sqrt()
        if self.stage=='relief':
            if coast is None: raise ValueError('Relief requires a coastline field')
            inputs = torch.cat((inputs,self.grid.project(coast)-.5),dim=1)
        f = self.net(inputs,sigma.log()/4,coverage)
        d = prior+sd*sd/(s*s+sd*sd)*(x-prior)+s*sd/(s*s+sd*sd).sqrt()*f.float()
        return self.grid.project(d)
