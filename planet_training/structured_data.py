"""Separate coastline geometry and unsigned relief, with a fixed zero datum.

The label is land/ocean at the existing area-averaged ~161 km resolution.
The magnitude model learns sqrt(abs(elevation)); the coastline supplies its
sign before any elevation is decoded. Neither sampling stage shifts sea level.
"""
import numpy as np
import torch

from .data import node_weights
from .layout import SphereGrid
from .train import LayoutData

STRUCTURED_VERSION = 'coast-relief-sphere-v1'
OBJECTIVE = 'conditioned-area-edm-v2'
DEFAULT_COAST_RUN = 'outputs/continents/structured-coast/run2-unbiased'
DEFAULT_RELIEF_RUN = 'outputs/continents/structured-relief/run2-unbiased'


def split_encoded(encoded):
    return (encoded > 0).astype(np.float32), np.abs(encoded).astype(np.float32)


def decode_relief(coast,magnitude):
    coast,magnitude = np.asarray(coast),np.asarray(magnitude)
    if coast.shape != magnitude.shape or not np.isfinite(coast).all() or not np.isfinite(magnitude).all():
        raise ValueError('Coast and magnitude must be matching finite arrays')
    # Sign is supplied by geometry. Squaring is the inverse of the magnitude
    # encoding; no global height shift, histogram remapping or sea-level edit.
    return np.where(coast>.5,1.,-1.) * magnitude.astype(np.float32)**2


class StructuredData:
    def __init__(self,folder,seed=42):
        self.source = LayoutData(folder,seed)
        self.meta,self.n = self.source.meta,self.source.n
        self.rng = self.source.rng
        self.legacy_grid = None

    def stats(self):
        from .data import PaleoRaster,layout_target
        weights = node_weights(self.n)
        first = second = 0.
        coverages = []
        for i in self.source.splits['train']:
            values = layout_target(PaleoRaster(self.source.maps[i]),np.eye(3),self.n,self.meta['supersample'])
            coast,magnitude = split_encoded(values)
            first += float((magnitude*weights).sum()/weights.sum())
            second += float((magnitude*magnitude*weights).sum()/weights.sum())
            coverages.append(float((coast*weights).sum()/weights.sum()))
        count = len(coverages)
        mean = first/count
        return dict(mean=mean,std=max(1e-6,(second/count-mean*mean)**.5),
                    training_coverage_min=min(coverages),training_coverage_max=max(coverages),
                    training_coverages=coverages)

    def encode_batch(self,values,stats,device):
        coast = (values>0).float()[:,None]
        magnitude = (values.abs()-stats['mean'])/stats['std']*.5
        return coast,magnitude[:,None]

    def batch(self,count,stats,device):
        if str(device).startswith('cuda'):
            if self.legacy_grid is None: self.legacy_grid = SphereGrid(self.n).to(device)
            values = self.source.cuda_batch(count,self.legacy_grid,dict(mean=0.,std=1.),device)*2
            values = values[...,:self.n+1,:self.n+1]
        else:
            values = torch.from_numpy(self.source.batch(count)).to(device)
        return self.encode_batch(values,stats,device)

    def validation(self,stats,device):
        values = torch.from_numpy(self.source.validation()).to(device)
        return self.encode_batch(values,stats,device)
