"""Cheap denoising intervention: fixed targets/noise, altered guide channels."""
import argparse
from pathlib import Path

import numpy as np
import torch

from planet_training.adaptation import STRUCTURE_RECIPE, degrade_guides
from planet_training.data import atomic_json, manifest_id
from planet_training.terrain_structure import StructureData, load, target_for, DEFAULT_DATA
from toy_meso.common import autocast


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544/structure-checkpoint.pt')
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--out', default='outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544/analysis/conditioning-check.json')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    data = StructureData(a.data); model, norm, ck = load(a.checkpoint, a.device)
    if ck['dataset_id'] != manifest_id(data.meta): raise ValueError('Checkpoint/data identity mismatch')
    raw, c, ctx = data.batch(0, 'val', np.arange(data.meta['val_count']), a.device)
    c = degrade_guides(ctx, c, torch.Generator(device=a.device).manual_seed(2026), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    target, _ = target_for(raw, c); _, up, cin = norm.condition(c)
    x = norm.residual(target, up)
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(42)).to(a.device)
    flat = cin.clone(); flat[:, 0] = 0
    shuffled_height = cin.clone(); shuffled_height[:, 0] = cin[permutation, 0]
    variants = {'correct': cin, 'flat_height_same_mask': flat,
                'shuffled_height_same_mask': shuffled_height, 'shuffled_both_channels': cin[permutation]}
    results = []
    with torch.no_grad():
        for sigma in [.05, .5, 2., 8., 20.]:
            g = torch.Generator(device=a.device).manual_seed(314)
            noise = torch.randn(x.shape, generator=g, device=a.device)
            noisy = x+noise*sigma; row = {'sigma': sigma, 'weighted_residual_mse': {}}
            for label, condition in variants.items():
                total = 0.
                for start in range(0, len(x), 16):
                    sl = slice(start, min(start+16, len(x)))
                    with autocast(a.device):
                        pred = model(noisy[sl], torch.full((len(noisy[sl]),), sigma, device=a.device), condition[sl])
                    weight = (sigma*sigma+.25)/(sigma*.5)**2
                    total += float(weight*(pred.float()-x[sl]).square().mean())*len(noisy[sl])/len(x)
                row['weighted_residual_mse'][label] = total
            results.append(row)
            print(row, flush=True)
    atomic_json(a.out, {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': ck['step'], 'dataset_id': ck['dataset_id'],
        'patches': len(x), 'method': 'Complete geographic holdout; sigma-one degraded guides. Each variant denoises identical normalized targets and identical noise. Flat/shuffled height leaves eligibility mask unchanged; shuffled-both changes both channels. Target residual baseline is always the original correct guide.',
        'results': results, 'limits': 'Counterfactual denoising sensitivity, not generation quality or a valid replacement guide. Wrong guides can be out of distribution. Seed correlation alone does not prove conditioning is ignored.',
        'training': False, 'model_generation': False, 'full_planet_generation': False})
    return 0


if __name__ == '__main__': raise SystemExit(main())
