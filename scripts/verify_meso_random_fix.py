"""Measure the production random source and render matching sampled tiles."""
import json
from pathlib import Path

import numpy as np
import torch

from toy_meso.conditioning import RandomElevation
from toy_meso.common import load_checkpoint
from toy_meso.planet_tiled import Grid, TileCache, compute_tiles, tile_condition, ensure_coarse_condition
from toy_meso.sphere import encode, decode

OUT = Path('outputs/meso/random-diagnosis')
settings = json.loads(Path('human_input/MESO1/settings.json').read_text())
keys = [tuple(k) for k in json.loads((OUT / 'diagnosis.json').read_text())['tile_keys']]
grid = Grid(settings['fine_intervals'], settings['tile'], settings['stride'], settings['factor'])
folder = OUT / 'production-v2'
folder.mkdir(exist_ok=True)
condition = encode(ensure_coarse_condition(folder, grid, keys, lambda: RandomElevation(settings['seed'])))
model, norm, ck = load_checkpoint('outputs/meso/run2-lowlr/latest.pt', 'cuda')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cache = TileCache(folder / 'tiles')
compute_tiles(keys, model, norm, condition, 8, grid, settings['seed'], cache, 32, 16, 'cuda')
generated = decode(np.stack([cache.load(k) for k in keys]))
baseline = decode(np.stack([tile_condition(condition, 8, grid, *k) for k in keys]))
land = baseline > 0
delta = generated - baseline
result = dict(sampled_tiles=len(keys), land_model_delta_rms_m=float(np.sqrt(np.mean(delta[land]**2))))
(folder / 'result.json').write_text(json.dumps(result, indent=2))
np.savez(folder / 'values.npz', generated=generated, baseline=baseline)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from toy_meso.evaluate import shaded
old = np.load(OUT / 'old.npz')
# Representative interior land tile; same grid location/seed across both guides.
i = 5
fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
for ax, values, title in zip(axes, [old['baseline'][i], old['generated'][i], baseline[i], generated[i]],
                            ['Old random input', 'Old model output', 'Revised random input', 'Revised model output']):
    ax.imshow(shaded(values, 2880), interpolation='nearest')
    ax.set_title(title)
    ax.set_axis_off()
fig.savefig(OUT / 'random-fix-comparison.png', dpi=150)
print(json.dumps(result, indent=2), flush=True)
