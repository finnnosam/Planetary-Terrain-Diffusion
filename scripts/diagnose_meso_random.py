"""Compare smoother and multiscale random guides with this trained model."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from planet_diffusion import procedural
from toy_meso.conditioning import RandomElevation
from toy_meso.common import load_checkpoint, load_dataset
from toy_meso.planet_tiled import Grid, TileCache, compute_tiles, tile_condition, ensure_coarse_condition
from toy_meso.sphere import encode, decode
from toy_meso.evaluate import radial_psd

OUT = Path('outputs/meso/random-diagnosis')
OUT.mkdir(exist_ok=True)
reference = Path('human_input/MESO1')
settings = json.loads((reference / 'settings.json').read_text())
grid = Grid(settings['fine_intervals'], settings['tile'], settings['stride'], settings['factor'])
old_coarse = np.load(reference / 'coarse.npy')
old_enc = encode(old_coarse)
all_keys = grid.all_tiles()
candidates = [key for key in all_keys[::7] if np.mean(decode(tile_condition(old_enc, 8, grid, *key)) > 0) > .95]
keys = [candidates[i] for i in np.linspace(0, len(candidates)-1, 16).astype(int)]
network, norm, checkpoint = load_checkpoint('outputs/meso/run2-lowlr/latest.pt', 'cuda')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
dataset, _ = load_dataset('outputs/meso/dataset.npz')
real = decode(dataset['val_cond'][:1024])

def guide_stats(c):
    dy, dx = np.diff(c, axis=-2), np.diff(c, axis=-1)
    ly = (c[...,1:,:] > 0) & (c[...,:-1,:] > 0)
    lx = (c[...,:,1:] > 0) & (c[...,:,:-1] > 0)
    return dict(land_neighbour_delta_rms_m=float(np.sqrt(np.mean(np.r_[dy[ly], dx[lx]]**2))))

report = {'training_input': guide_stats(real), 'tile_keys': keys, 'variants': {}}
for label, gain in [('old', None), ('gain-060', .60), ('gain-070', .70), ('gain-080', .80)]:
    source = RandomElevation(settings['seed'])
    if gain is None:
        coarse = old_coarse
    else:
        source.noise.fractal_octaves = 9
        source.noise.fractal_gain = gain
        coords = np.random.default_rng(0).uniform(0, 32768, (3, 1024*1024)).astype(np.float32)
        source.noise_quantiles = procedural.build_quantiles(source.noise.gen_from_coords(coords))
        folder = OUT / label
        folder.mkdir(exist_ok=True)
        coarse = ensure_coarse_condition(folder, grid, keys, lambda: source)
    condition = encode(coarse)
    cache = TileCache(OUT / label / 'tiles')
    start = time.perf_counter()
    compute_tiles(keys, network, norm, condition, 8, grid, settings['seed'], cache, 32, 16, 'cuda')
    seconds = time.perf_counter() - start
    generated = decode(np.stack([cache.load(k) for k in keys]))
    baseline = decode(np.stack([tile_condition(condition, 8, grid, *k) for k in keys]))
    land = baseline > 0
    delta = generated - baseline
    cells = np.stack([coarse[0, f, grid.starts[ky]//8-1+grid.coarse_halo: grid.starts[ky]//8+17+grid.coarse_halo,
                              grid.starts[kx]//8-1+grid.coarse_halo: grid.starts[kx]//8+17+grid.coarse_halo]
                      for f, ky, kx in keys])
    wl, p = radial_psd(generated, 2880)
    _, b = radial_psd(baseline, 2880)
    result = dict(seconds=seconds, guide=guide_stats(cells), land_fraction=float(land.mean()),
                  land_model_delta_rms_m=float(np.sqrt(np.mean(delta[land]**2))),
                  fine_power_model_over_input=float(p[wl < 23040].sum()/b[wl < 23040].sum()))
    report['variants'][label] = result
    np.savez(OUT / f'{label}.npz', generated=generated, baseline=baseline, keys=keys)
    print(label, json.dumps(result), flush=True)
(OUT / 'diagnosis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
