"""Compare tiled inference against a frozen pre-optimization implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from toy_meso import common, model as model_module, planet_tiled as tiled
from toy_meso.sphere import encode, decode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/meso/speed-verification'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def baseline():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('model', 'common', 'planet_tiled'):
        path = OUT / f'baseline_{name}.py'
        if not path.exists():
            path.write_bytes((ROOT / 'toy_meso' / f'{name}.py').read_bytes())
    old_model = module('toy_meso._speed_model', OUT / 'baseline_model.py')
    old_common = module('toy_meso._speed_common', OUT / 'baseline_common.py')
    old_common.build_model = lambda cfg: old_model.EDMDenoiser(
        sigma_data=.5, ch=cfg['ch'], mults=tuple(cfg['mults']), blocks=cfg['blocks'], dropout=cfg.get('dropout', 0))
    old_common.heun_sample = old_model.heun_sample
    old_tiled = module('toy_meso._speed_tiled', OUT / 'baseline_planet_tiled.py')
    old_tiled.heun_sample = old_model.heun_sample
    old_tiled.autocast = old_common.autocast
    return old_common, old_tiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('baseline', 'current', 'channels-last', 'graphs', 'autotune'), default='baseline')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    old_common, old_tiled = baseline()
    helper = old_common if args.variant == 'baseline' else common
    sampler = old_tiled if args.variant == 'baseline' else tiled
    torch.backends.cudnn.benchmark = args.variant == 'autotune'
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    network, norm, checkpoint = helper.load_checkpoint(ROOT / 'outputs/meso/run2-lowlr/latest.pt', 'cuda')
    if args.variant == 'channels-last':
        network.to(memory_format=torch.channels_last)
        original = network

        class ChannelsLast:
            def __call__(self, x, sigma, condition):
                return original(x.to(memory_format=torch.channels_last), sigma,
                                condition.to(memory_format=torch.channels_last))

        network = ChannelsLast()
    if args.variant == 'graphs':
        from scripts.meso_gpu_experiments import CUDAForward
        network = CUDAForward(network, (args.batch, 1, checkpoint['meta']['size'], checkpoint['meta']['size']), 'cuda')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    load_allocated = torch.cuda.memory_allocated()
    settings = json.loads((ROOT / 'outputs/meso/planet-fixed/settings.json').read_text())
    grid = sampler.Grid(settings['fine_intervals'], settings['tile'], settings['stride'], settings['factor'])
    condition = encode(np.load(ROOT / 'outputs/meso/planet-fixed/coarse.npy'))
    paths = sorted((ROOT / 'outputs/meso/planet-fixed/tiles').glob('f*/*.npy'))
    keys = [(int(p.parent.name[1:]), *map(int, p.stem.split('_'))) for p in paths[:args.batch]]
    if len(keys) < args.batch:
        keys = grid.all_tiles()[:args.batch]
    stamp = time.time_ns()
    # Warm convolution kernels separately from the measured sampling runs.
    warm = sampler.TileCache(OUT / f'{args.variant}-{stamp}-warm')
    sampler.compute_tiles(keys, network, norm, condition, grid.factor, grid, 0, warm, 8, args.batch, 'cuda')
    times = []
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(args.repeat):
        cache = sampler.TileCache(OUT / f'{args.variant}-{stamp}-{repeat}')
        torch.cuda.synchronize()
        started = time.perf_counter()
        sampler.compute_tiles(keys, network, norm, condition, grid.factor, grid, 0, cache, args.steps, args.batch, 'cuda')
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    arrays = np.stack([cache.load(key) for key in keys])
    path = OUT / f'{args.variant}-steps{args.steps}-batch{args.batch}.npz'
    np.savez(path, values=arrays, keys=np.array(keys))
    result = dict(variant=args.variant, steps=args.steps, batch=args.batch, tiles=len(keys),
                  gpu=torch.cuda.get_device_name(), load_seconds=load_seconds,
                  loaded_gpu_mib=load_allocated / 2**20, peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                  seconds=times, median_seconds=float(np.median(times)), tiles_per_second=len(keys) / float(np.median(times)),
                  values=str(path))
    reference_path = OUT / f'baseline-steps{args.steps}-batch{args.batch}.npz'
    if args.variant != 'baseline' and reference_path.exists():
        reference = np.load(reference_path)['values']
        result['encoded_identical'] = bool(np.array_equal(reference, arrays))
        difference = decode(reference) - decode(arrays)
        result['max_elevation_difference_metres'] = float(np.max(np.abs(difference)))
        result['rms_elevation_difference_metres'] = float(np.sqrt(np.mean(difference**2)))
    (OUT / f'{args.variant}-steps{args.steps}-batch{args.batch}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
