"""Time a fresh regional generation against the frozen implementation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rasterio
import torch

from scripts.profile_meso_speed import ROOT, OUT, baseline
from toy_meso import planet_tiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('baseline', 'current'))
    parser.add_argument('--source', choices=('map', 'random', 'mixed'), default='mixed')
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    old_common, old_tiled = baseline()
    old_tiled.load_checkpoint = old_common.load_checkpoint
    sampler = old_tiled if args.variant == 'baseline' else planet_tiled
    timings = {}
    for name in ('load_checkpoint', 'coarse_condition_cells', 'ensure_coarse_condition',
                 'compute_tiles', 'blend', 'blend_region'):
        if not hasattr(sampler, name):
            continue
        original = getattr(sampler, name)

        def timed(*a, _name=name, _original=original, **kw):
            start = time.perf_counter()
            result = _original(*a, **kw)
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            timings[_name] = time.perf_counter() - start
            return result

        setattr(sampler, name, timed)
    folder = OUT / f'region-{args.variant}-{args.source}'
    argv = ['--run', str(ROOT / 'outputs/meso/run2-lowlr'), '--source', args.source,
            '--seed', '42', '--out', str(folder), '--steps', '32', '--batch', '16',
            '--bounds', '5', '44', '7', '46', '--width', '64', '--height', '64',
            '--device', 'cuda', '--name', 'cached.tif' if args.cached else 'region.tif']
    if args.source != 'random':
        argv += ['--dem', str(ROOT / 'human_input/B2W.png'), '--black-metres', '-6000',
                 '--white-metres', '6000', '--map-weight', '.75']
    start = time.perf_counter()
    sampler.main(argv)
    elapsed = time.perf_counter() - start
    result = dict(variant=args.variant, source=args.source, cached=args.cached,
                  total_seconds=elapsed, stages=timings,
                  tiles=len(list((folder / 'tiles').glob('f*/*.npy'))))
    if args.variant == 'current':
        reference = OUT / f'region-baseline-{args.source}'
        with rasterio.open(reference / 'region.tif') as old, rasterio.open(folder / 'region.tif') as new:
            a, b = old.read(1), new.read(1)
        result['export_identical'] = bool(np.array_equal(a, b))
        result['max_difference_m'] = float(np.max(np.abs(a - b)))
        result['all_tiles_identical'] = all(
            np.array_equal(np.load(path), np.load(reference / path.relative_to(folder)))
            for path in (folder / 'tiles').glob('f*/*.npy'))
        valid = np.load(folder / 'coarse-valid.npy')
        result['used_input_identical'] = bool(np.array_equal(
            np.load(folder / 'coarse.npy')[0][valid], np.load(reference / 'coarse.npy')[0][valid]))
        result['input_cells_prepared'] = int(valid.sum())
        result['input_cells_total'] = int(valid.size)
    (OUT / f'region-{args.variant}-{args.source}{"-cached" if args.cached else ""}.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
