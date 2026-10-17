"""Resumable preparation of 40-km modern terrain from verified 23-km cell means."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import atomic_json, manifest_id
from .run_lock import training_lock
from .terrain_structure import (VERSION, SIZE, SPACING, FACTOR, OBJECTIVE,
                                DEFAULT_DATA, conservative_resample, decode)
from .train import BridgeData

SOURCE = 'outputs/planet_training/data/bridge-adapted'


def file_info(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def prepare(source=SOURCE, out=DEFAULT_DATA, train_count=None, val_count=None, chunk=64):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out: raise ValueError('Use a new structure dataset directory')
    data = BridgeData(source, 0)
    if (data.meta['size'], data.meta['factor'], data.meta['spacing_m']) != (112, 7, 23040.) or not data.meta.get('guide_recipe'):
        raise ValueError('Expected verified 112-cell adapted modern bridge data with geographic context')
    counts = {'train': train_count or data.meta['train_count'], 'val': val_count or data.meta['val_count']}
    if chunk < 1 or any(not 1 <= counts[s] <= data.meta[f'{s}_count'] for s in counts):
        raise ValueError('Invalid preparation counts/chunk')
    files = [f'{s}_{k}.npy' for s in counts for k in ('target', 'cond', 'context')]
    meta = {'version': VERSION, 'objective': OBJECTIVE, 'size': SIZE, 'factor': FACTOR,
            'spacing_m': SPACING, 'coarse_spacing_m': 161280.,
            'array_source': str(source), 'source_dataset_id': manifest_id(data.meta),
            'source_array_fingerprints': {f: file_info(source/f) for f in files},
            'train_count': counts['train'], 'val_count': counts['val'],
            'geographic_split': 'unchanged source train/val membership and verified context guard',
            'source_holdout_caps': data.meta['holdout_caps_lat_lon_radius_m'],
            'conversion': 'exact 7:4 cell interval overlaps in physical metres, weighted by 23-km cell-centre spherical Jacobian; source signed-root float16 decoding first',
            'limits': '40-km targets approximate averages of existing verified 23-km modern cell means, not fresh ETOPO quadrature. No new terrain information introduced. First N source patches retain source ordering.'}
    meta = json.loads(json.dumps(meta))
    out.mkdir(parents=True, exist_ok=True)
    manifest, progress_path = out/'manifest.json', out/'progress.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != meta: raise ValueError('Preparation settings/source changed; use a fresh output directory')
    else:
        if any(p.name not in ('training.lock', 'prepare-console.log') for p in out.iterdir()):
            raise ValueError('New preparation directory contains unrelated files')
        atomic_json(manifest, meta)
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'done': {'train': 0, 'val': 0}}
    if any(not 0 <= progress['done'][s] <= counts[s] for s in counts): raise ValueError('Invalid preparation progress')
    if progress['done'] == counts:
        for split in counts:
            if progress.get('array_fingerprints', {}).get(split) != file_info(out/f'{split}_metres.npy'):
                raise ValueError('Completed structure targets changed; use a fresh data directory')
        print('40-km structure data already prepared.', flush=True)
        return meta
    started = time.perf_counter(); prepared = 0
    for split, count in counts.items():
        path = out/f'{split}_metres.npy'
        if progress['done'][split] and not path.exists(): raise ValueError('Missing partially prepared target array')
        a = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32, shape=(count, SIZE, SIZE))
        if a.shape != (count, SIZE, SIZE) or a.dtype != np.float32: raise ValueError('Prepared target shape/dtype changed')
        try:
            for start in range(progress['done'][split], count, chunk):
                stop = min(start+chunk, count)
                z = torch.from_numpy(np.array(data.arrays[f'{split}_target'][start:stop], dtype=np.float32))
                converted = conservative_resample(decode(z), SIZE).numpy()
                if not np.isfinite(converted).all(): raise FloatingPointError('Nonfinite modern targets')
                a[start:stop] = converted; a.flush()
                progress['done'][split] = stop
                atomic_json(progress_path, progress)
                prepared += stop-start; elapsed = time.perf_counter()-started
                remaining = sum(counts[s]-progress['done'][s] for s in counts)
                print(f"Prepared {sum(progress['done'].values()):,}/{sum(counts.values()):,} patches; {prepared/max(elapsed,.001):.1f}/s; ETA {remaining*elapsed/max(prepared,1)/60:.1f} min", flush=True)
        finally:
            a.flush(); del a
    progress['array_fingerprints'] = {s: file_info(out/f'{s}_metres.npy') for s in counts}
    atomic_json(progress_path, progress)
    print(f'40-km data ready: {counts["train"]:,} train, {counts["val"]:,} held out. Source arrays unchanged.', flush=True)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=SOURCE); p.add_argument('--out', default=DEFAULT_DATA)
    p.add_argument('--train-count', type=int); p.add_argument('--val-count', type=int)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    if a.threads < 1 or any(v is not None and v < 1 for v in (a.train_count, a.val_count)): p.error('Counts/threads must be positive')
    torch.set_num_threads(a.threads)
    if a.smoke:
        if a.out == DEFAULT_DATA: a.out = 'outputs/structure/checks/data-smoke'
        a.train_count, a.val_count = 16, 4
    with training_lock(a.out):
        return prepare(a.source, a.out, a.train_count, a.val_count)


if __name__ == '__main__': main()
