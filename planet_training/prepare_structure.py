"""Reuse verified read-only bridge arrays with an explicit unchanged-target recipe."""
import argparse
import json
from pathlib import Path

from .adaptation import PROFILE, RECIPE, STRUCTURE_RECIPE
from .data import atomic_json, manifest_id
from .train import BridgeData

DEFAULT_DATA = 'outputs/planet_training/data/bridge-structure-v1'


def prepare(source='outputs/planet_training/data/bridge-adapted', out=DEFAULT_DATA):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError('Use a separate structure dataset folder')
    data = BridgeData(source, 0)
    if data.meta.get('guide_recipe') != RECIPE or data.meta.get('array_source'):
        raise ValueError(f'Expected the original {PROFILE} adapted bridge data')
    files = [f'{split}_{key}.npy' for split in ('train','val')
             for key in ('target','cond','context')]
    fingerprints = {name:{'bytes':(source/name).stat().st_size,
                          'mtime_ns':(source/name).stat().st_mtime_ns} for name in files}
    meta = dict(data.meta, guide_recipe=STRUCTURE_RECIPE,
                parent_dataset_id=manifest_id(data.meta),
                clean_parent_dataset_id=data.meta['parent_dataset_id'],
                array_source=str(source), array_source_dataset_id=manifest_id(data.meta),
                array_fingerprints=fingerprints,
                preparation='read-only references to verified adapted bridge arrays; no target modification')
    meta = json.loads(json.dumps(meta))
    out.mkdir(parents=True,exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != meta:
            raise ValueError('Structure preparation changed; choose a new dataset folder')
    else:
        if any(out.iterdir()):
            raise ValueError('New structure dataset folder must be empty')
        atomic_json(manifest,meta)
    atomic_json(out/'progress.json',{'done':{s:meta[f'{s}_count'] for s in ('train','val')}})
    BridgeData(out,0)
    print(f"Structure data ready: {meta['train_count']:,} training / {meta['val_count']:,} held-out patches. Original arrays reused read-only.",flush=True)
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/planet_training/data/bridge-adapted')
    p.add_argument('--out',default=DEFAULT_DATA)
    a=p.parse_args(argv);prepare(a.source,a.out)


if __name__=='__main__':main()
