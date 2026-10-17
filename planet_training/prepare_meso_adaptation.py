"""Crop existing fine terrain while using the saved outer guide cells as real context."""
import argparse
import json
from pathlib import Path
import numpy as np

from .adaptation import MESO_RECIPE
from .data import VERSION, atomic_json, fingerprint, manifest_id


def prepare(source, out, train=16000, val=512):
    source, out = Path(source), Path(out)
    d = np.load(source)
    parent = json.loads(str(d["meta"]))
    factor = parent["factor"]
    old_size = parent["size"]
    halo = MESO_RECIPE["extra_coarse_halo"]
    size = old_size-2*halo*factor
    if size <= 0 or size % 8:
        raise ValueError("Saved guides do not support the required safe fine-terrain crop")
    if min(train, val)<1: raise ValueError("Counts must be positive")
    selected = {s: np.linspace(0, len(d[f"{s}_cond"])-1,
                        min(count, len(d[f"{s}_cond"])), dtype=int).tolist()
                for s, count in (("train",train),("val",val))}
    meta = dict(parent, version=VERSION, kind="bridge", training_crop_size=size,
                train_count=len(selected["train"]), val_count=len(selected["val"]),
                guide_recipe=MESO_RECIPE, legacy_parent_meta_id=manifest_id(parent),
                source_dataset=fingerprint(source), parent_source_indices=selected,
                detail="Centre crop with original real guide halo; original geographic splits unchanged")
    out.mkdir(parents=True, exist_ok=True)
    mpath, ppath = out/'manifest.json', out/'progress.json'
    if mpath.exists() and json.loads(mpath.read_text()) != meta:
        raise ValueError("Meso adaptation settings changed; choose a fresh output folder")
    done=json.loads(ppath.read_text())['done'] if ppath.exists() else {'train':0,'val':0}
    atomic_json(mpath,meta)
    for split, indices in selected.items():
        if done[split]==len(indices): continue
        print(f"Preparing {len(indices):,} {split} fine-terrain crops...",flush=True)
        target=d[f'{split}_target'];cond=d[f'{split}_cond']
        arrays={}
        for key, shape, dtype in [('target',(len(indices),size,size),np.float16),
                                 ('cond',(len(indices),size//factor+2,size//factor+2),np.float32),
                                 ('context',(len(indices),old_size//factor+2,old_size//factor+2),np.float32)]:
            path=out/f'{split}_{key}.npy'
            if done[split] > 0 and not path.exists(): raise ValueError(f'Missing preparation cache: {path}')
            arrays[key]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
        def commit():
            for values in arrays.values():values.flush()
            atomic_json(ppath,{'done':done})
        try:
            for start in range(done[split],len(indices),256):
                ids=indices[start:start+256];stop=start+len(ids)
                margin=halo*factor
                arrays['target'][start:stop]=target[ids,margin:-margin,margin:-margin]
                arrays['cond'][start:stop]=cond[ids,halo:-halo,halo:-halo]
                arrays['context'][start:stop]=cond[ids]
                done[split]=stop;commit()
        finally:commit()
        del target,cond,arrays
    d.close()
    print(f'Meso adaptation data ready: {size}x{size} training crops; {old_size}x{old_size} generation tiles.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/meso/dataset.npz')
    p.add_argument('--out',default='outputs/planet_training/data/meso-adapted')
    p.add_argument('--train',type=int,default=16000)
    p.add_argument('--val',type=int,default=512)
    a=p.parse_args(argv);prepare(a.source,a.out,a.train,a.val)


if __name__=='__main__':main()
