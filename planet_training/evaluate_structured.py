"""Cheap continent-only quality gate, before expensive downstream generation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import manifest_id,atomic_json,layout_target,PaleoRaster
from .generate_structured import load
from .geography import measures,export_pixels,spectrum
from .structured_data import StructuredData,STRUCTURED_VERSION,decode_relief,DEFAULT_COAST_RUN,DEFAULT_RELIEF_RUN
from .evaluate_layout import CopyScreen
from .train_structured import evaluate,sample
from toy_meso.common import pick_device
from toy_meso.sphere import decode


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coast',default=DEFAULT_COAST_RUN+'/best.pt')
    p.add_argument('--relief',default=DEFAULT_RELIEF_RUN+'/best.pt')
    p.add_argument('--data',default='outputs/planet_training/data/layout')
    p.add_argument('--out',default='outputs/continents/structured-evaluation')
    p.add_argument('--seeds',default='42,314,2718,867')
    p.add_argument('--land-percent',type=float,default=30)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='auto')
    a=p.parse_args(argv)
    if not np.isfinite(a.land_percent) or not 0<a.land_percent<100 or a.steps<2 or a.batch<1: p.error('Invalid evaluation settings')
    seeds=[int(v) for v in a.seeds.split(',')]
    if not seeds or any(s<0 or s>=2**63 for s in seeds): p.error('Seeds must be nonnegative 63-bit integers')
    torch.set_num_threads(4);device=pick_device(a.device)
    cm,cc,csha=load(a.coast,'coast',device);rm,rc,rsha=load(a.relief,'relief',device)
    data=StructuredData(a.data)
    identity=manifest_id(dict(source=data.meta,representation=STRUCTURED_VERSION))
    if cc['dataset_id']!=identity or rc['dataset_id']!=identity: raise ValueError('Evaluation data must match both checkpoints')
    lo,hi=cc['stats']['training_coverage_min']*100,cc['stats']['training_coverage_max']*100
    if not lo<=a.land_percent<=hi: raise ValueError('Requested coverage is outside the training distribution')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    coast,magnitude=data.validation(rc['stats'],device)
    report=dict(version=STRUCTURED_VERSION,dataset_id=identity,coast_sha256=csha,relief_sha256=rsha,
                coast_step=cc['step'],relief_step=rc['step'],requested_land_percent=a.land_percent,
                sea_level_m=0,sampler_steps=a.steps,
                validation=dict(coast=evaluate(cm,coast,magnitude,cc['stats'],a.batch),
                                relief=evaluate(rm,coast,magnitude,rc['stats'],a.batch)),references={},samples=[])
    for split in ('train','val'):
        records=[]
        for i in data.source.splits[split]:
            z=decode(layout_target(PaleoRaster(data.source.maps[i]),np.eye(3),data.n,data.meta['supersample']))
            records.append(dict(age_ma=data.meta['records'][i]['age_ma'],**measures(z)))
        report['references'][split]=records
    copy_screen=CopyScreen(data.source.maps,data.source.splits['train'],data.meta['records'])
    fields=[]
    for seed in seeds:
        requested=torch.tensor([a.land_percent/100],device=device)
        coast=(sample(cm,cc['stats'],requested,a.steps,seed)>.5).float()
        magnitude=sample(rm,rc['stats'],rm.grid.mean(coast)[:,0],a.steps,seed+1,coast)
        z=decode_relief(coast[0,0].cpu().numpy(),magnitude[0,0].cpu().numpy()).astype(np.float32)
        metrics=measures(z)
        metrics.update(seed=seed,land_coverage_error_percentage_points=metrics['land_fraction']*100-a.land_percent,
                       spherical_power=spectrum(z))
        metrics.update(copy_screen.match(z))
        report['samples'].append(metrics);fields.append(z)
        print(f'Seed {seed}: {metrics["land_fraction"]*100:.1f}% land, {metrics["large_landmass_count"]} large land masses, shelves {metrics["shelf_planet_fraction"]*100:.1f}%',flush=True)
    report['limits']='New losses are not comparable with the old elevation-only loss. Test ages remain untouched. Learned coverage is approximate. Inspect coastal profiles, broad relief and fragmentation. Rotation/reflection similarity uses 2048 equal-area probes and local alignment refinement: it is an approximate copying screen, not proof against copying. Native coast signs are fixed before decoding, while interpolation and downstream refinement can move shorelines. No full intermediate or meso globe was generated.'
    atomic_json(out/'evaluation.json',report)
    np.save(out/'sample-cubes.npy',np.stack(fields))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(len(fields),2,figsize=(12,3.5*len(fields)),constrained_layout=True,squeeze=False)
    for row,(z,r) in enumerate(zip(fields,report['samples'])):
        axes[row,0].imshow(export_pixels((z>0).astype(np.float32),256),cmap=ListedColormap(['#16465e','#a6af70']),vmin=0,vmax=1)
        axes[row,1].imshow(export_pixels(z,256),cmap='terrain',vmin=-6000,vmax=6000)
        axes[row,0].set_title(f'Seed {r["seed"]}: {r["land_fraction"]*100:.1f}% land / requested {a.land_percent:.1f}%')
        axes[row,1].set_title(f'Shelves {r["shelf_planet_fraction"]*100:.1f}% of globe / land p95 {r["land_p95_m"] or 0:,.0f} m')
        for ax in axes[row]: ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Continent-only quality gate: coastline geometry first, relief around sea level 0 m')
    fig.savefig(out/'comparison.png',dpi=120);plt.close(fig)
    print(f'Saved {out}/evaluation.json and comparison.png; no downstream globe generated.',flush=True)


if __name__=='__main__': main()
