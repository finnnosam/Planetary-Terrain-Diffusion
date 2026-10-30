"""Broaden a saved coastline review and show closest aligned training layouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from planet_training.generate_structured import load
from planet_training.structured_data import StructuredData
from planet_training.train_structured import sample
from planet_training.data import layout_target,PaleoRaster,atomic_json
from planet_training.geography import export_pixels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review',help='folder containing frozen reviewed checkpoint, report and masks')
    args=parser.parse_args()
    root=Path(args.review);out=root/'expanded-readiness'
    if (out/'readiness.json').exists(): raise ValueError('Expanded review already exists; inspect it first')
    out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'review.json').read_text())
    masks=np.load(root/'sample-masks.npy')
    torch.set_num_threads(4)
    model,ck,digest=load(root/'reviewed-checkpoint.pt','coast','cuda')
    if digest!=previous['checkpoint_sha256']: raise ValueError('Reviewed snapshot changed')
    fields={(record['seed'],30):field for record,field in zip(previous['samples'],masks)}
    records=[]
    @torch.no_grad()
    def check(seed,percent):
        key=(seed,percent)
        if key not in fields:
            fields[key]=(sample(model,ck['stats'],[percent/100],64,seed)[0,0].cpu().numpy()>.5).astype(np.float32)
        actual=float(model.grid.mean(torch.from_numpy(fields[key])[None,None].cuda())[0,0])*100
        record=dict(seed=seed,requested_land_percent=percent,actual_land_percent=actual,error_pp=actual-percent)
        records.append(record)
        print(f'Seed {seed}, requested {percent}%: {actual:.2f}% land',flush=True)
    seeds=(42,314,2718,867,2026,1776,12345,919,101,202,303,404)
    for seed in seeds: check(seed,30)
    for seed in seeds[:4]:
        for percent in (20,25,35): check(seed,percent)
    summary={}
    for percent in (20,25,30,35):
        values=np.array([r['actual_land_percent'] for r in records if r['requested_land_percent']==percent])
        summary[str(percent)]=dict(samples=len(values),mean=float(values.mean()),minimum=float(values.min()),
                                  maximum=float(values.max()),mean_absolute_error_pp=float(np.abs(values-percent).mean()),
                                  maximum_absolute_error_pp=float(np.abs(values-percent).max()))
    report=dict(checkpoint_sha256=digest,step=ck['step'],sampling_steps=64,samples=records,summary=summary,
                note='Coverage remains learned and approximate. 12 seeds at 30%, four matched seeds at 20/25/35%. No test ages or terrain generation. Source alignment is an approximate comparison, not proof against copying.')
    atomic_json(out/'readiness.json',report)
    data=StructuredData('outputs/planet_training/data/layout')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['#16465e','#a6af70'])
    fig,axes=plt.subplots(4,3,figsize=(17,11),constrained_layout=True)
    for row,(record,field) in enumerate(zip(previous['samples'],masks)):
        index=record['closest_training_record']
        source=layout_target(PaleoRaster(data.source.maps[index]),np.asarray(record['alignment']),data.n,data.meta['supersample'])>0
        axes[row,0].imshow(export_pixels(field,192),cmap=cmap,vmin=0,vmax=1)
        axes[row,1].imshow(export_pixels(source.astype(np.float32),192),cmap=cmap,vmin=0,vmax=1)
        differences=(export_pixels(field,192)>.5).astype(np.int32)+(export_pixels(source.astype(np.float32),192)>.5).astype(np.int32)*2
        axes[row,2].imshow(differences,interpolation='nearest',
                           cmap=ListedColormap(['#16465e','#dfb061','#a986d1','#6d9154']),vmin=0,vmax=3)
        axes[row,0].set_title(f'Generated seed {record["seed"]}')
        axes[row,1].set_title(f'Closest aligned training layout: {record["closest_training_age_ma"]:g} Ma')
        axes[row,2].set_title(f'Approximate land overlap: {record["approx_best_training_land_iou"]:.2f} IoU')
        for ax in axes[row]: ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Copying screen: green common land / gold generated-only / purple source-only / blue ocean')
    fig.savefig(out/'closest-training-comparison.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(17,8),constrained_layout=True)
    for ax,seed in zip(axes.ravel(),seeds):
        field=fields[(seed,30)]
        record=next(r for r in records if r['seed']==seed and r['requested_land_percent']==30)
        ax.imshow(export_pixels(field,128),cmap=cmap,vmin=0,vmax=1)
        ax.set_title(f'Seed {seed}: {record["actual_land_percent"]:.1f}% land')
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Requested 30% land: 12 seeds / 64 sampling steps')
    fig.savefig(out/'twelve-seeds.png',dpi=110);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
