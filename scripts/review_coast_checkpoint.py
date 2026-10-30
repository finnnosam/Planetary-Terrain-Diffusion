"""Review a loaded coastline checkpoint without changing or stopping training."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from planet_training.generate_structured import load
from planet_training.structured_data import StructuredData,DEFAULT_COAST_RUN
from planet_training.train_structured import sample
from planet_training.geography import geometry,measures,export_pixels
from planet_training.evaluate_layout import CopyScreen


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default=DEFAULT_COAST_RUN+'/best.pt')
    parser.add_argument('--out')
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--skip-copy',action='store_true')
    args=parser.parse_args(argv)
    if args.steps<2: parser.error('Use at least two sampling steps')
    torch.set_num_threads(4)
    out=Path(args.out or str(Path('outputs/continents/coast-reviews')/datetime.now().strftime('review-%Y%m%d-%H%M%S-%f')))
    if (out/'review.json').exists(): raise ValueError('Choose a fresh review folder')
    out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'reviewed-checkpoint.pt'
    snapshot.write_bytes(Path(args.checkpoint).read_bytes())
    model,ck,digest=load(snapshot,'coast','cuda')
    data=StructuredData('outputs/planet_training/data/layout')
    coast,_=data.validation(ck['stats'],'cuda')
    preview_targets=model.grid.mean(coast[:2])[:,0]
    unique,_,weights,*_=geometry(model.grid.n)
    preview=sample(model,ck['stats'],preview_targets,steps=24,seed=42)[:,0].cpu().numpy()
    preview_coverage=[float(weights[(z>.5).ravel()[unique]].sum()*100) for z in preview]
    screen=None if args.skip_copy else CopyScreen(data.source.maps,data.source.splits['train'],data.meta['records'])
    records=[];fields=[]
    for seed in (42,314,2718,867):
        scores=sample(model,ck['stats'],[.30],steps=args.steps,seed=seed)[0,0].cpu().numpy()
        field=(scores>.5).astype(np.float32)
        # A +/-1 surrogate supplies geometry to the existing measurements;
        # it is not elevation or a generated terrain guide.
        geographic=measures(field*2-1)
        copying={} if screen is None else screen.match(field*2-1)
        record=dict(seed=seed,requested_land_percent=30,actual_land_percent=geographic['land_fraction']*100,
                    large_landmasses=geographic['large_landmass_count'],
                    largest_landmass_share=geographic['largest_landmass_share_of_land'],**copying)
        records.append(record);fields.append(field)
        print(json.dumps(record),flush=True)
    sweep=[]
    for percent in (20,30,35):
        scores=sample(model,ck['stats'],[percent/100],steps=args.steps,seed=42)[0,0].cpu().numpy()
        achieved=float(weights[(scores>.5).ravel()[unique]].sum()*100)
        sweep.append(dict(requested=percent,achieved=achieved))
    report=dict(step=ck['step'],checkpoint_sha256=digest,checkpoint=str(Path(args.checkpoint).resolve()),
                reviewed_snapshot=str(snapshot.resolve()),
                objective=ck.get('objective','legacy-balanced-v1'),sampling_steps=args.steps,
                config=ck['config'],initialized_from=ck.get('initialized_from'),
                copy_screen_performed=screen is not None,val_loss=ck['best_val_loss'],
                training_preview_targets_percent=(preview_targets*100).cpu().tolist(),
                training_preview_actual_percent=preview_coverage,samples=records,coverage_sweep_same_seed=sweep,
                note='Coastlines only. Four seeds are a small screen; copying alignment is approximate. No training was interrupted and no downstream generation was started.')
    (out/'review.json').write_text(json.dumps(report,indent=2))
    np.save(out/'sample-masks.npy',np.stack(fields))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['#16465e','#a6af70'])
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for ax,field,record in zip(axes.ravel(),fields,records):
        ax.imshow(export_pixels(field,256),cmap=cmap,vmin=0,vmax=1)
        ax.set_title(f'Seed {record["seed"]}: requested 30% land / actual {record["actual_land_percent"]:.1f}%')
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'Coastline checkpoint {ck["step"]:,}: green land / blue ocean; {args.steps} sampling steps')
    fig.savefig(out/'coverage-review.png',dpi=120);plt.close(fig)
    print(f'Saved {out}/review.json and coverage-review.png',flush=True)


if __name__=='__main__':main()
