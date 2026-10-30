"""Compare matched coastline reviews without rerunning any generation."""
import argparse
import json
from pathlib import Path
import numpy as np
from planet_training.geography import export_pixels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='outputs/continents/coast-correction-comparison')
    parser.add_argument('reviews',nargs='+',help='review.json files in display order')
    parser.add_argument('--labels',default='Old objective,Corrected warm start,Fresh restart')
    args=parser.parse_args()
    reports=[json.loads(Path(path).read_text()) for path in args.reviews]
    fields=[np.load(Path(path).parent/'sample-masks.npy') for path in args.reviews]
    labels=args.labels.split(',')
    if len(labels)!=len(reports): parser.error('Supply one label per review')
    seeds=[record['seed'] for record in reports[0]['samples']]
    for report in reports:
        if [record['seed'] for record in report['samples']]!=seeds:
            raise ValueError('Reviews must use matching seeds')
        if report.get('sampling_steps',64)!=64:
            raise ValueError('Reviews must use 64 sampling steps')
        if any(record['requested_land_percent']!=30 for record in report['samples']):
            raise ValueError('Reviews must request the same 30% land')
    summary=[]
    for label,path,report in zip(labels,args.reviews,reports):
        achieved=np.array([record['actual_land_percent'] for record in report['samples']])
        summary.append(dict(label=label,review=str(Path(path).resolve()),
                            checkpoint_sha256=report['checkpoint_sha256'],
                            actual_land_percent=achieved.tolist(),
                            mean_absolute_error_pp=float(np.mean(np.abs(achieved-30))),
                            maximum_absolute_error_pp=float(np.max(np.abs(achieved-30))),
                            coverage_sweep_same_seed=report['coverage_sweep_same_seed']))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(dict(requested_land_percent=30,seeds=seeds,
        sampling_steps=64,runs=summary,note='Small calibration screen. The warm start has 10,000 prior updates; fresh restart has none. Do not equate training maturity or compare the old loss numerically.'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(len(seeds),len(reports),figsize=(6*len(reports),3*len(seeds)),
                          constrained_layout=True,squeeze=False)
    for col,(label,report,cubes) in enumerate(zip(labels,reports,fields)):
        for row,(record,cube) in enumerate(zip(report['samples'],cubes)):
            axes[row,col].imshow(export_pixels(cube,192),cmap=ListedColormap(['#16465e','#a6af70']),vmin=0,vmax=1)
            axes[row,col].set_title(f'{label}\nSeed {record["seed"]}: {record["actual_land_percent"]:.1f}% land')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    fig.suptitle('Matched seeds, 30% requested land, 64 sampling steps; green land / blue ocean')
    fig.savefig(out/'matched-coastlines.png',dpi=110);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    ax.plot([15,40],[15,40],'--',color='gray',label='Requested = actual')
    for row in summary:
        sweep=row['coverage_sweep_same_seed']
        ax.plot([point['requested'] for point in sweep],[point['achieved'] for point in sweep],
                'o-',label=row['label'])
    ax.set_xlabel('Requested land (% of spherical surface)');ax.set_ylabel('Actual land (%)')
    ax.set_title('Same starting noise: coverage response, seed 42')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'coverage-response.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
