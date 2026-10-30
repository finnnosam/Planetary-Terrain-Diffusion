"""Saved-array duration comparison; no training or sampling."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from planet_training.terrain_spatial_plan import PlanCodec,continuous_fields
from scripts.audit_terrain_plan_targets import plan_stats
from scripts.diagnose_terrain_structure import image,legend

ROOT=Path(__file__).resolve().parents[1]/'outputs/structure/spatial-supervision-v1'
MODES=['categorical','continuous']
CASES=['alps','california','plains','holdout-rugged','holdout-quiet','generated-60']


def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))


def panel(ax,z,title,spacing,seen):
    key=z.tobytes()
    if key in seen:
        ax.text(.5,.5,'Identical field to\n'+seen[key],ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
    else:
        seen[key]=title;half=len(z)*spacing/2000
        ax.imshow(image(z,spacing),extent=(-half,half,-half,half),interpolation='nearest')
        ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
    ax.set_title(title,fontsize=9)


@torch.no_grad()
def compare(before,after):
    torch.set_num_threads(4);out=after/'analysis';out.mkdir(exist_ok=True)
    folders=[before,after];settings=[json.loads((p/'settings.json').read_text()) for p in folders]
    results=[json.loads((p/'results.json').read_text()) for p in folders]
    if any(x['status']!='complete' for x in results):raise ValueError('Both reviews must be complete')
    for key in ['renderer_sha256','renderer_updates','source_sha256','dataset_id','meso_sha256','cases','renderer_seeds','plan_seeds','meso_seeds']:
        if settings[0][key]!=settings[1][key]:raise ValueError('Unmatched review: '+key)
    checkpoints=[torch.load(p/'planner-pair.pt',map_location='cpu',weights_only=False) for p in folders]
    for key in ['spec','source_sha256','dataset_id','config','stats','plan_stats','normalization_source','optimization']:
        if checkpoints[0][key]!=checkpoints[1][key]:raise ValueError('Unmatched planner identity: '+key)
    codec=PlanCodec(checkpoints[0]['plan_stats']);steps=[x['planner_updates'] for x in settings]
    lookup=[{(r['case'],r['kind'],r['mode'],r['plan_seed'],r['renderer_seed']):r for r in x['records']} for x in results]
    summary={'before':str(before.resolve()),'after':str(after.resolve()),'planner_updates':steps,
        'renderer_updates':settings[0]['renderer_updates'],'renderer_seeds':settings[0]['renderer_seeds'],
        'plan_seeds':settings[0]['plan_seeds'],'meso_seeds':settings[0]['meso_seeds'],
        'limits':'Regional monitored diagnostics. Plan/render agreement, variability, connectivity and RMS changes are not terrain-realism scores or world-frequency estimates.',
        'checks':{'same_downstream_models':True,'same_planner_training_identity':True,'supplied_structure_reference_bit_exact':True,'supplied_meso_bit_exact':True,'all_saved_arrays_finite':True},'cases':{}}
    for key,row in lookup[0].items():
        a=dict(np.load(before/row['file']));b=dict(np.load(after/lookup[1][key]['file']))
        for name in ['coarse_encoded','baseline40_metres']:np.testing.assert_array_equal(a[name],b[name])
        if row['kind']=='supplied':
            for name in ['raw40_metres','projected40_metres','direct23_metres','encoded_plan']:np.testing.assert_array_equal(a[name],b[name])
            for ma,mb in zip(row['meso'],lookup[1][key]['meso']):
                np.testing.assert_array_equal(np.load(before/ma['file'])['raw_meso_metres'],np.load(after/mb['file'])['raw_meso_metres'])
    for folder in folders:
        for path in folder.glob('*.npz'):
            with np.load(path) as a:
                if not all(np.isfinite(a[k]).all() for k in a.files):raise ValueError(path)
    for case in CASES:
        summary['cases'][case]={}
        for mode in MODES:
            response=[];fine_response=[];diagnostics=[]
            for ps in settings[0]['plan_seeds']:
                for rs in settings[0]['renderer_seeds']:
                    rows=[l[case,'autonomous',mode,ps,rs] for l in lookup]
                    fields=[dict(np.load(p/r['file'])) for p,r in zip(folders,rows)]
                    response.append(rms((fields[1]['projected40_metres']-fields[0]['projected40_metres'])[0,0,16:48,16:48]))
                    for ma,mb in zip(rows[0]['meso'],rows[1]['meso']):
                        fine_response.append(rms(np.load(after/mb['file'])['raw_meso_metres']-np.load(before/ma['file'])['raw_meso_metres']))
                    if rs!=settings[0]['renderer_seeds'][0]:continue  # Each plan is counted once.
                    for time_index,a in enumerate(fields):
                        base=torch.from_numpy(a['baseline40_metres']);physical=codec.decode(torch.from_numpy(a['raw_generated_encoded_plan']),base,mode)
                        mask=base[0,0,16:48,16:48]>200
                        row={'step':steps[time_index],'plan_seed':ps,'active_cells_in_crop':int(mask.sum())}
                        if mode=='continuous':
                            q=physical[0,:,16:48,16:48]
                            row['negative_relief_fraction']=float((q[3][mask]<0).float().mean()) if mask.any() else None
                            row['physical_field_rms_m']=[rms(v[mask].numpy()) if mask.any() else None for v in q]
                            actual=continuous_fields(torch.from_numpy(a['projected40_metres']),base)[0,:,16:48,16:48]
                            row['own_terrain_field_correlation']=[]
                            for v,w in zip(q,actual):
                                v,w=v[mask],w[mask]
                                corr=float(torch.corrcoef(torch.stack((v,w)))[0,1]) if len(v)>1 and v.std()>1e-6 and w.std()>1e-6 else None
                                row['own_terrain_field_correlation'].append(corr)
                        else:
                            canonical=codec.decode(torch.from_numpy(a['encoded_plan']),base,mode)
                            row['categorical_plan']=plan_stats(canonical,base)
                        diagnostics.append(row)
            variations=[]
            for folder,look in zip(folders,lookup):
                values=[]
                for rs in settings[0]['renderer_seeds']:
                    maps=[np.load(folder/look[case,'autonomous',mode,ps,rs]['file'])['projected40_metres'][0,0,16:48,16:48] for ps in settings[0]['plan_seeds']]
                    values.extend(rms(maps[a]-maps[b]) for a in range(len(maps)) for b in range(a+1,len(maps)))
                variations.append(float(np.mean(values)))
            summary['cases'][case][mode]={'projected_height_change_rms_m_mean_all_4_plan_2_renderer_seeds':float(np.mean(response)),
                'fine_change_rms_m_mean_both_meso_seeds_fixed_first_plan_renderer':float(np.mean(fine_response)) if fine_response else None,
                'guide_variation_between_plan_seeds_rms_m_mean_by_training_step':variations,
                'unique_plan_diagnostics':diagnostics}
    summary['plan_diagnostic_scope']='Central 32x32 at 40.32 km, unique four plans per case/mode/training step, evaluated through renderer seed 7919 only. Plan-seed variation averages six pairs under each of two renderer seeds. No frequency estimates.'
    atomic_json(out/'duration-comparison.json',summary)
    for ms in settings[0]['meso_seeds']:
        fig,axs=plt.subplots(3,5,figsize=(16,10),squeeze=False,constrained_layout=True);seen={}
        for i,case in enumerate(['alps','california','generated-60']):
            ref=after/f'{case}-reference-m{ms}.npz'
            if ref.exists():panel(axs[i,0],np.load(ref)['raw_meso_metres'][0,0],case+'\nSUPPLIED feasible height → meso',2880.,seen)
            else:
                axs[i,0].text(.5,.5,'Generated prefix\nNo paired Earth reference',ha='center',va='center',transform=axs[i,0].transAxes);axs[i,0].set_axis_off()
            for j,(mode,t) in enumerate([(m,t) for m in MODES for t in [0,1]],1):
                row=lookup[t][case,'autonomous',mode,42,7919];entry=next(x for x in row['meso'] if x['seed']==ms)
                z=np.load(folders[t]/entry['file'])['raw_meso_metres'][0,0]
                panel(axs[i,j],z,case+f'\n{mode}, {steps[t]:,} updates',2880.,seen)
        fig.suptitle(f'Planner duration, meso seed {ms}: each 368.64 km\nSame guide, plan seed 42, renderer seed 7919, frozen 1,000-update renderers and selected meso. Fixed colours / NW 5x shade.',fontsize=11)
        legend(fig,axs);fig.savefig(out/f'fine-duration-m{ms}.png',dpi=115);plt.close(fig)
    for case in ['alps','california','generated-60']:
        fig,axs=plt.subplots(4,4,figsize=(13,12.6),squeeze=False,constrained_layout=True);seen={}
        for i,(mode,t) in enumerate((m,t) for m in MODES for t in [0,1]):
            for j,ps in enumerate(settings[0]['plan_seeds']):
                row=lookup[t][case,'autonomous',mode,ps,7919]
                z=np.load(folders[t]/row['file'])['projected40_metres'][0,0,16:48,16:48]
                panel(axs[i,j],z,f'{mode}, {steps[t]:,} updates\nplan seed {ps}',40320.,seen)
        fig.suptitle(f'{case}: all plan seeds, each 1290.24 km\nMatched training-duration comparison, renderer seed 7919; fixed metre colours and shading.',fontsize=11)
        legend(fig,axs);fig.savefig(out/f'{case}-duration.png',dpi=110);plt.close(fig)
    text='<!doctype html><html><head><meta charset="utf-8"><title>Planner duration comparison</title><style>body{font:16px system-ui;background:#f5f5f5;margin:24px}p{max-width:1100px;line-height:1.5}img{width:100%;max-width:1800px}details{margin:25px 0}</style></head><body>'
    text+=f'<h1>Planner training: {steps[0]:,} versus {steps[1]:,} updates</h1><p>Same source, guide, noise, renderer and meso. All four declared plan seeds and both renderer seeds were retained. Primary fine panels use the first declared plan/renderer seed; both meso seeds are available. Supplied heights are references, not autonomous successes. Quiet protected cells are not evidence of learned plains. Fixed metre colours and physical footprints.</p><img src="fine-duration-m113.png"><details><summary>Second meso seed, same comparison</summary><img src="fine-duration-m997.png"></details>'
    for c in ['alps','california','generated-60']:text+=f'<details><summary>{html.escape(c)}: all four plan seeds, before/after</summary><img loading="lazy" src="{c}-duration.png"></details>'
    text+='<p><a href="../index.html">Complete current review: all six inputs and both renderer seeds</a></p></body></html>'
    (out/'index.html').write_text(text,encoding='utf-8');print(f'Comparison saved: {out/"index.html"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',default=str(ROOT/'autonomous-review-1000'));p.add_argument('--after',required=True)
    args=p.parse_args();compare(Path(args.before).resolve(),Path(args.after).resolve())


if __name__=='__main__':main()
