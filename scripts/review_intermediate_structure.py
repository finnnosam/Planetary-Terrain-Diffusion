"""Regional baseline/pilot review; no training, selection changes, or full globes."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch

from planet_training.adaptation import STRUCTURE_PROFILE, STRUCTURE_RECIPE, paired_degradation
from planet_training.data import atomic_json, bridge_patch, manifest_id
from planet_training.prepare_structure import DEFAULT_DATA
from planet_training.run_lock import training_lock, RunAlreadyActive
from planet_training.train import BridgeData
from toy_meso.common import load_checkpoint, sample_patches, pick_device
from toy_meso.sphere import GlobalRaster, decode
from toy_meso.train import validation_loss
from scripts.train_intermediate_structure import RUN, BASELINE, BASELINE_SHA

MESO='outputs/meso/selected-adapted-1000/best.pt'
MESO_SHA='717e1c718fefe51fc0f5066de39618da9cc5a6b034c9df0fb1af09ebcbd7fcf6'
LAYOUT='outputs/continents/structured-previews/reviewed-10000-seed42/layout.tif'
LAYOUT_SHA='6a4f587f49aff675be9a04e90755d03ae2b024a4cd250abdac1a04f0412dff7c'
CASES=[('earth-alps',45.5,9.,'earth'),('earth-california',37.,-120.,'earth'),
       ('earth-plains',42.,-103.,'earth'),('new-highland',23.1221198,-70.1958525,'generated'),
       ('new-upland',-31.8317972,102.3387097,'generated'),
       ('new-plain',30.,-87.,'generated'),('new-coast',-4.2511521,-21.6705069,'generated')]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def frame(lat,lon):
    la,lo=np.radians([lat,lon])
    return (np.array([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)]),
            np.array([-np.sin(lo),np.cos(lo),0.]),
            np.array([-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)]))


def widths(profile,threshold,spacing_km):
    edges=np.diff(np.r_[False,profile>threshold,False].astype(int))
    lengths=(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*spacing_km
    return {'segments':len(lengths),'longest_km':float(max(lengths,default=0))}


def measurements(field,guide,spacing,coarse_encoded=None,factor=None):
    from scripts.diagnose_terrain_structure import gauss
    z=np.asarray(field)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite regional output')
    broad=gauss(z,2.)
    guide_broad=gauss(guide,2.)
    dy,dx=np.gradient(broad,spacing/1000)
    middle=z.shape[0]//2
    result={'range_m':[float(z.min()),float(z.max())],
        'land_fraction':float((z>0).mean()),'mean_m':float(z.mean()),
        'output_minus_guide_rms_m':float(np.sqrt(np.mean((z-guide)**2))),
        'broad_drift_rms_m':float(np.sqrt(np.mean((broad-guide_broad)**2))),
        'broad_smoothing_sigma_m':2*spacing,
        'broad_slope_m_per_km_p50_p90':[float(x) for x in np.percentile(np.hypot(dx,dy),[50,90])],
        'west_east_above_1000m':widths(z[middle],1000,spacing/1000),
        'south_north_above_1000m':widths(z[::-1,middle],1000,spacing/1000)}
    if coarse_encoded is not None:
        coarse=decode(np.asarray(coarse_encoded))[1:-1,1:-1]
        h,w=coarse.shape
        average=z.reshape(h,factor,w,factor).mean(axis=(1,3))
        result['coarse_metre_average_drift_rms_m']=float(np.sqrt(np.mean((average-coarse)**2)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default=RUN)
    p.add_argument('--checkpoint',default='latest.pt',choices=['latest.pt','best.pt'])
    p.add_argument('--data',default=DEFAULT_DATA)
    p.add_argument('--out')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,314,2718,2026])
    p.add_argument('--check-only',action='store_true',help='verify and freeze inputs without sampling or evaluating')
    p.add_argument('--smoke',action='store_true',help='test two regions, one seed, eight sampler steps and two validation patches')
    a=p.parse_args(argv)
    cases=[CASES[1],CASES[3]] if a.smoke else CASES
    steps=8 if a.smoke else 32
    if a.smoke:a.seeds=[42]
    if any(seed<0 or seed>=2**63 for seed in a.seeds) or len(set(a.seeds))!=len(a.seeds):
        p.error('Provide distinct seeds in 0..2^63-1')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=pick_device(a.device)
    data=BridgeData(a.data,0)
    if data.meta['guide_recipe']!=STRUCTURE_RECIPE:raise ValueError('Expected structure data')
    for path,expected in [(BASELINE,BASELINE_SHA),(MESO,MESO_SHA),(LAYOUT,LAYOUT_SHA)]:
        if sha(path)!=expected:raise ValueError(f'Frozen review input changed: {path}')
    out=Path(a.out) if a.out else Path(a.run)/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh review output folder')
    paths={}
    with training_lock(a.run):
        candidate=Path(a.run)/a.checkpoint
        if not candidate.exists():raise ValueError('Run the structure pilot first; checkpoint is missing')
        ck=torch.load(candidate,map_location='cpu',weights_only=False)
        if ck.get('guide_profile')!=STRUCTURE_PROFILE or ck.get('dataset_id')!=manifest_id(data.meta):
            raise ValueError('Pilot checkpoint does not match the structure objective/data')
        pilot_step=ck['step'];del ck
        out.mkdir(parents=True)
        for name,source in [('baseline',BASELINE),('pilot',candidate),('meso',MESO)]:
            target=out/f'{name}-checkpoint.pt';shutil.copyfile(source,target);paths[name]=target
    identity={'created_utc':datetime.now(timezone.utc).isoformat(),'pilot_step':pilot_step,
        'data_id':manifest_id(data.meta),'models':{name:{'file':str(path.resolve()),'sha256':sha(path)} for name,path in paths.items()},
        'layout':LAYOUT,'layout_sha256':LAYOUT_SHA,'seeds':a.seeds,'steps':steps,'smoke_test':a.smoke,
        'bridge_size':112,'bridge_spacing_m':23040.,'meso_size':128,'meso_spacing_m':2880.,
        'cases':cases,'full_planet_generation':False,
        'limits':'Independent north-up gnomonic patches, not production sphere tile blending. Earth guides are smoothed at sigma 1 coarse cell; actual Earth targets stay unchanged. Generated guides are sampled from the frozen layout. Shapes must be reviewed; numerical thresholds are not automatic approval.'}
    atomic_json(out/'review-settings.json',identity)
    print(f'\nREGIONAL STRUCTURE REVIEW\nPilot update {pilot_step}; {len(cases)} regions x {len(a.seeds)} seeds. Output: {out}',flush=True)
    if a.check_only:
        print('Review preflight passed. Inputs frozen; no sampling or validation performed.',flush=True);return 0
    limit=2 if a.smoke else None
    vt,vc,context=[torch.from_numpy(np.array(data.arrays[f'val_{key}'][:limit],dtype=np.float32))[:,None].to(device) for key in ('target','cond','context')]
    clean=BridgeData('outputs/planet_training/data/bridge',0)
    if manifest_id(clean.meta)!=data.meta['clean_parent_dataset_id']:raise ValueError('Clean regression data changed')
    ct,cc=[torch.from_numpy(np.array(clean.arrays[f'val_{key}'][:limit],dtype=np.float32))[:,None].to(device) for key in ('target','cond')]
    models={};validation={}
    for name in ['baseline','pilot']:
        model,norm,ck=load_checkpoint(paths[name],device)
        expected=data.meta['parent_dataset_id'] if name=='baseline' else manifest_id(data.meta)
        if ck.get('dataset_id')!=expected:raise ValueError('Review model/dataset mismatch')
        models[name]=(model,norm);del ck
        smooth_t,smooth_c=paired_degradation(norm,vt,vc,context,torch.Generator(device=device).manual_seed(2026),recipe=STRUCTURE_RECIPE)
        strong_t,strong_c=paired_degradation(norm,vt,vc,context,torch.Generator(device=device).manual_seed(2026),recipe=STRUCTURE_RECIPE,sigma_cells=1.5,clean_fraction=0.)
        validation[name]={}
        for label,t,c in [('clean_holdout',ct,cc),('mixed_holdout',smooth_t,smooth_c),('strong_smooth_holdout',strong_t,strong_c)]:
            print(f'Checking {name}: {label}...',flush=True)
            value=validation_loss(model,norm,t,c,device,seeds=8,chunk=16)
            validation[name][label]={'patches':len(t),'loss':value}
            print(f'  {value:.5f}',flush=True)
    meso,meso_norm,meso_ck=load_checkpoint(paths['meso'],device)
    del meso_ck
    earth=GlobalRaster(np.load('outputs/planet_training/data/cache/etopo-filtered.npy'))
    layout=GlobalRaster.from_geotiff(LAYOUT)
    records=[]
    from scripts.diagnose_terrain_structure import image,legend
    import matplotlib.pyplot as plt
    for case_index,(label,lat,lon,kind) in enumerate(cases):
        if kind=='earth':
            from planet_training.adaptation import coarse_context
            target,coarse=bridge_patch(earth,frame(lat,lon))
            ctx=coarse_context(earth,frame(lat,lon),112)
            _,coarse_t=paired_degradation(models['baseline'][1],torch.from_numpy(target.astype(np.float32))[None,None].to(device),
                torch.from_numpy(coarse)[None,None].to(device),torch.from_numpy(ctx)[None,None].to(device),
                recipe=STRUCTURE_RECIPE,sigma_cells=1.,clean_fraction=0.)
        else:
            _,coarse=bridge_patch(layout,frame(lat,lon))
            coarse_t=torch.from_numpy(coarse)[None,None].to(device);target=None
        for seed in a.seeds:
            columns=[];entry={'region':label,'latitude':lat,'longitude':lon,'kind':kind,'seed':seed,'models':{}}
            for name,(model,norm) in models.items():
                pred,up=sample_patches(model,norm,coarse_t,steps,seed,device)
                z=decode(pred[0,0].float().cpu().numpy());g=decode(up[0,0].cpu().numpy())
                np.save(out/f'{label}-seed{seed}-{name}-intermediate.npy',z)
                condition=torch.from_numpy(pred[0,0,47:65,47:65].float().cpu().numpy())[None,None].to(device)
                fine,fine_up=sample_patches(meso,meso_norm,condition,steps,seed,device)
                f=decode(fine[0,0].float().cpu().numpy());fg=decode(fine_up[0,0].cpu().numpy())
                np.save(out/f'{label}-seed{seed}-{name}-meso.npy',f)
                entry['models'][name]={'intermediate':measurements(z,g,23040.,coarse_t[0,0].cpu().numpy(),7),
                                      'meso':measurements(f,fg,2880.,condition[0,0].cpu().numpy(),8)}
                if target is not None:entry['models'][name]['true_earth_rms_m']=float(np.sqrt(np.mean((z-decode(target.astype(np.float32)))**2)))
                columns.append((name,z,g,f))
            fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
            for row,(name,z,g,f) in enumerate(columns):
                for col,(title,field,spacing) in enumerate([('Coarse guide',g[30:-30,30:-30],23040.),('Intermediate',z[30:-30,30:-30],23040.),('Meso',f,2880.)]):
                    half=field.shape[0]*spacing/2000
                    axs[row,col].imshow(image(field,spacing),extent=(-half,half,-half,half))
                    axs[row,col].set_title(f'{name}: {title} / {2*half:.0f} km');axs[row,col].set_xlabel('km');axs[row,col].set_ylabel('km')
            legend(fig,axs)
            prefix='Smoke test (not terrain-quality evidence): ' if a.smoke else ''
            fig.suptitle(f'{prefix}{label} ({lat:.2f}, {lon:.2f}) / seed {seed} / {steps} steps\nSame guide/noise per stage; colours and NW 5x hillshade fixed. Meso shows central 369 km.')
            fig.savefig(out/f'{label}-seed{seed}.png',dpi=120);plt.close(fig)
            # Raw elevation profiles reveal width and valley changes without shading.
            fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
            for direction,ax in enumerate(axes):
                for name,z,g,f in columns:
                    q=z[30:-30,30:-30];profile=q[26] if direction==0 else q[::-1,26]
                    ax.plot((np.arange(52)+.5-26)*23.04,profile,label=name)
                if target is not None:
                    q=decode(target.astype(np.float32))[30:-30,30:-30]
                    ax.plot((np.arange(52)+.5-26)*23.04,q[26] if direction==0 else q[::-1,26],label='Earth target',alpha=.6)
                ax.set_title(['West-east','South-north'][direction]);ax.set_xlabel('km');ax.set_ylabel('m');ax.grid(alpha=.2);ax.legend();ax.set_ylim(-6000,6000)
            fig.suptitle(f'{label}, seed {seed}: central intermediate transects; no vertical exaggeration')
            fig.savefig(out/f'{label}-seed{seed}-profiles.png',dpi=120);plt.close(fig)
            records.append(entry)
            atomic_json(out/'regional-review.json',{'settings':identity,'validation':validation,'regions':records,'status':'in progress'})
            print(f'Completed {len(records)}/{len(cases)*len(a.seeds)} region/seed comparisons: {label}, seed {seed}',flush=True)
    atomic_json(out/'regional-review.json',{'settings':identity,'validation':validation,'regions':records,'status':'complete','visual_review_required':True,'selection_changed':False})
    (out/'README.txt').write_text(('Smoke test only: two training updates and eight sampler steps are not terrain-quality evidence.\n' if a.smoke else '')+
        'Review the PNG pairs and unexaggerated profiles for belt widths, intervening valleys, secondary ranges, plains and shelves. Check coarse_metre_average_drift_rms_m, broad_drift_rms_m and clean validation regression in regional-review.json. Meso panels cover a smaller central window, explicitly labelled. No checkpoint was selected or promoted, and no full globe was generated.\n')
    print(f'\n{"Review smoke test" if a.smoke else "Regional review"} complete: {out}. No model promoted; no full globe started.',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RunAlreadyActive) as error:
        print(f'Cannot review: {error}',flush=True);raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nReview stopped. Completed regional comparisons remain saved. No model selection changed.',flush=True)
        raise SystemExit(130)
