"""Bounded guide-resolution and intermediate probes; no training or planet run."""
import hashlib
import json
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diagnose_terrain_structure import ROOT,OUT,FILES,EARTH_REGIONS,sample,image,gauss,legend
from planet_training.data import PaleoRaster,bridge_patch
from planet_training.adaptation import coarse_context,paired_degradation
from toy_meso.sphere import GlobalRaster,encode,decode
from toy_meso.common import load_checkpoint,sample_patches

def frame(lat,lon):
    a,b=np.radians([lat,lon])
    return (np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)]),
            np.array([-np.sin(b),np.cos(b),0]),
            np.array([-np.sin(a)*np.cos(b),-np.sin(a)*np.sin(b),np.cos(a)]))

def longest_run(a):
    padded=np.r_[False,a,False].astype(int)
    return float(max(np.flatnonzero(np.diff(padded)==-1)-np.flatnonzero(np.diff(padded)==1),default=0)*2.88)

def resolution_probe():
    rows=[];report=[]
    fig,axs=plt.subplots(3,5,figsize=(20,12),constrained_layout=True)
    for i,(label,lat,lon) in enumerate(EARTH_REGIONS):
        raw=sample('Modern Earth',lat,lon,n=560)
        variants=[raw]
        for factor in [56,28,14,8]:
            z=torch.from_numpy(raw)[None,None]
            coarse=F.avg_pool2d(z,factor)
            enc=coarse.sign()*coarse.abs().sqrt()
            up=F.interpolate(enc,size=raw.shape,mode='bilinear',align_corners=False)[0,0].numpy()
            variants.append(decode(up))
        records={'region':label,'lat':lat,'lon':lon,'variants':[]}
        for j,(name,z) in enumerate(zip(['Modern 2.88 km','161.28 km guide','80.64 km guide','40.32 km guide','23.04 km guide'],variants)):
            z=z[72:-72,72:-72] # same 1198-km central square, filters have context
            np.save(OUT/f'resolution-{i}-{j}.npy',z)
            axs[i,j].imshow(image(z,2880),extent=(-599.04,599.04,-599.04,599.04))
            axs[i,j].set_title(f'{label}\n{name}',fontsize=11)
            axs[i,j].set_xticks([-400,0,400]);axs[i,j].set_yticks([-400,0,400]);axs[i,j].tick_params(labelsize=8)
            records['variants'].append({'guide':name,'central_horizontal_above_1000m_longest_run_km':longest_run(z[208]>1000)})
        report.append(records)
    fig.suptitle('Representation probe: one modern Earth DEM, only the guide spacing changes\nSame metres, fixed sea level, 1198-km windows and 5x NW hillshade; encoded bilinear reconstruction',fontsize=15)
    legend(fig,axs)
    fig.savefig(OUT/'guide-resolution-probe.png',dpi=120);plt.close(fig)
    return report

def intermediate_probe():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    checkpoint=ROOT/'outputs/intermediate/selected-adapted-1750/best.pt'
    model,norm,ck=load_checkpoint(checkpoint,'cuda')
    earth=GlobalRaster(np.load(ROOT/'outputs/planet_training/data/cache/etopo-filtered.npy'))
    paleo=PaleoRaster.from_file(FILES['Paleo 0 Ma'])
    targets=[];conditions=[];entries=[];ups=[]
    for label,lat,lon in EARTH_REGIONS:
        fr=frame(lat,lon)
        t,c=bridge_patch(earth,fr)
        _,pc=bridge_patch(paleo,fr)
        context=coarse_context(earth,fr,112)
        tt=torch.from_numpy(t.astype(np.float32))[None,None].cuda()
        cc=torch.from_numpy(c)[None,None].cuda()
        ctx=torch.from_numpy(context)[None,None].cuda()
        transplanted,smoothed=paired_degradation(norm,tt,cc,ctx,
              sigma_cells=1.0,clean_fraction=0)
        for guide_name,condition,target in [('Modern guide',cc,tt),('Smoothed modern guide',smoothed,transplanted),('Paleo 0 Ma guide',torch.from_numpy(pc)[None,None].cuda(),None)]:
            up,_=norm.condition(condition)
            target_metres=decode(t.astype(np.float32))
            entry={'region':label,'lat':lat,'lon':lon,'guide':guide_name}
            if target is not None:
                delta=decode(target[0,0].cpu().numpy())-target_metres
                entry['training_target_change_rms_m']=float(np.sqrt(np.mean(delta**2)))
                entry['smoothed_46km_target_change_rms_m']=float(np.sqrt(np.mean(gauss(delta,2)**2)))
            targets.append(target_metres);conditions.append(condition);ups.append(decode(up[0,0].cpu().numpy()));entries.append(entry)
    guides=torch.cat(conditions)
    generated=[]
    started=time.perf_counter()
    for seed in [42,314]:
        # Each same-location guide gets identical noise via a separate call.
        results=[]
        for i in range(0,len(guides),3):
            for j in range(3):
                pred,_=sample_patches(model,norm,guides[i+j:i+j+1],steps=32,seed=seed,device='cuda')
                z=decode(pred[0,0].float().cpu().numpy())
                assert np.isfinite(z).all()
                np.save(OUT/f'bridge-probe-{seed}-{i+j}.npy',z)
                results.append(z)
        generated.append(results)
        print(f'Completed bounded intermediate probes seed {seed}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    for i,e in enumerate(entries):
        e['seeds']={str(s):{'output_minus_bilinear_guide_rms_m':float(np.sqrt(np.mean((generated[k][i]-ups[i])**2))),
          'broad_output_minus_guide_rms_m':float(np.sqrt(np.mean(gauss(generated[k][i]-ups[i],2)**2)))} for k,s in enumerate([42,314])}
    fig,axs=plt.subplots(3,7,figsize=(24,11),constrained_layout=True)
    for row,(label,*_) in enumerate(EARTH_REGIONS):
        i=row*3
        fields=[targets[i],ups[i],generated[0][i],ups[i+1],generated[0][i+1],ups[i+2],generated[0][i+2]]
        names=['Modern target','Modern guide','Sample on modern','Smoothed guide','Sample on smooth','Paleo guide','Sample on Paleo']
        for col,(name,z) in enumerate(zip(names,fields)):
            z=z[30:-30,30:-30]
            axs[row,col].imshow(image(z,23040),extent=(-599.04,599.04,-599.04,599.04))
            axs[row,col].set_title(f'{label}\n{name}',fontsize=9)
            axs[row,col].set_xticks([-400,0,400]);axs[row,col].set_yticks([-400,0,400]);axs[row,col].tick_params(labelsize=7)
    fig.suptitle('Selected intermediate checkpoint, same seed 42 per guide, 32 steps\nBounded north-up tangent-plane patches; central ~1198 km shown; same colours and 5x hillshade',fontsize=14)
    legend(fig,axs)
    fig.savefig(OUT/'intermediate-guide-probe.png',dpi=120);plt.close(fig)
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.file_digest(open(checkpoint,'rb'),'sha256').hexdigest(),
        'step':ck['step'],'seeds':[42,314],'sampling_steps':32,'patch_size':112,'spacing_m':23040,'seconds':time.perf_counter()-started,
        'limits':'18 independent bounded patches. No tiled blending, full intermediate globe, meso sampling or retraining. Modern source uses existing 0.05-degree area-mean cache, same source as bridge training. 0 Ma is test-age diagnostic only. This compares conditioning, not checkpoint selection.', 'cases':entries}

def meso_probe():
    checkpoint=ROOT/'outputs/meso/selected-adapted-1000/best.pt'
    model,norm,ck=load_checkpoint(checkpoint,'cuda')
    entries=[];display=[];started=time.perf_counter()
    for row,(label,lat,lon) in enumerate(EARTH_REGIONS):
        fields=[]
        for col,source_col in enumerate([1,3,4]):
            z=np.load(OUT/f'resolution-{row}-{source_col}.npy')
            # 144 fine cells -> 18 guide cells -> crop one 8-cell margin -> 128 output.
            central=z[136:280,136:280]
            cond_m=central.reshape(18,8,18,8).mean(axis=(1,3))
            cond=torch.from_numpy(encode(cond_m))[None,None].cuda()
            up,_=norm.condition(cond)
            input_m=decode(up[0,0].cpu().numpy())
            results=[]
            for seed in [42,314]:
                output,_=sample_patches(model,norm,cond,steps=32,seed=seed,device='cuda')
                pred=decode(output[0,0].float().cpu().numpy())
                assert np.isfinite(pred).all()
                np.save(OUT/f'meso-probe-{row}-{col}-{seed}.npy',pred)
                results.append({'seed':seed,'output_minus_guide_rms_m':float(np.sqrt(np.mean((pred-input_m)**2)))})
                if seed==42:fields.extend([input_m,pred])
            entries.append({'region':label,'guide_resolution_km':[161.28,40.32,23.04][col],'samples':results})
        display.append(fields)
    fig,axs=plt.subplots(3,6,figsize=(21,11),constrained_layout=True)
    for row,(label,*_) in enumerate(EARTH_REGIONS):
        for col,z in enumerate(display[row]):
            axs[row,col].imshow(image(z,2880),extent=(-184.32,184.32,-184.32,184.32))
            axs[row,col].set_title(f'{label}\n{[161,40,23][col//2]} km source / '+('meso' if col%2 else 'guide'),fontsize=10)
            axs[row,col].set_xticks([-150,0,150]);axs[row,col].set_yticks([-150,0,150]);axs[row,col].tick_params(labelsize=8)
    fig.suptitle('Bounded meso probe: same selected model, same sampling seed 42, 32 steps\nSource guide retains structure at 161 / 40 / 23 km; all meso conditions sampled at 23.04 km\nEach window 368.64 km square; identical colours and 5x hillshade',fontsize=14)
    legend(fig,axs)
    fig.savefig(OUT/'meso-guide-probe.png',dpi=120);plt.close(fig)
    return {'checkpoint_sha256':hashlib.file_digest(open(checkpoint,'rb'),'sha256').hexdigest(),'step':ck['step'],
        'seeds':[42,314],'steps':32,'patch_size':128,'seconds':time.perf_counter()-started,'cases':entries,
        'limits':'18 independent patch samples, no sphere tile blending. Supplying richer real guide is an oracle intervention, not a trained new terrain generator. Coarse conditions derived from the same area-mean DEM with different reconstruction scales.'}

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    report={'resolution':resolution_probe(),'intermediate':intermediate_probe(),'meso':meso_probe()}
    (OUT/'probe-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved all bounded probe results.',flush=True)
