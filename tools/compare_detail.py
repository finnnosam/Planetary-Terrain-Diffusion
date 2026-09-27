"""Pretrained latent/decoder assembly comparison with fixed coarse fields and noise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planet_diffusion import cube, detail
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.coarse import sample_coarse
from planet_diffusion.geometry_quality import cube_boundary_quality


def legacy_latents(backend,coarse,seed,source_weights=False):
    n=(coarse.shape[-1]-1)*32
    state=np.zeros((5,6,n+1,n+1),np.float32)
    for step,t in enumerate([float(np.arctan(160)),float(np.arctan(.7))]):
        xt=(np.cos(t)*state+np.sin(t)*cube.noise(seed,5819+step,5,n)).astype(np.float32)
        def predict(a,f,y,x):
            cond=cube.read(coarse,f,(y/32-1+np.arange(4))[:,None],(x/32-1+np.arange(4))[None,:])
            return backend.predict('latent',a,cond,t)
        options={} if not source_weights else dict(weight_window=cube.linear_weight_window(64),shared_weights=True,include_endpoint=True)
        prediction=cube.consensus(xt,predict,64,32,**options)
        state=cube.identify(np.cos(t)*xt+np.sin(t)*prediction)
    return state


def interpolated_latents(backend,coarse,seed):
    """Rejected experimental halo transport, kept here for reproduction only."""
    n=(coarse.shape[-1]-1)*32
    state=np.zeros((5,6,n+1,n+1),np.float32)
    for step,t in enumerate([float(np.arctan(160)),float(np.arctan(.7))]):
        noise=cube.noise(seed,5819+step,5,n)
        def predict(a,f,y,x):
            prior=cube.read(state,f,np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:])
            xt=(np.cos(t)*prior+np.sin(t)*a).astype(np.float32)
            cond=cube.read(coarse,f,(y/32-1+np.arange(4))[:,None],(x/32-1+np.arange(4))[None,:])
            return np.cos(t)*xt+np.sin(t)*backend.predict('latent',xt,cond,t)
        state=cube.consensus(noise,predict,64,32,weight_window=cube.linear_weight_window(64),
                             shared_weights=True,include_endpoint=True)
    return state


def decode(backend,latent,seed,variant):
    n=(latent.shape[-1]-1)*8
    t=float(np.arctan(160))
    xt=cube.noise(seed,6819,1,n)*np.sin(t)
    def predict(a,f,y,x):
        if variant in ('legacy','weights256','weights384'):
            cond=cube.read(latent[:4],f,np.floor_divide(np.arange(y,y+512),8)[:,None],
                           np.floor_divide(np.arange(x,x+512),8)[None,:],nearest=True)
        else:
            nodes=cube.read(latent[:4],f,(y//8+np.arange(64))[:,None],(x//8+np.arange(64))[None,:])
            cond=np.repeat(np.repeat(nodes,8,-2),8,-1)
        return backend.predict('decoder',a,cond,t)
    options={} if variant=='legacy' else dict(weight_window=cube.linear_weight_window(512),
                                             shared_weights=True,include_endpoint=True)
    stride=384 if variant in ('source384','weights384') else 256
    pred=cube.consensus(xt,predict,512,stride,**options,
                        progress=lambda d,n:print(f'{variant} decoder {d}/{n}',flush=True))
    residual=cube.identify(np.cos(t)*xt+np.sin(t)*pred)
    z=cube.reconstruct(residual*backend.residual_std+backend.residual_mean,latent[4:5]*38.6-31.4)
    return residual,cube.identify(np.sign(z)*z*z)


def metrics(latent,residual,elevation):
    def rms(a): return float(np.sqrt(np.mean(np.asarray(a,np.float64)**2)))
    def edge(a):
        v=cube_boundary_quality(a)
        return {'shared_node_error':v['shared_node_error_m'],
                'mean_gradient_ratio':float(np.mean([e['ratio'] for e in v['edges']])),
                'max_gradient_ratio':max(e['ratio'] for e in v['edges'])}
    tile={}
    for stride in (256,384):
        boundaries=[]; interiors=[]
        n=residual.shape[-1]-1
        for axis in (-1,-2):
            a=np.swapaxes(residual,axis,-1)
            for x in range(stride,n-8,stride):
                boundaries.append((a[...,x]-a[...,x-1]).ravel())
                interiors.extend([(a[...,x-8]-a[...,x-9]).ravel(),(a[...,x+8]-a[...,x+7]).ravel()])
        tile[str(stride)]=rms(np.concatenate(boundaries))/max(rms(np.concatenate(interiors)),1e-12) if boundaries else None
    return {'latent_edges':edge(latent),'residual_edges':edge(residual),
            'elevation_edges':edge(elevation),'residual_std':float(residual.std()),
            'residual_neighbor_rms':rms(np.diff(residual,axis=-1)),
            'tile_boundary_gradient_ratios':tile,
            'elevation_range_m':[float(elevation.min()),float(elevation.max())]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--face-coarse',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    backend=TerrainBackend('upstream','models/terrain-diffusion-90m',MODEL_REVISION,'cuda')
    n,seed=args.face_coarse,args.seed
    raw=cube.conditioning(seed,n,frequency_mult=backend.frequency_mult,drop_water_pct=backend.drop_water_pct)
    coarse=sample_coarse(backend,seed,raw,20,lambda _:None)
    np.save(out/'coarse.npy',coarse)
    report={'seed':seed,'face_coarse':n,'backend':backend.metadata,'variants':{},
            'scope':'Full pretrained latent and decoder passes; identical coarse field and per-node Gaussian innovations.'}
    latents={}
    for version in ('legacy','weights','adapted'):
        print(f'{version} latent',flush=True)
        start=time.perf_counter()
        latents[version]=(legacy_latents(backend,coarse,seed,source_weights=version=='weights') if version!='adapted'
                          else interpolated_latents(backend,coarse,seed))
        np.save(out/f'{version}-latent.npy',latents[version])
        report[f'{version}_latent_seconds']=time.perf_counter()-start
    maps=[]
    for variant in ('legacy','weights256','weights384','adapted256','source384'):
        start=time.perf_counter()
        latent=latents['legacy' if variant=='legacy' else 'weights' if variant.startswith('weights') else 'adapted']
        residual,elevation=decode(backend,latent,seed,variant)
        np.savez_compressed(out/f'{variant}.npz',residual=residual,elevation=elevation)
        values=metrics(latent,residual,elevation); values['decoder_seconds']=time.perf_counter()-start
        report['variants'][variant]=values
        print(json.dumps(values),flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        maps.append((variant,cube.to_equirectangular(elevation,512)[0],elevation[0,0]))
    fig,axes=plt.subplots(len(maps),2,figsize=(13,4*len(maps)),layout='constrained')
    minimum=min(float(face[64:320,64:320].min()) for _,_,face in maps)
    maximum=max(float(face[64:320,64:320].max()) for _,_,face in maps)
    for row,(name,global_map,face) in enumerate(maps):
        axes[row,0].imshow(global_map,cmap='RdBu_r',vmin=-6000,vmax=6000)
        axes[row,0].set_title(f'{name}: global elevation, +/-6000 m')
        # Same fixed face crop, with independent colour range avoided.
        crop=face[64:320,64:320]
        im=axes[row,1].imshow(crop,cmap='terrain',vmin=minimum,vmax=maximum)
        fig.colorbar(im,ax=axes[row,1],label='m')
        axes[row,1].set_title(f'{name}: face 0, nodes 64:320')
    fig.suptitle(f'Pretrained detail assembly: seed {seed}, {n} coarse face intervals')
    fig.savefig(out/'comparison.png',dpi=120); plt.close(fig)
    report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ('planet_diffusion/detail.py','planet_diffusion/cube.py','tools/compare_detail.py')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
