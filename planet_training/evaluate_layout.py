"""Matched whole-planet comparison, native spherical geography and copying screen."""
import argparse
import gc
import hashlib
import io
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from planet_diffusion import cube
from toy_meso.sphere import decode,to_latlon
from toy_meso.common import pick_device,autocast
from .data import VERSION,PaleoRaster,layout_target,random_rotation,manifest_id,atomic_json
from .layout import LayoutDenoiser
from .train import LayoutData,padded_target,sample_layout
from .geography import measures,spectrum


def probe_points(count=2048):
    z=1-2*(np.arange(count)+.5)/count
    lon=np.arange(count)*np.pi*(3-np.sqrt(5))
    r=np.sqrt(1-z*z)
    return np.stack((r*np.cos(lon),r*np.sin(lon),z),1)


def orientations(count=128):
    matrices=[]
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            matrices.append(np.eye(3)[list(order)]*np.asarray(signs)[:,None])
    rng=np.random.default_rng(1507)
    matrices.extend(random_rotation(rng) for _ in range(count))
    return np.stack(matrices)


class CopyScreen:
    """Approximate rotation/reflection search, followed by local alignment.

    IoU is estimated at equal-area sphere probes. This is a similarity screen,
    not a guarantee against memorization; related real ages also overlap highly.
    """
    def __init__(self,maps,indices,records,points=2048,rotations=128):
        self.points=probe_points(points);self.rotations=orientations(rotations)
        self.maps=maps;self.indices=np.asarray(indices);self.records=records
        lat,lon=to_latlon(self.points[None]@self.rotations)
        self.masks=np.stack([PaleoRaster(maps[i]).sample(lat,lon)>0 for i in indices])
        self.matrix=torch.from_numpy(self.masks.reshape(-1,points).astype(np.float32))
        self.counts=self.matrix.sum(1).numpy()

    def match(self,metres):
        candidate=cube.sample(metres[None],self.points)[0]>0
        overlap=(self.matrix@torch.from_numpy(candidate.astype(np.float32))).numpy()
        union=self.counts+candidate.sum()-overlap
        scores=np.divide(overlap,union,out=np.zeros_like(overlap),where=union>0)
        scores=scores.reshape(self.masks.shape[:2])
        choices=np.argsort(scores.max(1))[-min(2,len(scores)):]
        best_score=-1.;best_index=None;best_rotation=None
        def score(raster,rotation):
            la,lo=to_latlon(self.points@rotation)
            b=raster.sample(la,lo)>0
            u=(b|candidate).sum()
            return float((b&candidate).sum()/u) if u else 0.
        for index in choices:
            raster=PaleoRaster(self.maps[self.indices[index]])
            rotation=self.rotations[scores[index].argmax()].copy()
            current=score(raster,rotation)
            for degrees in (15.,7.5,3.75,1.875,.9375):
                for _ in range(3):
                    improved=False
                    for axis in range(3):
                        j,k=(axis+1)%3,(axis+2)%3
                        for sign in (-1,1):
                            t=np.radians(degrees*sign);delta=np.eye(3)
                            delta[j,j]=delta[k,k]=np.cos(t);delta[j,k]=-np.sin(t);delta[k,j]=np.sin(t)
                            proposed=rotation@delta;value=score(raster,proposed)
                            if value>current:
                                current,rotation=value,proposed;improved=True
                    if not improved:break
            if current>best_score:best_score,best_index,best_rotation=current,self.indices[index],rotation
        record=self.records[best_index]
        return dict(approx_best_training_land_iou=best_score,closest_training_age_ma=record.get('age_ma'),
                    closest_training_record=int(best_index),alignment=best_rotation.tolist())


@torch.no_grad()
def validation(model,stats,data,device,batch):
    raw=data.validation();vt=padded_target(raw,model,stats,device)
    g=torch.Generator(device=device).manual_seed(0);total=0.
    for sigma in torch.exp(torch.linspace(-3,1.5,8,device=device)):
        noisy=vt+model.grid.noise(len(vt),device,g)*sigma
        error=0.
        for start in range(0,len(vt),batch):
            chunk=noisy[start:start+batch]
            with autocast(device):d=model(chunk,sigma.expand(len(chunk)))
            error+=float(model.grid.weighted_mean((d-vt[start:start+batch])**2))*len(chunk)/len(vt)
        total+=error*float((sigma**2+.25)/(sigma*.5)**2)
    return total/8


def summarize(records):
    keys=['land_fraction','large_landmass_count','largest_landmass_share_of_land','land_p95_m',
          'highland_land_fraction','coastline_crossing_length_proxy_km','adjacent_land_mean_abs_change_m',
          'land_dipole_magnitude','land_quadrupole_magnitude','approx_best_training_land_iou']
    return {key:dict(median=float(np.median(v)),p25=float(np.percentile(v,25)),p75=float(np.percentile(v,75)))
            for key in keys if (v:=[r[key] for r in records if r.get(key) is not None])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',default='outputs/continents/run1/best.pt')
    p.add_argument('--candidate',default='outputs/continents/run2-lowlr/best.pt')
    p.add_argument('--data',default='outputs/planet_training/data/layout')
    p.add_argument('--out',default='outputs/continents/run2-lowlr/evaluation')
    p.add_argument('--seeds',default='42,314,2718,867,2026,1776,12345,919,101,202,303,404,505,606,707,808')
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=16)
    a=p.parse_args(argv)
    seeds=[int(v) for v in a.seeds.split(',')]
    if a.steps<2 or a.batch<1 or len(set(seeds))!=len(seeds):p.error('Invalid sampling steps, batch or duplicate seeds')
    torch.set_num_threads(4);device=pick_device(a.device);data=LayoutData(a.data,0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    print('Preparing native-scale real-world reference distributions and copying screen...',flush=True)
    screen=CopyScreen(data.maps,data.splits['train'],data.meta['records'])
    references={};rng=np.random.default_rng(2026)
    for split in ('train','val'):
        records=[]
        for i in data.splits[split]:
            z=decode(layout_target(PaleoRaster(data.maps[i]),random_rotation(rng),data.n,data.meta['supersample']))
            r=dict(record=int(i),age_ma=data.meta['records'][i].get('age_ma'),**measures(z),spherical_power=spectrum(z))
            if split=='val':r.update(screen.match(z))
            records.append(r)
        references[split]=dict(planets=records,summary=summarize(records))
    reports=[];samples={}
    for label,path in [('Original',a.reference),('Continued',a.candidate)]:
        payload=Path(path).read_bytes()
        checkpoint_sha=hashlib.sha256(payload).hexdigest()
        ck=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        del payload
        if ck.get('stage')!='layout' or ck.get('training_version')!=VERSION or ck.get('dataset_id')!=manifest_id(data.meta):
            raise ValueError('Layout checkpoint does not match evaluation data')
        model=LayoutDenoiser(n=data.n,**ck['config']).to(device).eval();model.load_state_dict(ck['ema'])
        loss=validation(model,ck['stats'],data,device,a.batch)
        print(f'{label}: held-out loss {loss:.6f}',flush=True)
        records=[];values=[]
        for seed in seeds:
            z=decode(sample_layout(model,ck['stats'],a.steps,seed,device)[0])
            r=dict(seed=seed,**measures(z),spherical_power=spectrum(z),**screen.match(z))
            records.append(r);values.append(z)
            print(f"{label}: seed {seed}, land {r['land_fraction']*100:.1f}%, large landmasses {r['large_landmass_count']}, nearest training IoU {r['approx_best_training_land_iou']:.3f}",flush=True)
        samples[label]=np.stack(values)
        reports.append(dict(label=label,checkpoint=path,checkpoint_sha256=checkpoint_sha,step=ck['step'],
                            initialized_from=ck.get('initialized_from'),val_loss=loss,planets=records,summary=summarize(records)))
        del model,ck;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    np.savez_compressed(out/'samples.npz',**samples,seeds=seeds)
    report=dict(dataset_id=manifest_id(data.meta),references=references,models=reports,sampler_steps=a.steps,
                protocol='native-sphere-geography-v1',notes=[
                    'Evaluation uses training and validation ages; final test ages remain untouched.',
                    'Training/validation ages represent different geological populations; statistics are references, not identical target distributions.',
                    'Land masses use four-neighbour connectivity on unique physical cube nodes; large means at least 1% of planet area.',
                    'Coastline metric is a grid-edge crossing length proxy, not exact shoreline length.',
                    'Spherical power is rotation-invariant degree 1–12 elevation power; low-resolution quadrature is approximate.',
                    'Copying screen searches rotations and reflections on 2048 equal-area probes, then refines two best source ages; approximate IoU is not proof against copying.',
                    'Paired seeds use identical initial noise; selection must consider generated geography as well as held-out loss.'])
    atomic_json(out/'evaluation.json',report)
    plots(out,report,samples,seeds)
    print(f'Saved whole-planet evaluation to {out}',flush=True)


def plots(out,report,samples,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count=min(8,len(seeds));fig,axes=plt.subplots(count,2,figsize=(12,2.5*count),squeeze=False)
    for i in range(count):
        for c,label in enumerate(('Original','Continued')):
            image=cube.to_equirectangular(samples[label][i][None],128)[0]
            axes[i,c].imshow(image,cmap='terrain',vmin=-6500,vmax=6000,extent=(-180,180,-90,90))
            axes[i,c].contour(image,levels=[0],colors='black',linewidths=.35,extent=(-180,180,-90,90),origin='upper')
            axes[i,c].set_xticks([]);axes[i,c].set_yticks([])
            axes[i,c].set_title(f'{label} / seed {seeds[i]}')
    fig.suptitle('Matched initial noise: original versus lower-rate continent continuation')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=120);plt.close(fig)
    groups=[('Real training',report['references']['train']['planets']),('Real held-out',report['references']['val']['planets'])]
    groups.extend((m['label'],m['planets']) for m in report['models'])
    keys=[('land_fraction','Land fraction'),('largest_landmass_share_of_land','Largest landmass / total land'),
          ('land_p95_m','Land elevation 95th percentile (m)'),('large_landmass_count','Landmasses >=1% planet area'),
          ('coastline_crossing_length_proxy_km','Coastline crossing proxy (km)'),('adjacent_land_mean_abs_change_m','Adjacent land relief (m)')]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(key,title) in zip(axes.ravel(),keys):
        ax.boxplot([[r[key] for r in records if r[key] is not None] for _,records in groups],tick_labels=[l for l,_ in groups])
        ax.set_title(title);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Native spherical geography distributions; source age populations differ')
    fig.tight_layout();fig.savefig(out/'distributions.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for label,records in groups:
        powers=np.array([[r['spherical_power'][str(ell)] for ell in range(1,13)] for r in records])
        ax.plot(range(1,13),np.median(powers,0),label=label)
    ax.set_yscale('log');ax.set_xlabel('Spherical harmonic degree');ax.set_ylabel('Median elevation power (m²)')
    ax.legend();fig.tight_layout();fig.savefig(out/'spherical-power.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
