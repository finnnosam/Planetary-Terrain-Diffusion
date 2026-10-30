"""Compare meso adaptation on the same three saved planet regions and noise seed."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from toy_meso.planet_tiled import main as generate
from toy_meso.evaluate import radial_psd, hillshade
from toy_meso.sphere import GlobalRaster


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='outputs/meso/run3-adapted')
    p.add_argument('--out',default='outputs/meso/adapted-both-planet42')
    a = p.parse_args()
    torch.set_num_threads(4)
    original_folder=Path('outputs/meso/adapted-planet42')
    report=json.loads((original_folder/'verification.json').read_text())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    guide=GlobalRaster.from_geotiff('outputs/intermediate/adapted-planet42/planet.tif')
    cases,metrics=[],[]
    for index, region in enumerate(report['regions']):
        dest=out/f'region-{index+1}.tif'
        if not dest.exists():
            generate(['--run',a.run,'--checkpoint','best.pt',
                      '--dem','outputs/intermediate/adapted-planet42/planet.tif',
                      '--out',str(out),'--name',dest.name,'--bounds',*map(str,region['bounds']),
                      '--width','256','--height','256','--seed','1135849674',
                      '--steps','32','--batch','16','--device','cuda'])
        with rasterio.open(region['after']) as src:r0=src.read(1); transform=src.transform
        with rasterio.open(dest) as src:
            r1=src.read(1)
            assert np.allclose(tuple(src.transform),tuple(transform),atol=1e-10)
            assert np.isfinite(r1).all()
        lat=transform.f+(np.arange(256)+.5)*transform.e
        lon=transform.c+(np.arange(256)+.5)*transform.a
        g=guide.sample(lat[:,None],lon[None,:])
        h=abs(transform.e)*np.pi/180*6371000
        wl,p0=radial_psd(r0[None],h);_,p1=radial_psd(r1[None],h)
        band=(wl>=6000)&(wl<23000)
        metrics.append(dict(name=region['name'],after=str(dest),bounds=region['bounds'],
            nominal_6_23km_power_new_vs_old=float(p1[band].sum()/p0[band].sum()),
            output_minus_input_rms_m=dict(original=float(np.sqrt(np.mean((r0-g)**2))),
                                         adapted=float(np.sqrt(np.mean((r1-g)**2))))))
        cases.append((region['name'],g,r0,r1,h))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,len(cases),figsize=(12,11),squeeze=False)
    for c,(name,g,z0,z1,h) in enumerate(cases):
        lo,hi=np.percentile(np.r_[g.ravel(),z0.ravel(),z1.ravel()],[1,99])
        for r,(label,z) in enumerate([('Adapted intermediate input',g),('Original meso',z0),('Adapted meso',z1)]):
            rgb=plt.cm.terrain(np.clip((z-lo)/max(hi-lo,1),0,1))[...,:3]
            rgb*=.3+.7*hillshade(z,h)[...,None]
            axs[r,c].imshow(rgb,interpolation='nearest');axs[r,c].set_xticks([]);axs[r,c].set_yticks([])
            if c==0:axs[r,c].set_ylabel(label)
            if r==0:axs[r,c].set_title(name)
    fig.suptitle('Same adapted intermediate input, output grid and initial noise\nMeso fine-tuning on smoother guides; ~740 km north–south close-ups')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    # Also show the user's original global result, so both adaptations are visible.
    from rasterio.windows import from_bounds
    original=[]
    with rasterio.open('human_input/MESOFromIntermediate/planet.tif') as src:
        for region in report['regions']:
            window=from_bounds(*region['bounds'],transform=src.transform).round_offsets().round_lengths()
            z=src.read(1,window=window)
            assert z.shape==(256,256)
            original.append(z)
    fig,axs=plt.subplots(2,len(cases),figsize=(12,8),squeeze=False)
    for c,((name,g,z0,z1,h),before) in enumerate(zip(cases,original)):
        lo,hi=np.percentile(np.r_[before.ravel(),z1.ravel()],[1,99])
        for r,(label,z) in enumerate([('Your original result',before),('Both models adapted',z1)]):
            rgb=plt.cm.terrain(np.clip((z-lo)/max(hi-lo,1),0,1))[...,:3]
            rgb*=.3+.7*hillshade(z,h)[...,None]
            axs[r,c].imshow(rgb,interpolation='nearest');axs[r,c].set_xticks([]);axs[r,c].set_yticks([])
            if c==0:axs[r,c].set_ylabel(label)
            if r==0:axs[r,c].set_title(name)
        metrics[c]['nominal_6_23km_power_vs_original_pipeline']=metrics[c]['nominal_6_23km_power_new_vs_old']*report['regions'][c]['nominal_6_23km_power_new_vs_old']
    fig.suptitle('Original result versus adapted intermediate + meso\nSame continent layout, meso seed, pixel grid and display scale per region')
    fig.tight_layout();fig.savefig(out/'combined-comparison.png',dpi=140);plt.close(fig)
    result=dict(regions=metrics,note=report['note'],meso_run=a.run)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
