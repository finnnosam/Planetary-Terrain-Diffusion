"""Three bounded meso trials for a reviewed continent/intermediate guide."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from toy_meso.evaluate import hillshade, radial_psd
from toy_meso.planet_tiled import main as generate
from toy_meso.sphere import GlobalRaster, EARTH_RADIUS_M


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', required=True)
    p.add_argument('--intermediate', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--run', default='outputs/meso/selected-adapted-1000')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = GlobalRaster.from_geotiff(a.layout)
    guide = GlobalRaster.from_geotiff(a.intermediate)
    z = guide.values
    lat = 90-(np.arange(guide.h)+.5)*180/guide.h
    lon = -180+(np.arange(guide.w)+.5)*360/guide.w
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    valid = (abs(yy)<45) & (abs(xx)<155)
    # Choose using the intermediate only, before looking at meso detail.
    regions = []
    for name, target in [('Highland', 2200), ('Upland', 1100), ('Coast', 0)]:
        distance = abs(z-target).astype(np.float64)
        distance[~valid] = np.inf
        if target:
            distance[z<100] = np.inf
        else:
            land = z>0
            crossing = np.zeros_like(land)
            crossing[:-1] |= land[:-1] != land[1:]
            crossing[1:] |= land[:-1] != land[1:]
            crossing[:,:-1] |= land[:,:-1] != land[:,1:]
            crossing[:,1:] |= land[:,:-1] != land[:,1:]
            distance[~crossing] = np.inf
        for region in regions:
            distance[(yy-region['latitude'])**2 +
                     ((xx-region['longitude'])*np.cos(np.radians(yy)))**2 < 15**2] = np.inf
        row, col = np.unravel_index(np.argmin(distance), z.shape)
        if not np.isfinite(distance[row,col]):
            raise ValueError('No eligible separated region')
        y, x = float(lat[row]), float(lon[col])
        span_y = 256*2880/EARTH_RADIUS_M*180/np.pi
        span_x = span_y/np.cos(np.radians(y))
        regions.append(dict(name=name, latitude=y, longitude=x,
                            intermediate_center_m=float(z[row,col]),
                            bounds=[x-span_x/2,y-span_y/2,x+span_x/2,y+span_y/2]))
    checkpoint = Path(a.run)/'best.pt'
    identity = dict(layout_sha256=hashlib.sha256(Path(a.layout).read_bytes()).hexdigest(),
                    intermediate_sha256=hashlib.sha256(Path(a.intermediate).read_bytes()).hexdigest(),
                    meso_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    seed=a.seed, steps=32, batch=16, shape=[256,256], regions=copy.deepcopy(regions))
    manifest = out/'trial-settings.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Trial settings differ; choose a fresh output folder')
    manifest.write_text(json.dumps(identity, indent=2)+'\n')
    cases = []
    for i, region in enumerate(regions):
        target = out/f'region-{i+1}.tif'
        if not target.exists():
            generate(['--run', a.run, '--checkpoint', 'best.pt', '--dem', a.intermediate,
                      '--out', str(out/'meso-cache'), '--name', target.name,
                      '--bounds', *map(str, region['bounds']), '--width', '256', '--height', '256',
                      '--seed', str(a.seed), '--steps', '32', '--batch', '16', '--device', 'cuda'])
            # Keep the generator's cache reusable, and copy the finished artifact.
            cached = out/'meso-cache'/target.name
            target.write_bytes(cached.read_bytes())
        with rasterio.open(target) as ds:
            terrain = ds.read(1)
            assert terrain.shape == (256,256) and np.isfinite(terrain).all()
            assert ds.crs is not None
            latitude = ds.transform.f+(np.arange(ds.height)+.5)*ds.transform.e
            longitude = ds.transform.c+(np.arange(ds.width)+.5)*ds.transform.a
            spacing = abs(ds.transform.e)*np.pi/180*EARTH_RADIUS_M
        coarse = layout.sample(latitude[:,None],longitude[None,:])
        intermediate = guide.sample(latitude[:,None],longitude[None,:])
        wl, p0 = radial_psd(intermediate[None],spacing)
        _, p1 = radial_psd(terrain[None],spacing)
        band = (wl>=6000)&(wl<23000)
        region.update(meso_minus_intermediate_rms_m=float(np.sqrt(np.mean((terrain-intermediate)**2))),
                      nominal_6_23km_power_ratio=float(p1[band].sum()/max(p0[band].sum(),1e-20)),
                      meso_range_m=[float(terrain.min()),float(terrain.max())],
                      finite=True, tiff=str(target.resolve()))
        cases.append((region,coarse,intermediate,terrain,spacing))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    # Sea level has a fixed position in the display; local land ranges show relief.
    cmap = LinearSegmentedColormap.from_list('sea-and-land',
        [(0,'#163657'),(.36,'#377db0'),(.499,'#8ad4d7'),(.5,'#486b47'),
         (.6,'#93a66b'),(.8,'#b7a278'),(1,'#f0ede5')])
    fig,axes = plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    for col,(region,*fields,spacing) in enumerate(cases):
        flat = np.concatenate([v.ravel() for v in fields])
        norm = TwoSlopeNorm(vmin=min(-100.,float(np.min(flat))),vcenter=0,
                            vmax=max(300.,float(np.max(flat))))
        for row,(label,field) in enumerate(zip(['Continent guide','Intermediate (~23 km)','Meso (~2.9 km)'],fields)):
            rgb = cmap(norm(field))[...,:3]
            rgb *= (.4+.6*hillshade(field,spacing,exaggeration=8))[...,None]
            axes[row,col].imshow(rgb,interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if not col: axes[row,col].set_ylabel(label)
            if not row: axes[row,col].set_title(f'{region["name"]}: {region["latitude"]:.1f}°, {region["longitude"]:.1f}°')
    fig.suptitle('Three regional trials, ~740 km across each\nSame display range per column; sea level fixed at 0 m; shading exaggerated 8×')
    fig.savefig(out/'regional-comparison.png',dpi=140);plt.close(fig)
    report = dict(identity=identity,regions=regions,
                  limits='Three regions of one planet, chosen before meso sampling. Higher spectral power is not a realism score. Wavelengths use nominal north-south spacing; longitude spacing varies with latitude. No full meso globe generated.')
    (out/'regional-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(regions,indent=2),flush=True)


if __name__=='__main__': main()
