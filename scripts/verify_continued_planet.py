"""Inspect the completed controlled continent -> intermediate -> meso globe.

This is an artifact inspection, not a terrain-realism score. Spectra use nominal
north-south raster spacing; longitude spacing varies with latitude.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from toy_meso.sphere import GlobalRaster
from toy_meso.evaluate import hillshade, radial_psd


def raster_stats(path):
    land_area = total_area = 0.0
    lo, hi = np.inf, -np.inf
    with rasterio.open(path) as ds:
        for row in range(0, ds.height, 256):
            count = min(256, ds.height - row)
            z = ds.read(1, window=Window(0, row, ds.width, count))
            if not np.isfinite(z).all():
                raise ValueError(f"Non-finite elevations in {path}")
            latitude = ds.transform.f + (np.arange(row, row + count) + .5) * ds.transform.e
            weight = np.cos(np.radians(latitude))[:, None]
            land_area += float(((z > 0) * weight).sum())
            total_area += float(weight.sum()) * ds.width
            lo, hi = min(lo, float(z.min())), max(hi, float(z.max()))
        return dict(path=str(path), shape=[ds.height, ds.width], finite=True,
                    min_m=lo, max_m=hi, area_weighted_land_percent=100 * land_area / total_area,
                    bounds=list(ds.bounds), transform=list(ds.transform),
                    units=ds.tags().get('units', 'metres'))


def choose_regions(guide_path):
    # Select before reading meso detail: intermediate elevation and separation
    # determine the three examples, without selecting on generated fine power.
    with rasterio.open(guide_path) as ds:
        z = ds.read(1)
        lat = ds.transform.f + (np.arange(ds.height) + .5) * ds.transform.e
        lon = ds.transform.c + (np.arange(ds.width) + .5) * ds.transform.a
        yy, xx = np.meshgrid(lat, lon, indexing='ij')
        valid = (abs(yy) < 55) & (abs(xx) < 170) & (z > 100)
        regions = []
        for title, target in [('Highland', 2500), ('Upland', 1200), ('Lowland', 350)]:
            distance = abs(z - target).astype(np.float64)
            distance[~valid] = np.inf
            for _, py, px in regions:
                distance[(yy - py)**2 + ((xx - px) * np.cos(np.radians(yy)))**2 < 15**2] = np.inf
            row, col = np.unravel_index(np.argmin(distance), z.shape)
            if not np.isfinite(distance[row, col]):
                raise ValueError('No separated land region available')
            regions.append((title, float(lat[row]), float(lon[col])))
        return regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', default='outputs/continents/continued-examples/land30-relief125/layout.tif')
    parser.add_argument('--intermediate', default='outputs/intermediate/continued-land30-relief125/planet.tif')
    parser.add_argument('--meso', default='outputs/meso/continued-land30-relief125/planet.tif')
    a = parser.parse_args(argv)
    paths = [Path(a.layout), Path(a.intermediate), Path(a.meso)]
    stats = [raster_stats(p) for p in paths]
    layout, guide = (GlobalRaster.from_geotiff(str(p)) for p in paths[:2])
    regions = choose_regions(paths[1])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    palette = np.array([[-6500,.035,.10,.23],[-2000,.04,.32,.60],[-200,.24,.65,.81],
                        [0,.52,.80,.86],[.01,.26,.49,.25],[500,.60,.67,.36],
                        [1500,.69,.54,.34],[3000,.61,.47,.34],[6000,.95,.95,.90]])
    levels = np.linspace(-6500, 6000, 1024)
    cmap = ListedColormap(np.stack([np.interp(levels, palette[:,0], palette[:,i]) for i in range(1,4)],1))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    records = []
    with rasterio.open(paths[2]) as ds:
        spacing = np.pi * 6371000 / ds.height
        for col, (title, latitude, longitude) in enumerate(regions):
            row, column = ds.index(longitude, latitude)
            row = min(max(row - 128, 0), ds.height - 256)
            column = min(max(column - 128, 0), ds.width - 256)
            window = Window(column, row, 256, 256)
            z = ds.read(1, window=window)
            transform = ds.window_transform(window)
            lat = transform.f + (np.arange(256) + .5) * transform.e
            lon = transform.c + (np.arange(256) + .5) * transform.a
            z0 = layout.sample(lat[:,None], lon[None,:])
            z1 = guide.sample(lat[:,None], lon[None,:])
            lo, hi = np.percentile(np.concatenate([z0.ravel(), z1.ravel(), z.ravel()]), [1,99])
            norm = Normalize(lo, max(hi, lo + 1))
            for r, (label, elevation) in enumerate([('Continent layout',z0),('Intermediate terrain',z1),('Meso terrain',z)]):
                # Each column shares a colour range; 8x shade exaggeration is
                # explicitly stated. Grey panels emphasize shapes, not height.
                rgb = plt.cm.terrain(norm(elevation))[...,:3]
                rgb *= (.3 + .7 * hillshade(elevation, spacing))[...,None]
                axes[r,col].imshow(rgb, interpolation='nearest')
                axes[r,col].set_xticks([]); axes[r,col].set_yticks([])
                if col == 0: axes[r,col].set_ylabel(label)
                if r == 0: axes[r,col].set_title(f'{title}: {latitude:.1f}°, {longitude:.1f}°')
            wl, power0 = radial_psd(z1[None], spacing)
            _, power1 = radial_psd(z[None], spacing)
            band = (wl >= 6000) & (wl < 23000)
            records.append(dict(name=title, center_latitude=latitude, center_longitude=longitude,
                window=[column,row,256,256],
                meso_minus_intermediate_rms_m=float(np.sqrt(np.mean((z-z1)**2))),
                nominal_6_23km_power_meso_over_intermediate=float(power1[band].sum()/max(power0[band].sum(),1e-20)),
                land_fraction=float((z>0).mean()), min_m=float(z.min()), max_m=float(z.max())))
            regional_path = paths[2].parent / f'closeup-{col+1}.tif'
            with rasterio.open(regional_path, 'w', driver='GTiff', width=256, height=256,
                               count=1, dtype='float32', crs=ds.crs, transform=transform,
                               compress='deflate', predictor=3) as dest:
                dest.write(z,1)
                dest.update_tags(units='metres', source=str(paths[2]))
        fig.suptitle('Chosen globe: 30% land, 1.25× continent relief\n'
                     'Same final pixel grid and colours within each column; ~740 km north–south; shade 8×')
        fig.savefig(paths[2].parent/'stage-closeups.png', dpi=140)
        plt.close(fig)
        from rasterio.enums import Resampling
        small = ds.read(1, out_shape=(868,1736), resampling=Resampling.average)
    fig, ax = plt.subplots(figsize=(14,7), constrained_layout=True)
    rgb = cmap(Normalize(-6500,6000)(small))[...,:3]
    rgb *= (.4 + .6 * hillshade(small, np.pi*6371000/868, exaggeration=4))[...,None]
    ax.imshow(rgb, extent=(-180,180,-90,90))
    ax.set_xticks([]); ax.set_yticks([])
    for index, (_, lat, lon) in enumerate(regions):
        ax.plot(lon,lat,'wo',markersize=5); ax.text(lon+3,lat,str(index+1),color='white')
    ax.set_title('Completed continent → intermediate → meso globe\nGlobal preview averages fine detail; numbered close-ups show it at native resolution')
    fig.savefig(paths[2].parent/'globe-overview.png', dpi=140); plt.close(fig)
    settings = [json.loads((p.parent/'settings.json').read_text()) for p in paths[1:]]
    report = dict(stages=stats, regions=records,
                  selected_continent=json.loads((paths[0].parent/'layout.json').read_text()),
                  intermediate_settings=settings[0], meso_settings=settings[1],
                  limitations='Three guide-selected regions of one seed. Higher small-scale power alone does not establish realistic mountains or drainage. Spectral distances use nominal north-south spacing; east-west spacing varies with latitude. Refinement can change land fraction; 30% is the continent-stage target.')
    target = paths[2].parent/'verification.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(stages=stats,regions=records,report=str(target)),indent=2),flush=True)


if __name__ == '__main__':
    main()
