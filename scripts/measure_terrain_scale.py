"""Independent Earth-reference benchmark for range envelopes and terrain scale.

Run from the project root. No diffusion models are evaluated, and generation
settings are never fitted to the same measurements used for evaluation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from planet_diffusion import cube, overview
from planet_diffusion import terrain_metrics as metrics

RADIUS = 6371000.
EARTH_CENTRES = [
    ('Rockies',-110,42),('Great Plains',-98,40),('Canadian interior',-106,54),
    ('Appalachians',-82,38),('Andean interior',-65,-22),('Amazon',-61,-6),
    ('Brazilian highlands',-46,-16),('Tibet',90,33),('Himalayan region',83,29),
    ('Central Asia',74,40),('Iran',57,33),('Mongolia',100,46),
    ('Central Siberia',95,55),('Kazakhstan',67,49),('China interior',110,35),
    ('India interior',79,23),('Congo basin',24,0),('Sahel',15,17),
    ('Ethiopian region',34,10),('Southern Africa',26,-25),
    ('Western Australia',126,-25),('Eastern Australia',142,-25),
    ('Russian interior',58,54),('Eastern Europe',30,48)]


def cube_average(a, size, spacing, lon0, lat0, radius, subsamples=4):
    # Integrate the same projected square-pixel footprint used for real terrain.
    out = np.zeros((size,size),np.float64)
    for dy in (np.arange(subsamples)+.5)/subsamples-.5:
        for dx in (np.arange(subsamples)+.5)/subsamples-.5:
            # Shift the tangent-plane pixel coordinates, not longitude degrees.
            # A larger coordinate grid with fractional node positions avoids
            # changing the physical footprint with latitude.
            lon,lat = np.deg2rad([lon0,lat0])
            centre = metrics.directions(lon,lat)
            east = np.array([-np.sin(lon),np.cos(lon),0.])
            north = np.array([-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)])
            xy = (np.arange(size)+.5-size/2)*spacing
            x,y = np.broadcast_arrays(xy[None,:]+dx*spacing,-xy[:,None]+dy*spacing)
            r = np.hypot(x,y)
            p = (np.cos(r/radius)[...,None]*centre
                 + np.sinc(r/(np.pi*radius))[...,None]*(x[...,None]*east+y[...,None]*north)/radius)
            out += cube.sample(a,p)[0]
    return (out/subsamples**2).astype(np.float32)


def globe_average(a, height, subsamples=4):
    out = np.zeros((height,2*height),np.float32)
    for start in range(0,height,32):
        rows = np.arange(start,min(height,start+32))[:,None]
        value = np.zeros((len(rows),2*height),np.float64)
        norm = np.zeros((len(rows),1),np.float64)
        for dy in (np.arange(subsamples)+.5)/subsamples-.5:
            theta = (rows+.5+dy)*np.pi/height
            weight = np.sin(theta)
            for dx in (np.arange(subsamples)+.5)/subsamples-.5:
                lon = -np.pi+(np.arange(2*height)+.5+dx)[None,:]*np.pi/height
                value += cube.sample(a,metrics.directions(lon,np.pi/2-theta))[0]*weight
                norm += weight
        out[start:start+len(rows)] = value/norm
    return out


def earth_patch(ds, name, lon, lat, size=128, spacing=8000):
    dst = np.full((size,size),np.nan,np.float32)
    half = size*spacing/2
    reproject(rasterio.band(ds,1),dst,src_transform=ds.transform,src_crs=ds.crs,
              dst_transform=from_origin(-half,half,spacing,spacing),
              dst_crs=f'+proj=aeqd +lat_0={lat} +lon_0={lon} +R={RADIUS} +units=m +no_defs',
              resampling=Resampling.average,dst_nodata=np.nan)
    return dst


def summarise(rows):
    if not rows:
        return None
    keys = ('length_km','effective_width_km','principal_width_km','area_km2')
    return {**{'count':len(rows)},**{key:np.quantile([r[key] for r in rows],[.1,.5,.9]).tolist() for key in keys}}


def band_summary(patches):
    if not patches:
        return None
    return [dict(wavelength_km=patches[0]['metrics']['bands'][i]['wavelength_km'],
                 rms_metres_q10_q50_q90=np.quantile([p['metrics']['bands'][i]['relief_rms_metres'] for p in patches],[.1,.5,.9]).tolist(),
                 variance_fraction_q10_q50_q90=np.quantile([p['metrics']['bands'][i]['fraction_of_resolved_band_variance'] for p in patches],[.1,.5,.9]).tolist())
            for i in range(len(patches[0]['metrics']['bands']))]


def candidate_centres():
    # Shuffle before taking a subset, so the sample does not fill up in the
    # northern hemisphere just because Fibonacci indices run north to south.
    for i in np.random.default_rng(81373).permutation(1024):
        lon = (int(i)*137.50776405)%360-180
        lat = np.rad2deg(np.arcsin(1-2*(i+.5)/1024))
        if abs(lat) <= 54:
            yield int(i),lon,float(lat)


def relief_classes(datasets):
    limits = np.quantile([p['metrics']['relief_90_percent_metres'] for p in datasets['Earth']],[1/3,2/3])
    names = ['Lower relief','Intermediate relief','Higher relief']
    result = {}
    for name,patches in datasets.items():
        # Keep Original/Auto paired in the same baseline relief class. Moving
        # suppressed mountains into a lower class would conceal that change.
        baseline = datasets['Original'] if name == 'Auto overview' else patches
        classes = np.searchsorted(limits,[p['metrics']['relief_90_percent_metres'] for p in baseline])
        result[name] = {}
        for i,label in enumerate(names):
            group = [p for p,c in zip(patches,classes) if c == i]
            result[name][label] = dict(count=len(group),bands=band_summary(group),
                relief_metres_q10_q50_q90=np.quantile([p['metrics']['relief_90_percent_metres'] for p in group],[.1,.5,.9]).tolist() if group else None,
                mean_elevation_metres_q10_q50_q90=np.quantile([p['metrics']['mean_elevation_metres'] for p in group],[.1,.5,.9]).tolist() if group else None)
    return dict(reference_relief_tercile_boundaries_metres=limits.tolist(),datasets=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='outputs/blobby-earth-native-guide')
    parser.add_argument('--reference',default='human_input/data_input_ready/heightmap.tif')
    parser.add_argument('--output',default='outputs/terrain-scale-benchmark')
    parser.add_argument('--global-height',type=int,default=1024)
    parser.add_argument('--generated-patches',type=int,default=48)
    parser.add_argument('--reference-patches',type=int,default=48)
    args = parser.parse_args()
    started = time.perf_counter()
    folder = Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    run = Path(args.run)
    meta = json.loads((run/'state/planet.json').read_text())
    if meta.get('terrain_scale',{}).get('strength',0) > 0:
        raise ValueError('Use a Learned-only or pre-overview run: applying the correction twice would invalidate the comparison')
    if not np.isclose(meta['radius_metres'],RADIUS):
        raise ValueError('This initial Earth benchmark requires the Earth-sized sphere')
    old = np.load(run/'checkpoints/cube-elevation.npy',mmap_mode='r')
    config_path = Path(meta['model'])/'config.json'
    if not config_path.is_file():
        raise ValueError('Supply an existing run with a local model calibration file')
    calibration = json.loads(config_path.read_text())
    residual = np.load(run/'checkpoints/residual.npy',mmap_mode='r')*calibration['residual_std']+calibration['residual_mean']
    config = overview.settings(SimpleNamespace(metadata=meta),old.shape[-1]-1,RADIUS)
    corrected = overview.cube_elevation(np.sign(old)*np.sqrt(np.abs(old)),residual,meta['seed'],config)
    del residual
    print('Preparing matched globe overviews...',flush=True)
    h = args.global_height
    with rasterio.open(args.reference) as ds:
        if ds.tags().get('NC_GLOBAL#title') != 'ETOPO1_Bed_c_gmt4.grd':
            raise ValueError('Reference provenance must identify the actual ETOPO1 bed elevation map')
        earth = ds.read(1,out_shape=(h,2*h),resampling=Resampling.average).astype(np.float32)
        reference_info = dict(path=args.reference,shape=list(ds.shape),source_tags=ds.tags(),
                              citation='https://doi.org/10.7289/V5C8276M',original_arcminutes=1.)
        named_patches = []
        for name,lon,lat in EARTH_CENTRES:
            a = earth_patch(ds,name,lon,lat)
            if not np.isfinite(a).all() or np.mean(a>0) < .98:
                continue
            named_patches.append(dict(name=name,centre=[lon,lat],metrics=metrics.spectrum(a,8000)))
        reference_patches = []
        for i,lon,lat in candidate_centres():
            p = metrics.patch_directions(lon,lat,16,64000,RADIUS)
            if np.mean(metrics.sample_global(earth,p)>0) < 1:
                continue
            a = earth_patch(ds,str(i),lon,lat)
            if not np.isfinite(a).all() or np.mean(a>0) < .98:
                continue
            reference_patches.append(dict(name=f'sample-{i}',centre=[lon,lat],metrics=metrics.spectrum(a,8000)))
            if len(reference_patches) >= args.reference_patches:
                break
        control = earth_patch(ds,'Rockies',-110,42)
    with open(args.reference,'rb') as handle:
        reference_info['sha256'] = hashlib.file_digest(handle,'sha256').hexdigest()
    fields = {'Earth':earth,'Original':globe_average(old,h),'Auto overview':globe_average(corrected,h)}
    np.savez_compressed(folder/'matched_global_fields.npz',**fields)
    belts = {}
    masks = {}
    lat = 90-(np.arange(h)+.5)*180/h
    domain = np.abs(lat)<60
    row_areas = RADIUS**2*(np.pi/h)*(np.sin(np.deg2rad(lat+90/h))-np.sin(np.deg2rad(lat-90/h)))/1e6
    for name,a in fields.items():
        print(f'Measuring geodesic mountain-belt envelopes: {name}',flush=True)
        rough = metrics.ruggedness(a,RADIUS,100)
        entries = {}
        land_area = float(np.sum(row_areas[:,None]*((a>0)&domain[:,None])))
        for threshold in (400,600,1000):
            mask = (a>300)&(rough>threshold)&domain[:,None]
            geometry = metrics.belt_geometry(mask,RADIUS,include_censored=True)
            large = [r for r in geometry if r['length_km'] >= 300 and not r['clipped_by_latitude_boundary']]
            censored = [r for r in geometry if r['clipped_by_latitude_boundary']]
            entries[str(threshold)] = dict(components=large,summary=summarise(large),
                censored_components=censored,
                censored_area_fraction_of_rugged_area=sum(r['area_km2'] for r in censored)/float(np.sum(row_areas[:,None]*mask)),
                rugged_area_fraction_of_land=float(np.sum(row_areas[:,None]*mask)/land_area),
                belts_per_million_land_km2=len(large)*1e6/land_area)
            if threshold == 600:
                masks[name] = mask
        belts[name] = dict(land_area_km2=land_area,thresholds=entries)
    print('Measuring area-averaged local terrain patches...',flush=True)
    original_patches,auto_patches = [],[]
    count = 0
    for i,lon,lat0 in candidate_centres():
        # Deterministic, approximately equal-area candidate centres; no picking
        # locations because they produce a desirable spectrum.
        p = metrics.patch_directions(lon,lat0,16,64000,RADIUS)
        if np.mean(metrics.sample_global(fields['Original'],p)>0)<1:
            continue
        a = cube_average(old,128,8000,lon,lat0,RADIUS)
        b = cube_average(corrected,128,8000,lon,lat0,RADIUS)
        if min(np.mean(a>0),np.mean(b>0)) < .98:
            continue
        original_patches.append(dict(name=f'sample-{i}',centre=[lon,lat0],metrics=metrics.spectrum(a,8000)))
        auto_patches.append(dict(name=f'sample-{i}',centre=[lon,lat0],metrics=metrics.spectrum(b,8000)))
        count += 1
        if count >= args.generated_patches:
            break
    # A counterexample test: doubling heights must leave normalized horizontal
    # spectra unchanged; stretching terrain should move variance to long bands.
    middle = control[32:96,32:96]
    axis = np.linspace(0,63,128)
    horizontal = np.stack([np.interp(axis,np.arange(64),row) for row in middle])
    stretched = np.stack([np.interp(axis,np.arange(64),col) for col in horizontal.T],axis=1)
    controls = dict(reference=metrics.spectrum(control,8000),double_heights=metrics.spectrum(control*2,8000),
                    double_horizontal_extent=metrics.spectrum(stretched,8000))
    datasets = {'Earth':reference_patches,'Original':original_patches,'Auto overview':auto_patches}
    spectrum_summaries = {name:band_summary(patches) for name,patches in datasets.items()}
    report = dict(reference=reference_info,generated_run=str(run),generated_seed=meta['seed'],
                  generated_guide_height=meta['coarse_height'],overview_settings=config,
                  method=dict(radius_metres=RADIUS,global_shape=[h,2*h],
                    global_meridional_spacing_km=np.pi*RADIUS/h/1000,
                    latitude_limit=60,local_relief_neighbourhood_radius_km=100,
                    relief_thresholds_metres=[400,600,1000],land_elevation_floor_metres=300,
                    geometry_minimum_area_km2=10000,reported_minimum_belt_length_km=300,
                    local_patch_shape=[128,128],local_spacing_metres=8000,
                    local_supported_wavelengths_km=[32,256],
                    spectrum='plane detrend, 2-D Hann window, squared FFT energy normalized by window energy; integrated annular band variances',
                    belt_definition='connected rugged-envelope proxies; no named-range or crest-path claim'),
                  belts=belts,spectra=spectrum_summaries,patches=datasets,controls=controls,
                  named_reference_patches=named_patches,relief_classes=relief_classes(datasets),
                  candidate_sampling='seeded permutation of 1024 equal-area Fibonacci centres, |latitude| <=54 degrees; >=98% land in 1024 km square patches',
                  seconds=time.perf_counter()-started,additional_model_calls=0,
                  limitations=['Initial benchmark: one generated globe/seed and one Earth DEM; no formal confidence interval.',
                    'Reference ETOPO1 is about 1.85 km at the equator; this analysis cannot validate sub-kilometre valleys.',
                    'Rugged belts can include several geological ranges, and curved belts distort principal-axis length/width.',
                    'Polar/ice terrain excluded; the mountain-belt proxy depends on relief threshold and sampling resolution.',
                    'Large all-land patches exclude coasts and narrow landmasses; this is an inland-terrain benchmark.',
                    'Components cut by the latitude boundary are recorded separately and excluded from length/width quantiles.',
                    'Relief-class comparisons need multiple seeds and independently held-out regions before fitting parameters.'])
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    axes = axes.ravel()
    colours = {'Earth':'#444444','Original':'#d27035','Auto overview':'#207ca0'}
    centres = [np.sqrt(lo*hi) for lo,hi in [(32,64),(64,128),(128,256)]]
    for name,rows in spectrum_summaries.items():
        values = np.array([r['variance_fraction_q10_q50_q90'] for r in rows])
        axes[0].plot(centres,values[:,1],'-o',label=name,color=colours[name])
        if name == 'Earth':
            axes[0].fill_between(centres,values[:,0],values[:,2],color=colours[name],alpha=.15,label='Earth 10–90% sample spread')
        amplitudes = np.array([r['rms_metres_q10_q50_q90'] for r in rows])
        axes[1].plot(centres,amplitudes[:,1],'-o',label=name,color=colours[name])
        if name == 'Earth':
            axes[1].fill_between(centres,amplitudes[:,0],amplitudes[:,2],color=colours[name],alpha=.15)
        geometry = belts[name]['thresholds']['600']['summary']
        if geometry:
            q = geometry['effective_width_km']
            axes[2].errorbar(q[1],name,xerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',color=colours[name],capsize=5)
        coverage = 100*belts[name]['thresholds']['600']['rugged_area_fraction_of_land']
        axes[3].bar(name,coverage,color=colours[name])
        axes[3].text(name,coverage+.6,f'{coverage:.1f}%',ha='center')
    axes[0].set(xscale='log',xlabel='Relief wavelength band centre (km)',ylabel='Fraction of resolved relief variance',
                title='Local relief structure (32–256 km)')
    axes[0].set_xticks(centres,['32–64','64–128','128–256'])
    axes[0].minorticks_off()
    axes[0].legend(fontsize=8)
    axes[1].set(xscale='log',xlabel='Relief wavelength band centre (km)',ylabel='Relief RMS (metres)',title='Absolute relief: texture alone is not enough')
    axes[1].set_xticks(centres,['32–64','64–128','128–256'])
    axes[1].minorticks_off()
    axes[2].set(xlabel='Effective rugged-belt width (km)',title='Unclipped belts: median and 10–90% spread')
    axes[3].set(ylabel='Percent of land classified as rugged',title='Mountain coverage under the same definition',ylim=(0,45))
    fig.savefig(folder/'comparison.png',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(13,4.5),layout='constrained',sharey=True)
    classes = report['relief_classes']['datasets']
    for ax,label in zip(axes,['Lower relief','Intermediate relief','Higher relief']):
        for name in datasets:
            group = classes[name][label]
            if not group['bands']:
                continue
            values = np.array([r['variance_fraction_q10_q50_q90'] for r in group['bands']])
            ax.plot(centres,values[:,1],'-o',color=colours[name],label=f'{name} (n={group["count"]})')
            if name == 'Earth':
                ax.fill_between(centres,values[:,0],values[:,2],color=colours[name],alpha=.15)
        ax.set(xscale='log',title=label,xlabel='Wavelength band (km)')
        ax.set_xticks(centres,['32–64','64–128','128–256'])
        ax.minorticks_off()
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Fraction of resolved relief variance')
    fig.savefig(folder/'relief_classes.png',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(3,1,figsize=(12,12),layout='constrained')
    for ax,(name,mask) in zip(axes,masks.items()):
        ax.imshow(mask,extent=(-180,180,-90,90),cmap='Greys',origin='upper',interpolation='nearest')
        ax.set(title=f'{name}: rugged mountain-belt proxy, 600 m relief within 100 km',ylim=(-60,60),xlabel='Longitude',ylabel='Latitude')
    fig.savefig(folder/'belt_maps.png',dpi=130)
    plt.close(fig)
    print(json.dumps({name:belts[name]['thresholds']['600']['summary'] for name in fields},indent=2),flush=True)
    print(f'Benchmark written to {folder} in {report["seconds"]:.1f}s',flush=True)


if __name__ == '__main__':
    main()
