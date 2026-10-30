"""Read-only terrain audit. Creates comparisons; never trains or generates a globe."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform
from rasterio.windows import from_bounds, Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT/'outputs/terrain-structure-audit'
R = 6371000.
GEO = '+proj=longlat +R=6371000 +no_defs'
FILES = {
    'Layout': ROOT/'outputs/continents/structured-previews/reviewed-10000-seed42/layout.tif',
    'Intermediate': ROOT/'outputs/intermediate/structured-10000-seed42/planet.tif',
    'New meso': ROOT/'human_input/For_Comparison/meso_diff/planet.tif',
    'B2W guide': ROOT/'human_input/B2W.png',
    'planetC': ROOT/'outputs/meso/planetC/planet.tif',
    'Modern Earth': ROOT/'data/etopo_2022_30s_surface.tif',
    'Paleo 0 Ma': next((ROOT/'data').rglob('*Holocene_0Ma.nc')),
    'Paleo 60 Ma': next((ROOT/'data').rglob('*_60Ma.nc')),
}
CMAP = LinearSegmentedColormap.from_list('fixed_elevation',
    [(0,'#163657'),(.42,'#377db0'),(.4999,'#9bcdd1'),(.5,'#50774c'),
     (.525,'#a1b67b'),(.58333,'#c9bc85'),(.66667,'#bd9770'),(.75,'#ad9587'),
     (.83333,'#dbd4cb'),(1,'#ffffff')])
NORM = Normalize(-6000,6000)

def legend(fig,axs):
    sm=matplotlib.cm.ScalarMappable(norm=NORM,cmap=CMAP)
    cb=fig.colorbar(sm,ax=np.asarray(axs).ravel().tolist(),orientation='horizontal',fraction=.018,pad=.015,aspect=65)
    cb.set_ticks([-6000,-3000,0,1000,2000,4000,6000]);cb.set_label('Elevation (m); colour limits ±6000 m, values outside are colour-clipped only',fontsize=9)

def image(z,spacing,exaggeration=5):
    dy,dx = np.gradient(z,spacing)
    normal = np.stack((-dx*exaggeration,dy*exaggeration,np.ones_like(z)),axis=-1)
    normal /= np.linalg.norm(normal,axis=-1)[...,None]
    az,alt = np.radians([315,45])
    light = np.array([np.sin(az)*np.cos(alt),np.cos(az)*np.cos(alt),np.sin(alt)])
    intensity = np.clip(normal@light,0,1)
    return np.clip(CMAP(NORM(z))[...,:3]*(.48+.65*intensity[...,None]),0,1)

def sample(name,lat,lon,n=416,spacing=2880):
    extent = n*spacing
    crs = f'+proj=aeqd +lat_0={lat} +lon_0={lon} +R={R} +units=m +no_defs'
    if name in ('Saved planetC guide','Native relief','Training 60 Ma target'):
        from planet_diffusion import cube
        from toy_meso.sphere import encode,decode
        q=(np.arange(n)+.5-n/2)*spacing
        xx,yy=np.meshgrid(q,-q)
        lo,la=transform(crs,GEO,xx.ravel(),yy.ravel())
        la,lo=np.radians(la),np.radians(lo)
        points=np.stack((np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)),axis=-1).reshape(n,n,3)
        if name=='Saved planetC guide':
            coarse=np.load(ROOT/'outputs/meso/planetC/coarse.npy')
            return decode(cube.sample(encode(coarse),points)[0])
        if name=='Native relief':
            coarse=np.load(ROOT/'outputs/continents/structured-previews/reviewed-10000-seed42/cube-elevation.npy')
        else:
            from planet_training.data import PaleoRaster,layout_target
            coarse=decode(layout_target(PaleoRaster.from_file(FILES['Paleo 60 Ma']),np.eye(3)))
        return cube.sample(coarse[None],points)[0]
    path = FILES[name]
    axis = np.linspace(-extent/2,extent/2,41)
    xx,yy = np.meshgrid(axis,axis)
    lons,lats = transform(crs,GEO,xx.ravel(),yy.ravel())
    bounds = [min(lons)-1,min(lats)-1,max(lons)+1,max(lats)+1]
    with rasterio.open(path) as ds:
        tr = from_origin(-180,90,360/ds.width,180/ds.height) if path.suffix=='.png' else ds.transform
        win = from_bounds(*bounds,transform=tr).round_offsets().round_lengths()
        win = win.intersection(Window(0,0,ds.width,ds.height))
        z = ds.read(1,window=win).astype(np.float32)
        if path.suffix=='.png':
            with Image.open(path) as im:
                z=np.asarray(im.convert('L').crop((int(win.col_off),int(win.row_off),int(win.col_off+win.width),int(win.row_off+win.height))),dtype=np.float32)
            z = -6000+z/255*12000
        target = np.empty((n,n),np.float32)
        # Use the same sphere for every case; avoid ellipsoid changes in scale.
        reproject(z,target,src_transform=rasterio.windows.transform(win,tr),src_crs=GEO,
                  dst_transform=from_origin(-extent/2,extent/2,spacing,spacing),dst_crs=crs,
                  resampling=Resampling.bilinear if name.startswith('Paleo') or name=='Layout' else Resampling.average,
                  src_nodata=ds.nodata,num_threads=2)
    return target

def gauss(z,sigma):
    """Separable physical smoothing, no optional scipy dependency."""
    r=int(np.ceil(3*sigma)); x=np.arange(-r,r+1)
    k=np.exp(-.5*(x/sigma)**2);k/=k.sum()
    z=np.apply_along_axis(lambda a:np.convolve(np.pad(a,(r,r),mode='reflect'),k,mode='valid'),1,z)
    return np.apply_along_axis(lambda a:np.convolve(np.pad(a,(r,r),mode='reflect'),k,mode='valid'),0,z)

def metrics(z,spacing=2880):
    land=z>0
    smooth=gauss(z,23040/spacing)
    dy,dx=np.gradient(smooth,spacing/1000)
    slope=np.hypot(dx,dy)
    # Relief is a 69-km local standard deviation, not pointwise elevation.
    mean=gauss(z,23040/spacing)
    rough=np.sqrt(np.maximum(gauss(z*z,23040/spacing)-mean*mean,0))
    return dict(land_fraction=float(land.mean()),land_median_m=float(np.median(z[land])) if land.any() else None,
        above_1000m_fraction=float((z>1000).mean()),above_2000m_fraction=float((z>2000).mean()),
        land_broad_slope_m_per_km_p50_p90=[float(v) for v in np.percentile(slope[land],[50,90])] if land.any() else [],
        low_roughness_land_fraction=float(((rough<50)&land).sum()/max(1,land.sum())),
        rms_m=float(np.sqrt(np.mean(z*z))))

def overview():
    names=['Layout','Intermediate','New meso','B2W guide','planetC','Modern Earth','Paleo 0 Ma','Paleo 60 Ma']
    fig,axs=plt.subplots(4,2,figsize=(15,15),constrained_layout=True)
    for ax,name in zip(axs.flat,names):
        with rasterio.open(FILES[name]) as ds:
            z=ds.read(1,out_shape=(720,1440),resampling=Resampling.average).astype(np.float32)
            if name=='B2W guide':
                with Image.open(FILES[name]) as im:z=np.asarray(im.convert('L').resize((1440,720),Image.Resampling.BOX),dtype=np.float32)
                z=-6000+z/255*12000
        ax.imshow(CMAP(NORM(z)),extent=(-180,180,-90,90))
        ax.set_title(name);ax.set_xticks(np.arange(-180,181,60));ax.set_yticks([-60,0,60]);ax.grid(alpha=.15)
    fig.suptitle('Global locator only: identical elevation colours; sea level = 0 m')
    legend(fig,axs)
    fig.savefig(OUT/'global-locator.png',dpi=120);plt.close(fig)

NEW_REGIONS=[('Reviewed highland',23.1221198,-70.1958525),('Central broad upland',-5,-60),('Northern lowland',30,-87),('Reviewed upland',-31.8317972,102.3387097),('Reviewed coast',-4.2511521,-21.6705069)]
EARTH_REGIONS=[('Alps and foreland',45.5,9),('California paired ranges',37,-120),('Rockies / Great Plains',42,-103)]

def panel(ax,z,title,extent=1198.08):
    ax.imshow(image(z,2880),extent=(-extent/2,extent/2,-extent/2,extent/2))
    ax.set_title(title,fontsize=11);ax.set_xticks([-400,0,400]);ax.set_yticks([-400,0,400]);ax.tick_params(labelsize=8)
    ax.plot([-500,-300],[-500,-500],color='black',lw=3);ax.text(-400,-465,'200 km',ha='center',fontsize=8)

def comparisons():
    report={'method':'1198.08 km square spherical azimuthal equidistant windows; 416 x 416 at 2.88 km. Same fixed -6000..6000 m colours and NW 45-degree light, 5x vertical exaggeration. ETOPO and generated TIFFs area resampled; point Paleo grids and layout bilinear. Display interpolation creates no new Paleo information. Local roughness uses Gaussian sigma 23.04 km; broad slope is gradient of that smoothed field.', 'new':[], 'earth':[]}
    for group,regions,names,filename in [('new',NEW_REGIONS,['Layout','Intermediate','New meso'],'new-stage-closeups.png'),('earth',EARTH_REGIONS,['Saved planetC guide','planetC','Modern Earth','Paleo 0 Ma'],'reference-closeups.png')]:
        fig,axs=plt.subplots(len(regions),len(names),figsize=(4*len(names),4*len(regions)),constrained_layout=True)
        for i,(label,lat,lon) in enumerate(regions):
            fields=[];entry={'region':label,'lat':lat,'lon':lon,'fields':{}}
            for j,name in enumerate(names):
                z=sample(name,lat,lon);fields.append(z)
                np.save(OUT/f'{group}-{i}-{j}.npy',z)
                panel(axs[i,j],z,f'{label}\n{name} ({lat:.2f}, {lon:.2f})')
                entry['fields'][name]=metrics(z)
            if group=='new':
                entry['differences']={f'{names[k+1]} minus {names[k]}':{'rms_m':float(np.sqrt(np.mean((fields[k+1]-fields[k])**2))), 'smoothed_23km_rms_m':float(np.sqrt(np.mean(gauss(fields[k+1]-fields[k],8)**2)))} for k in range(2)}
            report[group].append(entry)
        fig.suptitle('Stage comparison' if group=='new' else 'Earth structure supplied by B2W versus modern and Paleo references',fontsize=16)
        fig.supxlabel('All panels 1198 km square; axes in km; identical colours and 5x hillshade')
        legend(fig,axs)
        fig.savefig(OUT/filename,dpi=130);plt.close(fig)
    (OUT/'regional-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['new'],indent=2))

def finish_figures():
    regions=[('Northern highland (60 Ma)',46,-97),('Southern highland (60 Ma)',-17,-62),('Northern interior (60 Ma)',50,-85)]
    fig,axs=plt.subplots(3,2,figsize=(9,13),constrained_layout=True)
    for i,(label,lat,lon) in enumerate(regions):
        for j,name in enumerate(['Paleo 60 Ma','Training 60 Ma target']):
            panel(axs[i,j],sample(name,lat,lon),label+'\n'+name)
    fig.suptitle('Actual training-age examples: 1-degree source and ~161-km averaged target\n1198 km square, same colours, 5x hillshade; reconstructed geography at 60 Ma')
    legend(fig,axs)
    fig.savefig(OUT/'training-target-closeups.png',dpi=120);plt.close(fig)
    native=[]
    for label,lat,lon in NEW_REGIONS:
        z=sample('Native relief',lat,lon);exported=sample('Layout',lat,lon)
        native.append({'region':label,'native_minus_export_rms_m':float(np.sqrt(np.mean((z-exported)**2)))})
    (OUT/'native-export-check.json').write_text(json.dumps(native,indent=2)+'\n')
    fig,axs=plt.subplots(3,4,figsize=(16,12),constrained_layout=True)
    for row,((label,lat,lon),(earthlabel,elat,elon)) in enumerate(zip(NEW_REGIONS[:3],EARTH_REGIONS)):
        for col,(name,la,lo) in enumerate([('New meso',lat,lon),('planetC',elat,elon),('Modern Earth',elat,elon),('Paleo 0 Ma',elat,elon)]):
            panel(axs[row,col],sample(name,la,lo),f'{name}\n'+(label if col==0 else earthlabel))
    fig.suptitle('Matched physical scales: generated structure, older result, modern Earth, PaleoDEM\nAll panels 1198 km square; axes in km; fixed metres/colour and 5x NW hillshade')
    legend(fig,axs)
    fig.savefig(OUT/'three-case-closeups.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    x=(np.arange(416)+.5-208)*2.88
    choices=[('Reviewed new highland: west-east',0,225,'h'),('Central new upland: west-east',1,208,'h'),('Alps: south-north at centre longitude',0,208,'v'),('California: west-east at centre latitude',1,208,'h')]
    stats=[]
    for k,(title,row,line,direction) in enumerate(choices):
        ax=axs.flat[k]
        fields=['Layout','Intermediate','New meso'] if k<2 else ['Saved planetC guide','planetC','Modern Earth','Paleo 0 Ma']
        for j,name in enumerate(fields):
            z=np.load(OUT/f'{"new" if k<2 else "earth"}-{row}-{j}.npy')
            p=z[line] if direction=='h' else z[::-1,line]
            ax.plot(x,p,label=name,lw=1.5,alpha=.9)
            # Widths are transparent threshold measures, not mountain classification.
            edges=np.diff(np.r_[False,p>1000,False].astype(int))
            lengths=(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*2.88
            stats.append({'transect':title,'field':name,'longest_above_1000m_segment_km':float(max(lengths,default=0))})
        ax.set_title(title);ax.set_ylim(-1000,3500);ax.axhline(0,color='black',lw=.5);ax.set_xlabel('Distance (km)');ax.set_ylabel('Elevation (m)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Physical transects: identical horizontal and vertical axes; no vertical exaggeration\nNew regions are examples, not geographic matches to the Earth transects')
    fig.savefig(OUT/'elevation-transects.png',dpi=140);plt.close(fig)
    (OUT/'transect-metrics.json').write_text(json.dumps(stats,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--overview',action='store_true');p.add_argument('--compare',action='store_true');p.add_argument('--finish',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.overview:overview()
    if a.compare:comparisons()
    if a.finish:finish_figures()

if __name__=='__main__':main()
