"""Render the already exported seed-42 continent options using one elevation scale."""
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Normalize


def main():
    folder=Path('outputs/continents/continued-examples')
    palettes=np.array([[-6500,.035,.10,.23],[-2000,.04,.32,.60],[-200,.24,.65,.81],[0,.52,.80,.86],
                       [.01,.26,.49,.25],[500,.60,.67,.36],[1500,.69,.54,.34],[3000,.61,.47,.34],[6000,.95,.95,.90]])
    values=np.linspace(-6500,6000,1024)
    colors=np.stack([np.interp(values,palettes[:,0],palettes[:,i]) for i in range(1,4)],1)
    cmap=ListedColormap(colors);norm=Normalize(-6500,6000)
    choices=[('original','Original model'),('natural','Continued model / natural coverage'),
             ('land30','Continued model / 30% land'),('land30-relief125','Continued model / 30% land, relief 1.25×')]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    records=[]
    for ax,(name,title) in zip(axes.ravel(),choices):
        with rasterio.open(folder/name/'layout.tif') as ds:z=ds.read(1)
        meta=json.loads((folder/name/'layout.json').read_text());r=meta['geography']
        image=ax.imshow(z,cmap=cmap,norm=norm,extent=(-180,180,-90,90))
        ax.contour(z,levels=[0],colors='black',linewidths=.35,extent=(-180,180,-90,90),origin='upper')
        ax.set_title(f"{title}\n{r['land_fraction']*100:.1f}% land; land elevation 95th percentile {r['land_p95_m']:,.0f} m")
        ax.set_xticks([]);ax.set_yticks([])
        records.append(dict(name=name,layout=str(folder/name/'layout.tif'),controls=meta['controls'],geography=r))
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.75,label='Elevation relative to selected sea level (m)')
    fig.suptitle('Continent layout options: same seed 42, 64 sampling steps and elevation colours\nCoverage changes sea level; relief scales existing land height. Finer terrain comes later.')
    fig.savefig(folder/'options.png',dpi=130);plt.close(fig)
    (folder/'options.json').write_text(json.dumps(records,indent=2))
    print(folder/'options.png')


if __name__=='__main__':main()
