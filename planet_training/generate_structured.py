"""Sample coastlines first, then relief at a fixed zero sea-level datum.

Exports only the continent-stage preview and guide. It never starts a full
intermediate/meso globe automatically.
"""
import argparse
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json
from .geography import export_pixels,measures
from .spherical_net import CoastReliefDenoiser
from .structured_data import STRUCTURED_VERSION,OBJECTIVE,decode_relief,DEFAULT_COAST_RUN,DEFAULT_RELIEF_RUN
from .train_structured import sample
from toy_meso.common import pick_device


def load(path,stage,device):
    payload = Path(path).read_bytes()
    ck = torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    digest = hashlib.sha256(payload).hexdigest()
    if ck.get('version')!=STRUCTURED_VERSION or ck.get('stage')!=stage:
        raise ValueError(f'Expected a new structured {stage} checkpoint')
    model = CoastReliefDenoiser(stage,**ck['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck['ema'])
    return model,ck,digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coast',default=DEFAULT_COAST_RUN+'/best.pt')
    p.add_argument('--relief',default=DEFAULT_RELIEF_RUN+'/best.pt')
    p.add_argument('--out',help='fresh output folder; defaults to a new timestamped preview folder')
    p.add_argument('--land-percent',type=float,default=30,help='learned coastline conditioning, applied before relief')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--height',type=int,default=256)
    p.add_argument('--device',default='auto')
    p.add_argument('--threads',type=int,default=4)
    a = p.parse_args(argv)
    a.out = a.out or str(Path('outputs/continents/structured-previews')/datetime.now().strftime('planet-%Y%m%d-%H%M%S-%f'))
    if not np.isfinite(a.land_percent) or not 0<a.land_percent<100 or a.steps<2 or a.height<2 or a.threads<1 or not 0<=a.seed<2**63:
        p.error('Invalid coverage, seed, steps, height or threads')
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError('Choose a fresh output folder')
    torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    coast_model,cc,coast_sha = load(a.coast,'coast',device)
    relief_model,rc,relief_sha = load(a.relief,'relief',device)
    if any(ck.get('objective')!=OBJECTIVE for ck in (cc,rc)):
        raise ValueError('Use corrected-objective models for generation; legacy checkpoints are available for audit only')
    if cc['dataset_id']!=rc['dataset_id'] or cc['config']['n']!=rc['config']['n']:
        raise ValueError('Coast and relief models must share the source dataset and native scale')
    lo,hi = cc['stats']['training_coverage_min']*100,cc['stats']['training_coverage_max']*100
    if not lo<=a.land_percent<=hi:
        raise ValueError(f'Coverage is outside training support ({lo:.1f}–{hi:.1f}%); train additional layouts first')
    coverage = torch.tensor([a.land_percent/100],device=device)
    coast_scores = sample(coast_model,cc['stats'],coverage,a.steps,a.seed)
    # Resolve the geometry before generating any elevation, without changing
    # ocean/land height. No quantile threshold is used to force coverage.
    coast = (coast_scores>.5).float()
    actual_coverage = relief_model.grid.mean(coast)[:,0]
    magnitude = sample(relief_model,rc['stats'],actual_coverage,a.steps,a.seed+1,coast)
    mask = coast[0,0].cpu().numpy()
    metres = decode_relief(mask,magnitude[0,0].cpu().numpy()).astype(np.float32)
    image = export_pixels(metres,a.height)
    out.mkdir(parents=True,exist_ok=True)
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(out/'layout.tif','w',driver='GTiff',height=a.height,width=2*a.height,
                       count=1,dtype='float32',crs='+proj=longlat +R=6371000 +no_defs',
                       transform=from_origin(-180,90,180/a.height,180/a.height),compress='deflate') as ds:
        ds.write(image,1);ds.set_band_unit(1,'m');ds.update_tags(units='metres',sea_level_m='0',stage=STRUCTURED_VERSION)
    np.save(out/'cube-elevation.npy',metres)
    np.save(out/'cube-coast.npy',mask)
    np.save(out/'cube-coast-scores.npy',coast_scores[0,0].cpu().numpy())
    report = measures(metres)
    coverage_error = report['land_fraction']*100-a.land_percent
    metadata = dict(vars(a),version=STRUCTURED_VERSION,objective=OBJECTIVE,sea_level_m=0,
                    coverage_method='learned conditioning before coastline generation; no elevation shift',
                    requested_land_percent=a.land_percent,native_land_percent=report['land_fraction']*100,
                    coverage_error_percentage_points=coverage_error,
                    geography=report,coast_checkpoint_sha256=coast_sha,relief_checkpoint_sha256=relief_sha,
                    coast_training_step=cc['step'],relief_training_step=rc['step'],
                    dataset_id=cc['dataset_id'],face_intervals=cc['config']['n'],
                    note='Requested coverage is a learned condition, not an exact constraint. Inspect samples and regional refinements before a full globe.')
    atomic_json(out/'layout.json',metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    from matplotlib.colors import ListedColormap
    axes[0].imshow(export_pixels(mask,a.height),cmap=ListedColormap(['#16465e','#a6af70']),vmin=0,vmax=1)
    axes[0].set_title(f'Coastline fixed before relief / {report["land_fraction"]*100:.1f}% native land')
    axes[1].imshow(image,cmap='terrain',vmin=-6000,vmax=6000)
    axes[1].set_title('Generated relief and bathymetry / sea level 0 m')
    for ax in axes: ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(out/'layout-preview.png',dpi=130);plt.close(fig)
    print(f'Saved preview and guide: {out}; target {a.land_percent:.1f}% / achieved {report["land_fraction"]*100:.1f}%',flush=True)
    if abs(coverage_error)>3:
        print(f'Coverage misses the requested condition by {abs(coverage_error):.1f} percentage points. More training/review is needed before refinement.',flush=True)


if __name__=='__main__': main()
