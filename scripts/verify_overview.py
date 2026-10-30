"""Compare overview synthesis against an existing learned cube checkpoint."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from planet_diffusion import cube, overview


def shade(a, metres_per_pixel):
    # Shared palette and relief illumination, without per-image normalization.
    dy,dx = np.gradient(a,metres_per_pixel)
    light = np.clip((1+.55*dx-.55*dy)/np.sqrt(1+dx*dx+dy*dy),.35,1.4)
    stops = [-6000,-2000,0,1,1000,2500,4500,7000]
    colours = np.array([[10,30,60],[22,66,100],[65,112,140],[52,93,61],
                        [105,127,72],[173,149,102],[177,172,159],[235,235,230]])
    rgb = np.stack([np.interp(a,stops,colours[:,i]) for i in range(3)],-1)
    return Image.fromarray(np.uint8(np.clip(rgb*light[:,:,None],0,255)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='outputs/blobby-earth-native-guide')
    parser.add_argument('--output',default='outputs/horizontal-scale-audit/implementation')
    args = parser.parse_args()
    run,output = Path(args.run),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    metadata = json.loads((run/'state/planet.json').read_text())
    old = np.load(run/'checkpoints/cube-elevation.npy',mmap_mode='r')
    residual = np.load(run/'checkpoints/residual.npy',mmap_mode='r')
    # Calibration for the checkpoint used by this verification (90 m).
    config_path = Path(metadata['model'])/'config.json'
    if not config_path.is_file():
        config_path = Path('models/terrain-diffusion-90m/config.json')
    calibration = json.loads(config_path.read_text())
    residual = residual*calibration['residual_std']+calibration['residual_mean']
    z = np.sign(old)*np.sqrt(np.abs(old))
    backend = SimpleNamespace(metadata=metadata)
    config = overview.settings(backend,old.shape[-1]-1,metadata['radius_metres'])
    started = time.perf_counter()
    new = overview.cube_elevation(z,residual,metadata['seed'],config)
    seconds = time.perf_counter()-started
    # Locate an inland highland window without selecting the biggest contrast.
    index = np.unravel_index(np.argmax(old[:,:4]),old[:,:4].shape)
    centre = cube.directions(index[1],index[2],index[3],old.shape[-1]-1)
    lon0 = np.rad2deg(np.arctan2(centre[1],centre[0]))
    lat0 = np.rad2deg(np.arcsin(centre[2]))
    west,east = lon0-6,lon0+6
    south,north = max(-90,lat0-6),min(90,lat0+6)
    lon = np.deg2rad(np.linspace(west,east,768))[None,:]
    lat = np.deg2rad(np.linspace(north,south,768))[:,None]
    p = np.stack(np.broadcast_arrays(np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
    a,b = cube.sample(old,p)[0],cube.sample(new,p)[0]
    spacing = np.pi/180*metadata['radius_metres']*(north-south)/767
    image = Image.new('RGB',(1536,814),'#17202b')
    image.paste(shade(a,spacing),(0,46))
    image.paste(shade(b,spacing),(768,46))
    draw = ImageDraw.Draw(image)
    draw.text((20,15),'Original learned terrain',(255,255,255))
    draw.text((788,15),'Auto overview: physical-distance relief',(255,255,255))
    image.save(output/'comparison.png')
    members,ids,owners,_ = cube.edge_groups(old.shape[-1]-1)
    flat = new.reshape(1,-1)
    report = dict(run=str(run),settings=config,added_reconstruction_seconds=seconds,
                  additional_model_calls=0,old_range=[float(old.min()),float(old.max())],
                  new_range=[float(new.min()),float(new.max())],
                  shared_node_error_metres=float(np.max(np.abs(flat[:,members]-flat[:,owners[ids]]))),
                  finite=bool(np.isfinite(new).all()),
                  old_land_fraction=float(np.mean(old>0)),new_land_fraction=float(np.mean(new>0)),
                  bounds=[west,south,east,north],
                  note='Post-reconstruction comparison using actual learned checkpoints; not a fresh model run or a physical geology validation.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
