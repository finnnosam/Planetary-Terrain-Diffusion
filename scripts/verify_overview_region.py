"""Fresh real-model regional comparison at an affordable global grid density."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from planet_diffusion import cube
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.draft import Draft
from planet_diffusion.region import generate_region
from planet_diffusion.sparse import SparseWorld
from verify_overview import shade


def main():
    import torch
    torch.set_num_threads(4)
    folder = Path('outputs/horizontal-scale-audit/fresh-region-36-final')
    folder.mkdir(parents=True,exist_ok=True)
    backend = TerrainBackend('upstream','models/terrain-diffusion-90m',MODEL_REVISION,'cuda')
    source = Draft('outputs/blobby-earth-native-guide/checkpoints/draft.png',maximum=6250.,refinement=.2)
    world = SparseWorld(backend,2644365341129234824,36,20,6371000.,folder/'checkpoints',source=source)
    bounds = [-113.5,36,-107.5,42]
    started = time.perf_counter()
    auto,auto_meta,_ = world.region(bounds,256,256,progress=lambda text:print(text,flush=True))
    auto_seconds = time.perf_counter()-started
    metadata = dict(world.metadata)
    metadata['terrain_scale'] = {**metadata['terrain_scale'],'mode':'learned','strength':0.}
    learned,learned_meta = generate_region(backend,world.latent,world.seed,bounds,256,256,
                                           metadata,None,lambda text:print(text,flush=True))
    spacing = np.pi/180*6371000.*6/256
    comparison = Image.new('RGB',(1024,560),'#17202b')
    comparison.paste(shade(learned,spacing).resize((512,512)),(0,48))
    comparison.paste(shade(auto,spacing).resize((512,512)),(512,48))
    draw = ImageDraw.Draw(comparison)
    draw.text((16,16),'Original learned, guide height 36',(255,255,255))
    draw.text((528,16),'Auto overview, same model guides',(255,255,255))
    comparison.save(folder/'comparison.png')
    report = dict(guide_height=36,bounds=bounds,precision=getattr(backend,'precision','float32'),
                  auto_seconds=auto_seconds,auto_range=[float(auto.min()),float(auto.max())],
                  learned_range=[float(learned.min()),float(learned.max())],
                  mean_absolute_change_metres=float(np.mean(np.abs(auto-learned))),
                  learned_decoder_calls=learned_meta['regional_execution']['decoder_model_calls'],
                  auto_decoder_calls=auto_meta['regional_execution']['decoder_model_calls'],
                  settings=world.metadata['terrain_scale'],
                  note='Fresh learned coarse/latent/decoder inference for a regional crop on the coarse-height-36 globe grid, using identical guides for both reconstructions.')
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
