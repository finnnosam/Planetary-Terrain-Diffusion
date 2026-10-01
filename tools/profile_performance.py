"""Reproducible cache, reconstruction, and trained CUDA precision measurements.

Run with the project interpreter: python tools/profile_performance.py --mode all.
Reports include the pinned upstream revision and quality/repeatability metrics;
each run uses its own output directory and never overwrites existing checkpoints.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from planet_diffusion.backends import TerrainBackend, UPSTREAM_COMMIT
from planet_diffusion.sparse import SparseWorld, Stage


def cache_profile(directory):
    report = []
    coords = np.arange(64,704)
    for budget in (4,16,64):
        stage = Stage(directory/str(budget),{},'patches',1024,6,64,32,
            lambda f,y,x:np.full((6,64,64),f+y+x,np.float32),
            patch_cache_bytes=budget*1024**2)
        reads = []
        for _ in range(2):
            for field in stage.fields:
                field.block.cache_clear()
            before = stage.cache.hits
            started = time.perf_counter()
            value = stage.read(0,coords[:,None],coords[None,:])
            reads.append(dict(seconds=time.perf_counter()-started,
                              disk_reads=stage.cache.hits-before,checksum=float(value.sum())))
        report.append(dict(patch_budget_mib=budget,reads=reads,model_patches=stage.cache.misses,
            retained_patch_bytes=stage.patch_memory.bytes,retained_blend_bytes=stage.blend_memory.bytes,
            validation_scans=stage.cache.validation_scans))
    return report


def reconstruction_profile():
    rng = np.random.default_rng(12)
    halo = rng.normal(size=(102,96)).astype(np.float32)
    weights = np.exp(-.5*(np.arange(-3,4)/3)**2)
    weights /= weights.sum()
    windows = np.lib.stride_tricks.sliding_window_view(halo,7,axis=0)
    started = time.perf_counter()
    for _ in range(500):
        old = np.zeros((96,96),np.float64)
        for i,w in enumerate(weights):
            old += w*halo[i:i+96]
        old = old.astype(np.float32)
    scalar_seconds = time.perf_counter()-started
    started = time.perf_counter()
    for _ in range(500):
        new = np.einsum('...k,k->...',windows,weights,dtype=np.float64).astype(np.float32)
    return dict(tap_loop_seconds=scalar_seconds,vector_seconds=time.perf_counter()-started,
                max_absolute_difference=float(np.max(np.abs(new-old))))


def error(actual, expected):
    difference = actual.astype(np.float64)-expected
    return dict(rmse=float(np.sqrt(np.mean(difference**2))),
                max_absolute=float(np.max(np.abs(difference))),
                relative_rmse=float(np.sqrt(np.mean(difference**2))/max(1e-12,np.sqrt(np.mean(expected.astype(np.float64)**2)))))


def precision_profile(directory,model,steps):
    import torch
    torch.set_num_threads(4)
    backend = TerrainBackend(ROOT/'upstream',str(model),'unused','cuda')
    baseline = {}
    report = []
    rng = np.random.default_rng(42)
    inputs = [("latent",rng.normal(size=(5,64,64)).astype(np.float32),
               np.broadcast_to(backend.means[:,None,None],(6,4,4)).copy()),
              ("decoder",rng.normal(size=(1,512,512)).astype(np.float32),
               rng.normal(size=(4,512,512)).astype(np.float32))]
    for precision in ('float32','tf32','bfloat16'):
        if precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            report.append(dict(precision=precision,unavailable=True))
            continue
        backend.precision = precision
        backend.metadata.pop('precision',None)
        if precision != 'float32':
            backend.metadata['precision'] = precision
        stages = {}
        for stage,a,cond in inputs:
            backend.predict(stage,a,cond,.7)  # Warm kernels independently of timing.
            started = time.perf_counter()
            first = backend.predict(stage,a,cond,.7)
            second = backend.predict(stage,a,cond,.7)
            stages[stage] = dict(seconds_per_call=(time.perf_counter()-started)/2,
                                bitwise_repeatable=bool(np.array_equal(first,second)))
            if precision == 'float32':
                baseline[stage] = first
            stages[stage]['versus_float32'] = error(first,baseline[stage])
        regions = []
        for repeat in range(2):
            started = time.perf_counter()
            world = SparseWorld(backend,42,1024,steps,6371000.,directory/precision/str(repeat))
            elevation,metadata,_ = world.region((12,-12.04,12.04,-12),8,8,progress=lambda _:None)
            regions.append(elevation)
            print(f'{precision}: independent regional repeat {repeat+1} in {time.perf_counter()-started:.1f}s',flush=True)
        if precision == 'float32':
            baseline['region'] = regions[0]
        report.append(dict(precision=precision,stages=stages,
            elevation_metres_versus_float32=error(regions[0],baseline['region']),
            region_bitwise_repeatable=bool(np.array_equal(*regions)),
            regional_execution=metadata['regional_execution']))
    return dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,
        model_revision=backend.metadata['model_revision'],coarse_steps=steps,
        seed=42,bounds=[12,-12.04,12.04,-12],output_pixels=[8,8],modes=report)


def auto_profile(directory,model,steps):
    """Check two fresh Auto backends/generations and a lazy saved query."""
    import gc
    import torch
    from planet_diffusion.model_presets import sparse_guide_height
    from planet_diffusion.storage import save_state
    from planet_diffusion.query import query_region
    torch.set_num_threads(4)
    outputs, selections, timings = [],[],[]
    guide_height = sparse_guide_height(str(model))
    bounds = (12,-12.04,12.04,-12)
    for repeat in range(2):
        backend = TerrainBackend(ROOT/'upstream',str(model),'unused','auto')
        selections.append(backend.precision)
        print(f'Auto: {model.name}, {backend.device}, {backend.precision}, repeat {repeat+1}',flush=True)
        world = SparseWorld(backend,42,guide_height,steps,6371000.,directory/str(repeat)/'checkpoints')
        elevation,metadata,_ = world.region(bounds,8,8,progress=lambda _:None)
        outputs.append(elevation)
        timings.append(metadata['regional_execution'])
        if repeat == 1:
            state = directory/str(repeat)/'state'
            save_state(state,elevation,metadata)
        del world,backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Queries explicitly use the resolved mode in the saved identity.
    backend = TerrainBackend(ROOT/'upstream',str(model),'unused',metadata['device'],
                             precision=metadata.get('precision','float32'))
    queried,_,_ = query_region(backend,state,bounds,8,8,
        checkpoint_dir=directory/'1/checkpoints',progress=lambda _:None)
    report = dict(model=str(model),model_revision=metadata['model_revision'],
        device=metadata['device'],guide_height=guide_height,coarse_steps=steps,
        selections=selections,independent_generations_equal=bool(np.array_equal(*outputs)),
        cached_query_equal=bool(np.array_equal(queried,outputs[-1])),
        cached_query_loaded_models=list(backend.models),runs=timings)
    if (not report['independent_generations_equal'] or not report['cached_query_equal']
            or selections[0] != selections[1] or backend.models):
        raise RuntimeError(f'Auto reproducibility failed: {report}')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['caches','precision','auto','all'],default='caches')
    parser.add_argument('--model',type=Path,default=ROOT/'models/terrain-diffusion-90m')
    parser.add_argument('--coarse-steps',type=int,default=20)
    args = parser.parse_args()
    directory = ROOT/'outputs'/('performance-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True)
    report = dict(upstream_commit=UPSTREAM_COMMIT)
    if args.mode in ('caches','all'):
        report['caches'] = cache_profile(directory/'caches')
        report['reconstruction'] = reconstruction_profile()
    if args.mode in ('precision','all'):
        report['precision'] = precision_profile(directory/'precision',args.model,args.coarse_steps)
    if args.mode == 'auto':
        report['auto'] = auto_profile(directory/'auto',args.model,args.coarse_steps)
    path = directory/'report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(path,flush=True)


if __name__ == '__main__':
    main()
