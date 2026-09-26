"""Terrain Diffusion on six coupled cube charts at every generation stage."""
import hashlib
import copy
import json
import time
from pathlib import Path
import numpy as np
from . import cube
from .cache import PredictionCache


def generate_cube(backend, seed, face_coarse=4, coarse_steps=20, progress=print,
                  audit=None, checkpoint_dir=None, draft=None, region=None):
    if not 0 <= seed < 2**64 or face_coarse < 2 or face_coarse % 2 or coarse_steps < 2:
        raise ValueError("Require unsigned 64-bit seed, even face_coarse >= 2, coarse_steps >= 2")
    if region is not None:
        from .region import validate_region
        bounds, width, height = region
        bounds = validate_region(bounds, width, height, face_coarse*512)
    if draft is not None:
        backend = copy.copy(backend)
        backend.cond_snr = np.array([draft.refinement,.2,1.,.2,1.],np.float32)
    digest = hashlib.sha256()
    for name in ("cube.py","cube_generate.py","backends.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    if draft is not None:
        digest.update(Path(__file__).with_name("draft.py").read_bytes())
    if region is not None:
        digest.update(Path(__file__).with_name("region.py").read_bytes())
    metadata = {**backend.metadata, "algorithm":"cubed-sphere-v3", "seed":seed,
                "coarse_steps":coarse_steps, "coarse_height":2*face_coarse,
                "face_coarse_intervals":face_coarse, "face_native_intervals":face_coarse*256,
                "native_height":face_coarse*512, "numpy":np.__version__, "units":"m",
                "vertical_datum":"model-defined zero sea level",
                "grid":"generation: shared-node cubed sphere; export: equirectangular pixel centres",
                "geometry":"six gnomonic charts; poles are regular face interiors",
                "noise":"unit Gaussian per unique cube node; nearest-node model halos",
                "source_sha256":digest.hexdigest()}
    if region is not None:
        metadata["algorithm"] = "cubed-sphere-regional-v1"
        metadata["detail_noise"] = "cube-tile-seeded-v1"
    if draft is not None:
        metadata["draft"] = draft.metadata
        metadata["conditioning_noise"] = backend.cond_snr.tolist()
    directory = Path(checkpoint_dir) if checkpoint_dir else None
    if directory is not None:
        directory.mkdir(parents=True,exist_ok=True)
        manifest = directory/"identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != metadata:
                raise ValueError("Cube checkpoints belong to different parameters, code, weights or runtime")
        else:
            if list(directory.glob("*.npy")):
                raise ValueError("Cube checkpoint directory has states but no identity")
            temp = directory/"identity.json.tmp"
            temp.write_text(json.dumps(metadata,indent=2)+"\n")
            temp.replace(manifest)
        if draft is not None:
            snapshot = directory/"draft.png"
            if not snapshot.exists():
                temp = directory/"draft.png.tmp"
                temp.write_bytes(draft.png_bytes)
                temp.replace(snapshot)

    def record(name,a,save=False):
        if not np.isfinite(a).all():
            raise FloatingPointError(f"Nonfinite cube state: {name}")
        if audit is not None:
            audit(name,a)
        if save and directory is not None:
            temp = directory/(name+".npy.tmp")
            with temp.open("wb") as handle:
                np.save(handle,a,allow_pickle=False)
            temp.replace(directory/(name+".npy"))
        return a

    def restore(name,channels,n):
        path = directory/(name+".npy") if directory else None
        if path is None or not path.exists():
            return None
        a = np.load(path,allow_pickle=False)
        if a.shape != (channels,6,n+1,n+1) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Invalid cube checkpoint {path}")
        progress(f"resumed {name}")
        return record(name,a)

    nc, nl, nd = face_coarse,face_coarse*32,face_coarse*256
    guides_started = time.perf_counter()
    latent = restore("latent",5,nl)
    if latent is None:
        coarse = restore("coarse",6,nc)
        if coarse is None:
            channels = [0,2,3,4,5]
            means, stds = backend.means,backend.stds
            raw_guide = cube.conditioning(seed,nc) if draft is None else draft.conditioning(seed,nc)
            record("raw-conditioning",raw_guide)
            guide = (raw_guide-means[channels,None,None,None])/stds[channels,None,None,None]
            angles = np.arctan(backend.cond_snr)[:,None,None,None]
            guide = record("conditioning",(np.cos(angles)*guide+np.sin(angles)*cube.noise(seed,0,5,nc)).astype(np.float32))
            state = record("coarse-noise",cube.noise(seed,1,6,nc)*backend.start_coarse(coarse_steps))
            for step in range(coarse_steps):
                progress(f"cube coarse {step+1}/{coarse_steps}")
                def predict(a,f,y,x):
                    cond = cube.read(guide,f,np.arange(y,y+16)[:,None],np.arange(x,x+16)[None,:],nearest=True)
                    return backend.coarse_predict(a,cond,step)
                pred = cube.consensus(state,predict,size=16,stride=8)
                state = record(f"coarse-{step}",cube.identify(backend.coarse_advance(pred,state,step)))
            coarse = backend.coarse_finish(state)*stds[:,None,None,None]+means[:,None,None,None]
            coarse[1] = coarse[0]-coarse[1]
            coarse = record("coarse",cube.identify(coarse),save=True)
        state = np.zeros((5,6,nl+1,nl+1),np.float32)
        for step,t in enumerate([float(np.arctan(160)),float(np.arctan(.7))]):
            progress(f"cube latent {step+1}/2")
            xt = record(f"latent-noisy-{step}",(np.cos(t)*state+np.sin(t)*cube.noise(seed,5819+step,5,nl)).astype(np.float32))
            def predict(a,f,y,x):
                cond = cube.read(coarse,f,(y/32-1+np.arange(4))[:,None],(x/32-1+np.arange(4))[None,:])
                return backend.predict("latent",a,cond,t)
            pred = cube.consensus(xt,predict,size=64,stride=32,
                                  progress=lambda d,n:progress(f"cube latent {step+1}/2: {d}/{n} patches"))
            state = cube.identify(np.cos(t)*xt+np.sin(t)*pred)
            state = record(f"latent-{step}",state)
        latent = record("latent",state,save=True)

    if region is not None:
        from .region import generate_region
        guide_seconds = time.perf_counter()-guides_started
        progress(f"Regional detail: global guides ready in {guide_seconds:.1f}s; decoding requested cube patches and halos")
        result, metadata = generate_region(backend,latent,seed,bounds,width,height,metadata,directory,progress)
        metadata['regional_execution']['guide_seconds'] = guide_seconds
        return result, metadata

    residual = restore("residual",1,nd)
    if residual is None:
        t = float(np.arctan(160))
        xt = record("decoder-noise",cube.noise(seed,6819,1,nd)*np.sin(t))
        def decode(a,f,y,x):
            cond = cube.read(latent[:4],f,np.floor_divide(np.arange(y,y+512),8)[:,None],
                             np.floor_divide(np.arange(x,x+512),8)[None,:],nearest=True)
            return backend.predict("decoder",a,cond,t)
        if directory is not None:
            identity = {"generation":metadata,"latent_sha256":hashlib.sha256(latent.tobytes()).hexdigest()}
            decode = PredictionCache(directory/"decoder",identity,decode)
        pred = cube.consensus(xt,decode,size=512,stride=256,
                              progress=lambda d,n:progress(f"cube decoder: {d}/{n} patches"))
        residual = record("residual",cube.identify(np.cos(t)*xt+np.sin(t)*pred),save=True)
    progress("cube low-frequency reconstruction")
    z = cube.reconstruct(residual*backend.residual_std+backend.residual_mean,latent[4:5]*38.6-31.4)
    elevation = record("cube-elevation",np.sign(z)*z*z,save=True)
    progress("equirectangular export projection")
    projected = cube.to_equirectangular(elevation,nd*2)[0]
    return projected,metadata
