"""Terrain Diffusion on six coupled cube charts at every generation stage."""
import hashlib
import copy
import json
import time
from pathlib import Path
import numpy as np
from . import cube
from .cache import PredictionCache
from . import procedural
from .coarse import sample_coarse, VERSION as COARSE_VERSION, TILE_SIZE, TILE_STRIDE
from .detail import sample_latents, decoder_conditioning, DECODER_SIZE, DECODER_STRIDE, VERSION as DETAIL_VERSION


def generate_cube(backend, seed, face_coarse=4, coarse_steps=20, progress=print,
                  audit=None, checkpoint_dir=None, draft=None, region=None, latent_batch_size=1,
                  with_climate=False, conditioning=None, decoder_cache_bytes=64*1024**2):
    """Return elevation/metadata; with_climate adds compact climate features as a third result."""
    if (isinstance(latent_batch_size, bool) or not isinstance(latent_batch_size, (int, np.integer))
            or latent_batch_size < 1):
        raise ValueError("latent_batch_size must be a positive integer")
    if not 0 <= seed < 2**64 or face_coarse < 2 or face_coarse % 2 or coarse_steps < 2:
        raise ValueError("Require unsigned 64-bit seed, even face_coarse >= 2, coarse_steps >= 2")
    if region is not None:
        from .region import validate_region
        bounds, width, height = region
        bounds = validate_region(bounds, width, height, face_coarse*512)
    if draft is not None and conditioning is not None:
        raise ValueError("Choose either a PNG draft or a conditioning TIFF folder")
    source = conditioning if conditioning is not None else draft
    if source is not None:
        backend = copy.copy(backend)
        backend.cond_snr = (conditioning.cond_snr.copy() if conditioning is not None
                            else np.array([draft.refinement,.2,1.,.2,1.],np.float32))
    digest = hashlib.sha256()
    for name in ("cube.py","cube_generate.py","backends.py","procedural.py","coarse.py","detail.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for name in ("synthetic_map_stats.json", "elevation_reference.npz"):
        digest.update((procedural.DATA/name).read_bytes())
    frequency, drop_water = procedural.validate(
        getattr(backend, "frequency_mult", procedural.DEFAULT_FREQUENCY),
        getattr(backend, "drop_water_pct", procedural.DEFAULT_DROP_WATER))
    if draft is not None:
        digest.update(Path(__file__).with_name("draft.py").read_bytes())
    if source is not None:
        digest.update(Path(__file__).with_name("spherical_raster.py").read_bytes())
    if conditioning is not None:
        digest.update(Path(__file__).with_name("conditioning.py").read_bytes())
    if region is not None:
        digest.update(Path(__file__).with_name("region.py").read_bytes())
    if with_climate:
        digest.update(Path(__file__).with_name("climate.py").read_bytes())
    metadata = {**backend.metadata, "algorithm":"cubed-sphere-v3", "seed":seed,
                "coarse_steps":coarse_steps, "coarse_height":2*face_coarse,
                "face_coarse_intervals":face_coarse, "face_native_intervals":face_coarse*256,
                "native_height":face_coarse*512, "numpy":np.__version__, "units":"m",
                "vertical_datum":"model-defined zero sea level",
                "grid":"generation: shared-node cubed sphere; export: equirectangular pixel centres",
                "geometry":"six equi-angular charts; poles are regular face interiors",
                "noise":"unit Gaussian per unique cube node; nearest-node model halos",
                "source_sha256":digest.hexdigest(), "latent_batch_size":int(latent_batch_size)}
    import importlib.metadata
    metadata["procedural_conditioning"] = {
        "version":"source-perlin-sphere-v1", "frequency_mult":list(frequency),
        "drop_water_pct":drop_water, "radius_coarse_cells":2*face_coarse/np.pi,
        "pyfastnoiselite":importlib.metadata.version("pyfastnoiselite")}
    metadata["coarse_sampling"] = {"version":COARSE_VERSION, "tile_size":TILE_SIZE,
                                   "tile_stride":TILE_STRIDE, "blend":"completed tiles; shared weighted sums"}
    metadata["detail_sampling"] = {"version":DETAIL_VERSION,"latent_size":64,"latent_stride":32,
                                   "decoder_size":DECODER_SIZE,"decoder_stride":DECODER_STRIDE,
                                   "halos":"nearest shared nodes; decoder repeats latent nodes 8x"}
    if region is not None:
        metadata["algorithm"] = "cubed-sphere-regional-v1"
        metadata["detail_noise"] = "cube-tile-seeded-v1"
    if draft is not None:
        metadata["draft"] = draft.metadata
    if conditioning is not None:
        metadata["conditioning_tiffs"] = conditioning.metadata
    if source is not None:
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
        if conditioning is not None:
            conditioning.snapshot(directory)

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
    coarse = None
    if latent is None:
        coarse = restore("coarse",6,nc)
        if coarse is None:
            options = dict(frequency_mult=frequency, drop_water_pct=drop_water)
            raw_guide = (cube.conditioning(seed,nc,**options) if source is None
                         else source.conditioning(seed,nc,**options))
            coarse = sample_coarse(backend,seed,raw_guide,coarse_steps,progress,record)
            coarse = record("coarse",cube.identify(coarse),save=True)
        state = sample_latents(backend,coarse,seed,progress,record,latent_batch_size)
        latent = record("latent",state,save=True)

    climate = None
    if with_climate:
        from .climate import from_coarse
        if coarse is None:
            coarse = restore("coarse",6,nc)
        if coarse is None:
            raise ValueError("Climate reconstruction requires the coarse checkpoint; use a new checkpoint directory")
        progress("Reconstructing elevation-adjusted climate features")
        climate = record("climate-features",from_coarse(coarse))

    if region is not None:
        from .region import generate_region
        guide_seconds = time.perf_counter()-guides_started
        progress(f"Regional detail: global guides ready in {guide_seconds:.1f}s; decoding requested cube patches and halos")
        result, metadata = generate_region(backend,latent,seed,bounds,width,height,metadata,directory,progress,
                                          decoder_cache_bytes=decoder_cache_bytes)
        metadata['regional_execution']['guide_seconds'] = guide_seconds
        return (result, metadata, climate) if with_climate else (result, metadata)

    residual = restore("residual",1,nd)
    if residual is None:
        t = float(np.arctan(160))
        xt = record("decoder-noise",cube.noise(seed,6819,1,nd)*np.sin(t))
        def decode(a,f,y,x):
            cond = decoder_conditioning(latent,f,y,x)
            return backend.predict("decoder",a,cond,t)
        if directory is not None:
            identity = {"generation":metadata,"latent_sha256":hashlib.sha256(latent.tobytes()).hexdigest()}
            decode = PredictionCache(directory/"decoder",identity,decode)
        pred = cube.consensus(xt,decode,size=DECODER_SIZE,stride=DECODER_STRIDE,
                              weight_window=cube.linear_weight_window(DECODER_SIZE),shared_weights=True,include_endpoint=True,
                              progress=lambda d,n:progress(f"cube decoder: {d}/{n} patches"))
        residual = record("residual",cube.identify(np.cos(t)*xt+np.sin(t)*pred),save=True)
    progress("cube low-frequency reconstruction")
    z = cube.reconstruct(residual*backend.residual_std+backend.residual_mean,latent[4:5]*38.6-31.4)
    elevation = record("cube-elevation",np.sign(z)*z*z,save=True)
    progress("equirectangular export projection")
    projected = cube.to_equirectangular(elevation,nd*2)[0]
    return (projected, metadata, climate) if with_climate else (projected, metadata)
