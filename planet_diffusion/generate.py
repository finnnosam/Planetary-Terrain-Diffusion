"""Global-state coarse EDM and two-step latent / one-step decoder sampling."""
import numpy as np
from .topology import conditioning, noise, constrain, consensus, patch, sample
from .reconstruction import reconstruct


def generate(backend, seed, coarse_height=8, coarse_steps=20, progress=print, audit=None, decoder_cache=None):
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if coarse_height < 2 or coarse_height % 2:
        raise ValueError("coarse-height must be an even integer >= 2")
    if coarse_steps < 2:
        raise ValueError("coarse-steps must be >= 2")

    def record(name, state):
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite values at {name}")
        if audit is not None:
            audit(name, state)
        return state

    h = coarse_height
    means, stds = backend.means, backend.stds
    channels = [0, 2, 3, 4, 5]
    guide = (conditioning(seed, h)-means[channels, None, None])/stds[channels, None, None]
    angles = np.arctan(backend.cond_snr)[:, None, None]
    guide = record("conditioning", (np.cos(angles)*guide + np.sin(angles)*noise(seed, 0, 5, h)).astype(np.float32))
    state = record("coarse-noise", noise(seed, 1, 6, h)*backend.start_coarse(coarse_steps))
    for step in range(coarse_steps):
        progress(f"coarse {step+1}/{coarse_steps}")
        pred = consensus(state, lambda a, y, x: backend.coarse_predict(a, patch(guide, y, x, 64), step))
        state = record(f"coarse-{step}", constrain(backend.coarse_advance(pred, state, step)))
    coarse = backend.coarse_finish(state)*stds[:, None, None] + means[:, None, None]
    coarse[1] = coarse[0] - coarse[1]
    coarse = record("coarse", constrain(coarse))

    h *= 32
    state = np.zeros((5, h+1, 2*h), np.float32)
    for step, t in enumerate([float(np.arctan(80/.5)), float(np.arctan(.35/.5))]):
        progress(f"latent {step+1}/2")
        xt = record(f"latent-noisy-{step}", (np.cos(t)*state + np.sin(t)*noise(seed, 5819+step, 5, h)).astype(np.float32))
        def latent_predict(a, y, x):
            cond = sample(coarse, (y/32-1+np.arange(4))[:, None], (x/32-1+np.arange(4))[None, :])
            return backend.predict("latent", a, cond, t)
        pred = consensus(xt, latent_predict,
                         progress=lambda done, total: progress(f"latent {step+1}/2: {done}/{total} patches"))
        state = record(f"latent-{step}", constrain(np.cos(t)*xt + np.sin(t)*pred))
    elevation = decode_elevation(backend, seed, state, progress=progress, audit=audit, cache_dir=decoder_cache)
    return elevation, metadata_for(backend, seed, coarse_height, coarse_steps)


def decode_elevation(backend, seed, latent, progress=print, audit=None, cache_dir=None):
    """Resume the decoder from a completed spherical latent state."""
    h = (latent.shape[-2]-1)*8
    if latent.shape != (5, h//8+1, h//4) or not np.isfinite(latent).all():
        raise ValueError("Expected a finite five-channel spherical latent state")
    def record(name, state):
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite values at {name}")
        if audit is not None:
            audit(name, state)
        return state
    progress("decoder 1/1")
    t = float(np.arctan(80/.5))
    xt = record("decoder-noise", noise(seed, 6819, 1, h)*np.sin(t))
    def decode(a, y, x):
        # Same nearest-neighbour latent conditioning as upstream, with spherical indexing.
        from .topology import gather
        cond = gather(latent[:4], np.floor_divide(np.arange(y, y+512), 8)[:, None],
                      np.floor_divide(np.arange(x, x+512), 8)[None, :])
        return backend.predict("decoder", a, cond, t)
    if cache_dir is not None:
        import hashlib
        from pathlib import Path
        from .cache import PredictionCache
        digest = hashlib.sha256()
        for name in ["generate.py", "topology.py", "backends.py", "reconstruction.py"]:
            digest.update(Path(__file__).with_name(name).read_bytes())
        identity = {"backend": backend.metadata, "seed": seed, "source_sha256": digest.hexdigest(),
                    "latent_sha256": hashlib.sha256(latent.tobytes()).hexdigest(),
                    "noise_sha256": hashlib.sha256(xt.tobytes()).hexdigest(),
                    "size": 512, "stride": 256}
        decode = PredictionCache(cache_dir, identity, decode)
    pred = consensus(xt, decode, size=512, stride=256,
                     progress=lambda done, total: progress(f"decoder: {done}/{total} patches"))
    residual = record("decoder", constrain(np.cos(t)*xt + np.sin(t)*pred))
    sqrt_metres = reconstruct(residual*backend.residual_std + backend.residual_mean,
                              latent[4:5]*38.6-31.4)[0]
    elevation = record("elevation-metres", (np.sign(sqrt_metres)*sqrt_metres**2)[None])[0]
    return elevation


def metadata_for(backend, seed, coarse_height, coarse_steps):
    return {**backend.metadata, "seed": seed, "coarse_height": coarse_height,
                "coarse_steps": coarse_steps, "native_height": coarse_height*256, "algorithm": "spherical-consensus-v2",
                "numpy": np.__version__, "units": "m", "vertical_datum": "model-defined zero sea level",
                "grid": "internal nodes with unique poles; export pixel centres",
                "polar_constraint": "denoised first cap rings only; unit Gaussian node noise",
                "reconstruction": "spherical Laplacian low-frequency denoise, sigma=5, kernel=11"}
