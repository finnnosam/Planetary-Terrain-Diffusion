"""Continent-style guide degradation with genuine surrounding coarse cells."""
import numpy as np
import torch
import torch.nn.functional as F

PROFILE = "continent-smooth-v1"
HALO = 5
RECIPE = {"profile": PROFILE, "extra_coarse_halo": HALO, "clean_fraction": .25,
          "sigma_min_cells": .35, "sigma_max_cells": 1.5,
          "target": "original encoded terrain residual over original bilinear guide",
          "filter": "separable Gaussian in physical metres; radius 5; real context"}
MESO_PROFILE = "meso-smooth-v1"
MESO_RECIPE = dict(RECIPE, profile=MESO_PROFILE, extra_coarse_halo=3,
                   sigma_min_cells=.25, sigma_max_cells=1.,
                   filter="separable Gaussian in physical metres; radius 3; real context")
STRUCTURE_PROFILE = "continent-structure-v1"
STRUCTURE_RECIPE = dict(RECIPE, profile=STRUCTURE_PROFILE,
                        target="unchanged modern Earth encoded elevations; sea level stays zero")


def recipe_for(profile):
    return {PROFILE: RECIPE, MESO_PROFILE: MESO_RECIPE,
            STRUCTURE_PROFILE: STRUCTURE_RECIPE}[profile]


def decode_tensor(z):
    return z.sign() * z.square()


def degrade_guides(context, cond, generator, sigma_cells=None, clean_fraction=None, recipe=RECIPE):
    """Filter each guide differently without reflecting edges into its crop.

    Original central averages are retained exactly for clean examples. Smoothing
    occurs in metres, before encoding. The real five-cell halo supplies every
    filter tap used by the final condition, including its one-cell model margin.
    """
    halo = recipe["extra_coarse_halo"]
    if context.shape[-2:] != (cond.shape[-2]+2*halo, cond.shape[-1]+2*halo):
        raise ValueError("Guide context must include five extra coarse cells for the bridge, or three for meso")
    count = len(cond)
    device = cond.device
    if sigma_cells is None:
        sigma = torch.rand(count, generator=generator, device=device)
        sigma = recipe["sigma_min_cells"] + sigma*(recipe["sigma_max_cells"]-recipe["sigma_min_cells"])
    else:
        if not 0 < sigma_cells <= recipe["sigma_max_cells"]:
            raise ValueError("Evaluation sigma must be within the trained degradation range")
        sigma = torch.full((count,), sigma_cells, device=device)
    fraction = recipe["clean_fraction"] if clean_fraction is None else clean_fraction
    clean = torch.rand(count, generator=generator, device=device) < fraction
    offsets = torch.arange(-halo, halo+1, device=device)
    kernel = torch.exp(-.5*(offsets[None]/sigma[:, None]).square())
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    # Batch elements become separate convolution groups; both passes are valid.
    metres = decode_tensor(context).transpose(0, 1)
    filtered = F.conv2d(metres, kernel[:, None, None, :], groups=count)
    filtered = F.conv2d(filtered, kernel[:, None, :, None], groups=count).transpose(0, 1)
    encoded = filtered.sign() * filtered.abs().sqrt()
    return torch.where(clean[:, None, None, None], cond, encoded)


def augment_triplet(target, cond, context):
    k = int(torch.randint(4, ()))
    values = [torch.rot90(value, k, (-2, -1)) for value in (target, cond, context)]
    if torch.rand(()) < .5:
        values = [value.flip(-1) for value in values]
    return values


def paired_degradation(norm, target, cond, context, generator=None, **options):
    """Pair a degraded guide with the target defined by its saved recipe.

    In encoded elevation, the target residual remains exactly target-original_up.
    Only its broad baseline changes. This avoids asking the model to reverse
    the broad smoothing and overwrite the intended continent layout.
    The structure profile instead keeps Earth elevations unchanged so the
    intermediate model learns the missing belt/valley shape as well as detail.
    """
    guide = degrade_guides(context, cond, generator, **options)
    if options.get("recipe", RECIPE)["profile"] == STRUCTURE_PROFILE:
        # Learn the missing belt/valley geometry, including its broad correction.
        # Do not translate elevations, rescale relief, or adjust the zero datum.
        return target, guide
    original_up, _ = norm.condition(cond)
    new_up, _ = norm.condition(guide)
    return target+new_up-original_up, guide


def recovered_frames(folder, allow_partial=False):
    """Replay the original preparation RNG and identify saved patch orientations.

    Existing preparation stored centres but not orientations. Each candidate
    consumed random_frames(count=1), then one acceptance coin. Replaying those
    draws recovers exactly the frame for every saved centre, without rebuilding
    any fine-resolution terrain. Float32 keys match the stored centre dtype.
    """
    import json
    from pathlib import Path
    from toy_meso.sphere import random_frames, to_latlon
    folder = Path(folder)
    meta = json.loads((folder/"manifest.json").read_text())
    progress = json.loads((folder/"progress.json").read_text())
    if progress["done"] != {s: meta[f"{s}_count"] for s in ("train", "val")}:
        raise ValueError("Original bridge preparation must be complete")
    lookup, frames = {}, {s: [None]*meta[f"{s}_count"] for s in ("train", "val")}
    for split in frames:
        centres = np.load(folder/f"{split}_centre.npy")
        for index, centre in enumerate(centres):
            key = np.asarray(centre, dtype=np.float32).tobytes()
            if key in lookup:
                raise ValueError("Duplicate saved centres cannot identify patch orientations")
            lookup[key] = split, index
    rng = np.random.default_rng(meta["seed"])
    for _ in range(progress["tried"]):
        draw = random_frames(rng, 1)
        frame = tuple(a[0] for a in draw)
        rng.random()
        key = np.asarray(to_latlon(frame[0]), dtype=np.float32).tobytes()
        if key in lookup:
            split, index = lookup.pop(key)
            frames[split][index] = frame
            if not lookup:
                return frames
    if allow_partial:
        return frames
    raise ValueError(f"Could not recover {len(lookup)} patch frames; preparation RNG protocol changed")


def frame_matches_target(raster, frame, target, size, sub=6):
    """Verify orientation against exact fine-cell quadrature, not centre alone.

    Some older preparation sessions used a different RNG draw order. A centre
    can coincide while its rotation differs, so verify five noncentral cells
    against stored float16 targets before reusing a reconstructed frame.
    """
    from .data import FACTOR, SPACING_M
    from toy_meso.sphere import EARTH_RADIUS_M, to_latlon, encode
    ys = np.array([size//8, size//2, size*7//8, size//8, size*7//8])
    xs = np.array([size//8, size//2, size*7//8, size*7//8, size//8])
    full = size+2*FACTOR
    offsets = (np.arange(sub)+.5)/sub-.5
    u = (xs[:, None, None]+FACTOR-(full-1)/2+offsets[None, None, :])*SPACING_M/EARTH_RADIUS_M
    v = -(ys[:, None, None]+FACTOR-(full-1)/2+offsets[None, :, None])*SPACING_M/EARTH_RADIUS_M
    points = frame[0]+u[..., None]*frame[1]+v[..., None]*frame[2]
    points /= np.linalg.norm(points, axis=-1, keepdims=True)
    lat, lon = to_latlon(points)
    weights = np.maximum((points*frame[0]).sum(axis=-1), 0)**3
    metres = (raster.sample(lat, lon)*weights).sum(axis=(-1, -2))/weights.sum(axis=(-1, -2))
    return np.array_equal(encode(metres).astype(np.float16), target[ys, xs])


def coarse_context(raster, frame, size, sub=12):
    """Jacobian-weighted coarse means, centred on the original patch frame."""
    from .data import FACTOR, SPACING_M
    from toy_meso.sphere import patch_points, to_latlon, encode
    side = size//FACTOR + 2 + 2*HALO
    points = patch_points(frame, side, SPACING_M*FACTOR, supersample=sub)
    lat, lon = to_latlon(points)
    weights = np.maximum((points*frame[0]).sum(axis=-1), 0)**3
    values = raster.sample(lat, lon)
    means = (values*weights).reshape(side, sub, side, sub).sum(axis=(1, 3))
    means /= weights.reshape(side, sub, side, sub).sum(axis=(1, 3))
    return encode(means).astype(np.float32)
