"""Request additional spherical regions from a completed cube generation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import procedural
from .region import generate_region, validate_region
from .storage import load_state, load_climate


def checkpoint_directory(state, explicit=None):
    """Find the CLI or desktop launcher's checkpoint directory."""
    if explicit is not None:
        return Path(explicit)
    state = Path(state)
    candidates = [Path(str(state) + ".checkpoints"), state.parent / "checkpoints"]
    found = [path for path in candidates if (path / "identity.json").is_file()]
    if len(found) != 1:
        raise ValueError("Specify --checkpoint-dir; expected one matching checkpoint directory beside the state")
    return found[0]


def _current_source_hash(identity, with_climate):
    """Reproduce the generation source fingerprint for this checkpoint's mode."""
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("cube.py", "cube_generate.py", "backends.py", "procedural.py", "coarse.py", "detail.py"):
        digest.update((root / name).read_bytes())
    for name in ("synthetic_map_stats.json", "elevation_reference.npz"):
        digest.update((procedural.DATA / name).read_bytes())
    if "draft" in identity:
        digest.update((root / "draft.py").read_bytes())
    if "draft" in identity or "conditioning_tiffs" in identity:
        digest.update((root / "spherical_raster.py").read_bytes())
    if "conditioning_tiffs" in identity:
        digest.update((root / "conditioning.py").read_bytes())
    if identity["algorithm"] == "cubed-sphere-regional-v1":
        digest.update((root / "region.py").read_bytes())
    if with_climate:
        digest.update((root / "climate.py").read_bytes())
    return digest.hexdigest()


def load_query(state, checkpoint_dir=None, with_climate=False):
    """Load and validate the saved guide and its generation identity."""
    _, saved = load_state(state)
    directory = checkpoint_directory(state, checkpoint_dir)
    manifest = directory / "identity.json"
    if not manifest.is_file():
        raise ValueError(f"No cube checkpoint identity at {manifest}")
    identity = json.loads(manifest.read_text(encoding="utf-8"))
    if identity.get("algorithm") not in ("cubed-sphere-v3", "cubed-sphere-regional-v1", "sparse-cube-v1"):
        raise ValueError("Query requires cube generation checkpoints")
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("Saved state and cube checkpoints have different generation identities")
    if identity["algorithm"] == "sparse-cube-v1":
        return directory, identity, saved, None, None
    if _current_source_hash(identity, "climate" in saved) != identity.get("source_sha256"):
        raise ValueError("Generation source has changed since these checkpoints were created")
    n = identity["face_coarse_intervals"] * 32
    path = directory / "latent.npy"
    if not path.is_file():
        raise ValueError(f"No completed latent checkpoint at {path}")
    latent = np.load(path, mmap_mode="r", allow_pickle=False)
    if latent.shape != (5, 6, n + 1, n + 1) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError(f"Invalid latent checkpoint at {path}")
    climate = load_climate(state, saved) if with_climate else None
    return directory, identity, saved, latent, climate


def query_region(backend, state, bounds, width, height, checkpoint_dir=None,
                 with_climate=False, progress=print):
    """Decode requested bounds using saved global guides and persistent tile cache."""
    directory, identity, saved, latent, climate = load_query(state, checkpoint_dir, with_climate)
    if any(backend.metadata.get(key) != value for key, value in identity.items()
           if key in backend.metadata):
        raise ValueError("Backend does not match the saved model and runtime identity")
    bounds = validate_region(bounds, width, height, saved["native_height"])
    if identity["algorithm"] == "sparse-cube-v1":
        from .sparse import SparseWorld
        source = None
        if "draft" in identity:
            from .draft import Draft
            info = identity["draft"]
            source = Draft(directory/"draft.png",info["ocean_depth_hint_metres"],
                           info["white_metres"],info["refinement"])
        elif "conditioning_tiffs" in identity:
            from .conditioning import TiffConditioning
            source = TiffConditioning(directory/"conditioning",
                                      identity["conditioning_tiffs"]["snr"])
        world = SparseWorld(backend,saved["seed"],saved["coarse_height"],
                            saved["coarse_steps"],saved["radius_metres"],directory,
                            source=source,identity=identity)
        return world.region(bounds,width,height,with_climate=with_climate,progress=progress)
    metadata = dict(identity)
    metadata["algorithm"] = "cubed-sphere-regional-v1"
    metadata["detail_noise"] = "cube-tile-seeded-v1"
    if identity["algorithm"] == "cubed-sphere-v3":
        # Whole-globe identities predate the regional decoder and do not hash it.
        metadata["regional_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("region.py").read_bytes()).hexdigest()
    elevation, metadata = generate_region(backend, latent, saved["seed"], bounds,
                                          width, height, metadata, directory, progress)
    metadata["radius_metres"] = saved["radius_metres"]
    return elevation, metadata, climate
