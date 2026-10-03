"""Pinned upstream checkpoints and matching regional grid densities."""
import math
from pathlib import Path

MODEL_90M = "xandergos/terrain-diffusion-90m"
MODEL_30M = "xandergos/terrain-diffusion-30m"
REVISION_30M = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"


def is_30m_model(model):
    return model == MODEL_30M or Path(model).name == "terrain-diffusion-30m"


def native_metres(model):
    """Training metres per native pixel; a coarse cell is 256 native pixels."""
    return 30. if is_30m_model(model) else 90.


def parse_coarse_pooling(value):
    """Return a positive integer factor, or "auto"."""
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        factor = int(text)
    except ValueError:
        factor = 0
    if factor < 1:
        raise ValueError("Coarse pooling must be a positive integer or auto")
    return factor


def model_guide_height(radius_metres, metres_per_pixel):
    """Guide height whose nominal coarse cell equals the model's coarse cell."""
    return math.pi*radius_metres/(256*metres_per_pixel)


def auto_coarse_pooling(radius_metres, coarse_height, metres_per_pixel):
    """Pooling factor that runs the coarse model nearest its training spacing."""
    return max(1, round(model_guide_height(radius_metres, metres_per_pixel)/coarse_height))


def sparse_guide_height(model):
    # 2560 gives ~30.5 m equatorial native spacing at the default Earth radius.
    return 2560 if is_30m_model(model) else 1024


def default_revision(model):
    if is_30m_model(model):
        return REVISION_30M
    from .backends import MODEL_REVISION
    return MODEL_REVISION


def local_checkpoint(root, model):
    """Use a local download only after every stage is fully present."""
    path = Path(root) / Path(model).name
    required = [path / "config.json"]
    for stage in ("coarse_model", "base_model", "decoder_model"):
        required.extend((path / stage / "config.json",
                         path / stage / "diffusion_pytorch_model.safetensors"))
    return path if all(file.is_file() for file in required) else None
