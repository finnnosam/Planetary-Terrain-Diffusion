"""Pinned upstream checkpoints and matching regional grid densities."""
from pathlib import Path

MODEL_90M = "xandergos/terrain-diffusion-90m"
MODEL_30M = "xandergos/terrain-diffusion-30m"
REVISION_30M = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"


def is_30m_model(model):
    return model == MODEL_30M or Path(model).name == "terrain-diffusion-30m"


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
