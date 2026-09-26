"""Resolve random seeds before generation, preserving them for interrupted runs."""
import json
from pathlib import Path
import secrets


def resolve_seed(value="random", checkpoint_dir=None):
    if value is None or str(value).strip().lower() in ("", "random"):
        manifest = Path(checkpoint_dir)/"identity.json" if checkpoint_dir else None
        if manifest is not None and manifest.exists():
            seed = json.loads(manifest.read_text())["seed"]
        else:
            seed = secrets.randbits(64)
    else:
        try:
            seed = int(str(value).strip(),10)
        except ValueError as exc:
            raise ValueError("Seed must be an integer or 'random'") from exc
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    return seed
