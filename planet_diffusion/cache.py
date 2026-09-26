"""Atomic, identity-checked cache of intermediate denoiser predictions."""
import json
from pathlib import Path
import numpy as np


class PredictionCache:
    def __init__(self, directory, identity, predict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory/"identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Decoder cache belongs to different inputs, weights, code or runtime")
        else:
            if list(self.directory.glob("*.npy")):
                raise ValueError("Decoder cache has predictions but no identity manifest")
            temporary = self.directory/"identity.json.tmp"
            temporary.write_text(json.dumps(identity, indent=2)+"\n")
            temporary.replace(manifest)
        self.predict = predict
        self.hits = 0
        self.misses = 0

    def __call__(self, a, *location):
        return self.get_or_compute(lambda: a, (1, a.shape[-2], a.shape[-1]), *location)

    def get_or_compute(self, make_input, shape, *location):
        """Check disk before constructing expensive deterministic model inputs."""
        path = self.directory/("_".join(str(i) for i in location)+".npy")
        if path.exists():
            result = np.load(path, allow_pickle=False)
            self.hits += 1
        else:
            result = np.asarray(self.predict(make_input(), *location), dtype=np.float32)
            self.misses += 1
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, result, allow_pickle=False)
            temporary.replace(path)
        if result.shape != shape or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError(f"Invalid decoder prediction at {path}")
        return result
