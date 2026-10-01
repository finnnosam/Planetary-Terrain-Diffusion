"""Atomic, identity-checked cache of intermediate denoiser predictions."""
import json
import os
from collections import OrderedDict
from numbers import Integral
from pathlib import Path
import numpy as np


class ArrayCache:
    """LRU bounded by retained NumPy payload bytes; zero disables retention."""
    def __init__(self, max_bytes):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral) or max_bytes < 0:
            raise ValueError("Cache byte budget must be a nonnegative integer")
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous.nbytes
        if value.nbytes > self.max_bytes or self.max_bytes == 0:
            return
        # Views can retain an arbitrarily larger parent allocation.
        if not value.flags.owndata:
            value = value.copy()
        self.entries[key] = value
        self.bytes += value.nbytes
        while self.bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes -= evicted.nbytes


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
        self.validated = OrderedDict()
        self.validation_scans = 0
        self.hits = 0
        self.misses = 0

    def __call__(self, a, *location):
        return self.get_or_compute(lambda: a, (1, a.shape[-2], a.shape[-1]), *location)

    def get_or_compute(self, make_input, shape, *location):
        """Check disk before constructing expensive deterministic model inputs."""
        result = self.load(shape, *location)
        if result is None:
            result = self.save(self.predict(make_input(), *location), shape, *location)
        return result

    def load(self, shape, *location):
        path = self.directory/("_".join(str(i) for i in location)+".npy")
        try:
            handle = path.open('rb')
        except FileNotFoundError:
            return None
        # One open serves existence, identity/stat checks, and the array read.
        # Stat the opened file so an atomic replacement cannot mix signatures.
        with handle:
            stat = os.fstat(handle.fileno())
            result = np.load(handle, allow_pickle=False)
        signature = (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino)
        # Release handles immediately: retained arrays must not prevent atomic
        # patch replacement on Windows.
        if result.shape != shape or result.dtype != np.float32:
            raise ValueError(f"Invalid decoder prediction at {path}")
        if self.validated.get(path) != signature:
            self.validation_scans += 1
            if not np.isfinite(result).all():
                raise ValueError(f"Invalid decoder prediction at {path}")
            self.validated[path] = signature
        self.validated.move_to_end(path)
        if len(self.validated) > 4096:
            self.validated.popitem(last=False)
        self.hits += 1
        return result

    def save(self, result, shape, *location):
        path = self.directory/("_".join(str(i) for i in location)+".npy")
        # Own each patch so an in-memory entry cannot retain an entire batch.
        result = np.array(result, dtype=np.float32, copy=True)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"Invalid decoder prediction at {path}")
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, result, allow_pickle=False)
        temporary.replace(path)
        self.misses += 1
        return result
