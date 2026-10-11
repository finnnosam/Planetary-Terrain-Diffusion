"""Save a reviewable input layout and bind approval to its settings and images."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .conditioning import make_source, file_sha
from .sphere import decode
from planet_diffusion import cube

REVIEW_EXIT_CODE = 42
INPUT_PREVIEW = "input.preview.png"
RANDOM_PREVIEW = "random.preview.png"
REVIEW_FILE = "input-review.json"


def write_review(path, review):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(review, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_review(folder, settings, grid, coarse_enc, lat, lon, condition_sampler, approve=False):
    """Preview the cached model input; changed settings or images need new approval."""
    folder = Path(folder)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for axis in (lat, lon):
        digest.update(np.asarray(axis, dtype=np.float64).tobytes())
    signature = digest.hexdigest()
    manifest = folder / REVIEW_FILE
    review = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    expected_images = {INPUT_PREVIEW} | ({RANDOM_PREVIEW} if settings["input_source"] != "map" else set())
    reusable = (isinstance(review, dict) and review.get("version") == "meso-input-review-v1"
                and isinstance(review.get("approved"), bool) and review.get("signature") == signature
                and isinstance(review.get("images"), dict) and set(review["images"]) == expected_images
                and all((folder / name).is_file() and file_sha(folder / name) == sha
                        for name, sha in review["images"].items()))
    if approve and not reusable:
        raise ValueError("The input preview changed or is missing. Generate its preview again before approving it.")
    if not reusable:
        print("writing input layout previews (before model detail)", flush=True)
        pending = folder / ".incomplete"
        pending.mkdir(exist_ok=True)
        # Take a subset of the requested pixels, so even a sparse/low-resolution
        # regional output only reads coarse cells prepared for its actual tiles.
        ys = lat[::max(1, int(np.ceil(len(lat) / 512)))]
        xs = lon[::max(1, int(np.ceil(len(lon) / 1024)))]
        before = []
        for start in range(0, len(ys), 64):
            la, lo = np.radians(ys[start:start + 64])[:, None], np.radians(xs)[None, :]
            points = np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo),
                                                  np.cos(la) * np.sin(lo), np.sin(la)), -1)
            face, y, x = cube.coordinates(points, grid.N)
            values = np.empty(face.shape, np.float32)
            for f in np.unique(face):
                selected = face == f
                values[selected] = decode(condition_sampler(coarse_enc, grid.factor, grid,
                                                              int(f), y[selected], x[selected]))
            before.append(values)
        images = {INPUT_PREVIEW: np.concatenate(before)}
        if settings["input_source"] != "map":
            random = make_source("random", settings["seed"],
                                 random_version=settings["random_source_version"])
            random_lat = 90 - (np.arange(512) + .5) * 180 / 512
            random_lon = -180 + (np.arange(1024) + .5) * 360 / 1024
            images[RANDOM_PREVIEW] = random.sample(random_lat[:, None], random_lon[None, :])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        hashes = {}
        for name, values in images.items():
            # Use the same fixed elevation colours for input and random images.
            temporary = pending / name
            plt.imsave(temporary, plt.cm.terrain(Normalize(-6000, 6000)(values)))
            temporary.replace(folder / name)
            hashes[name] = file_sha(folder / name)
        review = dict(version="meso-input-review-v1", signature=signature, approved=False,
                      images=hashes, settings=settings, latitude_bounds=[float(lat[-1]), float(lat[0])],
                      longitude_bounds=[float(lon[0]), float(lon[-1])],
                      description="Input layout before the meso model adds finer terrain")
        write_review(manifest, review)
    if approve:
        review["approved"] = True
        write_review(manifest, review)
        print("input layout approved", flush=True)
    return review
