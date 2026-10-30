# Verification results

**Historical v1 boundary smoke test.** The user subsequently identified blobs
and latitude bands in this output. Those tests missed terrain-quality defects.
See [the v2 correction and controlled comparison](terrain-quality-v2.md).

Executed locally on Windows with Python 3.12.14, CPU PyTorch 2.14.0, four torch
threads, NumPy 2.5.3, Diffusers 0.40.0 and Rasterio 1.5.1.
Tested dependency versions are listed in `requirements-tested.txt`.

`python -m pytest -q`: **14 passed**, no skipped tests. Five upstream
Rasterio/Affine pending-deprecation warnings; no test failures.

Two complete pretrained runs used seed 42, coarse-height 2, 20 coarse steps,
two latent steps and one decoder step. Total generation/export/verification
time for both runs was approximately 107 seconds, excluding model loading and
download. Model source: `xandergos/terrain-diffusion-90m` at
`2bb1a93141140040091d44f0770d4ba22c8cf145`; local snapshot content digest is
recorded in the JSON report.

| Check | Result |
|---|---|
| Same seed, every recorded stage | Bitwise identical across both runs |
| Exact date-line identification | 0 m error |
| Two-sided date-line gap at ±1e-6 node spacing | 0.00000906 m |
| Longitude variation at each exact pole | 0 m |
| Pole constraint at every recorded stage | Passed |
| Polar spread when distance reduces 100× | Reduces approximately 100× |
| Seam-edge RMS / ordinary longitude-edge RMS | 0.779; no statistical outlier |
| Native global GeoTIFF | 1024×512, float32, metres, spherical CRS |
| Global bounds | −180°, −90°, 180°, 90° |
| Date-line tile versus native global pixels | Exactly equal |
| Native elevation range | 107.018 to 271.324 m |

Artifacts in `outputs/learned-smoke/`:

- `planet-native.tif`: native 1024×512 global elevation map.
- `planet.tif`: 256×128 global overview.
- `dateline-detail.tif`: 64×64 tile with four times the overview's sampling density,
  crossing the date line with continuous affine coordinates.
- `state/`: native node field and metadata, sufficient for additional exports.
- `verification.json`: numerical results and hashes of all generation stages.

These are real pretrained-model outputs, not the diagnostic backend. The tiny
coarse grid intentionally makes this a smoke test. This seed produces entirely
positive elevations over a narrow range; it is not evidence of realistic
continental geography. Topological continuity and reproducibility passed;
geographic realism, rotation invariance and Earth-scale performance remain
unvalidated. No raster was rendered or colour mapped.
