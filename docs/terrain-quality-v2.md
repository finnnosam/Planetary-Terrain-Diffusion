# Correcting the v1 blobs and latitude bands

Historical v2 results. These fixes restored terrain detail, but a subsequent
polar-view inspection exposed metric distortion in the generation grid.
[Version 3](cubed-sphere-v3.md) retains the reconstruction correction and moves
generation to a cubed sphere.

The user reported blobs separated by apparent latitude lines in the supplied
`outputs/learned-smoke/planet-native.tif`. This was a real generation defect,
not a QGIS display problem. The original continuity and repeatability tests did
not adequately test terrain quality.

## Causes found

1. **Missing low-frequency reconstruction.** The fifth latent channel was
   bilinearly enlarged directly, leaving bumps at the 8-pixel latent-grid knots.
   Upstream reconstructs a provisional field, reduces it with antialiasing,
   smooths the reconstructed low-frequency channel, then adds its learned
   residual. Omitting that step was incorrect.
2. **Distorted diffusion noise.** A latitude-dependent Fourier filter was applied
   to noise and intermediate states throughout the planet. Measured standard
   deviation was approximately 0.82 at the equator and 0.14 on the first polar
   ring, instead of the model's unit Gaussian innovations. Repeated filtering
   also altered learned latent features far from the poles.
3. **An inadequate demonstration grid.** The supplied file was generated from
   just two coarse latitude intervals and four longitude samples. That is useful
   for testing boundary code, but was a poor terrain demonstration.

## Changes

- Noise is now unit Gaussian at every distinct node, including a shared draw at
  each physical pole. It is not spectrally filtered.
- Denoised-state polar projection affects only the exact poles and their first
  adjacent rings; all other interior samples are unchanged.
- The upstream low-frequency reconstruction sequence is restored with spherical
  antialiasing and Gaussian halos. The high-frequency residual is retained.
  This is part of decoding, not a finished-image seam blend.
- Default coarse height is now 8 and default GeoTIFF export is native resolution.
- `verify` separately flags repeating latent-grid structure. A boundary `passed`
  result is explicitly not a terrain-quality verdict.
- The algorithm is versioned `spherical-consensus-v2`. Existing states are
  preserved; regenerating is required to obtain corrected terrain.

## Controlled comparison

Both complete runs used seed 42, coarse-height 2, identical pretrained weights
and the same CPU runtime. For each row phase modulo 8, measure east–west gradient
RMS in the broad interior latitude belt. The maximum/minimum ratio detects
whether detail systematically swells and fades between latent-grid rows.

| Measurement | Original v1 | Reconstruction fix alone | Complete v2 |
|---|---:|---:|---:|
| Row-phase gradient ratio | 1.403 | 1.055 | 1.036 |
| Column-phase gradient ratio | 1.345 | 1.006 | 1.015 |
| Native elevation range (m) | 107–271 | 143–190 | 16–706 |

The normalized decoder-residual standard deviation increased from 0.124 in v1
to 1.625 in v2. The former global filtering was suppressing the learned detail
channel, not merely altering the appearance of the low-frequency background.
This is a diagnostic measurement, not a target terrain-roughness calibration.

The reconstruction-only column is an ablation using captured model states to
isolate the missing operation; it is not the delivered replacement workflow.
Two full v2 runs matched bitwise in coarse state, latent state, decoder residual
and final elevation. Both pole limits and date-line checks passed. Unit tests
also verify latitude-independent noise variance, preservation of interior
features, retained high-frequency residuals and removal of latent-grid knots.
The suite has **22 passing tests**, including interrupted-decoder cache reuse,
identity mismatch rejection, and bitwise equality of resumed decoding.

Machine-readable evidence: `outputs/terrain-quality-comparison.json`. This
quantifies the reported artifact; it does not prove natural geology or establish
visual quality. No raster was rendered or colour mapped during these checks.

The larger regenerated example is in `outputs/terrain-v2-seed42/`, with an 8×16
coarse grid and 4096×2048 native export. Its `quality.json` records its own
boundary and grid-artifact diagnostics rather than reusing the small-run results.

The completed larger run passed its boundary checks, with a native-state
row-phase ratio of **1.0108** and column-phase ratio of **1.0076**. Measurements
on the exported GeoTIFF pixels are **1.0068** and **1.0072**, respectively. The
export contains finite float32 metre elevations from approximately **−3049 to
+1502 m**, exact global bounds, and a spherical CRS. A 64×128 tile crossing the
date line matches the native raster exactly. Replaying all cached decoder
predictions from the saved latent state reproduces the elevation state bitwise.
The latter is a cache/resume check, not a second independent large model run;
independent full-run repeatability was checked on the matched-resolution small
v2 runs above. Full results are in `outputs/terrain-v2-seed42/final-validation.json`.

## Resuming this comparison run

An interruption ended the first large decoder run after its coarse and latent
states were saved. Decoding can now resume from that captured seed-42 latent
state, saving each completed prediction atomically. Cache identity includes
the latent and noise hashes, model/runtime metadata and sampler source hash.
Cached predictions are still accumulated into the global generation state in
the same order; they are not stitched finished terrain tiles.

```powershell
.venv/Scripts/python -m scripts.compare_terrain --coarse-height 8 --output outputs/terrain-v2-seed42 --resume-latent
```

This is the seed-42 comparison script's resume option, not a generic migration
of arbitrary or v1 latent files. A completed output directory is refused.
