# Regional cube generation

## Source review

The pinned source's `terrain_diffusion/inference/world_pipeline.py` uses
`InfiniteTensor` and overlapping `TensorWindow` dependencies. `get(i1,j1,i2,j2)`
pulls only the requested planar output. `_compute_elev` expands that request
for low-frequency reconstruction (sigma 5), then crops it. Noise is seeded
by fixed tile coordinates; cached decoder predictions can serve later queries.
`inference/tiff_export.py` chunks requests rather than first generating a world.

Using this planar API directly would disconnect cube faces and mishandle poles.
The regional implementation instead retains the existing global coarse and
latent solver, and makes decoder/reconstruction fields lazy on the cube.

## Spherical dependencies

`region.Field` resolves halo coordinates through 3-D cube directions. At shared
edges and corners, all chart predictions contribute to one physical node.
Gaussian noise takes one chart owner's draw rather than averaging draws.
Interpolation, triangular reduction and Gaussian smoothing use the same
operations as `cube.reconstruct`, evaluated through bounded tile caches.
The decoder retains 512-pixel patches, 256-pixel strides and overlap weights.

The regional mode has its own version and checkpoint identity because its
tile-seeded detail noise differs from the existing whole-globe noise stream.
Bounds and output dimensions do not affect the underlying field or cache
identity. Code, model, seed, guide, native density and runtime still do.

Requested pixel centres are projected directly onto the cube, including across
the date line and near either pole. Signed squaring occurs before interpolation,
as in the existing cube elevation field. The output state is a regional pixel
grid, not the legacy global node grid. Its GeoTIFF affine uses the requested
bounds with unwrapped east longitude when necessary.

## Limits

Coarse/latent generation still has global cost; regional generation saves native
decoder work and native-grid memory, not all world-scale work. Very fine detail
on a small geographic extent still requires a larger global guide density.
The radius changes physical spacing, not angular bounds. Regional states can
re-export their saved grid; different coverage requires another generation query.
Low-resolution outputs evaluate pixel centres and are not area averages.

Tests compare lazy reconstruction against dense spherical reconstruction, check
shared noise, patch savings, cache resume, bounds validation and regional storage.
The decoder oracle also covers both poles, the date line, cube boundaries and
aligned overlapping requests. A pretrained CUDA smoke run on an RTX 3070
generated bounds (170, -4, 190, 4), 32×16 pixels at coarse height 4, with 20
decoder patches versus 54 for a whole globe. The TIFF passed finite-height and
digest checks and retained the radius-6371000 spherical CRS and metre units.

## Broad-region performance regression

The supplied coarse-height-32, 4096×2048 northwest quadrant run took 1304.6 s,
versus 728.2 s for the earlier globe. Both spent about 179 s generating the
global coarse/latent guides. The regional log contained 5441 patch-access
messages for only 544 distinct patches (the globe used 1734).

Those repeated messages were memory-cache misses, not extra GPU evaluations:
the saved prediction cache prevented duplicate inference. However, the old
lookup order regenerated a 512×512 spherical noise context before checking
disk. Small, fixed-entry reconstruction caches also discarded nearby work as
latitude scans crossed the cube faces. Single-tap filter reads repeatedly
sorted and gathered the same nodes.

The corrected path checks saved predictions before constructing noise, budgets
each field cache at 16 MiB, gathers rectangular tiles directly, and reads one
halo per separable filter. Progress counts only newly encountered patches and
reports them in batches of 16. `planet.json` records guide time, regional time,
decoder evaluation time/count and prediction disk reads under
`regional_execution` so future timings can distinguish computation from reuse.

Replaying all 4096×2048 pixels from the supplied saved decoder predictions took
38.34 s and matched the supplied regional elevation array bit for bit. This is
a cached reconstruction measurement, not a fresh generation time. Original
examples were read without modification. A strip profile improved from 18.83 s
to 3.65 s before the larger reconstruction caches were enabled.

A fresh end-to-end CUDA run on the RTX 3070, including model loading, both
global latent passes, all decoder inference, state saving and TIFF export,
completed in **424.97 s** with the same seed, draft, refinement 0.01, white
elevation 6670 m, coarse height 32, bounds (-180, 0, 0, 90), and 4096×2048
output. Its saved elevation array was **bit-identical** to the supplied region.

| Run | End-to-end seconds |
| --- | ---: |
| Supplied whole globe | 728.2 |
| Supplied regional implementation | 1304.6 |
| Corrected fresh regional implementation | 424.97 |

The new run used 544 decoder evaluations, 2570 inexpensive saved-prediction
reloads, and 35 distinct decoder progress messages (zero repeated counts).
Guides took 174.27 s. Regional detail/reconstruction/projection took 240.29 s,
including 183.23 s inside decoder calls. The GPU's convolution workspace
allocation retries occurred in both supplied runs and in the new run and are
included in those timings. The remaining fixed global guide cost prevents
end-to-end time from scaling directly with area.

Local evidence is saved under `outputs/regional-performance/`: `comparison.json`
contains the comparison, `cuda-full/generation.log` the fresh run log, and
`cuda-full/planet-native.tif` the verified result. The TIFF retains the requested
bounds, 4096×2048 shape, metre elevations, and spherical CRS.

Regression tests exercise saved-prediction input avoidance, repeated latitude
scans exceeding the old 128-entry limit, non-power-of-two face dimensions,
spherical reconstruction agreement, and resume timing/counters. Checkpoint
identities still include source hashes; use a new run after code changes.
