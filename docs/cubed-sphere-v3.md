# Polar geometry correction (v3)

The v2 heightmap was continuous, but its terrain was stretched in a polar view.
The cause was generation in latitude/longitude coordinates: the physical length
of an east/west model pixel approaches zero at a pole, while its north/south
length stays constant. Identifying a single pole value cannot correct this
metric distortion. This is a generation defect, not a QGIS projection problem.

## Smallest implemented geometry change

Keep the upstream pretrained coarse, latent and decoder networks and their
conditioning, scale factors and elevation conversion. Replace their shared
global array with six coupled equi-angular cube charts. A chart maps `(u,v)` to
the normalized 3-D vector `normal + tan(pi/4*u)*right + tan(pi/4*v)*down`, with
`u,v` in `[-1,1]`, so node indices are equally spaced in angle along each chart
axis. Both poles are at face centres. There is no polar singularity.

Neighbouring nodes are 0.71-1.00 times the nominal spacing (a quarter great
circle divided by the face intervals), and cell areas vary by sqrt(2). The
original gnomonic charts (`normal + u*right + v*down`) varied 0.60-1.27 times
nominal, and 5.2x in cell area, so landforms shrank toward face edges. The
local metric's singular-value ratio still reaches sqrt(3) at cube corners, where
three faces meet; any cube chart has this. Halo reads beyond a face continue
the chart angles, so a read along a chart axis follows the same great circle at
the same step into the neighbouring face. Raster imports weight subsamples by
the equi-angular solid-angle Jacobian (`cube.area_weight`).

`cube.py` handles chart coordinates, shared nodes, noise, context reads,
denoiser consensus, reconstruction and export projection. `cube_generate.py`
uses those operations throughout the pretrained workflow:

1. Sample conditioning in 3-D and one Gaussian draw per unique cube node.
   Duplicate face edges and corners reference the same draw, avoiding variance
   loss from averaging independent noise. There are `6*N*N+2` unique nodes.
2. At each coarse solver step, read model context across face boundaries through
   3-D directions, combine overlapping denoiser predictions and update a single
   shared state. Edge and corner aliases agree after every update.
3. Apply the same geometry to both latent consistency steps and the decoder.
   No per-face finished images are generated and subsequently blended. Only
   predictions within an ongoing diffusion stage are combined.
4. Restore low-frequency elevation with the upstream Laplacian reconstruction,
   including cross-face filtering and interpolation. Convert signed square-root
   heights to metres pointwise.
5. Project the completed scalar sphere to the saved 2:1 longitude/latitude node
   field, then export pixel-centred GeoTIFFs using the existing storage code.

Model noise/context reads use nearest shared nodes to preserve marginal Gaussian
variance. Smooth conditioning and elevation reconstruction use bilinear reads.
Nearest reads can correlate/duplicate samples across chart boundaries; planar
weights do not become spherical or rotation-equivariant through this adaptation.
Chart distortion is bounded, not eliminated. Exact edge agreement guarantees
continuous heights, not continuous derivatives or realistic geology. A 4k
Earth-sized map also samples much more coarsely than the model's training scale.

## Verification

Tests cover all face-edge/corner aliases, unit noise variance including halos,
analytic scalar interpolation across faces, bounded local metric distortion,
isotropic polar sampling, constant-preserving reconstruction, constraints at
every generation stage, independent same-seed diagnostic runs, checkpoint
resumption and GeoTIFF metadata. Tiny real upstream models exercise the adapter
and scheduler, including the five-dimensional cube state.

`verify` also compares directional elevation differences at equal great-circle
offsets on latitude rings. A ratio near one means similar east/west and
north/south variation at that scale; terrain may legitimately be anisotropic,
so this is a diagnostic rather than a universal pass/fail threshold.

For the old seed-42 4k v2 map the ratios were 5.153 at 85°N and 8.498 at 88°S,
despite passing pole-value continuity. Its measurements are retained in
`outputs/terrain-v2-seed42/polar-anisotropy.json`.

The replacement and its numerical evidence are written to
`outputs/terrain-v3-cube-seed42/` by `python -m scripts.verify_cube`.
Algorithm, source digest, model hashes and runtime are recorded in `planet.json`
and the TIFF tags. Repeatability is for the same configuration and runtime;
neither cross-device determinism nor identical terrain between v2 and v3 is
claimed. The pretrained example is a single generated world; independent full
pretrained v3 reruns are distinct from the diagnostic repeatability tests.

## Seed-42 pretrained result (25 September 2026)

The replacement `planet-native.tif` is 4096×2048, single-band float32 metres,
with a spherical geographic CRS of radius 6,371,000 m. All cells are finite;
the global bounds are (-180, -90, 180, 90), with unit scale and zero offset.
The date-line-crossing `dateline-detail.tif` agrees bitwise with the corresponding
two portions of the global raster.

| Latitude | v2 directional ratio | v3 directional ratio |
|---|---:|---:|
| 85°N | 5.153 | 0.907 |
| 85°S | 4.258 | 1.009 |
| 88°N | 4.986 | 1.062 |
| 88°S | 8.498 | 0.943 |

These are east/west divided by north/south RMS differences using ±0.087890625°
great-circle offsets. This controlled change uses the same seed, weights and
export size; the generated worlds differ because the sampling grid changed.
The figures support removal of systematic polar stretching at the tested scale,
not a claim that every terrain feature is isotropic.

Date-line identification error, both exact-pole longitude spreads and all cube
shared-node errors are zero. The date-line gradient RMS is 0.467 times ordinary
longitude edges. Across the twelve cube edges, transverse gradient RMS relative
to nearby face interiors ranges from 0.877 to 2.068. Thus there are no value
discontinuities, but elevated gradients at some chart boundaries remain a
limitation; this is not a proof of visually invisible chart transitions.

The suite passes 29 tests, including independent diagnostic generation from the
same seed and checkpoint resume. The added analytic cube-edge diagnostic also
passes its targeted tests. Full pretrained v3 generation was run once; a second
independent full pretrained v3 run has not been performed. See the machine-readable
`verification.json` beside the TIFF for all boundary, polar and format measurements.

## Underwater terrain audit

The upstream [paper, section 4](https://arxiv.org/html/2512.08309v1#S4) explicitly
includes ocean bathymetry from ETOPO, blurred and upsampled to match the finer
MERIT land DEM. Training favours crops containing land. This supports generating
ocean depths, but does not establish land-like fine detail on the seabed.
The pinned checkout corroborates this in
`terrain_diffusion/data/preprocessing/elevation_dataset.py`: missing high-resolution
DEM values are filled from a smoothed negative low-resolution depth field.

Our coarse, latent and decoder loops cover every cube face without land/ocean
selection. Reconstruction uses signed heights and the `sign(z)*z*z` conversion;
GeoTIFF export retains negative values. No ocean-skipping or sea-level clamp
needs removing. The existing seed-42 TIFF has 4,112,386 underwater pixels,
about 65.09% of sphere area after latitude weighting. Underwater pixel heights
range from -4,851.24 m to just below zero. There are no missing/nonfinite cells
or exactly zero cells. Underwater adjacent-longitude elevation differences have
6.43 m RMS, confirming spatial variation rather than a constant ocean fill.
That last number is an export-grid diagnostic, not a physical slope comparison
or a measure of geological realism. Measurements are recorded in
`outputs/terrain-v3-cube-seed42/bathymetry-audit.json`.

Following inspection, the generation algorithm and supplied TIFF remain
unchanged. The measured cube-edge gradients alone do not justify a larger
sampler change when no lines are visible; their bounded shared-node averaging
introduces no value overshoot, and no numerical instability was found in this
run. This is not a guarantee about every seed's visual quality.
