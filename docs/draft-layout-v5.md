# Preserving a drawn planet layout (version 0.5)

**Superseded by [version 0.6](native-guides-v6.md).** These historical mask
scores did not catch suppressed terrain detail or the coastal-depth artifact.
The hard guidance described below has been removed from current generation.

The supplied `human_input/blobby_Earth.png` is 8192×4096 with 12 brightness
levels. The reported run used seed 2644365341129234824, white elevation 6250 m,
ocean depth 0, and a 4096×2048 export. Its coarse model grid had just 98 unique
spherical nodes. The old importer averaged the draft down to those nodes, then
gave the model freedom to reinterpret the result. This was inadequate for
preserving the continents. Its all-nonnegative guide also treated oceans as
0 m ground: every learned coarse elevation became positive before detail began.

## Revised generation

Black is an ocean-coverage mask. A depth hint of zero selects automatic
bathymetry, starting the coarse model from a negative 2000 m depth prior; a
positive hint replaces that prior. The sea surface remains the zero reference.
The model supplies actual ocean depths rather than exporting a flat water plane.

The coarse result is constrained to the rough guide with bounded learned
variation in signed-square-root metres, retaining the model's mean-to-fifth-
percentile relief. In addition, the original draft is sampled directly at the
latent resolution (128 intervals per cube face for the default run). Thus the
layout does not have to survive a round trip through 98 coarse nodes.

The latent layout combines painted land elevations with interpolated learned
coarse bathymetry. Where a sea is too narrow for the coarse grid and the coarse
model predicts land, it uses the negative ocean prior. The low-frequency latent
channel is constrained after each consistency step. Its allowed variation is
20% of the target's signed-square-root magnitude, with a 2 sqrt(m) floor and a
smooth tanh bound. This limits geographic drift while leaving the four learned
detail channels free; those channels see the constrained low-frequency channel
in the next model step. The decoder runs on the resulting detail latents.

During Laplacian reconstruction, the known layout is removed before filtering
the free component and then restored at latent resolution before upsampling.
This prevents the reconstruction's Gaussian filter from erasing coastlines
again. It is part of the low-frequency reconstruction, not a blend of completed
maps. All constraints, context reads, interpolation and reconstruction use
shared-node cube topology. Alpha-weighted layout is zero in fully transparent
regions, preserving unconstrained generation there.

## Limits and checks

This mode intentionally keeps much more of a painted layout than the previous
weak conditioning. It is a constrained adaptation of the pretrained models,
not a retrained sketch-to-terrain model. It does not promise exact coastlines,
every small island, or identical physical elevations at each pixel. Strong
low-frequency guidance can retain some of the draft's broad plateaus; generated
local detail remains seed-dependent. Output pixel resolution is finer than the
guide resolution. Increasing it increases guide resolution as well.

Regression tests cover sub-coarse-grid islands, negative oceans despite an
all-land model proposal, layout-preserving reconstruction, transparent drafts,
same-seed repeatability, and the existing seam/pole/metadata checks.

The same-seed pretrained replacement is saved in `outputs/blobby-earth-v5/`.
`python -m scripts.verify_draft_layout --run outputs/blobby-earth-v5 --draft
human_input/blobby_Earth.png --white-metres 6250` measures land intersection over
union and agreement, validates the PNG hash against generation metadata, and
checks finite ocean depths and the spherical boundaries. These metrics measure
layout adherence separately from mathematical continuity and terrain realism.

The same-input replacement command is:

```powershell
.venv/Scripts/python -m planet_diffusion generate --draft human_input/blobby_Earth.png --draft-ocean-depth 0 --draft-white-metres 6250 --seed 2644365341129234824 --coarse-height 8 --model models/terrain-diffusion-90m --state outputs/blobby-earth-v5/state --output outputs/blobby-earth-v5/planet-native.tif --checkpoint-dir outputs/blobby-earth-v5/checkpoints
```

As usual, use fresh output paths for a new run. A completed state should be
exported with `export`, not passed to `generate` again.

All 37 tests pass. The preceding v0.4 run has 28.56% sphere-area-weighted
land/ocean agreement (all generated pixels were land) and 0.206 correlation
between draft brightness and generated elevation within painted land. Its
measured report is retained beside the original run as
`draft-layout-verification.json`; the original map was not modified.

## Finished pretrained replacement

The 4096×2048 float32-metre GeoTIFF was generated from the supplied PNG and
the same seed and requested settings. Numeric results:

| Measurement | Previous importer | Version 0.5 |
|---|---:|---:|
| Land/ocean agreement, area weighted | 28.56% | 98.41% |
| Land intersection over union, area weighted | 28.56% | 94.63% |
| Painted ocean below sea level | 0% | 98.59% |
| Painted land remaining land | 100% | 97.97% |
| Land elevation correlation with painted brightness | 0.206 | 0.908 |

New raster extrema are -3,116.37 m and 6,739.70 m. White elevation is a guide,
not a final-height cap. Painted-ocean elevation standard deviation is 682.91 m;
ocean depths are varying rather than zero-filled. The remaining coverage
differences include coast pixels and small features unresolved by the guide.

All elevations are finite. Date-line and exact-pole identification errors are
zero, as are shared cube-node errors. A date-line-crossing export agrees bitwise
with the matching global pixels. The periodic row/column gradient diagnostics
are both approximately 1.004. Cross-face gradient ratios range from 0.652 to
2.070; as before, exact value continuity does not guarantee derivative continuity.

The pretrained replacement was generated once. Same-seed repeatability is
covered by independent diagnostic pipeline tests; a second independent full
pretrained run has not been performed. This report establishes layout adherence
and numeric boundaries, not a guarantee of geological realism or visual quality.
