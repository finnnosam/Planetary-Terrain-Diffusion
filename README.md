https://github.com/xandergos/terrain-diffusion

# Spherical Terrain Diffusion

Generate spherical elevation with Xandergos's pretrained Terrain Diffusion
models. Generate a whole planet, or request a region without decoding the
whole globe. Generation runs on six connected cube faces. Longitude and
latitude are used for export.

## Install

### Requirements

Install **Python 3.11 or 3.12** with Tcl/Tk for the desktop launcher, and
**Git**. Create a virtual environment from the project folder:

```powershell
python -m venv .venv
```

Install each Python package:

| Package | Command |
|---|---|
| NumPy | `.venv/Scripts/python -m pip install "numpy>=1.26"` |
| Rasterio | `.venv/Scripts/python -m pip install "rasterio>=1.4"` |
| Pillow | `.venv/Scripts/python -m pip install "Pillow>=10"` |
| FastNoiseLite | `.venv/Scripts/python -m pip install "pyfastnoiselite==0.0.7"` |
| FastNoiseLite | `.venv/Scripts/python -m pip install "pyfastnoiselite==0.0.7"` |
| PyTorch | `.venv/Scripts/python -m pip install "torch>=2.4"` |
| Diffusers | `.venv/Scripts/python -m pip install "diffusers>=0.30"` |
| Safetensors | `.venv/Scripts/python -m pip install "safetensors>=0.4"` |
| Hugging Face Hub | `.venv/Scripts/python -m pip install "huggingface-hub>=0.25"` |
| This project | `.venv/Scripts/python -m pip install --no-deps -e .` |

The sampler also needs the pinned **Terrain Diffusion source**, already present
in `upstream/` in this workspace. For a fresh checkout:

```powershell
git clone https://github.com/xandergos/terrain-diffusion.git upstream
git -C upstream checkout e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230
```

The first model run downloads about 1.14 GB of weights. This workspace already
has them in `models/terrain-diffusion-90m/`. For GPU generation, install a
CUDA-enabled PyTorch build for your system and select `--device cuda`.

### Desktop launcher

Double-click **Launch Planet.cmd**. Choose an optional PNG draft and a seed, then
click **Generate / Resume**. The default output is an 8192×4096 GeoTIFF.
**Open run folder** shows the TIFF, saved state, checkpoints, and log.
**New run** picks a fresh folder. **Stop** preserves completed work; resume with
the same folder and settings. You can also launch with
`.venv/Scripts/python -m planet_diffusion gui`.

For regional generation, choose **Generation area → Region**, enter **West,
South, East, North** in degrees, and set **Output resolution** width and height
in pixels. East smaller than west crosses the date line (for example, west
170 and east −170 covers 20 degrees). Polar regions are supported. Whole-globe
output requires width = twice height; a region can have any aspect ratio.

**Coarse height** still sets detail density on the sphere. The launcher reports
the minimum coarse height if a region's requested pixel resolution exceeds
that density. Larger coarse heights increase global guide work, even for a
small region. Output dimensions do not change the sphere's radius or location.

### Procedural globe conditioning

The cube sampler follows upstream `synthetic_map.py`: five independent Perlin
FBm fields are quantile-matched to ETOPO elevation and WorldClim temperature,
temperature variability, precipitation, and precipitation variability. It applies
the source's precipitation-dependent lapse rate, temperature clipping and cold
stretch, temperature variability regression, and precipitation variability damping.
Elevation is then signed-square-root encoded; `WorldPipeline`'s channel ordering,
normalization and conditioning-noise mixture are preserved.

Noise is sampled in 3-D at shared sphere directions, using a radius of
`2 * face_coarse / pi` coarse cells. This keeps its scale tied to model cells as
the globe grows, with continuous fields across all edges and poles. The noise
quantiles are calibrated in 3-D against a fixed population; individual faces and
small planets are never independently histogram-matched. Small globes can therefore
cover only a narrow part of the reference distributions. No latitude-based climate
bands are imposed, matching the source's procedural approach.

Model `frequency_mult` and `drop_water_pct` settings now control these fields.
Reference statistics are bundled for offline generation; provenance and regeneration
instructions are in `planet_diffusion/data/README.txt`. TIFF imports follow the
source's import mode: merge raw channels, skip synthetic climate finalization, and
encode elevation after blending. PNG drafts retain their existing elevation-overlay
behavior over finalized procedural climate. Code, reference data, noise library
version and procedural settings participate in checkpoint identity; previous
checkpoints require a new run folder.

### Coarse tile sampling

The coarse stage follows upstream `WorldPipeline`: 64 x 64 tiles at stride 48,
20 denoising steps by default, and an independent scheduler history for each
complete tile. It converts the six output channels back to physical units and
reconstructs the elevation percentile channel before blending completed tiles
with the source's linear weight window. `--coarse-steps` remains configurable.

The spherical adaptation retains one Gaussian draw per physical globe node and
nearest-node reads across faces. Shared edge and corner entries combine weighted
sums before normalization. This preserves identical boundary values; it does not
prove derivative continuity. On small globes, a 64-cell context exceeds a face,
so the existing gnomonic halo projection stretches and repeats source nodes.
There is no intermediate per-step consensus in the coarse stage.

For a reproducible pretrained coarse-only comparison of the old loop, a larger
context with per-step blending, and completed-tile blending, install Matplotlib
and run `python tools/compare_coarse.py --output outputs/coarse-comparison`.
It saves arrays, area-weighted guide adherence metrics and comparison images.
These coarse elevation proxies are not final decoded terrain.

### Latent and decoder assembly

Latents follow the source's default `T=2` path: 64 x 64 tiles at stride 32,
a normalized blend between the two passes, and a 4 x 4 coarse window offset
by -1 coarse node. The decoder reads completed latent nodes, repeats each into
an 8 x 8 block, and uses 512 x 512 tiles at the source's default stride 384.
Both stages use the exact source linear weight ramp, include every overlapping
tile at the final shared node, and combine shared weighted sums before division.
The regional decoder uses the same placement and weighting rules.

Nearest-node cross-face sampling is retained for latent state and decoder
conditioning as well as Gaussian noise. Bilinear learned-field halos were
tested but reduced fine residual variation without consistent edge improvement.
The existing global and regional noise generators are unchanged. Shared node
values agree exactly; this is not a guarantee of matching boundary derivatives.

`tools/compare_detail.py` reproduces the pretrained before/after comparison,
including weighting-only and rejected interpolated-halo variants. It requires
Matplotlib and the local pretrained models. For example:

```powershell
.venv/Scripts/python tools/compare_detail.py --output outputs/detail-comparison --face-coarse 4 --seed 123
```

Use a new checkpoint directory after this assembly change. The implementation
and assembly settings are included in checkpoint identity.

### PNG draft convention

Use a **2:1 global PNG** with north at the top. The left and right edges are
adjacent longitudes and should match. Draw broad shapes: each coarse guide cell
expands to 256 output pixels per axis. An 8k output has 386 unique coarse nodes,
so small features may disappear.

- **Black:** ocean, with a default −2000 m elevation prior. The model generates
  the final seafloor. Change the prior with **Ocean depth hint**.
- **Lighter shades:** higher land. White defaults to 4000 m; change this with
  **White land elevation**.
- **Transparency:** fully transparent areas use a seeded procedural guide;
  partial transparency blends with it. RGB uses brightness; 16-bit grayscale works.

**Elevation refinement** defaults to `0.2` and accepts `0.01–4`. Smaller
values follow the draft more closely; larger values allow more change. The
model can still move coastlines and elevations. Climate remains procedural.
Avoid conflicting features at the poles. The PNG and its settings are saved
with the planet for repeatability.

```powershell
.venv/Scripts/python -m planet_diffusion generate --draft "E:/maps/my-draft.png" --coarse-height 16 --draft-refinement .2 --seed random --model models/terrain-diffusion-90m --state outputs/my-planet --output outputs/my-planet.tif
```

Omit `--seed` for a random seed, or give an unsigned integer. The seed is
saved with the planet. An interrupted run reuses its checkpoint seed. Optional
draft flags are `--draft-ocean-depth 0` and `--draft-white-metres 4000`.

## Generate and export

### Elevation and climate input maps

Ported from upstream's `tiff-export`: supply **Conditioning TIFF folder** in the
launcher, or `--conditioning-dir FOLDER` in the CLI, to guide both terrain and
climate. Use this instead of a PNG draft. The folder needs at least one of:

| File | Values |
|---|---|
| `heightmap.tif` | Signed elevation in metres; negative values are ocean |
| `temperature.tif` | Mean temperature in degrees C |
| `temperature_std.tif` | Temperature standard deviation in degrees C |
| `precipitation.tif` | Annual precipitation in mm |
| `precipitation_cv.tif` | Precipitation coefficient of variation in percent |

Each file must be a **single-band, north-up, 2:1 global GeoTIFF** with a
geographic CRS in degrees and bounds `(-180, -90, 180, 90)`. Different input
resolutions are supported. Scale and offset metadata are applied before unit
conversion. Missing channels, nodata, masked pixels, and nonfinite pixels use
the seeded procedural guide; coverage boundaries blend smoothly. Variability
and precipitation values must be nonnegative. Regional or projected input maps,
including unconverted planar Azgaar exports, must first be mapped onto this global
grid; assigning them a geographic CRS alone is insufficient.

Inputs are averaged over spherical coarse-cell footprints, wrap at the date
line, and share pole and cube-edge values. Elevation is converted to upstream's
signed square-root representation and temperature variability to its internal
hundredths of a degree. These maps guide model refinement, rather than replacing
the generated output. Very small input features may disappear at coarse resolution.

In the launcher, **Elevation refinement** controls elevation for either PNG or
TIFF inputs. **TIFF climate refinement** contains only the four climate values:
temperature, temperature standard deviation, precipitation, and precipitation
coefficient of variation (defaults `0.2,1.0,0.2,1.0`).
The CLI's `--snr` still accepts all five comma-separated values in table order,
each from `0.01` to `4`. Defaults match upstream's TIFF command:
`0.2,0.2,1.0,0.2,1.0`. Smaller values follow the respective input more closely.
The CLI's `--draft-refinement` applies only to PNG drafts.

```powershell
.venv/Scripts/python -m planet_diffusion generate --conditioning-dir E:/maps/global-guide --snr 0.2,0.2,1,0.2,1 --seed 42 --device cuda --coarse-height 16 --state outputs/guided-planet --output outputs/guided-planet.tif --climate-output outputs/guided-climate.tif
```

TIFF conditioning supports both whole-globe and regional cube generation. The
original TIFF bytes are saved in `STATE/conditioning/` and
`CHECKPOINT_DIR/conditioning/`. Content hashes and refinement values are part of
checkpoint identity, so changed inputs cannot silently resume an older run. To
resume after moving the original folder, pass the checkpoint's `conditioning/`
folder with the same refinement values. Existing checkpoints from before this
code change require a new run folder.

Set **Latent batch size** in the launcher or pass `--latent-batch-size 2` to
evaluate multiple latent patches per model call. This applies to whole-globe
and regional cube generation. The default is `1`; increase cautiously because
larger batches need more memory and can be slower on memory-limited GPUs.
On Windows, CUDA allocation is limited to 65% of GPU memory (or an existing
tighter limit). This prevents oversized cuDNN convolution workspaces from
spilling into system RAM. On the tested 8 GB RTX 3070, batch sizes 4 and 8
approximately halved latent model time after this fix; other generation stages
are unaffected by latent batching. Start with 4 on that card.
Batching preserves patch order and overlap blending. Model floating-point
results can vary slightly with batch size, so resume with the same setting.
Existing checkpoints from before this code change require a new run folder.

### Climate maps

The launcher enables **Export climate maps** by default, producing
`planet-climate.tif` alongside `planet-native.tif`. The climate TIFF has five
float32 bands on exactly the same grid, bounds, and spherical CRS as elevation:

| Band | Value | Unit |
|---|---|---|
| 1 | Temperature adjusted for generated elevation | degrees C |
| 2 | Temperature standard deviation | degrees C |
| 3 | Annual precipitation | mm |
| 4 | Precipitation coefficient of variation | percent |
| 5 | Temperature lapse rate | degrees C per metre |

Climate patterns originate in the generated coarse map. Temperature receives
additional detail from the final elevation using upstream's local land-weighted
regression; underwater elevation is treated as sea level for this adjustment.
Temperature variation is converted from upstream's internal hundredths of a
degree. These are model climate estimates, not a weather simulation.

The CLI accepts `--climate-output PATH` for both `generate` and `export`:

```powershell
.venv/Scripts/python -m planet_diffusion generate --seed 42 --device cuda --latent-batch-size 4 --model models/terrain-diffusion-90m --state outputs/climate-planet --output outputs/elevation.tif --climate-output outputs/climate.tif
.venv/Scripts/python -m planet_diffusion export --state outputs/climate-planet --height 512 --output outputs/elevation-overview.tif --climate-output outputs/climate-overview.tif
```

New CLI cube runs always save compact, checksum-verified climate features in
`state/climate.npy`, even if TIFF export is disabled. Later exports need no model
inference. Global overviews, date-line-crossing tiles, and regional exports are
supported; regional states retain their original bounds and resolution. Older
elevation-only states remain readable but need regeneration to provide climate.
Climate output is unavailable for the legacy equirectangular generator.

### Elevation export

```powershell
.venv/Scripts/python -m planet_diffusion generate --seed 42 --coarse-height 8 --model models/terrain-diffusion-90m --state outputs/planet42 --output outputs/planet42.tif
.venv/Scripts/python -m planet_diffusion verify --state outputs/planet42
.venv/Scripts/python -m planet_diffusion export --state outputs/planet42 --height 512 --output outputs/overview.tif
.venv/Scripts/python -m planet_diffusion export --state outputs/planet42 --output outputs/detail.tif --window 500 300 256 256
```

`coarse-height` must be divisible by four. Native output height is
`coarse-height × 256`; width is twice the height. Higher values preserve more
guide detail but increase compute and memory use. Checkpoints are saved in
`STATE.checkpoints`; rerun with the same settings to resume. Completed outputs
are protected from overwriting.

### Generate only regional detail

```powershell
.venv/Scripts/python -m planet_diffusion generate --seed 42 --device cuda --coarse-height 16 --bounds 170 -10 -170 10 --width 256 --height 256 --model models/terrain-diffusion-90m --state outputs/region42 --output outputs/region42.tif
```

`--bounds WEST SOUTH EAST NORTH` requires `--width` and `--height`. Bounds are
longitude/latitude degrees; dimensions are output pixels. The same global PNG
draft convention applies. The small coarse and latent fields still cover the
globe, while decoder patches and reconstruction halos are requested on demand.
No full-resolution globe field is allocated or exported. Tiny regions still
need overlapping model patches and surrounding context.

Regional decoder noise is seeded by cube tile, so requests with the same seed,
generation settings and aligned pixel centres agree regardless of bounds or
query order. Regional detail uses a different noise layout from the legacy
whole-globe path, so it is not an exact crop of that path. Global drafts and
spherical topology are preserved. Regional checkpoints can be reused for other
bounds/resolutions with the same generation settings and new state/output paths.

Regional states store their exact bounds and pixel grid, and `export` reproduces
that grid. They cannot export ungenerated areas or be used as full planet states.
`verify` checks regional integrity and finite heights, without claiming global
pole/date-line checks. Date-line-crossing TIFFs use continuous longitude bounds,
such as 170° to 190°, on the same custom spherical CRS as global output.

`--window X Y WIDTH HEIGHT` uses pixels in the selected global grid. Longitude
wraps at the date line; latitude windows must stay within the globe. Tiles
from the same saved global state and resolution agree exactly. This export-only
option requires a previously generated whole planet; use `generate --bounds`
to skip unneeded detail generation. Lower-resolution overviews may miss thin features.

At the default radius, `coarse-height 8` gives about 9.8 km per output pixel
at the equator, despite the model's 90 m training scale. Set
`--radius-metres 58670.88` for about 90 m equatorial spacing on a small sphere.

## Output contract

Oceans have generated seafloor elevations, not a flat water surface.

- One float32 elevation band, **metres**, scale 1 and offset 0, signed around
  the model's sea-level zero. No 0–65535 normalization or height remapping.
- North-up Plate Carrée longitude/latitude raster. Whole-globe output has bounds
  `(-180, -90, 180, 90)`, exact 2:1 aspect ratio. Pixels sample cell centres;
  neither duplicate date-line columns nor duplicate pole rows are written.
- A custom spherical geographic CRS with the requested radius; not WGS84.
  Horizontal angles are degrees. Vertical datum is model-defined zero sea
  level, not an Earth geoid, physical ocean surface or surveyed vertical CRS.
- Deflate compression, floating-point predictor, tiled storage, automatic
  BigTIFF when needed, NaN nodata. Valid generated cells are finite.
- GeoTIFF tags and `planet.json` record seed, model identity, source revision,
  algorithm, runtime versions, radius, resolution and units. `elevation.npy`
  stores the native node field (or the requested regional pixel grid); its digest detects damage.
- A date-line-crossing tile keeps a continuous affine longitude range (e.g.
  175° to 185°). Some GIS applications require splitting such tiles at 180°.

`verify` checks finite heights, the saved digest, poles, and date-line
continuity. These checks do not establish realistic geology. The pretrained
models were trained on planar patches, so spherical generation still has
geometric limits.
