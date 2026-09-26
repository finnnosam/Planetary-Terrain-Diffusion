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
