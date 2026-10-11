# Toy fixed-scale refinement model

A small, self-contained miniature of the planned 23 km → 2.9 km model, scaled up 6–7×. It learns Earth at one spacing (20 km), conditioned on the same terrain area-averaged 8× (160 km). It has only ever seen Earth at 20 km, so whatever it draws is sized correctly for 20 km. That is the property the pooled and overview runs lack.

It uses `upstream/data/global/etopo_10m.tif`, which is already on disk, and the project `.venv`. It is independent of `planet_diffusion`.

## Commands

Run from the project folder:

```powershell
# 1. Dataset: ~10 min on CPU, ~200 MB
.venv\Scripts\python.exe -m toy_meso.build_dataset --dem upstream\data\global\etopo_10m.tif --out outputs\toy_meso\dataset.npz

# 2. Train (resumable: rerun the same command)
.venv\Scripts\python.exe -m toy_meso.train --data outputs\toy_meso\dataset.npz --run outputs\toy_meso\run1

# 3. Evaluate on held-out regions
.venv\Scripts\python.exe -m toy_meso.evaluate --data outputs\toy_meso\dataset.npz --run outputs\toy_meso\run1

# 4. Large seamless region (5,120 km here), e.g. Scandinavia, which was held out of training
.venv\Scripts\python.exe -m toy_meso.demo_region --run outputs\toy_meso\run1 --dem upstream\data\global\etopo_10m.tif --lat 64 --lon 14 --cells 34

# 5. The same model on one of your planets: it refines the planet's 160 km layout
.venv\Scripts\python.exe -m toy_meso.demo_region --run outputs\toy_meso\run1 --dem human_input\CoarsePooling3\planet-native.tif --lat 20 --lon -40 --cells 34
```

Training defaults: 14.4M-parameter EDM U-Net (64 channels, multipliers 1,2,4, attention at 16×16), batch 32, 30,000 updates, bf16 on CUDA. Based on the earlier 3070 benchmark (112 patches/s at this size), expect roughly **2–3 hours**. `--steps 10000` gives a usable first look in under an hour. Every `--eval-every` updates, a preview PNG is written to the run folder.

### PNG sketches

`--dem` also accepts a 2:1 north-up grayscale PNG. Brightness maps linearly from `--black-metres` (default -2000) to `--white-metres` (default 4000), as with `planet_diffusion --draft`:

```powershell
.venv\Scripts\python.exe -m toy_meso.demo_region --run outputs\toy_meso\cpu-smoke --checkpoint smoke-ema.pt --dem human_input\Blobby_2.png --black-metres -6000 --white-metres 6000 --lat 0 --lon 0 --cells 34
```

For sketches where pure black means ocean and land starts just above black (like `Blobby_2.png`), add `--ocean-metres -4000 --black-metres 0 --white-metres 6250`.

The sketch is averaged to 160 km cells, so the model adds everything finer than that; detail you draw below ~160 km is smoothed away.

## Whole planet (cubed sphere)

```powershell
.venv\Scripts\python.exe -m toy_meso.planet --run outputs\toy_meso\run1 --dem path\to\sketch.png --black-metres -6000 --white-metres 6000 --out outputs\toy_meso\planet1
```

Same input options as `demo_region` (`--black-metres`, `--white-metres`, `--ocean-metres`, or a GeoTIFF). It uses `planet_diffusion.cube`: the input is averaged onto 63 coarse intervals per cube face (~159 km), and the model generates 504 per face (~19.9 km), blending 864 overlapping windows across face edges at every step. Outputs in `--out`:

- `planet.tif`: 2016 x 1008 global GeoTIFF in metres (spherical CRS, open in QGIS); change with `--height`
- `condition.tif`: the averaged input on the same grid, for comparison
- `planet-preview.png`, `planet.json`, `cube-elevation.npy` (the raw cube field)

`--steps 16` halves the time with little visible loss; `--seed` gives a different version of the same planet. Spacing varies by up to ~30% across each cube face (smallest near the corners), which the model doesn't know about, so features there are slightly small.

## Whole planet at the model's own spacing (tiled, region-consistent)

Use this generator for `outputs\meso\run2-lowlr\latest.pt`. The current `meso-tiled-v2-cell-centres` sampler matches the training cell-centre alignment without moving the fine output coordinates or adding a resampling shift. The checkpoint itself does not need retraining. The older `toy_meso.planet` consensus generator above does not include this correction.

### Desktop UI for the fixed model

Double-click `Launch Meso.cmd`, or run `.venv\Scripts\python.exe -m toy_meso.gui`. It starts with your `run2-lowlr\latest.pt` checkpoint, `human_input\B2W.png`, black/white elevations of -6000/+6000 m, 32 sampling steps, and 16 tiles per batch. The default output folder is `outputs\meso\planet-ui`.

- Choose **Input map**, **Random terrain**, or **Map + random**. Random terrain needs no map; its layout and finer model detail follow the seed. Mixed mode has a **Map contribution** slider: 0% uses random elevations, 100% uses the map, and intermediate values blend the elevation inputs in metres before the model runs. Transparent PNG areas use random terrain in mixed mode, even at 100% map contribution.
- Optional **Pause to approve the random layout before generating terrain** saves `random.preview.png` (the random globe) and `input.preview.png` (the averaged input for the selected globe/region, including the map blend). The run then stops before GPU inference. **Open input preview**, then **Approve & Generate** to continue with the same cached input. Leave the checkbox off for continuous generation. Approval is saved in `input-review.json`; changed input settings, area/pixel sampling, or preview images require a fresh review. To reject the layout, use **New run** and another seed.
- For map or mixed input, choose a global 2:1 PNG or a single elevation TIFF. Then choose the checkpoint, run folder, seed, and whole globe or regional bounds. Map controls are disabled for random terrain, and PNG elevation controls are disabled for TIFF inputs. The optional pure-black ocean elevation uses `--ocean-metres`; blank means no override.
- **Advanced settings** contains sampling steps, tile batch size, planet radius, optional tile stride, output height/region width, filename, and device. Blank dimensions use the model's native spacing; whole-globe width is always twice its height. There are no climate, refinement, guide-height, or 30 m / 90 m controls.
- **Generate / Resume** saves `meso-launch.json` and appends to `generation.log`. **Stop** keeps completed tiles. Intermediate arrays and exports are written to temporary files, so a stopped export can resume with the same settings and filename.
- **Load settings** accepts `meso-launch.json` or the fixed sampler's `settings.json`, including source and mix settings. The latter restores cache settings; choose the desired region/dimensions afterwards. New regions can reuse the same folder and tiles with an unused output filename. **New run** chooses a fresh folder while retaining model and input settings. Existing map-only v2 runs remain compatible; changing source or mix percentage needs a new run folder.
- On completion, **Open preview** shows the shaded PNG and **Open folder** opens the results. The main project's `Launch Planet.cmd` continues to launch its original UI.

`planet` above blends all windows at every step, which is fine at 20 km but too heavy at 2.88 km. `planet_tiled` uses a fixed grid of overlapping tiles on the cube instead. Each tile is denoised to completion on its own, from noise keyed to its absolute position on the sphere, and finished tiles are blended (as the upstream coarse stage does). A point depends only on the few tiles covering it, so:

- a region generated on its own is **identical** to the same area of the full planet
- tiles are cached in the planet folder: a stopped run resumes, and later regions reuse them

```powershell
# whole planet (2.88 km model: 8,664 tiles; writes planet.tif ~13,900 x 6,950)
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs\meso\run2-lowlr --dem human_input\B2W.png --black-metres -6000 --white-metres 6000 --out outputs\meso\planet-fixed --steps 32 --batch 16

# just a region (lon/lat box W S E N); same --out folder reuses / adds to the cache
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs\meso\run2-lowlr --dem human_input\B2W.png --black-metres -6000 --white-metres 6000 --out outputs\meso\planet-fixed --steps 32 --batch 16 --bounds 5 44 15 49 --name Alps.tif

# random planet; no input map needed
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs\meso\run2-lowlr --source random --seed 42 --out outputs\meso\planet-random

# mix 75% map elevations with 25% random elevations
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs\meso\run2-lowlr --source mixed --map-weight .75 --dem human_input\B2W.png --black-metres -6000 --white-metres 6000 --seed 42 --out outputs\meso\planet-mixed
```

- `--dem` can also be one of your `planet-coarse.tif` files (already at 23 km).
- `--source map|random|mixed` chooses the elevation input. Without `--source`, a supplied `--dem` selects map input; no `--dem` selects random terrain. `--map-weight` is 0..1 for mixed input (default 0.5). The random guide uses continuous 3-D fractal noise with a fixed continent scale and the main project's elevation distribution; it uses no climate channels or older pretrained models.
- The folder's `settings.json` pins the checkpoint, input, black/white values, seed, steps, batch size, sampler version and radius; a different setting needs a new `--out` folder. Old v1 sampler folders such as `planetC` are rejected: use a fresh folder to regenerate with the corrected sampler. Existing random-source v1 runs retain their original smooth guide for resuming; **New run** selects the revised multiscale random source.
- `--width/--height` set output pixels (default: native spacing). `--batch` is fixed per run for identical numerics; lower it if GPU memory spills.
- Outputs: the GeoTIFF, a `.preview.png`, `coarse.npy` (six cell-centred averaged input charts with halos), `coarse-valid.npy` (which input cells have been prepared), and `tiles\` (complete encoded-elevation patches, ~0.5 GB for a whole 2.88 km planet). Unprepared coarse cells are filled as new regions need them. Older coarse caches without a validity mask are complete and remain compatible. The v2 coarse arrays are chart cells, not shared cube nodes, and the v2 tile cache contains elevations rather than normalized residuals.
- Custom `--stride` values must be positive multiples of the model's factor (8 here), smaller than the tile size. The default remains 96 for 128-pixel tiles.
- Known limit: where tiles overlap (a third of each tile width), two slightly different tile results are averaged, which softens fine detail there by a few percent (measured ~7% lower detail amplitude where both axes overlap, ~2.5% where one does).

### Random input structure

The earlier random source used four noise octaves, leaving most land much smoother than the model's Earth training conditions. In the completed `human_input/MESO1` run, the model did run on all 8,664 tiles, but its land additions measured only about 3.4 m RMS across 130 sampled tiles. It therefore resembled the smooth random input even after a full sampling run.

New random/mixed runs use `meso-random-elevation-v2-multiscale`: nine octaves with gain 0.7 and a fixed global noise calibration, retaining broad shapes while adding variation toward the ~23 km input spacing. Across 16 sampled land tiles from the same seed, the model added about 142 m RMS with this guide versus 4 m with the earlier guide. The checkpoint and detail sampler are unchanged. New noise can change coastlines and elevations; use a fresh run rather than mixing versions in an existing cache. Random input still guides the layout, while the trained model supplies finer terrain, so the number of model calls is the same as for an input map.

CLI review: add `--pause-for-input` to a random or mixed run. Exit code 42 means the previews are ready and inference has not started. Rerun the same command with `--approve-input` to approve its unchanged previews and continue. Subsequent resumptions with that approved layout proceed without pausing again.

### Generation speed

The tiled generator prepares only the input cells read by requested tiles. Small regions also blend only the fine nodes needed by the export, instead of allocating fine arrays for the whole planet. Finished tiles and input cells are reused when you resume or export another region. These changes preserve the generated elevations and work automatically through the UI.

GPU inference keeps precision-converted weights cached through each sampling batch and avoids copying step values back to the CPU. Only the selected inference weights move to the GPU; optimizer state and the checkpoint's other weight copy stay in system memory. An export using entirely cached tiles keeps the model on the CPU.

Measured on the RTX 3070 with `run2-lowlr`, 32 steps and batch 16: a fresh 2° × 2° mixed-input region (64 × 64 export, eight tiles) took **7.8 s instead of 28.8 s**, with identical input cells, tiles, and exported elevations. The sampling-only benchmark improved from 5.71 to 5.60 s per 16 tiles, and GPU allocation fell by about 370 MiB. Whole-planet generation still spends most of its time in the model.

For a larger speed change, choose **16 sampling steps** in Advanced settings and **New run**. Sampling measured 2.82 s per 16 tiles, about twice as fast as 32 steps. This changes the terrain: the tested Alps tiles differed by 7.3 m RMS, and fine-scale power increased by about 4–9%. The default stays at 32 steps. Lowering export resolution primarily reduces export work; it does not proportionally reduce the number of model tiles for a whole globe.

## CPU smoke-test result (already run)

A small model (32 channels, 3.6M parameters) was trained for only 3,000 updates at batch 16, on CPU. Its EMA weights are in `outputs\toy_meso\cpu-smoke\smoke-ema.pt` with the images. You can try `demo_region` and `evaluate` with it before training, by adding `--run outputs\toy_meso\cpu-smoke --checkpoint smoke-ema.pt`. It can't be resumed, because it has no optimizer state.

Held-out power relative to real terrain (192 patches):

| Wavelength | Bilinear condition | Generated |
|---|---:|---:|
| 40–80 km | 0.004 | 0.82 |
| 80–160 km | 0.025 | 0.82 |
| 160–320 km | 0.14 | 0.93 |
| 320–1,280 km | 0.86 | 0.99 |

Slope RMS: real 0.0087, generated 0.0084, bilinear 0.0038. Land fraction and land elevation percentiles match within 1%. Nothing is oversized. The model is still slightly too smooth, which the full GPU run should improve.

## Data

- **Patches:** gnomonic grids around area-uniform random points, with random rotation, so latitude and poles are treated uniformly.
- **Targets:** area averages (3×3 supersampling per cell). They are encoded as signed square-root metres, as upstream does.
- **Condition:** the 8×8 block mean of the same metres, with one 160 km cell of margin on each side. The model predicts only the residual over the bilinear condition.
- **Held out:** Scandinavia, Tibet/Himalaya, Patagonia, East Africa, a Pacific abyss box, and ~8% of 15° blocks. No training patch touches them. Validation patches are centred in them.

## Reading the evaluation

`eval.json` and `eval-spectrum.png` compare power per wavelength band against the real held-out terrain:

- **Bilinear condition:** near 0 below 160 km. This is the "too smooth" baseline.
- **Generated:** should sit near 1.0 in every band. Above 1.0 in the 40–80 km band means features too large or too strong at the pixel scale, i.e. the oversized-fjord failure. Below 1.0 means too smooth.
- **`condition_rms_error_m`:** how well the generated terrain averages back to the 160 km condition. The real terrain gives ~0, which is the floor.

## Limits (deliberate, it's a toy)

- ETOPO 10′ is itself ~18.5 km, so 20 km targets have only a little real sub-cell detail. The real model would use ETOPO 30″ (~0.9 km) averaged to 2.9 km.
- Elevation only. No climate channels, no condition-noise augmentation for coarse-model inputs.
- `demo_region` works on a flat gnomonic frame, not the cube. Wiring into the cube sampler would reuse `cube.consensus`.
