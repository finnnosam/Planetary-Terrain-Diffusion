# Spherical Terrain Diffusion

Generate global or regional elevation GeoTIFFs with [Terrain Diffusion](https://github.com/xandergos/terrain-diffusion). Run these commands from the project folder in PowerShell.

## Install

Install Python 3.11 or 3.12 **with Tcl/Tk** (needed for the desktop UI) and Git. On Windows, for example:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
```

Open a new terminal after installing, then install this project and its model dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e '.[models]'
git clone https://github.com/xandergos/terrain-diffusion.git upstream
git -C upstream checkout e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230
```

`upstream/` is Git ignored and must be cloned for each fresh checkout. Model weights are downloaded on first use (about 1.14 GB per model), or you can supply a complete local model folder with `--model`.

**CUDA is highly recommended** for generation. It requires an NVIDIA GPU and an additional CUDA-enabled PyTorch installation; select `--device cuda` or **Compute device → cuda** afterward. The install above is sufficient for CPU use.

## Command line

Use `.venv\Scripts\python.exe -m planet_diffusion COMMAND`. Paths below are relative to the project folder unless absolute. `--state` is a saved state directory; `--output` and `--climate-output` are GeoTIFF paths. Output files and completed states must use new paths.

| Command | Inputs and units |
|---|---|
| `generate` | Required: `--state DIR --output FILE`. Generates a global or regional elevation GeoTIFF. Rerun with the same settings and checkpoint folder to resume an interrupted run. |
| `export` | Required: `--state DIR --output FILE`. Exports another resolution or pixel window from saved elevation. |
| `query` | Required: `--state DIR --bounds WEST SOUTH EAST NORTH --width PX --height PX --output FILE`. Generates another region from a completed cube run. |
| `verify` | Required: `--state DIR`. Checks saved state integrity. |
| `gui` | Opens the desktop UI; takes no inputs. |

### `generate` options

| Input | Meaning / default |
|---|---|
| `--seed N\|random` | Unsigned 64-bit integer or random seed (default `random`). The chosen seed is saved. |
| `--draft FILE` | Optional north-up 2:1 global PNG: brightness maps linearly between black and white elevations; transparency uses procedural terrain. |
| `--draft-black-metres M` | Signed elevation of black PNG pixels in metres (default `-2000`). Gray values interpolate linearly to white. |
| `--draft-ocean-depth M` | Compatibility option: positive depth sets black to its negative; `0` uses `-2000`. Cannot combine with `--draft-black-metres`. |
| `--draft-white-metres M` | Elevation of white PNG pixels in metres (default `4000`). |
| `--draft-refinement N` | PNG elevation refinement, `0.01–4` (default `0.2`); smaller follows the draft more closely. |
| `--conditioning-dir DIR` | Folder of global conditioning GeoTIFFs described below; use instead of `--draft`. |
| `--snr E,T,TS,P,PCV` | TIFF refinement for the five channels in the order below, each `0.01–4` (default `0.2,0.2,1,0.2,1`); smaller follows input more closely. Requires `--conditioning-dir`. |
| `--coarse-height N` | Logical guide height in cells, multiple of 4; default `8`, or `1024` for regional-only 90 m / `2560` for regional-only 30 m. Native output height is `N × 256` pixels. |
| `--preview` | With `--regional-only`, choose the smallest guide grid matching regional output pixel density. Omit `--coarse-height`. |
| `--regional-only` | Compute only the requested region; requires `--bounds`. |
| `--allow-large-region` | Proceed when the regional decoder estimate exceeds 2,000 patches; useful for unattended runs. |
| `--bounds W S E N` | Regional west/south/east/north in longitude/latitude degrees. West/east: `−180–180`; south/north: `−90–90`. East less than west crosses the date line. Requires `--width` and `--height`. |
| `--width PX`, `--height PX` | Regional output dimensions in pixels. Without bounds, `--height` selects global output height (default native); global width is twice height. |
| `--radius-metres M` | Sphere radius in metres (default `6371000`). |
| `--climate-output FILE` | Also writes five-band climate GeoTIFF; requires cube geometry. |
| `--coarse-steps N` | Coarse denoising steps (default `20`; minimum `2`). |
| `--latent-batch-size N` | Maximum latent patches per model call (default `1`); higher values use more memory. Applies to global and regional-only cube generation; regional batches contain only missing dependencies and may be smaller. Saved queries reuse this setting. |
| `--model ID\|DIR` | Hugging Face model ID or local model folder (default `xandergos/terrain-diffusion-90m`; `xandergos/terrain-diffusion-30m` is also supported). |
| `--revision REV` | Model revision; defaults to the pinned revision for the selected model. |
| `--device auto\|cpu\|cuda` | Compute device (default `auto`: CUDA when available, otherwise CPU). |
| `--precision auto\|float32\|tf32\|bfloat16` | Default `auto`: native CUDA bfloat16 when supported, otherwise float32. Explicit overrides are available. Checkpoints record the resolved mode. |
| `--patch-cache-mib N`, `--blend-cache-mib N`, `--decoder-cache-mib N` | Retained array budgets per sparse stage / regional decoder (defaults `64`, `8`, `64` MiB). Zero disables retention. Also supported by `query`. |
| `--threads N` | PyTorch CPU threads (default `4`). |
| `--upstream DIR` | Pinned Terrain Diffusion source directory (default `upstream`). |
| `--checkpoint-dir DIR` | Checkpoint directory (default `STATE.checkpoints` for cube runs). |
| `--backend terrain\|diagnostic` | Backend (default `terrain`); `diagnostic` is for testing and does not generate learned terrain. |
| `--geometry cube\|equirectangular` | Geometry (default `cube`); `equirectangular` is legacy and does not support PNG/TIFF conditioning or climate export. |

`--regional-only` is the practical choice when only a small area is needed. The 30 m and 90 m models need separate run folders. A requested regional pixel density cannot exceed the native density set by `--coarse-height`.
Regional generation and `query` ask for confirmation when the estimated region size exceeds 2,000 patches. The desktop launcher shows a confirmation dialog; the CLI prompts in a terminal. For unattended CLI runs, use `--allow-large-region`.

Reducing output dimensions alone does not lower generation detail. Use `--preview` or the desktop **Preview** level for lower-detail regional generation matched to your output dimensions in pixels. Preview can produce different terrain from Full detail; use separate run folders for Preview, Full detail, and different precision modes. Saved queries retain their run's detail level and precision.

In the desktop UI, **Stop** preserves completed progress. Resume with the same settings and checkpoint folder.

### `export` and `query` options

| Command | Optional input | Meaning / units |
|---|---|---|
| `export` | `--height PX` | Global output height in pixels (default native); width is twice height. |
| `export` | `--window X Y WIDTH HEIGHT` | Pixel rectangle in the chosen global output grid. |
| `export` | `--climate-output FILE` | Export climate from a state generated with climate output. |
| `query` | `--checkpoint-dir DIR` | Generation checkpoints; normally inferred beside the state. |
| `query` | `--climate-output FILE` | Export climate for the queried region; dense runs must have saved climate during generation. |
| `query` | `--upstream DIR` | Pinned source directory (default `upstream`). |

`query` uses the same longitude/latitude bounds and pixel dimensions as regional `generate`. `export` reproduces a regional state's saved grid; it cannot export areas outside that state.

PNG elevations use `black + brightness × (white − black)`, with brightness from 0 to 1. Both endpoints must be finite, and white must be greater than black. For black `-6000` and white `6000`, midpoint gray is sea level. This replaces the previous special treatment of black; use a new run folder for new generations. Saved queries retain their original mapping. TIFF inputs remain literal elevations in metres.

### Conditioning TIFF inputs and output units

The `--conditioning-dir` folder needs at least one of these single-band, north-up, 2:1 global GeoTIFFs with geographic coordinates in degrees and bounds `(-180, -90, 180, 90)`. Missing channels and nodata use a procedural guide.

| File | Input values |
|---|---|
| `heightmap.tif` | Signed elevation, metres; negative is ocean |
| `temperature.tif` | Mean temperature, °C |
| `temperature_std.tif` | Temperature standard deviation, °C |
| `precipitation.tif` | Annual precipitation, mm |
| `precipitation_cv.tif` | Precipitation coefficient of variation, percent |

Elevation output is one float32 band in metres. Climate output bands are temperature (°C), temperature standard deviation (°C), annual precipitation (mm), precipitation coefficient of variation (%), and lapse rate (°C/metre), in that order.

## Desktop UI

Double-click `Launch Planet.cmd` or run `.venv\Scripts\python.exe -m planet_diffusion gui`. **Generate / Resume** uses the run folder and current settings; **Stop** keeps completed checkpoints. **New run** selects a fresh folder; **Open run folder** opens it. For **Saved run query**, choose a completed run folder or its `state` folder, enter bounds and output dimensions, then click **Query saved run**.

| UI input | Meaning / default |
|---|---|
| Draft PNG | Optional global 2:1 PNG, as described above. |
| Conditioning TIFF folder | Optional folder of the TIFF files above, instead of a PNG. |
| Seed / Random | Unsigned 64-bit integer or random seed (default random). |
| Run folder | Folder for saved state, checkpoints, log, and GeoTIFFs. |
| Export climate maps | Also save `planet-climate.tif` (off by default). |
| Logical guide height / detail level | Cells, multiple of 4. Global default `16`; regional **Preview** chooses a grid matching output density. **Full detail** uses `1024` or `2560` for 90 m or 30 m. Editing the guide height selects **Custom**. |
| Radius | Metres (default `6371000`). |
| Output resolution | Width and height in pixels (default `8192 × 4096` globally; `512 × 512` when switching to a region). Global width must be twice height. |
| Black elevation | Signed elevation of black PNG pixels in metres (default `-2000`). |
| White elevation | Elevation of white PNG pixels in metres (default `6250`). |
| Elevation refinement | PNG or TIFF elevation refinement, `0.01–4` (default `0.2`); smaller follows input more closely. |
| TIFF climate refinement | Four values for temperature, temperature standard deviation, precipitation, precipitation variation (default `0.2,1,0.2,1`); each `0.01–4`. |
| Compute device / precision | Both default to `auto`: CUDA when available, with native bfloat16 when supported. Explicit device and precision overrides remain available. |
| Terrain model | `90 m` (default) or `30 m`. |
| Latent batch size | Positive number of patches per model call (default `1`); higher values use more memory. |
| Generation area / saved query | `Whole globe` (default), `Region`, or `Saved run query`. |
| Regional bounds | West, south, east, north in longitude/latitude degrees; east less than west crosses the date line. |
| Only generate requested region | For **Region**, compute only nearby model tiles (checked by default). |
