# Train continents and the intermediate terrain model

Run from `E:\projects\Infinite_Diffusion`. The existing `.venv`, RTX 3070,
PaleoDEM download and ETOPO 2022 are sufficient. No extra download is required.
The completed meso checkpoint is not modified.

## Coastline-first overhaul

The new **Train Structured Continents.cmd** trains coastlines and then land/
ocean relief in separate models, with sea level fixed at zero before elevation
generation. It uses neighbour-face convolutions, whole-globe attention and
CUDA batch 16. Source data is already present. These are fresh models; the short
pilot runs are implementation checks, not replacements for the trained models.

After training, **Evaluate Structured Continents.cmd** and
**Preview Structured Continents.cmd** generate small continent-only checks.
No full meso globe starts automatically. Treat full-planet meso generation as
an expensive final step, following acceptable previews and regional checks.
See `docs/structured-continent-overhaul.md` for the architecture, training paths,
known limits and the measured shelf distortion in the old sea-level control.

## Continue and evaluate continents

**Train Continents Continued.cmd** starts/resumes `outputs/continents/run2-lowlr`
from the original update-30,000 EMA checkpoint, with a fresh optimizer,
unchanged normalization/architecture, CUDA batch 16, learning rate 0.00003,
100-update warmup, a 12,000 additional-update ceiling and five-check early
stopping. It preserves `run1` and evaluates the starting model before adapting.
Update numbers in the new run count additional updates; `initialized_from`
records the original update number. Source dataset identity and cube scale must
match. **Evaluate Continents.cmd** compares original and continued models using
16 fixed seeds and 64 sampling steps, after training finishes.

Evaluation includes all 67 training and 18 validation layouts at native scale,
area-weighted land coverage and elevations, connected land mass areas across
cube seams/poles, coastline crossing length (a grid-dependent proxy), broad
spherical elevation power, and approximate rotation/reflection alignment to
every training layout. It locally refines the two best source ages. Copying IoU
uses 2,048 equal-area probe points, so it is a similarity screen rather than a
proof that copying cannot occur. Temporal test maps remain untouched.
Training and validation geological ages have different distributions; these
statistics are reference populations rather than identical targets.

```powershell
.venv\Scripts\python.exe -u -m planet_training.evaluate_layout
```

Reports, matched planet images, distributions, spectra and sampled native cubes
are saved in `outputs/continents/run2-lowlr/evaluation`. Checkpoint hashes are
computed from the same bytes that were loaded, including when a checkpoint
path changes during training.

## Continent generation controls

The continuation and evaluation are complete. The selected frozen model is
`outputs/continents/selected-continued-42000/best.pt`; the UI selects it
automatically. Full validation loss improved by 1.89%, and matching-seed
evaluation showed fewer large land masses. The user selected the seed-42
30%-land, 1.25x-relief example for the terrain stages. Its intermediate globe is
complete; full meso generation was left running at the user's request.
See `docs/continent-improvements.md` for results, output paths and the final
inspection command.

Double-click **Launch Continents.cmd** to choose a model, seed, fresh output
folder, optional land percentage and positive-land relief multiplier. Blank
land percentage and relief 1 preserve the model's natural result. The UI follows
`outputs/continents/selection.json` when a frozen model selection is available.

These are **generation-time elevation edits**, not a newly trained conditioned
model: land coverage moves sea level using native spherical area weights;
relief scales positive elevations after that shift. They do not invent new
mountain belts, and changing sea level can connect or split land masses.
Discrete or flat elevation levels may prevent an exact requested percentage;
metadata records requested and achieved coverage, sea-level shift and relief.
Extreme settings can move guides beyond the intermediate model's training range.

The GeoTIFF samples actual raster cell centres. The older node-preview helper
includes both poles and an extra endpoint row, so it is retained for previews
only. New exports avoid silent resizing and longitude half-pixel misalignment.
Native cube data and metres-based GeoTIFFs are saved together.

```powershell
.venv\Scripts\python.exe -m planet_training.generate --run outputs/continents/run2-lowlr --out outputs/continents/controlled-example --seed 42 --steps 64 --land-percent 30 --land-relief 1.25
```

## Start training

Both datasets are prepared. Double-click **Train Planet Models.cmd** to train
the continent model and then the intermediate model automatically, using the
same GPU in sequence. Rerunning this launcher resumes both stages. Ctrl+C stops
the current stage without proceeding to the next stage.

To start each stage separately, double-click:

1. **Train Continents.cmd** — a joint six-face continent model, 30,000 updates.
2. **Train Intermediate.cmd** — elevation refinement from ~161 km to 23.04 km,
   100,000 updates.

The intermediate model trains independently on ETOPO;
it does not need a completed continent checkpoint before training.

Equivalent commands:

```powershell
.venv\Scripts\python.exe -u -m planet_training.train layout
.venv\Scripts\python.exe -u -m planet_training.train bridge
```

Both commands resume their own `latest.pt` when rerun. The update limit means
**total** updates, including those already completed. Ctrl+C saves a checkpoint;
closing the window abruptly retains the last automatic checkpoint (every 250
updates). A partially written checkpoint never replaces the last complete one.

| Stage | Default folder | Batch | Base channels | Learning rate |
|---|---|---|---|---|
| Continents | `outputs/continents/run1` | 16 whole planets | 32 | 0.0001 |
| Intermediate | `outputs/intermediate/run1` | 16 patches | 64 | 0.0001 |

The first 1,000 updates warm up the learning rate. Continents use dropout 0.1.
Both use EMA, gradient clipping, and bfloat16 on the available CUDA device.
Both default to `--device cuda --batch 16`. These defaults also apply to the
double-click launchers; `--device cpu` or `--batch` can override them.
The bridge adds modest noise to the elevation condition during training.

The log reports updates per second, examples per second, batch size and a
remaining-time estimate. An update at batch 16 processes four times as many
examples as an update at batch 4, so compare examples per second for throughput.
Interactive terminals show a single updating progress line with completion,
loss, speed and time remaining. Checkpoint, validation and preview messages
appear separately. The detailed `log.jsonl` remains machine-readable and keeps
the example-throughput measurements. Existing running processes pick up output
changes only after a stop and resume.
Data sampling, validation, and checkpoint writes affect actual completion time.
The default update counts are starting budgets, not guarantees of quality.

Every 1,000 updates, validation and fixed-seed previews are saved. `best.pt`
tracks the lowest fixed-noise validation loss; inspect the saved previews too
before selecting a final model. Layout previews include **unconditional complete
globes** and a held-out reconstruction. Bridge previews compare the upsampled
input, generated refinement, and real held-out target.
Validation covers every held-out age for continents and all 512 held-out bridge
patches, evaluated in small GPU batches.

### Evaluate the intermediate model without training

After stopping training with Ctrl+C, double-click **Evaluate Intermediate.cmd**.
It compares `best.pt` and `latest.pt` on all 512 held-out patches using the same
fixed validation noise as training. It also samples eight held-out guides with
identical initial noise for both models. Results are saved in
`outputs/intermediate/run1/evaluation`: `comparison.png`, `validation-curve.png`
and `evaluation.json`. This does not update either checkpoint or the training log.

```powershell
.venv\Scripts\python.exe -u -m planet_training.evaluate_bridge
```

The selected update-10,000 intermediate model has been copied separately to
`outputs/intermediate/selected-10000/best.pt`. Use that folder with
`--checkpoint best.pt` to retain this selection independently of future training.
The original `run1/latest.pt` remains available to resume the stopped experiment.
An example generated from the completed continent model is saved under
`outputs/intermediate/evaluation-planet42`; its `intermediate/planet.tif` can be
used as the input elevation map for the existing meso model.

For a short first run, use `--steps 2000`; rerun with the full update limit to
continue. For a different experiment, always use a different run folder:

```powershell
.venv\Scripts\python.exe -u -m planet_training.train layout --run outputs/continents/run2 --lr 0.00005
.venv\Scripts\python.exe -u -m planet_training.train bridge --run outputs/intermediate/run2 --batch 8
```

On resume, network architecture and normalization come from the checkpoint.
The supplied learning rate takes effect. Changing the dataset, stage, or
condition-noise setting is rejected rather than silently mixing experiments.

## Adapt the intermediate to smooth continent maps

**Train Intermediate Adapted.cmd** prepares real surrounding guide context and
starts or resumes a separate experiment at `outputs/intermediate/run2-adapted`.
It initializes from the selected update-10,000 model using its EMA weights and
normalization, with a fresh optimizer. Original models and datasets are retained.
The default budget is 12,000 adaptation updates, CUDA batch 16, learning rate
0.00003, warmup 100, validation every 250 updates, and early stopping after five
checks without a new best. The budget is a ceiling, not a quality target.

Three quarters of the guides are randomly smoothed in physical metres, using
0.35–1.5 coarse-cell Gaussian sigma. One quarter remain clean. The training
target retains the real encoded terrain detail above its original bilinear guide,
then places that detail over the degraded guide. It therefore teaches the model
to add genuine relief without requiring it to undo the broad guide smoothing.

Filtering reads five extra coarse cells on each side sampled from the real
source terrain. It does not reflect the one-cell original margin. Geographic
holdouts cover the entire larger footprint. Original patch orientations are
recovered and verified against stored target cells; unrecoverable older examples
are excluded. This local preparation retained **5,248 training and 55 validation
patches**. Validate clean-guide behavior too; these larger-footprint validation
patches are a subset of the original held-out geography.

```powershell
.venv\Scripts\python.exe -u -m planet_training.prepare_adaptation
.venv\Scripts\python.exe -u -m planet_training.train bridge --data outputs/planet_training/data/bridge-adapted --run outputs/intermediate/run2-adapted --init-from outputs/intermediate/selected-10000/best.pt --guide-profile continent-smooth-v1 --steps 12000 --lr 0.00003 --warmup 100 --eval-every 250 --validation-patience 5
.venv\Scripts\python.exe -u -m planet_training.evaluate_adapted
```

The evaluator compares the original and adapted models using the same real
held-out guides, smoothing, and starting noise. It records clean, mixed and
strongly smoothed validation losses, terrain spectra, and sampled comparisons.
Its paired targets are real terrain detail over the smoother guide, rather than
unaltered Earth elevations. Run planet and regional meso checks before selecting
an adapted model for routine use.

## Adapt meso to smooth intermediate inputs

**Train Meso Adapted.cmd** starts or resumes `outputs/meso/run3-adapted` from
the original meso EMA at update 222,000, with a fresh optimizer and unchanged
normalization. CUDA batch 16, learning rate 0.00002, validation every 250 updates
and five-check early stopping are used. The 8,000-update budget is a ceiling.
The original meso checkpoint is retained.

Preparation reuses `outputs/meso/dataset.npz`, including its original geographic
splits. It selects 16,000 training and 512 held-out examples. Removing three
coarse cells from every side of the saved fine target produces 80x80 training
crops, with genuine saved guide context on every filter edge. No coarse ETOPO
cache is used as a substitute for 2.88 km targets. Guides use 0.25–1.0 coarse-cell
Gaussian sigma, with 25% clean examples; targets retain their original encoded
detail over the new guide. Inference still uses the original 128x128 tiles.

```powershell
.venv\Scripts\python.exe -u -m planet_training.evaluate_adapted --run outputs/meso/run3-adapted --reference outputs/meso/run2-lowlr/latest.pt --data outputs/planet_training/data/meso-adapted --clean-data outputs/meso/dataset.npz
```

This checks the selected 512 original full-size validation patches as well as
the cropped clean, mixed and strongly smoothed conditions. Compare regional
128-tile generation before adopting a cropped-training checkpoint. Greater
spectral power alone does not prove realistic drainage or valleys.

The first adaptation pilots are complete. The selected intermediate checkpoint
is `outputs/intermediate/selected-adapted-1750/best.pt`; the meso checkpoint is
`outputs/meso/selected-adapted-1000/best.pt`. A full adapted intermediate input is
saved at `outputs/intermediate/adapted-planet42/planet.tif`. Load
`outputs/meso/adapted-planet42-meso-launch.json` using the existing meso UI's
**Load settings** button for a fresh full meso generation. Regional before/after
comparisons and TIFFs are in `outputs/meso/adapted-both-planet42`. See
`docs/terrain-adaptation-fix.md` for measured improvements, clean-input regression
and the limits of this initial selection.

## Rebuild or resume preparation

**Prepare Planet Training.cmd** prepares both datasets and saves target previews.
It does not start training. Preparation was also run during setup of this workflow.
If interrupted, rerun it; intermediate patches resume from the last saved group.
Intermediate averaging uses four CPU threads by default; `--workers` changes
that number. Worker count does not change the selected examples or their values.

```powershell
.venv\Scripts\python.exe -u -m planet_training.prepare
.venv\Scripts\python.exe -u -m planet_training.train layout --check-only
.venv\Scripts\python.exe -u -m planet_training.train bridge --check-only
```

Prepared data lives under `outputs/planet_training/data`. Its manifests record
source files, scale, encoding, holdouts, averaging and sample counts.

### Continents

The 109 local PaleoDEMs are 181 x 361 **point** grids. Their importer respects
pole samples and duplicated dateline endpoints; it does not reinterpret them
as pixel-centred imagery. All maps were checked for coordinates, missing values
and dateline consistency. The included source license is CC BY 4.0; credit
Scotese and Wright (2018) when publishing work using these reconstructions.

The split is made by geological age before augmentation:

- **Training:** 67 maps.
- **Validation:** 18 maps at 100–140 and 280–320 Ma.
- **Test:** five maps at 0–20 Ma.
- **Excluded buffers:** 19 maps near validation and recent test ages.

Training chooses ages uniformly and generates fresh whole-sphere rotations and
reflections continuously. Training statistics use training ages only. ETOPO is
not included in layout training; modern ETOPO can serve as an additional final
reference. The test ages are not used for validation checkpoint selection.
On CUDA, spherical rotation, coordinate lookup and area integration run directly
on the GPU. CPU mode uses the original NumPy sampler. Both use the same age and
rotation RNG sequence, target footprints and area weights; float32 GPU arithmetic
can differ slightly from the CPU coordinate calculation. Resuming an existing
checkpoint needs no dataset rebuild.

Each example contains **all six faces of one rotated planet**. Faces are six
channels of a joint U-Net, so its attention and features see the full globe.
Shared edges and corners are identified differentiably during training and at
every denoising step. Noise has one independent draw per physical node. Loss is
weighted by spherical area, including partial boundary cells.

This is a joint-channel baseline, not a spherical-convolution network. It has
global context and identical shared edges, but generated global geography still
needs inspection for bias, fragmentation and memorization. Rotating a small
set of reconstructions does not create independent geological histories.

### Intermediate

The exact chain is **161.28 km -> 23.04 km -> 2.88 km**. The original proposed
156 km / 23 km ratio is not an integer; choosing 7x makes this bridge compatible
with the existing refinement and tiled-generation code. On the cubed sphere,
62 layout intervals give ~161.413 km actual nominal spacing; 434 intermediate
intervals give ~23.059 km. The existing meso sampler rounds to the same 434
conditioning intervals, so the rounded generation grids agree.

ETOPO is first streamed into a 0.05-degree raster using spherical row-area
means. The pipeline then takes 6 x 6 sub-samples per 23.04 km target footprint,
weighted by the gnomonic solid-angle Jacobian. Conditions are 7 x 7 area means
of target footprints, in metres before signed-square-root encoding.

There are **8,000 training patches and 512 validation patches**. Targets are
112 x 112 (~2,580 km across at their centre spacing); conditions are 18 x 18,
including one condition cell of margin on every side. Data stays in system
memory and is copied to the GPU one batch at a time.

Geographic validation uses two 3,500 km spherical caps centred on (55N, 20E)
and (45N, 115W). A training footprint cannot touch either cap; a validation
footprint must fit completely inside the held-out union. Footprints include
the condition margin, with a 25 km guard covering fine-cell extents. Mixed
footprints are rejected. Validation sampling is unbiased; training preferentially
retains land and coasts while retaining some deep-ocean samples.

The residual is measured above the bilinearly upsampled encoded condition,
matching the existing meso model convention. Independent geographic validation
shows whether this bridge generalizes outside the source regions.

## Use the trained models

Export a continent layout (choose a new output folder each time):

```powershell
.venv\Scripts\python.exe -m planet_training.generate --run outputs/continents/run1 --out outputs/continents/planet42 --seed 42
```

It writes `layout.tif` in metres, a preview, metadata and the raw six-face
elevation array. The default 512 x 256 export interpolates the coarse field;
it does not add learned resolution.

Then run the existing tiled generator with the new bridge checkpoint:

```powershell
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs/intermediate/run1 --checkpoint best.pt --dem outputs/continents/planet42/layout.tif --out outputs/intermediate/planet42 --seed 42 --steps 32 --batch 16
```

Finally use the intermediate `planet.tif` as the input map in **Launch Meso.cmd**,
with your existing `outputs/meso/run2-lowlr/latest.pt`, or use:

```powershell
.venv\Scripts\python.exe -m toy_meso.planet_tiled --run outputs/meso/run2-lowlr --dem outputs/intermediate/planet42/planet.tif --out outputs/meso/planet42 --seed 42 --steps 32 --batch 16
```

The intermediate stage uses the existing gnomonic-patch training / equi-angular
cube inference convention. Like the meso model, its physical spacing varies
within a cube face; this baseline does not add a spacing embedding. The bridge
uses a 7x factor and 112-pixel tile, whose default stride of 84 is also a valid
multiple of seven. Existing meso checkpoints and their UI remain compatible.

Before treating a final continent model as successful, compare land fraction,
continent sizes, elevation distributions, and rotation-aligned similarity to
training maps. For the bridge, also compare wavelength-band power and averaged
guide fidelity on held-out regions. Training loss alone does not establish this.
