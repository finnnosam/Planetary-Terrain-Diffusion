# Training data and models for planetary scale

Companion to `claude/scale-review.md`. Written 2026-10-04. Time estimates use the RTX 3070 benchmark in `outputs/training-estimate-2026-10-02/benchmark.json`: a 16M-parameter EDM U-Net trains at 112 patches/s at 64×64 and 28 patches/s at 128×128.

## Core idea

Each new model learns Earth **at one spacing**. Its training targets are real elevation data averaged down to exactly that spacing. Its condition is the same data averaged further, to the level above. Because the model never sees any other spacing, every feature it draws has a size that is correct at that spacing.

Two new models fill the gaps around the existing 23 km → 720 m → 90 m stack:

| Model | Spacing | Window | Condition | Role |
|---|---|---|---|---|
| L: layout | ~156 km | one cube face ≈ 10,000 km (64×64) | latitude only (unconditional elevation) | continents and ocean basins |
| M: meso | 2.9 km | 128×128 ≈ 370 km | 23 km coarse field (6 channels), noise-augmented | correct-size global map; guide for zooms |

The existing coarse model sits between L and M. It runs at true scale, conditioned on L's output upsampled to 23 km.

## Data

| Source | Use | Notes |
|---|---|---|
| ETOPO 2022, 30″ ice-surface GeoTIFF (NOAA NCEI) | M targets and conditions; present-day Earth for L | ~0.93 km, land and bathymetry in one grid; 15″ and 60″ also available, plus bedrock versions |
| WorldClim 2.1 30″ bio1, bio4, bio12, bio15 | climate channels of M's condition (same 6 channels as the coarse model) | already used upstream |
| Scotese & Wright 2018 PALEOMAP PaleoDEMs | L training set | 117 global paleo-topographies, 0–540 Ma at 5 Myr steps, 1°×1° (0.1° on Zenodo). Real continental layouts other than today's. Check the Zenodo license before use. |
| Tectonic-generator planets (scale-review layer A) | optional extra L data | adds variety; the model inherits their flaws, so keep them as a minority |
| GEBCO grid | optional bathymetry cross-check | — |

Storage: ETOPO 30″ is ~3.7 GB as float32. The preprocessed patch stores are a few GB. The 50–100 GB allowance in the earlier training review is ample.

## Preprocessing (shared sphere sampler)

1. **Sample patches on the sphere, not on lat/lon.** Pick an area-uniform random centre and a random rotation. Build a local equi-angular grid at the exact spacing h; reuse `cube.py` chart math so training matches inference. This covers the poles, which upstream excludes beyond ±60°.
2. **Area-average, never point-sample.** A 2.9 km target cell is the mean of ~9–10 ETOPO 30″ cells. The 23 km condition is the mean of the 8×8 target cells below it, plus the p5 channel computed the way upstream builds coarse data. Point sampling would put aliased sub-pixel detail into the targets, which is the oversized-feature failure again.
3. **Encoding** as upstream: signed square root of metres. For M, predict the residual over the upsampled condition (Laplacian style), so the model learns only the 23 km → 2.9 km detail.
4. **Sampling weights:** oversample land and coasts about 3:1 over open ocean, as upstream favours land crops.
5. **Geographic holdout before augmentation:** tile the sphere into ~1,000 km cells and hold out ~10%. Include named test regions such as Norway, Patagonia, British Columbia, the Himalaya and an abyssal plain. No training patch may overlap a held-out cell. For L, hold out several paleo time slices plus present-day Earth.
6. **Write to HDF5** (upstream's loaders use h5): ~200k–500k precomputed M patches, regenerated with fresh rotations every few epochs if disk allows.

### Augmentation

- **M:** rotations and flips only. The latitude and climate channels travel with the patch.
- **L:** random whole-sphere rotations of each map and mirror images. If L also predicts climate, keep rotations about the polar axis only and mirror north–south only, so climate stays consistent with latitude.

## Models and training

Both use upstream `edm_unet.py` (64 base channels, multipliers [1,2,4], two blocks per level, bottleneck attention, ~16M parameters). Each needs a new dataset class and config registered with upstream `terrain_diffusion train`. Both fit in 8 GB.

### M (meso, 2.9 km)

- Input: noisy residual plus upsampled 23 km condition (6 channels). Take the condition crop with ±2 coarse cells of margin, so the model sees context beyond its target.
- **Condition noise augmentation** (upstream's cond_snr idea): at inference the condition comes from the coarse diffusion model, not true averages, so train with jitter and noise on it. Optionally fine-tune on conditions produced by the coarse model itself.
- Schedule:
  - smoke test: 10k updates at 64×64, ~1 h
  - real run: 100k updates at batch 32 and 128×128, ~32 h (or ~8 h at 64×64 with less context)
  - optional consistency distillation to 2 steps: ~1 more day
- Inference: about 31k patches per step over the whole globe through the existing `cube.consensus` / `sparse.py` machinery. This is a new stage between coarse and latent.

### L (layout, ~156 km)

- At 64 intervals per cube face, a face is exactly one 64×64 training window (90° ≈ 10,000 km). Training examples are faces of rotated planets with halo from the neighbouring faces. Inference uses the existing six-face consensus, so no new architecture is needed.
- Data is scarce: 117 paleo maps plus today. Consecutive 5 Myr slices are highly correlated, so this is perhaps 15–30 effectively distinct layouts. Mitigate with:
  - a smaller model (32 base channels)
  - heavy rotation augmentation
  - early stopping on held-out slices
  - a minority of tectonic-generator planets
- Training: 20–50k updates, a few hours. Curating the data dominates the time.
- **Memorization check:** for each generated planet, rotation-align it to every training map and report the best overlap of land masks. Near-copies of a paleo-Earth mean overfitting.
- PaleoDEMs are reconstructions: smooth, with stylized mountains, and older slices are more schematic. Use L for layout only, i.e. land/ocean, shelves and broad highlands. Let the coarse model and M supply real relief.

## Evaluation (scale-aware)

On held-out regions, at each model's spacing:

- **Radially averaged power spectrum** against real data, per wavelength band. This is the direct test for oversized features: excess power at wavelengths of a few pixels means features are too large or too strong.
- Coastline indentation, measured as fractal dimension or coast length ratio at 2.9 km, on held-out fjord coasts.
- Hypsometry and bathymetry histograms, and the slope distribution at that spacing.
- Widths of valleys, ridges and inlets in pixels (fjords should be ≤ 1–2 px at 2.9 km).
- For L: land fraction, the size distribution of land masses, shelf fraction, and the memorization check.

Validation loss alone doesn't establish quality.

## Order and rough timeline

1. Sphere sampler, ETOPO and WorldClim preprocessing, holdout, HDF5 store: 2–4 days of development.
2. M smoke run, then the full run: ~1–2 days of GPU time. Integrate as a cube stage and compare global 8k maps against pooled or overview output.
3. Feed M's output into the latent stage as a low-frequency guide, so 90 m zooms match the global map.
4. L: PaleoDEM import (1° text/netCDF), rotated-face dataset, training, memorization check. A few days, mostly curation.
5. Later, optionally: one scale-conditioned model trained with the same sampler. Draw h log-uniformly from 90 m to 25 km and add a spacing embedding. It needs finer data (Copernicus GLO-90/GLO-30, tens to hundreds of GB) and much more GPU time.
