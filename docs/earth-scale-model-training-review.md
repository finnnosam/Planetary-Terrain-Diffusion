# Earth-scale map model: upstream review and local training estimate

Reviewed October 2, 2026. Local upstream checkout: `e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230`. Current upstream documentation was also checked. This is a feasibility estimate, not a trained-model quality result.

## Finding

An Earth-radius map with believable continents at a manageable point count needs a model trained on kilometre-scale targets and continental context. Reusing the existing 90 m or 30 m outputs at full physical scale is far too expensive for a complete raster. Enlarging their features changes physical scale; reducing the export dimensions does not reduce the internal generation workload.

The practical experiment is a new continent-layout model, followed by one or two new conditional refinement models that finish at roughly 5–10 km per point. The existing high-resolution models can optionally serve local zooms later. They do not need to be part of this experiment.

## What upstream actually recommends

The [upstream README](https://github.com/xandergos/terrain-diffusion#modifying-world-generation-advanced) describes three affordable customization routes: modify `synthetic_map.py`, retrain the coarse model using modified sampling/data/model settings, or pool its outputs. For coarse training it calls for ETOPO 30-arcsecond elevation and WorldClim 30-arcsecond bioclimatic rasters, then training and exporting an EMA checkpoint.

The [coarse configuration](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/configs/diffusion_coarse/diffusion_coarse.cfg) specifies 16×16 training crops, 128 channels, one resolution level, batch 256, 150 epochs and 1,024 batches per epoch. This totals 153,600 optimizer steps and 39,321,600 crop presentations. The loaded model has about 2.8 million trainable parameters.

At the 90 m model's 23.04 km coarse spacing, a 16-cell crop spans about 369 km. The 30 m model's equivalent is about 123 km. Larger inference windows do not add training examples of whole continents. This explains why changing the coarse model's style alone does not reliably establish continent-scale organization.

The [dataset implementation](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/training/datasets/coarse_dataset.py) resamples latitude bands to reduce physical scale distortion, builds six channels, and samples randomly augmented crops. It excludes latitudes beyond ±60°. The supplied coarse training and validation configurations use the same cached Earth data; a custom project needs a stronger geographic holdout and polar coverage.

Full training also includes an autoencoder, latent diffusion, decoder diffusion and consistency distillation. [TRAINING.md](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/TRAINING.md) gives 24 GB GPU memory for the complete workflow and 18 GB for the autoencoder. The [paper, Appendix E](https://arxiv.org/html/2512.08309v4#A5) reports approximately one week on an RTX 5090, and about one day for base fine tuning when reusing the decoder. These are not RTX 3070 measurements.

## Physical scale and output size

Use radius 6,371 km and equatorial circumference approximately 40,030 km. The following are 2:1 equirectangular exports; ground spacing varies with latitude. A cube or equal-area internal representation has a different point count and should be resampled to these exports.

| Export | Equatorial spacing | Points | Elevation float32 only |
|---|---:|---:|---:|
| 256×128 | 156 km | 32,768 | 0.125 MiB |
| 1,024×512 | 39.1 km | 524,288 | 2 MiB |
| 4,096×2,048 | 9.77 km | 8.39 million | 32 MiB |
| 8,192×4,096 | 4.89 km | 33.55 million | 128 MiB |
| Approximately 444,780×222,390 | 90 m | 98.9 billion | 396 GB |
| Approximately 1,334,339×667,170 | 30 m | 890 billion | 3.56 TB |

Last two storage estimates use decimal GB/TB and exclude climate, model intermediates, overlap caches and checkpoints. They include equirectangular polar oversampling; they are not equal-area surface pixel counts.

## Proposed experiment — design targets, not upstream guarantees

1. Generate continent/ocean layout at approximately 100–200 km per point. At 156 km spacing, a 64×64 training patch covers roughly 10,000 km, enough to see continent and ocean-basin relationships. Train with cross-face context or another spherical sampling method, including seam and polar examples.
2. Refine to approximately 40 km per point, conditioned on the planetary layout. A 64-cell crop then covers about 2,500 km.
3. Refine to approximately 10 km per point; optionally add a final 5 km stage. The upper-level land structure must remain a strong condition so local noise cannot break continents into islands.

These stages require new physical-scale metadata and sampling/adapter support. They cannot be plugged into the current fixed 256× coarse-to-output relationship without changes. Larger context is also an architectural change: simply increasing crop dimensions on the shallow upstream coarse model is insufficient.

An economical candidate per stage is an EDM U-Net with 64 base channels, multipliers [1,2,4], two blocks per level and bottleneck attention. This was measured at **15.84 million trainable parameters**. Start with 64×64 patches and an effective batch of 32, using microbatches of 8 and correct gradient accumulation. Checkpoints at 10,000, 50,000, 100,000 and 200,000 optimizer updates give approximately 0.32, 1.6, 3.2 and 6.4 million crop presentations. The script must count accumulated optimizer updates correctly; the supplied epoch counter counts incoming batches.

Use ETOPO topography/bathymetry, and the four WorldClim variables if joint climate prediction is wanted. The currently bundled 10-arcminute rasters are 2,160×1,080, roughly 18.5 km at the equator: adequate for an initial continent experiment, too coarse for truthful 5–10 km training targets. Download the finer rasters for refinement.

Five full global 30-arcsecond rasters contain about **18.7 GB** at float32 before compression. Stream preprocessing instead of loading every channel at once. Reserve **50–100 GB** as an initial disk allowance for selected raw data, transformed patches, experiments and checkpoints; downloading all bioclimatic bands can require more. There is approximately **1.26 TB free** on E: at review time.

The critical data limitation is that there is only one real Earth. Millions of rotated or overlapping crops are not millions of independent continents. A starting diversity experiment could combine Earth crops with **1,000–10,000 distinct synthetic planetary layouts**, with varying land fractions, continent sizes and ocean widths. This is an experimental range, not a proven minimum. Synthetic maps must be curated or generated by a credible structural method; otherwise the model learns their defects. Hold out geographic regions and synthetic seeds before augmentation. Climate augmentation must preserve latitude relationships.

Evaluate physical continent sizes, connected land-area distribution, small-island area fraction, mountain-belt widths, elevation/bathymetry distributions, and spherical seams/poles. A default land fraction around 30% is a useful controllable target, not a substitute for these tests. Validation loss alone will not establish success.

## This machine and measured throughput

- NVIDIA GeForce RTX 3070, 8 GB VRAM; native bfloat16 supported.
- AMD Ryzen 7 9800X3D, 8 cores / 16 threads.
- 31.1 GiB usable system memory, consistent with 32 GB installed.
- Project environment: PyTorch 2.14.0+cu130. Training-only dependencies are not installed yet.

The short local probe used the actual upstream model implementation, diffusion loss, backward pass, gradient clipping, Adam and two EMA parameter copies. It used bfloat16 arithmetic with float32 parameters. Each measurement followed four warmups and timed sixteen microbatches. It saved no model weights. Inputs were synthetic and already on the GPU: preprocessing, loading, validation, exact upstream EMA implementation and checkpoint I/O are excluded. Results establish throughput and memory feasibility, not convergence.

| Model / crop / microbatch | Seconds per batch | Patches per second | Peak allocated VRAM |
|---|---:|---:|---:|
| Existing coarse / 16×16 / 256 | 0.0878 | 2,916 | 1.44 GiB |
| New candidate / 64×64 / 8 | 0.0711 | 112.5 | 0.945 GiB |
| New candidate / 128×128 / 2 | 0.0713 | 28.0 | 1.32 GiB |

Allocator reserves, display memory and other processes increase total GPU use beyond this table. There is room to experiment with larger batches; these estimates do not assume that optimization.

At the measured 64×64 speed, 100,000 effective-batch-32 updates require 3.2 million patches, or **7.9 GPU hours**. Allow approximately **10–16 hours per stage** with 25–100% workflow overhead. A 200,000-update attempt is approximately **20–32 hours per stage**. For 128×128 patches, the corresponding 100,000-update estimate is **31.7 GPU hours**, or about **40–64 hours including overhead**.

| Work | Planning estimate on this machine |
|---|---|
| Existing coarse model, complete original schedule | 3.75 GPU hours; budget 5–10 hours with optimized loading, excluding initial data preparation |
| New continent model, 10,000-update smoke experiment | 0.8 GPU hours; budget 1–3 hours |
| New continent model, 100,000–200,000 updates | About 10–32 hours per attempt with 64×64 patches |
| Three new stages, 100,000–200,000 updates each | About 2–5 days of continuous training per complete attempt |
| Larger 128×128-context experiment | About 2–3 days per stage at 100,000 updates |
| Dataset setup, integration and several quality iterations | Plan 1–3 weeks of elapsed development; data creation may dominate |

The training-time bands are estimates, not promises of decent quality after a particular update count. Full retraining of the original high-resolution stack is not a supported unchanged 8 GB workflow. Its existing latent checkpoint alone is about 254 million stored values; optimizer state, gradients, EMA copies and activations are substantially larger than inference memory. A credible completion estimate would require a separately designed smaller model or memory-saving workflow.

## Recommendation

Target **4,096×2,048 at roughly 10 km per point first**, then **8,192×4,096 at roughly 5 km**. Train the continent stage before committing to refinement. If its held-out maps still consist of dispersed islands, improve the global training data and conditioning; increasing local detail will not fix the layout. Budget roughly one day for a serious first continent training run and 2–5 days per complete hierarchy attempt. Expect the dataset and quality iteration to be harder than fitting this candidate into GPU memory.

Raw timing evidence: `outputs/training-estimate-2026-10-02/benchmark.json`.
