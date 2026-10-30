# Intermediate structure pilot

This is the first isolated rework following the terrain structure diagnosis. It changes the intermediate adaptation target, not the reviewed coast/relief pair or meso selection. The new `continent-structure-v1` profile smooths guides with exactly the existing mixture/filter (25% clean, 75% Gaussian-smoothed), while keeping original modern-Earth elevations unchanged. The legacy `continent-smooth-v1` and `meso-smooth-v1` profiles retain their original baseline-translation behavior.

Earth targets keep sea level at zero. There is no global elevation shift, land-percentage correction, histogram remapping or relief multiplication. The pilot can change local terrain and shore geometry; review broad-height drift, shelves and clean-input regression before considering selection.

## Launch the pilot

Double-click **Train Intermediate Structure Pilot.cmd**, or run from PowerShell:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\Train Intermediate Structure Pilot.cmd'
```

The launcher prepares a small manifest with read-only references to the existing verified bridge-adapted arrays. It does not rebuild or duplicate the dataset. Its source manifest and array fingerprints are checked on every load. Original target arrays and geographic holdouts are preserved.

Training warm-starts from the frozen `selected-adapted-1750/best.pt`, whose hash is checked before launch. A fresh optimizer, validation baseline and EMA schedule are used. Defaults: **500-update ceiling**, CUDA batch 16, learning rate 0.00003, 100-update warmup, EMA 0.999, condition noise 0.03, seed 42, validation/previews every 100 updates, checkpoints every 50, progress every 10.

Expect several minutes; the console reports rate and ETA once training starts. It includes starting-model validation and clean-guide checks. Ctrl+C saves `latest.pt`; rerunning the same command resumes. A run lock prevents simultaneous trainers. Resume rejects changed objective, data, learning rate, warmup, EMA, condition noise or seed. Batch can be reduced, for example with `--batch 8`.

Outputs stay under **`outputs/intermediate/structure-pilot-v1`**:

- `training-job.json` and `launch.json`: initialization provenance, effective arguments and dataset identity.
- `log.jsonl`: durable update/validation/memory records.
- `latest.pt`, `best.pt`: candidate checkpoints, not promoted selections.
- `preview-000100.png` through `preview-000500.png`: bounded held-out previews.
- `smoke-test/`: separate developer checks; does not count toward the 500-update pilot.

Nothing starts automatically after training. In particular, the launcher does not run a full intermediate globe or meso globe.

## Launch the regional review after training

```powershell
& '.\Review Intermediate Structure Pilot.cmd'
```

This is a separate bounded evaluation/generation command. Expect a few minutes, depending on validation speed. It checks the original complete 512-patch clean holdout and 55-patch mixed/strongly-smoothed holdouts, comparing both models against the **same unchanged Earth targets**. The default candidate is `latest.pt`; `--checkpoint best.pt` reviews that candidate instead. The recorded step matters because a loss-selected best can remain at initialization.

Seven regions (Alps, California, Great Plains and four fixed generated windows) use four seeds, 32 sampler steps and identical initial noise between baseline and candidate. Intermediate patches are 112 × 112 at 23.04 km. Each generates a central 128 × 128 meso patch at 2.88 km with the same frozen meso model. These are north-up diagnostic patches, not production sphere-tiled generation.

Each review writes a fresh timestamped `review-*` folder inside the pilot run. It freezes baseline, candidate and meso checkpoints, records hashes and the fixed layout hash, and saves:

- Seven-by-four baseline/candidate PNG comparisons, each with its physical extent and a common colour/shading scale. Intermediate panels show about 1,198 km; meso panels show the central 369 km.
- Raw elevation transects without vertical exaggeration.
- Intermediate/meso arrays and incremental `regional-review.json` measurements, including slope percentiles, threshold-defined highland widths, actual coarse metre-average drift, mean heights and land fraction.
- `review-settings.json` and `README.txt` explaining scope and interpretation.

Review width and transition length into compact belts, intervening valley depth/continuity, secondary-range separation, plains, coastline and shelves. Higher peaks, finer noise or lower loss alone do not establish success. Threshold widths are descriptive transect measures, not an automatic mountain classifier. Coarse-average drift and clean regression expose the risk that recovering shape also rewrites too much of the broad guide.

No review automatically selects a checkpoint or launches a globe. An active trainer blocks snapshot creation; finish or stop training first.

## Checks completed before handoff

The unchanged-target/sign behavior, legacy behavior, read-only array reuse, changed-source detection, warm-start/resume guards, physical drift measurements and run locking were checked. Existing training/adaptation tests passed. A real CUDA smoke test made **two updates in a separate folder**, peaked at about **4.48 GiB**, and saved/reloaded candidate checkpoints. The review preflight and a two-region, one-seed, eight-step review smoke test also passed. Those smoke results establish execution, not terrain quality. All four selected coast/relief/intermediate/meso checkpoint hashes remained unchanged. The 500-update pilot was left for the user to launch.

The user explained the historical `B2W.png` hash difference: near-coast ocean colours were edited to make the water shallower. The original diagnosis/provenance records now include that explanation; historical planetC comparisons still use its saved coarse cache.
