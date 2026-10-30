# Regional 40-km terrain structure prototype

**Completed-pilot update:** The 2,000-update pilot and 28-case review completed. The candidate was not selected: belt/valley arrangements remain strongly seed driven. The review also exposed a meso postprocessing grid, now fixed by continuous footprint-mean correction. The reviewer defaults to `smooth-v1`; `--meso-constraint block-v1` reproduces the original review correction. This changes meso review postprocessing, not the existing structure checkpoint or training objective. See [the assessment and repeated-pattern diagnosis](../outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544/analysis/assessment.md) before launching more training or a globe.

The earlier 500-update intermediate target pilot was not selected. It steepened and raised terrain without reliably recovering compact belts or valley/range separation, and increased broad-height and shelf drift. This new experiment adds a separate learned structural field at **40.32 km**, ahead of the frozen 23.04-km intermediate detail and 2.88-km meso model. It is a regional prototype; no planet sampler or automatic selection has been added.

## Run these three jobs separately

From PowerShell in the project:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\Prepare Terrain Structure.cmd'
```

Preparation creates `outputs/planet_training/data/terrain-structure-40-v1`. It converts the existing verified modern-Earth 23-km training cell means into 40-km cell means in physical metres, using exact interval overlap weights and the spherical Jacobian. It never averages signed-square-root encodings. Its 5,248 training and 55 held-out patches retain the original geographic split and verified surrounding context. Condition/context arrays are reused read-only; about 87 MB of new target arrays are written. Source manifest identity, array fingerprints and completion state are checked. Preparation reports count/rate/ETA and safely resumes committed batches when rerun. It does not rebuild the large ETOPO source cache. Counts come from the verified source, not a new independent Earth history.

After preparation finishes:

```powershell
& '.\Train Terrain Structure Pilot.cmd'
```

This starts a **new model from scratch**, with a **2,000-update ceiling**, CUDA batch 16, 3.60 million parameters, 32 base channels, learning rate 0.0002, 200-update warmup and seed 42. It does not initialize from, alter or select any reviewed checkpoint. Conditions use the existing 25% clean / 75% smoothed mixture with real context. The model conditions on interpolated coarse elevation and the permitted-interior mask. Training combines residual denoising with physical 40/80-km slope and curvature errors against constrained modern terrain targets.

Expect several minutes of training. Live rate and ETA exclude intermittent validation/preview overhead. Progress prints every 25 updates, checkpoints every 100 and complete held-out checks/previews every 250. Peak GPU memory in the two-update batch-16 smoke test was about **0.77 GiB**; that is a smoke measurement, not a promise for other configurations. Ctrl+C saves `latest.pt`; rerunning resumes. The resume guard checks representation, objective, dataset identity, architecture and optimization settings. A run lock prevents simultaneous training or review snapshots. Batch size may be reduced with `--batch 8`; the default command resumes a default run with unchanged settings.

Training outputs are under `outputs/structure/terrain-40-pilot-v1`: `launch.json`, `log.jsonl`, `latest.pt`, `best.pt`, previews and `training-result.json`. `best.pt` is selected by this new objective's smoothed holdout loss, not morphology approval. New and legacy losses use different representations and are not directly comparable. Checkpoints save RNG and optimizer state for resume.

After training finishes:

```powershell
& '.\Review Terrain Structure Pilot.cmd'
```

Expect a few minutes for this separate regional generation/review job. It uses the latest candidate, all seven fixed Earth/generated windows, four paired seeds and 32 sampling steps. It prints region/seed count and ETA. `--checkpoint best.pt` can explicitly review the loss-selected checkpoint instead. Each run gets a fresh timestamped review folder and frozen snapshots/hashes of the candidate and selected intermediate/meso models. The frozen layout hash is also checked. An active trainer blocks snapshot creation.

Every launcher saves its console output and errors to a fresh file in **`outputs/structure/job-logs`** and prints that path immediately. No follow-on job starts automatically. If a job fails, its logged exit/error remains available; avoid closing the window before checking its status.

## What this stage changes

The structural model emits 64×64 cells at 40.32 km, covering exactly the same 2,580.48-km tangent-plane footprint as the existing 112×112 intermediate patches. Its elevation correction is passed to the 23-km grid in metres. The frozen intermediate contributes only detail shorter than the 80.64-km Gaussian filtering scale; its older broad correction cannot simply replace the new structural field. This is a deterministic regional handoff, not retraining the intermediate network to ingest a new channel and not averaging the structural field back to 161 km.

Sea level remains **0 m**. At each constrained stage, input ocean/shelf terrain and input terrain at or below 200 m retain their elevations; allowed correction tapers between 200 and 600 m in target construction. Redistribution cannot make permitted land cells ocean. Shallow-input protection also freezes already-low inland plains: this is a deliberate first-prototype limitation to review, not an assertion that all lowlands should be immutable forever. The modern-Earth training target is therefore constrained near shallow terrain and in regional mean; it is not globally unchanged raw Earth.

The structural field and 23-km handoff preserve Jacobian-weighted means over **645.12-km regional blocks**, relative to their interpolated input baselines. They do **not** preserve every original 161-km coarse value. Smaller-scale shifts are permitted to arrange a compact belt beside a basin and are measured separately. Initial Earth-reference tests showed that conserving every 161-km block excessively suppresses the Alpine reference; the larger scale retains clearer compact geometry. At meso, the constrained branches preserve the supplied 23-km footprint means. These are local redistribution constraints, not a planet-wide sea-level shift, land-percentage adjustment, histogram remapping or blanket relief multiplication.

The 40-km structural baseline interpolates physical coarse metres; the frozen intermediate retains its original encoded interpolation. Their difference fields are transferred on explicitly aligned physical grids. Protection/sign guarantees are relative to each stage's own input grid; later interpolation can move finer shoreline crossings inside an input cell. The regional prototype has no global seam/blending guarantees. Regional block boundaries and target clipping can introduce visible artifacts; they must be reviewed rather than hidden by passing conservation checks.

## Review and acceptance

The review includes four branches where Earth targets exist:

1. **Original:** unchanged selected intermediate/meso pipeline.
2. **Control:** filtered old intermediate detail and the same coastal/conservation rules, without a learned structural field.
3. **New stage:** learned 40-km structure plus exactly the control's frozen shorter-scale detail and constraints.
4. **Earth oracle:** known modern 40-km structure through the same handoff, as a representation reference; it is not a learned success. Generated regions have the first three branches only.

Compare **new stage to control** to isolate learned structure. Original-to-control differences identify the effects of filtering and constraints. Oracle-to-control shows what the representation can carry. Original modern Earth profiles remain a reference; generative samples need convincing belt/basin relationships, not identical Earth feature placement.

All structural/intermediate panels show the same 1,198-km square. The first column displays 40-km fields resampled onto the matched 23-km display grid; original/control show the coarse guide. Meso panels show the central 369 km, explicitly labelled. Elevation colours and NW 5x hillshading are fixed. Raw elevation profiles, input/output arrays, loss checks, physical slopes, absolute-threshold spans, coarse drift and protection/conservation checks are saved in `regional-review.json`. A passing constraint check does not establish realistic terrain.

Acceptance requires compact belts with shorter transitions, continuous intervening lowlands and reliable secondary-range separation across seeds, gentle plains, acceptable 161-km drift and no obvious block/grid artifacts. Check the Alps, California, Great Plains, generated broad highland, upland, plain and coast together. Do not select a candidate because it is taller, noisier or has lower denoising loss. No reviewer promotes a checkpoint, edits `structured-selection.json`, or launches a full globe.

## Checks before handoff

Targeted checks passed for noninteger conservative conversion, immutable source reuse, preparation resume/source-change detection, protected elevations, fixed sign, regional means, differentiable training, structural handoff, training resume and locks. Existing intermediate/adaptation tests also passed. Real smoke preparation used only 16 train / 4 held-out patches in `outputs/structure/checks/data-smoke`; real CUDA smoke training used **two updates** in `train-smoke`; the eight-step review smoke covered California and a generated highland in `review-smoke-fixed`. These verify execution, not terrain quality.

The CPU-only Earth-reference representation/handoff test can be reproduced with:

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.check_structure_representation
```

It writes matched close-ups and constraint measurements to `outputs/structure/checks/representation-v1`, without learned-model sampling or training. The production dataset preparation and 2,000-update pilot were left for the user to launch, in accordance with the runtime preference.
