# Dedicated terrain planning comparison

The frozen-model influence probes found meaningful but limited plan-to-height control in the joint model. This experiment tests a stronger separation: generate a plan from the coarse guide, then use the **same frozen plan-conditioned renderer** for every planning method.

It is an unselected regional experiment. If adopted, separate planning/rendering would add a learned pass within the intermediate role. That compute/stage-count tradeoff is real; this experiment does not establish that production should gain another stage. It compares plan-generation tasks under a common renderer, not complete equal-cost production architectures.

## Matched starting point and task change

Both branches use the same five-channel architecture and the identical frozen joint EMA from `outputs/structure/joint-structure-v1/review-20261014-143545-003794/joint-checkpoint.pt`, SHA256 `42b0fa2ed2c4f472021cfeeba417a1c09f1994a6e842cbdfebaed1fada4558d0`. New optimizers and EMA schedules are used. A single shared mini-batch, augmentation, guide degradation, sigma vector and five-channel Gaussian noise field is drawn for each pair of updates.

- **Coupled continuation** retains the previous height + 0.25 × plan + physical shape objective exactly, including the noisy height state.
- **Dedicated planner** zeros the noisy height channel during training and every sampling evaluation. Its loss is solely the four-channel plan denoising objective, weight 1. The height output is unused and receives no direct loss. It cannot recover a training plan by reading a noisy modern target-height field.

Both receive the same coarse height and permission conditioning. Modern-Earth-derived plans are training labels only. At generation, plans start from noise. The morphology definitions, feasible targets, 40.32 km grid, geographic split and low-elevation protection remain unchanged.

Removing a shortcut and focusing optimization change together. This is a test of the **dedicated planning task**, not an isolated claim about one loss weight, masking choice or capacity change. Common-renderer comparisons also hold the extra rendering network fixed; its learned capacity is not credited as an improvement in the planner.

## Bounded run completed

The first paired pilot completed **1,000 additional updates per branch in 85.7 seconds**. The combined job had a 100-second wall-time budget, including setup and validation before its final save, with a short reserve for saving/validation. It was launched under the user's authorization for jobs taking a couple of minutes. No long training or planet run starts automatically.

Outputs are under `outputs/structure/plan-first-v1/paired-1000`. Both branch states, optimizer states, EMA states and random states are saved atomically in one resumable `latest.pt`, so saved update counts remain paired. Ctrl+C is deferred to a pair boundary. Run locks and strict task/data/source/optimizer checks protect resume. A wall-time stop can leave fewer than the requested steps; inspect `training-result.json` before comparison.

Loss validation uses all 55 existing holdouts with the same smoothed-guide recipe, six noise levels and common random draws. These are monitored development holdouts from two geographic caps, not a world prevalence sample or fresh statistical confirmation. Loss reduction is not terrain-quality approval.

## Shared-renderer regional review

The predeclared review uses Alps, California, Rockies/Plains, the formal rugged/quiet holdouts and a saved genuinely generated coast/relief prefix. Plan seeds are 42, 314, 2718 and 2026; renderer seeds are 7919 and 104729. Original joint plans are retained as a frozen baseline.

Every planning method feeds the same frozen oracle renderer, hash `bd5e1fc86bbe835d5b7f96d73736e780bb32d9456f26e506a285845ade0ceb6f`. On Earth-derived cases, supplied feasible reference plans are rendered once per renderer seed as labelled references. They are not autonomous successes. The generated prefix has no invented Earth reference.

Fine-stage evaluation is fixed before inspection: all six cases at first plan seed 42 and first renderer seed 7919, with meso seeds 113 and 997. The selected frozen meso hash remains `717e1c718fefe51fc0f5066de39618da9cc5a6b034c9df0fb1af09ebcbd7fcf6`. All other generated plan seeds and both renderer seeds have saved structural outputs and expandable comparisons. This deliberately bounds fine sampling rather than generating a planet.

The viewer shows one primary raw-meso comparison, one alternative meso seed, and expandable guide/seed evidence. Shared references appear once per region. Exact duplicate fields are labelled rather than redrawn. Raw/projected/handoff arrays remain saved; redundant engineering columns are not the default visual presentation.

Guide panels share the same 1,198.08 km context; fine panels share the central 368.64 km footprint. Elevation colours are fixed at −6,000 to +6,000 m, with zero datum and NW 45° illumination at 5× shading. Formal holdouts retain their stored frames. Compare compact belts, connected lowlands, secondary systems and transitions; plan agreement and roughness alone are not quality measures.

The review records reused frozen baseline outputs explicitly and checks exact guide identity before reusing their fine results. Repeats and caches are reproducibility controls, not independent samples. The whole regional review has a separate short wall-time budget and no automatic follow-on job. The first execution reached its conservative 110-second budget guard after 22/24 groups at 96 seconds. Those saved groups were preserved, and an explicit resume finished the remaining two plus figures in 16 seconds. No sample was discarded because of its appearance. Future review defaults allow 120 seconds including a reserve for figures; a budget-limited run prints resume instructions and returns exit code 2.

## Commands and logs

These commands have already been run for the first bounded comparison. They do not need to be repeated to inspect it. Repeating training in the same folder resumes to the completed ceiling, while review needs a fresh folder.

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_plan_first --steps 1000 --max-seconds 100
```

The separate review command is:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_plan_first --max-seconds 110
```

If a bounded review stops before all groups finish, use its printed output directory:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_plan_first --resume --out 'THE-EXISTING-REVIEW-DIRECTORY' --max-seconds 60
```

Resume verifies the snapshot identity and reuses saved fields rather than regenerating completed groups. The original 110-second command above records the executed first run; 120 seconds is the current default for a fresh review.

Each prints a fresh durable console/error log under `outputs/structure/plan-first-v1/job-logs`. In another PowerShell window:

```powershell
Get-Content -LiteralPath 'THE-PRINTED-CONSOLE-LOG-PATH' -Tail 30 -Wait
```

Any substantial extension must be a separately declared comparison, with source snapshots, equal budgets and a reason supported by terrain evidence. Do not extend repeatedly merely because a plan loss is falling.

## Verification and limits

Twelve targeted tests passed before training, including dedicated-planner independence from target/noisy height, exact preservation of the old coupled objective with shared draws, zero planner height-state during generation, deterministic seeds, prior joint-task checks and the earlier influence-probe invariants. A separate two-update paired execution check and a two-region shared-renderer/meso review check completed before the substantive bounded comparison.

The inherited low-height lock still imposes quietness and suppresses some lowland/coastal geometry. The experiment does not isolate the PaleoDEM/modern-Earth mismatch or measure the frequency of good terrain across a world. PlanetC remains supplied detailed Earth-guide refinement, not an autonomous round trip.

Sea level remains **zero**. No global elevation shift, forced land coverage, relief multiplication or histogram matching is introduced. Selected production checkpoints are preserved.
