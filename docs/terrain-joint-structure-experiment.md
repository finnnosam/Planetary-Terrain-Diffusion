# Autonomous structural intermediate pilot

This experiment tests the next missing capability after the [oracle-plan review](../outputs/structure/plan-oracle-v1/review-20261013-220148-733529/analysis/assessment.md): can an intermediate model **generate its own structural arrangement** from a coarse guide, then supply useful structure to the existing meso model?

The candidate jointly generates elevation and a coarse morphology map. It receives **only coarse height and its baseline permission mask** as conditioning. At generation time, all five output fields start as noise. No target-derived plan, modern Earth elevation, reference image or region identity is passed into its sampler.

The proposed production role remains coast → broad relief → structural intermediate → meso. There is one candidate intermediate network, not an additional selected production stage. Its joint fields evolve together during diffusion; the pilot does not enforce a literal chronological “plan first, heights second” sequence inside that network. Meso supplies subsequent fine refinement. This is a hypothesis about a better structural learning task, not an established replacement.

## What changes and what stays controlled

Both pilots use exactly the same 3,603,749-parameter, five-output-channel network:

- Height: the existing normalized encoded-elevation residual.
- Four morphology outputs: relative high/low corridor category, quiet cue, and cosine/sine of twice the corridor-axis angle. These use the previous oracle's definitions at 40.32 km, not a new terrain classifier.

The **joint** pilot learns all five fields. The matched **control** learns the same height field with four zero morphology targets. Both have noisy auxiliary channels, the same noise/data/augmentation draws and equal update budgets. New noisy-plan input weights and new output weights start at zero. The height response at initialization is verified against the source model. This holds architecture and initialization fixed; the different auxiliary tasks necessarily produce different gradients and may change their subsequent optimization. It is not a comparison against a scalar network with identical gradient dynamics.

Both initialize from the **frozen height-only control** in `outputs/structure/plan-oracle-v1/review-20261013-220148-733529/control-checkpoint.pt`, SHA256 `84127921814f560379bc805a7011d6d51ae439d37ffb8df6b1f6228485a3eb8a`. This source already inherited 2,000 original structure updates and completed another 2,000 height-only updates. The new pilot's default ceiling is **2,000 additional updates**, with a fresh optimizer and EMA schedule in each mode. Neither pilot inherits oracle-conditioned weights.

The height targets, baseline degradation, geographic split, normalization, 40.32 km spacing, 2,580.48 km training context, physical slope/curvature loss and conservation/protection rules remain the same. This deliberately leaves the previously diagnosed target limitations in place so the joint-task test is not confounded with a target rewrite.

Morphology targets are multiplied by 0.5. Their mean denoising error over four channels and permitted cells receives a predeclared weight of 0.25 relative to the unchanged height loss. Averaging the four channels avoids multiplying their contribution by four, while retaining the original height-loss scale. This weight is a bounded pilot choice, not an established optimum. Height and shape terms retain their previous scaling. Plan loss excludes protected cells; noisy plan states and denoised outputs are zero there in both modes. Physical plan extraction remains float32 with mixed precision disabled, using the tested oracle helper. There is no new loss weight sweep or outcome-based tuning in this preparation.

These extra fields are deterministic functions of feasible terrain, so they add no new geographic information at inference. They may provide a useful training representation, or the network may ignore them / learn weak arrangements in both fields. That is exactly what the comparison must resolve. A successful supplied-plan renderer does not guarantee that joint training will learn a good plan distribution.

## Launch separately

The substantive jobs have **not** been launched. From PowerShell in the project, first run:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_joint_structure --mode control
```

After that finishes, separately run:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_joint_structure --mode joint
```

Equivalent double-click launchers are **Train Joint Terrain Control.cmd** and **Train Joint Terrain Structure.cmd**. Defaults are CUDA, batch 16, learning rate 0.0002, warmup 200, seed 42, EMA 0.999 and 2,000 additional updates. Allow several minutes per job; the earlier scalar pilots took about 94 seconds each after setup, but this version has more outputs and validation work. Use the measured rate and ETA printed by this implementation.

Each job prints a fresh console/error log under `outputs/structure/joint-structure-v1/job-logs`. Follow that **exact printed path** in another PowerShell window:

```powershell
Get-Content -LiteralPath 'THE-PRINTED-CONSOLE-LOG-PATH' -Tail 30 -Wait
```

Training writes component losses every 25 updates, resumable checkpoints every 100, and all-55-holdout clean/smoothed validation every 250. Losses are diagnostics, not terrain-quality measures. `best.pt` uses smoothed **height** denoising loss, since the modes have different morphology targets; the review defaults to equal-budget `latest.pt` checkpoints instead. Run locks prevent concurrent writers. Ctrl+C saves a resumable checkpoint. Rerunning resumes to the same additional-update ceiling. Mode, source hash, task definition, data identity, normalization, architecture and optimizer settings must match before resume.

If batch 8 is needed, use `--batch 8` for **both fresh runs** with corresponding fresh `--run` folders; do not change only one mode or attempt to resume a batch-16 run with a different optimizer identity. A second matched training seed is desirable after a first positive result; it is not included in this initial pair.

## Launch the review separately

After both jobs finish at the same additional update:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_joint_structure --with-meso
```

Or double-click **Review Joint Terrain Structure.cmd**. It snapshots checkpoints under their run locks and writes a fresh timestamped folder under `outputs/structure/joint-structure-v1`, plus a durable console/error log. It does not promote any checkpoint or start a planet. Allow roughly several to fifteen minutes for the complete regional review, subject to the GPU's measured performance. It processes 44 regional/structure-seed groups and prints progress after each group.

Omit `--with-meso` for an isolated intermediate-only review. That option makes the same autonomous structural samples and renderer intervention, without the fine-stage work. It is a separate fresh review, not a continuation of an earlier meso run.

## What the review compares

Eleven regional inputs are fixed before generated outputs are inspected:

- Alps, California, and Rockies/Great Plains diagnostics.
- The previous quiet, rugged and coastal formal geographic holdouts.
- Three additional inland holdouts, selected from the unused examples by target-relief order at q10/q50/q90, sequentially without replacement: indices **28, 13, 21**. Their exact indices and rule are saved in settings and [the predeclared input manifest](../outputs/structure/joint-structure-v1/checks/predeclared-review-inputs.json). They are new visual diagnostics, **not fresh statistical confirmation**: all 55 holdouts have been validation-monitored, and they cover two geographic caps.
- The exact two saved generated coast→relief prefix inputs. These have no geographically paired modern-Earth target. They share upstream seed 42 and differ in scalar land-fraction conditions; their historical “60/200” labels are not model age conditioning.

Structure seeds are 42, 314, 2718 and 2026. Meso seeds are independently specified as 113 and 997. The same coarse guide and initial noise are used for matched joint/control samples. Modern-Earth data are used only to construct labelled reference rows; they are never passed to autonomous generation.

Rows are:

1. **Control**: height with zero auxiliary targets, directly handed to meso.
2. **Joint**: jointly generated height and plan, directly handed to meso. This is the primary candidate.
3. **Joint-plan-rendered**: the candidate's generated plan is discretized/masked and supplied to the frozen previous oracle renderer at independent fixed seed 7919. This diagnostic separates a potentially useful generated plan from weak joint elevation rendering. It is not a production-stage selection. Generated plans can lie outside the oracle renderer's training distribution, so failure here does not uniquely diagnose the planner.
4. **Feasible-target reference**, on the nine Earth-derived cases only: supplied terrain through the exact same handoff and meso. This establishes a representation/downstream reference, not learned success or an exact-Earth requirement for generative samples.

The frozen renderer SHA256 is `bd5e1fc86bbe835d5b7f96d73736e780bb32d9456f26e506a285845ade0ceb6f`. Selected meso remains `outputs/meso/selected-adapted-1000/best.pt`, hash `717e1c718fefe51fc0f5066de39618da9cc5a6b034c9df0fb1af09ebcbd7fcf6`. The structural handoff uses **zero sampled detail from the old intermediate**, keeping that potential confound out of this comparison.

The full review saves 168 branch rows, including 36 repeated reference rows, and 336 raw meso branch comparisons when meso is enabled. These are not 336 independent world samples. In particular, reference fields and their same-seed meso results repeat across structure-seed groups. There are 132 learned structural realizations across the two direct models and the fixed-renderer interventions; the renderer interventions are not independently trained planners.

Raw 40 km output, constrained 40 km output, generated raw plan channels, interpreted plan and direct 23 km handoff are saved separately. Meso receives the exact saved central encoded guide; its raw output and baseline are saved. **No meso correction is selectively applied**. Model input identity, finite arrays, structural protection/sign and regional mean constraints are checked. The reviewer rechecks original formal holdout targets and reconstructs their degraded coarse guides from the verified source data rather than trusting an edited derived archive. Checkpoint copies, dataset and input archive hashes are recorded.

Elevation panels use fixed −6,000 to +6,000 m colours, NW illumination at 45°, 5× shading, zero datum, and labelled physical extents. All structural columns cover 1,198.08 km; fine columns cover the central 368.64 km. Morphology maps have categorical colours and show generated versus terrain-implied high/low corridors and axes, plus quiet cues. Formal holdouts retain their stored local frame. Colour/shading settings do not vary by branch or seed.

## How to decide what it means

Primary success requires more plausible *generated alternatives* in joint versus control: compact, elongated systems; separate secondary ranges where resolved; connected low terrain; and gradual transitions into quiet areas. It must recur across seeds and include the generated-prefix guides. Exact reconstruction of the Alps or California is not required.

Generated-plan agreement with the plan measured from its own heights is a **self-consistency** descriptor. It can be high for a bad landscape and can miss a useful plan when the joint height renderer is poor. High/low/quiet precision and recall include cell counts and null values for absent classes, preventing a high majority-class accuracy from pretending to establish good plains. Broad relief, roughness, height, spectral power and training loss cannot substitute for visual morphology assessment.

Interpret outcomes separately:

- Joint improves broad structure and the improvement survives meso: supports this representation within the intermediate role.
- Generated plan yields plausible organization through the frozen renderer, while joint heights do not: supports a planning capability but exposes a rendering/coupling issue; repeat with another renderer seed before relying on that intervention.
- Good direct structure becomes disorganized only in meso: address preservation in the handoff/fine model rather than blindly increasing intermediate training.
- Joint plan and heights remain seed-driven blobs or diffuse texture: auxiliary joint targets alone have not solved structural invention. Check task weighting, context, conditional signal, capacity and target feasibility with distinct controlled experiments.
- Quiet/coastal holdouts remain calm while permitted mountain tasks fail: inspect masks and physical transitions; the existing low-height lock can impose apparent success in quiet cases.

The hard protection rule and constrained targets still suppress some lowland/coastal geometry, and 40 km cannot resolve narrow secondary systems. Those limitations are recorded, not fixed here. This test also does not isolate the PaleoDEM/modern-Earth mismatch. PlanetC's stepped detailed B2W Earth guide remains supplied-structure evidence, not an end-to-end generation result.

Before selecting a replacement, compare against the current selected intermediate route and test continuous regional mosaics / production blending. Single central meso patches cannot establish kilometre-scale system connectivity across an entire 1,198 km context. Before a full planet, require those checks, additional seeds/geographic evidence, and a coherent generated-plan distribution. The probability/stratified/declared-best-feature survey of the **existing** world remains a separate way to measure capability versus prevalence; no new planet is needed for it.

## Preparation checks

The targeted suite passes **22 tests**, covering the new joint task plus the existing plan, structural handoff/conservation and stage-role helpers. Tests verify preserved initial height response, target-only auxiliary changes without added conditioning, equal initial height/shape losses, gradients into new outputs, reproducible coarse-only sampling, inactive-cell behaviour, plan interpretation and CUDA mixed-precision geometry/gradients. The sampler test deliberately makes target-plan extraction unavailable during generation.

Real execution checks are recorded under `outputs/structure/joint-structure-v1/checks`: two optimizer updates per mode in separate `control-smoke-v1` / `joint-smoke-v1` folders; matching end data/CPU/CUDA random states and task/optimizer identities; successful resume preflight; rejection of a wrong-mode resume before checkpoint changes. Both the first review smoke and the final `review-smoke-v2` completed. The final smoke checks the source-derived degraded guides and improved morphology plots, using one seed, eight sampling steps and two inputs (California and an actual generated prefix), including both meso seeds. Smoke images are labelled as execution checks. The four selected production checkpoint hashes still match. See [execution verification](../outputs/structure/joint-structure-v1/checks/verification.json).

These checks establish execution, not a substantive learning result. No substantive training, model selection or full planet generation is performed by preparation.

Sea level remains **0 m**. No global elevation shift, forced land fraction, relief multiplication or histogram remapping is introduced. Existing selected production models and historical outputs are preserved.
