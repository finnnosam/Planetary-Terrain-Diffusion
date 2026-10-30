# Controlled terrain-plan experiment

The completed target/stage-role experiment supports changing the structural learning task before adding another production stage. See `outputs/terrain-stage-roles/paired-v1/assessment.md` and its matched-scale, all-seed galleries. PlanetC is not an end-to-end round trip: its stepped Earth-derived B2W.png guide already contained substantial organization.

This new experiment tests whether explicit coarse landform information improves **regional structural synthesis**. It does not train a global planner, modify selected models, solve the target/protection tradeoff, or provide a new planet sampler.

## Matched control and oracle

Both pilots initialize from the frozen, unselected 2,000-update 40 km structure EMA (`65562cb3…`). They use the same 3,602,593-parameter architecture, source normalization, feasible constrained modern targets, data draws, augmentations, noise draws, optimization schedule and 2,000-additional-update budget. Both have four extra conditioning channels; the control always receives zeros. The new input weights start at zero, preserving the old noisy/height/mask response at initialization.

The oracle receives a target-derived plan with four 40.32 km channels:

1. Relative position: coarse ridge/high corridor, neutral terrain, or valley/low corridor, using a ±150 m departure from an 80.64 km Gaussian neighbourhood.
2. Quiet permitted terrain: a local relief/slope indicator; this is a coarse cue, not a validated global plains classifier.
3. Cosine of twice the corridor-axis angle.
4. Sine of twice that angle, so the orientation is axial rather than a downhill arrow.

Plans are derived from the **feasible constrained target**, with no exact elevation channel, and are zero in protected input terrain. The old height/permission channels remain available. Training target and denoising/physical-shape objective are unchanged so the first comparison isolates conditioning information. An oracle plan deliberately supplies information unavailable to current world generation; its success would prove a conditional capability, not end-to-end world synthesis.

The target audit found that current low-height protection can erase desired coast/lowland geometry. This pilot intentionally retains that limitation. A target/protection revision should be tested separately rather than conflated with the new plan input.

## Verification completed

- Tests check finite bounded plan channels, inactive-cell masking, high/low/quiet cues, rotated corridor axes, preserved initial source-model response, matched initial control/oracle losses and gradients into the new plan weights. The final targeted suite passes **18 tests**, including a CUDA regression confirming that training and review extract identical plans and that a 3,500 m flat plateau remains quiet.
- Physical geometry is calculated in float32 with mixed precision disabled. Squared metre elevations can overflow float16; this was corrected during testing, and the affected execution checks were repeated.
- Final real CUDA execution checks performed **two additional updates per mode**, batch 2, in separate `checks/control-smoke-v2` and `checks/oracle-smoke-v2` folders. They are not the substantive pilots; earlier v1 execution artifacts are retained separately.
- The control's new channel weights remained exactly zero; the oracle's became nonzero. Dataset, source initialization, architecture, optimizer settings, normalization, checkpoint step and noise RNG states matched.
- Resume preflight passed without changing checkpoint weights. A two-region/eight-step paired review smoke passed, saved frozen checkpoint copies and checked protection, sign, regional means and finite arrays.
- Selected intermediate/meso and the frozen source structure checkpoint hashes remain unchanged.

The substantive default control/oracle runs have **not** been launched. Two optimizer updates cannot establish plan-following or improved terrain.

## Launch these three jobs separately

From PowerShell:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_plan_oracle --mode control
```

After it finishes, separately launch:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_plan_oracle --mode oracle
```

Equivalent double-click launchers are **Train Terrain Plan Control.cmd** and **Train Terrain Plan Oracle.cmd**. Both default to CUDA batch 16, learning rate 0.0002, 200-update warmup, seed 42 and 2,000 additional updates. The completed predecessor took about 105 seconds for 2,000 updates on this GPU; this pilot adds plan extraction and held-out checks, so allow several minutes and use its measured rate/ETA rather than treating that timing as a guarantee.

Each job prints a fresh durable console/error log under `outputs/structure/plan-oracle-v1/job-logs`. It logs progress every 25 updates, saves resumable checkpoints every 100 and checks all 55 holdouts every 250. `latest.pt` is the fixed-budget candidate; `best.pt` is only loss-selected and is not automatically promoted. Ctrl+C saves resumable state. Rerunning the same command resumes to the total additional-update ceiling. Resume checks require the original mode, dataset, target/plan definition, architecture, source initialization, normalization and optimizer settings. Runs have locks.

Use `--batch 8` on **both** fresh pilots if needed. Their batch/optimizer identities must match for the paired review. Do not initialize one comparison from the other's trained weights or change their seeds/budgets independently. `--run` selects a separate experiment directory; matching variants need matching settings.

In a second PowerShell window, follow the exact log path printed at launch:

```powershell
Get-Content -LiteralPath 'THE-PRINTED-CONSOLE-LOG-PATH' -Tail 30 -Wait
```

After both jobs finish at the same additional update, launch the separate review:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_plan_oracle
```

Or use **Review Terrain Plan Oracle.cmd**. It freezes both checkpoints under their run locks, rejects mismatched training histories, and creates a fresh timestamped review folder. It does not start meso, select a checkpoint or generate a planet.

## What the review tests

Six regional guides reuse the completed stage-role experiment's exact inputs: Alps/California/Plains diagnostics and three formal geographic holdouts. The same coarse height and seed are held fixed across:

- height-only control;
- oracle model with correct plan;
- oracle model with absent plan;
- oracle model with a feasible changed plan.

The changed-plan target rotates the feasible target residual by 90 degrees under the fixed baseline and reapplies the same constraints; it therefore remains admissible to the tested representation. Its plan is then extracted. This may still be outside the natural conditional training distribution and should be treated as an intervention, not a guaranteed realistic target.

Seeds are 42/314/2718/2026, 32 sampling steps. Target and counterfactual reference rows are supplied fields, not learned results. All elevation panels cover the same 1,198.08 km square with fixed metre colours and NW 5× shading. Raw and constrained 40 km outputs are saved separately and shown, followed by the deterministic direct 23 km handoff. There is no sampled intermediate detail or meso confound in this first test.

Look for meaningful, coherent changes in corridor placement/type/direction when the plan changes, better compact belt/lowland arrangements than the matched control, and retained quiet terrain. Agreement with the supplied coarse plan is recorded, but is not a standalone realism score. Read all seeds and both raw/projected output. Lower denoising loss, stronger heights or correct average elevations do not establish success.

These inspected regions are development diagnostics, not fresh confirmation. A successful pilot needs new geographic tasks/seeds, downstream meso and small production-mosaic checks before any production selection. Global coherent plan generation and world-frequency evaluation remain unresolved.

Sea level stays 0 m. No global elevation shift, forced land fraction, histogram remapping or relief multiplication is introduced.
