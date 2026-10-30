# Spatial supervision training comparison

The current pilots generate a categorical high/low/quiet plan with axial directions. The bounded target audit found that spatial category arrangement is useful, axes have little effect through the frozen renderer, generated low corridors remain sparse, and categorical fields omit regional strength/context. See `outputs/structure/target-design-v1/audit-20261014-v1/assessment.md`.

The paired renderer and dedicated-planner trainers are now implemented in `scripts/train_terrain_spatial_plan.py`, with their encoding/model definitions in `planet_training/terrain_spatial_plan.py` and regional review in `scripts/review_terrain_spatial_plan.py`. **No production objective or checkpoint is changed.** The categorical and continuous branches start from the same frozen height-only control, SHA256 `84127921814f560379bc805a7011d6d51ae439d37ffb8df6b1f6228485a3eb8a`, with all new plan inputs/heads zero. Old categorical plan weights are not repurposed.

The first real pilot is **complete**: 1,000 updates per branch for renderers (90.8 seconds) and planners (83.3 seconds), followed by supplied-plan and autonomous regional reviews. See [the assessment](../outputs/structure/spatial-supervision-v1/assessment.md). Supplied continuous plans render usefully; autonomous terrain remains weak at this budget. The commands below record completed first-pilot steps and the separately user-launched continuation. Repeating the same 1,000-update command resumes at its already completed ceiling and does not add 1,000 more.

## Hypothesis

Generating continuous terrain structure at several physical scales will make connected highs/lows, secondary systems and transitions easier to learn than independently quantized local labels plus largely redundant directions. This is a claim about a potentially easier task for the finite model/budget, not a claim that categorical diffusion cannot represent topology.

Keep the current stage hierarchy. A separately generated plan and learned renderer already require two learned passes within the experimental intermediate role. This proposal does not approve that additional production cost or add more resolution levels. The original selected coast -> unsigned relief -> intermediate -> meso chain remains selected.

## Four-channel candidate, same native grid and channel budget

Let `Z` be the existing **feasible modern-Earth 40.32 km target**, constructed with the same degraded guide, low-height permission, sign protection and local conservation as the current structure pilot. Do not replace it with a translated original-Earth residual and do not silently relax the constraints.

Let `G_s` be a Gaussian blur with physical standard deviation `s`, and `M = (baseline > 0)` the known guide-land support. Define `A_s = G_s(M Z) / max(G_s(M), epsilon)`. Compute statistics over land so a flat coastal landmass is not classified as a mountain merely because its neighbourhood includes deep ocean. This mask is used to construct labels; it does not edit any generated elevation or force land coverage. The four clean plan fields on guide land are:

| Channel | Physical definition | Intended information |
|---|---|---|
| Local relative height | `Z - A_80.64km` | Range/valley separation and local contrast |
| Regional relative height | `Z - A_161.28km` | Compact systems, adjacent lower regions and secondary systems |
| Broad relative height | `Z - A_322.56km` | Regional arrangement and transitions within the patch context |
| Local relief | `sqrt(max(G_80.64km(M Z^2) / max(G_80.64km(M), epsilon) - A_80.64km^2, 0))` | Quiet versus active relief, independently of absolute elevation |

These are overlapping physical scales, not disjoint frequency bands or literal mountain classes. A broad high plateau can be quiet locally. A strong relative-height response is not by itself a mountain quality score. The training implementation matches the previously audited physical candidate. A synthetic flat coastal landmass and rotated ridge check passed. The comparison tests a bundle of continuity, multiscale context and land-restricted statistics; it does not isolate the cause of any gain. A later ablation can use identical land statistics in both encodings if needed.

For a first controlled test, apply the existing permission mask exactly as before. Keep protected cells explicit in evaluation, and do not count them as learned quietness. The proposal does not solve the independent question of how much coastal/lowland terrain the protection rule removes.

All physical calculations run in float32 outside mixed precision. Continuous compression is `asinh(field / 100m)` for signed fields and `log1p(relief / 100m)` for relief; categorical physical values are unchanged. Both branches estimate per-channel centring/scaling over active cells in the same 256 declared training patches, chosen without replacement with seed 1099 and the existing guide-degradation recipe. Standard deviations have a 0.05 floor. Encoded target scale is 0.5. Per-channel quantiles, covariance/correlation, indices and degradation recipe are saved with the checkpoint. No holdouts are used for normalization; no generated histogram/coverage is forced. Unit scales are encoding choices, not desired terrain amplitudes. Targets are not clipped to fit a display range. All encoded protected cells are exactly zero.

Both branches have the same normalization protocol, not the same numerical channel statistics. The categorical control therefore differs in conditioning scale from historical unnormalized category pilots; the valid comparison is the newly matched pair. Plan denoising losses from different encodings are not a morphology ranking. Continuous plans are supplied directly in their normalized encoding to their own renderer. Generated categorical plans are decoded, interpreted with the historical category/quiet/axis rules, and re-encoded for their renderer.

The 322.56 km blur requires context: calculate before display crops and exclude at least its three-sigma filter margin from metrics that claim unpadded physical support. The current 64-cell patch has little such uncontaminated interior at that scale (only 16 cells across), so native edge-reflection effects must be reported. For a first matched pilot retain the existing context and padding in both branches; a larger-context experiment is separate. The central 32-cell figures in the audit are illustrative for this broad channel, not boundary-free validation. Reflect-versus-replicate padding changes that broad field by about **0.81 m RMS** across the 55 central holdout crops, while the other three channels are unaffected there. This is a sensitivity check, not a comparison against true missing terrain. A substantial full-patch/edge effect would require extracting real fine-resolution DEM context with the geographic split guard, or dropping the broadest scale; the existing coarse guide halo is not equivalent to fine target context.

## Matched experiment order

1. **Encoding/data check.** Verify rotation/reflection behaviour, finite float32 fields under training autocast, decoding of compression/normalization, protected-cell handling, training-only statistics and broad-channel edge sensitivity. Inspect full modern and feasible targets together so conservation/protection losses are not blamed on a generator. Record exact source and dataset hashes.
2. **Supplied-plan rendering test.** Train a categorical renderer and a continuous renderer from the same frozen height-only source, with identical architecture (four plan conditioning channels), optimizer, normalization protocol, guide batches/degradations, noise fields, seed, updates and fresh EMA. Initialize plan-input weights to zero in both. The height target/loss/shape penalties stay identical. Do not reuse categorical plan weights under a new meaning. This tests whether the representation is useful when supplied correctly.
3. **Autonomous planning test, conditional on step 2.** Train categorical and continuous dedicated planners with the same five-channel architecture and zero noisy-height channel. Initialize both new plan heads/input paths identically from the same height-only source; do not compare one warm-started category planner with a cold continuous one. Equal plan loss weighting and budgets. Generate four declared plan seeds and pass each to its matching renderer under two renderer seeds. Include two meso seeds on a predeclared subset.
4. **Separate optimization ablation if needed.** If richer supplied plans render well but generated plans remain weak, distinguish representation from duration/capacity and removal of the noisy-height shortcut or loss weighting. The last dedicated/joint comparison changed those together. Do not add connectivity penalties, change masks, enlarge networks and extend training at the same time.

The trainer has a declared update ceiling and wall-time guard, atomic **paired** checkpoints, strict task/data/source/optimizer/batch/seed resume identity, and fresh durable stdout/stderr logs. Ctrl+C is deferred to a pair boundary. Both optimizer states, EMA states and random states are saved together. Resume reproduced an uninterrupted four-update renderer run bit-for-bit for model and EMA tensors. A changed batch was rejected without touching the checkpoint. Both tasks passed actual short execution checks. A wall-time stop saves equal update counts and returns exit code 2, an interrupted run returns 130, and completion returns 0. No next job or planet starts automatically.

Validation uses the same 55 monitored holdouts, six noise levels and common draws; it does not provide a new geographic confirmation sample. Default training is batch 16, learning rate 0.0002, 200-pair warmup, Adam (0.9, 0.99), gradient clip 1, ramped EMA capped at 0.999. One update means one update **per branch**. `--steps` is the total paired ceiling, including any prior completed pairs, not an increment.

## Ready-to-run commands and logs

Run the paired supplied-plan renderers first. The default budget includes setup and validation, with a saving/validation reserve:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_spatial_plan --task renderer --steps 1000 --max-seconds 110
```

Check `outputs/structure/spatial-supervision-v1/renderer-paired-1000/training-result.json` for the actual paired count. A guarded stop may finish fewer than 1000 updates. Review the actual checkpoint, with supplied-plan rows explicitly labelled:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_spatial_plan --max-seconds 120
```

If both representations can be rendered usefully, train the matched dedicated planners. This is a separate command, not an automatic chained job:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_spatial_plan --task planner --steps 1000 --max-seconds 110
```

Then review autonomous planning plus labelled supplied references:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_spatial_plan --planner 'outputs/structure/spatial-supervision-v1/planner-paired-1000/latest.pt' --max-seconds 120
```

The review freezes the checkpoint bytes in a fresh review folder. It verifies planner/renderer source, architecture, dataset, height normalization and plan-codec compatibility. Earth inputs are Alps, California, Rockies/Plains and the rugged/quiet formal holdouts; autonomous review adds the saved genuinely generated coast/relief prefix. Four declared plan seeds (42, 314, 2718, 2026), two renderer seeds (7919, 104729), and fine seeds 113/997 on Alps/California plus the generated prefix at the first declared plan/renderer seed. The generated-prefix fine subset was added before substantive autonomous inspection; its Earth-reference column stays blank. A review budget stop leaves saved arrays and exit code 2; choose a new review folder with a larger explicitly user-launched budget to finish. It does not silently discard regions/seeds or start another job.

Every command prints a fresh durable combined stdout/stderr log under `outputs/structure/spatial-supervision-v1/job-logs`. Training also writes `log.jsonl`, `launch.json`, `training-result.json`, and atomic `latest.pt` in its run folder. In another PowerShell window, monitor the actual printed console log:

```powershell
Get-Content -LiteralPath 'THE-PRINTED-CONSOLE-LOG-PATH' -Tail 30 -Wait
```

After this pilot, the next controlled extension is to train **both planners longer while holding the already tested renderer snapshots fixed**. The following command is **for the user to launch**, not automatically executed. It resumes the existing paired planner run from 1,000 to a 5,000-update ceiling, with a ten-minute wall-time limit. At the observed rate the additional 4,000 paired updates should take roughly five to six minutes; actual speed can vary. The folder's original `1000` suffix does not change; actual counts are recorded in the checkpoint/result:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_terrain_spatial_plan --task planner --steps 5000 --max-seconds 600
```

Review that continuation against the **frozen 1,000-update renderers**, whose bytes are preserved in the completed renderer review:

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.review_terrain_spatial_plan --renderer 'outputs/structure/spatial-supervision-v1/renderer-review-1000/renderer-pair.pt' --planner 'outputs/structure/spatial-supervision-v1/planner-paired-1000/latest.pt' --max-seconds 120
```

If a later capability problem warrants extending the renderers too, their matching command is `python -u -m scripts.train_terrain_spatial_plan --task renderer --steps 5000 --max-seconds 600` using the same project interpreter, but that is a separate experiment, not part of this proposed planner-duration comparison. Do not extend merely because loss decreases. Preserve categorical and continuous counts and assess actual terrain before another extension. Checkpoint resume retains the original learning-rate/EMA trajectory; it does not restart warmup.

## Assessment and decision

Use fixed physical footprints, elevation colours and shading, with a primary predeclared seed and all alternatives available. Compare full feasible target and supplied-plan renderings to diagnose stage capability; generated samples are allowed different plausible geography.

Inspect compact connected systems, connected low regions, width/height transitions, secondary ranges, plateau/plains quietness and changed-region responses. Report connected-component descriptors on fixed masks/scales with denominators; they supplement visual judgement. Do not optimize a single largest-component score: one giant ridge or flattened plain could score well. Keep labels measured from the plan separate from those measured from actual rendered terrain.

The candidate succeeds only if it improves plausible organization across several regions and seeds, maintains quiet examples without crediting locks, and survives the meso handoff. Its rendered fidelity/loss or higher relief alone is insufficient. Dense continuous plans contain more height information than categorical labels despite having the same channel count: a better supplied-plan reconstruction is an information/capability check, not evidence that a coarse-conditioned planner can generate that information. Regional performance does not establish world frequency: follow with the existing-world geographic probability sample, guide strata and separately declared search for distinct strongest systems. No full planet run or production stage-count decision until these distinct capability and prevalence questions have evidence.

## Completed audit command

The no-training intervention audit is already complete. To reproduce it in a **new** output folder:

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.audit_terrain_plan_targets --out 'outputs/structure/target-design-v1/audit-repeat' --max-seconds 120
```

It prints a fresh durable console/error log in `outputs/structure/target-design-v1/job-logs`. Monitor that printed file from another window:

```powershell
Get-Content -LiteralPath 'THE-PRINTED-CONSOLE-LOG-PATH' -Tail 30 -Wait
```

The audit performs only bounded regional frozen-model sampling and saved-array analysis. It does not train or promote models. Sea level remains zero.
