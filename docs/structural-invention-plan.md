> **Superseded on 14 Oct 2026 by [`docs/tectonic-prototype.md`](tectonic-prototype.md).** Kept for the record; do not run the long planner jobs below.

# Structural invention: decisive experiments (plan v2)

This replaces the "four stages with more jointly generated fields" proposal in
`docs/terrain-generation-flow-plan.md` as the next working plan. The stage list stays:
land/ocean → broad relief → **structural (planner → renderer)** → meso, sea level 0 m.

## Why this, and not more fields

The existing experiments already narrowed the failure to one step:

- **Rendering supplied structure works.** Given an Earth-derived 40 km plan, the frozen continuous
  renderer and the selected meso produce organized belts and lowlands.
- **Inventing structure fails.** Every autonomous planner so far (joint, dedicated, categorical,
  continuous, up to 5,000 updates) produces ramps. In the guide-regime probe it fails even on a
  *training* example once the guide is degraded, while the feasible reference under that same guide
  is still structured.
- **The proposed stage-3 outputs (mountain/lowland/relief support generated with height) were already
  tested** as the joint-structure pilot: 35–39 m RMS change against the control, weak plan→height use.
- **Relief potential at ~160–240 km cannot carry the missing arrangement.** Coarsening a supplied plan
  to 161 km damaged the result as much as removing it. Direction channels made little difference.

So the open question is whether a model can learn to *invent* arrangements, and if not, what supplies
them. Every negative result on invention came from ≤5,000 updates of a 3.6 M-parameter model
(≈15 passes over 5,248 patches), which cannot distinguish "wrong task" from "undertrained".

## What was added (no selected model changed)

| File | Purpose |
|---|---|
| `planet_training/structure_plan_v2.py` | Continuous plan codec, constraint **policy** (`locked-200` = the existing rule exactly; `coast-only` = only ocean frozen), plan-only planner network (width configurable, exact weight copy from the v1 planner), renderer/planner losses and samplers, 40→23 km handoff, constraint checks. Verified bit-identical to v1 targets, projection, codec, renderer sampling and planner sampling for `locked-200`. |
| `planet_training/arrangement_metrics.py` | Arrangement metrics and **predeclared gates** (below). |
| `planet_training/procedural_plan.py` | Plan source B: rule-based layouts (`arc-foreland`, `paired-ranges`, `plateau-margin`, `branching`, `basin-range`, `composite`) placed on the guide's upland and converted to the renderer's four channels with the same function that made its training labels. |
| `scripts/train_structure_plan_v2.py` | Long-budget planner (A) or renderer training. Resumable, run-locked, EMA snapshots, durable logs. |
| `scripts/review_plan_sources.py` | Scores reference / supplied / zero / learned / rule-based plans on identical guides, with rule-selected training and holdout cases, optional meso and a memorization check. |
| `scripts/compare_constraint_policies.py` | What the 200 m lock removes from the learning target (seconds, no model). |
| `planet_training/elevation_shading.py` | Same fixed colours / NW 5× shading, without importing the raster-loading diagnose script. |
| `tests/test_structure_plan_v2.py` | 14 CPU tests (no data/checkpoints needed). |

## Arrangement gates (fixed before any new run)

Metrics on 40 km fields, central 48×48 cells (1,935 km), land only:

- **long-ridge share** — high cells (≥150 m above the land-only 80 km mean) in ridges ≥300 km long with elongation ≥2.
- **flanked-lowland share** — land cells ≥150 m below the 161 km mean with high ground within 242 km in two
  directions ≥135° apart (lowland between or beside ranges: Po, Central Valley).
- **median 322 km window contrast**; also connected-low share, contrast ≥1,500 m share, 90th-percentile slope.

A sample **passes** when, against the feasible reference on the same guide, long-ridge ≥0.5×,
flanked-lowland ≥0.6× and contrast ≥0.6×. Only cases whose reference itself is structured are gated.
A **set passes** at ≥60% of (case, seed) samples. Calibration on four Earth guides through the frozen
renderer: supplied reference plans pass 4/4 (flanked ratio 0.98–1.11); the 5,000-update planner passes 1/8
(flanked ratio 0.14–0.43 on Alps / California / rugged holdout). Gates decide what to do next; the galleries
still need a visual check.

Memorization: generated holdout plans are flagged if their median nearest-training-plan correlation
(8 symmetries, 161 km channel) exceeds that of the holdout references by >0.1.

## Constraint policy finding

`Compare Constraint Policies.cmd` → `outputs/structure/plan-v2/constraint-policies.png`. The 200 m lock
freezes 45% of Alpine land, 73% of the quiet holdout and 63% of the coastal holdout, but only 2–4% of
mountain interiors (California, rugged holdouts). Under `coast-only` the Alpine target gains the connected
Po lowland and the Apennine ridge (flanked-lowland share 0.05 → 0.18). The lock mainly removes exactly the
lowland-beside-range geometry the project is missing.

## Baseline results (CPU review, already run)

*Pending.* A CPU review of the existing 5,000-update planner and the rule-based plans (no meso) is running
in Claude's workspace; its gate results and galleries will be added here and copied to
`outputs/structure/plan-sources-v2/baseline-5000-cpu/`. Command 1 below produces the same review, plus meso,
on your GPU in a few minutes.

## Commands, in order

All from the project folder in PowerShell. Each prints a durable log path under the run's `job-logs`.
Nothing starts automatically afterwards. Durations are estimates from the measured v1 rate (~25 updates/s
for one 32-wide model at batch 16); trust the printed rate/ETA.

### 1. Baseline review with meso (≈3–6 min) — optional

Adds the meso check to the CPU baseline above.

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -u -m scripts.review_plan_sources --planner 'outputs\structure\spatial-supervision-v1\autonomous-review-5000\planner-pair.pt' --procedural --meso --memorization
```
(or **Review Plan Sources Baseline.cmd**)

### 2. A1 — the long planner run (≈70–90 min)

Continues the existing 5,000-update continuous planner (exact copy, fresh optimizer) to 100,000 more
updates. Same width, data, targets and plan encoding as the frozen renderer, so duration is the only change.
EMA snapshots at +10k, +25k, +50k and +100k.

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_structure_plan_v2 --task planner --policy locked-200 --init v1 --steps 100000 --max-seconds 9000
```
(or **Train Planner Long A1.cmd**). Ctrl+C saves; rerun the same command to resume. Output:
`outputs\structure\plan-v2\planner-locked-200-v1cont\`.

Then review each snapshot (≈2 min each; add `--meso` on the last):

```powershell
foreach ($s in '010000','025000','050000','100000') {
  & '.\.venv\Scripts\python.exe' -u -m scripts.review_plan_sources --planner "outputs\structure\plan-v2\planner-locked-200-v1cont\ema-$s.pt" --memorization --out "outputs\structure\plan-sources-v2\A1-$s"
}
```

**Decision after A1** (read `gates.json` → `decisions`):

| Learnability (training cases) | Generalization (holdout + named) | Next |
|---|---|---|
| pass | pass, no memorization flag | Learned planner is viable. Visual check, then retrain the renderer/planner under `coast-only` (step 4) and test small mosaics. |
| pass | fail | Learnable but not generalizing: more data/augmentation, or condition the planner on rule-based skeletons (hybrid). |
| fail, but gate ratios rising across snapshots | – | Still learning: run A2 (wider). |
| fail, flat across snapshots | – | Duration is not the bottleneck. Run A2 once; if A2 also fails on training cases, stop learned invention at this stage and use plan source B. |

### 3. A2 — only if A1 fails learnability (≈3–5 h)

Same task, from scratch, 64 channels (≈4× the parameters).

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_structure_plan_v2 --task planner --policy locked-200 --init scratch --width 64 --steps 100000 --max-seconds 21600
```
(or **Train Planner Wide A2.cmd**), then review its snapshots as above with `planner-locked-200-w64`.

### 4. Coast-only renderer (≈3 min) and rule-based review (≈3 min)

Can run any time, independent of A1. Fine-tunes the v1 continuous renderer for the `coast-only` policy
(new plan normalization fitted on training data only).

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.train_structure_plan_v2 --task renderer --policy coast-only --init v1 --steps 3000 --max-seconds 900 --snapshots 1000
& '.\.venv\Scripts\python.exe' -u -m scripts.review_plan_sources --renderer 'outputs\structure\plan-v2\renderer-coast-only-v1cont\latest.pt' --procedural --meso --out 'outputs\structure\plan-sources-v2\B-coast-only'
```
(or **Train Renderer Coast-only.cmd**). The "supplied" rows of that review show whether the coast-only
renderer reproduces Earth's lowland-beside-range structure; the procedural rows are plan source B
without the lock. A coast-only planner (if A1 succeeds) must share this renderer's normalization:
`--task planner --policy coast-only --init v1 --stats-from outputs\structure\plan-v2\renderer-coast-only-v1cont\latest.pt`.

## Limits

- Rule-based plans render as clean, somewhat tube-like belts at 40 km; meso adds detail but the layouts
  are stylized. They are a guaranteed route to organized arrangements, not proof of realism.
- With `locked-200`, belts next to frozen lowland can produce visible steps at 645 km conservation blocks
  (the renderer adds net mass that conservation can only remove from eligible cells). `coast-only` mostly
  avoids this.
- Earth references are a fair comparison only on structured guides; generated coast/relief prefixes have
  no reference and are reported ungated.
- The gates are regional. World-scale arrangement and frequency still need the existing-world survey.
