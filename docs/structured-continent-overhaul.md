# Coastline-first continent overhaul, 11 October 2026

Sea level is established at **0 metres before elevation generation**. Land
percentage describes coastline geometry, rather than a height shift applied to
an existing DEM. No full intermediate/meso generation is part of this workflow.
The user has explicitly identified full-planet meso generation as a significant
expense: use small continent previews and regional refinement checks, and obtain
agreement on the result before starting another full meso globe.

## Why the previous control failed

The old coverage control selected an elevation quantile and subtracted it from
every cell; its relief control then multiplied positive heights. For the
continued seed-42 example, selecting 30% land required a -673.07 m sea-level
offset. It exposed 14.11% of the planet that had previously been shallow ocean
(-1,000 to 0 m), then raised that new land further with the 1.25 multiplier.

| Native spherical measurement | Natural output | 30% land / 1.25 relief |
|---|---:|---:|
| Land coverage | 15.90% | 30.00% |
| Ocean shallower than 200 m, fraction of planet | 8.19% | 0.97% |
| Median land elevation | 214 m | 848 m |

This is a sea-level edit, not generation of more continental crust with its own
shelves and relief. The original controls remain available for explicit legacy
use, but the UI now calls the coverage control **Sea-level land coverage** and
recommends blank coverage and relief 1. Audit:
`outputs/continents/continued-examples/coverage-audit.json`.

The full controlled meso globe did finish. Its 13,888 × 6,944 elevations are
finite, range from -5,149 to 5,242 m, and cover 29.813% land by spherical area.
Its new 740 km close-ups show added detail, but smooth broad plateaus still
dominate some regions. Added relief RMS over the intermediate guide is 109 m
in the highland example, 54 m in the upland and 23 m in the lowland. Higher fine
power did not establish the visual improvement the user wanted. Verification
and same-grid stage comparisons are in
`outputs/meso/continued-land30-relief125/verification.json` and
`outputs/meso/continued-land30-relief125/stage-closeups.png`.

## New representation and model

The new implementation is independent of old checkpoints and has two stages:

1. **Coastlines:** a diffusion model learns land/ocean labels at the existing
   ~161 km resolution, conditioned on the requested spherical land percentage.
   The sampled coast is resolved into a land/ocean field before relief starts.
   Coverage is learned conditioning, not an exact quantile constraint.
2. **Relief and bathymetry:** a second diffusion model learns square-root height
   and depth magnitudes conditioned on that fixed coastline. The coastline
   supplies the sign when decoding metres. Known land/ocean domains receive
   balanced reconstruction weight; weights do not depend on unknown depths. Sea level stays
   zero; no final height shift or histogram remapping occurs.

Labels are the sign of the area-averaged DEM, not a claim that mixed ~161 km
coastal footprints are entirely land or water at fine resolution. This keeps the
sign and magnitude representation exactly invertible on the training grid.
Magnitude squaring is its decoder. Interpolated TIFF shores and later learned
refinement can still move within those coarse cells.

The architecture treats cube faces as spatial charts with shared convolution
weights. Every 3 × 3 convolution reads geometric neighbouring-face halos, using
bilinear spherical lookup and shared-edge feature identification. The pyramid
keeps chart endpoints aligned, and attention spans all six faces at the coarsest
level. This replaces the original use of faces as six image channels.

Face-aware padding is an established cubed-sphere CNN approach; see
[NVIDIA's DLWP documentation](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/weather.html)
and [Weyn et al. (2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020ms002109).
Using it here is an architectural inference, not evidence that weather-model
results establish terrain realism. This network is not exactly rotation
equivariant. Random whole-sphere rotations and reflections remain in training.

Existing PaleoDEM data, source averaging, 67 training ages, 18 validation ages
and temporal buffers are reused. Test ages remain untouched. Magnitude
normalization and supported land percentages are computed from training ages
only. Present support is approximately 14.35–35.37% land; generation rejects
requests outside that observed range. No new downloads are needed.

PaleoDEMs remain smooth reconstructions from one geological history. This
overhaul addresses topology, geometry/elevation coupling and error balance;
it does not manufacture independent training planets, tectonic simulation or
fine mountain/river supervision. Better visual quality must still be measured.

## Train and inspect

Double-click **Train Structured Continents.cmd**. It trains the two stages in
sequence, using CUDA batch 16, 24 base channels, learning rate 0.0001,
500-update warmup, 30,000-update ceilings and eight-check validation stopping.
Each has about 1.04 million parameters. Both keep EMA checkpoints and preserve
the existing continent, intermediate and meso models. Closing with Ctrl+C saves
the current stage and prevents the next stage from starting.

Folders:

- `outputs/continents/structured-coast/run2-unbiased`
- `outputs/continents/structured-relief/run2-unbiased`

Or use **Train Coastlines.cmd** and **Train Continent Relief.cmd** separately.
The relief model trains from known source coastlines and does not require an
already-trained coast model. Resuming a run requires its original source,
architecture, learning rate, warmup, EMA and seed; batch size and total budget
may be changed. Old elevation-only checkpoints cannot resume these models.

After training, use **Evaluate Structured Continents.cmd** for four matching
seed previews at 30% requested land, full held-out losses and reference land/shelf/relief
statistics. It records checkpoint hashes from the actual loaded bytes.
Reports and images are in `outputs/continents/structured-evaluation`. These new
losses cannot be numerically compared with the old elevation-only loss.
The evaluator also screens rotation/reflection-aligned land-mask similarity
against all training maps. This approximate screen does not prove absence of
copying; inspect the closest source maps too.

**Preview Structured Continents.cmd** generates a continent guide and side-by-
side coast/elevation image in a fresh timestamped folder. It never starts an
intermediate or meso run. Optional arguments include `--land-percent`, `--seed`
and `--out`; elevation/sea-level shift controls are intentionally absent.

```powershell
.venv\Scripts\python.exe -u -m planet_training.train_structured coast
.venv\Scripts\python.exe -u -m planet_training.train_structured relief
.venv\Scripts\python.exe -m planet_training.evaluate_structured
.venv\Scripts\python.exe -m planet_training.generate_structured --land-percent 30 --seed 42
```

The exported metre-based `layout.tif` remains compatible with the existing
intermediate stage, at the same native cube scale. The next acceptance checks
are coastline plausibility, achieved coverage, retained shelves, broad relief,
training-map similarity, then a few intermediate/meso regions. Full globe
generation should follow those checks and a preview the user accepts.

## Corrected objective and checkpoint review

The user's first coastline run reached 10,200 updates. The exact 10,000-update
checkpoint was reviewed and frozen at
`outputs/continents/structured-coast/run1/review-010000/reviewed-checkpoint.pt`.
It fails the coverage gate: four seeds requested at 30% produced 43.6–45.5%
land, and a matched seed requested at 20/30/35% produced 42.65/44.08/45.20%.
The old decreasing validation loss was not a sufficient quality measure.

The first loss balanced land/water according to the **unknown clean label**.
For an ambiguous pixel with true land probability p, its weighted mean moves
toward the class-balanced probability, distorting the diffusion sampling score.
At a completely uninformative noise level, population-balancing pushes the
global prediction toward 50/50 regardless of the requested coverage. This is
a loss-design bug, not a sea-level problem or a reason to shift generated DEMs.

`conditioned-area-edm-v2` uses ordinary EDM reconstruction weighted by spherical
area for coastlines. Relief can balance the **known conditioning mask**, but no
longer reweights unknown heights/depths by shallow/deep class. Old data and model
shapes remain compatible. Old objectives cannot resume into the new objective
in-place. `--init-from` starts a fresh optimizer/EMA schedule and validation
baseline from a source EMA, preserving the source; omitting it restarts from
random weights. Fresh coast runs additionally center EDM around the analytic
conditional mean, requested land fraction minus 0.5. Whole-planet rotations
make this the appropriate clean marginal mean. Both noisy inputs and the skip
connection are centered consistently. This improves high-noise conditioning;
it does not force a quantile threshold or guarantee exact generated coverage.
Legacy checkpoints keep their old preconditioning for reproducible comparison.
A corrected run requires its original optimization settings
when resumed. Batch size and total update budget may change.

Both options are being compared using CUDA batch 16 and the same spherical-area
coverage measurements. The warm-start pilot lives in
`outputs/continents/structured-coast/corrected-warmstart-pilot`; the fresh run
uses the new `run2-unbiased` folder. Periodic previews and JSON sidecars now show
requested **and actual** land coverage, with green land and blue ocean.
`Review Coastlines.cmd` checks four seeds and a matched 20/30/35% sweep without
requiring a trained relief model. Reports use hashes of the actual loaded bytes.

### Initial correction trials

Both pilots completed 2,000 corrected updates. Matched four-seed, 64-sampling-step
checks requested 30% land:

| Candidate | Actual land range | Mean absolute coverage error |
|---|---:|---:|
| Original objective, 10,000 updates | 43.59–45.48% | 14.46 percentage points |
| Corrected warm start, 10,000 original + 2,000 corrected updates | 23.04–32.56% | 3.14 points |
| Fresh corrected restart, 2,000 updates | 22.83–28.10% | 4.71 points |

At this early comparison, the warm start was more mature and more accurate. Neither pilot passed
the coverage screen: maximum errors are still about seven points. The fresh run
is being continued to 10,000 updates to compare at the same training age as the
original rejected model, rather than declaring the immature restart the winner.
Its validation loss has decreased from 0.23765 at 500 to 0.12285 at 2,000 updates.
The old weighted loss is not numerically comparable. The warm start's corrected
baseline was 0.11699; it reached 0.09899 at 2,000 corrected updates.

Comparison: `outputs/continents/coast-correction-comparison/comparison.json`,
`matched-coastlines.png` and `coverage-response.png`. The fresh model's small
approximate training-map alignment screen gives land IoUs of 0.39–0.51; this
does not establish independence or lack of memorization.

**Train and Review Coastlines.cmd** resumes the fresh run to a 10,000-update
ceiling and then automatically reviews four seeds plus the 20/30/35% sweep.
The preliminary coverage gate requires mean absolute error <=3 points on the
four 30% samples and maximum error <=5 points across samples and sweep. These
are practical screening thresholds, not validated guarantees of visual quality.
The final review retains an exact checkpoint snapshot and writes
`coast-quality-status.json`. It requires visual review and never automatically
starts relief, intermediate or meso generation. Ctrl+C prevents the review;
the current checkpoint is retained. An OS run-folder lock prevents concurrent
trainers and is released automatically after a crash.

The completed coastline continuation's readable output is in
`outputs/continents/structured-coast/run2-unbiased/training-console.log`.
`training-job.json` records its supervised session.

### Completed 10,000-update coast review and relief trial

The fresh run completed 10,000 updates with no recorded errors. Its best held-out
loss is 0.0988754. Exact reviewed checkpoint SHA256:
`5cbba200e14f4732cbb05dfa5ac1765f58d1e8a9e5e6c40480926bfda026e0c1`.
Frozen checkpoint and four-seed review:
`outputs/continents/structured-coast/run2-unbiased/review-20261011-200952-428278`.
Four matched seeds requested at 30% achieve 27.80, 30.81, 30.57 and 29.62% land
(mean absolute error 0.99 percentage points), versus the old 43.6–45.5%.

The expanded review uses 12 seeds at 30% and four matched seeds at each of
20/25/35%, all with 64 sampling steps:

| Requested land | Actual range | Mean absolute error |
|---|---:|---:|
| 20% (4 seeds) | 17.47–20.35% | 1.17 points |
| 25% (4 seeds) | 21.94–25.34% | 0.88 points |
| 30% (12 seeds) | 25.67–32.95% | 1.99 points |
| 35% (4 seeds) | 33.29–36.09% | 1.07 points |

All expanded checks meet the preliminary coverage thresholds. There is a mild
undercoverage tendency and requests remain approximate. The quick periodic
previews use 32 steps and can miss coverage more; use the reviewed 64-step
sampler for generation. Visual inspection of the 12-seed grid shows coherent,
varied layouts. The four closest training-layout comparisons show substantial
differences; approximate land IoUs of 0.43–0.53 do not prove independence or
absence of memorization. Final test ages remain untouched.

**Decision: approve the frozen coast checkpoint for a relief-stage trial.**
This approves coastline geometry, not elevations, shelves or final terrain.
`outputs/continents/structured-selection.json` records this limited selection.
Expanded results and figures are in the review's `expanded-readiness` folder.

Fresh corrected relief training is now running in
`outputs/continents/structured-relief/run2-unbiased`, CUDA batch 16, up to
10,000 updates with validation stopping. Expected time is about 45 minutes,
plus the end-to-end review. It trains on known source masks at fixed sea level;
the approved generated coast is used for the later evaluation, not mislabeled
as training truth. **Train and Review Relief.cmd** runs the same workflow.
It verifies the selected coast's hash, objective and dataset before training,
then freezes the finished relief checkpoint and evaluates four small
coast/elevation planets with 64 sampling steps. The report is recorded in
`relief-quality-status.json`; readable progress is in `training-console.log`.
It never starts a full intermediate or meso globe. Shelves, coastal profiles
and broad highlands must pass visual review, followed by a few regional tests.

The first relief command session subsequently disappeared after logging update
3,150, with checkpoint 3,000 intact, validation improving to 0.16421, and no
recorded error or early-stopping message. The cause was not recorded. It was
resumed from 3,000 in an independent background process outside the command
sandbox, with unchanged CUDA batch 16 and the 10,000-update limit. Both the
launcher and Python worker were verified alive after launch, and training
advanced past 3,300. Current progress is in
`training-console-resumed-20261011-205407.log`, and errors are in the matching
`training-errors-resumed-20261011-205407.log`. The original console log is retained.
`resume-note.md` and `training-job.json` record provenance and last observed
state; job metadata is not a live heartbeat. The automatic small end-to-end
review still follows training, with no full terrain generation.

The old run and its audit remain intact. No full intermediate/meso generation
is being started during these checks. The Mars MOLA elevation product is a
possible later source of dry-planet relief or explicitly hypothetical flooded
layouts, not observed coastlines or continental shelves: its reference zero is
an areoid, not a sea surface. Untagged Mars elevations must not be pooled into
Earth coastline training by calling all negative elevations ocean.

50 targeted checks passed for objective calibration, conditional mean at high
noise, warm-start/reset and legacy
resume guards, representation, geographic convolution, mixed precision, generation,
evaluation and existing layout/GUI/training behavior, plus relief review snapshot,
selected coast hash/dataset/objective guards and interruption handling. Compilation passed.

### Completed relief review and regional trials

The resumed relief run completed update 10,000 and its automatic four-seed review
without recorded errors. Best validation loss was 0.1546347 at update 10,000.
The exact reviewed relief checkpoint is
`outputs/continents/structured-relief/run2-unbiased/review-20261011-212327-265861/reviewed-relief.pt`,
SHA256 `68857d2275eb1c0b55081bafbbf0cb17edea37d5f67d5e715ce0b9bd1efeea45`.

For the four approved coastline samples at requested 30% land, generated shelves
occupy 3.6–5.0% of the globe; land median elevations are 340–391 m and land p95
is 1,140–1,250 m. Shelves use native elevations between -200 and 0 m. Comparing
against sources with 27–33% land, rather than all geological ages, gives train
shelf coverage 2.2–8.1% (median 4.4%) and land p95 882–1,370 m. Thus modest
relief and shelf coverage are compatible with this training distribution.
This does not establish mountain realism. Native cube seams have zero mismatch.
The coastline supplies elevation sign before relief, without shifting sea level.

**Decision: provisionally approve relief for bounded regional trials.** Selection
records preserve both exact reviewed checkpoints. The continental preview at
`outputs/continents/structured-previews/reviewed-10000-seed42` reproduces the
reviewed seed 42. A cheap global intermediate guide (~23 km) was generated at
`outputs/intermediate/structured-10000-seed42`; its land area is 27.42%, versus
27.53% in the exported continent TIFF and 27.80% on native continent nodes.
Interpolation and refinement can change shores within coarse cells; the zero
datum stays fixed. The intermediate export is finite, in metres, and has the
global spherical CRS.

`scripts/review_structured_regions.py` chooses highland, upland and shoreline
locations from that intermediate guide before looking at meso output. It runs
three ~740 km, 256×256 regions through the frozen adapted meso model, seed 42,
32 sampling steps, CUDA batch 16. The three trials required only 48 meso tiles;
no full meso globe was generated. Settings and source/checkpoint hashes are
retained alongside the results in
`outputs/meso/structured-10000-seed42-regional`.

All three TIFFs are finite. Meso changes the intermediate by RMS 61, 43 and 25 m
for highland, upland and coast respectively. The comparison uses consistent
colour ranges within each column and labels its 8× shaded-relief exaggeration.
Visible ridges and valleys have been added, but broad smooth uplands remain.
Higher spectral power is not a terrain realism verdict. These trials establish
a usable candidate for user visual assessment; **final terrain and another full
meso globe remain unapproved**. Review `regional-comparison.png` before selecting
further terrain changes or an expensive full run.

The user subsequently **provisionally approved the reviewed model pair and
regional terrain results** after viewing the meso zooms. This supersedes the
pending user visual assessment above. `structured-selection.json` records the
approval, the exact frozen checkpoint hashes and its provisional scope; the
regional and relief review records also retain it. Broad smooth uplands remain
a known limitation. No further training or full meso generation was requested
or started as part of recording this approval.
