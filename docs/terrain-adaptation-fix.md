# Terrain detail adaptation, 11 October 2026

The failure was a training/inference mismatch. Both refinement stages learned
from real Earth guides with considerably more local variation than the generated
continent/intermediate fields. Controlled smoothing tests with matched sampling
noise suppressed learned detail in both stages. Continent generation statistics
did not show a comparable obvious failure; the continent checkpoint is retained.

## Change and selection

Both stages now have a separate warm-start experiment with guide degradation.
Three quarters of training guides are smoothed in physical metres; a quarter
remain clean. Targets retain the original real encoded terrain residual above
its original guide, placed over the smoother guide. This teaches detail synthesis
without asking the network to reverse the broad guide smoothing.

Filtering uses real surrounding guide cells. Bridge preparation verifies original
patch orientations and the expanded geographic holdout footprints. Fine meso
preparation crops the original saved 2.88 km targets and reuses their saved guide
halo; it does not use the coarse ETOPO cache as a fine-resolution target.

| Stage | Source checkpoint | Selected adaptation | Training examples | Held-out examples |
|---|---|---|---:|---:|
| Intermediate | selected-10000/best.pt | selected-adapted-1750/best.pt | 5,248 | 55 expanded-footprint patches; all 512 original clean patches also checked |
| Meso | run2-lowlr/latest.pt, update 222,000 | selected-adapted-1000/best.pt | 16,000 | 512 crops and their original full-size patches |

Selections are frozen separately from resumable runs. Both retain source
normalization and architecture, initialize from EMA weights and use a fresh
optimizer. Meso training crops are 80x80; ordinary 128x128 inference tiles are
retained and checked through the existing regional sampler.

## Verification

Intermediate smooth-guide validation loss improves by 10.4%; the full clean
Earth holdout loss worsens by 1.6%. Sampled 46–323 km wavelength power rises
roughly 3.6–4.8 times on strongly smoothed guides, while still remaining below
the paired target power. The same continent layout and bridge seed produce the
new full intermediate TIFF at `outputs/intermediate/adapted-planet42/planet.tif`.
Land fraction is 15.48% versus 15.40% in the initial layout. RMS disagreement
between averaged intermediate elevations and layout cells increases from 69 to
119 metres; stronger relief is not perfectly constrained to its guide averages.

Meso's full-size clean Earth holdout loss increases from 0.25248 to 0.28486
(12.8%). This adapted checkpoint is intended for smooth generated guides;
retain the original checkpoint for real Earth inputs. Cropped mixed-guide loss
improves from 0.30834 to 0.29913 (3.0%). The complete clean/smooth sampling
comparison is saved at `outputs/meso/run3-adapted/evaluation`. Strongly smoothed
guide loss improves from 0.39930 to 0.31391 (21.4%).
On those guides, sampled 5.76–23.04 km wavelength power rises from 1.8–2.9%
to 10.8–12.3% of paired real-target power. This is a substantial improvement,
but the first adapted selection still underproduces fine relief on that test.

Three 256x256 regional generations use the same original planet grid and meso
noise seed (1135849674), so differences are not output resolution or reseeding.

| Region | Fine wavelength power versus original pipeline | Added relief RMS, original meso on adapted input | Added relief RMS, adapted meso |
|---|---:|---:|---:|
| Southern highland | 5.10x | 41.6 m | 85.7 m |
| Northern highland | 5.84x | 30.6 m | 76.2 m |
| Lowland | 3.35x | 1.49 m | 5.10 m |

Fine wavelength bands are nominal 6–23 km, measured on equirectangular regional
exports. East–west physical spacing varies with latitude. Higher power does not
prove terrain realism. Visual comparisons show stronger ridges and valleys;
lowlands remain comparatively gentle. These models do not enforce drainage
connectivity, and narrow rivers are below 2.88 km cell resolution. Evidence covers
one planet, one noise seed and three selected regions, rather than every planet.

Comparisons and regional TIFFs:

- `outputs/meso/adapted-both-planet42/combined-comparison.png`: user's original result versus both adaptations.
- `outputs/meso/adapted-both-planet42/comparison.png`: original versus adapted meso with the same adapted intermediate input.
- `outputs/meso/adapted-both-planet42/verification.json`: regional measurements.
- `outputs/intermediate/run2-adapted/evaluation`: paired held-out bridge comparisons.

All 79 targeted checks pass: adaptation/preparation/resume, existing planet
training, meso conditioning/tiled generation, launch settings and GUI behavior.
Compilation succeeds. No original checkpoint, dataset or user's output is replaced.

## Use in the existing UI

Open **Launch Meso.cmd**, choose **Load settings**, then select
`outputs/meso/adapted-planet42-meso-launch.json`. It selects the frozen adapted
meso checkpoint, the regenerated intermediate TIFF, CUDA batch 16 and a fresh
output folder at `outputs/meso/planet42-adapted-full`.

The full final meso globe has not been regenerated; the three matched regional
outputs above verify the change before another hour-long run. The UI preset is
ready to generate it. Its frozen checkpoint remains stable if training continues.

**Train Intermediate Adapted.cmd** and **Train Meso Adapted.cmd** resume the
separate experiments with early stopping. Their update limits are ceilings.
Inspect clean-input regression and regional results before replacing frozen
selections with later checkpoints; lower training loss alone is insufficient.
