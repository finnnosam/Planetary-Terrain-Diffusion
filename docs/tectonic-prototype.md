# Tectonic + landscape-evolution prototype (plan of record, 14 Oct 2026)

## Why the direction changed

Re-evaluation of the whole pipeline (not just the structural stage):

- **Below ~40 km everything works.** The meso model (23 → 2.9 km) is trained on real Earth averages with no
  tricks and reproduces Earth's spectrum; every oracle test shows it renders credible terrain when given credible
  23–40 km structure.
- **Above ~100 km the data cannot teach structure.** PaleoDEMs are smooth reconstructions (perhaps 15–30
  distinct layouts); modern Earth has a few dozen orogen systems. Learned coarse stages therefore produce smooth
  guides (generated land 95th percentile ≈ 1.1 km), and every learned 40 km "planner" has produced ramps.
- **The constraints added to cope** (guide degradation, adapted/translated targets, the 200 m lock, frozen
  land/ocean sign, 645 km mean conservation, "feasible targets") stop the models training on Earth, pin the coastline
  at 161 km resolution and create block artifacts. They are not physical and are dropped.
- The planner/renderer split was redundant: the "plan" was the 40 km terrain under another name.

So: **mechanism above ~100 km, learning below.** This was the recommendation of the 4 October scale review
(layer A, procedural tectonics) and had not been built. The earlier `docs/structural-invention-plan.md` and the
long planner run it describes are superseded.

## What the prototype does

`planet_training/landscape_evolution.py` evolves a 2,949 km window at 11.52 km (256 × 256) in metres and years.
Each 50 kyr step:

1. **Tectonic rock uplift** from a template (`planet_training/tectonic_templates.py`).
2. **Flow routing**: open sea = water connected to the sea (a landlocked cell below 0 m is a basin, not ocean);
   depressions are filled to their spill level (priority flood); D8 steepest descent; drainage area.
3. **Erosion and deposition together** (Yuan et al. 2019 implicit erosion–deposition, n = 1): steep upland rivers
   incise, large lowland rivers become transport-limited and aggrade. Cell-to-cell slopes are capped at 0.05
   (landslide limit at this cell size).
4. **Sediment**: fills closed depressions towards their spill level; at the coast it builds deltas (the river
   mouth aggrades to sea level so the coast progrades) and moves downslope underwater to build shelves and fill
   shallow seas.
5. **Hillslope and marine diffusion.**
6. **Flexural isostasy** (elastic thickness 30 km, flexural parameter ≈ 67 km): mountains bend the plate down
   beside them, which is what makes foredeeps such as the Po plain; erosion unloads and lifts the crust back up.

Sea level is the 0 m datum only. Nothing is frozen, conserved or projected.

**Drainage is part of the process, not an add-on.** Every run exports drainage area, rivers, a drainage-basin label
for every cell (its outlet), lakes / closed basins (depth below spill level), and whether each cell drains to the
sea or out of the window. Current limitation: water always fills a depression until it overflows (a humid
climate), so true endorheic basins (Great Basin, Caspian) never form; that needs a precipitation–evaporation balance,
a natural next step.

### Templates (mechanisms, not landforms)

- **collision** — continent–continent convergence on a curved suture: asymmetric uplift belt, sea beyond the
  southern foreland; in ~75% of seeds a younger oblique belt rises on the far side of the foreland after 3–5 Myr
  (Apennine type), so a lowland can end up between two ranges.
- **margin** — ocean–continent subduction: segmented coast range, slowly subsiding forearc, tilted arc block with a
  steep back side (Sierra type), back-arc plateau with basin-and-range blocks.

Templates are seeded; each seed varies geometry, rates and erodibility within plausible ranges.

## First results (CPU, 10 Myr, seeds 1–3)

Figures: `outputs/tectonic-prototype/cpu-preview/index.html`. Each run takes ~10 s with numba.

| Run | land p50 / p95 / max (m) | long ridges | flanked lowland | median 322 km contrast |
|---|---|---|---|---|
| collision s1 (two belts) | 102 / 840 / 5,884 | 1.00 | 0.07 | 185 m |
| collision s2 (one belt) | 107 / 571 / 5,393 | 1.00 | 0.03 | 65 m |
| collision s3 (two belts) | 96 / 832 / 5,748 | 1.00 | 0.06 | 143 m |
| margin s1 | 182 / 1,017 / 4,720 | 0.82 | 0.10 | 70 m |
| margin s2 | 158 / 818 / 6,153 | 0.88 | 0.12 | 106 m |
| margin s3 | 127 / 817 / 3,867 | 0.77 | 0.11 | 280 m |
| **Earth, unconstrained 40 km:** Alps / California / Plains | | 0.61 / 0.74 / 0.44 | 0.19 / 0.32 / 0.10 | 930 / 1,493 / 759 m |

What works: organized belts with drainage divides, forelands and lakes beside them, a lowland between two belts
with an axial river, a coast range / forearc / arc sequence, rivers integrating across thousands of km to the sea
or out of the window, peaks of 4–6 km.

What is wrong, and why:
- **Plains are far too featureless** (window contrast 65–280 m vs Earth 760–1,500 m). Real continents carry
  inherited relief everywhere: older worn-down ranges, rifts, cratonic basins, several overlapping tectonic events.
  The templates start from a flat 250 m plain with one event.
- **Belts are too tidy** (long-ridge share 0.8–1.0 vs Earth 0.4–0.7) and look tube-like at 11.5 km; one uplift
  curve per belt is too simple.
- **The forearc stays a shallow sea** in the margin runs (sediment-limited), like Puget Sound rather than the
  Central Valley.
- Regional window only, flat (not on the sphere), humid drainage only.

## Meso results (GPU run 20261014-223854)

`outputs/tectonic-prototype/20261014-223854/index.html`; four 368 km windows per run, two meso seeds each.

- **Mountains: convincing.** The meso turns simulated belts into ranges with branching ridges and valleys, foothills
  and range fronts dropping onto plains (collision s1 / s3 high-relief windows, margin arc fronts). The domain gap
  between simulated and Earth 23 km input does not show in mountain windows.
- **Coasts and straits: plausible.** Margin windows read as a coast range island, a strait and an arc front;
  collision coasts and a peninsula look natural, though some 23 km stair-steps survive in the shoreline.
- **Plains: dead flat.** Centre windows are almost featureless, because the simulated plains are (see "What is
  wrong"). The meso adds little to a flat input, as it should; the fix belongs in the simulation.
- **Belts are single and uniform**: parallel, comb-like ridge texture along one smooth belt, not the segmented
  complexity of the Alps.
- **Shelves are a uniform -60 m**, an artefact of the delta/shelf fill rule.
- The "lowland-between-ranges" window rule mostly picks mountain windows; it needs a better score.

Verdict: the direction works; mechanism above ~100 km plus meso below produces credible terrain where the
simulation has structure. Next work is in the simulation: inherited relief, multiple and segmented tectonic events,
realistic shelf profiles.

## Version 2 (15 Oct): inherited relief, segmented belts, graded shelves

Changes, all in the simulation:

- **Inherited history.** Each run first simulates 20 Myr of older events: 1-2 older orogens that rise and then stop
  and wear down, usually a rift (subsiding trough with raised shoulders), and slow mantle-driven swells and basins.
  The continent starts at ~500 m instead of 250 m. The young events then run 10 Myr on that aged landscape
  (30 Myr total, ~30 s per run). `--no-inherited` reproduces the flat-plain start.
- **Harder old crust.** Erodibility is ~3x lower on stable crust than in young belts (~7x since version 3, below), with patchy hardness at
  ~140 km and ~460 km, so differential erosion leaves uplands, escarpments and worn old ranges.
- **Segmented belts.** Young belts are built from 3-5 offset (en-echelon) segments, each with optional outer and
  inner sub-ranges, so longitudinal valleys can form; the margin arc block steps along strike.
- **Graded shelves.** Marine sediment fills to a shelf surface that deepens ~1.2 m/km offshore, out to a shelf break
  150 km from land; beyond it sediment passes to the deep sea. (Replaces the flat -60 m shelves.)
- Underwater sediment routing now uses a height-ordered queue (no sediment is lost by processing order).
- The "lowland-between-ranges" meso window now looks for a low land core with high ground on opposite sides.

| Run (CPU, 30 Myr) | land p50 / p95 / max (m) | long ridges | flanked lowland | median 322 km contrast |
|---|---|---|---|---|
| collision s1 | 201 / 1,193 / 5,247 | 0.92 | 0.11 | 834 m |
| collision s2 | 165 / 920 / 4,644 | 0.81 | 0.08 | 323 m |
| collision s3 | 243 / 1,324 / 6,354 | 0.87 | 0.08 | 403 m |
| margin s1 | 224 / 769 / 3,617 | 0.58 | 0.10 | 541 m |
| margin s2 | 245 / 787 / 7,798 | 0.57 | 0.14 | 482 m |
| margin s3 | 225 / 745 / 4,913 | 0.65 | 0.12 | 317 m |
| *version 1, same seeds* | | 0.77-1.00 | 0.03-0.12 | 65-280 m |
| **Earth** Alps / California / Plains | | 0.61 / 0.74 / 0.44 | 0.19 / 0.32 / 0.10 | 930 / 1,493 / 759 m |

Window contrast rose 3-5x and the margin's long-ridge share is now Earth-like. Remaining: collision belts are still
tidier than Earth (0.8-0.9 long-ridge share); flanked lowland is still below Earth; margin seed 2 builds one 7.8 km
massif where arc segments overlap (too high); closed basins still always overflow (humid assumption).
CPU figures: `outputs/tectonic-prototype/v2-cpu-preview/index.html`.

## Version 2 meso results (GPU run 20261015-190930) and the next question

Better than version 1: coasts and islands are the strongest part (peninsulas, straits, a coast-range island);
Alps-type fronts with long foredeep lakes and branching valleys; the interior has uplands and worn old ranges
instead of nothing. Still weak: plains read as soft gradients at 2.88 km, without the incised river-valley texture
real plains show; some dead-level terraces from basins filled to a flat surface; 23 km stair-steps on some coasts.

**Is the meso the reason plains are smooth?** `scripts/diagnose_meso_plains.py` (no training, a few minutes)
runs the meso on held-out Earth patches, where the true 2.88 km terrain is known, and on simulated windows, by
23 km relief class (flat < 300 m, plain/upland 300-800 m, hills 800-2,000 m). It measures 5.8-46 km band power,
river-channel incision and fine roughness. Decision rule, fixed in advance: if on Earth flat and plain/upland
patches meso/real >= 0.7 for both incision and 5.8-11.5 km power, the model is adequate and the fix belongs in the
simulation (finer-scale erosion). Otherwise a new model (for example one also given the river network) could
help, and it is only adopted if it beats the current meso on this same held-out test.

**Result (20261015-193134, 24 held-out Earth patches per class, 2 seeds):**

| Channel incision / 5.8-11.5 km power, generated ÷ real | flat | plain/upland | hills |
|---|---|---|---|
| adapted meso (fine-tuned for smoothed learned guides; used until now) | 0.34 / 0.18 | 0.57 / 0.55 | 0.70 / 0.56 |
| original Earth-trained meso `outputs/meso/run2-lowlr/latest.pt` | **1.00 / 1.06** | **1.05 / 1.73** | 0.97 / 1.00 |

No new model is needed. The adaptation, made for the old smoothed learned guides, cost the meso its plain and
upland texture. The simulation produces true area-averaged 23 km fields, which is exactly what the original model
was trained on, so `simulate_tectonic_region` now uses run2-lowlr by default (`--meso-checkpoint` to override).
Watch-point: run2-lowlr puts ~1.7x Earth's 5.8-11.5 km power into plain/upland patches (slightly rough).
Also note real Earth plains are themselves fairly smooth at 2.88 km; part of the "smooth plains" impression is real.

## Version 3: too much of the land was plain (23 km comparison with Earth, no GPU)

With run2-lowlr the simulated plains (run 20261015-195458) were still smooth at 2.88 km: on the land windows that
were plains, incision was 0.35-0.41 of Earth's and roughness 0.39-0.52. `scripts/compare_plain_inputs.py` compares
the 23 km inputs themselves, modern Earth (intermediate-stage dataset, |lat| <= 60) against the simulation:

- **Within a relief class the simulated plains already look like Earth's at 23 km**: undulation 0.86-1.05,
  23 km channel incision 0.84-1.09, slope within ~15%. So the plains are not under-textured for their class.
- **The difference is how much of the land is plain.** For 368 km land windows (one meso window):

| p95-p5 relief of 368 km land windows | p10 / p25 / p50 / p75 | < 300 m | < 100 m |
|---|---|---|---|
| Earth | 133 / 237 / 456 / 872 m | 34% | 6% |
| version 2 | 63 / 109 / 186 / 466 m | 67% | 22% |
| **version 3** | **175 / 245 / 370 / 843 m** | **37%** | **1%** |

  The interior was worn down to a low, nearly flat surface (land median ~220 m vs Earth's 366 m). Late-stage
  denudation of stable land was already Earth-like (median ~7 m/Myr), so the problem was not erosion running too
  fast but nothing keeping interior relief alive.

**What was tried** (6 runs each, 30 s per run on CPU): less hillslope diffusion (Kd 10, 2) and less deposition
(G 0.5) did not help or made it worse; crustal provinces at Airy-isostatic height (thickness std 3-5 km) did not help
(the thick-crust plateaus wear down; thin-crust lowlands go flat); swell patterns that change every ~10 Myr help
less than one persistent pattern. What works is the two mechanisms that maintain relief on real continents:

- **Stronger mantle swells and basins:** 22-40 m/Myr uplift/subsidence amplitude (was 5-9). Observed dynamic
  topography rates are ~10-100 m/Myr.
- **Harder stable crust:** erodibility 0.15 of young belts (was 0.3). Old crystalline rock erodes much more slowly
  than the young, fractured and sedimentary rock of active belts; old orogens and escarpments survive longer.

Both are template settings (`SWELL_RATE_YR`, `STABLE_ERODIBILITY` in `tectonic_templates.py`). 40 km arrangement
also moved toward Earth: contrast 483 -> 869 m (Alps 930), flanked lowland 0.10 -> 0.16 (Alps 0.19), long
ridges 0.73 -> 0.68 (Alps 0.61), slope p90 11.1 -> 12.9 m/km (Alps 12.0, California 12.4).

New visible issue: subsiding swell basins hold larger lakes, filled to their spill level; on these flat fills the
routing runs in straight grid lines (rectangular lake edges in the maps). This is the same dead-level-fill problem
as before, now more common; the water-balance step (closed basins that do not fill) and sloping fills address it.
The 2.88 km check (meso on the new plains) is the GPU command under "Run it".

## Version 3 meso check (GPU run 20261015-205816, diagnosis 20261015-210612): the fine cue is missing

The tectonic templates are only a test bed for the processes; the findings below are about what any generator must
hand the meso, not about tuning the templates (user, 15 Oct).

The meso is still fine on real Earth input (incision 1.00 / 1.05, verdict unchanged), but on simulated windows it
gives about a third of Earth's detail **at the same relief class**: incision 10.6 vs 22.0 m (flat), 18.4 vs 56.6 m
(plain/upland), 51 vs 141 m (hills). The meso outputs show stepped flats with scarps rather than valley networks.

Compared with the meso's own validation inputs, the simulated 23 km fields have the right relief and roughly the
right neighbour differences, but far less **cell-to-cell curvature** (|Laplacian| at 23 km):

| median per 368 km window | flat | plain/upland | hills |
|---|---|---|---|
| Earth (meso validation input) | 31 m | 68 m | 179 m |
| simulation, 11.52 km cells | 17 m | 34 m | 66 m |
| simulation, 5.76 km cells (`--dx 5760`, ~3 min per run) | 20 m | 44 m | 81 m |
| Earth input blurred by 0.7 cells | 14 m | 36 m | 82 m |

On Earth that curvature is the 23 km footprint of real valleys and ridges 20-50 km across. A landscape model at
11.52 km (Nyquist 23 km) cannot make them, and halving the cell size closes only about a quarter of the gap; less
hillslope diffusion makes it worse (more dead-flat fills). The simulation looks like Earth seen through a ~0.7-cell
blur, and the meso, which adds detail as a residual on the upsampled input, reads a soft input as soft terrain.

**Decisive test (GPU, a few minutes):** `diagnose_meso_plains --blur-earth 0.7` runs the meso on Earth input blurred
that much. If its detail drops to the simulated level, the fine cue is the cause, and the fix belongs at the
meso boundary rather than in any particular generator: fine-tune the meso with randomly softened inputs so it
generates Earth-like detail when the input does not carry it. Rule, fixed now: the fine-tuned model is adopted only
if, on held-out Earth, blurred (0.7) flat and plain/upland input gives incision and 5.8-11.5 km power >= 0.7 of
real while unblurred input still passes. (Generating at <= 5.76 km everywhere is the costly alternative.)

**Result (20261015-213919): confirmed.** On Earth input blurred by 0.7 cells, the original meso keeps 0.30 / 0.37 /
0.32 of real incision (flat / plain-upland / hills) and 0.04-0.16 of the 5.8-11.5 km power: the same level as on
simulated windows (incision 10.6 / 18.4 / 51 m vs 6.5 / 20.8 / 46 m on blurred Earth). The missing fine cue explains
the smooth simulated plains.

**Fix in progress: a detail-aware, softening-robust meso** (`toy_meso/finetune_detail.py`, fine-tuned from
run2-lowlr, ~2 h on the GPU). Two parts, following the user's suggestion of extra inputs alongside the height map:

- *Softened training input:* 60% of training conditions are blurred (sigma 0.3-1.1 cells) while the target stays
  the real 2.88 km terrain, so the model must produce Earth-like valleys when the input does not hint at them.
- *Detail input:* a per-cell "sub-cell detail" map (std of the 2.88 km terrain about the smooth 23 km surface,
  metres; `toy_meso/detail.py`) plus a known/unknown mask. Hidden 40% of the time, so the model works without it
  and picks a plausible amount itself; when a generator can say "rough here, smooth there" it can pass the map.
  New input weights start at zero, so the starting point is exactly run2-lowlr.

Adoption rule (fixed before training): `diagnose_meso_plains --blur-earth 0.7` must report both "plain test passes"
and "robust to softened input" (blur 0.7, detail unknown: flat and plain/upland incision and 5.8-11.5 km power
>= 0.7 of real). The run also reports how close it gets with the true detail supplied. Later candidates for further
inputs: a roughness *type* (dissected, dunes, karst, glacial: needs labels or a learned code), drainage area /
rivers, climate (the multi-input meso).

**First check at fine-tune step 10,000 (diagnosis 20261015-230146): passes both parts of the rule.**

| incision / 5.8-11.5 km power, generated ÷ real | flat | plain/upland | hills |
|---|---|---|---|
| original meso, real input | 1.00 / 1.06 | 1.05 / 1.73 | 0.97 / 1.00 |
| original meso, blurred (0.7) input | 0.30 / 0.04 | 0.37 / 0.16 | 0.32 / 0.07 |
| fine-tuned, real input | 1.12 / 1.15 | 1.10 / 1.69 | 0.96 / 0.94 |
| fine-tuned, blurred input, detail unknown | 1.35 / 1.84 | 1.20 / 1.92 | 0.88 / 0.77 |
| fine-tuned, blurred input, true detail given | 1.16 / 1.58 | 1.18 / 2.03 | 0.97 / 1.08 |

On the simulated windows river incision rose from 10.6 / 18.4 / 51 m to 24.2 / 40.7 / 107 m (Earth 22.0 / 56.6 / 141).
Watch-point: with blurred input and no detail it now slightly over-textures plains (5.8-11.5 km power ~1.9x; the
original already had 1.7x on plain/upland with real input). Copy kept as `outputs/meso/detail-ft/step10000-passed.pt`.

**Full run (48,000 batch-16 steps, diagnosis 20261016-203650): passes, but overfits; step 10,000 adopted.**

| incision / 5.8-11.5 km power, generated ÷ real | flat | plain/upland | hills |
|---|---|---|---|
| step 10,000, blurred input, detail unknown | 1.35 / 1.84 | 1.20 / 1.92 | 0.88 / 0.77 |
| step 48,000, blurred input, detail unknown | 1.39 / 2.06 | 1.21 / 2.34 | 0.92 / 0.94 |
| step 10,000, blurred input, true detail | 1.16 / 1.58 | 1.18 / 2.03 | 0.97 / 1.08 |
| step 48,000, blurred input, true detail | 1.04 / 1.12 | 1.09 / 1.61 | 0.96 / 1.01 |

Simulated-window incision at step 48,000: 26 / 45 / 133 m (Earth 22 / 57 / 141). But held-out validation loss rose
steadily after ~12-24k steps (real input 0.2489 -> 0.2595, +4%) while training loss kept falling: the later model fits
these few measured statistics better but Earth terrain as a whole worse, and over-textures soft input more when no
detail is given. Step 10,000 (`outputs/meso/detail-ft/step10000-passed.pt`) is now the default meso of
`simulate_tectonic_region` and `diagnose_meso_plains`; step 48,000 stays as `latest.pt`. The fine-tune now also keeps
`best.pt` (lowest mean validation loss), so a future run (for example once a generator supplies detail maps) can
stop at the right point.

Reading: with the true detail map the model is close to Earth everywhere; without it, it must guess and leans rough,
because 60% of its training inputs were softened. Supplying detail from the generator is therefore worth doing later.

## Run it

Install numba once (without it the simulation runs in plain Python, ~7 minutes per run instead of ~10 s):

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -m pip install numba
```

Simulate both scenarios, seeds 1–3, and refine four 368 km windows per run with the selected meso
(highest relief, most lowland-between-ranges, a coastal window, the centre), two meso seeds each. About 2–4 minutes
on the GPU (~30 s per simulated run with numba, plus meso):

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.simulate_tectonic_region --meso
```

(or **Simulate Tectonic Region.cmd**). Then measure the meso on the new run's plains against held-out Earth
(**Diagnose Meso Plains.cmd**, a few minutes):

```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.diagnose_meso_plains --blur-earth 0.7
```

Fine-tune the detail-aware meso (~2 h; rerun the same command to resume), then test it against the original:

```powershell
& '.\.venv\Scripts\python.exe' -u -m toy_meso.finetune_detail
& '.\.venv\Scripts\python.exe' -u -m scripts.diagnose_meso_plains --meso outputs\meso\detail-ft\latest.pt --compare outputs\meso\run2-lowlr\latest.pt --blur-earth 0.7
```

The 23 km comparison needs no GPU: `python -m scripts.compare_plain_inputs` (newest run). Simulation output: `outputs\tectonic-prototype\<timestamp>\index.html`. Options:
`--scenarios collision`, `--seeds 1,2,3,4`, `--years 40e6`, `--no-inherited`, `--param Sc=0.04` (any parameter in
`landscape_evolution.DEFAULTS`).

## Planet-scale layer (next)

Research survey for the planet layer (plates, global landscape evolution, existing tools, Earth validation targets):
`docs/research/planet-scale/README.md`; prompt for the next chat: `docs/research/planet-scale/next-chat-prompt.md`.

**Design (proposed 17 Oct, awaiting agreement): `docs/planet-layer-plan.md`.** Plate layer at 46 km over ~250 Myr
(Cortial rules on per-plate frames), our landscape model ported to the cube sphere at 11.53 km for the last ~30 Myr in
present-day geometry with parcel histories, exact area-mean hand-off to `toy_meso.planet_tiled`.

## Next steps, in order

1. **Look at the meso windows.** The decisive question for this direction: does simulated 23 km structure become
   convincing 2.9 km terrain through the existing meso, or does the domain gap (simulated vs Earth statistics) show?
2. ~~Inherited relief and multiple events~~ and 3. ~~segmented belts~~: first versions done (version 2); keep
   judging them through the meso.
4. **Water balance** (precipitation vs evaporation) so closed basins can stay closed. Later: a **climate layer**
   (precipitation, temperature) that would also drive erodibility and the water balance.
6. **Future: a multi-input meso.** Condition the fine model on the simulation's actual rivers / drainage area,
   climate and similar fields as well as height, so fine detail follows real rivers rather than river-like texture
   inferred from heights. Priority is lower than fixing the stages before it (user, 15 Oct).
5. Only then: plates and boundaries on the cube sphere (Euler poles, boundary classification) to place templates
   automatically, and a whole-planet run feeding meso.

> **Note (user, 15 Oct):** before building a serious plate-tectonic simulation, search online first — there is
> likely a lot of existing work (papers and open-source code) to reuse or learn from. Right now the goal is only to
> make sure every process in the chain works.
