# Planet layer: decision log

One line per thing that was decided: adopted, rejected, kept, or fixed. Newest at the bottom of each table. The
reasoning, the numbers and the rule that decided it are in the journal entry linked in the last column
(`journal/`, one file per pass, verbatim from the old plan doc). Current settings and scores: `STATE.md`.

Every sweep is in `outputs/planet-calibration/<sweep>/`, with its reviews in `<sweep>/reviews/`.

**How to add a row:** when a pass adopts, rejects, keeps or fixes something, add one row here in the same pass. Give
the parameter or setting, the outcome, the sweep (or "none"), one clause of reason, and the journal file.

## Model settings: adopted, rejected, kept

"Default" means it is the `PARAMS` default in `planet_layer/plates.py`. "Working baseline" means every sweep `.cmd`
passes it explicitly while `PARAMS` still holds the old value (see the warning in `STATE.md`).

| Date | Setting | Outcome | Sweep | Why, in one clause | Journal |
|---|---|---|---|---|---|
| 17 Oct | landscape dt 50 kyr, window 30 Myr | kept | timing run | 30 Myr takes ~12 min at n = 868; testing 100 kyr not worth it | [03](journal/03_2026-10-17_step-4-done.md) |
| 17 Oct | plate erosion θ 8 at n = 217 | adopted, later replaced | calibrate_plate_erosion | won the fixed rule on land hypsometry (θ calibrated at the grid it runs on) | [07](journal/07_2026-10-17_step-6-result.md) |
| 17 Oct | plate merging (1,000 km for 40 Myr; smaller < 35% of larger; result < 150 Mkm²) | adopted | none | plates split but never merged; unconstrained merges made 182–266 Mkm² giants | [06](journal/06_2026-10-17_step-6-continued.md), [07](journal/07_2026-10-17_step-6-result.md) |
| 17 Oct | subduction initiation 0.004 /Myr | adopted | none | ocean ages were young | [07](journal/07_2026-10-17_step-6-result.md) |
| 17 Oct | landscape θ 20 | adopted, later replaced | theta-20261017-152517 | best on denudation rates alone | [10](journal/10_2026-10-17_step-7-result.md) |
| 17 Oct | `crust_kappa_normal_m2_yr` 1000 | kept, later replaced (24 Oct) | single-seed texture test | lowering it cost land (a test confounded by the deep-sea sink) | [12](journal/12_2026-10-17_step-8-meso-and-texture-sweep.md) |
| 18 Oct | water inventory (sea level holds Earth's ocean volume) | adopted, default | none | physical bookkeeping; the fixed datum acted as extra water | [13](journal/13_2026-10-18_step-8-results.md) |
| 18 Oct | dynamic topography 800 m, degree 30, slope 1; landscape θ 10 | adopted, later revised | sweep-20261017-223508 | won the texture score (later found to be passed by round swells) | [13](journal/13_2026-10-18_step-8-results.md) |
| 18 Oct | `dyn_std_m` 500, `sediment_accretion` 0.75, Andes-type margins (H 2.5 km) | adopted, default | sweep-20261018-144253 | best score; accretion matched Earth's absolute subducted volume while the deep-sea sink was live | [14](journal/14_2026-10-18_step-8-mass-bug.md) |
| 18 Oct | `erosion_K_factor` 0.75, θ 8 | adopted, default (K still in PARAMS) | sweep-20261018-164500 | best score; export barely responds to K (supply-limited) | [15](journal/15_2026-10-18_step-8-erosion-sweep.md) |
| 18 Oct | pelagic 4 m/Myr; rift necking zone 150 km | rejected | single seed | no gain | [15](journal/15_2026-10-18_step-8-erosion-sweep.md) |
| 21 Oct | shelves (wave-base planation, cliff retreat) | rejected (later adopted 22 Oct) | sweep-20261021-182714 | seed spread swamps it; one-cell margins give it nothing to act on | [17](journal/17_2026-10-21_pass-02.md) |
| 21 Oct | κ_normal 1000 / 100 / 10 | no change | sweep-20261021-211308 (invalid, towers) | lateral flow did not erase thin margins | [19](journal/19_2026-10-21_pass-04.md) |
| 22 Oct | `breakup_thick_m` 17.5 km, plate θ 6, shelves κ 3·10⁴ R 0.003 | adopted with the user; breakup later replaced | sweep-20261022-194148 | passed the shelf gate; better on almost every measure | [24](journal/24_2026-10-22_pass-08.md) |
| 23 Oct | landscape θ 5 (`run_planet_landscape.THETA`) | adopted, working default | theta-20261022-214812 / 224344 / 234017 | three-seed rule on denudation and 23 km texture | [25](journal/25_2026-10-23_pass-09.md) |
| 23 Oct | margin shortening + accretion 0.13 | rejected by the shelf rule, then working baseline | sweep-20261023-192601 | Earth's margin budget; the old margins built 3–5× Earth's belts | [29](journal/29_2026-10-23_pass-13.md), [32](journal/32_2026-10-23_pass-16.md) |
| 23 Oct | erosion-law test rule: classes by basin slope quartile, not by measured rate | agreed with the user | none | classing by measured rate biases the ratio (regression to the mean) | [31](journal/31_2026-10-23_pass-15.md) |
| 23 Oct | Earth erosion law at 46 km: K × 1.0, θ 200, Sc 0.036 (Kd × 4) | working baseline (PARAMS still 0.75 / 6 / 0.05) | erosion_law_check vs OCTOPUS; sweep-20261023-214324 | the old law is ×8 too fast against ¹⁰Be rates; with Earth margins it gives Earth's export | [31](journal/31_2026-10-23_pass-15.md), [32](journal/32_2026-10-23_pass-16.md) |
| 23 Oct | plate speeds | no change | plate_speeds vs Bird 2003 | Earth-like | [29](journal/29_2026-10-23_pass-13.md) |
| 24 Oct | `collapse_edge` | rejected (kept, default off) | sweep-20261023-231817 | the thick crust is inland, not at the coast | [34](journal/34_2026-10-24_pass-18.md) |
| 24 Oct | `thinning_local_isostasy` True | adopted, default | sweep-20261024-114002 | without it the ocean beside rifts stands km above sea level | [35](journal/35_2026-10-24_pass-19.md) |
| 24 Oct | `continental_sediment` True, `crust_kappa_normal_m2_yr` 30 | adopted, defaults | sweep-20261024-143735 | Earth values (sediment 2,400 kg/m³; lower-crust channel at ~10²¹ Pa s); elevation tests pass | [36](journal/36_2026-10-24_pass-20.md) |
| 24 Oct | `rift_rule` pure_shear, `breakup_thick_m` 10 km, `rift_width_km` 100 | adopted, defaults | sweep-20261024-173200 | margin width 70 → 141 km of thin crust per km of edge (band 100–200) | [37](journal/37_2026-10-24_pass-21.md) |
| 25 Oct | continental crust read label-free (total ≥ 20 km, both sides) | agreed with the user | none | reproduces CRUST1.0's own continent on Earth | [38](journal/38_2026-10-25_pass-22.md), [39](journal/39_2026-10-25_pass-23.md) |
| 25 Oct | deforming-crust threshold 3.5 nanostrain/yr (equal-area criterion) | adopted | earth_deformation (GSRM v2.1) | criterion fixed before reading the planet | [40](journal/40_2026-10-25_pass-24.md) |
| 25 Oct | `start_crust` earth, `orogen_rule` belt | adopted, defaults | sweep-20261024-193840 | won the interior rule; neither thins the thick tail (the high land is plateau) | [41](journal/41_2026-10-25_pass-25.md) |
| 25 Oct | E: extension of thick unheld crust | rejected (default off) | sweep-20261024-215311 | 0 of 4 E tests; the "held" rule shields the plateaus | [44](journal/44_2026-10-25_pass-28.md) |
| 25 Oct | E2: held pieces cut across, hot clock | rejected (default off) | sweep-20261025-111119 | trigger reach 0.01–0.04; collapse flow spreads crust untimed | [46](journal/46_2026-10-25_pass-30.md) |
| 25 Oct | E3: clock carried by flow, band cut, pair motion union | rejected (default off) | sweep-20261025-141028 | 1 of 4 E tests; 60–70% of events lost when the microplate docks | [48](journal/48_2026-10-25_pass-32.md) |
| 26 Oct | `boundary_normal` ring | adopted, default | sweep-20261025-170555 | node typing misread transforms; ring passes gates on 3 of 3 seeds | [50](journal/50_2026-10-26_pass-34.md) |
| 26 Oct | E4: free side moves | rejected (default off) | sweep-20261025-170555 | 0–1 of 4 E tests; ledger guard fails | [50](journal/50_2026-10-26_pass-34.md) |
| 26 Oct | C (lithosphere buoyancy) narrowed | re-scoped | earth_lithosphere_residual | Earth's orogens stand on the planet's own Airy line: the slope pair is not C's | [50](journal/50_2026-10-26_pass-34.md) |
| 26 Oct | `sediment_compaction` True (B-a) | adopted, default | sweep-20261025-201623 | deep floor 4,524 → 4,317 m (Earth 4,205); gates pass | [53](journal/53_2026-10-26_pass-37.md) |
| 26 Oct | `piece_rule` plates (B-b) | adopted, default | sweep-20261025-220434 | detached plate pieces 6–10% → < 0.4% of the surface; T2–T5 closer to Earth | [54](journal/54_2026-10-26_pass-38.md) |
| 26 Oct | `start_plates` coast, `start_poles` forces | rejected (defaults voronoi / random) | sweep-20261026-184745, sweep-20261026-200857 | fails T5 (ridge 71k vs 91k km) and gates on 2 of 3 seeds | [56](journal/56_2026-10-26_held-out-seeds-check.md) |
| 27 Oct | `piece_rule` plates on six seeds | kept, as a recorded exception to the .cmd's gate clause | sweep-20261026-204140 | the one gate miss came from a 0.064%-of-surface snapshot | [57](journal/57_2026-10-27_pass-39.md) |
| 27 Oct | `arc_H_ocean_m` 1,000 m (candidate), `arc_accretion` (built) | not swept; Arc Rate Sweep withheld | cloud pre-checks | docking made the plate layer run away on one seed | [57](journal/57_2026-10-27_pass-39.md) |
| 28 Oct | Arc Dock Probe | read; no setting changed | sweep-20261027-210209 | no runaway on six seeds; arc flux 2× Earth's per km because the band law over-counts | [58](journal/58_2026-10-28_pass-40.md) |
| 29 Oct | B-d2 arm B: `trench_polarity` pair alone | rejected by the rule (default 'node') | sweep-20261028-200644 | fails T4, T5 and the margin guard against the control (small misses, inside seed spread) | [59](journal/59_2026-10-29_pass-41.md) |
| 29 Oct | B-d2 arm C: pair + `trench_flux` consumed + `arc_H_ocean_m` 1,000 + `arc_accretion` | rejected by the rule (defaults unchanged) | sweep-20261028-200644 | passes A1–A3 and moves volume and water toward Earth, but fails T2–T4, > 45 km and margins | [59](journal/59_2026-10-29_pass-41.md) |

## Correctness fixes (no choice involved)

| Date | Fix | Effect | Journal |
|---|---|---|---|
| 17 Oct | subduction erosion on oceanic overriding plates removes only arc crust above 7 km | removed −10 km pit trails on the ocean floor | [09](journal/09_2026-10-17_step-7-first-full-run.md) |
| 17 Oct | coupling: continental forcing only where the parcel is continental at both ends of an interval | removed false island arcs up to 2 km high | [09](journal/09_2026-10-17_step-7-first-full-run.md) |
| 18 Oct | `basin_to_continent` False | basins became "continent" without mass (12.8% of the surface over a run) | [14](journal/14_2026-10-18_step-8-mass-bug.md) |
| 21 Oct | `keep_deep_sediment` True (deep-sea sediment kept, not deleted) | ~7–9 km³/yr had vanished | [17](journal/17_2026-10-21_pass-02.md) |
| 22 Oct | `fill_and_spill` replaces the first deep-sea spread | the first fix built 100–250 km sediment towers; sweeps -201744, -211308, -230144 invalid | [20](journal/20_2026-10-22_pass-05.md) |
| 23 Oct | `elevation_shading.CMAP` N = 65,536 | the "staircase" plains were a 256-level colour map | [26](journal/26_2026-10-23_pass-10.md) |
| 24 Oct | `elevation_check` sorts snapshots by time, not as text | "last 40 Myr" had been the first 40 | [35](journal/35_2026-10-24_pass-19.md) |
| 24 Oct | `ocean_crust_erosion` True (ledger) | ocean floor eroded below its sediment now loses crust; leak of 8 km³/yr closed | [34](journal/34_2026-10-24_pass-18.md) |
| 25 Oct | `boundary_normal` ring (typing from the two-ring) | transforms were typed ridge / collision / subduction since step 7 | [49](journal/49_2026-10-25_pass-33.md) |
| 25 Oct | `ext_pair_motion` union added without asking first (confirmed by the user) | band strips took the collision boundary and sped the parent up | [47](journal/47_2026-10-25_pass-31.md) |
| 26 Oct | ledgers: consumed nodes booked when they leave; `cont_lost_other` double count removed | crust and sediment ledgers close to 1e-16 | [54](journal/54_2026-10-26_pass-38.md) |
| 27 Oct | arc-to-continent booked once per plate node | double booking (ledger only) | [57](journal/57_2026-10-27_pass-39.md) |
| 27 Oct | docked crust on oceanic plates spread over the arc band | one-node continents speckled the ocean | [57](journal/57_2026-10-27_pass-39.md) |
| 28 Oct | trench-flux over-count and ocean–ocean polarity defect found (fix built, default off; not adopted 29 Oct, defect remains) | band law 1.8–2.7× the consumed area; polarity wrong at 22% of arc nodes | [58](journal/58_2026-10-28_pass-40.md) |

## Process and review rules

| Date | Rule | Journal |
|---|---|---|
| 17 Oct | Calibrate at the grid spacing you run at; scripts refuse n < 96 | [06](journal/06_2026-10-17_step-6-continued.md) |
| 17 Oct | Texture is part of the check for any erosion setting | [11](journal/11_2026-10-17_step-8-first-pass.md) |
| 22 Oct | Runs are deterministic on the user's machine; "seed spread" is chaotic sensitivity, not noise | [19](journal/19_2026-10-21_pass-04.md) |
| 23 Oct | Realism: values from Earth measurements or mechanism, never fitted to a band (user) | [28](journal/28_2026-10-23_pass-12.md) |
| 23 Oct | Do not judge a law on a transient | [28](journal/28_2026-10-23_pass-12.md) |
| 24 Oct | Sweep folders named only by the parameters that differ (`MAX_NAME` 80) | [37](journal/37_2026-10-24_pass-21.md) |
| 26 Oct | Decisions go to the user only when very important (user) | [54](journal/54_2026-10-26_pass-38.md) |
| 26 Oct | Effects smaller than the drift between runs are measured paired on the same states | [52](journal/52_2026-10-26_pass-36.md) |
| 27 Oct | Sweeps decide on six seeds (1001–1006) (user) | [57](journal/57_2026-10-27_pass-39.md) |
| 27 Oct | `elevation_check` offsets pool the window's nodes (area-weighted median) | [57](journal/57_2026-10-27_pass-39.md) |
| 28 Oct | Docs reorganized: STATE / decisions / journal replace the single plan doc | (this reorganization) |
