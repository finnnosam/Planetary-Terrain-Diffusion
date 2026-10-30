# Next chat: planet layer, item B: B-d2 built (trench fluxes from consumed area, ocean-ocean polarity per plate pair), default off; the user runs `Trench Flux Sweep.cmd`; read it by its rule (handoff of 28 Oct 2026, fortieth pass)

Project: E:\projects\Infinite_Diffusion (Windows, PowerShell, `.venv` Python 3.12, numba, RTX 3070 8 GB, 16 threads).
Read first: `docs/project-map.md` (where everything is, commands), `docs/planet-parameter-audit.md` (which values are measured, assumed or fitted, and the replacement order),
`docs/planet-layer-plan.md` (step 7, step 8, "Step 8, follow-up 1" up to its **thirty-fifth pass**; the thirty-third builds E4 and
`boundary_normal` 'ring', the thirty-fourth reads their sweep, adopts ring, rejects E4, measures what the mantle lithosphere adds to
Earth's elevation, finds sea level 600–900 m below the datum and proposes B (continental budget and ocean basin) before C; the
thirty-fifth measures B's budget with `scripts/crust_budget.py` and proposes B-a / B-b / B-c),
`docs/planet-ideal-flow.md`, `docs/tectonic-prototype.md`. The **thirty-sixth pass** has B-b's diagnosis,
B-c's probe, GlobSed, B-a as built and its paired floor test; the **thirty-seventh pass** (end of the plan doc) reads
the compaction sweep (B-a adopted), adds Earth's ocean-floor age (Seton et al. 2020), builds B-b, pre-checks it, and
shows that the start's convergence comes from its geometry; the **thirty-eighth pass** (end of the plan doc) reads the
Plate Pieces sweep (B-b adopted), builds the ringed start and its force-law poles, restates the start test (one-sided,
S1 / S2), closes the crust and sediment ledgers, and writes `Coast Start Sweep.cmd`; then the dynamic-topography
investigation and the held-out seeds check; the **thirty-ninth pass** (end of the plan doc) reads the B-b six-seed
check (kept, gate statistic pooled), builds B-d (arc rate from Earth, arc docking) default off, and records three
pre-checks: docking fixes the budget, the plate layer runs away; the **fortieth pass** (end of the plan doc) reads the
Arc Dock Probe (six seeds: no runaway; arc flux 2x Earth's per km), finds that the trench fluxes follow node-typed
trench rather than consumed crust and that ocean-ocean polarity differs between the node typing and the overlap
resolution (22% of arc nodes), builds B-d2 (`trench_flux` 'consumed', `trench_polarity` 'pair'), pre-checks it on two
seeds in the device shell, and writes `Trench Flux Sweep.cmd`.

## Working rules
- **Bring a decision to the user only when it is very important** (user, 26 Oct evening: "only bring a decision to
  me if it's very important"). Very important means: it cannot be undone cheaply, it changes the project's goal or
  direction, or it needs something only the user has (data, hardware, money, a long run on their machine that is not
  already the routine sweep). Everything else (model design, which candidate to build next, test bands from Earth
  data, adopting by a rule fixed before the runs) Claude decides by the overall goal (realism) and these rules, writes
  the reasoning in the plan doc, and carries on. Running the next `.cmd` sweep is routine: just say which one to run.
- **Tell the user when it is a good time to start a new chat** (requested 23 Oct): when the conversation has grown
  long (many sweeps read, large tool outputs) or a natural checkpoint is reached (results read, docs and handoff
  updated, nothing half-built). Update this file, the plan doc and the Project copies first, then say so plainly,
  e.g. "This is a good point to start a new chat; the handoff is up to date."
- Don't run long generations or trainings yourself; give simple PowerShell commands with real paths (no `<placeholders>`;
  use `--plates latest` / `--plates sweep`). Short CPU tests in your own workspace are fine (a 250 Myr n=217 plate run
  takes ~15 min there, two in parallel; the cloud workspace has 2 cores, 7 GB).
- Edits through the bridge: stage each sync in a fresh folder **with a new file name** (a re-commit from the same
  staged path can write the old content), check the device md5 before overwriting, verify md5 after committing.
  Mirror the plan doc to the Project (`claude/planet-layer-plan.md`). `Projects project_read` returns the whole plan
  doc inline (~200 KB): don't read it just to compare. Upload it with `device_stage_files`, copy the staged file into
  the cloud working directory, then `project_write` with `local_path` (done 25 Oct night: the Project copy is
  complete through the twenty-fifth pass). 25 Oct late: that session had no `device_stage_files`, so the twenty-sixth
  and twenty-seventh passes sit in the Project as `claude/planet-layer-plan-pass26.md` and `-pass27.md`, and the
  Project copies of this handoff and the audit were not refreshed; re-upload all three when staging works.
  25 Oct morning: again no staging; the twenty-eighth pass is `claude/planet-layer-plan-pass28.md`, and this handoff
  was written to the Project inline (`claude/next-chat-prompt.md` is current; the audit copy is still old).
  25 Oct midday: still no staging; the twenty-ninth pass is `claude/planet-layer-plan-pass29.md`, this handoff was
  written inline again; the full plan doc and the audit copies are still old (re-upload when staging works).
  25 Oct afternoon: staging works again; the full plan doc (through the thirtieth pass), this handoff and the
  audit were uploaded to the Project with `local_path`; `claude/planet-layer-plan-pass26..29.md` are now superseded
  by the full copy (left in place). 25 Oct evening: plan doc (through the thirty-first pass) and this handoff
  uploaded again the same way. 25 Oct night: plan doc (through the thirty-second pass) and this handoff uploaded again.
  25 Oct evening (second session): plan doc (through the thirty-third pass), this handoff and the audit uploaded.
  26 Oct: plan doc (through the thirty-fourth pass), this handoff, the audit and the project map uploaded.
  26 Oct (second session): plan doc (through the thirty-fifth pass), this handoff, the audit and the project map uploaded.
  26 Oct (third session): plan doc (through the thirty-sixth pass), this handoff, the audit and the project map uploaded.
  26 Oct (fourth session): plan doc (through the thirty-seventh pass), this handoff, the audit and the project map uploaded.
  26 Oct (fifth session): plan doc (through the thirty-eighth pass), this handoff, the audit and the project map uploaded.
  27 Oct: plan doc (through the thirty-ninth pass), this handoff and the audit uploaded.
  28 Oct: plan doc (through the fortieth pass), this handoff, the audit and the project map uploaded.
  Long shell commands fail through the bridge (`spawn E2BIG`): write big files in the cloud workspace and copy them
  with `device_commit_files` from `/mnt/user-data/outputs/<fresh folder>/<new name>`.
- **Cloud workspace works for real runs** (25 Oct evening): `pip install --break-system-packages numpy numba ducc0
  pytest`, then the code as tarballs staged from `outputs/claude-sync/` (planet_layer, planet_diffusion, planet_training,
  toy_meso, scripts, tests, tools, data/crust1, data/tectonicplates-master; no ETOPO, so `elevation_check` does not
  run there). A 250 Myr n = 217 run took ~50–60 min with two in parallel (slower than the 15 min noted before).
- **Step-by-step identity checks need `params={'threads': 1}`:** with ducc0's threads two runs of the same code differ
  at round-off (1e-12) from the first steps.
- Without numba / ducc0 / pytest in the device shell, plate code can still be smoke-tested there:
  `python3 tools/device_stubs/run.py <test names>` (from `tests/test_plates.py`; local-Airy flexure, cosine-sum random
  fields, empty `ducc0`; ~2 s per step at n = 48; keep each call under ~170 s; `import run` from a script in that
  folder applies the stubs). Plumbing only, never for judging. The shell had no pip network on 25 Oct midday.
- The user's `.venv` has no scipy or scikit-image: planet-layer code must stay numpy / numba / ducc0.
- Total re-evaluation when something looks off. Fix pass bands / selection rules before running.
- **Direction (user, 23 Oct): realism; any system may be changed if it moves toward realism.** User, 25 Oct: free to
  add, remove or change anything as needed; comments may not reflect the actual goal or code: always decide by the
  overall goal (realism), not by what a comment says. Values come from Earth measurements or from mechanism, never from fitting the
  planet's output to a band; replace values that only make a band pass. Don't judge a law on a transient (twelfth pass).
- Mechanism above ~100 km, learning below. Propose a new model only with its test set first.
- Always compare against `outputs/meso/dataset.npz` (and ETOPO on the same grid via `scripts/earth_reference.py`).
- Avoid: learned planners above ~100 km, artificial constraints (locks, frozen sign, projections, mean
  conservation), scale hacks, over-tuning the test bed, training past the best point, and metric gaming.
- **Sweeps decide on six seeds (1001–1006) from 27 Oct (user agreed; ~2× machine time).** Never on one. Runs are deterministic on the user's machine; the cloud workspace's
  trajectories differ (seed 1001 there: 26–28 plates at 0 Myr vs 20 on the user's).
- **Judge terrain on numbers or on pictures with enough colour levels.** `elevation_shading.CMAP` had 256 levels over
  ±6 km (47 m each) until 23 Oct; it drew false terraces on plains. Fixed (N = 65,536).
- Planet-layer tests: `pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py
  tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py
  tests\test_crust_reviews.py` (86 since the fortieth pass; 65 pass; 26 Oct all 65 ran in the device shell after `pip install --user`; `test_coupling` must run from a copy there,
  since the shell cannot delete its old `outputs/test-tmp` folder).
- Continental crust in the reviews is label-free: total (crust + sediment) ≥ 20 km on both sides (`crust_review`
  at 46 km, `margin_width` on CRUST1.0's 1° cells, `to_crust1_cells`); the planet-flag reading is printed beside it.
- `planet_meso_check --landscape latest` prefers the newest *theta calibration*: pass explicit landscape folders.
- Sweep folders are named only by the parameters that differ between combinations (`MAX_NAME` 80, else `combo-NN`);
  shared ones are in `sweep.json`, each combination's in `<combo>/params.json`. Old sweeps were renamed the same way
  on 24 Oct (`outputs/planet-calibration/renamed-2026-10-24.json`); no path in `outputs` is over 260 characters.
- **Work on the user's machine through its Linux shell** (device_bash; repo at `$HOME/mnt/Infinite_Diffusion`): reviews
  and probes run there (`NUMBA_CACHE_DIR=$HOME/numba-cache PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=$PWD` for ad hoc
  scripts; each call ≤ 180 s, background jobs are stopped). Its Python packages can vanish between sessions: 25 Oct
  needed `pip install --user --break-system-packages numpy numba ducc0 pytest` again; on 25 Oct (evening) the
  shell had no network at all. The reviews still run on the repo's pure-Python numba fallback (18-run sweep:
  `elevation_check` 43 s, `crust_review` 2 s, `interior_probe` 63 s, `margin_width` 129 s); `PlateSim` and pytest
  need ducc0 and do not. Save review printouts beside the sweep (`<sweep>/reviews/*.txt`). Edit files there in place with
  short Python read-modify-write scripts that assert each old string matches once; check md5 before and after.
- `pkill -f` patterns can match your own command and kill your shell: use `simulate_[p]lates` style patterns
  (26 Oct: even that killed the cloud shell once; kill by PID from `ps aux | grep "simulate_[p]lates"`).
- **26 Oct: the device shell has numba, ducc0, pytest and h5py again** (`pip install --user --break-system-packages`
  worked): all 69 planet-layer tests run there in ~40 s (`test_coupling` from a copy: `$HOME/work/tc/tests`), and
  PlateSim at n = 217 builds in seconds, so paired probes on saved states run there (`scripts/compaction_effect.py`).
  Cloud: two 250 Myr n = 217 runs in parallel with `--param threads=1` took ~25 min (Bash `run_in_background`).
- **Runs drift apart from the first steps** (the cloud's seed 1001 equals the user's at −240 Myr, not at −230): an
  effect smaller than the drift between two runs (B-a's floor, ~120 m) must be measured paired, on the same states,
  not read off a sweep.

## Pipeline now
`scripts.simulate_plates` (n=217, 250 Myr) → `scripts.run_planet_landscape` (n=868, 30 Myr, θ=5) →
`scripts.planet_meso_check` (23 km hand-off, meso `detail-ft/step10000-passed.pt`, detail unknown).
Calibration tools: `scripts.sweep_plates` (`--rule shelf`), `scripts.calibrate_landscape_theta`,
`scripts.theta_seeds_review`, `scripts.shelf_review`, `scripts.plate_texture`, `scripts.boundary_lengths`
(true line lengths of plate boundaries by type, kinematically typed, vs Bird 2003). Crust reviews: `scripts.crust_review`,
`scripts.margin_width`, `scripts.margin_cross_section`, `scripts.interior_probe` (`--earth`, `--latest`), `scripts.elevation_check`.
Earth references: `scripts.earth_deformation` (GSRM v2.1 in `data/gsrm`: deforming crust vs Bird's orogens), `scripts.earth_crust_elevation`.
Margin modes: `--param margin_shortening=True --param sediment_accretion=0.13` (see plan doc, eleventh pass).

## State (26 Oct, after the thirty-fourth pass; details in the plan doc)
Working baseline (sweeps since 23–24 Oct pass these explicitly; PARAMS defaults in brackets where they differ): Earth
erosion law K × 1.0, θ 200, Sc 0.036 (PARAMS 0.75 / 6 / 0.05), Kd × 4; realistic margins `margin_shortening` True,
`sediment_accretion` 0.13 (PARAMS False / 0.75); `continental_sediment` True; κ_normal 30; `thinning_local_isostasy`
True; `rift_rule` pure_shear, breakup 10 km, `rift_width_km` 100 (defaults since 24 Oct); `start_crust` earth, `orogen_rule`
belt (defaults since 25 Oct); `boundary_normal` ring (default since 26 Oct). Latest sweep: `sweep-20261025-170555` (Typing E4,
read: ring adopted, E4 not); `Ring Defaults Tests.cmd` confirms the tests on Windows. Maps every 25 Myr (`maps.png`) now use
`simulate_plates.MAP_STOPS_M` (ocean blues, land hypsometric tint, colour bar; 'terrain' drew shallow sea green). Landscape θ 5.
- **Reviews read the label-free continent** (total ≥ 20 km on both sides; built 25 Oct; flag reading printed
  beside). Baseline (pure_shear, 3 seeds): area 0.313 ✗, < 30 km 0.063 ✗, > 45 km 0.283 ✗, > 55 km 0.092, median
  40.1 ✗; margins at 1°: thin per km 98 ✗ (100–200), abrupt 16% ✓, edge T 26.0 ✓, inner T 34.6 ✗ (30–34). Earth:
  0.384 / 0.190 / 0.086 / 0.018 / 37.0; 155 / 11% / 25.2 / 31.6.
- Margin profile itself Earth-like (−200..+100 km of the 20 km contour within ~1 km). Passive edge 40% of Earth's
  length (planet 60–69k km vs 173k; small enclosed basins 0–7k vs 34k).
- **Thick interior diagnosed (twenty-third pass, `scripts/interior_probe.py --earth`):** (1) 95–99% of interior crust
  is the supercontinent's start, untouched (p50 ~40 vs Earth's stable Precambrian crust 38.2, p10–p90 3–4 km wide vs
  Earth's 10); (2) Earth's 150–1000 km inland bins hold 20–40% shelf and extended crust (26–31 km) that the planet
  does not make (→ item 2b): the start alone cannot pass the ±1.5 km profile test; (3) the > 45 km tail (0.27–0.29)
  is crust thickened by shortening belts and spread by collapse that never gets the orogen mark: young-orogen
  erodibility covers 1–2% of land and 2–4% of land > 2 km vs Bird's orogens 16% / 76%, with which the erosion law was
  calibrated; land > 2 km is 32–36% of land (Earth 5–6%); (4) Earth's elevation is nearly flat with thickness from 30
  to 48 km (98 → 885 m), the planet's Airy ~3× steeper; Earth's thick Precambrian crust stands at 466 m.
- Older open items unchanged: lowland-plain windows 5.7% (≥ 8%), flat over-texture of the meso (detail value
  unknown), trench length 1.4–2.1× Earth, 11.53 km erosion law form (fifteenth pass).

- **Interior sweep read (twenty-fifth pass, `sweep-20261024-193840`, reviews in `<sweep>/reviews/`):** all six
  combinations pass the gates; by the rule **`start_crust` earth + `orogen_rule` belt adopted** (2 of 6 tests: median
  39.6 ✓, young orogen on land > 2 km 0.87 ✓; tie with earth-strain broken by > 45 km 0.298 vs 0.306, inside seed
  spread). PARAMS and two test asserts changed; the user ran `Interior Defaults Tests.cmd`: no failures.
  A works (median 39.5–39.7, < 30 km 0.09–0.10, p10 −1 to −3 km). B over-marks (young orogen on 0.40–0.47 of land vs
  Bird 0.16) and thins nothing: > 45 km 0.30, land > 2 km 0.34–0.37, export 5.2–6.8. **Cause: the high land is
  plateau** (slope of land > 2 km at 46 km: median 1.7–3.0 m/km, 64–79% under 5 m/km; Earth 11.1, 23%): a thresholded
  stream-power law cannot lower flat surfaces; collapse diffusion spreads thick crust into 45–55 km plateaus at Airy
  height. Young-ocean offset −179…−278 m (means), three earth-start seeds beyond −300: watch-point.

- **E references measured (twenty-sixth pass; `interior_probe` now prints them as "E tests", saved in
  `sweep-20261024-193840/reviews/interior_probe_E.txt`):** slope of land > 2 km p50 Earth 11.1 m/km vs planet 2.0–2.5,
  share under 5 m/km 0.23 vs 0.67–0.74; > 45 km crust depth from its edge p50 / p90 Earth 92 / 322 km vs 247–358 /
  667–955; Earth has one Tibet (7.9 Mkm², 47% of its thick crust), the planet three or four of 5–24 Mkm². The start is
  Earth-like (slope p50 13.3, depth p50 96 km): the plateaus grow during the run. All six combinations pass 0 of 4.
  Tests for E proposed and literature values sourced (onset ~45–50 km with no active convergence, end ~30 km, no
  breakup, strain rate ~1–5e-8 /yr, zones 500–800 km wide; Rey et al. 2001: the diffusion is the fixed-boundary mode,
  extension needs a free boundary). The collapse κ at 50–60 km matches a 15 km channel at 10¹⁹–10²⁰ Pa s, so
  "reduce the diffusion" is open, not decided.

- **E built (twenty-seventh pass), default off:** `extension` True splits a plate along an unheld > onset piece
  (≥ 0.5 Mkm², no collision boundary or subduction belt in or next to it), halves diverge at 2e-8 /yr × zone width,
  gaps within the zone's half-width (piece's p90 distance from the cut, 100–400 km) fill by pure shear, crust E
  touches is marked (`ext` / `extension_age_myr`), halves weld when the marked crust's mean is ≤ 30 km. Area ledger
  by mechanism in `stats.json`. Collapse diffusion unchanged. Two new tests; both and the main plate tests pass in
  the device shell with stubs; pytest on the user's machine runs first in `Extension Sweep.cmd`.

- **Extension sweep read (twenty-eighth pass, `sweep-20261024-215311`, reviews in `<sweep>/reviews/`): E not
  adopted.** Control identical to the interior sweep. E passes 0 of 4 E tests at onset 45 and 50 km (slope p50 3.0–3.1
  m/km, under 5 m/km 0.62–0.64, depth p50 193–254 km); at 45 km the volume (−14.3%) and ledger-residual guards fail,
  at 50 km the young-ocean gate is −301 m. It does move crust the right way (> 45 km 0.298 → 0.268 / 0.234, < 30 km
  0.09 → 0.15–0.16, margin review passes). Cause: the "held" rule. 81–100% of > 45 km area is in pieces touching a
  collision or belt, the largest plateau always, though 50–76% of it is > 300 km from that boundary; 8 of 36 events
  fired on the start's cold crust at −249 Myr; the whole-plate cut added more rift area (+0.029) than E (+0.020).
  `orogeny_age_myr` is 0 on all thick crust (reset each step): not a thermal clock.

- **E2 built (twenty-ninth pass), default off:** `ext_held_rule` 'direction' (held pieces cut across their long
  axis; 'exempt' = E as built), `ext_hot_myr` 100 Myr (Sonder et al. 1987: thickening-to-extension delay 0–100 Myr,
  observed 0 / 20–40 / ~70 Myr; 0 = hot rule off), thickening clock `thk` (snapshots `thickening_age_myr`; set where
  subduction-term + collision-volume thickening / thickness > `orogen_strain_rate`; not collapse diffusion), trigger
  reach in `stats.json` `present.extension_trigger` and printed by `interior_probe` ("E trigger"). Three new tests;
  E's test pinned to 'exempt' + 0. Device shell (stubs): new tests, E's two and the main plate tests pass; default
  path identical step by step to the pre-edit code at n = 48 (incl. a split and a merge). Not run at 46 km.
  **Risk:** crust spread by the collapse diffusion gets no clock, so plateau interiors may stay "cold" and the trigger
  reach fall below 0.5; remedy (not built, to agree only then): the clock carried by the collapse flow (plan doc).

- **E2 sweep read (thirtieth pass, `sweep-20261025-111119`, reviews in `<sweep>/reviews/`): E2 not adopted.** Control
  identical to the extension sweep's. 0 of 4 E tests at 45 and 50 km; **design check fails: eligible 0.01 / 0.04**
  (hot nodes 0.22 / 0.25 of the > 45 km area). Snapshot probe: > 45 km crust hot 0.11–0.29, timed > 100 Myr ago
  0.00–0.10, **never timed 0.64–0.89**, though 98% of it is the run's own (provenance "collided"): the collapse
  diffusion spreads belt crust inland untimed. Still, at 45 km E2 moves crust the right way (> 45 km 0.298 → 0.253,
  < 30 km 0.090 → 0.123, margin review passes). Gates and other guards pass (young ocean −248 / −286 m, export 6.6 /
  7.0, volume −12.6 / −11.9%; ledger residual at 50 km 0.005 off, on the edge). **The whole-plate cut rifts elsewhere:**
  rift area over the control +0.012 / +0.030 vs E2's own +0.003 / +0.006, above the control in 11 of 12 extension runs
  over both sweeps: the condition for the zone-bounded cut is met.

- **E3 built (thirty-first pass), default off; agreed 25 Oct 11:53.** E2 + (b′) `ext_clock_flow` (`carry_clock` in the
  diffusion substeps) + (c′) `ext_cut` band (microplate between two transforms, pole carried in the parent's frame)
  + **(e) `ext_pair_motion` union**, added after the first pre-check without asking first (confirmed by the user 25 Oct 14:08):
  while an event runs, the pair's mean motion is `_driving` on both halves together. Without it a band strip cut along
  a collision belt takes the collision boundary, the parent loses the collision resistance (target speed / (1 + L_coll
  / 1500 km)) and speeds up: pre-check area-weighted speed 7.3 vs 3.7 cm/yr, volume −25%, plateaus ran away. E/E2
  reproducible with `ext_clock_flow` False, `ext_cut` whole, `ext_pair_motion` mean (checked step by step, 30 steps).
  39 plate tests (7 new) and 60 planet-layer tests pass in the device shell. Pre-check with union (cloud, seed 1001,
  approximate label-free reading): eligible 0.91, rift over control +0.037 ≤ own +0.051, speed 4.0, volume −12.0%,
  > 45 km 0.299 → 0.239, < 30 km 0.068 → 0.145, median 37.8, largest plateau 6.9 Mkm², depth p50 / p90 159 / 435 km,
  slope p50 4.3 m/km, under 5 m/km 0.54; 21 of 33 events "lost" (a half docked onto another plate), 29 plates vs 18.

- **E3 sweep read (thirty-second pass, `sweep-20261025-141028`, reviews in `<sweep>/reviews/`): E3 not adopted.**
  Control identical to the E2 sweep's. Every candidate passes 1 of 4 E tests (depth p90); rule needs 2. Design checks
  pass for the band cut (eligible 0.84 / 0.93; rift over control +0.020 / +0.016 ≤ own 0.030 / 0.022), the whole cut
  fails the rift check and the ledger guard again. E3 50 km is the best run so far: > 45 km 0.171, land > 2 km 0.227,
  volume −8.5%, area 0.332, margins pass, young ocean −241 m (all seeds within ±300), speed 3.9 cm/yr. **Events do not
  finish:** 60–70% of band events are lost (a half docks onto a third plate after ~36–38 Myr, the merge rule's 40 Myr;
  far side on a collision, median share 0.45 / 0.58), so E-marked crust stays at 41–44 km (Earth's extended 29–33) and
  < 30 km is only 0.10–0.12; the whole cut (5.7 lost per run) reaches 0.218 (Earth 0.190). **Slope pair out of E's
  reach:** > 45 km crust stands at 2.3–3.4 km (Earth 1.2), land > 2 km 0.23–0.28 of land (Earth 0.05): item C decides it.
- **E4 proposed, not built:** `ext_side` 'free' (the band side with the smaller far-edge collision share moves; ties to
  the smaller area; Rey et al. 2001). Tests g1 / g2, design checks lost ≤ 1/3 of starts and far-side collision median
  ≤ 0.2, pre-check seed 1001 at 50 km, then `Extension4 Sweep.cmd` (control / E4 45 / E4 50). Details in the plan doc.

- **Thirty-third pass (25 Oct evening): E4 built, default off** (`ext_side` 'free': at a band cut the side whose far
  edge has the smaller collision share moves; tests g1 and d; g2 by hand: 'smaller' = E3 step by step at n = 48).
  Pre-check (cloud, seed 1001, 50 km): E4 2 of 4 E tests (depth p50 142, p90 385) vs E3 1 of 4, but lost 0.67 of
  starts (E3 0.53). Logged rerun: **5 of 9 lost E4 events were the microplate docking onto its own parent after exactly
  40 Myr**, across the band's side edges: transforms by construction, typed ridge / collision by the simulator.
- **Boundary-typing defect (general, since step 7):** `classify_boundaries` types a node by one grid neighbour's
  direction; staircase transforms become ridge / collision / subduction (pure transforms on random great circles:
  median 30% typed transform, min 0%; a run's transforms 11% of node-typed boundary vs 36% of contour-typed length).
  **Built, default 'node': `boundary_normal` 'ring'** (normal from the two-ring; 96–100% typed transform; node shares
  then match the contour typing). Tests b1–b5; 65 planet-layer tests pass in the cloud; 'node' = as built step by step.
  Ring pre-check (seed 1001): control > 45 km 0.286, largest plateau 28 Mkm², subduction length still 1.7× Bird
  (contour); ring E4 50 km lost 0.35, far side 0.27, 0 of 4 E tests (5 of 8 ends at the 100 Myr guard, extended crust
  at 43 km). One seed: the sweep decides. Logged-run tool: `tools/extension_event_log.py`.

- **Thirty-fourth pass (26 Oct): Typing E4 sweep read** (`sweep-20261025-170555`, reviews in `<sweep>/reviews/`). Node
  control identical to the E3 sweep's, field by field. **Ring adopted** (PARAMS, test renamed
  `test_ring_normal_is_the_default_and_keeps_the_ledgers`, audit 4c): its control passes the elevation tests on 3 of 3
  seeds (node 1 of 3, young-ocean offsets −356 / −307 m), export 6.2, volume −12.6%, median 39.3, margins pass (node
  fails, inner T 34.5); crust numbers otherwise within seed spread (> 45 km 0.304 vs 0.298, land > 2 km 0.381 vs
  0.342, largest plateau 12–36 Mkm²); contour lengths ridge 97.6 / subduction 83.9 / collision 5.4 / transform 99.4k km
  (Bird 94.8 / 51.3 / 40.5 / 73.9); node shares now match the contour typing; plates 16.8, speed 4.0. **E4 not
  adopted:** ring 45 km 0 of 4 E tests though its design checks pass (lost 0.33, far side 0.07: docking was the typing
  defect), ledger residual −0.019 vs −0.006, volume −14.0%; ring 50 km 1 of 4, lost 0.47; node 50 km 1 of 4, lost 0.55.
  Thinning is what fails now: zones always at the 400 km cap, 17 of 39 events guard-ended at 31–60 km, E-marked crust
  p50 33–46 km and 30–99% of it re-thickened later.
- **C re-examined (new `scripts/earth_lithosphere_residual.py`, json in `outputs/planet-reference/`):** Earth's
  crystalline density is 2,830 kg/m³ in every thickness bin (composition does not flatten Earth's relation); the mantle
  lithosphere does (dL −0.5 km at 20–25 km to −3.1 km at 45–50 km; rifted crust ~1 km above normal). **Inside Bird's
  orogens Earth's 45–55 km crust stands on the planet's own Airy line** (1,383 vs 1,559 m; 2,282 vs 2,294 m); only old
  crust outside orogens stands low. The planet's thick crust is all young orogen, so C would lift thin margins and lower
  cratons but not the plateaus: the slope pair is not C's.
- **Sea level 590–890 m below the datum** (Earth: on it by the datum's calibration). Water below the datum: Earth
  2,629 m GEL, planet 2,860 at the start, 3,133–3,285 at present: half extra deep-ocean area (continental area lost),
  half a floor ~0.3 km deeper than Earth's (not ages; sediment only partly: deep-ocean sediment 188–574 m vs GlobSed
  ~404; missing plateaus / seamounts / margin sediment). Land shifted by the offset: land > 2 km 0.24–0.32 of land
  (now 0.34–0.45, Earth 0.05): sea level is about a quarter of the high-land problem; the rest is too much thick crust on
  too little area. **Budget (km³/yr):** island arcs add 0.6 (Earth 1.5) and never become continental (area ledger `arc`
  0); sediment subduction 2.2–2.6 (Earth ~1.0–1.65); continental erosion 6.8–8.8, to the ocean 3.3–4.4.

- **Thirty-fifth pass (26 Oct, B started): budget measured, no model change.** `scripts/crust_budget.py` (B's tests,
  every ledger in km³/yr over the whole run and the last 40 Myr beside Scholl & von Huene 2009, island-arc stock and
  what reaches trenches, contour trench length split continental / intra-oceanic: new keys in `boundary_lengths`);
  read on `sweep-20261025-170555` (`<sweep>/reviews/crust_budget.txt` / `.json`). Ring control: all five B tests fail
  (water below the datum 3,206 m vs 2,629; floor 4,524 vs 4,205; volume −12.6%; area 0.303; sediment subducted 3.5
  km³/yr in the last 40 Myr vs ~1.0). Readings: (1) per km of continental-side trench, arcs and subduction erosion are
  Earth's and that trench is Earth's length (26k km); the loss is eroded crust leaving as sediment (4.0–4.8 km³/yr onto
  oceanic crust, 2.5–3.5 subducted; margins on continental crust already hold Earth's sediment thickness on half
  Earth's thin-crust area; 70% of the ocean's share is turbidite fill that ponds at the nearest low). (2) The excess
  trench is intra-oceanic and inherited from the start: 33–36k km at −240 Myr, 54–58k at 0 (Earth ~17k, Leat & Larter
  2003). (3) Island arcs grow 15 km³/Myr per km (Earth 30–95, Dimalanta et al. 2002), stand p90 11.5 km (Earth 20–30)
  and never turn continental; 0.30 km³/yr of them is subducted at trenches facing continents. (4) Sediment density
  (2,400 vs bulk ~2,015 near the top, Sykes 1996, GlobSed seismic thickness) is ~40% of the floor misfit (−130 to −190 m
  if bulk). (5) Area: collisions take area at twice the rifts' rate; the lost area is the largest term of the water
  excess.

- **Thirty-sixth pass (26 Oct): B's two measurements, then B-a built.** New scripts `trench_origin` (B-b),
  `earth_sediment` (GlobSed v3 ASCII on n = 217; the `.nc` is HDF5, the ASCII writes `1.#QNAN`), `ocean_sediment_probe`
  (B-c), `compaction_effect` (paired); `boundary_lengths.typed_segments`. Reviews in `sweep-20261025-170555/reviews/`.
  **B-b:** Earth (Bird + CRUST1.0) intra-oceanic SUB 20.8k km, continental 30.7k, OCB 17.4k (slow, no slab),
  production 3.37 km²/yr. Planet: intra-oceanic 33–38k already at −250/−240 Myr (random start poles: 0.31–0.41 of
  the ocean–ocean boundary converges vs Earth 0.25 incl. OCB, 0.14 as trench), 55–58k at 0, of which 24–28k borders
  **detached plate pieces** (31–65 Mkm², 6–13% of the surface, 35–53% of all boundary; a plate's pieces keep one pole;
  seen: a plate inside another grows until it cuts its host). Speeds follow `_driving` (0.98–1.00), and that law gives
  Earth's speeds on Earth's geometry within 9%: not the cause. Production 4.6–6.2 km²/yr (Earth 3.37), excess consumed
  at intra-oceanic trenches. **B-c:** the fill lands at active-margin feet (23–48% of the net arrival on 2–3% of the
  ocean, 1.6–4.6× the passive feet's rate); intra-oceanic trenches get 8% but subduct the open-ocean stock at 2.3–3×
  Earth's consumption: half the sediment-subduction excess is B-b's. Earth's label-free ocean holds 610 m (GlobSed,
  bulk); planet open ocean thin (no pelagic). No fan rule now. **B-a:** `sediment_compaction` (off): `sed` = solid
  volume (`rho_grain` 2650), Athy (Sclater & Christie 1980: shale 0.63 / 0.51 /km on oceanic crust, shaly sandstone
  0.56 / 0.39 on continental), water-filled pores, kernels unchanged (solid heights within a step), snapshots gain
  `sediment_bulk_m`, reviews read bulk. Paired on the ring control's states: deep floor −71 / −115 / −171 m (mean −119,
  37% of the gap), water −112 m, sea level +135 m; two seeds' bulk ocean sediment then exceeds GlobSed by 27–54%.
  4 new tests; default identical step by step (70 steps, n = 48). Cloud pre-check (seed 1001): gates pass, export 5.1,
  volume −9.6% as its control.

- **Thirty-seventh pass (26 Oct): B-a adopted, B-b built and pre-checked.** Compaction sweep (`sweep-20261025-201623`):
  control identical to the ring control; B-a passes its rule (gates 3 of 3, median 39.6, volume −12.5% vs −12.6%,
  margins pass); deep floor 4,317 m (B test passes; control 4,524), water below the datum 3,036 m, ocean bulk sediment
  569 m (GlobSed 610); sediment subducted 4.13 km³/yr (control 3.52, Earth ~1). PARAMS `sediment_compaction` True.
  Runs now save `params_effective`; reviews read older runs with `crust_review.LEGACY_DEFAULTS` (compaction off).
  New `scripts/earth_seafloor_age.py` (numpy CDF1 reader for `data/EarthByte/Grids/age.2020.1.GTS2012.2m.grd`): Earth
  production 2.9–3.0 km²/yr (A = 2–40 Myr), mean age 63.9 Myr, GlobSed median by age ~3 m/Myr (pelagic reference).
  **B-b** (`piece_rule` 'plates', default 'shared'; `piece_min_area_mkm2` 1 from Bird: 20 plates ≥ 1 Mkm² hold 98%):
  detached pieces with an interior node become plates (≥ 1 Mkm², parent's Euler vector, then own forces) or weld to
  the plate around most of their edge (microplate capture); boundary crumbs are left. 5 new tests, 74 pass; default
  identical step by step. New `scripts/plate_pieces_review.py`. Pre-check (cloud, seed 1001): detached max 0.0034 of
  the surface (control 0.023); window −100…0 Myr: intra-oceanic trench 33.7k (control 39.1k), continental side 34.0k
  (27.1k), production 4.72 (4.73), ridge 83.6k, shape ratio 1.51 (Earth 1.51; control 1.71); volume −10.8% vs −9.6%
  (guard edge, one seed); watch-point: continental-side trench and collision longer. **The remaining intra-oceanic
  excess is the start's** (33–38k km at −250 Myr): relaxing start poles to the force law makes it worse (0.45–0.50 of
  the ocean–ocean boundary), ridge push only 0.32–0.41; cause: the supercontinent's plates reach into the ocean, its
  coast is passive (continental-side trench 0.2–0.7k km at −250), so all closing is ocean–ocean (Pangea was ringed by
  subduction). Found, not fixed: continental-sediment ledger drift 1e-5–3e-4 over 60 steps at n = 48 in
  `_update_rasters` (default path).

- **Thirty-eighth pass (26 Oct evening): B-b adopted; ringed start built; ledgers closed.** Plate Pieces sweep
  (`sweep-20261025-220434`): control identical to the B-a combination; B-b passes its rule (T1 detached max
  0.0018–0.0033 vs 0.06–0.095; window T2 41.4k vs 55.5k, T3 30.7k, T4 5.15 vs 5.72, T5 91.1k; gates 3 of 3; median
  39.2; volume −10.8% vs −12.5%; margins pass). PARAMS `piece_rule` 'plates'. **`start_plates` 'coast'** (default
  'voronoi'): seeds grow only in their own domain, the supercontinent's plates end at the coast. **`start_poles`
  'forces'** (default 'random'): oceanic start plates move as `_driving` (fully relaxed, `relax=1.`) sets them with the
  coast as trench and ocean-ocean boundaries as ridges. Start probe at −240 (3 seeds, device): continental side /
  intra-oceanic voronoi 3.1k / 33.9k, coast 32.6k / 24.2k, coast + forces 41.3k / 13.4k. Cloud pre-check (seed 1001,
  baseline): coast + random fails over the run (window T2 40.8k vs 33.7k, young-ocean offset −524 m: diverging rim
  opens a ridge at the continent's edge); **coast + forces passes every B-b window test** (T2 32.9k, T3 27.7k, T4 4.23,
  T5 81.3k), gates (−147 m), volume −7.0% (control −10.8%), ocean-floor ages closest to Seton et al. so far (mean 57
  Myr; Earth 63.9). Start test restated one-sided (S1 continental-side ≥ 20k km, S2 share ≥ 0.60; van der Meer et al.
  2014: Triassic subduction ~1.55× today's; Pangea ringed: Collins 2003, Cawood 2005). Ledgers: consumed nodes booked
  when they leave, `cont_lost_other` no longer double-counted (crust and sediment close to 1e-16; fields unchanged).
  79 tests pass in the device shell.

## Next, in order (written 28 Oct, fortieth pass)
1. **The user runs `Trench Flux Sweep.cmd`** (repo root; tests, then A control / B `trench_polarity` pair /
   C pair + `trench_flux` consumed + `arc_H_ocean_m` 1000 + `arc_accretion` True on seeds 1001-1006, 18 runs ~1.5 h,
   then identity (A vs the probe's control: must say IDENTICAL), plate_pieces_review, arc_dock_probe, crust_budget,
   elevation_check, crust_review, margin_width into `<sweep>/reviews/`). Read it by the rule written in the .cmd
   (and in the plan doc, end of the fortieth pass); if C or B is adopted: PARAMS defaults, default tests
   (`test_trench_flux_band_is_the_default`, `test_pair_polarity_follows_the_overlap_resolution`,
   `test_arc_accretion_is_off_by_default` assert today's defaults: change them with the adoption), audit rows.
   Pre-check risks: C's T3 / T4 (continental-side trench 38k, production 5.2 on two seeds), plates faster with pair
   (4.9 cm/yr), sediment subducted 64 km^3/Myr/km.
2. Then the remaining B items: the intra-oceanic trench excess (T2 fails in every arm, 30-40k vs 12-25k; the
   control's trench grows through the run: node-km 104k at -240 to 268k at 0, plates 12 -> 28) and the
   sediment-subduction excess (2.6-3.4 km^3/yr vs ~1). D (dynamic topography from slabs) before C.
3. **C, lithosphere buoyancy**; 4. **E, the thick crust itself**; 5. the rest of 2b; 6. detail value for the meso;
   7. freeze and look.
Tools new on 28 Oct: `tools/arc_growth_probe.py` (band terms vs contour consumption on snapshots);
`tools/device_runs/chunk_run.py` + `simpickle.py` (full n = 217 runs in the device shell, pickled between 180 s calls,
two in parallel ~12 min per run; pre-checks only: trajectories differ from Windows). Pre-check outputs:
`outputs/planet-calibration/precheck-20261028-trenchflux/` (a stale `_state.pkl` sits in its first combination's
seed1001 folder: the shell could not delete it; harmless).

## Next, in order (written 27 Oct, thirty-ninth pass; superseded)
1. **The user runs `Arc Dock Probe.cmd`** (27 Oct evening; gathers information, decides nothing): tests, then
   control / rate / rate + docking on seeds 1001-1006 (18 runs; runs now save `event_list` in stats.json), then
   `tools/compare_runs.py` (control vs the earlier runs: must say IDENTICAL twice) and the new
   `scripts/arc_dock_probe.py` (when the trench parts from the control's, plate events by window, where the trench
   sits: near small continental bodies < 2 Mkm^2 / thin continent / oceanic arcs > 10 km / large continents / far,
   small bodies' thickness, spikes, detach cutters, arc fluxes; `.png` of trench and plates per seed), plus the usual
   reviews. On the cloud's seed 1001 (three pre-check runs) it read, at 0 Myr: small continental bodies 98 / 140 /
   319, trench node-km near oceanic arcs 4 / 24 / 76k, near small bodies 17 / 42 / 88k (control / rate / docking).
   Read it, then:
   **B-d: why docking arcs lengthens the trench and cuts plates** (plan doc, end of the thirty-ninth pass). Pre-check
   runs (cloud, seed 1001; reviews `runs/rev*_*.txt` there, gone with the workspace; numbers in the plan doc): control
   300 m, rate 1,000 m, rate + docking (single node), rate + docking (spread). Docking: arcs lost 0.61 → 0.01–0.06,
   volume −10.8% → −4.3 / −7.3%, deep floor passes; but 34–40 plates, node trench 119k vs 82–91k from −140 Myr, detach
   events 13–18 vs 2. Measure paired and cheaply first (several seeds at n = 96, or a per-step trench-length ledger by
   cause), then fix, then a full pre-check, then `Arc Rate Sweep DRAFT.cmd` (in `outputs/claude-sync/s20261027a/`;
   rule fixed there: six seeds, three arms) moved to the repo root and run by the user. Leads: arc continents ~13 km
   thick (turned continental at 20 km then thinned; Earth's arcs 26 ± 6 km, Tetreault & Buiter 2014); docked crust
   landing inside the hidden overlap region.
2. The sediment-subduction excess (3.4–4.1 km³/yr vs ~1). D (dynamic topography from slabs) before C.
3. **C, lithosphere buoyancy**; 4. **E, the thick crust itself**; 5. the rest of 2b; 6. detail value for the meso;
   7. freeze and look (as in the list below).
Watch-points: young-ocean offset (now pooled; B-b ~50 m deeper on average; arcs add young ocean beside thick arc
crust: −313 … −352 m on the docking pre-checks); T1 reads detachment lasting two samples; the leak gate reads normal
oceanic crust only.

## Next, in order (written 26 Oct, thirty-eighth pass; superseded)
1. **Done (26 Oct night): Coast Start sweep and the held-out seeds check read** (plan doc, end). Coast Start: neither
   candidate adopted (coast:forces fails T5, ridge 71k vs 91k, and the gates on 2 of 3 / 3 of 6 seeds); PARAMS unchanged.
   Held-out (`sweep-20261026-200857`, seeds 1004–1006): crust / elevation numbers hold; the baseline's plate-geometry
   windows flag 3 (T2, T4, T5; seed 1004 ends with 13 plates), so by the rule B-b is decided again on six seeds;
   young-ocean gate fails on 1004 (−313 m): 5 of 6. coast:forces 0 flags. Not the dynamic-topography field's doing.
2. **The user runs `B-b Six Seeds Check.cmd`** (piece_rule plates / shared on 1004–1006; plates must equal
   `sweep-20261026-200857\start_plates-voronoi_start_poles-random` field by field); read by the rule in the .cmd on
   six-seed means (1001–1003 from `sweep-20261025-220434`). Ask the user once whether future sweeps run six seeds
   (1001–1006; ~2× machine time) — proposed, not agreed.
3. Then arcs at Earth's rate (thirty-fifth pass); the sediment-subduction excess (3.4–4.1 km³/yr vs ~1).
   D (dynamic topography from the slabs, see the investigation section at the end) before C.
4. **C, lithosphere buoyancy** (narrowed: thin and rifted crust up, old thick crust down).
5. **E, the thick crust itself** (also the main area lever: collision area loss).
6. **The rest of 2b** (back-arc basins, microcontinents, passive edge in small basins: Earth 34k km vs 0–7k).
6. Detail value for the meso (independent): `planet_meso_check` with fixed detail values on flat windows first.
7. Freeze and look (GeoTIFF, `toy_meso.planet_tiled --bounds`); drainage vs RiverATLAS.
Small fixes: a pelagic rule from GlobSed by age (~3 m/Myr bulk median).
Watch-points: young-ocean offset; the ringed start's deep floor (old at the coast: 4,596 m at −250 on seed 1001);
> 45 km share; late plate count; boundary crumbs (B-b leaves them).

## Next, in order (written 26 Oct, thirty-seventh pass; superseded)
1. **The user runs `Plate Pieces Sweep.cmd`** (first try 26 Oct night stopped at the tests: the B-b run test needed
   welds within a fixed 60 steps and Windows' trajectory had none; the test now runs 60–150 steps until a plate and
   two welds; 74 pass in the device shell; plan doc end of the thirty-seventh pass)
   (tests, Earth age reference, `piece_rule` shared / plates × 3 seeds,
   reviews into `<sweep>/reviews/`); read it by the rule in the .cmd ("Decision" lines of `plate_pieces_review.txt`,
   then gates and guards). If adopted: PARAMS default, a default test, audit row. Check the control against
   `sweep-20261025-201623\sediment_compaction-True` field by field first.
2. **Start geometry, to agree with the user (plan doc, thirty-seventh pass):** a supercontinent start ringed by
   subduction (continental plates end at the coast on most of the rim; the oceanic side subducts there) instead of
   continental Voronoi cells reaching into the ocean. Tests: B-b's T2 / T3 at −240 Myr and over the run.
3. Then arcs at Earth's rate (thirty-fifth pass); the sediment-subduction excess (4.1 km³/yr vs ~1).
4. **C, lithosphere buoyancy** (narrowed: thin and rifted crust up, old thick crust down).
5. **E, the thick crust itself** (also the main area lever: collision area loss).
6. **The rest of 2b** (back-arc basins, microcontinents, passive edge in small basins: Earth 34k km vs 0–7k).
7. Detail value for the meso (independent): `planet_meso_check` with fixed detail values on flat windows first.
8. Freeze and look (GeoTIFF, `toy_meso.planet_tiled --bounds`); drainage vs RiverATLAS.
Small fixes: the continental-sediment ledger drift in `_update_rasters` (thirty-seventh pass); a pelagic rule from
GlobSed by age (~3 m/Myr bulk median).
Watch-points: young-ocean offset; continental-side trench with B-b; boundary crumbs (no interior node) can exceed
1% detached at n = 48 (seed 7: 0.0142); at n = 217 the pre-check had 0.0034.

## Next, in order (written 26 Oct, thirty-sixth pass; superseded)
1. **The user runs `Compaction Sweep.cmd`** (tests, GlobSed, 2 × 3 seeds, reviews into `<sweep>/reviews/`); read it by
   the rule in the .cmd (floor test already passed paired; the sweep decides on gates and guards against its
   control). If adopted: PARAMS default, a default test, audit row.
2. **B-b candidate, to agree (plan doc, thirty-sixth pass): plates are connected.** Each detached piece above a size
   floor becomes its own plate (parent's Euler vector at the split, then its own forces); smaller pieces join the plate
   around most of their edge. Tests: detached area <= 1% of the surface; intra-oceanic trench 12–25k km and continental
   side 20–35k at 0 Myr; production 2.5–4.5 km²/yr (last 40 Myr); ridge length Bird ± 15%; plate count and sizes vs
   Bird; fixed B tests and reviews. Then, only if the start still converges too much: start poles from the oceanic
   plates' own slab and ridge torques instead of random axes. Then arcs at Earth's rate (thirty-fifth pass).
2. **C, lithosphere buoyancy** (narrowed: thin and rifted crust up, old thick crust down; after B).
3. **E, the thick crust itself** (also the main area lever: collision area loss).
4. **Trench length**: now known to be intra-oceanic (3× Earth's) and inherited from the start; folded into B-b.
5. **The rest of 2b** (back-arc basins, microcontinents, passive edge in small basins: Earth 34k km vs 0–7k).
6. Detail value for the meso (independent): `planet_meso_check` with fixed detail values on flat windows first.
7. Freeze and look (GeoTIFF, `toy_meso.planet_tiled --bounds`); drainage vs RiverATLAS.
Watch-point: young-ocean offset (eased with ring: no seed beyond −300 m).
Useful data to download (user, browser): an ocean-floor age grid (EarthByte: Müller et al. 2016 or Seton et al. 2020) for
GlobSed by floor age and a pelagic rate (GlobSed v3 is in `data/GlobSed/` since 26 Oct).
Later: lowland-plain band, water balance / endorheic basins, climate layer, dispersed start, 20/40 Myr windows,
microplates, collision length.

## State addendum (27 Oct, thirty-ninth pass)
- **B-b Six Seeds Check read** (`sweep-20261026-204140`): plates equals the held-out control field by field; six-seed
  means T1–T5 pass (T2 38.5k vs 47.4k, T3 29.4k, T4 4.59 vs 5.39, T5 81.9k), guards pass (volume −9.9% vs −10.4%,
  margins pass; shared's fail on 1004–1006). The .cmd's gate clause failed as written (plates 5 of 6, shared 6 of 6:
  seed 1004 young-ocean offset −313 m from one snapshot whose sample was 0.064% of the surface). **B-b kept** as a
  recorded exception. **`elevation_check` offsets now pool the window** (area-weighted median over all the window's
  nodes; old reading printed as "snapshot mean"; re-read on all sweeps since 25 Oct in `<sweep>/reviews/
  elevation_check_pooled.txt`: no past decision moves).
- **B-d built, default off** (`arc_accretion`, `arc_buoyant_thick_m` 15 km, `arc_accrete_upper_m` 3.5 km,
  `spread_to_band`; candidate `arc_H_ocean_m` 1,000 m from Dimalanta et al. 2002 over Bird's intra-oceanic closing
  rate 61.5 mm/yr). Arc tests A1–A3 in `crust_budget`. Review corrections: leak gate on normal oceanic crust; T1 two
  samples. Ledger bug fixed (arc-to-continent booked shared plate nodes twice; ledger only). Default path identical
  step by step; 82 tests pass in the cloud, the new ones on the device.

## State addendum (28 Oct, fortieth pass)
- **Arc Dock Probe read** (`sweep-20261027-210209`): six seeds show no plate runaway (plates 27.8 / 28.8 / 32.2 for
  control / rate / rate + docking; the extra plates are < 1 Mkm^2; contour subduction +8%, all continental side, beside
  arc continents and small continental bodies). Arc growth per km of intra-oceanic trench 111-120 (Earth 30-95);
  docking moves 2.2 km^3/yr onto continents (> 45 km +0.06); gates 6 / 2 / 3 of 6; margins fail with docking.
- **Trench fluxes over-count** (`tools/arc_growth_probe.py`): the band law follows node-typed trench, 1.8-2.7x H x the
  consumed area for ocean arcs (three seeds: more arc crust than all subduction consumed), 1.2-1.5x continental arcs,
  1.5-1.7x subduction erosion; straight line exact; resolution-dependent (0.7x at n = 96).
- **Polarity defect (default path since step 7):** node typing vs `_ocean_polarity` disagree at 22% of oceanic
  overriding nodes (arcs on the downgoing plate, slab pull on the overriding one).
- **B-d2 built, default off:** `trench_flux` 'consumed', `trench_polarity` 'pair'; 4 tests; 86 planet-layer tests pass
  in the device shell; default path identical step by step (70 steps). Pre-check (pair + consumed + docking, seeds
  1001 / 1004, device): routed 0.92, A1-A3 pass (35.0 / 15.9 / -0.16), arc growth 1.1 km^3/yr, docked on continents
  0.8, volume -6.0%, gates 2 of 2, margins pass; T3 38.3k, T4 5.23, speed 4.9 cm/yr.
- The device shell had numba / ducc0 / pytest again after `pip install --user --break-system-packages` (28 Oct);
  it has 2 cores and 3.9 GB.

## Investigation (26 Oct, night; plan doc, end): the shared-within-seed noise is the dynamic-topography field
The user saw a noise pattern shared by runs of one seed at the same checkpoint. It is `Mantle`'s field: own stream
`default_rng([seed, 7])`, all knots drawn up front, nothing reads the plates, so it is bit-identical across settings
and added to elevation at full height (slope 0.8–1.0). Ocean high-pass: 0.13–0.32 correlation between settings, ~0
with the field removed (3 seeds); land: only 4–8% of high-pass variance. Lag matrix non-zero only at equal times
(independent knots every 40 Myr; rms pulses 510 → 370 m between knots). Earth's is red (85% in degrees 1–3, peak at
degree 2; Valentine & Davies 2020) and tied to slabs (Rubey et al. 2017); the planet's is flat to degree 30 (12% in
degrees 1–3): the ~650 km blobs that passed the texture score on 18 Oct. **Proposed D (not built):** dynamic
topography from the plates' own slab history plus a weak constant-variance short-wavelength part; tests in the plan
doc. Order: after reading `Coast Start Sweep` (item 1, its reviews exist: `sweep-20261026-184745`), before C.
