# Project map (planet layer work; structure as of 23 Oct 2026, pointers updated 28 Oct)

A quick orientation for a new chat. Read with `docs/planet-layer/STATE.md` (what is running, the model, the
scorecard, next steps), `docs/next-chat-prompt.md` (working rules), `docs/planet-parameter-audit.md` (which values
are measured, assumed or fitted) and `docs/planet-layer/decisions.md` (what was adopted or rejected). The history is
in `docs/planet-layer/journal/` (one file per pass); `docs/planet-layer-plan.md` is the frozen old single plan doc
(code comments and `.cmd` headers cite its passes: "Nth pass" = `journal/*_pass-NN.md`). The root `README.md`
documents the original GeoTIFF generator (`planet_diffusion` CLI), not this work.

## The chain

```
P  plate layer      planet_layer/plates.py        cube n=217 (46 km), 1 Myr steps, 250 Myr from a supercontinent
        |  snapshots: plate ids, rotations, crust, elevation, dynamic topography, tectonic thickening
        v
L  landscape stage  planet_layer/coupling.py +    cube n=868 (11.53 km), 50 kyr steps, last 30 Myr,
                    graph_landscape.py            present-day geometry with parcel histories
        v
H  hand-off         scripts/planet_meso_check.py  exact area mean onto the meso's 23.04 km cells
        v
M  meso             toy_meso/ (planet_tiled.py)   diffusion model 23 km -> 2.88 km, outputs/meso/detail-ft/step10000-passed.pt
```

Direction (user, 23 Oct): realism. Values come from Earth measurements or mechanism, never from fitting the planet's
output to a band; any system may be changed if it moves toward realism.

## Code

| Path | What it is |
|---|---|
| `planet_layer/plates.py` | plate layer: `PARAMS` (all tunables), `PlateSim` (step, rifting, collisions, margins, crust flow, edge collapse, coarse erosion, isostasy, water inventory, ledger) |
| `planet_layer/mantle.py` | dynamic topography (random spherical-harmonic fields blended in time), hotspots, LIPs |
| `planet_layer/shelf.py` | wave-base planation and cliff retreat (plate layer only) |
| `planet_layer/graph_landscape.py` | landscape kernels on any node graph: routing, Yuan erosion–deposition (n = 1), lakes, marine sediment, fill-and-spill, diffusion; `GraphLandscape` |
| `planet_layer/coupling.py` | parcel back-trace through plate snapshots, forcing (load, surface motion, z0), erodibility |
| `planet_layer/sphere_graph.py`, `sphere_flexure.py`, `fields.py` | cube-sphere graph (unique nodes), spherical-harmonic flexure (ducc0), lookups |
| `planet_training/landscape_evolution.py` | the original regional landscape model; its `DEFAULTS` (K, m, Sc, Kd, ...) are used by both grids |
| `planet_training/elevation_shading.py` | colours for every review picture (65,536 levels since 23 Oct; 256 drew false terraces) |
| `planet_training/arrangement_metrics.py` | high-ground / flanked-lowland / slope metrics (needs torch) |
| `toy_meso/` | the meso model: `planet_tiled.py` (whole-planet tiles), `detail.py` / `finetune_detail.py` (detail-aware fine-tune) |
| `planet_diffusion/` | the original generator; `cube.py` (cube geometry) is used by the planet layer |
| `tests/` | `test_plates.py`, `test_shelf.py`, `test_sphere_graph.py`, `test_graph_landscape.py`, `test_coupling.py`, `test_boundary_lengths.py`, `test_erosion_law_check.py`, `test_crust_reviews.py` (86 planet-layer tests since 28 Oct; some older test files elsewhere do not import) |

## Scripts (`python -m scripts.<name>`)

| Stage | Script | Purpose |
|---|---|---|
| P | `simulate_plates` | one plate run (`--seeds`, `--param KEY=VALUE`) |
| P | `sweep_plates` | 3-seed sweeps (`--grid K=V1,V2`, `--grid K1,K2=A1:A2,B1:B2`, `--rule shelf`); scores texture, hypsometry, arrangement, budget, export flux |
| P review | `crust_review` | continental area and thickness distribution vs CRUST1.0, label-free (total ≥ 20 km on both sides; flag reading beside) (`--latest`) |
| P review | `margin_width` | passive margin width and inland thickness profile vs CRUST1.0 at 1°, label-free (`--latest`) |
| Earth ref | `earth_deformation` | GSRM v2.1 strain rates on n = 217 vs Bird's orogens; the measured deforming-crust threshold (3.5 nanostrain/yr) |
| Earth ref | `earth_lithosphere_residual` | CRUST1.0 column Airy (type layer densities) vs ETOPO: the mantle lithosphere's part of elevation by thickness, crust type, inside / outside Bird's orogens (item C reference, thirty-fourth pass) |
| P review | `interior_probe` | provenance of interior and thick crust (start / collided / new), young-orogen and deforming land vs Bird's orogens; `--earth` adds CRUST1.0 by type group |
| P review | `margin_cross_section` | passive margins on both sides of the edge, label-free (total ≥ 20 km = CRUST1.0's continent) vs Earth |
| P review | `boundary_lengths` | true plate-boundary line lengths by type vs Bird 2003 (subduction split continental / intra-oceanic) |
| P review | `crust_budget` | item B: water below the datum, deep floor, volume, area tests; every ledger in km³/yr (whole run, last 40 Myr) vs Scholl & von Huene 2009; island-arc stock and trench arrivals; per-km rates (thirty-fifth pass) |
| P review | `tools/arc_growth_probe.py` | B-d2 (fortieth pass): rebuilds the trench-flux band terms on snapshots and compares them with the contour-typed consumption |
| P run | `tools/device_runs/chunk_run.py`, `simpickle.py` | full n = 217 runs in the device shell, pickled between 180 s calls (pre-checks; trajectories differ from Windows) |
| P review | `trench_origin` | item B-b: boundary geometry (shape ratio, detached plate pieces), ocean–ocean boundary and intra-oceanic trench, rates, ridge production and consumption, speeds vs the `_driving` law and Earth's plates, trench by plate pair; Earth from Bird's PB2002_steps + CRUST1.0 types (thirty-sixth pass) |
| P review | `ocean_sediment_probe` | item B-c: ocean-floor sediment stock by distance to trenches / continents / floor age vs GlobSed, and where it arrives (last 40 Myr, following the plates) (thirty-sixth pass) |
| Earth ref | `earth_sediment` | GlobSed v3 (ASCII grid, numpy only) on the cube grid: `outputs/planet-reference/globsed-n217.npz` |
| P review | `plate_pieces_review` | item B-b's tests: detached area every step, intra-oceanic and continental-side trench, ridge production and length (0 Myr and the −100…0 Myr window, against the control), plate count and sizes vs Bird, ocean-floor age vs Seton et al. 2020 (thirty-seventh pass); the start's S1 / S2 at −240 Myr (`--control` names the control; thirty-eighth pass) |
| Earth ref | `earth_seafloor_age` | Seton et al. 2020 age grid (`data/EarthByte`, the CDF1 `.grd` read with numpy): production, age distribution, GlobSed by floor age: `outputs/planet-reference/earth-seafloor-age-n217.npz` / `.json` |
| P review | `plate_speeds` | plate RMS speeds in a no-net-rotation frame vs Bird 2003 |
| P review | `plate_texture` | 46 km relief, band power and arrangement vs ETOPO |
| L | `run_planet_landscape` | the 11.53 km run (`--plates sweep` / `latest` / folder, `--seed`, `--theta`) |
| L | `calibrate_landscape_theta`, `theta_seeds_review` | θ calibration (fitted; to be replaced, see audit) |
| L review | `shelf_review` | final bands on 11.53 km runs (p70, drowned crust, land p25, lowland plains, land) |
| H/M | `planet_meso_check` | 23 km hand-off and meso on planet vs Earth windows (GPU) |
| laws | `erosion_law_check` | each erosion law on Earth's surface vs OCTOPUS ¹⁰Be basin rates (`--grid 217` / `868`) |
| Earth | `earth_reference` | ETOPO averaged onto the cube grids (`outputs/planet-reference/earth-n217.npz`, `-n868.npz`) |

The `.cmd` files in the project root chain these for the user (double-click): tests, the sweep, then the reviews into
`<sweep>/reviews/`; each one's header states its decision rule and the pass that wrote it. Current: `Trench Flux
Sweep.cmd` (fortieth pass, 28 Oct). The other sweep `.cmd`s in the root are finished (their sweeps are listed in
`docs/planet-layer/decisions.md`); they pass the working baseline explicitly, so the newest one is the template for
the next. Older ones (`Train *`, `Evaluate *`, `Launch *`, `Review Terrain *`) belong to the earlier learned-model work.

## Data (`data/`)

| Folder | Use |
|---|---|
| `etopo_2022_30s_surface.tif` | Earth's surface; averaged onto the grids by `earth_reference` |
| `tectonicplates-master/` | Bird 2003: plates, boundaries, poles, orogens (orogen zones = Earth's young belts for erodibility) |
| `octopus/crn_int_basins.json` | OCTOPUS v2.3 ¹⁰Be basin-averaged erosion rates (5,376 basins, polygons, EPSG:4326) |
| `crust1/` | CRUST1.0 thickness, sediment, crust types; summary `crust1-continental-summary.json` |
| `EarthByte/` | Seton et al. 2020 ocean-floor age and spreading grids (`Grids/`: `age.2020.1.GTS2012.2m.grd` is classic netCDF, read by `scripts.earth_seafloor_age`; the `.nc` files are netCDF-4 / HDF5) and the Müller et al. 2019 rotation / isochron / ridge feature files (`FeatureData/`, GPlates `.rot` / `.gpml`, not read yet) |
| `GlobSed/` | GlobSed total sediment thickness (Straume et al. 2019): `V3/GlobSed_package3/ArcGIS_ascii/globsed-v3.asc` (read by `scripts.earth_sediment`; the `.nc` files are HDF5) |
| `RiverATLAS_Data_v10_shp/` | drainage validation (not used yet) |
| `Terrain Ruggedness Index/` | 1 km ruggedness (possible meso detail target; low priority) |
| `Scotese_Wright_2018_.../` | paleo-DEMs (earlier work) |

## Outputs (`outputs/`)

| Folder | Contents |
|---|---|
| `planet-calibration/sweep-<time>/` | sweeps: `results.json` (`table`, `chosen`, `chosen_dir`), `sweep.json` (parameters shared by all combinations; since 24 Oct `<combo>` names only the parameters that differ, e.g. `rift_rule-pure_shear`, or `combo-01`..; full set in `<combo>/params.json`), `<combo>/seed<s>/` with `stats.json` (ledger in `present.erosion_totals_km3`, time `series`) and `snapshots/t<time>.npz` + `-plates.json` |
| `planet-landscape/` | 11.53 km runs: `present.npz`, `stats.json`, `preview.png`, cached `forcing/` |
| `planet-meso-check/<time>/` | meso check: `results.json`, `windows.npz`, pictures |
| `erosion-law-check/<grid>-<time>/` | erosion-law tests: `results.json`, `scatter.png` |
| `planet-reference/` | Earth on the grids and cached Earth metrics (`earth-plate-speeds-nnr.json`, textures) |
| `meso/` | meso data set (`dataset.npz`) and models (`detail-ft/step10000-passed.pt`) |
| `test-tmp/` | scratch and sync staging (safe to clean) |

Key runs: the current model's six-seed control is `sweep-20261027-210209/arc_H_ocean_m-300_arc_accretion-False`
(see `STATE.md`); landscape θ 5 `planet-landscape/theta-20261022-214812-seed1001/theta5` (on the 22 Oct plate
baseline). The newest sweeps are read with
`crust_review --latest` and `results.json`.

## Commands (PowerShell, project root)

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py
& '.\.venv\Scripts\python.exe' -u -m scripts.simulate_plates --seeds 1001 --param collapse_edge=True
& '.\.venv\Scripts\python.exe' -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid breakup_thick_m,rift_width_km=17500:0,10000:100
& '.\.venv\Scripts\python.exe' -u -m scripts.crust_review --latest
& '.\.venv\Scripts\python.exe' -u -m scripts.run_planet_landscape --plates sweep --seed 1001
& '.\.venv\Scripts\python.exe' -u -m scripts.shelf_review outputs\planet-landscape\<run1> outputs\planet-landscape\<run2> outputs\planet-landscape\<run3>
& '.\.venv\Scripts\python.exe' -u -m scripts.planet_meso_check --landscape outputs\planet-landscape\theta-20261022-214812-seed1001\theta5
& '.\.venv\Scripts\python.exe' -u -m scripts.erosion_law_check --grid 217
& '.\.venv\Scripts\python.exe' -m scripts.boundary_lengths outputs\planet-calibration\sweep-20261022-194148\breakup_thick_m-17500_erosion_theta-6\seed1001\snapshots\t+0000.0.npz
& '.\.venv\Scripts\python.exe' -m scripts.plate_speeds outputs\planet-calibration\sweep-20261022-194148\breakup_thick_m-17500_erosion_theta-6\seed1001\snapshots\t+0000.0.npz
```

(`<run1>` etc. stand for real folder names; give the user complete paths.)

## Settings

The current model, and which of its values are passed by the `.cmd` files rather than set in `PARAMS` (the old
23 Oct table here was stale): `docs/planet-layer/STATE.md`, section 3. Every value's source:
`docs/planet-parameter-audit.md`.

## Working with the user's machine (cloud sessions)

- Read and run jobs in the user's folder through the device shell (details, package reinstalls and limits: the
  working rules in `docs/next-chat-prompt.md`). Heavy work: stage the needed files into the cloud workspace (≤ ~400 MB
  per file; large files time out, so slim them first on the device).
- Sync edits: copy to a fresh staging folder under a new file name, check the device md5, commit, verify the md5.
- The user's `.venv` has no scipy or scikit-image: planet code stays numpy / numba / ducc0 / matplotlib.
- Neither shell reaches most data servers; the user downloads in a browser (right-click, Save Link As).
- After a pass: a new journal file, `STATE.md` rewritten, rows in `decisions.md`, the audit if values changed, this
  map if scripts or `.cmd`s changed, and the Project copies (`claude/...`); see "How to keep the docs" in
  `docs/next-chat-prompt.md`. Tell the user when a new chat is a good idea.

Old sweep folders were renamed to the short naming on 24 Oct; `outputs/planet-calibration/renamed-2026-10-24.json` and each sweep's `sweep.json` (`renamed_from`) list the old names; stored paths in planet-landscape runs were rewritten.
