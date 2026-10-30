# Planet layer: design (proposed 17 Oct 2026, awaiting agreement)

What decides where continents, plate boundaries, young and old mountain belts, cratons, rifts and basins sit on a
whole planet, and how that feeds landscape evolution on the cube sphere and then the meso at 23.04 km.
Background: `docs/research/planet-scale/README.md` (survey), `docs/tectonic-prototype.md` (plan of record so far).
Nothing here is built yet. Numbers marked *est.* are estimates, to be replaced by measurements.
Long-term target (all stages, outputs and statistics, later additions included): `docs/planet-ideal-flow.md`.

## 1. The chain

```
P  plate layer      cube n=217 (46 km), 1 Myr steps, ~250 Myr      kinematics, crust, coarse isostatic elevation, coarse erosion
        |  saved every 1 Myr: plate ids + rotations, boundary types and distances, rates, crust attributes
        v
L  landscape layer  cube n=868 (11.53 km), 50 kyr steps, last ~30 Myr   our landscape model on a sphere graph
        |  present day: elevation, drainage, basins, lakes, erodibility, rock age, recent uplift
        v
H  hand-off         exact area mean onto the meso's n=434 coarse cells (23.06 km)  [+ detail map, later]
        v
M  meso             toy_meso.planet_tiled with outputs/meso/detail-ft/step10000-passed.pt (2.88 km)
```

The grids nest exactly: `planet_tiled` uses n = round(quarter circle / 23.04 km) = 434 coarse intervals per face
(N = 3,472 fine), so L at n = 868 has exactly 2x2 L cells per meso condition cell, as in the test bed.

## 2. Options compared

Key point: **goSPL and fastscapelib are alternatives only for L.** Neither makes plates; both consume uplift and
displacement maps (goSPL usually from GPlates reconstructions). A plate layer (P) has to be built in every option.

| | A. Own P (Cortial rules) + our model ported to the cube (recommended) | B. Own P + goSPL as L | C. Own P + fastscapelib HEALPix as L |
|---|---|---|---|
| Processes in L | exactly the validated set (Yuan erosion-deposition, priority flood, deltas, graded shelves, hillslope and marine diffusion, flexure) | similar set, but MFD routing and its own deposition scheme: unvalidated against the meso inputs | planetary example covers stream power and sink resolution only; no marine, deposition or flexure |
| Grid vs meso | the meso's own cube: no resampling | unstructured mesh, so it must be resampled to the cube | HEALPix, so it must be resampled to the cube |
| Why resampling matters | — | interpolating at the target spacing is the ~0.7-cell blur that caused the smooth-plains gap (version 3 check) | same |
| Horizontal plate motion | not needed with the present-geometry frame (sect. 4) | yes: inverse-distance interpolation at the end of each tectonic interval (itself smoothing) | no |
| Flexure on the sphere | spherical harmonics (to build; goSPL does the same) | yes (pyshtools, spherical harmonics) | no |
| Windows / RTX 3070 desktop | numba, as now | PETSc/MPI, Docker or WSL; published global runs on HPC | HEALPix only on Linux/macOS (WSL) |
| Licence | ours | GPL-3 | BSD-3 |
| Effort | port: neighbour lists instead of grid offsets, plus flexure | install, mesh, two-way remapping, recalibration | WSL, missing processes, remapping |

**What I'd take from the others:** goSPL's spherical-harmonic flexure (with Picard iteration for variable Te
later) and its tip to smooth Te steps; Cortial's plate rules (sect. 3); and the framing in Hergarten & Fraters 2026
(EGUsphere preprint): Eulerian advection on a fixed grid diffuses, Lagrangian moving nodes need remeshing, which can
dam rivers, and "frozen flow pattern" variants exist. That framing informs sect. 4.

**Not worth it:** learned layout models (superseded), platec/worldengine (flat), GPlates as the engine (laborious
for invented planets; Earth reference only), mantle convection (too costly, not steerable).

## 3. P: the plate layer

**Representation: per-plate frames on a fixed grid.** Cortial moves crust samples and periodically re-triangulates
them, which second-system.de found fiddly (self-intersections, mesh repair). Instead, every plate k keeps its own
raster on the same n = 217 cube graph *in its own rotating frame*, plus a rotation Q_k(t). The raster stores mask,
crust type, crustal thickness, crust age, last orogeny age and type, oceanic lithosphere age and sediment
thickness. Each step:

1. **Compose:** for every world node p and plate k, look up the nearest node of Q_k(t)^-1 p in plate k's raster.
   That is ~282k x ~15 lookups (*est.* tens of ms in numba).
2. **Gaps** (no plate) are divergence: new oceanic crust of age 0 joins the nearer plate (ridge).
3. **Overlaps** are convergence: oceanic under continental, and older (denser) ocean under younger. The subducting
   cells leave that plate's mask. Two continents overlapping is a collision.
4. **Crust is never resampled from itself.** Plate rasters are only written where crust is created or changed:
   erosion, thickening, a new ridge. So 250 Myr of motion adds no interpolation blur.

**Kinematics (Cortial plus Earth's force balance).** Each plate has an Euler pole and an angular speed. Plates
pulled by attached slabs drift toward their trenches and move fast (5–10 cm/yr). Plates that are mostly continent
move slowly (1–3 cm/yr). This is Forsyth & Uyeda's observation, and Cortial's slab-pull axis drift.

**Events:**
- **Rifting.** Poisson events. Their rate grows with plate area, continental fraction and time since the last
  rift. The rift path prefers old sutures (the Wilson cycle).
- **Collision.** Convergence slows and the belt thickens. The suture is recorded. Small terranes transfer to the
  overriding plate (Cortial).
- **Hotspots** (fixed in the mantle frame). They add volcanic crust as plates pass over, giving seamount chains,
  oceanic plateaus and LIPs. Optional, but cheap here, and Cortial lacks them.
- **Mantle swells and basins.** A slowly evolving low-degree field in the mantle frame (~50–100 Myr timescale).
  It is passed as **non-load** vertical motion, with Earth-like amplitude (residual topography ~±1 km, rates
  10–100 m/Myr) in surface terms. See the note at the end of this document.

**Start: a supercontinent at ~-250 Myr.** It is assembled procedurally from cratons (Voronoi aggregates of very old,
thick, hard crust), welded by old orogens of assigned ages (300–1,500 Ma, low relief) and ringed by ocean. The
simulation then rifts it apart, opens oceans, starts subduction where plates converge, and makes collisions. That
gives an Earth-like mix of passive margins, active margins and orogens of several ages. A dispersed start is an
option.

**Elevation in P is isostatic.** Continents use Airy isostasy from crustal thickness and density, with the water
load below 0 m. Ocean depth follows thermal age (GDH1: ~2.6 km at the ridge, deepening as root age, flattening to
~5.6 km). Dynamic topography is added on top. The constants are Earth's: ~35–40 km crust stands near 0–500 m.
**Land fraction is an output** of continental crust area and thickness. Nothing projects it, and sea level is only
the 0 m datum.

**Coarse erosion in P.** The same graph landscape kernels (routing, erosion-deposition, lakes, sea) run at 46 km with
dt 250 kyr, four steps per plate step, on the composed elevation. Flexure is off, because P's isostasy replaces it.
Erosion and deposition are scattered back to the plate rasters as thickness changes. K and Kd at 46 km are
calibrated once against the 11.52 km model on the same forcing; the rule is in sect. 6. The fallback is Cortial's
simple decay rule (ε 0.03 mm/yr).

**Cost (est.):** ~1–2 s per Myr, so ~5 min for 250 Myr on the 9800X3D.

**Saved every 1 Myr (world frame, 46 km):** plate id, plate rotations, velocity, boundary type, signed distance to
each boundary type, convergence rate, overriding and subducting crust type, time since collision onset, crust type,
crust age and thickness, orogeny age and type, elevation, and the dynamic-topography rate.

## 4. L: landscape on the sphere

**Port.** The kernels of `planet_training/landscape_evolution.py` are rewritten on an explicit graph: neighbours
(N x 8, -1 for none), edge lengths and node areas. They are D8 on true distances, priority flood, routing and stack,
accumulation, Yuan erosion-deposition, lakes, marine sediment with graph distance-to-land (multi-source Dijkstra),
and conservative hillslope and marine diffusion. The same kernels then run on the flat test window (regression
against the current module) and on the cube sphere.

The cube graph has 6n²+2 unique nodes, built from `planet_diffusion/cube.py` (`edge_groups`, `directions`). Edge
nodes get 8 neighbours across two faces, and the 8 cube corners get 6. Node spacing varies by up to √2 over a face;
true distances and areas handle that. The old module stays untouched for the test bed.

**Flexure on the sphere.** Thin-plate response per spherical-harmonic degree, with k² = l(l+1)/R². The load is
area-averaged onto a Gauss–Legendre grid, transformed, filtered, and the deflection interpolated back. With Te 30 km
the response falls below ~1.5% beyond l ≈ 380, so lmax 512 is enough. pyshtools and ducc0 both have Windows wheels
for Python 3.10. Constant Te first, as now; variable Te (thick cratons) later.

**Two forcing channels:**
- thickening rate (m/yr of rock), which goes into the flexural load, as uplift does today;
- non-load vertical rate (dynamic topography, thermal subsidence of the ocean floor), which moves the surface
  directly.

**Frame: present-day geometry with parcel histories (recommended).** L runs on one fixed grid in the present-day
plate configuration. For each node, the crust parcel now there is traced back through the saved plate-id snapshots
and rotations, one Myr step at a time. Its forcing at time t is P's forcing at the position it occupied then.

- This is exact inside plates, which are rigid, and is the same as evolving each plate in its own co-moving frame.
- There is **no resampling at all**, so there is no blur in the 20–50 km band the meso depends on.
- Mantle-frame effects (swells, hotspots) come out right automatically, because a parcel feels them as it drifts
  over them.
- **What it gets wrong:** the kinematics near converging and diverging boundaries during the L window. A collision
  belt is thickened in place, by the right volume, rather than built by material moving through it. Before a
  collision, the closing ocean between the two continents is narrower than it really was. A young ocean is ocean
  for the whole window. Rivers are not carried along by thrusting.
- With convergence of 2–8 cm/yr over 30 Myr, these errors sit within a few hundred km of active sutures. The test
  bed also uplifted in place, and the meso found those belts convincing.

**Alternative, only if validation shows a failure traceable to this.** Per-plate remap every 1 Myr by nearest node:
no blur, at most half a cell of jitter, with ridge infill and trench removal at boundaries. Its test would be
random sub-cell remaps on the flat window, with 23 km curvature per relief class within 5% of no remap.
Interpolating advection (cubic, or goSPL's inverse-distance weighting) is ruled out. Each remap attenuates the
4-cell band a little, and ~15–30 remaps compound.

**Fine forcing shapes.** The fine forcing is computed on the 11.53 km grid from P's smooth signed-distance fields,
interpolated from 46 km.

- **Amplitudes come from mass balance:** convergence x crustal thickness, spread over the belt. For example, 3 cm/yr
  x 35 km over a 300 km belt is a mean thickening of ~3.5 mm/yr. This matches the test-bed peaks of 5.5–7.5 mm/yr.
- **Shapes generalise the test-bed families:**
  - collision: asymmetric pro/retro belt with en-echelon segments;
  - ocean–continent: coast range, forearc, arc block and back-arc;
  - rift: trough and shoulders, then thermal subsidence;
  - transform: none.
- **Erodibility** comes from rock age and type: young belts and sediment high, cratons 0.15, with the test bed's
  patchy hardness.

**Initial surface** at the start of the window is the parcel elevations from P, plus small seed roughness for
routing. **Open sea** is the present-day connected ocean.

**L window: 30 Myr by default** (20 / 40 tested at step 7), dt 50 kyr. dt 100 kyr is adopted only if the flat-window
regression shows the same statistics (sect. 6).

**Cost.** The test bed measures 0.76 µs per cell-step (256², 600 steps, 30 s). Allowing ~1.3x for graph
indirection (*est.*):

| L grid | nodes | s/step (*est.*) | 30 Myr at 50 kyr | use |
|---|---|---|---|---|
| n=868, 11.53 km | 4.52 M | ~4.5 | ~45 min | production (the spacing the processes were tuned at) |
| n=434, 23.06 km | 1.13 M | ~1.1 | ~11 min | plumbing checks only, never for judging terrain |
| n=217, 46.1 km | 0.28 M | ~0.3 | (P's coarse erosion, dt 250 kyr) | P |

A global 250 Myr run at 11.53 km would take ~6 h (*est.*, 5,000 steps). That is less out of reach than the survey
assumed, but cost is not the main reason for two clocks. Over 250 Myr the plate geometry reorganises completely,
so the fine state would have to be remapped hundreds of times, with blur or jitter each time. The present-geometry
frame is only valid for a window that is short compared with plate reorganisation. Iteration speed matters too.

## 5. H: hand-off to the meso

- **Height.** The exact area mean of L's node cells onto the meso's n=434 coarse cells, passed to `planet_tiled`
  as a cube array. This is a new input mode. Going through an equirectangular GeoTIFF would interpolate twice and
  soften the input. A test checks that the operator equals what `coarse_condition_cells` computes from a raster of
  the same field, when that field is smooth.
- **Detail map** (step 9, later). A predictor of sub-cell detail fitted on Earth from inputs available on both sides:
  province class and age (Hasterok 2022 on Earth, P on the planet) and 23 km relief over a window. The rule, fixed
  before fitting: on held-out Earth with blurred (0.7) input, incision and 5.8–11.5 km power with the predicted map
  must be closer to 1 than with detail unknown, in all three relief classes.

## 6. Tests, fixed before building

| Piece | Test | Pass |
|---|---|---|
| sphere graph | total area vs 4πR²; neighbour symmetry; counts 6–8; spacing ratio | area to 1e-6; symmetric; ratio ≤ √2 |
| graph port (flat) | collision s1 on the 256² window vs the current module | identical for the first 20 steps; 30 Myr statistics within 1% |
| dt 100 kyr | same run at dt 100 vs 50 kyr | 368 km relief percentiles and 23 km curvature per class within 5% |
| sphere processes | synthetic cone on a cube corner and an edge; mass ledger | continuous drainage area across faces; uplift - erosion + deposition - export balances to 1e-6 |
| sphere flexure | 500 km disc load on a cap vs flat FFT | deflection within 2% |
| global timing | 1 Myr at n=868 with synthetic uplift (your machine, a few min) | sets the L window and dt by measurement |
| P coarse-erosion calibration | test-bed collision forcing at 46 km vs 11.52 km, 30 Myr | 368 km-window mean elevation and denudation within 15% |
| P layout (5 seeds, present day) | land fraction; hypsometry; ridge depth; plate sizes; provinces | land 25–35% (Earth 29%); bimodal, with peaks within 500 m of Earth's; ridge crest 2.3–2.8 km; largest plate 40–110 Mkm² and log-size distribution vs Bird 2003; province shares vs Hasterok 2022 (Earth shares computed from the dataset first) |
| L present day | 368 km land relief | p10/p25/p50/p75 within ±25% of 133/237/456/872 m; 25–45% under 300 m (Earth 34%) |
| L present day | 40 km arrangement metrics, by setting | inside the Alps / California / Plains reference spread |
| L present day | 23 km curvature per relief class vs `outputs/meso/dataset.npz` inputs | not below the test bed (17 / 34 / 66 m) |
| end to end | `diagnose_meso_plains` on planet windows | incision per class ≥ 0.7 of Earth |

## 7. Build order (small, testable pieces)

1. `planet_layer/sphere_graph.py`: the unique-node cube graph (neighbours, distances, areas, face↔node maps). CPU,
   my workspace.
2. Graph landscape kernels and the flat-window regression. CPU, my workspace.
3. Sphere processes and spherical-harmonic flexure, with their tests. CPU, my workspace.
4. Global timing run at n=868. Your machine, a few minutes. Fixes the L window and dt.
5. P1: plates, per-plate frames, composition, ridges, subduction, ages, from a supercontinent start. Maps every
   10 Myr. Checks: area closure, age-depth, plate sizes.
6. P2: collision, rifting, terranes, isostasy, hotspots, swells, coarse erosion and its calibration. Checks:
   hypsometry, land fraction, provinces. Needs the Hasterok 2022 dataset downloaded from Zenodo on your machine.
7. Coupling: parcel back-trace, fine forcing, erodibility, then the L run at n=868. Your machine, ~45 min (*est.*).
8. Hand-off into `planet_tiled`; meso on chosen windows (your GPU); the full validation table.
9. Detail-map predictor, with its rule fixed in advance.

## 8. Decisions (agreed 17 Oct 2026)

1. **L frame:** present-day geometry with parcel histories. Per-plate nearest-node remapping stays the fallback,
   with the test in sect. 4.
2. **L window:** 30 Myr default. Shorter windows give drainage less time to integrate and dissect the inherited
   surface. Longer windows grow the present-geometry error and cost more. 20 and 40 Myr are run at step 7, and the
   one that scores best on the validation table (sect. 6) becomes the default.
3. **Coarse erosion in P:** graph landscape kernels at 46 km, calibrated per sect. 6.
4. **Start:** supercontinent at ~-250 Myr. A dispersed start is a later option (`--start dispersed`).
5. **Order:** port first (steps 1–4), then P.

## Note: the test-bed swells were a load

`tectonic_templates.inherited_events` adds the swells to the uplift field. `Landscape.step` adds all uplift to the
flexural load. At the swell wavelength (~650 km), Te 30 km leaves only ~22% of a load as surface uplift (100%
below ~100 km, 18% beyond ~1,000 km). So the version 3 swells (22–40 m/Myr as load) raised the surface by ~5–9
m/Myr, and the version 2 swells by ~1–2 m/Myr. The test-bed conclusion stands, in surface terms. The planet layer
passes swells as non-load motion with Earth-like surface rates. The test bed itself is not changed.

## Sources added in this step

- [Hergarten & Fraters 2026, Horizontal movement and deformation in large-scale landform evolution models (EGUsphere)](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2060/)
- [goSPL tectonic forcing: advection and flexure](https://gospl.readthedocs.io/en/latest/tech_guide/tecto.html)
- Survey sources: `docs/research/planet-scale/README.md`

## 9. Progress

### Steps 1–3 done (17 Oct): sphere graph, graph landscape kernels, sphere flexure

New code (the test-bed module `planet_training/landscape_evolution.py` is unchanged):
- `planet_layer/sphere_graph.py`: `CubeSphere(n)` builds the unique-node graph; `flat(ny, nx, dx)` builds the same
  arrays for the test window. Includes `to_nodes` / `to_cube` conversion to the shared cube layout.
- `planet_layer/graph_landscape.py`: `GraphLandscape` with every process of the regional model on a graph, plus a
  second forcing channel, `surface_rate` (non-load motion). It also reports a timing breakdown per process.
- `planet_layer/sphere_flexure.py`: `SphereFlexure`, using ducc0 non-uniform spherical-harmonic transforms straight
  from node values with cell-area weights. lmax 480 for Te 30 km.
- `tests/test_sphere_graph.py`, `tests/test_graph_landscape.py`: 8 tests, ~10 s.
- `scripts/planet_landscape_timing.py` (**Planet Landscape Timing.cmd**).

| Test (sect. 6) | Result |
|---|---|
| sphere graph | area exact to 1e-15; neighbours symmetric; 8 corners with 6 neighbours, the rest 8; axis spacing 0.707–1.0 of nominal (√2). n=868: 4.52 M nodes, built in 15 s, 1.8 GB peak |
| flat regression, first 20 steps | identical routing; heights within 1e-11 m |
| flat regression, 30 Myr, collision s1 | max difference 2.5e-11 m after 600 steps |
| flat regression, 30 Myr, margin s1 | diverges at step 408. The original model diverges identically from itself under a 1e-9 m perturbation (see below) |
| graph distance vs chamfer | equal (rtol 1e-12) |
| sphere: cone on a cube corner, 2 Myr | every node on a drainage tree; all land drains to the sea; the three faces' land within 15%; volume ledger exact (1e-9) |
| surface_rate | moves the surface, adds no load |
| sphere flexure vs flat FFT | ratio 0.999–1.000 at 0–500 km under a 500 km disc (n=434); within 2% of peak at n=217; broad-scale compensation 0.818 = ρc/ρm |

**On the margin run and the 1% rule.** At step 408 a near-tie of ~1e-11 m (floating-point order, from the
graph-form diffusion) flips one routing choice. From there the two runs differ cell by cell: up to 4.5 km, with 47%
of cells differing by more than 1 m.

The original module, given a 1e-9 m random perturbation of its starting surface, ends in *the same* state as the
ported code. Its statistics differ from the unperturbed run by the same amounts: land p50 / p95 within 0.3 m,
23 km curvature 50.32 vs 50.37 m, 368 km relief p25–p75 within 0.4%, p10 199.8 vs 208.0 m (4%).

So the 4% is the model's own chaotic noise floor, not a porting error, and the 1% rule was tighter than the model
can meet. The regression rule is now: per-step identity on the flat window (tested), and 30 Myr statistics within
the spread the original model shows under a 1e-9 m perturbation.

**Speed changes, with identical results:**
- distance-to-land stops at the shelf break, the only range used;
- the priority flood seeds only outlets that touch land.

At 23 km this took the step from 1.5 s to 0.85 s; both changes are covered by the flat regression.

**Timing in my workspace (2 cores, synthetic planet, 30% land):**

| Grid | Time per step | Breakdown | 30 Myr at 50 kyr | Peak memory |
|---|---|---|---|---|
| n=434 | 0.85 s | | ~14 min | |
| n=868 | 4.1 s | routing 1.1, erosion 1.0, marine 0.6, diffusion 0.5, flexure 0.6 | ~41 min | 2.2 GB |

The run on your machine sets the real numbers (step 4).

### Step 4: global timing run (your machine)

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -m pip install ducc0
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_sphere_graph.py tests\test_graph_landscape.py
& '.\.venv\Scripts\python.exe' -u -m scripts.planet_landscape_timing
```

**Planet Landscape Timing.cmd** does the install and the run. Output goes to
`outputs\planet-landscape-timing\<timestamp>\`: `timing.json` and `preview.png`.

### Step 4 done (17 Oct): global timing on your machine

On the 9800X3D at n=868 (4.52 M nodes, 11.53 km), a step takes **1.22 s**:

| Process | Seconds per step |
|---|---|
| routing | 0.36 |
| erosion | 0.31 |
| marine | 0.24 |
| flexure | 0.13 |
| diffusion | 0.13 |

So **30 Myr takes ~12 minutes** and 40 Myr ~16 minutes. All land drained to the sea. A zoom on the cube corner at
35°N 135°W shows rivers crossing face edges without breaks; the straight line in the preview is one of the
synthetic belts.

**Decision:** keep dt 50 kyr (testing 100 kyr isn't worth it at this cost) and keep the 30 Myr default window.

**Correction to an earlier remark.** The regional model's substep rule (Kd_marine dt / h_min² ≤ 0.2) is *more*
conservative on the cube sphere than the graph's true explicit-diffusion bound (6/h² vs 10/h²), so marine diffusion
was always stable. `Graph.diffusion_bound()` now computes the true bound, and `GraphLandscape` uses whichever rule
needs more substeps. On the flat window that is still the regional rule, so the regression is unchanged.

### Step 5 done (17 Oct): plate layer part 1

New: `planet_layer/plates.py` (`PlateSim`), `planet_layer/fields.py` (fast nearest-node lookup, random spherical
fields, rotations), `scripts/simulate_plates.py` (**Simulate Plates.cmd**), `tests/test_plates.py`.

**What is built** (each plate keeps its crust in its own rotating frame, sect. 3):

- **Supercontinent start.** One continent (38% of the sphere) of 7–11 cratons, welded by old orogens of 300–1,500 Ma
  and thinned at its rim. It sits on 5 of 12 Voronoi plates whose poles move them outward. The surrounding ocean is
  aged by distance from the initial ridges.
- **Each step (1 Myr):**
  - rotate the plates;
  - build the world by backward lookup;
  - gaps become new 7 km ocean crust on the nearest plate (ridges);
  - overlaps: continental crust beats oceanic. Ocean–ocean overlaps get **one polarity per pair of plates**, kept while
    they stay in contact (deciding per node gave comb-like fingers);
  - overlapped crust is **hidden for 3 steps** before it is consumed (Cortial keeps subducted samples similarly);
    single-step jitter overlaps along jagged or transform boundaries otherwise destroyed crust and left holes. When
    hidden continental crust is consumed, its volume goes to the overriding continent (collision thickening);
  - continental crust spreads laterally with a diffusivity that rises e-fold per 6 km above 45 km (gravitational
    collapse), which keeps plateaus near 70–75 km.
- **Changes reach the plate rasters only where the world reads them.**
  - Thickness changes go to the plate nodes the world actually samples: diffusion increments are averaged, collision
    volume is added as volume. Sending them through the forward map instead let unread nodes drift without feedback,
    to ±600 km.
  - Continental crust is never duplicated by edge sampling: only genuinely new ridge crust, or ocean behind a ridge,
    switches a plate node on.
  - Single-node holes in a plate raster are closed from a neighbour.
- **Motion.** Slab pull turns the pole toward the plate's trenches and speeds it with trench length (to 9 cm/yr).
  Ridge push adds up to 1.5 cm/yr away from ridges. Continental area (×(1 − 0.3 f)) and collision length slow it.
- **Boundaries.** Ridge, subduction, collision, or transform. A boundary is a transform when its normal rate is under
  half its shear rate.

**Results** (n=217, 46 km, seed 1, 250 Myr; 15 min on my 2-core workspace):

| Quantity | Planet | Earth |
|---|---|---|
| Land fraction (present) | 30% | 29% |
| Largest plate | 113.5 Mkm² | 104 Mkm² (Pacific) |
| Plates | 10 (12 at start; 2 consumed) | 52 (Bird 2003), ~15 large |
| Ocean age, mean / max | 77 / 362 Myr | ~64 / ~180 Myr |
| Ridge / trench length (node count) | 129,000 / 90,000 km | ~65,000 / ~55,000 km |
| Plate speeds | 2.1–10.5 cm/yr | ~1–10 cm/yr |
| Elevation p95 | 4.2 km | — |
| Continental volume over 250 Myr | +0.9% | conserved (no erosion yet) |
| Holes inside plates | 0 at most steps; 16 at worst | 0 |

Notes on the table:
- Ridge and trench lengths are node counts and include the staircase factor (~1.3), so ~100,000 / ~70,000 km as
  lines.
- The tests also check that every node is owned, there are no holes, the continental volume ledger closes,
  ocean ages are non-negative, and new crust has age 0.

The supercontinent breaks into pieces separated by Atlantic-style oceans with ridges. Trenches line the outer
margins, the pieces disperse over 250 Myr, collision belts form where pieces meet, and the seafloor shows ridge-
centred age bands.

**What part 2 has to fix, from these numbers:**
1. **Plate count** (10, no small plates): rifting, i.e. plate splitting with Poisson timing, preferring old sutures.
2. **Ocean that never subducts** (max 362 Myr): subduction initiation where old ocean meets a passive margin, or
   inside old ocean. This uses the same splitting machinery.
3. **Plateaus and high p95** (4.2 km): coarse erosion (graph landscape kernels at 46 km, calibrated per sect. 6),
   whose sediment also rebuilds continental margins. Without erosion, continental area shrinks from 38% to 30% while
   volume is conserved, because collisions thicken crust.
4. **Abrupt rifted margins** (full-thickness crust next to new ocean): stretching of continental crust at divergent
   continental boundaries before breakup, to give passive margins.
5. Terrane transfer, hotspots and mantle swells (sect. 3).

### Step 6 (17 Oct): plate layer part 2. Built, calibrated at n=96, full 46 km check running

**Part 1 on your machine.** Seeds 1001–1003, ~9.5 min each:

| | Result | Earth |
|---|---|---|
| Land | 28.5–29% | 29% |
| Ocean age, mean / max | 69–85 / 328–375 Myr | ~64 / ~180 Myr |
| Plates | 9–10 | — |
| Largest plate | 164–231 Mkm² | 104 Mkm² |
| Elevation p95 | 4.1–5.3 km (uneroded plateaus) | — |

Part 2 addresses all of these.

**New and changed:** `planet_layer/mantle.py`. Also changed:
- `plates.py`: erosion, flexural elevation, subduction zones, rifting, splitting, sediment basins;
- `graph_landscape.py`: `surface_step()`, refactored out of `GraphLandscape.step` (the flat regression is unchanged),
  and an optional incision threshold `theta`;
- `sphere_flexure.py`: `rho_load` / `rho_mantle` arguments;
- `scripts/calibrate_coarse_erosion.py`;
- tests, 15 in total.

**1. Coarse erosion, calibrated (sect. 6 rule).** The test-bed templates are built once at 11.52 km and area-averaged
to 46 km, so both grids get identical forcing. A first attempt compared separately rendered templates, whose uplift
volumes differed by −16% to +39%, so it was discarded. Results, using `graph_landscape` on both grids:
- best settings: K ×1, Sc 0.05, Kd ×4, dt 250 kyr;
- 368 km block mean elevation within 11%, total denudation within 6%;
- stable-land denudation over the last 10 Myr within 18%: the coarse grid is slightly *lower* (29–35 vs 30–41
  m/Myr). This misses the 15% rule narrowly and is recorded as such.

**2. Incision threshold for long runs** (`erosion_theta` = 25, plate layer only).
- Over 250 Myr without a threshold, low-gradient land kept eroding at tens of m/Myr. That planed continents to sea
  level: 14–18% land, half of the continental crust under water. Earth's cratons and platforms erode at ~1–10 m/Myr.
- E = K(AᵐS − θ). θ = 25 was chosen against Earth, not the test bed, from θ ∈ {0, 10, 25, 50}:

| Land elevation | Denudation at θ = 25 | Earth (roughly) |
|---|---|---|
| 200–500 m | 5 m/Myr | ~1–10 |
| 0.5–1 km | 55 m/Myr | ~50–100 |
| 1–2 km | 222 m/Myr | ~200–500 |
| Mean land | 63 m/Myr | ~55 |

- Land below 200 m aggrades.
- **Watch-point:** the 11.52 km stage (θ = 0) erodes stable land at 30–40 m/Myr. That is harmless over its 30 Myr
  window but faster than Earth. Revisit if the final hypsometry is low.

**3. Elevation by flexure** (`PlateSim.elevation`). Loads are given as mass, with two plate responses:
- continental crust anomaly: density ρc against the mantle;
- oceanic crust anomaly and ocean-floor sediment: (ρ − ρw) against (ρm − ρw), switched to dry loading above sea level.
  A basin filled to sea level then holds ~2.5× its depth of sediment, as on Earth.
- The ocean floor follows GDH1 age-depth.
- A thinned continental column is water-loaded on its own Airy depth only. Applying the factor to neighbours'
  deflection made −29 km pits.

**4. Subduction zones** (from the previous step's boundaries, on the overriding plate). Flux = convergence × H,
spread as Gaussian bands:
- continental margin thickening: H = 1,000 m at 250 ± 120 km from the trench;
- island arcs on oceanic crust: H = 300 m at 150 ± 50 km. They become continental crust at 20 km.
- **Subduction erosion** removes crust at 50 ± 40 km: H = 700 m. This is the forearc loss that balances arc growth
  on Earth (~1.3–1.7 km³/yr).
- 30% of trench sediment is accreted.

These were set by the continental volume budget. With arcs at H = 800 m (Earth's per-km rate) and no subduction
erosion, continents grew 17–29% in 250 Myr. 85% of that growth came from intra-oceanic arcs, because every
subduction initiation creates a new ocean–ocean trench.

**5. Continental rifting.** A gap opening amid continental crust becomes stretched continental crust: 0.6 × its
neighbours' thickness, with the volume taken from them. Below 20 km, ocean floor forms (breakup). This also removed
one-node gaps that became 20+ km sediment traps. Wider rifting (0.75, breakup at 15 km) left half of the continental
crust submerged.

**6. Sediment basins become continent.** Oceanic nodes carrying more than 10 km of sediment, or sediment above sea
level, become continental crust with the thickness that keeps their elevation unchanged. Enclosed old-ocean remnants
otherwise filled without limit (36 km) and were also the "oldest ocean".

**7. Mantle** (`mantle.py`, mantle frame).
- Dynamic topography: degree ≤ 12 random fields blended every 40 Myr, std 400 m, non-load motion.
- ~10 hotspots plus Poisson births: steady volcanic crust, an 800 m swell, and a LIP burst at birth.

**8. Plate splitting.** It keeps the plate's frame, with no resampling.
- Rifting: 0.006 per Myr per 100 Mkm², doubled for mostly continental plates. The line prefers old sutures; the halves
  diverge at 2 cm/yr.
- Subduction initiation: 0.006 per Myr × min(5, old area / 2 Mkm²) for ocean older than 170 Myr. The old ocean
  converges on the rest of the plate at 3 cm/yr.

**Calibration result** (n=96, seed 1, 250 Myr, final defaults):

| Quantity | Planet | Earth |
|---|---|---|
| Land fraction | 28.8% | 29% |
| Continental crust area | 37% | ~40% |
| Share of continental crust above sea level | 78% | ~72% |
| Mean land elevation | ~640 m | ~840 m |
| Continental volume change | +0.7% | ~steady |
| Plates | 22 | — |
| Ocean age, mean / max | 44 / 281 Myr | ~64 / ~180 Myr |
| Thickest ocean-floor sediment | ~10 km | 10–20 km |

**Still open:**
- the oldest ocean (small enclosed remnants);
- mean ocean age is on the young side;
- land is ~200 m lower than Earth's.
The full 46 km check (3 seeds) comes next. Then terrane transfer, and the coupling to the landscape stage (step 7).

### Step 6, continued (17 Oct): your three-seed run, merging, and θ at the right grid

**Your run (20261017-103320, n=217, seeds 1001–1003, ~9 min each):**

| | Result | Earth |
|---|---|---|
| Land | 26.6 / 32.3 / 30.8% | 29% |
| Continental crust | 36.5–37.6% | — |
| Mean land elevation | 932 / 1,347 / 1,218 m | — |
| Land median | 587 / 967 / 785 m | — |
| Ocean age, mean / max | 52–67 / 277–296 Myr | — |
| Plates | 31–38 (24–35 over 1 Mkm²) | — |
| Largest plate | 59–70 Mkm² | — |
| Continental volume | +4 to +8% | — |

The maps read as Earth-like: an Atlantic-type ocean with a central ridge, trenches along active margins, collision
belts, sediment along the margins, and hotspot swells.

Three findings:
1. **Land too high.** θ was calibrated at n=96 (~100 km cells), but A^m S thresholds depend on the grid spacing. That
   broke the rule "calibrate at the spacing you run at". θ is now calibrated at n=217 against Earth on the same grid
   (`scripts/earth_reference.py`: ETOPO 2022 30″ area-averaged onto the n=217 nodes, ice sheets excluded;
   `scripts/calibrate_plate_erosion.py`). Selection rule, fixed in advance: pick the smallest worst error over land
   fraction and land p25 / p50 / p75 / p90, averaged over seeds.
2. **Too many plates and no large ones.** Plates split but never merged. Now two plates in continental collision along
   more than 1,000 km for 40 Myr in a row merge. Every node of the larger plate's raster that lies in the smaller
   plate's territory reads its crust (backward lookup, no holes); overlapping continental volume is added; the seam is
   filled. With 15 Myr / 800 km, the supercontinent's initial plates welded within 50 Myr (12 → 4 plates).
   n=96 check: 16 plates at present (14 over 1 Mkm², largest 76 Mkm²), land 29.7%, continental volume +0.4% over
   250 Myr.
3. **The oldest ocean** (~280–300 Myr) is in small enclosed remnants, like Earth's eastern Mediterranean (~230–280
   Myr). Accepted.

**Guard.** On grids much coarser than the calibration grid the erosion settings lose meaning: at n=48 the slope cap
allows 10 km steps between 200 km cells, and the run blows up. The scripts refuse n < 96. The tests use n=32–48 only
for short plumbing checks.

`scripts/simulate_plates.py` now takes `--param KEY=VALUE`, for example `--param erosion_theta=16`.

### Step 6 result (17 Oct): θ calibrated at n=217; part 2 closed with one watch-point

**Earth at the plate grid** (ETOPO 2022 30″ → n=217, no ice sheets): land 28.1%, mean land elevation 652 m, land
p10/25/50/75/90/99 = 66/166/384/838/1,461/4,680 m.

**θ calibration** (`calibrate_plate_erosion`, 2 seeds, last 50 Myr; selection rule fixed in advance):

| θ | land | p25 | p50 | p75 | p90 | worst error |
|---|---|---|---|---|---|---|
| 8 | 27.0% | 167 | 327 | 558 | 786 | 46% (p90) |
| 12 | 30.6% | 256 | 482 | 765 | 1,043 | 54% (p25) |
| 16 | 31.1% | 318 | 581 | 926 | 1,243 | 91% (p25) |
| 20 | 33.1% | 358 | 606 | 875 | 1,154 | 116% (p25) |
| 25 | 33.5% | 360 | 625 | 936 | 1,340 | 117% (p25) |

**θ = 8 adopted** (it wins by the rule). Your three seeds at θ = 8:
- land 23–32%;
- land p10/25/50 = 62–114 / 159–239 / 316–401 m: lowlands match Earth closely;
- **the high end is short:** p75 546–625 m vs 838, p90 786–883 m vs 1,461, p99 1.6–2.9 km vs 4.7 km.

No θ fits both ends: θ sets lowland erosion, and the shortfall is in high terrain. Present-day active belts are rebuilt
by the 11.53 km stage from the plate layer's uplift history, so the high end is judged on the final terrain (step 7)
before changing the plate layer's mountain building.

Other changes:
- Merging produced giant plates (182–266 Mkm² in every seed). A merge now needs the smaller plate under 35% of the
  larger, and the result under 150 Mkm².
- Ocean ages were young (mean 41–52 Myr), so subduction initiation is down to 0.004 per Myr.

### Step 7 (17 Oct): coupling built, first full run next

New: `planet_layer/coupling.py`, `scripts/run_planet_landscape.py`, `tests/test_coupling.py`. The plate layer now saves
its tectonic thickening per step (`tectonic_thickening_m`: subduction, arcs, rifting, volcanism, crust spreading,
collision volume; no erosion).

- **Parcel back-trace:** p(t−1) = Q_k(t−1) Q_k(t)ᵀ p(t) for the plate k owning the parcel at t. A plate born in the
  step falls back to the plate owning the position at t−1. Tested: parcels that stay on one plate land exactly on that
  plate's rotation (1e-9). Crust-age consistency along 10 Myr paths: median 9.9 Myr, and 96% within 3 Myr away from
  boundaries (the rest is interpolation of a 100 km age field).
- **Forcing per Myr, sampled bilinearly along the paths:**
  - load: tectonic thickening on continental crust, × 0.78 so broad isostatic uplift matches between the two models'
    crust densities (2,830 vs 2,700 kg/m³), spread conservatively over 80 km first;
  - non-load on continents: change of dynamic topography along the path;
  - non-load on ocean floor: change of the plate layer's ocean-floor elevation.
- **Initial surface:** the plate layer's elevation at −30 Myr along the paths.
- **Erodibility:** from time since the last orogeny. θ = 0 at 11.53 km (validated).
- **Why the 80 km spread:** without it, collision thickening, which arrives as one-node pulses at 46 km, drew beaded
  belts (rows of isolated peaks). Spread, the belts are continuous. Belt shape families (segments, sub-ranges) can come
  later if the final terrain needs them.
- **Small end-to-end test** (plates at 100 km, landscape at 52 km, 10 Myr): drainage integrates into large river
  systems.
- **Outputs:** present.npz, preview.png, and stats.json with hypsometry against Earth at 11.53 km and 46 km.

**Run it:**
```powershell
& '.\.venv\Scripts\python.exe' -u -m scripts.simulate_plates --seeds 1001
& '.\.venv\Scripts\python.exe' -u -m scripts.run_planet_landscape --plates latest
```
(`<run>` placeholders are not valid PowerShell; `--plates latest` picks the newest complete plate run.)

### Step 7, first full run (17 Oct): two bugs and the erosion watch-point

**Your run** (plates 20261017-141621 seed 1001 → landscape 20261017-142515, n=868, 30 Myr, 14 min, 1.22 s/step):

| | Planet (11.53 km) | Earth (same grid, no ice) |
|---|---|---|
| Land | 23.4% | 28.0% |
| Land p10/25/50/75/90/99 | 30/82/203/397/908/3,103 m | 63/163/382/839/1,479/4,707 m |
| Continental crust under water | 40% | ~28% |
| Basins > 1 Mkm² | 9 (largest 8.8, 4.4, 3.4, 3.2) | ~20 (Amazon 7.0) |

Drainage integrates into Earth-sized rivers. Everything else needed a re-evaluation:

1. **Pit trails on the ocean floor (plate-layer bug).** Subduction erosion also ran on oceanic overriding plates, capped
   only at 30% of the column per step. Repeated, it stripped oceanic crust to nothing: 9% of the ocean floor had
   crust under 5 km, 1% lay below −8.4 km, and moving trenches left rows of −10 km pits. **Fix:** on oceanic crust
   subduction erosion removes only arc crust above the normal 7 km (an intra-oceanic forearc), never the basement.
   n=96 check, 250 Myr, same seed: deepest 0.1% of the ocean floor −6.7 km; continental area 36.5% both ways;
   continental volume change −9.9% (−11.1% before; n=96 is outside the θ calibration grid, so only the comparison
   counts). New invariant in the tests: oceanic crust never thinner than 7 km.
2. **Ocean forcing did not telescope (coupling bug).** The surface channel used the parcel's continental fraction at
   the end of each interval only. Paths that touched a converting arc, a coast or a pit for one sample lost or
   gained the plate layer's whole elevation step, up to +9 km: arcs of false islands up to 2 km high. **Fix:** the
   continental treatment applies only where the parcel is continental at both ends of the interval. A label change
   (arc or basin turned continent, breakup) now moves the surface by the plate layer's own elevation change, and every
   path that is never continental at both ends of an interval ends exactly at the plate layer's present elevation.
   New test (fails on the old rule).
3. **The watch-point fired: the 11.53 km stage erodes stable land far faster than Earth.** Continental land that
   started at 200–500 m lost 175 m of elevation in 30 Myr (incl. −65 m of dynamic topography) under 0.8 km of mean
   tectonic load; 0.5–1 km land lost 350 m; 1–2 km lost 780 m. That is roughly 30–50 m/Myr of rock off plains where
   Earth's cratons and platforms lose 1–10. Gross erosion was 16 km³/yr against ~6 km³/yr of pre-human sediment
   delivery. The plate layer got the same problem at 46 km and solved it with an incision threshold (θ = 8 at
   n=217); the 11.53 km stage ran with θ = 0, which reproduces the test bed but is not tuned for 30 Myr of plains.
   **Next:** θ at 11.53 km, calibrated against Earth's denudation rates (the step 6 table), not against the
   hypsometry. The 11.53 km stage starts from the plate layer's surface; fitting the hypsometry here would tune it to
   hide the plate layer's own shortfall (its start surface already has p75 585 m and p90 1,084 m vs Earth's
   838 / 1,479). `scripts/calibrate_landscape_theta.py`: forcing built once, θ ∈ {0, 5, 10, 20}, full 30 Myr runs.
   Rule fixed in advance: worst log10 distance outside Earth's bands for 200–500 m (1–10 m/Myr), 0.5–1 km (50–100),
   1–2 km (200–500) and the land mean (40–70), over the last 10 Myr. Smallest wins; within 0.05, the smaller θ.
   After the choice: check the test bed's 368 km relief and 23 km curvature targets at that θ.

Also noted: all land drains to the sea (no endorheic basins: lakes always overflow without a water balance; that is
the deferred climate and water-balance layer).

`run_planet_landscape` now takes `--theta`, `--plates latest`, reuses a matching forcing folder, writes
denudation by elevation over the last 10 Myr to stats.json and present.npz, and draws rivers more legibly.

**Run it:**
```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests
& '.\.venv\Scripts\python.exe' -u -m scripts.simulate_plates --seeds 1001
& '.\.venv\Scripts\python.exe' -u -m scripts.calibrate_landscape_theta --plates latest
```

### Step 7 result (17 Oct): θ = 20 at 11.53 km; the remaining gap is the plate layer's margins

**Calibration** (plates 20261017-151726 seed 1001 with both fixes; `theta-20261017-152517`; denudation over the last
10 Myr, m/Myr):

| θ | 200–500 m | 0.5–1 km | 1–2 km | land mean | score | land | p25 / p50 / p75 / p90 / p99 (m) |
|---|---|---|---|---|---|---|---|
| 0 | 93 | 196 | 358 | 81 | 0.97 | 19.5% | 61 / 148 / 302 / 649 / 2,321 |
| 5 | 40 | 128 | 286 | 60 | 0.60 | 21.3% | 111 / 239 / 424 / 786 / 2,428 |
| 10 | 21 | 87 | 211 | 48 | 0.32 | 21.5% | 134 / 289 / 503 / 935 / 2,624 |
| **20** | **6.6** | **47** | **143** | **37** | **0.14** | 21.4% | 150 / 320 / 578 / 1,195 / 2,950 |
| Earth | 1–10 | 50–100 | 200–500 | 40–70 | — | 28.0% | 163 / 382 / 839 / 1,479 / 4,707 |

**θ = 20 adopted** (`run_planet_landscape.THETA`). It is the edge of the grid, but the trend shows the score rising
beyond it: the 1–2 km class is already under its band and falls with θ. Lowlands now match Earth (p10/25/50 within
5–16%). The pit trails and false island chains are gone from the ocean floor.

A side check that does not work as a reference: the same erosion law applied for one step to Earth's own ETOPO
surface. Earth's DEM at 11.5 km is full of closed depressions, which act as sediment traps, so its net rates on plains
come out negative or near zero (deposition 57 m/Myr on 200–500 m land). The long-term literature bands stay the rule.

**What still separates the planet from Earth is the plate layer, not the 11.53 km stage:**
- **Land 21% vs 28%**: 41% of the continental crust is under water (Earth ~29%). The excess sits at −200 to −1,000 m
  (7.5% of the planet's surface vs 4.1% on Earth), a drowned platform of crust ~31–36 km thick.
- **Cause: the plate layer's continents erode from their coasts inward.** At 46 km and θ = 8, interior plains
  (>300 km from the sea) lose 9.6 m/Myr, inside Earth's band, but 200–500 m land within 100 km of the coast loses
  216 m/Myr and within 100–300 km 73 m/Myr (instantaneous rates on the 20261017-141621 present surface). The coast
  is a one-cell drop from ~300 m to ocean depth, so the rim keeps retreating, and over 250 Myr this thins and drowns a
  fifth of the continental crust (continental volume −4 to −7% per run). On Earth coastal plains and shelves are
  built by deposition and the coast does not retreat like this.
- **The high end is short** (p75 578 vs 839 m, p99 2.95 vs 4.7 km), as already seen in the plate layer.

No single θ fixes the plate layer: θ = 20 at 46 km stops the coastal retreat (0–100 km: 96, 100–300 km: 16 m/Myr)
but leaves lowlands too high (calibration table above: p25 358 vs 166 m).

**Order:** step 8 (exact 23 km hand-off and meso on windows, full validation table) comes first, because it exercises
the remaining untested interface and its texture checks (23 km curvature per relief class, 368 km relief, 40 km
arrangement) may raise their own issues. The plate layer's margins are next after that, with Earth's land fraction,
the share of continental crust under water and the −200 to −1,000 m band as the checks, fixed before the change.

### Step 8, first pass (17 Oct): the hand-off, and what the 23 km input already shows

New: `scripts/planet_meso_check.py`. It builds the meso's 23.04 km input from the 11.53 km landscape exactly as
`toy_meso.planet_tiled` does (n = 434 face charts, each cell the mean of 8 × 8 sample points over fine nodes
8c..8c+7; the landscape grid is exactly twice as fine). It then cuts 18 × 18 windows (stride 9), classes them as the
meso tests do, and compares them with the meso's own held-out Earth split (`outputs/meso/dataset.npz`, val). On the
GPU it runs the adopted meso (`detail-ft/step10000-passed.pt`, detail unknown, seeds 113 / 997) on planet and Earth
windows of each class and scores them with `diagnose_meso_plains`' metrics.
Rule, fixed before the run: a class passes if meso-on-planet incision and 5.8–11.5 km power are ≥ 0.7 of real Earth;
above 2 is flagged as over-textured.

**Input-side result (no model needed; θ = 20 run, plates 20261017-151726):**

| 368 km land windows | planet | Earth (meso val) | Earth (doc reference) |
|---|---|---|---|
| relief p10 / 25 / 50 / 75 / 90 (m) | 59 / 101 / 221 / 677 / 1,229 | 138 / 263 / 502 / 1,095 / 3,080 | 133 / 237 / 456 / 872 / — |
| share with relief < 300 m | 56% | 31% | 34% |
| share with relief < 100 m | 24% | 5% | 6% |

| 23 km curvature (median abs Laplacian, m) | flat | plain/upland | hills | mountains | coast |
|---|---|---|---|---|---|
| planet | **1.1** | **14** | 137 | 260 | 36 |
| Earth (meso val) | 31 | 67 | 180 | 402 | 87 |

The plains are dead flat at 23 km: 1 m of cell-to-cell curvature against Earth's 31 m. That is 13× softer than
Earth's input blurred by 0.7 cells (14 m), the softest case the detail-aware meso was made robust to, and a quarter
of the land is in windows with less than 100 m of relief (Earth 5–6%). Hills and mountains are much closer (0.65–0.76
of Earth). Large shallow seas (−5 to −65 m) also cover the interior of the biggest continent.

The meso cannot be expected to make Earth-like plains from this; the GPU run measures by how much, and how the hills
and mountains fare. Likely causes, to be separated next (none tested yet):
- θ = 20 stops fluvial incision on low-gradient land entirely: plains no longer cut valleys at all, so they keep the
  smooth surface handed down from the 46 km plate layer (calibrated rates, but no texture);
- the plate layer's surface itself is smooth at 46 km, and 30 Myr is short for an uneroding plain to gain relief;
- nothing keeps interior relief alive (the test bed needed persistent mantle swells of 22–40 m/Myr; here dynamic
  topography is ~10 m/Myr).

**Where the smoothness comes from (same windows, power of the 23 km input per wavelength band, planet ÷ Earth meso
val, median per class; all land windows in the last column group):**

| surface | flat windows | flat curvature | flat 70–110 / 110–180 / 180–370 km | all land 70–110 / 110–180 / 180–370 km |
|---|---|---|---|---|
| plate layer, start of the window (−30 Myr) | 75% | 2.6 m | 0.01 / 0.09 / 0.14 | 0.004 / 0.02 / 0.04 |
| landscape, θ = 0 | 67% | 13.6 m | 0.28 / 0.61 / 0.47 | 0.07 / 0.14 / 0.14 |
| landscape, θ = 20 | 56% | 1.1 m | 0.00 / 0.02 / 0.06 | 0.00 / 0.02 / 0.05 |
| Earth | 31% | 31 m | 1 | 1 |

- **The plate layer hands down nearly featureless continents:** 75% of land windows flat, and 1–4% of Earth's relief
  power at 70–370 km over all land. This is the same symptom as the earlier paleo-DEM-trained relief model (whose
  1° training data were themselves smooth reconstructions) and the test bed's version 2: every generator so far has
  supplied too little continental-interior relief at 50–400 km, and the meso, which only adds detail below ~23 km,
  cannot make it up.
- **θ trades one failure for another:** θ = 0 incises plains (curvature 13.6 m, power at 110–180 km 0.6 of Earth in
  flat windows) but wears more land flat; θ = 20 keeps elevations but leaves the plate layer's smooth surface
  untouched. θ was chosen on denudation rates alone; the texture targets (368 km relief, 23 km curvature) were in
  the validation list and would have flagged θ = 20. Texture is now part of the check for any erosion setting.
- **What is missing is mechanism at 50–400 km on stable continents**, which on Earth comes from inherited crustal
  structure (old orogen roots, accreted terranes), rift shoulders and escarpments, intracratonic basins and domes,
  shorter-wavelength dynamic topography (the test bed needed persistent 22–40 m/Myr swells; here degree ≤ 12 at
  ~10 m/Myr), volcanic plateaus, and long-term fluvial dissection of uplifted surfaces.

**Next, in this order:** (1) a texture check on the plate layer itself against ETOPO on the same 46 km grid
(368 km window relief and band power at 92–370 km, pass bands fixed before any change), so the plate layer is judged
on texture as well as hypsometry; (2) add the missing relief mechanisms one at a time, each kept only if it moves
that check toward Earth without breaking hypsometry; (3) re-choose θ at 11.53 km on rates and texture together.

### Step 8, meso run and the plate-layer texture sweep (17 Oct)

**Meso on the planet** (`planet-meso-check/20261017-172251`, rule fixed before the run): generated ÷ real Earth
incision / 5.8–11.5 km power: flat 0.16 / 0.07, plain/upland 0.45 / 0.23, hills 0.77 / 0.34, mountains 0.79 / 0.34,
coast 1.31 / 0.84 (passes). The meso on Earth's own input in the same classes: 1.06 / 1.28, 0.95 / 0.71,
0.83 / 0.46, 0.74 / 0.49. Plains fail because of the input, as the 23 km check predicted.

**Texture check of the plate layer itself** (`scripts/plate_texture.py`). Earth (ETOPO on CubeSphere(217), no ice):
368 km land relief p10/25/50/75/90 = 115 / 206 / 412 / 861 / 1,533 m (matches the 133 / 237 / 456 / 872 reference
from another grid), 38% of land windows under 300 m. Scored, fixed before any change: band power at 245–368 and
368–736 km and relief p25/50/75, worst |log10|. The 92–245 km bands are reported only: at 46 km they are 2–5 cells,
the 11.53 km stage's job.

One seed (1001), 250 Myr each, run here:

| plate layer | land | 368 km relief p25 / 50 / 75 | < 300 m | power 92–147 / 147–245 / 245–368 / 368–736 | score |
|---|---|---|---|---|---|
| current (κ_normal 1000, dyn 400 m, degree ≤ 12) | 25.8% | 85 / 151 / 270 | 78% | 0.14 / 0.15 / 0.18 / 0.21 | 0.74 |
| κ_normal 100 | 21.5% | 116 / 176 / 284 | 77% | 0.10 / 0.14 / 0.20 / 0.22 | 0.69 |
| κ_normal 10 | 21.3% | 143 / 229 / 423 | 64% | 0.10 / 0.11 / 0.17 / 0.23 | 0.76 |
| κ_normal 0 | 15.8% | 143 / 239 / 481 | 61% | 0.11 / 0.14 / 0.22 / 0.30 | 0.65 |
| dyn 800 m, degree ≤ 12 | 12.0% | 142 / 237 / 548 | 61% | 0.22 / 0.23 / 0.33 / 0.37 | 0.49 |
| dyn 800 m, degree ≤ 30, flat spectrum | 16.2% | 306 / 479 / 692 | 24% | 0.10 / 0.22 / 0.38 / 0.52 | 0.42 |
| Earth | 28.1% | 206 / 412 / 861 | 38% | 1 | 0 |

- **Lateral crust flow is not the main cause.** Turning it down (new parameter `crust_kappa_normal_m2_yr`, default
  unchanged at 1000) raises window relief a little, barely changes band power, and costs land. Default kept.
- **Dynamic topography is the strongest lever,** as in the test bed (version 3): stronger and less red mantle
  topography gives Earth-like 368 km relief and 2–3× the band power. But land falls to 12–16%: half the continents
  sit in mantle lows, and the coastal retreat found in step 7 drowns them further. Dynamic-topography parameters are
  now PlateSim parameters (`dyn_std_m`, `dyn_lmax`, `dyn_slope`, `dt_knot_myr`).
- Earth's residual topography is debated: ±~1 km with power to degree ~30 (Hoggard et al. 2016), or smaller. Both
  ends are inside the sweep below.

**Sweep on your machine** (`scripts/sweep_plates.py`, 2 seeds; rule fixed now: score = the larger of the texture
score and the hypsometry score, the worst |log10| over land fraction and land p25/50/75/90; smallest wins):
dyn 400 / 600 / 800 m × (degree ≤ 12, slope 2.5 | degree ≤ 30, slope 1). Then θ at 11.53 km is re-chosen on the
winner with the revised rule (denudation and 23 km texture together, `calibrate_landscape_theta --plates sweep`).
The coastal-retreat fix for land fraction stays the next separate item.

### Step 8 results (18 Oct): sweep, θ, meso, and the water inventory

**Plate sweep** (`sweep-20261017-223508`, 2 seeds, rule fixed before the run): winner dyn 800 m, degree ≤ 30,
slope 1: texture 0.23 (old defaults 0.77), 368 km relief p25/50/75 = 352 / 500 / 724 m (Earth 206 / 412 / 861),
band power 245–368 / 368–736 km 0.74 / 0.83 of Earth. But land only 17%.

**θ on that plate run** (`theta-20261017-230010`, revised rule): θ 0 / 5 / 10 / 20 scored 1.01 / 0.84 / **0.67** /
0.87 (denudation 1.01 / 0.84 / 0.67 / 0.35; texture 0.27 / 0.23 / 0.40 / 0.87). θ = 10 chosen and now the default.
Stronger mantle motion raises erosion everywhere (200–500 m land 47 m/Myr at θ = 10), so denudation now limits.

**Meso** (`planet-meso-check/20261018-082913`): it read the θ = 20 run (the default at the time; the check now picks
the newest calibration's chosen θ). Incision / 5.8–11.5 km power vs real Earth: flat 0.33 / 0.21 (was 0.16 / 0.07),
plain/upland 0.60 / 0.23 (0.45 / 0.23), hills 0.73 / 0.30, mountains 0.69 / 0.30, coast 0.67 / 0.26.

**New class: plains-lowlands** (user's suggestion): flat windows (relief < 300 m) with core median elevation
< 200 m, reported beside the others in `planet_meso_check` and scored in the θ rule's texture term. Earth's meso
validation split has 150 such windows, curvature 27.6 m. The θ = 10 landscape has 0.27 of that, the worst class.

**Water inventory (physical bookkeeping, new default).** The planet's ocean held 1.48–1.51 × 10⁹ km³ below the
fixed 0 m datum, 11–13% more than Earth's 1.34 × 10⁹ (ETOPO on the same grid). The ocean floor is 150–300 m deeper
than Earth's (mean −4,411 to −4,571 m vs −4,276 m for Earth below −2 km; little pelagic sediment: 84–109 m mean vs
Earth's ~500 m) and a fifth of the continental crust was drowned. With the fixed datum this acted as extra water.
Now sea level is the level that holds Earth's ocean volume (`water_volume_km3` = 1.335 × 10⁹, Charette & Smith 2010):
elevations are computed on the datum, then returned relative to that level. Submarine loads use the previous
step's level. Snapshots carry `sea_level_m`, and the landscape forcing moves land by −Δ(sea level), as it moves the
ocean floor by the plate layer's own change. Test: the ocean holds exactly the inventory.

One seed here (n = 217, 250 Myr, with the water inventory):

| | land | sea level on datum | 368 km relief p25/50/75 | band power 245–368 / 368–736 | texture | land p50 / p90 / p99 | continental volume |
|---|---|---|---|---|---|---|---|
| old mantle defaults | 21.3% | −250 m | 84 / 132 / 209 | 0.14 / 0.19 | 0.85 | 302 / 747 / 3,250 | −2.0% |
| dyn 800 m, degree ≤ 30 | 21.8% | −454 m | 307 / 455 / 737 | 0.45 / 0.52 | 0.34 | 409 / 1,260 / 3,646 | −4.9% |
| Earth | 28.1% | — | 206 / 412 / 861 | 1 | 0 | 384 / 1,461 / 4,680 | ~0 |

(Mid-run land is 27–28%; it falls in the last ~100 Myr as the coasts retreat.)

**Adopted defaults:** water inventory on; dynamic topography std 800 m to degree 30, slope 1 (the high end of Earth
estimates); θ = 10. Still open: the coastal retreat (land 22% vs 28%), denudation on plains under strong mantle
motion, lowland-plain curvature (0.27), and the stair-step coastal plains seen in the meso pictures (graded shelf
fills raised by mantle uplift, to check).

### Step 8 (18 Oct, afternoon): a mass bug in the plate layer, and ranges instead of blobs

Your chain (plates 20261018-090819, θ calibration theta-20261018-124231, meso 20261018-134454): θ = 10 again; land
17.8%, land p99 only 1.7 km, 59% of continental crust under water; the meso over-textures the few flat windows
(incision 1.9×) and under-textures hills and mountains (5.8–11.5 km power 0.17–0.20).

**Bug: sediment basins became continental crust without mass.** `_sediment_basins_to_continent` gave converted nodes
the continental thickness that matched their elevation (Airy, minus dynamic topography), not their mass: ~1 km of
sediment on 7 km of oceanic crust could become ~30 km of "continental crust". Tracked over 250 Myr (n = 217, seed
1001): conversions added continental area over **12.8% of the planet's surface**, rifting 3.7%; the mean continental
thickness fell from 38.5 to 34.9 km. Nodes converted in a mantle high later sank when it moved on: the drowned
−200 to −1,000 m platforms. It also hid the true crust budget. **Now off** (`basin_to_continent` = False): thick
fills stay sediment on oceanic crust.

**The texture score was passed by round mantle swells.** Degree-30 dynamic topography raised band power and window
relief, but the landscape maps show ~700 km round blobs. New: `plate_texture.arrangement` runs
`planet_training.arrangement_metrics` on ~2,950 km windows of Earth and planet on the same grid. Earth: high-ground
share 0.10, flanked lowland 0.09, slope p90 7.3 m/km. Every earlier plate setting had 0.02–0.03 / 0.01 / 1.7–3.5.

One seed here (n = 217, 250 Myr, water inventory, dyn 800 m degree ≤ 30):

| | land | continental area / under water | volume change | land p50 / p90 / p99 | high-ground / flanked / slope p90 | texture |
|---|---|---|---|---|---|---|
| with conversion | 21.8% | 41% / 48% | −5% | 409 / 1,260 / 3,646 | 0.03 / 0.01 / 3.5 | 0.34 |
| no conversion | 22.0% | 30% / 27% | **−33%** | 592 / 1,442 / 4,225 | 0.07 / 0.05 / 6.0 | 0.29 |
| no conversion + Andes-type margins (H 2.5 km at 150 ± 60 km) | 20.9% | 27% / 24% | −34% | 721 / 2,606 / 5,003 | 0.11 / 0.07 / 9.5 | 0.30 |
| Earth | 28.1% | ~40% / ~29% | ~0 | 384 / 1,461 / 4,680 | 0.10 / 0.09 / 7.3 | 0 |

Without the conversion the continents look much more like Earth's (under-water share, ranges, slopes, the high end),
but they lose a third of their volume in 250 Myr: plate-layer erosion exports ~8.9 km³/yr to the ocean against
Earth's ~5–7 km³/yr delivered (much of which Earth stores on continental margins), and margin growth does not keep up.

**Sweep, fixed before the run** (`sweep_plates`, now scored on the largest of four: texture, hypsometry,
arrangement (high-ground share, flanked lowland, slope p90; worst |log10|), and budget (|continental volume change|,
a fraction)): dyn 500 / 800 m (degree ≤ 30, slope 1) × θ 8 / 16 at 46 km × margins default / Andes-type, 2 seeds.

**Sweep result** (`sweep-20261018-144253`): winner dyn 500 m, θ 8, Andes-type margins: score 0.42 (texture 0.28,
hypsometry 0.34, arrangement 0.42, budget 0.28). Continental volume still −20 to −39% in every combination.
**θ on it** (`theta-20261018-151244`): θ = 10 again (0.48). Landscape: land 30.4% (Earth 28.0%), land p10/25/50/75/90/99
= 227 / 440 / 720 / 1,116 / 1,751 / 4,635 m (Earth 63 / 163 / 382 / 839 / 1,479 / 4,707): the high end now matches,
the lowlands stand far too high; 6% of continental crust under water (Earth ~29%); lowland-plain windows 1% of land.
**Meso** (`planet-meso-check/20261018-160543`): flat and coast pass; plain/upland, hills, mountains 5.8–11.5 km power
0.22–0.37 of real (incision 0.66–0.83); lowland plains incision 0.52.

**Reading:** the continents are too small and too tall. With 23% of their volume lost, they cover 31% of the planet
(Earth ~40%), the water inventory puts the sea ~700 m below the datum, so everything stands high and shelves vanish.
The leak: 12–14 km³/yr of rock eroded off the continents (Earth delivers ~6–7), deposited on oceanic crust (the
coast is a one-cell cliff at 46 km, so there is almost no continental shelf to hold it), and at trenches 70% of the
sediment was subducted (2.5–6.4 km³/yr; Earth ~1.65).

One seed here (dyn 500, Andes-type margins, θ 8):

| trench sediment accreted | sediment subducted | export | volume | land | land p25/50/75/90/99 | texture | high-ground / flanked / slope p90 |
|---|---|---|---|---|---|---|---|
| 0.30 | 2.5 km³/yr | 12.8 km³/yr | −27% | 26.7% | 384 / 667 / 1,034 / 1,641 / 4,319 | 0.28 | 0.04 / 0.03 / 4.5 |
| 0.75 | 1.2 km³/yr | 14.2 km³/yr | −21% | 26.1% | 311 / 600 / 1,116 / 2,104 / 4,155 | 0.12 | 0.07 / 0.04 / 8.2 |

**New defaults:** trench sediment accretion 0.75 (subduction ~1.2 km³/yr, Earth ~1.65), dyn 500 m (degree ≤ 30,
slope 1), Andes-type margins (H 2.5 km at 150 ± 60 km). The remaining leak is the export itself, ~2× Earth's.
The sweep now also scores it: flux score = log10 distance of the continents' net rock export from Earth's 4–8 km³/yr.
**Next sweep:** erosion K factor 0.5 / 0.75 / 1 × θ 8 / 16 at 46 km, 2 seeds; then θ at 11.53 km on the winner,
then the meso check.

### Step 8 (18 Oct, evening): erosion sweep and where the planet still differs from Earth

**Sweep** (`sweep-20261018-164500`, flux score added): winner K × 0.75, θ 8 at 46 km (score 0.37; texture 0.17,
hypsometry 0.37, arrangement 0.31, budget −17%). The rock export barely responds to K (13.6 / 13.5 / 16.2 km³/yr at
K × 0.5 / 0.75 / 1): erosion is supply-limited, set by the rock uplift of the margins and belts. Now the default.
**θ at 11.53 km** (`theta-20261018-171320`): θ = 10 again. Land 28.9%, 12% of continental crust under water,
land p10/25/50/75/90/99 = 180 / 392 / 718 / 1,159 / 2,106 / 4,375 m. **Meso** (`20261018-181029`): flat and coast
pass (flat over-textured, 3× power); plain/upland, hills, mountains, lowlands 5.8–11.5 km power 0.13–0.37.

Tested here and **not adopted** (single seed, no clear gain; parameters kept, default off):
- pelagic sediment 4 m/Myr (`pelagic_m_myr`): score 0.34 vs 0.38. The ocean floor already carries ~0.6 km of
  sediment since accretion 0.75, and its median depth (−3,836 m) is now *shallower* than Earth's (−4,083 m).
- rift necking zone 150 km (`rift_width_km`; volume-conserving, tested): score 0.49, land higher.

**What is left, read off the whole-planet hypsometric curve** (percentiles of all elevations, Earth vs planet):
the planet has no continental shelf step. Earth goes from −1,990 m (60th percentile) to −54 m (70th) to +67 m
(75th); the planet from −2,487 to −316 to +145. 10% of Earth's ocean is shallower than −531 m (shelves), the
planet's −1,588 m. Its coasts are ramps rather than a planed shelf with a break, and the land above them stands
200–350 m too high (p25 392 vs 166 m), so lowland plains are nearly absent (2% of land windows). Earth's shelves
are planed at wave base during sea-level swings and built out by deposition at the shelf edge; neither
process is in the model. That is the next mechanism to add, then the continental budget (−17%, export ~2× Earth).

### Step 8, follow-up 1 (21 Oct): continental shelves in the plate layer. Built; the 3-seed decision is on your machine

**Where the planet differs at the coast** (your sweep winner `sweep-20261018-164500`, K × 0.75 θ 8, seeds 1001/1002,
present day, 46 km; Earth = ETOPO on the same grid, no ice sheets):

| | planet | Earth |
|---|---|---|
| whole-surface p60 / p65 / p70 / p75 | −2,487 / −1,749 / −316 / +145 m | −2,266 / −687 / −43 / +76 m |
| surface at −200..0 m | 1.9–2.1% | 5.0% |
| surface at 0..200 m | 3.0–3.2% | 8.3% |
| sea 0–50 km off the coast: elevation p25 / p50 | −1,028 / −259 m | −231 / −59 m |
| sea 50–100 km off: p50 / p75 | −808 / −288 m | −309 / −65 m |
| land 0–50 / 50–100 / 100–150 km inland, median | 105 / 320 / 544 m | 63 / 199 / 272 m |
| continental area / under water / mean thickness | 30–32% / 14–16% / 39.4 km | ~40% / ~29% / — |

Ocean-floor sediment is there (mean 3.7–5.9 km within 150 km of land, but median 0: piles at deltas and trenches,
bare floor between). Half the coast faces a trench within 400 km; passive coasts already have −121 m in their first
sea ring, active ones −613 m. The coast is a ramp with no planed shelf and no break.

**New: `planet_layer/shelf.py`**, called after every coarse-erosion substep in `PlateSim._erode` (plate layer only;
the 11.53 km stage and the landscape kernels are unchanged, so the flat regression is untouched).
- **Wave-base planation:** diffusion in the open sea with κ = κ_wave·exp(−depth/wave_base), depth at the edge
  midpoint (the stratigraphic-model form: Kaufman et al. 1991, Rivenæs 1992, Granjeon & Joseph 1999). Shoals above
  wave base are planed, hollows filled.
- **Shelf-edge progradation** follows from the same law: transport collapses where the water deepens, so sediment
  carried over the topset stops at the shelf edge and builds out (clinoform). Topset gradient ≈ supply/κ:
  ~50 m²/yr per metre of coast over 3·10⁴ m²/yr gives ~1.7 m/km (Earth ~1–2 m/km).
- **Coastal retreat:** land facing the open sea loses R × (cliff face down to wave base) per unit coast length into
  its sea neighbour, damped by 1 − exp(−depth/wave_base) of the water in front: a coast behind a shallow shelf stops
  retreating (wave energy is spent on the shelf; Anderson et al. 1999, Trenhaile 2000).
- **Conservation:** fluxes between node pairs only; oceanic nodes can give only their sediment, never basement.
- Parameters (`PARAMS`): `shelf_kappa_m2_yr` (0 = off, the default until the sweep decides), `shelf_wave_base_m` 50,
  `cliff_retreat_m_yr` 0.01. Cost: none measurable (670 vs 679 s per 250 Myr here).
- Tests: `tests/test_shelf.py` (volume exact, basement never eroded, shoal at −10 m planed while the same bump at
  −600 m is not, sediment at the coast fills the topset and piles at the shelf edge, cliff retreat at R × face and
  damped behind a shelf); `test_plates.py` gains a ledger test with shelves on. 26 planet-layer tests pass.

**Single-seed check here** (seed 1001, n = 217, 250 Myr; one seed only, so read as direction, not as decision:
this workspace's own control run lands far from your control on the same seed, p70 +34 m vs −316 m):

| run | land | continental area | under water | p70 | land p25 | sea above −200 m | volume | score (tex / hyp / arr) |
|---|---|---|---|---|---|---|---|---|
| control | 29.8% | 32.2% | 8.8% | +34 m | 510 m | 1.6% | −14% | 0.49 (0.37 / 0.49 / 0.33) |
| κ 3·10⁴, R 0.01 | 28.7% | 33.6% | 15.9% | −100 m | 344 m | 2.6% | −15% | 0.62 (0.62 / 0.32 / 0.48) |
| κ 10⁵, R 0.01 | 27.3% | 32.2% | 16.3% | −200 m | 340 m | 2.8% | −18% | 0.52 (0.52 / 0.31 / 0.40) |
| κ 3·10⁴, R 0.03 | 29.2% | 31.2% | 6.9% | −139 m | 499 m | 1.4% | −17% | 0.62 (0.43 / 0.48 / 0.62) |
| Earth (46 km) | 28.1% | ~40% | ~29% | −43 m | 166 m | 5.0% | ~0 | 0 |

Shelves move all three plate-level bands the right way (p70 into its band, under-water share ×1.8, p25 −30%) but on
one seed do not reach the under-water and p25 bands; strong coastal retreat (R 0.03) undoes it. The texture score
got worse on this seed (band power 245–736 km 0.43/0.68 → 0.24–0.30/0.38–0.45); whether that is the shelves or
seed noise is for the three seeds to show.

**Reading:** the shelf mechanism alone cannot supply the missing continental area. Sea level sits ~480 m below the
datum because the continents cover 32% instead of ~40%, so the ocean is wider and the fixed water inventory stands
lower; everything on land stands high. Earth's submerged continental crust is mostly stretched margin (necking
zones 50–200 km wide), which shelves then cap. So the sweep crosses shelves with the existing necking-zone option
(`rift_width_km` 150, volume-conserving; alone it scored worse in step 8, but there it had no shelf to fill it).

**Decision on your machine, rule fixed now** (`Shelf Sweep and Review.cmd`):
1. `sweep_plates --rule shelf`, seeds 1001–1003: (κ, R) ∈ {(0, –) control, (3·10⁴, 0.003), (3·10⁴, 0.01),
   (10⁵, 0.003), (10⁵, 0.01)} × `rift_width_km` ∈ {0, 150}. Only combinations whose **seed means** pass the 46 km
   bands compete: p70 −150..0 m, continental crust under water 20–35%, land p25 < 250 m. Among them the smallest
   sweep score (texture, hypsometry, arrangement, budget, flux) wins; within 0.03, the one listed first. If none
   passes, nothing is chosen and the chain stops.
2. The 11.53 km landscape on the winner for each seed (θ 10), then `scripts/shelf_review.py`, the final bands on the
   3-seed mean: p70 −150..0 m, continental crust under water 20–35%, land p25 < 250 m, lowland-plain windows ≥ 8%
   (planet_meso_check's definition, input side), guard land fraction 25–35%. It also reports the continental budget
   (volume change, rock export) for the next item.
3. `planet_meso_check` on seed 1001 for the texture classes.
Adopt the winner as the default only if step 2 passes.

`sweep_plates` now prints a shelf table (p70, under-water share, land p25, sea above −200 m, continental area) for
every run; `run_planet_landscape --plates sweep` refuses to run if the newest sweep chose nothing.

### Step 8, follow-up 1, result (21 Oct, evening): shelves rejected; a deep-sea sediment sink found and fixed

**Your sweep** (`sweep-20261021-182714`, 3 seeds, rule fixed before the run): **no combination passed the gate,
nothing chosen**, the chain stopped as designed.

| (κ, R), rift width | p70 | crust under water | land p25 | sea above −200 m | continental area | score |
|---|---|---|---|---|---|---|
| off, 0 (control) | −321 m | 13% | 411 m | 1.8% | 30.5% | **0.39** |
| off, 150 km | −446 m | 11% | 472 m | 1.7% | 29.4% | 0.50 |
| 3·10⁴, 0.003, 0 | −303 m | 11% | 463 m | 1.9% | 30.0% | 0.44 |
| 3·10⁴, 0.003, 150 | −256 m | 12% | 432 m | 2.0% | 30.4% | 0.44 |
| 3·10⁴, 0.01, 0 | −347 m | 12% | 437 m | 1.9% | 30.7% | 0.52 |
| 3·10⁴, 0.01, 150 | −423 m | 12% | 493 m | 1.8% | 29.3% | 0.47 |
| 10⁵, 0.003, 0 | −222 m | 7% | 512 m | 1.4% | 30.0% | 0.51 |
| 10⁵, 0.003, 150 | −350 m | 12% | 401 m | 1.9% | 30.7% | 0.41 |
| 10⁵, 0.01, 0 | −175 m | 8% | 492 m | 1.7% | 31.0% | 0.55 |
| 10⁵, 0.01, 150 | −369 m | 12% | 435 m | 2.1% | 29.3% | 0.42 |
| band / Earth | −150..0 / −43 | 20–35% / ~29% | < 250 / 166 | — / 5.0% | — / ~40% | |

- **Seed spread swamps the settings:** within one setting p70 ranges from −680 to −148 m (κ 3·10⁴, R 0.01).
  The single-seed gain seen in this workspace was noise (its control was an outlier).
- **Continental area explains it:** over all 30 runs, p70 rises ~95 m per +1% of continental area (R² 0.66); land
  p25 falls ~25 m per +1% (R² 0.32). After removing the area effect, shelves shift p70 by only +35 to +72 m.
- The shelf process moves 0.2–0.5 km³/yr off the coasts, against ~14 km³/yr of river export: too small to matter at
  46 km. **Not adopted** (`shelf_kappa_m2_yr` stays 0; the code and tests stay). The stretched-margin option
  (`rift_width_km` 150) did not help either (mean score worse).

**Bug: the plate layer deleted deep-sea sediment.** The landscape kernels' marine step places sediment only on
shelves (within 150 km of land, to the graded profile) and drops what finds no room: "the rest goes to the deep
sea", right for the regional window, a sink on a closed planet. Measured on the control runs: river export
13.3–15.5 km³/yr, ocean-floor sediment gain 6.0–6.6 km³/yr, so **~7–9 km³/yr vanished**, more than the continents'
net loss (4.4–7.9 km³/yr). That sediment would have ridden the ocean floor to the trenches, where 75% is accreted.
- **Fix:** `graph_landscape.marine_sediment_kept` returns the leftover at the sea-floor low where its path ended
  (`surface_step(..., keep_deep=True)`; `marine_sediment` itself is unchanged, so the regional model and the
  11.53 km stage are untouched). The plate layer spreads it over the open sea as fans / turbidites (conservative
  diffusion, Gaussian σ = `deep_sea_spread_km` 300 km). `keep_deep_sediment` = True is the new default (mass
  conservation, not a tuning choice). Tests: the surface step closes its volume with the leftover returned; the
  spread conserves volume.
- **One seed here (1001), fix on, no shelves:** continental volume **−6%** over 250 Myr (control −14 to −26%), sea
  level −229 m on the datum (control ~−495 m), budget score 0.06. But continental area 31.8%, mean continental
  thickness **43.6 km** (control ~39.5), land 31.6%, crust under water 7.6%, land p25 500 m, p70 +123 m; score 0.48
  (hypsometry 0.48, texture 0.24, arrangement 0.17, flux 0.18 with export 12.1 km³/yr). The volume comes back as
  thicker crust (trench accretion 6.9 km³/yr vs 3.7), not as area, so land stands higher.

**Continental area audit** (fix on, seed 1001, 250 Myr; area changes as a share of the planet's surface):
collision and overlap −10.1%, rifting +5.0%, arcs +1.5%. Thickness: 48% of continental area at 35–40 km, 28% at
40–45, 19% at 45–55, and only 2% below 30 km (Earth's continental crust includes wide thinned margins).
Two structural causes:
1. **Sediment on continental nodes is counted as crust** (thickness at ρ 2,830). A rifted margin at −2.7 km needs
   ~4.8× its depth of fill to reach the shelf (vs ~2.5× for sediment of 2,400 kg/m³), and once filled it *is*
   full-thickness crust: stretched margins never survive as thin, submerged continental crust.
2. **Area only shrinks at collisions and never spreads back:** lateral crust flow is confined to continental nodes
   (zero flux at the continent edge), so collapse thickens or thins in place but cannot widen a continent, and
   rifting returns only half of what collisions consume.

**Next (proposed, not built):** with the sink closed, the plate layer's remaining gap is continental *area and
thickness*, not shelves. Candidates, each to be tested against the same bands and the budget: (a) track sediment
on continental crust separately (density 2,400, its own load) so margins stay thin crust under a sediment wedge;
(b) let gravitational spreading move the continent edge (orogenic collapse outward onto the neighbouring plate
margin), with a test that it conserves volume and does not create area from nothing. First the 3-seed baseline
with the fix: **Deep Sediment Sweep and Review.cmd** (shelves off / 3·10⁴ / 10⁵ at R 0.003, same gate and final
bands).

### Step 8, follow-up 1, third pass (21 Oct, night): the 3-seed baseline with the sediment fix

**Your sweep** (`sweep-20261021-201744`, 3 seeds, fix on): nothing passed the gate, as expected.

| 3-seed mean | before the fix (control, `sweep-20261021-182714`) | fix, shelves off | fix, κ 3·10⁴ | fix, κ 10⁵ | band / Earth |
|---|---|---|---|---|---|
| continental volume, 250 Myr | −20% | **−5.5%** | −5.5% | −6.8% | ~0 |
| continental area | 30.5% | 32.8% | 32.6% | 31.3% | ~40% |
| land | 27.2% | 32.1% | 33.1% | 31.9% | 28.1% |
| crust under water | 13% | 11% | 7% | 7% | 20–35% |
| land p25 | 411 m | 453 m | 554 m | 540 m | < 250 m |
| p70 | −321 m | +154 m | +229 m | +156 m | −150..0 |
| score (budget) | 0.39 (0.20) | 0.49 (0.05) | 0.54 | 0.51 | |

- The budget closes (−5.5%, export 11.8 km³/yr, trench accretion 5.4–7.6 km³/yr, deep-sea sediment 11–18 km³/yr).
- Shelves add nothing again. With land above 30%, p70 sits on land by construction.
- The remaining gap is shape: almost all continental crust stands above the sea (89%), with no thinned, drowned
  margin. The score rose because hypsometry worsened (land 32%, p25 453 m).
- Watch-point: max pinholes 11–142 per run (earlier 7–65). Pinholes are kinematic (gaps inside plates), not touched
  by sediment; the different trajectories change the plate events. To be checked if it grows.

**What erases thin margins:** lateral crust flow at normal thickness, `crust_kappa_normal_m2_yr` = 1000, a diffusion
length of ~220 km over 50 Myr. Rifted crust (0.6 × its neighbours, ~23 km) is smoothed into the interior within
~10–20 Myr, and the area audit finds only 2% of continental area under 30 km. Step 8 already noted that cold crust of
normal thickness barely flows. κ_normal was kept at 1000 only because lowering it cost land (25.8% → 15.8% at
κ 0). That test ran while the deep-sea sink was deleting continental volume; land now overshoots (32%), so the
same direction is what the bands need. This revisits a parameter whose earlier rejection was confounded by the
bug, not a new tuning knob. Thick crust keeps its collapse flow (above 45 km, unchanged).

**Run, rule unchanged** (`Crust Flow Sweep and Review.cmd`): `crust_kappa_normal_m2_yr` ∈ {1000, 100, 10}, seeds
1001–1003, `--rule shelf` gate (p70 −150..0 m, crust under water 20–35%, land p25 < 250 m on seed means; smallest
score among the passing), then the 11.53 km landscape, `shelf_review` (adds lowland-plain windows ≥ 8%, land
25–35%) and the meso check on seed 1001. If nothing passes, the separate sediment layer on continental crust
(density 2,400, own load) is next: filled margins then stay thin crust under a wedge.

### Step 8, follow-up 1, fourth pass (21–22 Oct): crust flow is not it; breakup thickness is

**Your sweep** (`sweep-20261021-211308`, 3 seeds, deep-sea fix on): nothing passed.

| `crust_kappa_normal_m2_yr` | crust under water | land p25 | p70 | continental area | score |
|---|---|---|---|---|---|
| 1000 | 11% | 453 m | +154 m | 32.8% | 0.49 |
| 100 | 9% | 499 m | +127 m | 32.5% | 0.51 |
| 10 | 10% | 555 m | +116 m | 32.3% | 0.51 |

Lateral flow does not erase the thin margins (arrangement moved from under to over Earth: flanked lowland
0.77 → 1.6). Default unchanged. **Correction:** the κ = 1000 rows equal the previous sweep's control digit for digit,
so runs on your machine are deterministic; "seed spread" is the trajectory's sensitivity to any change (chaotic),
not run-to-run noise. Three seeds remain the rule.

**What is missing is margin width.** Earth's submerged continental crust is mostly stretched margin: crust thinned
to ~10–25 km over 100–300 km on each side before the ocean opens (hyper-extended margins break up at ~5–15 km of
crust). The plate layer declares breakup once a rift gap would be thinner than 20 km, i.e. after about one cell.

**Area audit, seed 1001 here (deep-sea fix on), breakup 20 vs 10 km, 250 Myr:**

| | 20 km (default) | 10 km | Earth |
|---|---|---|---|
| continental area added by rifting / consumed at collisions (share of the surface) | +2.7% by −50 Myr / −10.1% | +16.6% / −11.1% | ~steady |
| continental area, present | 31.8% | 50.3% | ~40% |
| continental crust under water | 7.6% | 53% | ~29% |
| crust < 30 km / 30–35 km (share of continental area) | 0% / 2% | 5% / 14% | wide thinned margins |
| land / land p25 / p70 | 31.6% / 500 m / +123 m | 24.9% / 236 m / −264 m | 28.1% / 166 m / −43 m |
| surface at −200..0 m | 1.8% | 4.0% | 5.0% |
| hypsometry score | 0.48 | 0.19 | 0 |
| continental volume | −6% | +25% (sediment on the wide drowned margins is booked as crust) | ~0 |

The lever works and Earth lies between the two.

**Two scoring fixes found on the way:**
- The flux score became NaN when the continents gained more sediment than they lost (10 km run); export ≤ 0 now
  scores as 0.1 km³/yr.
- Export is now measured as Earth's figure is: rock eroded minus what stays *on land*. Deposits on submerged
  continental crust (shelves, drowned margins) count as delivered to the sea; with wide margins they would otherwise
  count as staying on the continent. New total `continental_deposited_submerged_m3`; `shelf_review` uses the same
  definition.

**Run, rule unchanged** (`Breakup Sweep and Review.cmd`): `breakup_thick_m` ∈ {20, 17.5, 15, 12.5} km, seeds
1001–1003, `--rule shelf` gate, then the 11.53 km landscape on the winner, `shelf_review`, and the meso check on
seed 1001. Watch the budget score (continental volume growth at small breakup thickness).

### Step 8, follow-up 1, fifth pass (22 Oct): the first deep-sea fix built towers; corrected

**Your breakup sweep** (`sweep-20261021-230144`, 3 seeds): nothing passed. The trend was monotonic: at 12.5 km,
continental area 40.9%, crust under water 25% (in band), land p25 346 m, score 0.37 (best so far).

**But every run since the deep-sea fix carried an artifact.** Land band power at 92–147 km was 10–36× Earth's (≈0.2
before the fix). Single nodes carried 100–250 km of sediment and stood up to 200 km high, and continental thickness
reached 150–220 km at a few nodes (trench accretion of those piles). Cause: the first version spread the kept
sediment by diffusion over the *connected open sea* only. Sediment ending in an enclosed pocket (an inland rift
basin below −1,500 m, which seeds the sea, or a pocket cut off within the step) could not leave its node.
`sweep-20261021-201744` (fix baseline), `-211308` (crust flow) and `-230144` (breakup) are therefore **invalid**.
The shelf sweep `-182714` came before the fix and stands.

**Corrected:** `graph_landscape.fill_and_spill` places the leftover the way turbidites do. It ponds at the lowest open
level and fills the depression from the bottom up (a priority flood raising the pond to the next lowest neighbour).
Where the pond reaches a lower neighbour it spills, and the rest runs down from there and ponds again. Abyssal
plains and basins fill flat, and every m³ is placed. It runs after each coarse-erosion substep; the diffusion spread
and `deep_sea_spread_km` are gone. Tests: ponds flat and spills over (flat strip); 15 Myr at n = 96 with sediment
< 25 km and elevation < 9 km (the towers appeared within ~10 Myr).

**Seed 1001 here with the correction (250 Myr):**

| | breakup 20 km | breakup 12.5 km | band / Earth |
|---|---|---|---|
| continental area / land | 33.3% / 28.9% | 42.5% / 25.4% | ~40% / 28.1% |
| crust under water | 15.3% | 42% | 20–35% |
| p70 / land p25 | −87 m / 358 m | −211 m / 271 m | −150..0 / < 250 |
| surface at −200..0 m | 2.4% | 3.7% | 5.0% |
| max sediment / max single-node spike | 36 km / 10.9 km | 33 km / 15.0 km | (pre-fix runs: 21 / 9.8 km) |
| band power 92–147 km vs Earth | 0.27 | 0.22 | |
| score (tex / hyp / arr / budget / flux) | 0.44 (0.37 / 0.33 / 0.44 / 0.03 / 0.40) | 0.34 (0.33 / 0.23 / 0.14 / 0.11 / 0.34) | |
| export (Earth's definition) | 20.0 km³/yr | 17.5 km³/yr | 4–8 |

The breakup lever behaves as in your sweep; the best setting is probably between 12.5 and 17.5 km. A new standing
gap: with deposits on submerged continental crust counted as delivered (Earth's definition), the plate layer's
export is 17–20 km³/yr, 2–3× Earth's. That is the coarse-erosion rate itself, the next item after the margins.

**Rerun:** `Breakup Sweep and Review.cmd`, unchanged (`breakup_thick_m` ∈ {20, 17.5, 15, 12.5} km, 3 seeds, the
`--rule shelf` gate, then the 11.53 km landscape, `shelf_review` and the meso check).

### Step 8, follow-up 1, sixth pass (22 Oct, evening): the clean breakup sweep brackets the bands

**Your sweep** (`sweep-20261022-173642`, 3 seeds, corrected deep-sea sediment; band power 92–147 km back to ~0.3 of
Earth, no towers): nothing passed the gate, but the bands are bracketed.

| breakup | p70 (−150..0) | crust under water (20–35%) | land p25 (< 250) | land | continental area | sea above −200 m | volume | score |
|---|---|---|---|---|---|---|---|---|
| 20 km | −168 m | 16% | 385 m | 28.0% | 32.3% | 2.9% | −5.7% | 0.46 |
| 17.5 km | **−128 m** | **22%** | 301 m | 28.4% | 35.2% | 3.3% | −1.5% | 0.42 |
| 15 km | −219 m | **35%** (edge) | **233 m** | 25.2% | 37.5% | 4.4% | +1.4% | 0.43 |
| 12.5 km | −459 m | 57% | **207 m** | 20.1% | 44.5% | 4.7% | +13.1% | 0.36 |
| Earth | −43 m | ~29% | 166 m | 28.1% | ~40% | 5.0% | ~0 | 0 |

- Breakup thickness is the lever for continental area and submerged crust, as seed 1001 showed. The budget is
  closest to zero at 15–17.5 km (−1.5% / +1.4%).
- **The bands conflict between 15 and 17.5 km:** p70 needs ≥ ~17 km, land p25 ≤ ~15.5 km. Picking a value in
  between would be fitting the test, so it is not done.
- **Physical reading:** the wider margins exist now, but sit too deep. At 15 km the extra submerged crust lies at
  −1,000 to −200 m; the sea above −200 m (4.4%) is still short of Earth's shelves (5.0%). Planing and filling
  margins to shelf level is the shelf process (`planet_layer/shelf.py`). Its earlier rejection
  (`sweep-20261021-182714`) was with one-cell margins, where it had nothing to act on.
- Standing gap: export (Earth's definition) 18–21 km³/yr vs 4–8 (flux score 0.35–0.42), every setting.

**Run, rule unchanged** (`Margin Shelf Sweep and Review.cmd`): `breakup_thick_m` ∈ {17.5, 15} km ×
shelves {off, (κ 3·10⁴, R 0.003), (κ 10⁵, R 0.003)}, seeds 1001–1003, `--rule shelf` gate, then the 11.53 km
landscape on the winner, `shelf_review` and the meso check on seed 1001. If nothing passes again, the gap is
margin depth, not shelf transport, and the separate continental sediment layer (density 2,400) is next.

### Step 8, follow-up 1, seventh pass (22 Oct, night): shelves work on wide margins; one band short

**Your sweep** (`sweep-20261022-183426`, 3 seeds): nothing passed the gate.

| breakup, shelves | p70 (−150..0) | crust under water (20–35%) | land p25 (< 250) | sea above −200 m | land | volume | score |
|---|---|---|---|---|---|---|---|
| 17.5 km, off | **−128** | **21.6%** | 301 | 3.3% | 28.4% | −1.5% | 0.42 |
| 17.5 km, κ 3·10⁴ | **−137** | **27.8%** | 268 | 4.1% | 27.2% | +1.3% | 0.42 |
| 17.5 km, κ 10⁵ | −173 | **31.2%** | 264 | 4.0% | 26.1% | −0.7% | 0.51 |
| 15 km, off | −219 | 35.2% | **233** | 4.4% | 25.2% | +1.4% | 0.43 |
| 15 km, κ 3·10⁴ | −203 | 37.8% | **221** | 4.6% | 25.5% | +4.4% | 0.45 |
| 15 km, κ 10⁵ | −188 | 40.0% | **222** | **5.8%** | 24.3% | +5.2% | 0.38 |
| Earth | −43 | ~29% | 166 | 5.0% | 28.1% | ~0 | 0 |

- **With margins to act on, shelves do what they were built for:** at 17.5 km, κ 3·10⁴ raises the submerged share
  from 21.6% to 27.8% and lowers p25 from 301 to 268 m. Sea above −200 m reaches Earth's 5% at κ 10⁵.
- The best candidate (17.5 km, κ 3·10⁴) fails only land p25, by 18 m (seed values 213 / 311 / 282). The band stays.
- Export (Earth's definition) 18.6–19.2 km³/yr in every setting, unchanged (flux score 0.36–0.38).

**What is still set from the old regime:** the plate layer's incision threshold θ = 8 (step 6). It was chosen because it
matched Earth's lowlands (p25 167 m) while the deep-sea sink and one-cell margins were in place. θ is the main lever
on lowland height (step 6 table: p25 167 → 256 m from θ 8 to 12). It is re-chosen now under the current physics,
not tuned to the gate: lower θ lets rivers keep incising low-gradient land.

**Run, rule unchanged** (`Plate Theta Sweep and Review.cmd`): `breakup_thick_m` ∈ {17.5, 15} km × `erosion_theta`
∈ {8, 6, 4}, shelves κ 3·10⁴ R 0.003, seeds 1001–1003, `--rule shelf` gate, then the 11.53 km landscape on the
winner (`outputs\planet-landscape\theta-plates-1022-seed100X`), `shelf_review` and the meso check. If one passes, the
11.53 km stage's own θ (10, calibrated on the old plate surfaces) is the next check.

### Data added by the user (22 Oct): what each is for

- **`data/tectonicplates-master` (Bird 2003, PB2002)**: 52 plates, boundary steps with class and relative velocity,
  orogens. Earth computed from it (plate areas from the polygons, which sum to 4πR² exactly):
  - plates over 1 / over 10 Mkm²: 20 / 10; largest PA 104.6, AF 58.5, AN 58.2, NA 55.4, EU 48.6, AU 46.0,
    SA 41.8 Mkm²; median plate 0.52 Mkm² (32 microplates under 1 Mkm²);
  - boundary length: ridge (OSR + CRB) 94,800 km, subduction (SUB) 51,300 km, convergent/collision (CCB + OCB)
    40,500 km, transform (OTF + CTF) 73,900 km; length-weighted median rates: ridge opening 24.5 mm/yr, subduction
    closing 54 mm/yr.
  - Plate layer (seed 1001, breakup 20 km, here): 18 plates, 16 over 1 and 11 over 10 Mkm², largest 90.6. The large
    plates match; microplates are absent.
  - Boundary lengths, from node counts ÷ the assumed ~1.3 staircase factor: ridge ~157,000 km, trench ~160,000 km,
    collision ~11,000 km, transform ~44,000 km. **The trench length would be ~3× Earth's.** This bears on the export
    gap (18–21 km³/yr vs 4–8), because Andes-type margin thickening (H 2.5 km × convergence) acts along every trench.
    First step: a true line-length measure of the boundaries (the node-count staircase factor is an assumption), then
    a P-layout check by type against these numbers (sect. 6 already asks for Bird 2003 sizes).
- **`data/RiverATLAS_Data_v10_shp` (HydroATLAS v1.0, 13 GB)**: river reaches with upstream area, length, order,
  discharge and an endorheic flag. Use: drainage validation of the 11.53 km landscape (basin-size distribution,
  Hack's law, share of land draining to closed basins). A one-off extraction of a few fields to a compact table on
  your machine first; later, the water-balance layer.
- **`data/Terrain Ruggedness Index/tri_1KMmn_GMTEDmd.tif` (EarthEnv, Amatulli et al. 2018; m, mean 3.5)**:
  1 km ruggedness, below the meso's 2.88 km output. It could be a feature or target for the meso detail channel
  (item 4), but it is derived from GMTED and can be recomputed from the ETOPO 30″ we already use. Low priority.

### Step 8, follow-up 1, eighth pass (22 Oct, night): the gate passes; new plate baseline adopted

**Your sweep** (`sweep-20261022-194148`, 3 seeds, shelves κ 3·10⁴ R 0.003): the gate chose **breakup 17.5 km,
plate θ 6**.

| breakup, plate θ | p70 | crust under water | land p25 | sea above −200 m | land | volume | score | gate |
|---|---|---|---|---|---|---|---|---|
| 17.5 km, 8 | −137 | 27.8% | 268 | 4.1% | 27.2% | +1.3% | 0.42 | no |
| **17.5 km, 6** | **−147** | **27.5%** | **246** | 4.2% | 26.3% | −2.3% | 0.41 | **yes** |
| 17.5 km, 4 | −215 | 36.1% | 200 | 5.3% | 24.7% | −1.0% | 0.38 | no |
| 15 km, 8 / 6 / 4 | −203 / −269 / −290 | 38–42% | 217–231 | 4.6–5.1% | 23.5–25.5% | +4% | 0.41–0.50 | no |

**The 11.53 km landscape on the winner, all three seeds** (`outputs\planet-landscape\theta-plates-1022-seed100X`,
θ 10), final bands (`shelf_review`):

| | seed 1001 | 1002 | 1003 | mean | band | Earth (11.53 km) |
|---|---|---|---|---|---|---|
| whole-surface p70 | −77 m | −103 m | +19 m | **−53 m ✓** | −150..0 | −54 m |
| continental crust under water | 32.6% | 29.1% | 20.7% | **27.4% ✓** | 20–35% | ~29% |
| land p25 | 229 m | 230 m | 323 m | 261 m ✗ | < 250 | 163 m |
| lowland-plain windows | 3.8% | 4.2% | 3.1% | 3.7% ✗ | ≥ 8% | 16% (meso val) |
| land fraction (guard) | 27.2% | 27.0% | 29.5% | **27.9% ✓** | 25–35% | 28.0% |
| sea above −200 m | 5.4% | 5.2% | 3.8% | 4.8% | — | ~5% |
| plate continental volume, 250 Myr | −3.0% | −4.0% | +0.1% | −2.3% | — | ~0 |
| plate export (Earth's definition) | 17.7 | 19.6 | 21.1 km³/yr | 19.5 | — | 4–8 |

Since step 7 (18 Oct): p70 −316 → −53 m, crust under water 12% → 27%, land p25 392 → 261 m, lowland plains 2% →
3.7%. Seed 1001's land p10/25/50/75/90/99 = 93 / 229 / 491 / 906 / 1,514 / 3,087 m (Earth 63 / 163 / 382 / 839 /
1,479 / 4,707). Seed 1003 alone misses p25 and p70.

**Meso** (`planet-meso-check/20261022-211853`, seed 1001; incision / 5.8–11.5 km power vs real Earth): flat
1.17 / 1.88 (passes), plain/upland 1.08 / 0.39, hills 0.87 / 0.37, mountains 0.69 / 0.28, lowland plains
0.50 / 0.23, coast 1.05 / 0.57. Only flat passes, as before; power on the other classes is similar to or a little
above 18 Oct (0.13–0.37). The 23 km input: lowland-plain curvature 0.21 of Earth, flat 0.44, plain/upland and hills
0.77; lowland plains 3.8% of land windows (Earth's validation split 16%).

**Decision (with you, 22 Oct):** adopted as the working baseline although two final bands are open, because it
improves almost every measure over the old defaults. New `PARAMS`: `breakup_thick_m` 17,500, `erosion_theta` 6,
`shelf_kappa_m2_yr` 3·10⁴, `cliff_retreat_m_yr` 0.003 (`keep_deep_sediment` True since the earlier fix). Open:
land p25, lowland plains, the meso's texture on plain/upland/hills/mountains, and export.

**Next (the open bands):** the 11.53 km stage's θ = 10 was chosen on the old plate surfaces, and θ is what stops
rivers incising plains (step 8: θ 20 left plains at 1 m of 23 km curvature). It is re-chosen on the new baseline on
three seeds (`Landscape Theta Three Seeds.cmd`): `calibrate_landscape_theta` (θ 0 / 5 / 10 / 20) for seeds
1001–1003, then `scripts/theta_seeds_review.py`. Rule, fixed now: per θ, the mean over the seeds of the calibration's
score (the larger of denudation and 23 km texture incl. lowland plains); smallest wins, within 0.05 the smallest θ.
Then `shelf_review` on the chosen θ's three runs and the meso check on seed 1001.

**Outputs cleanup (22 Oct, with you):** 169 folders deleted (48 GB): plate snapshots of every sweep and plate run
except the winner (`sweep-20261022-194148/breakup_thick_m-17500_erosion_theta-6`, renamed 24 Oct from
`…_shelf_kappa_m2_yr-3e4_cliff_retreat_m_yr-0.003`), and all cached landscape forcing (rebuilt on demand). Every run kept results.json,
stats.json, index.html, pictures and present.npz, so all numbers here stay checkable; superseded plate runs can no
longer feed the landscape stage without re-running them. List: `outputs\test-tmp\cleanup-1022.txt`.

### Step 8, follow-up 1, ninth pass (23 Oct): landscape θ 5 on three seeds; the open band and a staircase

**Three-seed θ calibration** (`theta-20261022-214812 / 224344 / 234017`, `theta-seeds-review.json`; rule fixed
before the runs):

| θ (11.53 km) | mean score (denudation, texture) | per seed | land p25 | 23 km curvature flat / lowland plains / plain-upland / hills vs Earth |
|---|---|---|---|---|
| 0 | 0.92 (0.92, 0.34) | 0.93 / 0.86 / 0.95 | 166 m | 0.54 / 0.46 / 0.66 / 0.69 |
| **5** | **0.65** (0.65, 0.30) | 0.69 / 0.57 / 0.70 | 222 m | 0.69 / 0.50 / 0.80 / 0.73 |
| 10 | 0.67 (0.40, 0.67) | 0.66 / 0.88 / 0.47 | 261 m | 0.36 / 0.23 / 0.80 / 0.79 |
| 20 | 1.06 (0.17, 1.06) | 1.10 / 1.15 / 0.95 | 302 m | 0.11 / 0.09 / 0.34 / 0.88 |

θ 5 chosen (within 0.05 of the best, the smaller θ). Denudation at θ 5 on seed 1001: 200–500 m 49, 0.5–1 km 109,
1–2 km 276, land mean 103 m/Myr (Earth 1–10 / 50–100 / 200–500 / 40–70): low land erodes too fast; that is the
denudation score.

**Final bands at θ 5** (`shelf_review`): p70 −64 m ✓, crust under water 28.4% ✓, **land p25 222 m ✓**, land 27.6% ✓,
sea above −200 m 5.1% (Earth ~5%), **lowland-plain windows 5.7% ✗** (seeds 6.1 / 7.0 / 4.0%; band ≥ 8%; was 3.7%).
Seed 1003 is again the outlier (land 29.2%, p25 283 m).

**Meso** (`planet-meso-check/20261023-003922`, seed 1001; incision / 5.8–11.5 km power vs real Earth; θ 10 in brackets):
lowland plains **1.07 / 0.58** (0.50 / 0.23), plain/upland 1.11 / 0.55 (1.08 / 0.39), hills 0.79 / 0.31 (0.87 / 0.37),
mountains 0.75 / 0.36 (0.69 / 0.28), coast 1.99 / 1.12 (passes), **flat 2.28 / 4.92: over-textured** (1.17 / 1.88).

**THETA = 5 is the working default** (`run_planet_landscape.THETA`; same footing as the plate baseline: four of five
bands, lowland plains open).

**Why flat windows are over-textured: a staircase in emerged plains.** The first flat window (face 0, cells 0–17,
72–89; also the first lowland-plain window) shows flat treads stepping up from the coast in the 11.53 km landscape,
~2 cells (≈ 46 km) apart, and the meso sharpens each riser into a scarp. Checked so far:
- Not sediment and not a label change: the window's parcels were continental throughout the 30 Myr, sediment 0.
- It rose ~850 m in 15 Myr (from −670 m on its path at −30 Myr) by dynamic topography and sea level. The rate peaked
  at 138 m/Myr around −25 Myr, then the window sank at ~40 m/Myr.
- Globally, along the parcel paths of present land: |Δ dynamic topography| per Myr median 23, p90 60, p99 108 m/Myr
  (max 757). The field's own change at a fixed place is only median 9 / p90 28 / p99 51: most of the rate is plates
  drifting across the degree-30 field. 9% of land parcels exceed 100 m/Myr at some point (Earth: ~10–100).
- A uniform 120 m/Myr rise and 40 m/Myr fall through sea level on the flat test window (θ 5, 16 Myr) makes a smooth
  coastal ramp (0.3–0.9 m/km) and normal dissection, **no staircase**. So the steps need the real forcing's spatial
  pattern (or something else in the coupling), not the rate alone.

**Next:** reproduce the staircase on the flat window with that window's own extracted forcing (load, surface, z0),
then switch parts off (dynamic-topography gradient, sea-level change, per-Myr steps) until the steps go. Fix it at
its cause, then re-run the three-seed review: lowland plains and flat texture are the open checks. Export (19.5
km³/yr vs 4–8) and the 3× trench length stay next after that.

**Note from the user (23 Oct) on the meso verdicts:** the adopted meso was fine-tuned (from step 222,000) with blurred
inputs and an extra per-cell roughness (detail) value. With the value unknown, as in every `planet_meso_check` so far,
flat comes out too detailed and mountains too smooth. So the flat over-texture and the weak hill/mountain power are
partly the missing detail value. The staircase still matters: same setting, Earth's flat windows 1.28× vs the planet's
4.9×. The detail value (derived from the simulation or predicted by a model) moves up.

### Step 8, follow-up 1, tenth pass (23 Oct, evening): the staircase was the colour map; boundary lengths; the export loop

**The staircase is a rendering artifact, not terrain.** `planet_training/elevation_shading.CMAP` (used by every review
picture: `planet_meso_check`, `diagnose_meso_plains`, `simulate_tectonic_region`, ...) was a `LinearSegmentedColormap`
with matplotlib's default 256 levels over ±6,000 m: one colour step every 47 m. On a gentle plain (~1 m/km) that
draws contour-like bands ~47 km apart, read as flat treads and risers. Checked on the first flat window
(face 0, cells 0–17, 72–89, seed 1001, θ 5):
- re-rendered with 65,536 levels, the 11.53 km landscape and the 23 km input show smooth slopes, no steps; the window's
  elevation histogram is single-peaked (no tread levels);
- a regional re-run here (76,504 nodes within 1,600 km, the window's own extracted forcing, sphere flexure on the
  region) reproduces the global run's surface closely; switching off the load, making the surface rate uniform,
  Kd_marine 0 and θ 0 change the landscape but there was never a staircase to remove.
- **Fixed:** `CMAP` now has `N=65536` (0.18 m per level). The earlier "stair-step coastal plains seen in the meso
  pictures" (18 Oct) were most likely the same artifact. `scripts/diagnose_terrain_structure.py` and
  `review_structured_regions.py` keep their own 256-level copies (older test-bed reviews; not changed).

**The flat over-texture (meso 5.8–11.5 km power 4.9× real Earth) is real and not explained by the input.** Same
picked windows (16 per side): planet flat 23 km inputs have relief 220 vs 157 m, mean gradient 1.30 vs 0.89 m/km,
median elevation 379 vs 186 m, curvature 25.8 vs 27.0 m. Real Earth flat windows matched to the planet's (relief
≥ 180 m, elevation > 250 m, n = 78) have only 1.3× the mean power of all Earth flat windows (2.4 vs 1.8 · 10⁹),
while the meso gives 7.9 · 10⁹ on the planet's. So it is the meso's response to these inputs with the detail value
unknown, the user's point of 23 Oct. It goes with item 4 (detail value), and needs the GPU.

**Boundary lengths, measured as lines** (new `scripts/boundary_lengths.py`, numpy only; test
`tests/test_boundary_lengths.py`): each plate's mask is smoothed by 1 cell on every face chart and contoured at 0.5;
segments are typed from the two plates' relative motion across them with the plate layer's own rule. Great circles
at n = 217 measure 2πR within 0–0.5% (node counting: +24 to +38%; unsmoothed marching squares +5–9%). Present day,
winner `sweep-20261022-194148/...`:

| km | seed 1001 | 1002 | 1003 | Earth (Bird 2003) |
|---|---|---|---|---|
| ridge | 90,000 | 76,400 | 101,800 | 94,800 |
| subduction | 72,400 | 79,900 | 106,500 | 51,300 (SUB) |
| collision | 6,500 | 13,200 | 15,800 | 40,500 (CCB + OCB) |
| transform | 86,100 | 86,600 | 133,300 | 73,900 |
| total (incl. junction slivers) | 272,600 | 272,900 | 379,400 | 260,500 |
| length-median subduction rate | 56 | 63 | 52 mm/yr | 54 |

**Trench length is 1.4–2.1× Earth's, not ~3×** (the node count overstated it); ridges and transforms are Earth-like
(the node types undercount transforms: a node takes its fastest-closing neighbour, diagonals included). Collision
length is short. 1003 is the large-boundary outlier, as it is the outlier in the bands.

**Where the export (19.5 km³/yr vs 4–8) comes from**, 50 Myr blocks of the plate ledger, your three runs (km³/yr):

| block | eroded | deposited on submerged continental crust | trench sediment accreted | Andes-type margin thickening | export | dV/dt |
|---|---|---|---|---|---|---|
| −250..−200 | 15–19 | 6–10 | 1.0–1.3 | 0.5–1.1 | 13–17 | −4 to −7 |
| −50..0 | 22–32 | 7–9 | 10–16 | 5–8 | 22–30 | +0.3 to +2.6 |

1. **The coarse erosion law is ~2× too fast on Earth's own surface.** The plate layer's step (K × 0.75, θ 6,
   Kd × 4, 4 × 250 kyr) applied to ETOPO at 46 km (no ice sheets): export 9 / 20 / 24 km³/yr (last substep, still
   falling) at erodibility 0.15 / 0.5 / 1 everywhere; 200–500 m land 20–56 m/Myr (Earth 1–10). The planet's
   present surface erodes ~1.3× Earth's under the same law (mean 97 vs 68 m/Myr at erodibility 0.15). Even the first
   50 Myr, before margins and accretion ramp up, export 13–17 km³/yr.
2. **Andes-type margins add 5–8 km³/yr of new crust** (H 2.5 km × convergence along every trench with a continental
   overriding plate). Earth's arc addition at continental margins is ~1.0 km³/yr (2.7 in total with intra-oceanic
   arcs; Scholl & von Huene 2009). On Earth Andean thickening is mostly shortening (volume-conserving); here it
   is new volume.
3. **A recycling loop through the trenches.** Export ends on the ocean floor (mean 1.9–2.2 km of sediment on your
   runs, ~4× Earth's), every subducting km² carries it to a trench, and `sediment_accretion` 0.75 returns three
   quarters to the continent, where it is eroded again: gain up to 1/(1 − 0.75) = 4. Accretion and export still
   climb at 0 Myr. The 0.75 was set on 18 Oct to match Earth's *absolute* subducted volume (1.65 km³/yr) while the
   deep-sea sink was still deleting half the export; that decision is confounded by the bug, as crust flow was.
   One seed here (1001; this workspace's trajectory differs from yours): accretion 0.3 instead of 0.75 gives export
   19.6 vs 26.4 km³/yr in the last 50 Myr, accreted 4.0 vs 12.3, subducted 9.3 vs 4.1, continental volume −10% vs
   +3.5%, land 23.9 vs 26.0%, p25 227 vs 233 m. The loop is about a quarter of the excess; the margin source and the
   erosion law are the rest.

**Next (proposed, not run):** a 3-seed sweep crossing `sediment_accretion` {0.75, 0.3} with the Andes-type margin
source {2.5 km, scaled to Earth's ~1 km³/yr}, `--rule shelf` gate unchanged, export reported. The erosion law
itself (item 1) is the "P coarse-erosion calibration" of sect. 6: an Earth-surface test like the one above
(denudation by elevation class against Earth's bands) is a cleaner target than the planet's own output.

### Step 8, follow-up 1, eleventh pass (23 Oct, night): margin shortening built; one seed here says the erosion law is next

**Agreed with the user:** the margin source is worked on first, because in a supply-limited plate layer the export
follows the rock supply; the erosion law mostly sets how high land stands for a given supply.

**Built** (`planet_layer/plates.py`, `margin_shortening` = True; default False until the sweep decides):
- **Shortening, volume-conserving.** Rate s = max(0, u − `rollback_mm_yr`), u the overriding plate's speed toward
  the trench in the mantle frame. Oncken et al. (2006): Central Andean shortening tracks upper-plate drift minus slab
  rollback, which held at ~17 mm/yr; shortening 5–14 mm/yr since 30 Ma, 9–12 mm/yr today, 250–275 km since the
  Eocene. Per trench segment, s × T is taken from a foreland band (350 ± 100 km) and added to the belt (150 ± 60 km),
  conserved exactly. The rigid plate raster cannot lose area inside a plate, so the foreland is refilled by lateral
  crust flow instead of the plate shrinking (a deviation from real shortening; area is still lost at collisions).
- **Arc addition is the only new volume:** `arc_H_cont_m` 750 m × convergence, ~40 km³ per km of trench per Myr
  (Earth's ~1.0 km³/yr at continental margins, Scholl & von Huene 2009). Subduction erosion unchanged (700 m × v,
  ~38 km³/km/Myr; Earth 25–50, von Huene & Scholl 1991).
- **Earth's accreted share of trench sediment:** 0.13, length-weighted from von Huene & Scholl (1991): ~20% at
  small/medium prisms (16,300 km), ~30% at large prisms (8,200 km), none at nonaccretionary margins (19,000 km).
- Tests: `test_margin_shortening_moves_volume_into_the_belt` (exact per segment), `test_margin_shortening_run_keeps_
  the_ledger`. With the defaults the code path is the old one (the baseline stays deterministic). 32 tests pass.

**One seed here** (1001, 250 Myr; this workspace's trajectory differs from the user's), last 50 Myr in km³/yr,
present-day sweep scores (`sweep_plates.score`):

| run | export (last 50 / whole run) | accreted | subducted | shortening moves | volume change | land / p25 | high-ground / flanked / slope p90 vs Earth | score |
|---|---|---|---|---|---|---|---|---|
| baseline (new volume 2.5 km, accretion 0.75) | 26.4 / 19.0 | 12.3 | 4.1 | — | +3.5% | 26.0% / 233 m | 0.89 / 0.71 / 1.23 | 0.37 |
| baseline, accretion 0.3 | 19.6 / 15.5 | 4.0 | 9.3 | — | −10.1% | 23.9% / 227 m | 0.70 / 0.44 / 1.21 | 0.36 |
| shortening, accretion 0.75 | 16.9 / 15.8 | 6.3 | 2.1 | 1.4 | −9.9% | 23.2% / 195 m | 0.32 / 0.27 / 0.51 | 0.57 |
| shortening, accretion 0.13 | 10.8 / 11.7 | 0.8 | 5.6 | 1.2 | −15.8% | 23.1% / 208 m | 0.35 / 0.23 / 0.40 | 0.64 |

- **The supply reading holds:** without the free margin volume and the accretion loop, export falls from 26 to
  11 km³/yr.
- **But ranges collapse** (high-ground share 0.35 of Earth's) **and the continents lose 16%** of their volume. Export
  is still ~2× Earth's, so erosion wears the belts down faster than shortening builds them, and with accretion at
  Earth's share almost everything exported is subducted (5.6 km³/yr; Earth 1.0–1.5).
- Shortening moved only 1.2–1.4 km³/yr here: this trajectory's continental plates are slow (trenchward speed 0–9
  mm/yr at −190 Myr, under the 17 mm/yr rollback). On the user's runs the overriding continents move faster late
  (seed 1001: median 16–29 mm/yr at −100..0 Myr; 1002/1003 slower), so the sweep may differ.
- **Reading:** the margin change is the right budget but cannot be judged on its own. The 46 km erosion law (~2× too
  fast on Earth's own surface, tenth pass) removes the ranges and drives the volume loss, so it is next whatever the
  sweep shows. Watch-point: continental plate speeds (planet's large continents 0.5–1.2 cm/yr vs Earth's ~1–3 in the
  hotspot frame).

**Run on the user's machine** (`Margin Shortening Sweep.cmd`, rule fixed now = the existing `--rule shelf`):
`sweep_plates`, seeds 1001–1003, (margin_shortening, sediment_accretion) ∈ {(False, 0.75) baseline, (True, 0.13),
(True, 0.75)}. Only seed means passing the shelf gate compete; smallest score wins, within 0.03 the first listed
(the baseline). A new mode is adopted only if it wins. Either way, the export, volume change and arrangement of the
three are read for the erosion-law step.

**Next:** the coarse-erosion law against Earth's own surface (sect. 6 "P coarse-erosion calibration"): the plate layer's
erosion step on ETOPO at 46 km, denudation by elevation class against Earth's long-term bands (200–500 m 1–10,
0.5–1 km 50–100, 1–2 km 200–500, mean 40–70 m/Myr), erodibility from province age (Hasterok 2022) if available,
otherwise bracketed. Test and pass bands to be fixed before the change.

### Step 8, follow-up 1, twelfth pass (23 Oct, night): direction, and a correction on the erosion law

**Direction from the user (23 Oct):** move the simulation toward realism: values come from Earth measurements or from
mechanism, not from fitting the planet's output to a band. A value that only makes a band pass is to be replaced.

**Correction to the tenth pass: the Earth-surface test does not show the 46 km law is ~2× too fast.** Repeated with
measured erodibility (Bird 2003 `PB2002_orogens`, 13 zones: 16% of Earth's land and 76% of land above 2 km; 1 inside,
0.15 outside) and followed in time, the rates are a transient. On land above 2 km (7.3 Mkm²): 1,933 / 1,093 / 471 m/Myr
after 0.25 / 1 / 2 Myr; gross erosion there 14.3 → 3.4 km³/yr. Earth's surface at 46 km is not in balance with this
law, and without uplift the steep fronts are planed off within ~1 Myr. Over the first Myr, by class (m/Myr; Earth's
long-term bands in brackets): 200–500 m 23 (1–10), 0.5–1 km 94 (50–100), 1–2 km 233 (200–500), land mean 118
(40–70), export 15.6 km³/yr. The excess over the bands is in the high, steep terrain that relaxes fastest. Changing one
constant at a time (θ 12, K × 0.35, stable erodibility 0.05) brings the classes into their bands but would fit a
transient. **No change is made on this basis.**

**What a realistic test of the law needs instead:** a steady state, i.e. Earth's uplift (or erosion) rate and its
relief together. Proposed (not built): for mountain belts and cratons whose long-term erosion rates are measured
(cosmogenic ¹⁰Be basin averages, e.g. the OCTOPUS compilation; thermochronology for the large belts), run the 46 km law
to steady state under that rate on a synthetic belt and compare its 46 km relief and mean elevation with Earth's
belt on the same grid. Test set and bands to be fixed before any constant changes. The margin sweep
(`Margin Shortening Sweep.cmd`, running on the user's machine) is read first.

### Step 8, follow-up 1, thirteenth pass (23 Oct, night): the margin sweep keeps the baseline; plate speeds are Earth-like

**Your sweep** (`sweep-20261023-192601`, 3 seeds, `--rule shelf`): the rule chose the **baseline** (margin_shortening
False, accretion 0.75), which reproduced the 22 Oct winner to the digit (score 0.413, p70 −147 m, crust under water
27.5%, land p25 246 m): the code change left the default path unchanged.

| mode (3-seed means) | score | export km³/yr | volume | high-ground / flanked / slope p90 vs Earth | land | p70 | gate |
|---|---|---|---|---|---|---|---|
| baseline (new volume 2.5 km × v, accretion 0.75) | **0.41** | 19.5 | −2.3% | 0.62 / 0.48 / 0.75 | 26.3% | −147 | yes |
| shortening, accretion 0.13 (Earth) | 0.61 | 13.2 | −16.5% | 0.44 / 0.36 / 0.38 | 25.2% | −276 | no |
| shortening, accretion 0.75 | 0.54 | 17.0 | −8.9% | 0.51 / 0.49 / 0.46 | 24.1% | −276 | no |

Last 50 Myr per seed (km³/yr): baseline export 22–30, accreted 10–16, new margin volume 5–8; shortening with 0.13:
export 11–13, accreted 0.5–0.7, subducted 3.5–4.6 (Earth 1.0–1.5), shortening moved 1.9–3.3 (run means 1.1–1.4),
continental volume −14 to −18%.

**Plate speeds against Earth** (new `scripts/plate_speeds.py`; Earth in
`outputs/planet-reference/earth-plate-speeds-nnr.json`: Bird 2003 poles and polygons, no-net-rotation frame; the
Pacific frame's net rotation came out 0.65°/Myr, as in published NNR models). Area-RMS speeds, plates over 10 Mkm²:

| | mostly continental (share > 0.45) | mostly oceanic |
|---|---|---|
| Earth | 1.2, 2.0, 2.3, 2.9 cm/yr (SA, NA, EU, AF) | 1.5–7.5 (AN, SO, IN, AU, PA, NZ) |
| planet, 3 seeds at 0 / −50 / −100 Myr | 0.5–5.3 (mostly 1.5–4.5) | 1.8–11.3 |

**The kinematics are Earth-like** (continents as fast or slightly faster than Earth's; the fastest oceanic plates
~1.5× Earth's). The "slow continents" watch-point (eleventh pass) came from this workspace's own early trajectory in
the mantle frame; withdrawn. Planet net rotation 0.03–0.23°/Myr.

**Reading:**
- Shortening at Earth's rule moves 1–3 km³/yr into belts, about what Earth's Andes-type margins do (the Central Andes,
  ~250 km over 45 Myr). The baseline's 5–8 km³/yr of new margin crust built **3–5× Earth's margin belts**, and that
  is what the arrangement score has been passing on. With Earth's margins, high ground falls to 0.44–0.51 of Earth's.
  On Earth most high ground is collisional (Alpine–Himalayan); the planet's collision boundaries are 7–16k km
  (`boundary_lengths`), and collision belts are spread by the 80 km coupling smoothing. That is where to look for the
  missing high ground, not in more margin volume.
- The volume loss with Earth's accretion share (−16%) follows from the export staying 2–3× Earth's: whatever leaves the
  continents ends subducted. The export with realistic margins (11–13 km³/yr) has no large tectonic supply behind it;
  it comes from the continents themselves (low land eroding fast, and the drowned-margin deposits being re-eroded).
  That points at the erosion law on low ground (200–500 m: 20–50 m/Myr vs Earth's 1–10), which the steady-state test
  is to measure.
- Neither margin mode is adopted; `margin_shortening` and accretion 0.13 stay available. They are the realistic
  values and come back once the erosion law is set from Earth data.

**Next:** step 3 of the audit's order, the erosion law at steady state against measured erosion rates (twelfth pass),
with the test set and bands fixed first. Then the margin modes again.

### Step 8, follow-up 1, fourteenth pass (23 Oct, night): the erosion-law test against measured rates, fixed before the data

**Data:** OCTOPUS v2, CRN Global collection (Codilean et al. 2022, ESSD 14, 3695; CC BY 4.0): 4,152 basin-averaged
¹⁰Be denudation rates (mm/kyr = m/Myr), recalculated consistently, with basin polygons. Neither the cloud workspace nor
the user's VM shell can reach the server, so the user downloads the basins layer in a browser to
`data\octopus\crn_int_basins.json` (WFS GetFeature, `be10-denude:crn_int_basins`, GeoJSON, EPSG:4326).

**Refinement of the twelfth pass:** the law applied once to Earth's present surface *is* a valid test when it is compared
with the erosion rates measured in the same places. The twelfth pass compared it with elevation-class bands and
watched it decay without uplift (which it must); neither says whether the law is right.

**Test** (`scripts/erosion_law_check.py`, `Erosion Law Check.cmd`, test `tests/test_erosion_law_check.py`):
- each law at its own grid and step on ETOPO at that grid: n = 217 (plate layer: K × 0.75, Sc 0.05, Kd × 4, θ 6,
  250 kyr) and n = 868 (landscape stage: DEFAULTS, θ 5, 50 kyr); erodibility as in the planet, 1 in young orogens
  (Bird 2003 orogen zones) and 0.15 elsewhere;
- the law's gross erosion rate (deposition not subtracted, as a ¹⁰Be basin average measures the lowering of the eroding
  surface), averaged over each basin polygon (2 km sampling lattice, nearest node);
- only basins of at least one grid cell are scored (~2,100 km² at n = 217, ~133 km² at n = 868): a law predicts the
  mean over its cells, and small steep catchments are not comparable to it.

**Rule, fixed now:** per class of measured rate (< 20, 20–100, 100–500, ≥ 500 m/Myr), a class with ≥ 20 basins passes
if the median log10(law / measured) is within ±0.2 (a factor 1.6); all scored classes must pass, and at least one must
be scored. Spearman correlation reported.

**If a law fails** (also fixed now): its K and θ (and at 46 km Kd) are fitted to the measured rates on half of the
basins, split by source study so that one study's region never sits on both sides, and the other half must then pass
the same rule. Those values are then Earth-calibrated (kind M in the audit) and replace the fitted ones. Only after
that are the planet's bands and the margin modes read again.

### Step 8, follow-up 1, fifteenth pass (23 Oct, night): both erosion laws fail; the rule corrected before the fit

**Result of the fourteenth-pass check** (`outputs/erosion-law-check/217-20261023-202816`, `868-20261023-203547`;
OCTOPUS v2.3 global basins, 5,376 of them, 710 at least one cell at n = 217 and 2,046 at n = 868):

| law (current settings) | basins | median law ÷ measured | median law / measured (m/Myr) | Spearman |
|---|---|---|---|---|
| plate layer, 46 km (K × 0.75, θ 6, Kd × 4, Sc 0.05) | 710 | ×8.1 | 1,066 / 79 | 0.62 |
| landscape, 11.53 km (K 8·10⁻⁵, θ 5, Kd 50, Sc 0.05) | 2,046 | ×12.5 | 2,029 / 101 | 0.63 |

- River incision carries 95–97% of the law's erosion; the slope cap and diffusion set a floor below it. With K × 0.001
  the 11.53 km law still gives ×3.4 overall: Earth's surface at 11.53 km is steeper than Sc = 0.05 on 3.1% of land
  (p99 0.076), and the law cuts those slopes to Sc in one step. Diffusion alone gives ~2× measured in slow basins at
  both grids. **Sc = 0.05 is not Earth-consistent at 11.53 km** (audit: from "measured-ish" to contradicted).
- The fit under the fourteenth-pass rule failed (best worst-class |log10| 0.44 at n = 217, 0.77 at n = 868).

**Flaw in the fourteenth-pass rule:** classing basins by their *measured* rate biases the per-class ratio whenever
the law has scatter not tied to topography (climate and lithology, which the laws do not have): low classes come out
too high and high classes too low (regression to the mean). Diffusion alone shows it (×5.6 → ×0.2 across classes).

**Corrected rule (agreed with the user, fixed before any result under it is computed):**
- classes: quartiles of the basin's mean slope from OCTOPUS (`SLP_AVE`, from 30–90 m DEMs), a property of the terrain
  independent of the law and of the measured rate; quartiles over the scored basins of each grid;
- statistic: log10(median law / median measured) per class and overall; pass if every class and the overall value
  lie within ±0.2 (factor 1.6);
- fit: K factor, θ, Kd and Sc on the calibration half (studies split with `numpy.random.default_rng(0)` over the sorted
  STUDYIDs, first half); Sc not below Earth's own p99 slope at that grid (0.036 at n = 217, 0.076 at n = 868);
  objective = the largest |log10 ratio| over the four classes and the overall value; the other half must pass;
- if no setting passes on the validation half, the next candidate is a mechanism change backed by global ¹⁰Be
  analyses: slope exponent n > 1 (the Yuan solver is n = 1 only).

**Fifteenth pass, result** (cloud workspace; basin samples and fits in `outputs/test-tmp/claude-sync-1023` notes only):

*46 km (plate layer), 710 basins, 94 studies.* Best on the calibration half: K × 0.1, θ 100, Kd 100, Sc 0.1, worst
class 0.17 (passes); on the held-out half every class is −0.31 to −0.37 (fails), but the shape across the four slope
classes is flat to within 0.06: only the level differs. **The level differs between study halves by sampling alone:**
studies disagree with each other by 0.59 dex (std of per-study median ratio; climate and lithology are not in the law),
and two random halves of the studies differ by 0.23 dex at the median, 0.47 at p90. A ±0.2 band on one held-out half
is at the noise floor of this data set. Pooled over all basins: (K × 1.0, θ 200, Kd 200 = × 4, Sc 0.036) worst 0.13,
or (K × 0.1, θ 100, Kd 200, Sc 0.036) worst 0.14. The current law (K × 0.75, θ 6) is ×8.1 too fast. **What the data
fix is the incision threshold: θ 100–200 at 46 km instead of 6.**

*11.53 km (landscape stage), 2,046 basins, 182 studies.* Best on the calibration half: K × 0.003 (the grid's lower
edge), θ 200, Kd 25, Sc 0.076 (Earth's p99 bound), worst 0.18; held out: overall +0.00, classes 0.17 / 0.07 / 0.32 /
0.10 (fails one class). **Degenerate:** rivers barely erode and the rate comes from diffusion and the slope cap. At
11.5 km most valleys are narrower than a cell, so an instantaneous test on Earth's surface constrains the total rate
but cannot separate river incision from hillslopes; adopting this would remove valley cutting. Scaling the current
law's level only (K and Kd together, θ 5, Sc 0.076) needs ×0.03 and still has the wrong slope dependence (gentlest
quartile −0.5, others −0.04 to +0.19). **The 11.53 km law's form (n = 1 stream power with this cell size) does not
match Earth's rate–slope relation;** a mechanism change is needed there (slope exponent n > 1, the pre-agreed next
candidate, or a sub-cell hillslope–channel split), with a second Earth constraint for the river/hillslope split
(the 23 km texture against the meso data set, or RiverATLAS drainage).

**Decision proposed to the user:** the 46 km law is replaced by Earth-calibrated values only after a 3-seed plate
sweep reads the bands with them (not fitted to the bands): `Earth Erosion Law Sweep.cmd`, (K, θ, Sc) ∈ {(0.75, 6,
0.05) current, (1.0, 200, 0.036), (0.1, 100, 0.036)} × margin {(False, 0.75), (True, 0.13)}, Kd × 4 in all,
`--rule shelf`. Read: export, continental volume, high ground, the shelf bands.

### Step 8, follow-up 1, sixteenth pass (23 Oct, late): Earth's erosion rates give Earth's export; the continents stand too high

**Your sweep** (`sweep-20261023-214324`, 3 seeds, `--rule shelf`; Kd × 4 in all): the rule kept the baseline (the only
setting passing the gate).

| 46 km law, margins (3-seed means) | export km³/yr | volume | land | land p25 / p50 (m) | crust under water | p70 | high-ground / flanked / slope p90 | score |
|---|---|---|---|---|---|---|---|---|
| K × 0.75, θ 6 (current), old margins | 19.5 | −2.3% | 26.3% | 246 / 573 | 28% | −147 | 0.62 / 0.48 / 0.75 | 0.41 |
| K × 0.75, θ 6, realistic margins | 13.2 | −16.5% | 25.2% | 247 / 486 | 27% | −276 | 0.44 / 0.36 / 0.38 | 0.61 |
| K × 1, θ 200, Sc 0.036 (Earth), old margins | 9.9 | +2.1% | 31.8% | 1,047 / 2,000 | 5% | +359 | 0.82 / 0.80 / 1.41 | 0.80 |
| **K × 1, θ 200, Sc 0.036 (Earth), realistic margins** | **4.8** | −8.8% | 29.9% | 871 / 1,415 | 8% | −100 | 0.44 / 0.47 / 0.71 | 0.72 |
| K × 0.1, θ 100, Sc 0.036 (Earth), old margins | 8.1 | +1.1% | 31.2% | 1,116 / 1,878 | 4% | +201 | 0.58 / 0.41 / 0.87 | 0.82 |
| **K × 0.1, θ 100, Sc 0.036 (Earth), realistic margins** | **5.0** | −10.0% | 28.9% | 963 / 1,515 | 7% | −578 | 0.51 / 0.44 / 0.72 | 0.76 |
| Earth | 4–8 | ~0 | 28.1% | 166 / 384 | ~29% | −43 | 1 | 0 |

- **With Earth's erosion law and Earth's margin budget the export is Earth's** (4.8–5.0 km³/yr, flux score 0) and the
  continental volume loss halves (−9 to −10%, from −16.5%). Mountains hold up better with the slower erosion
  (texture 0.42 / 0.27; slope p90 0.71–0.72 vs 0.38).
- **But the land stands 1 km too high** (median 1.4–1.5 km vs 384 m) and almost no continental crust is drowned
  (7–8% vs ~29%). The old θ 6 had been eroding the lowlands down to Earth's p25, hiding this.
- **Why** (seed 1001, present day): continental crust p25 / p50 / p75 / p90 = 37 / 39 / 45 / 50 km (Earth erosion,
  realistic margins) and 38 / 41 / 51 / 63 km (Earth erosion, old margins), vs 36 / 38 / 40 / 47 km at θ 6;
  continental area 31–35% of the surface (Earth ~40%); with the fixed water inventory the smaller continents put sea
  level 450–630 m *below* the datum, lifting all land further. This is the structural cause noted on 21 Oct: collisions
  consume continental area (−10% of the surface over 250 Myr) and pile the volume up, and thick crust never spreads
  back past the continent's edge (crust flow stops at the edge; rifting returns only 5%).

**Next (to agree):** first the test, against Earth data: the continental crust thickness distribution (CRUST1.0,
Laske et al. 2013, 1°) and continental area (~40%); then the mechanism that is missing on the planet: thick crust
spreading outward past the continent edge (orogenic collapse, extensional margins), volume-conserving, gaining area.

### Step 8, follow-up 1, seventeenth pass (23 Oct, late): the continental crust against CRUST1.0

**Earth** (CRUST1.0, Laske et al. 2013, 1°, in `data/crust1`; summary `data/crust1/crust1-continental-summary.json`;
continental = crust types 01–25, shelf, slope, thinned and transitional continental crust; total thickness incl.
sediment, as the plate layer counts it; cos-latitude weights): continental area **38.3%** of the surface; thickness
p10 / 25 / 50 / 75 / 90 / 99 = 26.5 / 31.7 / 37.0 / 41.0 / 44.5 / 64 km; **19%** thinner than 30 km (shelves 9.7%,
slopes 6.4%, extended crust 3.8%), **8.7%** thicker than 45 km, 1.8% thicker than 55 km.

**Planet** (`scripts/crust_review.py`, 3-seed means, `sweep-20261023-214324`): area 30–35%; thinner than 30 km 0.6–2%;
thicker than 45 km 21% (current baseline) to 32–52% (Earth erosion law); median 38.8–46 km. Every setting fails.
The planet lacks Earth's thin crust almost entirely and has 2.5–6× its thick crust.

**Bands, fixed now** (`crust_review.BANDS`): area 0.35–0.42, thinner than 30 km 0.12–0.26, thicker than 45 km ≤ 0.13,
median 34–40 km.

**Two mechanisms, both from Earth:**
1. **Rifted margins at Earth's values** (existing code): breakup when the stretched crust would be thinner than
   10 km (Earth's hyper-extended margins break up at ~5–15 km; the fitted 17.5 km goes), necking zone 100 km
   (`rift_width_km`; Earth 50–200 km). Middle of Earth's ranges, not chosen by a sweep.
   `Rifted Margins Sweep.cmd`: Earth erosion law (K × 1, θ 200, Sc 0.036) + realistic margins, (breakup, width) ∈
   {(17.5 km, 0), (10 km, 100 km)}, 3 seeds, then `crust_review --latest`.
2. **Collapse of thick crust past the continent edge** (to build): crust flow currently stops at the coastline, so
   thick belts can only thicken or spread inward. On Earth, crust thicker than ~45 km collapses and extends, adding
   continental area (Basin and Range, Aegean). Proposed: the collapse part of the crust flow (above
   `collapse_thick_m`) may cross the continent edge into oceanic nodes of the same plate that are not on a plate
   boundary, volume-conserving; such a node becomes continental crust through the existing arc rule (≥ 20 km).
   Normal-thickness crust does not flow across the edge (cold crust is strong).

**Seventeenth pass, results.** `Rifted Margins Sweep.cmd` (`sweep-20261023-225025`; Earth erosion law K × 1, θ 200,
Sc 0.036; realistic margins; 3 seeds), continental crust (`crust_review`) and sweep scores:

| rifted margins | area | < 30 km | > 45 km | > 55 km | median | volume | crust drowned | land p25 / p50 | export | score |
|---|---|---|---|---|---|---|---|---|---|---|
| fitted (17.5 km, no necking) | 0.311 | 0.016 | 0.352 | 0.119 | 40.4 km | −9.0% | 8% | 911 / 1,512 m | 4.8 | 0.73 |
| **Earth (10 km, 100 km)** | 0.483 | **0.226** | 0.194 | 0.074 | **36.9 km** | +22.7% | 48% | 502 / 999 m | 3.7 | 0.60 |
| Earth (CRUST1.0) | 0.383 | 0.19 | 0.087 | 0.018 | 37.0 km | ~0 | ~29% | 166 / 384 m | 4–8 | |

Earth's breakup values produce Earth's thin crust (p10 27.9 km vs 26.5) and median for the first time, and halve the
thick tail. They overshoot continental area (48%), drowned crust (48%) and volume (+23%: sediment on the wide submerged
margins is booked as crust of density 2,830; the 21 Oct candidate "sediment as its own layer" addresses this, and also
lowers land that stands on sediment-thickened crust).

**Edge collapse built** (`collapse_edge`, default False; `edge_spread` kernel inside the crust-flow substeps; tests
`test_edge_spread_moves_only_collapse_excess_across_the_coast`, `test_edge_collapse_run_keeps_the_ledger`; 35
planet-layer tests pass). One seed here (Earth law + realistic margins, fitted rifting): area 31.2 → 34.0%, > 55 km
10.0 → 3.6%, > 45 km 31 → 27%, volume −9.8 → −4.3%, texture 0.44 → 0.16; it moved 0.54 km³/yr across coastlines.
`Edge Collapse Sweep.cmd`: the Earth rifted-margin setting above plus `collapse_edge` True, 3 seeds, then `crust_review`.

**Next candidates, in order:** sediment as its own layer on continental crust (density 2,400, its own load; volume
budget and land height), then the area overshoot of Earth-value rifting (margin width and how much stretched crust
counts as continent; CRUST1.0's continental area includes shelf and slope).

### Step 8, follow-up 1, eighteenth pass (24 Oct): edge collapse works where it reaches, but the thick crust is inland

**Your sweep** (`Edge Collapse Sweep.cmd`, `sweep-20261023-231817`; Earth erosion law K × 1, θ 200, Sc 0.036;
realistic margins; Earth rifting 10 km / 100 km; `collapse_edge` True; 3 seeds; `crust_review` re-run from the
saved snapshots on 24 Oct):

| Earth rifting | area | < 30 km | > 45 km | > 55 km | p10 / median / p90 | volume | export | land | score |
|---|---|---|---|---|---|---|---|---|---|
| no edge collapse (`sweep-20261023-225025`) | 0.483 | 0.226 | 0.194 | 0.074 | 27.9 / 36.9 / 52.3 km | +22.7% | 3.7 | 27.2% | 0.60 |
| **edge collapse** | 0.489 | 0.232 | 0.204 | 0.080 | 27.4 / 36.7 / 53.2 km | +23.8% | 4.0 | 26.6% | 0.57 |
| Earth (CRUST1.0) | 0.383 | 0.19 | 0.087 | 0.018 | 26.5 / 37.0 / 44.5 km | ~0 | 4–8 | 28.1% | |

- **The collapse ran** (0.85–1.17 km³/yr across coastlines, 2.1–2.9 × 10⁸ km³ over 250 Myr; more than the one-seed
  test here) **but changed nothing that matters**: every crust number within seed noise, still failing area and > 45 km.
- **Why: the thick crust is not at the coast.** Share of the > 45 km crust within 3 nodes (~70 km) of a coastline:
  0.15–0.26 without edge collapse, 0.10–0.12 with it (fitted rifting: 0.10–0.11). Edge collapse empties the coastal
  part, but 85–90% of the thick crust sits in continental interiors (collision belts), where only the interior crust
  flow acts. The thick tail is an interior problem: how much collisions thicken and how fast thick belts relax inland.
- **Where the +23% volume comes from** (ledger, `erosion_totals_km3`, 3 seeds): with Earth rifting, deposits on
  *submerged* continental crust are 2.5–3.1 × 10⁹ km³ against 1.0–1.2 × 10⁹ km³ eroded off the continents (fitted
  rifting: 0.44–0.49 × 10⁹ vs 1.2–1.6 × 10⁹). Deposited − eroded ≈ +1.9 × 10⁹ km³ ≈ the volume gain. So the gain is
  sediment trapped on the wide submerged margins and booked as crust (density 2,830), and more is deposited there than
  the continents lose: the extra must come from the ocean side (deep-sea fill and spill, or ocean-floor diffusion onto
  continental nodes). Check this ledger before building the sediment layer; the snapshot field `sediment_m` on
  continental nodes is ~0–90 m, so the margin sediment is not in it.

**Decision:** `collapse_edge` stays available (default False); not adopted (no measurable gain, slightly worse score).

**Next candidates, in order (to agree):**
1. Ledger check: where the sediment on submerged continental crust comes from (one seed, cheap; a test that
   continental deposits never exceed continental erosion + volcanic + accreted input unless a named ocean source is
   booked).
2. Sediment as its own layer on continental crust (density 2,400), as planned: fixes the +23% volume and lowers land
   on sediment-thickened crust.
3. The thick interior tail (> 45 km 0.19–0.20 vs 0.087; > 55 km 4×): collision thickening and interior collapse,
   tested against CRUST1.0's thick tail and Earth's collision-belt widths; joins "missing high ground" (item 2 of the
   handoff), since both are collision belts.
4. Area overshoot (48% vs 38%).

**Eighteenth pass, ledger check (24 Oct, here; seed 1001, Earth erosion law + realistic margins; cloud trajectory).**
`scripts/ledger_probe.py` (an instrumented copy of `_erode`), `scripts/subaerial_ocean_probe.py`,
`scripts/thinned_strip_probe.py`. 250 Myr means, km³/yr:

| rifting | ocean floor cut below its sediment, never booked | of it on nodes above sea level | deposited on submerged continental crust (of it at z0 ≤ −1 km) | continental net in `_erode` |
|---|---|---|---|---|
| fitted (17.5 km, 0) | 0.29 | 0.26 | 1.69 (1.04) | −3.12 |
| Earth (10 km, 100 km) | **8.02** | **7.98** | **9.44 (8.35)** | **+5.76** |

- `surface_step` is closed (sum of dz·a = 0). The leak is in `_erode`'s clamp `dsed = max(dz, −sed)`: an oceanic node
  eroded below its sediment loses the rest from the surface but not from its crust. 2.0 × 10⁹ km³ over 250 Myr, the
  same size as the continental volume gain (+1.4 × 10⁹ km³ here; ~1.9 × 10⁹ in your sweep's ledger).
- **Which nodes:** ordinary young ocean floor (crust anomaly 0, sediment ~0, mostly < 10 Myr), all within 200 km of a
  continent, standing at +2 to +12 km (median). Their uplift is the flexural share of *neighbouring continental
  loads* (median 5.5–9 km); without it they sit at −2.4 to −3 km (ridge depth), and only 1–5% would be above sea level.
- **Why:** `elevation()` loads continental crust by `rho_crust × (thick − 38 km)`. A narrow strip thinned to 10–20 km
  is a large negative load; the Te = 30 km plate (flexural parameter ~68 km) spreads its rebound onto the ocean beside
  it. At −240 Myr: continental crust < 20 km sits 2.5 km *below* its own Airy depth (median; p10 5.7 km below), young
  ocean within 100 km of a coast stands 4.4 km above its own depth (p90 11.6 km), 65% of it above sea level; beyond
  100 km it is normal. The over-deep strips are the sink for the debris of the lifted ocean floor: a closed artifact
  loop that builds the +23% volume and much of the 48% drowned crust. Fitted rifting has few thin strips, so 0.3 km³/yr.
- On Earth, crust thinned by stretching is in local isostatic equilibrium to first order (McKenzie 1978's stretching
  model assumes local Airy isostasy; young rifted and oceanic lithosphere is weak, Te of a few km, Watts 2001), and
  ocean floor next to passive margins is at ridge or subsidence depth, not above sea level (exceptions: plume-fed
  Iceland, Afar).

**Earth's elevation of continental crust against its thickness** (`scripts/earth_crust_elevation.py`, CRUST1.0 total
thickness and type × ETOPO 2022 on the n = 217 graph, ice excluded; `outputs/planet-reference/earth-crust-elevation-n217.json`),
node medians, m:

| crust (km) | area | Earth p25 / p50 / p75 | CRUST1.0 sediment (km) | planet local Airy (all crust) | planet Airy, sediment as a 2,400 layer |
|---|---|---|---|---|---|
| 15–20 | 0.7% | −1,944 / −1,112 / −432 | 1.1 | −3,585 | −3,327 |
| 20–25 | 1.5% | −1,306 / −635 / −110 | 1.5 | −2,690 | −2,341 |
| 25–30 | 3.6% | −286 / −72 / −6 | 1.9 | −1,475 | −1,117 |
| 30–35 | 6.1% | −26 / 98 / 381 | 1.0 | −748 | −412 |
| 35–40 | 10.2% | 115 / 321 / 647 | 0.5 | 239 | 373 |
| 40–45 | 7.9% | 167 / 359 / 706 | 0.8 | 814 | 1,013 |
| 45–55 | 2.8% | 354 / 885 / 1,593 | 0.2 | 1,582 | 1,746 |

Even locally (no flexure artifact) the planet's Airy relation (38 km at +300 m, 2,830 against 3,300) puts thinned
crust 1.0–2.0 km deeper than Earth and 45–55 km crust 0.7 km higher; a sediment layer explains only ~0.35 km of it.
Caveat: CRUST1.0 thickness is uncertain by several km, and binning on a noisy thickness flattens the apparent slope
(regression dilution), so the slope is not to be fitted from this table; the thin end (shelves near sea level) is
robust.

**Proposed (to agree), tests first:**
1. **Ledger closes** (bookkeeping, no choice): ocean floor eroded below its sediment is charged to the oceanic crust.
   Test: in every `_erode`, Σ over nodes of (crust + sediment volume change) = 0 to round-off.
2. **Thinned crust compensated locally**: the negative part of the continental anomaly (thinning) enters the
   elevation as local Airy (water-loaded), not as a flexural load; thickening (positive anomaly) and erosional
   unloading of belts stay flexural. Mechanism: McKenzie 1978, weak young lithosphere. Tests (fixed now): young ocean
   (< 20 Myr, anomaly < 1 km, sediment < 500 m) within 100 km of a coast: share above sea level ≤ 1%, median offset
   from its own depth within ±300 m; continental crust < 30 km: median offset from its own Airy depth within ±300 m;
   ledger leak (`ledger_probe` "unbooked") ≤ 0.1 km³/yr.
3. Then re-read the crust bands (`crust_review`) and the thickness–elevation table above against the planet before
   touching densities or the reference column; the sediment layer after that.

**Eighteenth pass, the two fixes (24 Oct; agreed, built, one seed here).**
1. `ocean_crust_erosion` (default **True**): in `_erode`, ocean floor eroded below its sediment loses the rest from
   its crust (`ocean_crust_eroded_m3`, `ocean_crust_eroded_subaerial_m3` in the erosion totals). Crust + sediment
   change sums to 0 in every call. The shelf test's invariant "oceanic crust never thinner than `ocean_thick_m`" is
   now "thinner only by the booked ocean-crust erosion".
2. `thinning_local_isostasy` (default False until your sweep): in `elevation()`, the negative part of the continental
   anomaly stands at its local Airy depth (water-loaded) and is no flexural load; thickening and the erosion of thick
   belts stay flexural. Identical to before for wide uniform thinning.
Tests (`tests/test_plates.py`, 18 pass; 36 planet-layer tests pass here without `test_erosion_law_check`):
`test_ocean_floor_eroded_below_its_sediment_loses_crust` (ocean patch lifted by dynamic topography; the old clamp
created volume), `test_thinned_strip_stands_at_its_own_depth_and_does_not_lift_the_ocean` (n = 217, 100 km strip of
12 km crust along every coast, young ocean in front: old elevation lifts that ocean by a median 5.4 km, 89% above sea
level, strip 10 km below its own depth; new: ocean +22 m, 0% above sea level, strip +6 m).
Side note: at n = 48 (plumbing tests) `SphereFlexure` deflects a uniform load 1.83× Airy; at n = 217 it gives Airy
exactly (857.6 m per km of crust) and conserves a point load's compensation. Plumbing tests only; no effect on runs.
Not changed yet: the landscape stage (`coupling.py`) flexes the plate layer's `tectonic_thickening_m`, where rift
thinning enters as a negative load in the same way; check after the plate layer is settled.

`scripts/elevation_check.py` (new): the tests above on the last 40 Myr of snapshots, plus crust bands, hypsometry,
shelf metrics, volume and export. One seed here (1001, cloud trajectory; Earth law, realistic margins, Earth rifting
unless noted):

| | old code | ledger fix only | **both fixes** | fitted rifting + both |
|---|---|---|---|---|
| young ocean ≤ 100 km from coast above sea level / offset | 44% / +2.9 km | 27% / +3.4 km | **0% / −7 m** | 0% / −36 m |
| thin crust offset from own Airy | −1,189 m | −665 m | **−7 m** | −108 m |
| ocean crust eroded (above sea level), km³/yr | (leak) | 0.70 | 0.01 | 0.01 |
| continental area / < 30 km / > 45 km / median | 0.414 / 0.151 / 0.262 / 37.3 | 0.356 / 0.118 / 0.249 / 39.1 | **0.344 / 0.128 / 0.245 / 38.1** | 0.332 / 0.018 / 0.254 / 39.0 |
| volume change | +9.7% | −4.2% | −8.3% | −8.5% |
| deposited on submerged continental crust, km³/yr | 7.4 | 2.8 | 1.1 | 0.7 |
| export km³/yr | 4.3 | 4.0 | 3.8 | 3.2 |
| crust drowned (Earth ~29%) / whole p70 | 35% / −144 m | 16% / +191 m | **23% / −429 m** | 6% / +204 m |
| land / p25 / p50 (Earth 28.1% / 166 / 384 m) | 28.1% / 551 / 1,120 | 30.6% / 863 / 1,419 | 26.8% / 638 / 1,193 | 31.3% / 802 / 1,298 |

- With both fixes the tests pass and the crust passes 2 of 4 bands (thin share, median), area just below (0.344 vs
  0.35); drowned crust enters the shelf band (0.20–0.35). The area overshoot of Earth rifting (48%) was the leak.
- Left: the thick tail (> 45 km 0.245 vs ≤ 0.13; inland collision belts) and land 0.5–0.8 km too high; volume −8%.
- The ledger fix alone is not a candidate (the lifted ocean remains); it is in the sweep to separate the effects.

`Thinning Isostasy Sweep.cmd`: 3 seeds, both settings (local isostasy off / on) with Earth law, realistic margins
and Earth rifting, then `elevation_check --latest` and `crust_review --latest`. Decision rule: adopt
`thinning_local_isostasy` if its 3-seed means pass the elevation tests; then the crust bands and the shelf bands
are read as they come out (no band tuning). Next after that, in order: the thick inland tail with the missing high
ground (collision thickening, interior collapse; CRUST1.0's thick tail), Earth's thickness–elevation table above,
then the sediment layer.

### Step 8, follow-up 1, nineteenth pass (24 Oct): local isostasy adopted; rifting now makes too little continent

**Correction first:** `elevation_check.py` sorted snapshot names as text (`t+0000` first, `t-0249` last), so its
"last 40 Myr" were the first 40 Myr. Fixed (sorted by time). Re-read on the cloud runs (last 40 Myr): old code 31% of
young coastal ocean above sea level, +1.8 km; both fixes 0%, −14 m, thin crust +6 m. Conclusions unchanged.

**Your sweep** (`Thinning Isostasy Sweep.cmd`, `sweep-20261024-114002`; Earth law, realistic margins, Earth rifting
10 km / 100 km, ledger fix on in both; 3 seeds; `elevation_check` and `crust_review` here from your snapshots):

| `thinning_local_isostasy` | young coastal ocean above sea / offset | area | < 30 km | > 45 km | median | volume | drowned | land p25 / p50 | export | score (tex / hyp / arr) |
|---|---|---|---|---|---|---|---|---|---|---|
| False | 31–33% / +3.3 to +4.0 km ✗ | 0.372 | 0.185 | 0.233 | 38.2 | −2.3% | 27% | 730 / 1,343 | 4.5 | 0.66 (0.45 / 0.64 / 0.44) |
| **True** | **0% / −30 to −33 m ✓** | 0.325 | 0.089 | 0.294 | 39.7 | −9.4% | 16% | 829 / 1,465 | 4.5 | 0.70 (0.33 / 0.70 / 0.42) |
| Earth | ~0 | 0.383 | 0.19 | 0.087 | 37.0 | ~0 | ~29% | 166 / 384 | 4–8 | |

Neither passes the shelf bands (nothing chosen). **Decision (rule fixed before the sweep): `thinning_local_isostasy`
adopted, default True.** Without it the ocean floor beside rifts stands kilometres above sea level on every seed.

**Why the crust got worse** (`scripts/area_ledger_probe.py`, cloud seed 1003, continental area by mechanism over 250
Myr, fractions of the surface): collision −0.102 / −0.104 (same), subduction 0, arcs 0; **rifting +0.092 without,
+0.063 with** the local isostasy: the 0.029 difference is the whole area gap (end 0.361 vs 0.333). Without it, the
3.3–4.5 km³/yr deposited on submerged continental crust (lifted ocean floor and its sediment; 1.3 km³/yr with it)
rethickened the thinned margins above the 16.7 km the rift rule needs (0.6 × neighbours ≥ 10 km), so rifting kept
adding continent. The old area was held up by the artifact.

**The rift rule is the next suspect.** `rift_thinning` 0.6 per gap node is per grid cell, not from Earth: from 38 km
a rifted margin gets 22.8, then 13.7 km, then breaks up, i.e. ~2 nodes (~90 km) of stretched crust per side plus the
100 km necking band. Earth's extended margins are wider: crust thins from ~35 to < 10 km over ~100–300 km (e.g.
Iberia–Newfoundland, Péron-Pinvidic & Manatschal 2009), and CRUST1.0 counts 19% of continental crust below 30 km.

**Next candidates (to agree), tests first:**
1. **Earth's margin width, measured** (CRUST1.0, cheap): area-weighted distance over which continental crust goes
   from ≥ 30 km to oceanic, per margin, as the test for any rift-rule change; then a rift rule stated in distance (a
   thinning profile from Earth margins), not per node. Bands already fixed: area 0.35–0.42, < 30 km 0.12–0.26.
2. The thick inland tail with the missing high ground (> 45 km 0.29 vs 0.087; collision belts).
3. Earth's thickness–elevation table (planet Airy 1–2 km too deep for thin crust; land p25/p50 2–5× Earth's).
4. Syn-rift magmatism (volcanic margins add igneous crust) only if 1 leaves the area short.

**Nineteenth pass, Earth's margin width (24 Oct).** `scripts/margin_width.py` (CRUST1.0 type and total thickness on
the n = 217 graph; passive edge = continental node beside ocean floor, > 300 km from a plate boundary: Bird 2003
PB2002 on Earth, `boundary_type` on the planet; Earth resolved to ~50–110 km), `outputs/planet-reference/earth-margin-width-n217.json`.

| | thin crust (< 30 km) per km of passive edge | abrupt edges (no crust < 30 km) | median T, 0–50 / 50–100 / 100–150 / 200–300 / 500–1000 km from the edge |
|---|---|---|---|
| **Earth** | **140 km** | **17%** | **26.4 / 30.0 / 32.0 / 35.0 / 38.0 km** |
| planet, local isostasy (your 3 seeds) | 30–80 km | 48–78% | 30.6–33.9 / 31.7–34.6 / 32.0–34.9 / 33.5–37.6 / 38.6–40.6 |
| planet, without (your 3 seeds) | 81–103 km | 27–47% | 27.3–30.3 / 29.8–31.3 / 30.4–31.7 / 32.5–34.0 / 37.2–39.3 |
| cloud seed 1001, local isostasy, κ_normal 1000 / 30 | 87 / 69 km | 48% / 50% | 30.3 / 30.4 at the edge |

**Margin bands, fixed now** (`margin_width.BANDS`): thin crust per km of passive edge 100–200 km; abrupt edges
≤ 30%; median T 0–50 km from the edge 24–29 km; 100–150 km from the edge 30–34 km.

- Beyond ~100 km inland the planet is close to Earth; the thin edge is what is missing.
- **Crust flow is not it** (second time, now on margin structure): κ_normal 30 instead of 1000 (a 10 km lower-crust
  channel at ~10²¹ Pa s; κ 1000 is ~4 × 10¹⁹, hot crust) leaves 50% abrupt edges and 30.4 km at the edge.
- **Margins are born right and thicken with age** (`scripts/margin_age_probe.py`, passive edges by the age of the
  ocean floor beside them; your 3 seeds with local isostasy and cloud seed 1001): beside ocean < 25 Myr old the edge
  is 20–27 km and 15–24% abrupt (Earth-like); beside 100–200 Myr ocean it is 29–39 km and 47–82% abrupt.
- **Likely cause: sediment booked as crust.** Deposits on continental crust become crust of density 2,830; filling 1 m
  of water on a submerged margin then takes (ρm−ρw)/(ρm−ρc) = 4.8 m of material, against 2.5 m for sediment of 2,400,
  so margins trap ~2× Earth's volume and thicken ~2× as fast (~1.3 km³/yr lands there with local isostasy, several
  km per 100 Myr). Earth's margin sediment is in CRUST1.0's total thickness too, but lighter and with the
  crystalline crust staying thin.

**Next (to agree): sediment as its own layer on continental crust** (density 2,400; deposits go to `sed`, erosion
takes sediment first, then crust; sediment is a flexural load like ocean sediment; the ledger counts crust and
sediment separately; `crust_review` and `margin_width` compare crust + sediment, as CRUST1.0 total thickness does).
Tests fixed before building: crust + sediment ledger closes; the margin bands above; passive edges beside 100–200 Myr
ocean ≤ 30% abrupt (Earth's 180 Myr Atlantic margins are not abrupt; an ocean-age grid, e.g. Seton et al. 2020,
would let this be measured on Earth); the elevation tests stay passing; crust bands read as they come out.

### Step 8, follow-up 1, twentieth pass (24 Oct): the sediment layer, and what really thickens old margins

**Built: `continental_sediment`** (default False until your sweep). Deposits on continental nodes go to `sed`
(density 2,400) instead of becoming crust of 2,830; erosion takes sediment first, then crust; sediment on continental
crust is a flexural load (dry where its top, on the crust's own water-loaded Airy level, is above sea level, as the
ocean rule; water-displacing below); sediment of continental crust consumed in a collision joins the receiving belt's
crust (`collision_sediment_to_crust_m3`); merges add the smaller plate's sediment. New `sim.sed_ledger` (increments,
arcs, new nodes, holes, collision, subduction, merge) and `continental_sediment_volume()`. With the layer off the code
is bit-identical to before (checked: n = 96, 12 steps, every state field equal).
Tests (`tests/test_plates.py`, 20 pass; 39 planet-layer tests pass): `test_continental_sediment_layer_keeps_both_ledgers`
(every erosion call: crust + sediment change = the kernels' surface change; continental crust never thickened by
deposits; run: crust ledger and sediment ledger both close), `test_continental_sediment_is_a_sediment_load` (n = 217,
1 km of sediment on wide dry crust raises it by s (ρm − ρs) / ρm = 273 m, on submerged crust by s (ρm − ρs) / (ρm − ρw)
= 396 m, within 5%: 2.5 m of sediment per m of water filled; ignored with the layer off).
Reviews count crust + sediment (as CRUST1.0's total thickness) for runs with the layer only, so earlier numbers stay
reproducible: `crust_review.total_thickness_m` / `run_params`, used by `crust_review`, `margin_width`,
`margin_age_probe`, `elevation_check` (thin-crust test: crust + sediment column at local Airy). `simulate_plates`:
`continental_volume_change` is crust + continental sediment with the layer (crust only otherwise), plus
`continental_crust_volume_change`, `continental_sediment_km3`, `sediment_ledger_km3`. `--latest` added to
`margin_width` and `margin_age_probe`.
Earth's sediment on continental crust (CRUST1.0 `sedthk`, n = 217; `crust_review.EARTH_SEDIMENT_M`, a read-out, not a
band): mean 1,569 m; on crust < 30 km 2,394; submerged 2,398; land 1,250 (the land figure includes Palaeozoic and older
basins; the planet starts with none, so the margin figures are the comparable ones).

**Found on the way, pre-existing:** `graph_landscape.erode_and_deposit` is not exactly closed: now and then (both
modes; ~2 × 10⁻⁵ of the gross at n = 48, e.g. 190 km³ on the first call of a run) it sends a little more to the sea
than it erodes. Negligible for the budget; the ledger test now checks the partition against the kernel's own total.
The plate-raster increments lose ~17% of the world's continental sediment deposits to the nearest-node averaging
(0.31 km³/yr deposited, 65 Mkm³ arrived over 250 Myr instead of 78; the crust increments share this property).

**One seed here** (`sweep-20261024-135611`, cloud seed 1001; Earth law, realistic margins, Earth rifting, local
isostasy; the sweep's texture score needs torch, absent here, so read with the reviews):

| setting | elevation tests (young ocean / thin crust offset) | area | < 30 km | > 45 km | median | sediment on cont. / on < 30 km / submerged | volume (crust) | land p25 / p50 | drowned | export |
|---|---|---|---|---|---|---|---|---|---|---|
| layer off | pass (−20 / −47 m) | 0.337 | 0.113 | 0.255 | 38.5 | — | −9.7% | 628 / 1,337 | 25% | 4.5 |
| layer on | pass (−112 / −25 m) | 0.313 | 0.102 | 0.298 | 40.0 | 371 / 1,017 / 821 m | −11.4% (−12.2%) | 848 / 1,592 | 14% | 4.2 |
| layer on, κ_normal 30 | pass (−222 / −156 m) | 0.309 | 0.066 | 0.331 | 39.9 | 877 / 4,619 / 3,173 m | −11.2% (−13.1%) | 882 / 1,387 | 8% | 4.9 |
| Earth | | 0.383 | 0.19 | 0.087 | 37.0 | 1,569 / 2,394 / 2,398 m | ~0 | 166 / 384 | ~29% | 4–8 |

| setting | thin per km of edge | abrupt edges | edge T 0–50 / 100–150 / 200–300 km | abrupt beside 0–25 / 50–100 / 100–200 Myr ocean |
|---|---|---|---|---|
| layer off | 78 km | 51% | 30.8 / 32.8 / 34.9 | 52% / 45% / 62% |
| layer on | 85 km | 47% | 29.8 / 32.0 / 34.9 | 11% / 30% / 66% |
| layer on, κ_normal 30 | 69 km | **28% ✓** | **27.4 ✓** / 37.5 / 43.3 | 5% / 32% / **36%** |
| Earth (bands) | 140 (100–200) | 17% (≤ 30%) | 26.4 (24–29) / 32.0 (30–34) / 35.0 | (≤ 30% beside 100–200 Myr) |

**The sediment layer alone does not fix the old margins**: sediment on thin crust is only ~1 km, far too little for
the 5–7 km by which old margins thicken. The nineteenth-pass reading ("sediment booked as crust") was at most part of it.

**What thickens them** (`scripts/margin_track_probe.py`, new: thin crust within 150 km of a passive edge at −100 Myr,
followed in its plate frame through the saved rotations; and the tectonic-thickening field over the last 40 Myr):
at quiet margins (no collision, > 300 km from any plate boundary) the **crust** thickens 25.6 → 31 km in 100 Myr
(layer off) and ~2 km per 40 Myr with the layer on, where no deposit can reach the crust. Over −40…0 Myr the
tectonic term is 1.1 km (layer off; erosion/deposits +0.4 km) and 1.8 km (layer on; rest +0.1 km). Running the plate
layer's own crust-flow kernel on the snapshots: **lateral flow at κ_normal 1000 moves 44–109 m/Myr into the thin
margins** (κ_normal 30: 1.4–3.4 m/Myr), as much as the whole tectonic term. With κ 30 + the layer the tracked margins
stay thin: crust 19.9 → 18.9 km, sediment wedge 2.7 → 3.8 km over 100 Myr.
Why the earlier κ tests missed it: 21 Oct (fitted rifting, deep-sea and lifted-ocean artifacts) nearly no thin margins
existed; 24 Oct (κ 30, one seed) deposits were still booked as crust and only the present-day abrupt share was read.
The two thickeners add; each alone leaves the old margins abrupt.

**κ_normal 30 is an Earth value, not a fit:** κ ≈ ρc g (1 − ρc/ρm) h³ / (3η) for a lower-crust channel of h = 10 km
gives ~40 m²/yr at η = 10²¹ Pa s (stable continental lower crust, ~10²¹–10²²) and ~1,000 at 4 × 10¹⁹ (hot crust);
independently, Earth's 180 Myr Atlantic margins survive, which needs a diffusion length well below their ~150 km
width, i.e. κ ≪ 125 m²/yr. The collapse flow of thick crust (> 45 km, crust_kappa × factor) is unchanged.

**What κ 30 exposes:** with the margins right, the crust 100–500 km inland is 37–44 km (Earth 32–37), the > 45 km
share rises to 0.33 and the thin share falls to 0.066; κ 1000 had been smoothing the thick interior along with the
margins. That is the known thick inland tail (collision belts; handoff item 2), now in the open. The margin sediment
overshoots on this seed (4.6 km on thin crust vs Earth's 2.4; 1° CRUST1.0 averages wedges with shelves). One seed;
offsets of the elevation tests grow (−222 / −156 m, limit ±300).

`Sediment Layer Sweep.cmd`: three settings (neither; the layer; the layer + κ_normal 30) with Earth law, realistic
margins and Earth rifting, 3 seeds, then `elevation_check`, `crust_review`, `margin_width`, `margin_age_probe`,
`margin_track_probe --from -100` (all `--latest`). **Decision rule (fixed now): adopt each Earth value
(`continental_sediment` True; `crust_kappa_normal_m2_yr` 30) whose 3-seed means pass the elevation tests; every other
band is read as it comes out.**
**Next after the sweep:** the thick inland tail with the missing high ground (collision thickening, interior collapse
of 38–45 km crust; CRUST1.0's thick tail and Earth's inland profile 33.7 / 35.0 / 36.7 km at 150–500 km from passive
edges), then Earth's thickness–elevation table (land 2–5× too high), then the margin sediment volume.

**Twentieth pass, your sweep** (`Sediment Layer Sweep.cmd`, `sweep-20261024-143735`; Earth law, realistic margins,
Earth rifting, local isostasy; 3 seeds; reviews run on your snapshots from here):

| setting | elevation tests: young ocean / thin crust offset | area | < 30 km | > 45 km | > 55 km | median | sed. on < 30 km / submerged | volume | drowned | score (tex / hyp / arr) |
|---|---|---|---|---|---|---|---|---|---|---|
| neither | pass (−31…−37 / −19…−39 m) | 0.325 | 0.089 | 0.294 | 0.107 | 39.7 | — | −9.4% | 7–22% | 0.70 (0.33 / 0.70 / 0.42) |
| layer | pass (−75…−119 / −66…−95 m) | 0.328 | 0.091 | 0.307 | 0.097 | 39.7 | 1,077 / 921 m | −9.6% | 10–16% | 0.78 (0.51 / 0.70 / 0.41) |
| **layer + κ_normal 30** | **pass (−261…−288 / +10…−80 m)** | 0.289 | 0.068 | 0.354 | 0.196 | 40.6 | 4,084 / 3,095 m | −13.2% | 4–8% | 0.85 (**0.19** / 0.85 / **0.15**) |
| Earth | | 0.383 | 0.19 | 0.087 | 0.018 | 37.0 | 2,394 / 2,398 m | ~0 | ~29% | |

| setting | abrupt edges (3 seeds) | edge T 0–50 km | abrupt beside 100–200 Myr ocean | tracked thin margin crust, −100 → 0 Myr (quiet) |
|---|---|---|---|---|
| neither | 51 / 48 / 78% | 30.7 / 30.6 / 33.9 | 67 / 47 / 82% | 25.4 → 33.3, 25.1 → 26.9, 26.3 → 31.3 km |
| layer | 63 / 66 / 80% | 31.6 / 33.6 / 36.9 | 72 / 71 / 81% | — |
| layer + κ 30 | 49 / 28 / 50% | 31.4 / 26.7 / 30.9 | 61 / 34 / 66% | 19.7 → 25.9, 18.4 → 26.4, 21.7 → 25.8 km (+ 3–4 km sediment) |
| Earth (bands) | 17% (≤ 30%) | 26.4 (24–29) | (≤ 30%) | |

**Decision (rule fixed before the sweep): both adopted, `continental_sediment` True and `crust_kappa_normal_m2_yr` 30
are the defaults** (3-seed means pass the elevation tests; the young-ocean offset, −278 m mean, is close to its
±300 m limit: ocean floor beside 3–4 km margin sediment wedges is flexed down by their load, as on Earth, but watch it).
No setting passes the shelf gate; nothing tuned. With κ 30 the texture (0.19) and arrangement (0.15: high ground
survives) are the best yet; hypsometry is the worst (land p50 1.5–1.9 km).

**Why κ 30 helps less on your seeds than on the cloud seed:** the planet's margins are only 1–2 cells wide. A 20 km
step across one 46 km cell still diffuses at ~280 m/Myr at κ 30, and the measured normal flow into tracked quiet thin
margins is 26–28 m/Myr on all three seeds (`outputs/test-tmp/claude-sync-1024b/tect_probe2.py`, last 40 Myr, nodes
> 300 km from any boundary throughout: crust +1.3–2.0 km per 40 Myr, collapse flow ~0). The diffusion length over
100 Myr at κ 30 (~55 km) is one cell: enough to fill a one-cell margin, nothing on Earth's 100–300 km margins. So the
remaining cause is margin *width*: the rift rule makes 45–70 km of thin crust per km of passive edge (Earth 140).
Also: the plate-raster nearest-node sampling ("rest", +0.4–0.6 km per 40 Myr on two seeds) adds to quiet margins.

**Next (to agree), tests fixed before building:**
1. **A rift rule stated in distance** (nineteenth pass, candidate 1): stretched crust thinned along a profile from
   Earth's margins (crust 35 → < 10 km over ~100–300 km, e.g. Iberia–Newfoundland, Péron-Pinvidic & Manatschal 2009)
   instead of `rift_thinning` 0.6 per gap node. Tests: the margin bands (thin per km of edge 100–200 km, abrupt ≤ 30%,
   edge T 24–29, inner T 30–34), abrupt beside 100–200 Myr ocean ≤ 30%, the elevation tests, ledgers closed.
2. The thick inland tail with the missing high ground (> 45 km 0.35, > 55 km 0.20 vs 0.087 / 0.018; inland profile
   38–46 km at 100–500 km from passive edges vs Earth 32–37), now exposed since flow no longer smooths it.
3. Earth's thickness–elevation table (land 2–5× too high), the young-ocean offset watch-point, margin sediment
   (3–4 km on thin crust vs Earth's 2.4).

### Step 8, follow-up 1, twenty-first pass (24 Oct, evening): a rift rule stated in distance

**The 1 degree instrument first** (`scripts/margin_width.py`, measurement fix, bands unchanged): Earth's margins have
crust < 10 km at the continent–ocean boundary on seismic sections (hyper-extended slivers 70–200 km wide, Brune et
al. 2014), yet CRUST1.0 reads 26.4 km 0–50 km from the edge: its 1° cells average the thin strip with the shelf. A
planet margin read at 46 km was therefore not comparable with the edge band. `to_crust1_cells` now shows the planet
through the same cells (each node takes the area-weighted mean thickness of the planet nodes in its 1° cell;
continental where continental nodes cover ≥ half of it); the bands are judged on that reading, the 46 km reading is
printed beside it. On the current baseline (`sweep-20261024-143735`, layer + κ 30) it changes little: thin per km of
edge 70 / 83 / 58 km at 1° vs 63 / 70 / 45 at 46 km; abrupt 49 / 27 / 42% vs 49 / 28 / 50%.

**Built: `rift_rule`** (`'node'` default, unchanged code path; `'pure_shear'` new). With `pure_shear`, a gap opening
between continental crust is the new area of a stretching zone: continental nodes within `rift_width_km` (100 km, the
Earth necking value already used by the realistic-margin setting) of the gap cell's edge, reached through continental
crust only, with a linear taper of the strain rate. Each zone node thins as pure shear, T exp(−c), c its share of the
opened area (a_g k / Σ a k per gap); every gap receives exactly the volume the zone loses (the strain-weighted mean
thickness for small strain; less for a whole 46 km cell opening into a 100 km zone, a finite stretch). A gap whose
crust would be thinner than `breakup_thick_m` (10 km) is ocean floor (iterated, so no gap is born thinner). Kernel
`plates.rift_pure_shear` (numba, bounded Dijkstra per gap). `rift_thinning` 0.6 per cell is not used by this rule.
1D prototype first (`rift1d.py`, workspace only): a fixed zone of half-width W stretched until breakup gives a flat
~10 km hyper-extended domain with a necking zone at its inner edge; crust < 30 km over 1.6–2.2 W per side (lin taper,
W = 100 km: 160 km), insensitive within Earth's narrow-rift widths (full zone 100–200 km: 90–230 km).

Tests (`tests/test_plates.py`, 23 pass; 41 planet-layer tests pass here without `test_erosion_law_check`):
`test_pure_shear_rift_kernel_conserves_volume_and_is_resolution_independent` (a great-circle rift through 38 km crust
at n = 217 and 434: volume exact to 1e-9, gap ≤ the zone's mean and ≥ its most-stretched node, volume-weighted mean
distance of the source from the gap edge W/3 within 6 km at both resolutions: 32.7 / 30.6 km, nothing from beyond
W), `test_pure_shear_rift_run_conserves_volume_and_breaks_up_at_breakup_thickness` (n = 48 run: ledger closes, no
rifted gap thinner than 10 km), `test_node_rule_is_the_default_and_unchanged`.

**One seed here** (cloud seed 1001; Earth law, realistic margins, Earth rifting 10 km / 100 km, local isostasy,
continental sediment, κ_normal 30; `outputs` in the workspace only):

| `rift_rule` | thin per km of edge (1°) | abrupt | edge T 0–50 / 50–100 / 100–150 / 200–300 km (1°) | area | < 30 km | > 45 km | > 55 km | median | volume | land p25 / p50 / p90 | elevation tests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| node | 75 km ✗ | 26% ✓ | 26.3 ✓ / 33.4 / 37.0 ✗ / 42.9 | 0.309 | 0.066 | 0.331 | 0.145 | 39.9 | −11.2% | 882 / 1,387 / 3,930 | pass (−222 / −156 m) |
| **pure_shear** | **165 km ✓** | **6% ✓** | 16.6 ✗ / 25.5 / 29.5 ✗ / 35.3 | **0.348** | **0.128 ✓** | **0.227** | **0.002** | **39.1 ✓** | −10.0% | 872 / 1,343 / 2,925 | pass (−287 / −94 m) |
| Earth (bands) | 140 (100–200) | 17% (≤ 30%) | 26.4 (24–29) / 30.0 / 32.0 (30–34) / 35.0 | 0.383 (0.35–0.42) | 0.19 (0.12–0.26) | 0.087 (≤ 0.13) | 0.018 | 37.0 (34–40) | ~0 | 166 / 384 / 1,461 | |

| `rift_rule` | abrupt beside 0–25 / 50–100 / 100–200 / > 200 Myr ocean | edge T beside 100–200 Myr ocean | tracked quiet thin margins, −100 → 0 Myr (crust / sediment) |
|---|---|---|---|
| node | 5% / 32% / **36%** / 27% | 26.8 km | 19.9 → 21.9 km / 2.7 → 3.6 km |
| pure_shear | 6% / 2% / **16% ✓** / 12% | 18.9 km | 16.3 → 18.8 km / 3.2 → 3.1 km |

- **The margin width was the rift rule.** Thin crust per km of passive edge 75 → 165 km (band 100–200), abrupt edges
  26 → 6%, old margins stay thin (16% abrupt beside 100–200 Myr ocean). The crust passes 2 of 4 bands (thin share,
  median) and misses area by 0.002; the thick tail halves (> 45 km 0.33 → 0.23) and > 55 km nearly vanishes (0.145
  → 0.002; Earth 0.018): wider margins also mean less continent to pile into collision belts (collision length 9,050 →
  5,970 km on this seed). High land lower (land p90 3.9 → 2.9 km; Earth 1.5).
- **Now too thin at the edge**: 16.6 km at 1° (band 24–29) and 29.5 km 100–150 km inland (30–34). The rule makes
  Earth's seismic profile (a ~10 km hyper-extended domain); CRUST1.0's edge cells are thicker. Not tuned: the next
  candidates are (a) thermal subsidence and the sediment wedge (the planet's margins carry 3.7 km of sediment on
  crust < 30 km vs Earth's 2.4, so it is not missing sediment), (b) CRUST1.0's coarse type profiles at margins
  (its 1° "thinned / transitional" types), (c) asymmetric conjugates (Earth: one side < 100 km, the other 100–350 km;
  the rule is symmetric). Read on 3 seeds first.
- Young-ocean offset −287 m (limit ±300): the flexural load of the margin sediment wedges, as on Earth; watch.

`Rift Rule Sweep.cmd`: the current baseline (Earth law, realistic margins, Earth rifting, layer, κ 30) with
`rift_rule` node and pure_shear, 3 seeds, then `elevation_check`, `crust_review`, `margin_width`, `margin_age_probe`,
`margin_track_probe --from -100` (all `--latest`). **Decision rule (fixed now): adopt `pure_shear` if its 3-seed means
pass the elevation tests and its thin crust per km of passive edge (1° reading) is inside 100–200 km; every other
band is read as it comes out.**
**Next after the sweep:** the edge-thickness gap above (read on 3 seeds, then the candidates in order), then the thick
inland tail (now 0.23 on one seed) with the missing high ground, then Earth's thickness–elevation table.

**Twenty-first pass, your sweep** (`Rift Rule Sweep.cmd`, `sweep-20261024-173200`, the first with short folder names;
current baseline; 3 seeds; reviews run on your machine). Margin numbers are the 1° reading:

| `rift_rule` | elevation tests: young ocean / thin crust offset | thin per km of edge | abrupt | edge T 0–50 | inner T 100–150 | abrupt beside 100–200 Myr ocean | area | < 30 km | > 45 km | > 55 km | median | volume | land p50 / p90 | score (tex / hyp / arr), export |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| node | pass (−261…−288 / +10…−80 m) | 70 / 83 / 58 ✗ | 49 / 27 / 42% | 31.1 / 25.9 / 30.0 | 42.2 / 36.1 / 35.7 | 61 / 34 / 66% | 0.289 | 0.068 | 0.354 | 0.196 | 40.6 | −13.2% | 1,772 / 4,780 m | 0.84 (0.19 / 0.84 / 0.15), 6.3 |
| **pure_shear** | **pass (−210…−286 / −79…−99 m)** | **141 / 130 / 144 ✓** | **9 / 17 / 10% ✓** | 18.1 / 19.8 / 19.8 ✗ | **30.8 / 30.1 / 31.2 ✓** | **4 / 34 / 17%** (mean 18% ✓) | 0.329 | 0.108 | 0.270 | 0.087 | 39.9 ✓ | −10.2% | 1,563 / 3,597 m | 0.81 (0.26 / 0.81 / 0.21), 5.2 |
| Earth (bands) | | 140 (100–200) | 17% (≤ 30%) | 26.4 (24–29) | 32.0 (30–34) | (≤ 30%) | 0.383 (0.35–0.42) | 0.19 (0.12–0.26) | 0.087 (≤ 0.13) | 0.018 | 37.0 (34–40) | ~0 | 384 / 1,461 m | |

Tracked quiet thin margins, −100 → 0 Myr (crust): node 19.7 → 25.9, 18.4 → 26.4, 21.7 → 25.8 km; pure_shear
15.6 → 19.6, 16.4 → 19.6, 16.2 → 19.9 km (sediment 3–5 km on both). Neither passes the shelf gate (nothing chosen
by the sweep's own rule; the rift decision has its own).

**Decision (rule fixed before the sweep): `rift_rule` pure_shear adopted, default since 24 Oct**, with the Earth
rifting values it was decided with as the new defaults (`breakup_thick_m` 10 km, `rift_width_km` 100 km; they were
passed in every sweep since 23 Oct, and pure_shear needs a width: it now refuses `rift_width_km` 0). `'node'`
reproduces earlier runs. Tests updated (`test_pure_shear_is_the_default_with_earth_rifting`); 41 pass here.
- Margins pass 3 of 4 bands on the 3-seed means (thin per km 138, abrupt 12%, inner T 30.7); the edge is too thin
  (19.2 km vs 24–29), as on the cloud seed. Old margins stay non-abrupt (18% beside 100–200 Myr ocean, from 54%).
- Crust: median passes; thin share 0.108 (band from 0.12), area 0.329 (from 0.35); thick tail 0.354 → 0.270 and
  > 55 km 0.196 → 0.087, still 3× and 5× Earth. Land lower (p50 1.77 → 1.56 km, p90 4.8 → 3.6 km) but 4× / 2.5×
  Earth. Quiet margins still thicken ~3.5 km per 100 Myr (from 6–8).
- Texture and arrangement slightly worse (0.19 → 0.26, 0.15 → 0.21), hypsometry better (0.84 → 0.81).

**Old sweep folders renamed (24 Oct, with you):** 28 combination folders in 9 sweeps (names of 84–199 characters,
2,660 files over Windows' 260-character path limit, longest path 319) now follow `sweep_plates`' short naming
(`MAX_NAME` 80: only the differing parameters, or `combo-NN`); `sweep.json` in each lists the shared parameters and
`renamed_from`; `outputs/planet-calibration/renamed-2026-10-24.json` maps all; stored plate paths in 26
planet-landscape JSON files rewritten; `results.json` gained `chosen_dir`. Longest path in `outputs` now 183.

**Next (to agree):**
1. The edge-thickness gap (19.2 vs 24–29 km at 1°): candidates in order, thermal subsidence and the sediment wedge
   (margin sediment on thin crust 4.4 km vs Earth's 2.4), CRUST1.0's coarse margin types, asymmetric conjugates.
2. The thick inland tail with the missing high ground (> 45 km 0.27, > 55 km 0.087; land p50 4× Earth): collision
   thickening and interior collapse.
3. Earth's thickness–elevation table; area 0.329 and thin share 0.108 read again after 1–2.

### Step 8, follow-up 1, twenty-second pass (25 Oct, night): the edge gap was the continent label; the real gaps are the thick interior and the short passive edge

**New: `scripts/margin_cross_section.py`** (read-only review; Earth = CRUST1.0 on n = 217, planet = `sweep-20261024-173200`,
both rules, 3 seeds, run here on copies of your snapshots). It measures passive margins on both sides of the edge and
without labels: total thickness (crust + sediment, ocean side too, as CRUST1.0's `crsthk`) in distance bins from the
passive edge (negative = ocean side), the same aligned on the 20 km thickness contour, and the area of 10–20 / 20–30 km
crust of any label per km of passive edge.

**Earth's continent label sits at ~20 km of crust; the planet's at breakup (10 km).**
- CRUST1.0 calls the cells next to passive margins oceanic (type A1/B-) although they carry 11–16 km of crust (12.9 km
  crystalline 0–50 km off the edge, 10.1 km at 50–100 km). 87% of Earth's 10–20 km crust near passive edges is
  labelled oceanic; the planet labels it continental down to 10 km.
- Earth's label as a function of thickness: continental share 0.17 at 16–18 km, 0.41 at 18–20, 0.44 at 20–22, 0.85 at
  24–26 km (total). **A label-free rule "total thickness ≥ 20 km" reproduces CRUST1.0's own continent on Earth:** area
  0.384 (types 0.383), < 30 km 0.189 (0.189), > 45 km 0.086 (0.087), median 37.0 (37.0); margins thin per km of
  edge 155 (140), abrupt 11% (17%), edge T 25.2 (26.4), inner T 31.6 (32.0).
- So the edge band (24–29 km at 0–50 km) measured the planet ~50 km further out on the same profile than Earth.
  Aligned on the 20 km contour, the planet's pure-shear margin matches Earth from −200 to +100 km within ~1 km
  (total, 3-seed means at 1°: 8.7 / 10.3 / 13.5 / 16.1 | 24.5 / 26.8 km vs Earth 9.9 / 11.0 / 13.5 / 15.5 | 23.8 /
  25.9); the node rule has bare 7 km ocean crust up to the edge (its "thick" ocean side is sediment).

**The planet read through Earth's rule (≥ 20 km on both; pure_shear, 3-seed means; bands unchanged):**

| | area | < 30 km | > 45 km | > 55 km | median | thin per km (1°) | abrupt | edge T | inner T |
|---|---|---|---|---|---|---|---|---|---|
| planet flag (the reviews so far) | 0.329 | 0.108 | 0.270 | 0.087 | 39.9 | 137 | 13% | 19.8 ✗ | 30.7 |
| planet, ≥ 20 km | 0.313 ✗ | 0.063 ✗ | 0.283 ✗ | 0.092 | 40.1 ✗ | 97 ✗ (100–200) | 18% ✓ | **25.9 ✓** | 34.6 ✗ (30–34) |
| Earth, ≥ 20 km (types) | 0.384 (0.383) | 0.189 | 0.086 | 0.018 | 37.0 | 155 (140) | 11% (17%) | 25.2 (26.4) | 31.6 (32.0) |

The edge now passes; the area, thin share and median are worse than the flag said (the flag counted 10–20 km
transitional crust that Earth calls ocean). Node rule under the same reading: area 0.289, < 30 km 0.066, edge T 28.2,
inner T 36.5, abrupt 31%.

**What is really different** (label-free, pure_shear, 3 seeds):
1. **The interior is 3.5 km too thick, in the crystalline crust.** Inland profile from passive edges (1°, km, 150–200 /
   200–300 / 300–500 / 500–1000): total 36.5 / 38.2 / 39.5 / 40.7 vs Earth 33.0 / 35.0 / 36.0 / 37.1; crystalline
   35.3 / 37.5 / 39.2 / 40.3 vs 31.6 / 33.8 / 35.4 / 36.6. Not sediment (inland it is thinner than Earth's). This is
   handoff item 2 (thick tail, high land), and it starts 100 km from the edge.
2. **The passive edge is 40% of Earth's length.** Continent–ocean edge (≥ 20 km, 1°, same staircase on both): Earth
   306k km (passive 173k, active 133k); planet 153–183k (passive 60–69k, active 83–115k). Earth's small enclosed ocean
   basins (< 20 Mkm²: Gulf of Mexico, Mediterranean, Black Sea, Arctic, …) carry 62k km of edge (34k passive); the
   planet has almost none (0–7k). Facing the world ocean Earth's passive edge is still 2× the planet's.
   Earth's 20–30 km continental crust covers 7.3% of the surface, the planet's 2.0%: ≈ edge length 0.38 × width per km
   of edge 0.63 (97 vs 155) — the thin-share gap is mostly edge length, not margin profile.
3. The margin profile itself (−200..+100 km of the 20 km contour) is Earth-like; nothing to fix there.

**Proposed (to agree; nothing in the code changed):**
- Reviews read continental crust label-free as total ≥ 20 km on both sides (`crust_review`, `margin_width`), the
  planet flag reading printed beside it. The rule comes from Earth (it reproduces CRUST1.0's types), not from the
  planet, and it makes the planet's crust numbers worse; the bands stay as they are. Noted honestly: I read the planet
  numbers before proposing it.
- Item 1 (edge thickness) closed: instrument, not mechanism; thermal subsidence, CRUST1.0 types and asymmetric
  conjugates are not needed for it.
- Next in order: (a) the thick interior (crystalline +3.5 km from 150 km inland on; collision thickening, interior
  collapse; tests: the inland profile 150–1000 km within ±1.5 km of Earth's, > 45 km ≤ 0.13, median 34–40, the
  elevation tests); (b) the short passive edge: small ocean basins and microcontinents (back-arc opening, failed and
  short-lived rifts; Bird's 32 microplates vs the planet's ~3), tests to be fixed first (passive edge length and
  edge in basins < 20 Mkm² against Earth's 173k / 34k km).

**Agreed with the user (25 Oct):** the label-free reading (total ≥ 20 km on both sides) becomes the reviews' rule,
the planet flag printed beside it, bands unchanged; then (a) the thick interior with the tests above, then (b) the
short passive edge. The review change is the first job of the next chat.

### Step 8, follow-up 1, twenty-third pass (25 Oct): label-free reviews; the thick interior is the start, the thick tail is unmarked orogens

**Reviews changed (agreed 25 Oct).** `crust_review` and `margin_width` read continental crust label-free: total
thickness (crust + sediment) ≥ 20 km (`crust_review.CONTINENT_MIN_TOTAL_M`), on both sides. Planet columns
(`column_thickness_m`): crust + sediment on every node, ocean side too (as CRUST1.0's crsthk), continental sediment as
`total_thickness_m`. Earth is computed from CRUST1.0 (cos-latitude weights): area 0.384, < 30 km 0.190, > 45 km 0.086,
> 55 km 0.018, median 37.0 (types: 0.383 / 0.190 / 0.087 / 0.018 / 37.0); margins thin per km of edge 155, abrupt 11%,
edge T 25.2, inner T 31.6 km (types 140 / 17% / 26.4 / 32.0). `crust_review` stays at 46 km; `margin_width` reads the
1° cells (cell-mean column ≥ 20 km) and prints the 46 km label-free reading, the flag reading at 1° and 3-seed means
per combination. The bands are unchanged. `elevation_check`'s crust line now shows the label-free reading (it calls
`crust_review.metrics`, whose `flag_*` keys keep the old one). New test `tests/test_crust_reviews.py` (2 pass): the
rule reproduces CRUST1.0's types on Earth within 0.005 / 0.5 km; sediment counted on both sides.

**`sweep-20261024-173200` re-read** (3-seed means; margins at 1°):

| `rift_rule` | area | < 30 km | > 45 km | > 55 km | median | thin per km | abrupt | edge T | inner T |
|---|---|---|---|---|---|---|---|---|---|
| node | 0.289 | 0.066 | 0.354 | 0.195 | 40.6 | 94 | 27% | 27.7 | 36.4 |
| pure_shear | 0.313 ✗ | 0.063 ✗ | 0.283 ✗ | 0.092 | 40.1 ✗ | 98 ✗ (94 / 109 / 90) | 16% ✓ | 26.0 ✓ | 34.6 ✗ |
| Earth (bands) | 0.384 (0.35–0.42) | 0.190 (0.12–0.26) | 0.086 (≤ 0.13) | 0.018 | 37.0 (34–40) | 155 (100–200) | 11% (≤ 30%) | 25.2 (24–29) | 31.6 (30–34) |

The rift decision stands (its rule was stated on the flag reading; under the label-free reading pure_shear is still
better on every margin band and on area). The twenty-second pass's ad hoc numbers (97 / 18% / 25.9) and the review's
(98 / 16% / 26.0) differ by rounding of the cell mean.

**Item 2, the thick interior: diagnosis** (new `scripts/interior_probe.py`, read-only, `--earth` adds the Earth side;
pure_shear, 3 seeds, on your machine). Provenance of present-day continental crust: *inherited* (older than the run,
never received collision volume), *collided* (orogeny age < 250 Myr), *new* (born in the run).

1. **The interior is the start, untouched.** 95–99% of the crust 150 km and more inland from passive edges is
   inherited, crystalline median 39.9–40.7 km at present, the same as at −250 Myr (39.1–40.1: `_init_supercontinent`
   sets 37.5 km + 2 km more beyond 300 km from old sutures + 4 km near sutures + a ±1.5 km field; audit kind A). Cold
   crust at κ_normal 30, eroding at erodibility 0.15, keeps its starting thickness for 250 Myr. Earth's stable
   continental crust (CRUST1.0 Precambrian and platform types D–I, 0.48 of the continent; > 500 km from plate
   boundaries) has crystalline p10 / 25 / 50 / 75 / 90 = **32.9 / 35.9 / 38.2 / 40.5 / 43.0 km**; the planet's start
   (> 200 km from the coast) has p50 39.2 / 39.8 / 40.1 km and p10–p90 3.2–4.1 km wide (Earth 10.1): 1–2 km thicker,
   a third of Earth's spread.
2. **Earth's inland bins are not all craton.** From passive edges (crystalline p50; share Precambrian / shelf and
   slope / extended): 150–200 km 31.6 (0.40 at 35.8 / 0.32 at 26.6 / 0.11 at 30.9), 200–300 km 33.8 (0.49 / 0.23 /
   0.10), 300–500 km 35.4 (0.55 / 0.16 / 0.10), 500–1000 km 36.6 (0.62 / 0.11 / 0.09), 1000–5000 km 39.1 (0.68 / 0.02 /
   0.03; orogens 0.22 at 43.6). The planet's deep interior (1000–5000 km: 39.4–40.7) is near Earth's 39.1; the gap at
   150–1000 km is a third start (~1.5–2 km) and two thirds Earth's wide epicontinental shelves and extended crust
   (26–31 km over 20–40% of those bins), which the planet does not make: item 2b's family (short passive edge, no
   small basins, failed rifts). Earth's start distribution alone would leave the 150–500 km bins ~2 km thick:
   **the agreed profile test (±1.5 km) cannot pass by the interior alone.**
3. **The thick tail is unmarked orogens.** > 45 km: 0.265–0.294 of the continent (Earth 0.086), at −250 Myr 0.000.
   0.24–0.27 of it is "inherited": crust thickened by the margin-shortening belts and spread by collapse (the collapse
   diffusivity rises above `collapse_thick_m` 45 km, so collision volume flattens into broad 45–55 km plateaus: 0.19 of
   the continent is 45–55 km, Earth 0.068). Only collision contact nodes reset the orogeny clock, so 96–98% of the
   > 45 km crust erodes at the stable erodibility (mean 0.17). Young-orogen erodibility (≥ 0.5) covers 1–2% of the
   planet's land and 2–4% of its land above 2 km; **Bird's orogens (with which the 46 km erosion law was calibrated at
   erodibility 1, `erosion_law_check`) cover 16% of Earth's land and 76% of its land above 2 km.** The law is applied
   inconsistently with its calibration. At present 69–70% of the > 45 km crust deforms at > 1 nanostrain/yr (|tectonic
   thickening| / T), 30–40% at > 3, 7–17% at > 10 (land: 33–35%, 17–21%, 5–8%; GSRM: 14% of Earth deforms, its prior
   for deforming cells 10 nanostrain/yr, Kreemer et al. 2014; GSRM's grid is blocked by the allowlist).
   Land above 2 km: 0.32–0.36 of the planet's land, 0.01–0.03 at the start; Earth 0.052 (0.062 without |lat| > 60).
   The > 45 km land stands at p50 2.96–3.32 km; Earth's at 1.23 km.
4. **Earth's thick crust stands low.** Earth's elevation by total thickness (CRUST1.0 × ETOPO, ice excluded,
   `earth-crust-elevation-n217.json`): 30–35 km 98 m, 35–40 km 321, 40–45 km 359, 45–55 km 885, 55–80 km 4,356; the
   planet's local Airy relation for the same bins −748 / 239 / 814 / 1,582 / 4,131 m. Between 30 and 48 km Earth rises
   ~800 m, the planet ~2,300 m. Earth's > 45 km crust is 52% inside Bird's orogens (> 50 km 78%, > 55 km 85%); outside
   them it is Precambrian (H1, G1, G2, I1, D-) and stands at 466 m. Thin crust stands higher than the Airy relation,
   thick old crust lower: the lithosphere's thermal state (young hot under rifted margins, old cold under cratons), not
   in the plate layer.

**Proposed (to agree; nothing in the model changed yet):**
- **A. Start from Earth's stable crust** (audit A → M): the supercontinent's interior thickness (beyond the 200 km rim)
  quantile-mapped onto Earth's Precambrian / platform crystalline distribution above (p50 38.2, p10–p90 32.9–43.0),
  keeping the field's spatial structure (sutures thicker, as now). Default off until the sweep.
- **B. Orogen = deforming crust** (consistency with the erosion law's calibration): the orogeny clock is also reset on
  continental nodes in the margin-shortening belt (the band holding 95% of the belt volume, d0 ± 2σ, while its
  shortening rate is > 0) and on crust in the collapse regime (T > `collapse_thick_m`, already the model's "hot, weak
  crust"), not only at collision contacts. No new parameter. Default off until the sweep.
- **C. Thickness–elevation from the lithosphere** (later; larger): Earth's table (item 4) as the test; candidates
  thermal age of the lithosphere (rift-flank uplift and thermal subsidence; cold cratonic roots).
- Tests, fixed now: the agreed item-2 tests (inland profile 150–1000 km within ±1.5 km of Earth's 31.6 / 33.8 / 35.4 /
  36.6, > 45 km ≤ 0.13, median 34–40, the elevation tests), plus young-orogen erodibility on 0.10–0.25 of land and on
  ≥ 0.5 of land above 2 km (Bird 0.16 / 0.76), land above 2 km ≤ 0.10 of land (Earth 0.05–0.06), export 4–8 km³/yr.
  Expected from item 2: A and B can pass > 45 km, the median and the high-land tests; the 150–500 km profile needs 2b.
- Sweep proposed: A × B (2 × 2), 3 seeds, current baseline; decision rule: adopt each part whose 3-seed means move
  the tests above toward Earth without failing the elevation tests or the export band; read the rest as it comes.

### Step 8, follow-up 1, twenty-fourth pass (25 Oct): A and B built; Earth's deforming-crust threshold from GSRM; the interior sweep

**Agreed with the user (25 Oct):** build A and B, prepare the sweep. The user downloaded GSRM v2.1 (`data/gsrm`).

**Earth's deforming crust, measured** (new `scripts/earth_deformation.py`; `outputs/planet-reference/
earth-deformation-n217.json` / `.npz`). GSRM v2.1 (Kreemer et al. 2014; `GSRM_average_strain_v2.1.txt`, unpacked from
the `.Z`): strain-rate tensors of the deforming cells (0.25° × 0.2°); rigid plates are exactly zero. Each graph node
takes the area-weighted mean tensor of its cells. The planet's proxy is |dT/dt| / T, for incompressible crust the
areal dilatation |exx + eyy| (strike-slip shows in neither). On continental land (CRUST1.0 ≥ 20 km, ice excluded):
GSRM's deforming cells cover 24% of land (12.4% of the surface); inside Bird's orogens the dilatation is p10 / 25 / 50
/ 75 / 90 = 0.4 / 1.7 / 5.0 / 14.9 / 38.1 nanostrain/yr, outside p90 1.5, p95 16.1.
- Separation of Bird's orogens from the rest (Youden index) is flat from 0.1 to ~1 nanostrain/yr (TPR 0.95 → 0.83,
  FPR 0.13 → 0.11): GSRM's zeros make it a deforming/rigid split, so it does not fix a rate.
- **Adopted criterion (fixed before reading the planet with it): equal area.** The threshold at which Earth's flagged
  land equals Bird's orogen land (the area the erosion law's erodibility-1 class had in its calibration):
  **3.5 nanostrain/yr** dilatation (flags 17.0% of land; TPR 0.59, FPR 0.08; 67% of land above 2 km). Second
  invariant for comparison: 6.2 nanostrain/yr.

**Built (`planet_layer/plates.py`, defaults unchanged):**
- `start_crust` 'assumed' (default) / 'earth' (A): the start's thickness field on continental nodes is quantile-mapped
  (`quantile_map`, area-weighted mid-ranks) onto `EARTH_STABLE_CRUST_KM` (21 quantiles, 1–99%, of CRUST1.0's
  crystalline Precambrian / platform crust > 500 km from plate boundaries: p10 / 50 / 90 = 32.91 / 38.17 / 43.0 km),
  the field's order kept; the 200 km rim taper applied after, as before. Same random stream, so 'assumed' runs are
  unchanged.
- `orogen_rule` 'collision' (default) / 'belt' / 'strain' (B), `orogen_strain_rate` 3.5e-9 /yr. The orogeny clock is
  reset on collision contacts (always) and, by the rule, on `_orogen_marks()`: 'belt' = continental nodes of the
  subduction belt (`self.belt`: within 2σ of `subduction_d0_cont_km` on the overriding plate, converging) or thicker
  than `collapse_thick_m`; 'strain' = continental nodes with |tectonic thickening| / max(T, 1 km) per year above
  `orogen_strain_rate` (the step's tectonic field: crust flow, rifting, subduction belt and arc, volcanism, collision
  volume; no erosion). Marks go to the world and the plate rasters (`self.orogen_marked`).
- Tests (`tests/test_plates.py`, 27 pass on your machine; 48 planet-layer tests in all, the coupling ones run from a
  copy because the shell there cannot delete its old temp folder): `test_earth_stable_crust_quantiles_match_crust1`,
  `test_quantile_map_keeps_order_and_takes_the_target_distribution`, `test_earth_start_keeps_the_field_and_takes_earths_thickness`
  (n = 96: same continent, rank correlation > 0.999, interior p50 38.17 ± 0.5 km, p10–p90 > 7 km),
  `test_orogen_rules_mark_more_than_collisions_and_keep_the_ledger`.
- Smoke runs (n = 217, seed 1001, 20 Myr, baseline; transients, not judged): 'earth' start, crystalline median 38.0
  km; 'strain' marks 14% of land and 68% of land above 2 km after 20 Myr; 'belt' 6% / 84%.
- `interior_probe` gained `--latest` and 3-seed means of the item-2 tests per combination (its "collided" now means
  any orogeny-clock reset). The current baseline passes 0 of 5 (profile worst +4.0 km at 46 km, > 45 km 0.283,
  young orogen on 1.6% of land and 2.8% of land > 2 km, land > 2 km 0.340).

**`Interior Sweep.cmd`:** `start_crust` {assumed, earth} × `orogen_rule` {collision, belt, strain}, 3 seeds (18 runs,
~1 h), current baseline (Earth law, realistic margins, pure-shear rifting, local isostasy, sediment layer, κ 30), then
`elevation_check`, `crust_review`, `margin_width`, `interior_probe` (all `--latest`). **Decision rule (fixed before
the sweep):** gates = the elevation tests pass and the export is 4–8 km³/yr (3-seed means); among combinations through
the gates, the one passing most of the five `interior_probe` tests plus the crust median 34–40 km wins, ties to the
smaller > 45 km share; its `start_crust` and `orogen_rule` become defaults; nothing is adopted if none passes the
gates. Expected: the 150–500 km profile stays ~2 km thick (needs 2b).

### Step 8, follow-up 1, twenty-fifth pass (25 Oct, evening): the interior sweep adopts A and B by its rule; neither thins the tail; the high land is plateau

**The sweep** (`Interior Sweep.cmd`, `sweep-20261024-193840`, 6 combinations × 3 seeds, current baseline). The review
printouts are saved in `<sweep>/reviews/` (`elevation_check.txt`, `crust_review.txt`, `margin_width.txt`,
`interior_probe.txt`; rerun in the device shell, which had no network and no numba this time: the reviews run on the
repo's pure-Python fallback in 2–130 s). 3-seed means; crust label-free (total ≥ 20 km), margins at 1°.

| start, orogen rule | young ocean / thin crust offset (±300 m) | export (4–8) | interior tests passed (of 5) | median (34–40) | > 45 km (≤ 0.13) | young orogen on land / on land > 2 km (0.10–0.25 / ≥ 0.5) | land > 2 km (≤ 0.10) | profile 150–200 / 200–300 / 300–500 / 500–1000 km (±1.5) |
|---|---|---|---|---|---|---|---|---|
| assumed, collision | −250 / −91 | 5.2 | 0 | 40.1 ✗ | 0.283 | 0.016 / 0.028 | 0.340 | 35.6 / 37.3 / 39.0 / 40.1 |
| assumed, belt | −179 / −97 | 5.8 | 1 | 40.4 ✗ | 0.296 | 0.404 / 0.846 ✓ | 0.350 | 36.7 / 38.4 / 39.8 / 40.9 |
| assumed, strain | −219 / −117 | 6.1 | 1 | 40.5 ✗ | 0.307 | 0.470 / 0.751 ✓ | 0.346 | 36.3 / 38.0 / 39.4 / 40.3 |
| earth, collision | −216 / −135 | 6.0 | 0 | 39.7 ✓ | 0.301 | 0.030 / 0.034 | 0.360 | 34.7 / 36.6 / 38.6 / 40.2 |
| **earth, belt** | −278 / −110 | 5.9 | 1 | **39.6 ✓** | **0.298** | 0.422 / 0.868 ✓ | 0.342 | 35.4 / 37.0 / 38.6 / 39.7 |
| earth, strain | −262 / −135 | 6.8 | 1 | 39.5 ✓ | 0.306 | 0.459 / 0.747 ✓ | 0.372 | 36.2 / 37.9 / 39.2 / 39.9 |
| Earth | ~0 | 4–8 | | 37.0 | 0.086 | 0.16 / 0.76 | 0.05–0.06 | 31.6 / 33.8 / 35.4 / 36.6 |

| start, orogen rule | area / < 30 km / > 55 km | p10 (km) | thin per km / abrupt / edge T / inner T | volume | land p25 / p50 (m) |
|---|---|---|---|---|---|
| assumed, collision | 0.313 / 0.063 / 0.092 | 33.7 | 98 / 16% / 26.0 / 34.6 | −10.2% | 1,072 / 1,563 |
| assumed, belt | 0.310 / 0.072 / 0.065 | 32.7 | 94 / 18% / 26.4 / 35.6 | −11.7% | 1,031 / 1,541 |
| assumed, strain | 0.307 / 0.089 / 0.088 | 31.1 | 101 / 18% / 26.0 / 35.3 | −12.0% | 947 / 1,499 |
| earth, collision | 0.306 / 0.095 / 0.076 | 30.3 | 128 / 16% / 25.6 / 34.3 | −11.7% | 972 / 1,595 |
| earth, belt | 0.303 / 0.090 / 0.088 | 30.7 | 117 / 16% / 26.1 / 34.5 | −11.7% | 913 / 1,510 |
| earth, strain | 0.296 / 0.096 / 0.080 | 30.3 | 104 / 16% / 26.0 / 35.3 | −13.7% | 878 / 1,515 |
| Earth (bands) | 0.384 / 0.190 / 0.018 | 26.5 | 155 (100–200) / 11% (≤ 30%) / 25.2 (24–29) / 31.6 (30–34) | ~0 | 166 / 384 |

**Decision (rule fixed before the sweep):** every combination passes the gates (3-seed means). Interior tests + median:
0 / 1 / 1 / 1 / 2 / 2; earth-belt and earth-strain tie at 2, the smaller > 45 km share (0.298 vs 0.306) wins.
**`start_crust` 'earth' and `orogen_rule` 'belt' are the defaults since 25 Oct** (`PARAMS`; `tests/test_plates.py`: the
two default asserts, and `test_earth_start_keeps_the_field_and_takes_earths_thickness` passes 'assumed' explicitly for
its reference run). Saved runs store only their overrides and no review reads these two keys, so older runs read as
before; an older setting is re-run with `--param start_crust=assumed --param orogen_rule=collision`. pytest was not
run here (no numba / ducc0 in the device shell this session): `Interior Defaults Tests.cmd`.
Stated with the decision:
- The gate is on 3-seed means. Three earth-start seeds miss the young-ocean offset alone (earth-belt 1001 −356,
  1002 −307 m; earth-strain 1002 −367 m); all 18 runs lie at −166 to −367 m. Watch-point (twentieth pass: ocean floor
  flexed down beside margin sediment wedges); the earth start makes it a little more negative.
- belt over strain is the tie-break, 0.298 vs 0.306, well inside the seed spread (0.21–0.40 per seed). The measured
  rule (strain, GSRM) and the geometric one (belt) behave alike.

**What the sweep says:**
1. **A does what it was built for.** The three earth-start combinations pass the median (39.5–39.7 vs 40.1–40.5), the
   < 30 km share rises (0.090–0.096 vs 0.063–0.089), p10 falls 1–3 km, thin crust per km of edge rises (104–128 vs
   94–101), the 150–200 km bin comes down ~1 km. It replaces an assumed value with a measured one (audit A → M).
2. **B marks a lot and thins nothing.** Young-orogen erodibility covers 0.40–0.47 of land (Bird's orogens 0.16; the
   collision rule 0.016–0.030) and 0.75–0.87 of land > 2 km (0.76). But > 45 km rises (0.283 → 0.296–0.307), land
   above 2 km does not fall (0.340 → 0.342–0.372) and the export rises by 0.6–1.6 km³/yr only. The inconsistency with the
   law's calibration was real; it is not what keeps the thick crust thick and high.
3. **Why: the planet's high land is plateau.** Steepest-descent slope at 46 km on land above 2 km (ETOPO on the same
   grid, no ice; planet present day, seeds 1001–1003 of assumed-collision, earth-belt, earth-strain):

   | | p25 / p50 / p75 (m/km) | share under 5 m/km | all land p50 |
   |---|---|---|---|
   | Earth | 5.4 / 11.1 / 22.5 | 0.23 | 1.7 |
   | planet (9 runs) | 1.0–1.5 / 1.7–3.0 / 3.8–9.1 | 0.64–0.79 | 1.7–2.8 |

   A thresholded stream-power law (θ 200 at 46 km) cannot lower a surface without gradient, whatever its
   erodibility; plateaus lose height only at their edges. The planet's thick crust is spread into broad 45–55 km
   plateaus by the collapse diffusivity (e-fold per 6 km above `collapse_thick_m` 45 km; twenty-third pass, item 3)
   and stands at its Airy height (> 45 km land p50 ~3.0 km, Earth 1.2). Earth's land above 2 km is steep belts; its
   thick crust outside orogens is old and stands at 466 m.
4. The inland profile is still +3.6–5.1 km in its worst bin (expected ~2): with belt marks, 50–90% of the 150–1000 km
   bins is crust marked as orogen at 45–50 km, the same thick tail.

**Next (to agree), tests first:**
- A new candidate beside 2b and C. **E, collapse of thick crust:** on Earth, crust thickened beyond ~45–50 km
  collapses by extension that thins it while spreading it (Basin and Range, Aegean, E–W extension of Tibet), and Tibet
  is the only plateau of its size; the planet's collapse is a diffusion that keeps the volume at 45–55 km over broad
  plateaus. Proposed tests (to fix before building): > 45 km ≤ 0.13, land > 2 km ≤ 0.10, slope of land above 2 km
  (median within ±0.2 dex of Earth's 11.1 m/km, share under 5 m/km ≤ 0.35), and the width of thick belts against
  Bird's orogen widths (to be measured on Earth first, read-only).
- **2b** (short passive edge; back-arc opening, failed rifts, microcontinents) adds area and Earth's inland extended
  crust; the pure-shear pass showed wider margins also shrink the thick tail. **C** (thickness–elevation from the
  lithosphere's thermal state) lowers old thick crust but not the plateaus' thickness.
- Proposed order: measure E's Earth references first (cheap), then choose between E and 2b.

**Twenty-fifth pass, judgement for the next chat (25 Oct, night).** The user ran `Interior Defaults Tests.cmd`: no
failures (pytest's node cache was rewritten at 21:01 after the edits; `lastfailed` unchanged since 23 Oct). The new
defaults stand.

*Judgement: E comes first, built as continental extension; it takes over the failed-rift part of 2b.* Reasons, from
numbers already in hand:
1. **The largest gaps are the thick tail and the high land:** > 45 km 0.30 vs 0.086 (3.5×), land above 2 km 0.34 vs
   0.05 (6×), land p50 ~1.5 km vs 0.38 (4×); hypsometry has been the worst score in every sweep since 23 Oct. 2b's
   targets are smaller ratios (passive edge 0.4×, < 30 km 0.09 vs 0.19), and C moves elevation, not thickness.
2. **The plateau shape is the collapse law itself.** κ = κ_normal + 1000 (e^((T − 45 km)/6 km) − 1) m²/yr (cap 200):
   1,300 at 50 km, 4,300 at 55, 11,000 at 60 km (diffusion length over 20 Myr 160 / 290 / 470 km), but 30 below
   45 km (25 km). Thick crust spreads fast until it is just above 45 km and stalls: broad 45–50 km plateaus with a
   sharp foot (0.19 of the continent at 45–55 km, Earth 0.068; slope of land above 2 km median 1.7–3.0 m/km, Earth
   11.1). The values were set to hold Tibet-like ~70 km plateaus (audit: kind M-target), not from a mechanism; under
   the realism rule they are to be replaced.
3. **The area deficit is the same process seen from the other side.** Collisions consume ~0.10 of the surface in
   continental area over 250 Myr and rifting returns ~0.06 (area ledger, nineteenth pass, cloud seed); the earth-start
   continent goes from 0.377 to 0.30 (Earth 0.384). On Earth, crust thickened in orogens gives its area back by
   extension once convergence slows (Basin and Range, Aegean, the E–W rifts of Tibet) and thins to ~30 km. On the
   planet, crust flow conserves volume inside a fixed plate raster: it can move thickness around but never regain area.
   Edge collapse (seventeenth and eighteenth passes) failed because 50–75% of the > 45 km crust lies more than 500 km
   from any plate boundary, inland.
4. **The volume says area, not thinning alone.** Planet continental volume ≈ 0.30 × ~41 km against Earth's
   0.384 × ~36 km, about 0.9 of Earth's (and −10 to −14% over the run). Spreading the tail within today's area would
   raise the median (39.6, band 34–40); only extension that makes area can shrink the tail and keep the median.
5. **It also makes Earth's inland extended crust** (0.09–0.11 of the 150–1000 km bins at ~31 km, twenty-third pass),
   the failed-rift part of 2b, so the inland profile test gets its first real lever.

*What E would be (to agree; nothing built):* intracontinental extension as a kinematic event, on machinery already
validated: a plate split whose halves diverge, plus `rift_rule` pure_shear (volume-conserving thinning over a zone;
the gap is new area). Trigger and path from gravitational potential energy instead of old sutures: crust above a
threshold thickness that is no longer converging (collision and margin belt inactive). Slow divergence, ending when
the zone's crust reaches an Earth post-extension thickness (~30 km in the Basin and Range and the Aegean), with no
breakup: a failed rift. Rates, onset thickness and end thickness come from published values (to be sourced at build
time, not fitted). Then the collapse diffusion is reduced to what a mechanism supports. Design questions for the
chat: a plate split vs a stretch operation inside one plate that opens gaps; zone width (the Basin and Range is
~800 km wide; `rift_width_km` is 100).

*Earth references to measure first (read-only, cheap); then the tests are fixed from them before building:*
- the bands already fixed: > 45 km ≤ 0.13, land above 2 km ≤ 0.10, area 0.35–0.42, median 34–40, the elevation
  tests, export 4–8 km³/yr;
- slope of land above 2 km at 46 km: median within ±0.2 dex of Earth's 11.1 m/km, share under 5 m/km ≤ 0.35 (read
  this pass with an ad hoc script; to become a function of `interior_probe`);
- width distribution of thick crust belts (> 45 km, CRUST1.0 at 1°) and of Bird's orogens (to measure);
- share and thickness of extended crust inland (CRUST1.0 types, `interior_probe --earth`: 0.09–0.11 at ~31 km);
- the continental area ledger by mechanism closes over the run (collision loss vs extension gain), on the user's
  three seeds.

*Order after E:* (1) the rest of 2b (back-arc basins, microcontinents, passive edge in small basins: Earth 34k km vs
the planet's 0–7k), its tests reread after E, which adds area and inland extended crust; (2) the continental volume
budget (−10 to −14% over 250 Myr, Earth ~steady; arc addition vs subduction erosion and sediment subduction, all Earth
values now; read the 50 Myr ledger blocks first); (3) C, thickness–elevation from the lithosphere's thermal state.
Independent, any time: the detail value for the meso. Watch-point: young-ocean offset (three earth-start seeds beyond
−300 m).

### Step 8, follow-up 1, twenty-sixth pass (25 Oct, late): E's Earth references measured, its tests proposed, its values sourced; design to agree

Step E (a)–(c) of the handoff, read-only. No model code changed; the device shell again had no network (no numba,
ducc0 or pytest; the reviews ran on the pure-Python fallback).

**(a) Earth references, measured** (ETOPO and CRUST1.0 on CubeSphere(217), no ice sheets; now in
`scripts/interior_probe.py`: `slope_m_km`, `edge_depth_km`, `e_refs`, printed by `--earth` and for every planet snapshot,
and as "E tests" under the 3-seed summary, not counted in the five item-2 tests). Printout for the interior sweep:
`sweep-20261024-193840/reviews/interior_probe_E.txt`.

| | Earth | planet start (−250 Myr, earth start) | planet present, 6 combinations × 3 seeds |
|---|---|---|---|
| slope of land > 2 km, p25 / p50 / p75 (m/km) | 5.4 / **11.1** / 22.5 | 1.5 / 13.3 / 52.3 (seed 1003) | 1.0–1.5 / **2.0–2.5** / 3.8–9.1 |
| share of land > 2 km under 5 m/km | **0.23** | 0.45 | **0.67–0.74** |
| slope of all land, p50 | 1.7 | 1.8 | 1.7–2.8 |
| > 45 km crust: depth from its edge p50 / p90 (km) | **92 / 322** | 96 / 511 | **247–358 / 667–955** (3-seed means) |
| > 45 km crust: pieces, largest four (Mkm²) | 79; 7.9 / 1.4 / 1.1 / 0.5 (Tibet–Pamir 47%) | 17; 4.2 / 1.5 / 0.9 / 0.7 | 20–39; three or four of 5–24 each |
| > 50 km crust: depth p50 / p90 | 105 / 368 | | 247–432 / 756–1,133 |
| Bird's orogens on continent: depth p50 / p90, medial width p50 | 267 / 794, 683 km | | |
| extended crust (CRUST1.0 types M, N, T) | 0.07 of the continent, p10 / 50 / 90 25.6 / 30.9 / 36.1 km; 0.09–0.11 of the 150–1000 km inland bins at 30.7–31.7 km | | (no planet marker yet) |

Depth is the graph distance from a node of the mask to the nearest node outside it (a one-node belt reads one spacing,
~46 km), area-weighted. Bird's orogens are deformation zones (Basin and Range, Aegean, Tibet as a whole), ~3× wider
than Earth's thick crust; the thick crust itself is the narrower measure, so the test is on it.

Reading: Earth's thick crust is narrow and broken (half of it within ~90 km of its edge, one Tibet), and its land
above 2 km is steep. The planet starts Earth-like (start slope p50 13.3, depth p50 96 km) and grows three or four
Tibet-sized flat plateaus during the run, 3× deeper from their edges than Earth's thick crust. This confirms the
twenty-fifth pass: the shape comes from what happens to thickened crust during the run, not from the start.

**(b) Tests for E, proposed (fixed before building; 3-seed means; to agree):**
- gates unchanged: the elevation tests pass, export 4–8 km³/yr;
- > 45 km ≤ 0.13 and land > 2 km ≤ 0.10 (item-2 tests, unchanged);
- slope of land > 2 km: p50 within ±0.2 dex of 11.1 (7.0–17.6 m/km), share under 5 m/km ≤ 0.35;
- > 45 km depth from its edge: p50 within ±0.2 dex of 92 km (58–146), p90 within ±0.2 dex of 322 km (203–510);
- continental area 0.35–0.42, crust median 34–40 km (unchanged);
- extended crust inland: share of the 150–1000 km bins thinned by extension (a new planet marker: crust that E thinned,
  not breakup margins) 0.05–0.15, its median 29–33 km (Earth 0.09–0.11 at 30.7–31.7);
- area ledger closes: continental area change by mechanism (collision, rifting, E, arcs, other) sums to the net
  change within 0.002 of the surface over 250 Myr (instrument `PlateSim`, as `scripts/area_ledger_probe.py` did ad hoc);
- guards: the passing tests stay passing (median, young orogen on land > 2 km, abrupt margins, edge T), continental
  volume change no worse than −14%, young-ocean offset reported.
- Decision rule (proposed): E is adopted if it passes the gates and at least two more of the four E tests (slope p50,
  slope share, depth p50, depth p90) than the baseline (0 of 4 in all six combinations), with no guard failing.

**(c) Values from the literature (not fitted):**
- *Onset thickness.* The Basin and Range crust was thickened to at least 45–50 km by Cretaceous thrusting and thinned
  to ~30 km by > 50% extension (Colgan et al. 2006, Am. J. Sci.); southern Arizona and Nevada reached ~55–60 km in the
  Late Cretaceous, ~28 km today (Jepson et al. 2022, JGR). The Aegean: 40–50 km under mainland Greece and Turkey taken as
  the pre-extension crust, ~25 km under the sea (Zhu et al. 2006, JGR). Dewey (1988): collapse by outward flow once a
  plateau stands above ~3 km. → onset at a zone-mean crust of ~45–50 km.
- *Trigger is not thickness alone.* Basin and Range faulting began tens of Myr after thickening ended, after the
  flat Farallon slab was removed and the plate boundary changed (Lund Snee 2022). Rey, Vanderhaeghe & Teyssier (2001)
  separate two modes: with the surrounding lithosphere fixed, collapse moves potential energy toward the lowlands by
  lower-crust flow and upper-crust sliding (the planet's collapse diffusion); with a free boundary, the thick crust
  thins homogeneously (extension, new area). The Aegean extends by gravitational collapse that slab retreat allowed
  after ~30 Ma (Gautier et al. 1999; Jolivet). → trigger: thick crust no longer held by convergence (no active
  collision or margin belt across the zone).
- *End thickness.* ~28–31 km in the Basin and Range, 25–28 km in the Aegean (refs above). → ~30 km, and no breakup.
- *Rate and width.* The Basin and Range is ~750 km wide at 40°N; it opened at ~20 mm/yr from 10–15 Ma, slowing to
  ~12 mm/yr (GSA Today 2000, BARGEN); strain rates today ~10–20 nanostrain/yr (Bennett et al. 2003). The Aegean extends
  at ~3 cm/yr with rollback (Sodoudi et al. 2006), 30–50 nanostrain/yr (Zhu et al. 2006), stretched by a factor ~2.
  → strain rate ~1–5 × 10⁻⁸ /yr; 50 → 30 km (ln 1.67 = 0.51) at 2 × 10⁻⁸ /yr takes ~25 Myr, as the Basin and Range
  (~15–35 Myr). Zone widths 500–800 km.
- *The diffusion is the fixed-boundary mode and has a mechanism.* κ = ρc g (1 − ρc/ρm) h³ / (3η) for a 15 km lower-crust
  channel gives 1,400 m²/yr at η = 10²⁰ Pa s and 14,000 at 10¹⁹, the range of the planet's collapse κ at 50–60 km
  (hot thickened crust; Clark & Royden 2000 infer ~10¹⁸ in Tibet's channel, from memory, to confirm at build time). So the handoff's "reduce the collapse
  diffusion" is not obviously right: what is missing is the free-boundary mode. The assumed parts are the 45 km
  threshold and the 6 km e-fold (a viscosity law in disguise); whether to keep, derive or reduce them is a design question.

**(d) Design, to agree with the user (nothing built).** Inside one rigid plate raster, inland crust cannot gain area
without resampling (forbidden, sect. 3), so a free boundary has to be a plate boundary:
1. *Plate split (recommended):* when a thick zone (zone-mean > ~45–50 km over a connected > 45 km piece) has had no
   active collision or margin belt for a set time, split the plate along the zone's long axis (the existing `split`,
   path through the zone instead of a Voronoi line), halves diverge at a strain-rate-derived speed (zone width ×
   2 × 10⁻⁸ /yr, ~1–1.5 cm/yr), `rift_rule` pure_shear with the stretching half-width set to the zone's half-width
   instead of `rift_width_km` 100, no breakup (the gap fills from the stretched zone; stop at ~30 km zone mean), then
   stop the relative motion and weld the halves (a merge without a collision contact). Validated parts: split,
   pure-shear thinning with exact volume, merge. New parts: the trigger, the path, the stop and the weld.
2. *Toward an existing free boundary:* the piece between the thick zone and a nearby retreating trench or passive
   margin moves away (Aegean style). Kinematically the same split; only the path differs. Covers fewer cases (half
   of the > 45 km crust is > 500 km from any boundary).
3. *Edge collapse* (seventeenth/eighteenth passes, `collapse_edge`): already tested, reaches only coastal thick crust.
Questions: split (1) vs (1)+(2); whether the collapse diffusion is kept as the fixed-boundary mode (with its law
written as channel flow), reduced, or left alone for the first sweep; zone width from the thick piece itself vs a
fixed 300–400 km half-width; whether the planet's > 45 km threshold for the trigger is 45 or 50 km (sweep both?).

### Step 8, follow-up 1, twenty-seventh pass (25 Oct, late): E built (default off); the extension sweep is ready

**Decisions (the user left the open questions to my judgement):** (1) a plate split through the thick zone only (the
Aegean-style split toward an existing free boundary waits for the sweep); (2) the collapse diffusion stays as it is,
as the fixed-boundary mode, so the sweep measures E alone (rewriting it as channel flow is a later, separate change);
(3) onset 45 and 50 km both swept; (4) the tests of the twenty-sixth pass, with one change: the area ledger does not
close to 0.002 even without E (below), so it becomes a guard (E adds no unexplained area).

**Built (`planet_layer/plates.py`; defaults unchanged, `extension` False):**
- PARAMS `extension`, `ext_onset_thick_m` 45 km, `ext_end_thick_m` 30 km, `ext_strain_rate_per_yr` 2e-8,
  `ext_min_area_mkm2` 0.5, `ext_max_half_width_km` 400, `ext_max_myr` 100 (a guard only); sources in the comment.
- Trigger (`_extension_events`, every step after merges): connected pieces of continental crust thicker than the
  onset, at least 0.5 Mkm², with no collision boundary node or subduction-belt node in or next to them (convergence no
  longer holds them; a merged suture is inside one plate, so crust thickened by a finished collision qualifies). One
  running event per plate.
- Start (`_start_extension`): the cut is the great circle through the piece's centroid along its long axis (largest
  eigenvector of its tangent-plane spread); the smaller side of the plate becomes the new plate (`split`, no
  resampling). Zone half-width W = the piece's area-weighted p90 distance from the cut, within [`rift_width_km`,
  400 km]; relative speed v = 2e-8 /yr × 2W (0.4–1.6 cm/yr). The piece is tracked as raster nodes of both halves.
- Motion (`_extension_motion`, after `_update_motion`): the halves' mean motion follows the usual forces; their
  relative rotation is held at v (pole 90° from both centroids, as `split` sets it), since the speed relaxation
  (τ 10 / 20 Myr) would otherwise stop the divergence within one event.
- Gaps (`_rift_pure_shear`): gaps between the two halves within W of the tracked piece are the event's: pure shear
  of the two halves' continental crust within W (same kernel, exact volume, same breakup rule). The rest of the cut
  rifts as before (W = `rift_width_km`). The crust E thins or makes gets a time mark: new raster field `ext`, carried
  by split, merge, new nodes and hole filling; snapshots have `extension_age_myr`.
- End: when the area-weighted mean thickness of the crust marked since the event began is ≤ 30 km, the halves weld
  (`merge`, Euler vectors averaged by area). Event records: `extension`, `extension end` (duration, zone thickness and
  area, guard flag), `extension lost` (a half merged away).
- Area ledger (`area_ledger`, m²): collision, subduction, rift, extension, arc, merge overlap, holes; `stats.json`
  gets `continental_area_change_frac`, `area_ledger_frac`, `extension_starts`, `extension_ends`.
- `scripts/interior_probe.py`: extended-by-E share of the 150–1000 km inland bins and its median, and the area
  ledger with its residual and the E event summary per combination.
- Tests (`tests/test_plates.py`): `test_extension_is_off_by_default_and_inert_without_thick_crust` (an unreachable
  onset gives the default run exactly), `test_extension_splits_unheld_thick_crust_thins_it_and_welds` (n = 48, onset
  40 km: events start, the halves diverge at the event's rate, gaps fill, events weld at ≤ 30 km, marked crust is
  continental, volume ledger within 2e-3).

**Checked here** (device shell: no numba, ducc0 or network; flexure stubbed by local Airy and the random fields by a
cosine sum, numba kernels in pure Python; plumbing only): both new tests pass; `test_short_run_invariants`,
`test_erosion_lowers_belts_and_volume_ledger_closes`, the split, merge and pure-shear run tests pass. Smoke run n = 48,
seed 3, 60 Myr, onset 40 km: 12 events welded after 1–24 Myr (median ~12; expected ~17 from 40 to 30 km at 2e-8),
volume ledger error 1e-4, extended crust median 30.5 km.
- The first version counted every gap of the cut as the event's: thin rim crust far from the piece pulled the zone
  mean below 30 km within 1–2 Myr. Fixed by tracking the piece and restricting the event's gaps and stretching to
  within W of it.
- Area ledger: net change minus the counted mechanisms leaves −0.0065 of the surface over 60 Myr at n = 48 without
  E and −0.0094 with it (nearest-node bookkeeping between rasters and world; collisions are booked when the hidden
  crust is consumed). The 0.002 closure test is replaced by the guard: residual with E within 0.005 of the control's.

**Not done here:** pytest on the user's machine (ducc0 and numba), and any 46 km run.

**The sweep (`Extension Sweep.cmd`):** the planet-layer tests, then `extension`, `ext_onset_thick_m` ∈ {False 45 km
(control), True 45 km, True 50 km} × seeds 1001–1003 (9 runs) on the current baseline, then `elevation_check`,
`crust_review`, `margin_width`, `interior_probe` (all `--latest`). The control should reproduce
`sweep-20261024-193840/start_crust-earth_orogen_rule-belt` digit for digit (deterministic runs; nothing in the default
path changed); a difference means the build touched the default path.
**Decision rule (fixed before the sweep):** gates = the elevation tests pass and the export is 4–8 km³/yr (3-seed
means). E is adopted if it passes the gates and at least 2 of the 4 E tests (slope of land above 2 km p50 7.0–17.6
m/km; share under 5 m/km ≤ 0.35; > 45 km depth from its edge p50 58–146 km, p90 203–510 km; the control passes 0)
with no guard failing (crust median 34–40 km, young orogen on land above 2 km ≥ 0.5, abrupt margins ≤ 30%, edge T
24–29 km, continental volume no worse than −14%, area-ledger residual within 0.005 of the control's). Between onsets
that both qualify: more E tests, ties to the smaller > 45 km share. Read as they come: > 45 km (≤ 0.13), land above 2
km (≤ 0.10), area (0.35–0.42), extended inland share (0.05–0.15 at 29–33 km), the inland profile, the event counts
and durations, and the area ledger (extension gain against collision loss).

**What to watch:** the cut runs across the whole plate, so outside the zone it is an ordinary slow rift
(0.4–1.6 cm/yr) and adds rift area elsewhere; if that shows (rift area up, breakup oceans in odd places), the next
step is a cut that ends at the zone (a microplate bounded by transforms) or option 2 (split toward a free boundary).
If E fires rarely, the "held" test (adjacency to a collision or belt) is the first suspect: 49% of the > 45 km crust
lies within 500 km of a boundary.

### Step 8, follow-up 1, twenty-eighth pass (25 Oct, morning): the extension sweep; E not adopted, the "held" rule shields the plateaus

**Your sweep** (`sweep-20261024-215311`, 3 × 3 seeds; reviews rerun in the device shell on the pure-Python fallback and
saved in `<sweep>/reviews/`: `elevation_check`, `crust_review`, `margin_width`, `interior_probe`).
- **Control reproduces the interior sweep exactly:** every value of `sweep-20261024-193840/start_crust-earth_orogen_rule-belt`
  `stats.json` is identical; the new run only adds keys (`extension*`, `area_ledger_frac`, `continental_area_change_frac`).
  The build did not touch the default path.

| 3-seed means | control | E, onset 45 km | E, onset 50 km | band / Earth |
|---|---|---|---|---|
| **E tests** (slope p50 / under 5 m/km / depth p50 / p90) | 2.4 / 0.68 / 261 / 721: 0 of 4 | 3.0 / 0.64 / 254 / 653: **0 of 4** | 3.1 / 0.62 / 193 / 553: **0 of 4** | 7.0–17.6 / ≤ 0.35 / 58–146 / 203–510 |
| > 45 km (label-free) / land > 2 km | 0.298 / 0.342 | 0.268 / 0.341 | 0.234 / 0.302 | ≤ 0.13 / ≤ 0.10 (Earth 0.086 / 0.05) |
| < 30 km / median | 0.090 / 39.6 | 0.162 / 38.4 | 0.147 / 38.6 | Earth 0.190 / 37.0 |
| continental area (label-free) | 0.303 | 0.304 | 0.313 | 0.35–0.42 (Earth 0.384) |
| inland profile, worst | +3.8 km | +2.1 | +2.9 | ±1.5 |
| margins at 1° (thin per km / abrupt / edge T / inner T) | 117 / 16% / 26.1 / 34.5: fails | 144 / 13% / 25.1 / 32.7: passes | 137 / 12% / 24.9 / 33.1: passes | 100–200 / ≤ 30% / 24–29 / 30–34 |
| extended by E, inland bins | — | 0.194 at 33.0 km | 0.138 at 37.4 km | 0.05–0.15 at 29–33 |
| young-ocean offset (gate ±300 m) | −278 m | −252 m | **−301 m** (seed 1001 −451) | |
| continental volume (guard ≥ −14%) | −11.7% | **−14.3%** | −12.5% | ~0 |
| export | 5.9 | 7.7 | 6.6 | 4–8 km³/yr |
| area ledger: collision / rift / E / residual | −0.113 / +0.057 / 0 / −0.008 | −0.146 / +0.086 / +0.020 / −0.014 (**guard: 0.006 off**) | −0.124 / +0.070 / +0.016 / −0.011 | |
| E events per run (ended), median duration | — | 10.7, 25 Myr | 5.7, 35 Myr | Basin and Range ~15–35 Myr |

**Decision by the rule: E is not adopted** (0 of 4 E tests at both onsets; at 45 km two guards fail, at 50 km the
young-ocean gate misses by 1 m). `extension` stays False. E moves the crust-thickness numbers the right way (> 45 km
−0.03 to −0.06, < 30 km +0.06, the margin review now passes, the inland profile closer), but not the shape of the high
land, which the E tests measure.

**Why (read before changing anything):**
1. **The mechanics work.** Events start, diverge at 0.6–1.6 cm/yr, fill by pure shear and weld at 30 km after
   1–72 Myr (median 25 / 35 Myr); none ended on the 100 Myr guard.
2. **The "held" rule shields almost all thick crust.** Probe on the control's snapshots (held = in or next to a
   collision boundary node, or within ~6 nodes of a subduction boundary on the same plate as a stand-in for the belt,
   which snapshots do not store): 81–100% of the > 45 km area lies in pieces that touch a holding boundary somewhere,
   at −200, −100, −50 Myr and today, and the largest plateau (6–17 Mkm²) is held in every seed at every time. Yet
   50–76% of that crust lies > 300 km from the holding boundary (32–65% > 500 km, 4–38% > 1,000 km). A piece is held
   as a whole if any node of it touches a collision. So E only reaches small unheld pieces (0.5–3 Mkm²; two of
   10–13 Mkm² in seed 1002), and cuts a single ≤ 800 km band through the large ones, whose events end when that band's
   marked crust reaches 30 km.
3. **E fires on the start's crust.** 8 of 36 events at 45 km start at −249/−248 Myr on the earth start's 45–48 km
   crust (thick old crust that has not been thickened in the run). Earth's thick Precambrian crust (38–50 km) does not
   extend: it is cold and strong.
4. **The cut across the whole plate costs as much area as E makes.** At 45 km: E +0.020 of the surface, but rifting
   elsewhere along the cut +0.029 over the control, and collisions −0.033 (more plates, more collisions); net area
   unchanged (0.303 → 0.304).
5. Side finding: `orogeny_age_myr` is 0 on all > 45 km crust today (p10–p90) in every run, so it cannot serve as a
   thermal clock (it is reset each step on thick crust).

**Earth on point 2:** Tibet's interior extends east–west (north–south grabens, ~10–20 nanostrain/yr) while India
still converges: convergence does not stop extension, it sets its direction (orogen-parallel stretching). The Basin
and Range and the Aegean extend perpendicular to the former belt once the boundary is free. So "held" decides the
direction, not whether extension happens.

**Proposal E2 (to agree; tests and rule of the twenty-seventh pass unchanged):**
- (a) No held exemption. A piece touching a collision or belt is cut *across* its long axis (orogen-parallel
  stretching, Tibet); a free piece along it, as now (Basin and Range).
- (b) Only hot crust extends: a new per-node clock, time since tectonic thickening (collision volume, margin
  shortening; not collapse diffusion), carried like `ext`; the trigger needs the piece's area-weighted share of crust
  thickened within the lithosphere's thermal time (~60–100 Myr, to source) above one half. This also keeps E off the
  start's crust.
- (c) The cut stays whole-plate for E2; if rift area outside the zone again matches E's own gain, the next step is a
  cut that ends at the zone (a microplate bounded by transforms).
- Design check (not a pass band): the trigger reaches ≥ 50% of the > 45 km area summed over the run.
- Sweep: control / E2 onset 45 / 50 km × 3 seeds, the four reviews, the same decision rule and guards.

**Agreed by the user (25 Oct, 10:33).** Build notes (where in the code, new PARAMS `ext_hot_myr` and `ext_held_rule`,
the clock `thickening_age_myr`, tests, trigger reach in `stats.json`, `Extension2 Sweep.cmd`) are in
`docs/next-chat-prompt.md`, item 1.

### Step 8, follow-up 1, twenty-ninth pass (25 Oct, midday): E2 built (default off); the E2 sweep is ready

**What E2 is** (agreed 25 Oct, 10:33; twenty-eighth pass): E with three changes, nothing else. (a) No held exemption:
a thick piece touching a collision boundary or subduction belt is cut *across* its long axis (orogen-parallel
stretching, Tibet), a free piece along it (Basin and Range, Aegean). (b) Only hot crust extends: more than half of the
piece (area-weighted) thickened tectonically within `ext_hot_myr`. (c) The cut stays whole-plate.

**`ext_hot_myr` = 100 Myr, sourced before building.** Sonder, England, Wernicke & Christiansen (1987, *A physical
model for Cenozoic extension of western North America*, Geol. Soc. Spec. Publ. 28): the vertically averaged strength
of thickened lithosphere is too great for significant extension until the Moho exceeds ~700 °C (± 100), so extension
follows thickening after a delay set by the initial geotherm and the strain: none if the initial Moho was above
~700 °C, up to ~100 Myr if it was below 450 °C. Observed gaps between compression and extension: ~0 (Pacific
Northwest), 20–40 Myr (northern Great Basin), ~70 Myr (southern Great Basin). 100 Myr is that model's upper bound and
covers every observed case; crust thickened longer ago has had time to heat and, if it did not extend, to cool and
strengthen. For scale, McKenzie's (1978) lithosphere time constant is ~63 Myr (125 km plate). Not fitted; written in
the PARAMS comment.

**Built (`planet_layer/plates.py`; `extension` stays False, so the defaults are unchanged):**
- PARAMS `ext_held_rule` 'direction' (E2; 'exempt' = E as built) and `ext_hot_myr` 100 (0 switches the hot rule off;
  E as built = 'exempt' + 0).
- **The thickening clock**: new plate-raster field `thk` (Myr, NaN = never thickened in the run), carried exactly like
  `ext` (world field, split, both merge copies, new nodes, hole filling); snapshots have `thickening_age_myr`. Set
  each step on nodes whose tectonic thickening, divided by thickness, exceeds `orogen_strain_rate` (3.5e-9 /yr, the
  measured GSRM threshold of deforming crust: no new number). Tectonic thickening = the subduction term (arc and
  continental-margin thickening, margin shortening, net of subduction erosion) + collision volume at the receiving
  node; not collapse diffusion, not rifting, not erosion, not mantle volcanism. The start's crust has no time (cold).
- **Trigger** (`_extension_events`): pieces as before (> onset, connected, ≥ `ext_min_area_mkm2`); eligible if more
  than half hot (with 'direction') and, with 'exempt', unheld. A held eligible piece is cut across its long axis
  (`_start_extension(..., across=True)` takes the tangent-plane covariance's second eigenvector, the short axis).
  Event records add `across`, `hot_share` (and `cut_normal` in memory).
- **Trigger reach**: `PlateSim.ext_reach` sums, over steps, the > onset area: all, in eligible pieces, in hot pieces,
  in held pieces, hot nodes. `stats.json` `present.extension_trigger` holds them as fractions of the first;
  `interior_probe` prints "E trigger … eligible" with the design check (≥ 0.5, not a pass band), the starts per run,
  how many were cut across, and the median hot share.
- `scripts/simulate_plates.py` (stats), `scripts/interior_probe.py` (printout).

**Tests** (`tests/test_plates.py`): E's run test is pinned to E as built (`ext_held_rule` 'exempt', `ext_hot_myr` 0),
so E stays reproducible. New: `test_held_piece_is_cut_across_its_long_axis` (an elongated piece in the largest plate:
across, the cut's pole lies along the long axis, |m·e| > 0.9; along, < 0.3), `test_crust_without_a_thickening_time_
never_triggers_extension` (the same state with the clock cleared starts no event with the hot rule and does without it;
in a run every event's hot share is > 0.5), `test_thickening_clock_is_not_set_by_collapse_diffusion` (subduction
thickening off and no crust consumed: strong collapse diffusion thickens crust faster than the threshold, yet no node
gets a time; with margin shortening on, the clock is set exactly where marked).

**Checked here** (device shell: no numba, ducc0, pytest or network; local-Airy flexure and cosine-sum random fields
stubbed, numba kernels in pure Python; plumbing only): the three new tests, E's two, `test_short_run_invariants`, the
split, merge, erosion-ledger and pure-shear run tests pass. **Default path untouched:** the pre-edit `plates.py` was
rebuilt by inverting the edit (md5 matches the pre-edit file) and run beside the new one at n = 48: world fields, owner,
plate rasters and rotations identical every step (12 steps on PARAMS defaults and on the sweep baseline; 32 steps on
the baseline, including a split and a merge). The sweep's control must still equal `sweep-20261024-215311` digit for
digit on your machine. Smoke (n = 48, onset 40 km, 40 Myr; not for judging): two events, both on held pieces, cut
across, hot shares 0.59 and 0.50+; trigger reach 0.10 (in hot pieces 0.11, in held pieces 0.73, hot nodes 0.20). Early
in a run the start's (cold) crust dominates the thick area, so this number says nothing yet about 46 km.

**Risk to read first in the sweep (the design check).** Collision volume lands on the contact nodes and margin
shortening on the belt; the collapse diffusion then spreads that crust inland (at 55 km κ ≈ 4,300 m²/yr, √(κt) ≈ 65 km
in 1 Myr, ~650 km in 100 Myr), and crust thickened by that flow gets no time, as agreed. 50–76% of the > 45 km crust
lies > 300 km from the holding boundary (twenty-eighth pass). If the plateau interiors are mostly diffused-in crust,
their hot share stays below one half and the trigger reach falls short of 0.5. Physically, crust that flowed in is hot
(it flows because it is hot): the clock should move with the crust. **Remedy, to agree only if the check fails (not
built):** the clock carried by the collapse flow: a node thickened by the diffusion faster than `orogen_strain_rate`
takes the most recent time among its thicker neighbours (the volume's source), never the present time, so start crust
spreading into start crust stays cold. Tests first: start crust that flows never gains a time; crust flowing from a
freshly marked belt carries the belt's time; the clock is never set to the present by the diffusion alone.

**The sweep (`Extension2 Sweep.cmd`):** the planet-layer tests, then `extension`, `ext_onset_thick_m` ∈ {False 45 km
(control), True 45 km, True 50 km} × seeds 1001–1003 on the extension sweep's baseline, with `ext_held_rule`
direction and `ext_hot_myr` 100 passed explicitly (recorded in `sweep.json`), then `elevation_check`, `crust_review`,
`margin_width`, `interior_probe` (all `--latest`). Decision rule and guards unchanged from the twenty-seventh pass.
Read as they come: the E trigger line (eligible ≥ 0.5; across vs along; hot share), > 45 km and land above 2 km, the
area ledger (rift area outside the zone against E's own gain: if they match again, the next step is the cut that ends
at the zone, change (c)), the young-ocean offset (watch-point), and the event durations.

### Step 8, follow-up 1, thirtieth pass (25 Oct, afternoon): the E2 sweep; E2 not adopted, the clock does not follow the crust and the cut rifts elsewhere; E3 to agree

**Your sweep** (`sweep-20261025-111119`, 3 × 3 seeds; planet-layer tests 53 pass; reviews saved by the script in
`<sweep>/reviews/`). **Control identical** to `sweep-20261024-215311/extension-False...`: all 24–26k values of each seed's
`stats.json` equal; only the runtime and the new keys (`params.ext_*`, `present.extension_trigger`, series entries)
differ. The E2 build did not touch the default path.

| 3-seed means | control | E2, onset 45 km | E2, onset 50 km | band / Earth |
|---|---|---|---|---|
| **E tests** (slope p50 / under 5 m/km / depth p50 / p90) | 2.4 / 0.68 / 261 / 721: 0 of 4 | 2.6 / 0.66 / 268 / 719: **0 of 4** | 2.8 / 0.63 / 241 / 663: **0 of 4** | 7.0–17.6 / ≤ 0.35 / 58–146 / 203–510 |
| **E trigger: eligible** (design check ≥ 0.5) | — | **0.01** | **0.04** | |
| in hot pieces / in held pieces / hot nodes | — | 0.03 / 0.88 / 0.22 | 0.06 / 0.81 / 0.25 | |
| starts per run (all across), median hot share | — | 7.7, 0.60 | 10.7, 0.61 | |
| > 45 km (label-free) / land > 2 km | 0.298 / 0.342 | 0.253 / 0.324 | 0.289 / 0.354 | ≤ 0.13 / ≤ 0.10 |
| < 30 km / median | 0.090 / 39.6 | 0.123 / 38.9 | 0.145 / 39.7 | Earth 0.190 / 37.0 |
| continental area (label-free) | 0.303 | 0.305 | 0.303 | 0.35–0.42 |
| inland profile, worst | +3.8 km | +3.8 | +4.8 | ±1.5 |
| margins at 1° (thin per km / abrupt / edge T / inner T) | 117 / 16% / 26.1 / 34.5: fails | 130 / 14% / 25.4 / 32.9: passes | 118 / 15% / 25.4 / 33.4: passes | 100–200 / ≤ 30% / 24–29 / 30–34 |
| extended by E, inland bins | — | 0.037 at 41.3 km | 0.033 at 44.0 km | 0.05–0.15 at 29–33 |
| young-ocean offset (gate ±300 m) | −278 m | −248 m | −286 m | |
| continental volume (guard ≥ −14%) | −11.7% | −12.6% | −11.9% | ~0 |
| export | 5.9 | 6.6 | 7.0 | 4–8 km³/yr |
| area ledger: collision / rift / E / residual | −0.113 / +0.057 / 0 / −0.008 | −0.119 / +0.069 / +0.003 / −0.009 | −0.135 / +0.087 / +0.006 / −0.013 (guard: 0.005 off, on the edge) | |
| E events ended per run, median duration | — | 4.3, 24 Myr (1 on the guard) | 6.7, 39 Myr (2) | Basin and Range ~15–35 Myr |

Young-ocean offset per seed beyond −300 m: control 1001 −356, 1002 −307; E2 45 km 1003 −376; E2 50 km 1001 −307,
1003 −337 (watch-point, unchanged: the means pass).

**Decision by the rule: E2 is not adopted** (0 of 4 E tests at both onsets; the design check fails, 0.01 / 0.04
against ≥ 0.5). `extension` stays False. The gates and the other guards pass.

**Why (read before changing anything):**
1. **The design check fails as the twenty-ninth pass's risk said: the clock does not follow the crust.** Probe on the
   snapshots (control and E2 45 km, 3 seeds, at 0, −12, −24, −36, −120 and −240 Myr; flag continent, crust > 45 km):
   carrying a time within 100 Myr ("hot") 0.11–0.29 of the area, thickened more than 100 Myr ago 0.00–0.10, **never
   timed 0.64–0.89**. Yet that crust is the run's own: today 98% of the label-free > 45 km crust (E2 45 km, seed 1001:
   0.245 of 0.250) went through a collision or belt during the run (`interior_probe` provenance "collided"), and the
   start's (cold, correctly untimed) thick crust is 0.045–0.07 of the continent at −250 Myr and ~0 today. Collision
   volume and margin shortening land on contact and belt nodes; the collapse diffusion spreads them inland
   (√(κt) ≈ 65 km per Myr at 55 km); the crust it spreads gets no time, as built. Crust thickened more than 100 Myr
   ago is a minor share, so a clock that moves with the crust could lift the hot share to at most ~0.8–0.9 (hot +
   never; an upper bound, since not all untimed crust is flow-fed within 100 Myr).
2. **Even at 1–4% reach, E2 moves the crust the right way at 45 km** (> 45 km 0.298 → 0.253, < 30 km 0.090 → 0.123,
   margin review passes), because each event runs 24–39 Myr; it does not change the shape of the high land, which
   the E tests read (slope p50 2.4 → 2.6 m/km).
3. **The whole-plate cut rifts elsewhere, again.** Rift area over the control: E2 +0.012 (45 km) / +0.030 (50 km)
   against E2's own extension area +0.003 / +0.006. Per seed, rift area is above the control in 11 of 12 extension
   runs over both sweeps (E: +0.043, +0.025, +0.017 / +0.013, +0.020, +0.007; E2: +0.020, +0.028, −0.011 / +0.048,
   +0.031, +0.011), and collisions consume more (more plates). The twenty-seventh / twenty-ninth pass condition for
   change (c) is met: the next cut ends at the zone.
4. The mechanics work: all 23 / 32 starts were held pieces cut across (orogen-parallel), median hot share 0.60–0.61,
   events weld at 30 km after a median 24 / 39 Myr, 1–2 per run on the 100 Myr guard.

**Proposal E3 (to agree; E2 with two changes, nothing else; decision rule and guards unchanged):**
- **(b′) The clock carried by the collapse flow** (the twenty-ninth pass's remedy). In each substep of the crust
  diffusion (`step`, the `diffuse_variable` loop; also `edge_spread` if `collapse_edge` is on), a continental node
  whose thickness gain from the flow, divided by its thickness and the substep, exceeds `orogen_strain_rate` takes the
  most recent time among its neighbours that were thicker before the substep and carry a time (the volume's source),
  if that is more recent than its own. The flow never sets the present time; start crust spreading into start crust
  stays untimed. No new number (the tectonic marking's threshold). Basis: crust flows in the collapse diffusion
  because it is hot and weak (κ at 50–60 km ≈ a 15 km channel at 10¹⁹–10²⁰ Pa s, twenty-sixth pass), so its thermal
  age is its source's. PARAMS `ext_clock_flow` (True = E3; False = E2 as built).
- **(c′) The cut ends at the zone: a microplate between two transforms.** The relative pole is the point u on the
  cut's great circle 90° from the piece's centroid c (rotation about u moves c straight across the cut). The moving
  part is the plate's nodes on one side of the cut inside the band |asin(x·u)|·R ≤ L, L = the piece's p90 distance from
  c along u (the zone's half-length); the band's edges are small circles about u, i.e. flow lines of the relative
  motion, so they are transforms (no gap, no overlap). The microplate runs from the cut to the plate's edge on that
  side; the smaller band part moves, as now. `split` gets an optional pole (default: unchanged). The cut then opens
  only inside the band. Speed as now (`ext_strain_rate_per_yr` × 2W at c). The event records the boundary type where
  the band leaves the plate (if the microplates often push into collisions, the side facing a free boundary, Rey et
  al. 2001, is the next step; not built). PARAMS `ext_cut` ('band' = E3; 'whole' = E2 as built).
- **Tests, fixed before building** (`tests/test_plates.py`): (b′1) start crust that flows never gains a time (thick
  start, no tectonic thickening, strong collapse: `thk` stays all NaN with `ext_clock_flow` True); (b′2) crust flowing
  from a belt marked at t₁ carries t₁ (tectonics off after t₁, diffusion on: the timed area grows, every new time is
  t₁); (b′3) after any step, nodes timed at the present are a subset of the tectonically marked nodes; (c′1) the
  microplate lies inside the band on one side of the cut, and the parent keeps every node outside the band; (c′2) the
  relative Euler vector is parallel to u (|cos| > 0.99) and the relative velocity on the band's edges is tangent to
  them (normal part < 5% of the speed); (c′3) in a short run with one band event and no other splits, new rift crust
  appears only inside the band; (d) `ext_clock_flow` False with `ext_cut` 'whole' reproduces E2 step by step at n = 48.
  E's two tests and E2's three still pass.
- **Design checks (not pass bands):** trigger reach (eligible) ≥ 0.5; rift area over the control no more than E3's own
  extension area (3-seed means).
- **Pre-check before your sweep:** one 250 Myr run at n = 217 in my workspace (seed 1001, onset 45 km), for the two
  design checks only (its trajectory differs from your machine's); if the reach is still below 0.5, stop and look.
- **Sweep (`Extension3 Sweep.cmd`):** planet-layer tests; control / E3 45 km / E3 50 km / clock only (`ext_cut`
  whole) 45 km × seeds 1001–1003 = 12 runs on the E2 baseline; the four reviews. The clock-only combination separates
  (b′) from (c′) and is a candidate under the same rule. Control must equal this sweep's control digit for digit.

### Step 8, follow-up 1, thirty-first pass (25 Oct, afternoon): E3 built (default off); the pre-check found the pair-motion flaw; fixed; the E3 sweep is ready

**Agreed by the user (25 Oct, 11:53):** E3 as proposed in the thirtieth pass, both changes.

**Built (`planet_layer/plates.py`; `extension` stays False, so the defaults are unchanged):**
- PARAMS `ext_clock_flow` True (E3; False = E2 as built) and `ext_cut` 'band' (E3; 'whole' = E2 as built), with the
  reasons in the PARAMS comment. E as built = `ext_held_rule` exempt + `ext_hot_myr` 0 + `ext_clock_flow` False +
  `ext_cut` whole; E2 = direction + 100 + False + whole.
- **(b′) `carry_clock`** (numba kernel): over one substep of the crust diffusion, a node whose flow gain / thickness /
  substep exceeds `orogen_strain_rate` takes the most recent time among its axis neighbours that were thicker before
  the substep and carry a time, if more recent than its own (NaN takes it). Called after every `diffuse_variable`
  substep in `step` (and after `edge_spread` with `collapse_edge`, targets then include the oceanic nodes it fills),
  only when `extension` and `ext_clock_flow` are on. The carried times go into the world clock and the plate rasters
  (`thk_carried`, max per raster node) before the tectonic marking, which then sets the present time where it applies.
- **(c′) band cut** (`_start_extension`): u = the point on the cut's great circle 90° from the piece's centroid (the
  eigenvector that spans the cut with the centroid); L = the piece's p90 |distance along u| from the centroid, at least
  `rift_width_km`; the candidate parts are the plate's nodes inside |asin(x·u)|·R ≤ L on each side of the cut,
  restricted to the components that touch the piece; the smaller moves (a side with < 4 nodes is skipped); the pole is
  ±u, signed so the part moves away from the cut. `split(..., pole=None)` takes an imposed pole (default unchanged).
  The event stores `pole`, `pole_body` (in the parent's frame), `L`, `cut`; `_extension_motion` re-imposes the carried
  pole every step (E/E2 events still use the centroids). The event also records where the microplate meets other
  plates: `far_nodes` and `far_collision` (share of those nodes on a collision boundary at the start).
- **(e) `ext_pair_motion` 'union'** (added after the first pre-check, see below; 'mean' = as built): while an event
  runs, the halves' mean motion is `_driving` (the old per-plate body of `_update_motion`, now a method) applied to the
  pair's nodes together, relaxed from the speed and axis the plate had when it split (`base_speed`, `base_axis` in the
  event); then the halves are set to base ∓ d/2 about the event's pole as before. `_update_motion` calls `_driving`
  per plate (same arithmetic).
- `scripts/simulate_plates.py` (`extension_starts` adds `cut`, `L`, `far_nodes`, `far_collision`; `extension_lost`
  counts events ended because a half merged away or was consumed),
  `scripts/interior_probe.py` (the E trigger line adds band cuts, median L, median far-side collision share; the
  area-ledger line adds lost events per run).

**Tests** (`tests/test_plates.py`; E's run test pinned to `ext_clock_flow` False + `ext_cut` whole as well, so E stays
reproducible): `test_flow_clock_gives_no_time_to_crust_without_a_timed_source` (b′1),
`test_flow_clock_carries_the_belts_time` (b′2), `test_flow_clock_never_sets_the_present_time` (b′3),
`test_band_cut_moves_a_microplate_between_two_transforms` (c′1, c′2, for an along and an across cut, and the carried
pole over three steps), `test_band_cut_opens_no_rift_along_the_cut_beyond_the_band` (c′3),
`test_e3_switches_are_inert_without_extension` (d), `test_pair_moves_as_the_unsplit_plate_would` (e: one motion update
after a band cut gives the pair exactly the unsplit plate's motion; with 'mean' the pair is off by 8–20% after one
update). One change from the test text agreed in the thirtieth pass: c′3
first read "new rift crust only inside the band". At n = 48 and 5× Earth's strain rate the band's side edges, 5,000–
7,000 km from the cut, show 1–9 new-crust nodes 1.7–3 spacings outside the edge (the transforms' staircase on the
coarse grid), none along the cut beyond the band; the whole-plate cut opens 51–111 there. At 2e-8 /yr over 12 Myr the
band cut adds no rift area over a run without the event (8.44 vs 8.49 Mkm²), the whole cut +0.62 Mkm². So c′3 tests
what the change is for: no new crust along the cut beyond the band (within 4 spacings of the cut, more than 1.5 beyond
the edge), and some for the whole cut.

**Checked on your machine's shell** (real numba, ducc0, pytest installed there again): the 39 plate tests (7 new, E's
two, E2's three) and the other planet-layer files pass; `test_coupling`'s three need their temp folder recreated, which
that shell cannot delete, and pass from a copy that writes elsewhere (60 in all). **Default path and E2 untouched:**
the pre-edit `plates.py` run beside the new one at n = 48 for 30 steps is identical (world fields, owner, plate
rotations, rasters and clocks) on PARAMS defaults, on the sweep baseline, and on E2 at onset 40 km with the E3
switches off (`ext_clock_flow` False, `ext_cut` whole, `ext_pair_motion` mean; 7 extension starts, welds, a merge);
checked again after (e).

**Pre-check** (my workspace, seed 1001, onset 45 km, the E2 sweep's baseline, 250 Myr at n = 217; its trajectory
differs from your machine's; approximate label-free reading, total ≥ 20 km, not the review scripts):

| seed 1001 | control | E3, pair motion 'mean' | E3, 'union' | Earth / band |
|---|---|---|---|---|
| E trigger: eligible (≥ 0.5) | — | 0.82 | **0.91** | |
| rift over control / own extension area | — | +0.043 / +0.042 (on the line) | **+0.037 / +0.051** | rift ≤ own |
| plate speed, area-weighted mean −225…0 Myr | 3.7 cm/yr | **7.3** | 4.0 | Earth ~3.5–4 |
| continental volume | −11.2% | **−24.8%** | −12.0% | guard ≥ −14% |
| area ledger: collision | −0.106 | −0.232 | −0.154 | |
| continent / > 45 km / < 30 km / median | 0.303 / 0.299 / 0.068 / 39.2 | 0.252 / 0.404 / 0.159 / 42.6 | 0.317 / 0.239 / 0.145 / 37.8 | 0.384 / 0.086 / 0.190 / 37.0 |
| slope p50 / under 5 m/km (land > 2 km) | 2.0 / 0.73 | 2.5 / 0.68 | 4.3 / 0.54 | 7.0–17.6 / ≤ 0.35 |
| > 45 km depth p50 / p90 | 290 / 826 km | 352 / 1091 | 159 / 435 | 58–146 / 203–510 |
| largest > 45 km pieces | 17.4 / 11.0 / 8.5 Mkm² | 25.7 / 18.8 / 2.7 | 6.9 / 5.1 / 4.5 | 7.9 / 1.4 / 1.1 |
| plates today; events started / welded / lost | 18 | 22; 24 / 15 / 7 | 29; 33 / 6 / 21 | Bird 52 |

**The first pre-check (E3 as agreed) passed the design check and failed the run:** the trigger reached 0.82, but the
plateaus ran away (one event started on a 44 Mkm² piece at 62 km mean at −76 Myr), volume −25%, collisions consumed
twice the continent. **Cause: the pair's motion.** A plate's target speed is divided by (1 + collision length / 1500
km). Many band microplates are strips cut along a collision belt (far side on a collision: 0.96–1.0 for the first
events), so the strip takes the collision boundary; the large parent loses the collision resistance and speeds up;
`_extension_motion` set the pair's mean to the area-weighted mean of each half's own force result, which follows the
parent, and pushed the strip into the collision at the parent's unbraked speed. Area-weighted plate speed doubled
(7.3 vs 3.7 cm/yr). E and E2 show a mild version on your machine (4.6–5.4 vs 4.4–4.8 cm/yr; a whole-plate cut
splits the collision boundary between two large halves). The docstring said the mean motion "follows the usual
forces"; the forces on the pair as one plate is what it meant. Built as (e) without asking first (a fix of E3's
mechanics, not a new process; E/E2 reproducible with 'mean'). **Confirmed by the user (25 Oct, 14:08)**, with a standing note: free to add, remove
or change anything as needed; comments may not reflect the actual goal or code, so decide by the overall goal.

**The second pre-check (E3 with 'union') passes both design checks** and moves every crust number the right way on
this seed: plate speed back to 4.0, volume −12.0%, > 45 km 0.299 → 0.239, < 30 km 0.068 → 0.145, median 37.8 km,
largest plateau 6.9 Mkm² (Earth's Tibet 7.9), > 45 km depth p90 435 km (inside 203–510), p50 159 km (band to 146);
slope p50 4.3 m/km and the under-5 share 0.54 still fail. **Watch:** 21 of 33 events ended "lost": a half docked onto
another plate by the merge rule (20 docking merges vs the control's 4); a strip cut along a collision belt touches the
plate across it and docks, which is accretion of the orogen to one side (plausible) but cuts the extension short. The
extra plates (29 vs 18) also come from more rifting and subduction starts (12 / 9 vs 9 / 6). One seed: the sweep
decides.

**The sweep (`Extension3 Sweep.cmd`):** the planet-layer tests, then `extension`, `ext_onset_thick_m`, `ext_cut` ∈
{False 45 km band (control), True 45 km band, True 50 km band, True 45 km whole (clock only)} × seeds 1001–1003 on the
E2 sweep's baseline, with `ext_held_rule` direction, `ext_hot_myr` 100, `ext_clock_flow` True and `ext_pair_motion` union passed explicitly,
then the four reviews (`--latest`). Decision rule and guards unchanged (twenty-seventh pass); the clock-only
combination (whole cut with the carried clock and union motion) is a candidate under the same rule. Read first: the control equals `sweep-20261025-111119/extension-False...`
digit for digit (only new keys may differ); then the E trigger line (eligible ≥ 0.5) and the area ledger (rift over
the control ≤ the candidate's own extension area); then the decision rule. Also read: band cuts and their far-side
collision share (if the microplates often push into collisions, the side facing a free boundary is the next step),
the young-ocean offset (watch-point), event durations, lost events per run, the plate count and the
area-weighted plate speed (from `stats.json` series; control ~4.4–4.8 cm/yr).

### Step 8, follow-up 1, thirty-second pass (25 Oct, night): the E3 sweep; E3 not adopted (1 of 4 E tests); the design works, the events do not finish; E4 to agree

**Your sweep** (`sweep-20261025-141028`, 4 × 3 seeds; planet-layer tests 60 pass; reviews in `<sweep>/reviews/`).
**Control identical** to `sweep-20261025-111119/extension-False...`: every common `stats.json` value of each seed equal
(25–27k), only `seconds` differs; new keys only (`params.ext_clock_flow`, `ext_cut`, `ext_pair_motion`, `extension_lost`).

| 3-seed means | control | E3, 45 km (band) | clock + union, whole cut, 45 km | E3, 50 km (band) | band / Earth |
|---|---|---|---|---|---|
| **E tests** (slope p50 / under 5 m/km / depth p50 / p90) | 2.4 / 0.68 / 261 / 721: 0 of 4 | 3.5 / 0.58 / 169 / 483: **1 of 4** | 4.3 / 0.54 / 171 / 468: **1 of 4** | 3.9 / 0.57 / 165 / 440: **1 of 4** | 7.0–17.6 / ≤ 0.35 / 58–146 / 203–510 |
| **E trigger: eligible** (design check ≥ 0.5) | — | **0.84** | **0.83** | **0.93** | E2: 0.01 / 0.04 |
| **rift over control / own E area** (design check: rift ≤ own) | — | +0.020 / 0.030 ok | **+0.052 / 0.035 fails** | +0.016 / 0.022 ok | |
| > 45 km / land > 2 km (label-free) | 0.298 / 0.342 | 0.200 / 0.255 | 0.215 / 0.280 | **0.171 / 0.227** | ≤ 0.13 / ≤ 0.10 (Earth 0.086 / 0.05) |
| < 30 km / median | 0.090 / 39.6 | 0.122 / 38.3 | **0.218** / 37.1 | 0.104 / 38.3 | Earth 0.190 / 37.0 |
| continental area (label-free) | 0.303 | 0.318 | 0.314 | 0.332 | 0.35–0.42 (Earth 0.384) |
| inland profile, worst | +3.8 km | +2.5 | +2.7 | +2.9 | ±1.5 |
| margins at 1° (thin per km / abrupt / edge T / inner T) | 117 / 16% / 26.1 / 34.5: fails | 129 / 14% / 25.4 / 33.4: passes | 133 / 12% / 25.0 / 33.0: passes | 145 / 14% / 25.2 / 33.2: passes | 100–200 / ≤ 30% / 24–29 / 30–34 |
| extended by E, inland bins | — | 0.215 at **41.3 km** | 0.230 at 39.5 km | 0.173 at **44.3 km** | 0.05–0.15 at 29–33 |
| young-ocean offset (mean; gate ±300 m) | −278 m | −227 | −229 | −241 | |
| continental volume (guard ≥ −14%) | −11.7% | −12.0% | −13.2% | **−8.5%** | ~0 |
| export (gate 4–8 km³/yr) | 5.9 | 6.2 | 7.2 | 5.0 | |
| area ledger: collision / rift / E / residual (guard: residual within 0.005 of the control's) | −0.113 / +0.057 / 0 / −0.008 | −0.134 / +0.077 / +0.030 / −0.013 (on the edge) | −0.150 / +0.109 / +0.035 / **−0.019 fails** | −0.113 / +0.073 / +0.022 / −0.012 | |
| E starts per run; welded per run, median duration; guard-ended | — | 35.0; 9.3, 37 Myr; 7 of 28 | 32.7; 22.3, 26 Myr; 9 of 67 | 30.0; 5.0, 78 Myr; 6 of 15 | Basin and Range ~15–35 Myr |
| **lost per run** (a half docked or was consumed) | — | **21.0** (60%) | 5.7 (17%) | **21.0** (70%) | |
| far side on a collision (node share, median; starts with > 0.5) | — | 0.45; 48% | — | 0.58; 62% | |
| plates today; area-weighted speed −225…0 Myr | 24.3; 4.6 cm/yr | 26.7; 3.8 | 23.7; 4.2 | 23.0; 3.9 | Bird 52; Earth ~3.5–4 |
| largest > 45 km piece per seed (Mkm²) | 13.2 / 15.6 / 14.2 | 13.0 / 3.6 / 11.6 | 8.5 / 6.6 / 4.1 | 25.1 / 6.3 / 9.4 | Tibet 7.9 |
| > 45 km land elevation p50 per seed | 2.6 / 2.8 / 3.4 km | 2.5 / 2.5 / 3.1 | 2.9 / 2.7 / 2.8 | 2.3 / 2.9 / 2.8 | Earth 1.2 km |

Elevation-check gate per seed: control seeds 1001 / 1002 and the whole cut's 1001 fail on the young-ocean offset (−356,
−307, −308 m); every band-cut seed passes (−177 … −297 m). The young-ocean watch-point is better with the band cut.

**Decision by the rule: E3 is not adopted.** Every candidate passes 1 of the 4 E tests (the > 45 km depth p90); the rule
needs 2. The whole cut also fails a design check (rift over the control > its own E area, third sweep in a row) and the
ledger-residual guard. `extension` stays False. Gates and the other guards pass for both band cuts; E3 at 50 km is the
best run so far on almost every crust number (> 45 km 0.171, land > 2 km 0.227, volume −8.5%, area 0.332, margin
review passes).

**Why (read before changing anything):**
1. **The two E3 changes did what they were for.** The carried clock lifts the trigger reach from 0.01–0.04 (E2) to
   0.83–0.93; the band cut keeps the rift added elsewhere below E's own area (the whole cut does not: +0.052 against
   0.035, as in the E and E2 sweeps). Union motion holds the plate speed at 3.8–3.9 cm/yr (control 4.6): no runaway.
2. **The events do not finish.** 60–70% of band events end "lost" (21 per run): a half docks onto a third plate after
   a median 36–38 Myr, about the merge rule's 40 Myr of contact. Half of the microplates (median far-side collision
   share 0.45 / 0.58) are strips whose far side is a collision: the relative motion pushes them into it, they dock,
   and the extension stops. So the crust E marks stays at 41–44 km (Earth's extended crust 29–33 km), only 5–9 events
   per run weld at 30 km, and at 50 km 40% of the welds come from the 100 Myr guard. The whole cut, whose halves rarely
   dock (5.7 lost per run), makes the most thin crust (< 30 km 0.218, Earth 0.190): the thinning works where the
   events run to the end.
3. **The slope pair is out of E's reach as long as thick crust stands this high.** Slope p50 of land > 2 km moved
   2.4 → 3.5–4.3 m/km (Earth 11.1) and the under-5 m/km share 0.68 → 0.54–0.58 (≤ 0.35). The planet's > 45 km crust
   stands at 2.3–3.4 km (Earth 1.2 km) and land > 2 km is 0.23–0.28 of land (Earth 0.05): the test samples plateau
   tops that Earth's thickness–elevation relation (Next item C) would put below 2 km. Of the four E tests, the depth p50
   (165–171 km, band to 146) is the one an E change can still reach.

**Proposal E4 (to agree; E3 at both onsets with one change; decision rule and guards unchanged):**
- **(g) `ext_side` 'free': the microplate is the side that faces a free boundary.** At a band cut, for each candidate
  side, the far edge = its nodes next to a third plate; the side with the smaller collision share on its far edge
  moves (Rey et al. 2001: gravitational spreading needs a free boundary to extend toward; the side pushed into a
  collision cannot open); ties, including two free sides, go to the smaller area as now. 'smaller' = E3 as built. No new
  number.
- **Tests, fixed before building** (`tests/test_plates.py`): (g1) a plate with a thick piece between a collision on one
  side and a subduction or ridge boundary on the other: 'free' moves the free side, 'smaller' moves the smaller side
  even when it faces the collision; (g2) `ext_side` 'smaller' reproduces E3 step by step at n = 48 (incl. a split and a
  merge); E3's seven tests and E/E2's pass.
- **Design checks (new, besides the two kept):** lost events ≤ 1/3 of starts; far-side collision share median ≤ 0.2.
- **Pre-check** in my workspace (seed 1001, 50 km, 250 Myr) for the design checks; stop and look if lost stays > 1/3.
- **Sweep (`Extension4 Sweep.cmd`):** control / E4 45 km / E4 50 km × 3 seeds on this sweep's baseline (E3 50 km need
  not be rerun: runs are deterministic, read from this sweep). If lost events stay high with the free side, the next
  step is to keep a pair from docking while its event runs (not built).
- **Alternative order:** item C (thickness–elevation from the lithosphere's thermal state) first, since it decides
  the slope pair, then E4 on top. My judgement: E4 first (one small change, its own test within reach), C next.

**Side change (user request, 25 Oct night): colours of the maps every 25 Myr** (`scripts/simulate_plates.py`, `draw`,
used by every plate run and sweep for `maps.png`). New `MAP_STOPS_M` / `map_colours()`: blues from −6000 m (navy) to
the coast (pale), then a hypsometric tint (green lowland, olive and sand uplands, browns, grey-brown, snow at 6000 m),
65,536 levels, with a colour bar. matplotlib's 'terrain' had put 0 m mid-ramp, so sea between −1500 and 0 m was drawn
green like lowland; the shallow interior seas of the start's supercontinent show now. Checked by redrawing
`sweep-20261025-141028` E3 50 km seed 1001 with the new `draw` (preview in `outputs/palette-preview/real/`).
No numbers change.

### Step 8, follow-up 1, thirty-third pass (25 Oct, evening): E4 built; its pre-check found that boundary typing reads transforms as collisions; `boundary_normal` 'ring' built; one sweep for both

**Agreed by the user (25 Oct, 15:06):** the suggested path, E4 first, then C.

**E4 built (`planet_layer/plates.py`; `extension` stays False, defaults unchanged):** PARAMS `ext_side` 'free' (E4;
'smaller' = E3 as built). At a band cut, each candidate side's far edge is its nodes next to another plate; the side
whose far edge has the smaller share of collision-boundary nodes moves; equal shares, two free sides, and a side
without a far edge go to the smaller area (Rey et al. 2001). The whole-plate cut ignores it. The event and its start
record carry `side`, `far_collision_sides` (both sides' shares) and `moved_larger`; `scripts/simulate_plates.py` saves
them in `extension_starts`; `scripts/interior_probe.py` prints an "E4 design checks" line (lost / starts ≤ 1/3, far
side on a collision median ≤ 0.2, larger side moved in k of n band starts).

**Tests** (`tests/test_plates.py`): `test_free_side_moves_away_from_the_collision` (g1: for an along and an across
cut, collision on the smaller side, on the larger side, and on neither (ridge / subduction); 'free' moves the free side
whichever is smaller, 'smaller' always the smaller side; checked to fail with the rule disabled),
`test_e4_side_rule_is_inert_without_extension_and_for_the_whole_cut` (d). **g2 by hand** (not a pytest: it needs the
pre-edit file): the pre-edit `plates.py` beside the new one at n = 48, 30 steps, world fields, owner, plate rotations,
speeds and all rasters compared by value: identical on PARAMS defaults, the sweep baseline, E3 at onset 40 km with
`ext_side` smaller (6 starts, a weld, 2 lost, merges, rifts) and E2; 'free' first differs at the first event (step 4).
**Note for such checks:** two runs of the same code differ at round-off (1e-12) unless `threads` is 1: ducc0's
multithreaded transform in the flexure. Earlier "identical step by step" checks hold only if they ran single-threaded;
compare with `params={'threads': 1}`.

**Pre-check** (my workspace, seed 1001, onset 50 km, 250 Myr at n = 217, the E3 sweep's baseline; one seed, a
trajectory unlike your machine's):

| seed 1001 | E3 50 km | E4 50 km | control, ring typing | E4 50 km, ring typing | band / Earth |
|---|---|---|---|---|---|
| E tests (slope p50 / under 5 / depth p50 / p90) | 3.1 / 0.62 / 183 / 469: 1 of 4 | 3.2 / 0.62 / **142 / 385: 2 of 4** | 2.5 / 0.65 / 277 / 777: 0 | 3.7 / 0.56 / 195 / 579: 0 | 7.0–17.6 / ≤ 0.35 / 58–146 / 203–510 |
| lost / starts (E4 check ≤ 1/3) | 16 / 30 = 0.53 | **16 / 24 = 0.67** | — | **7 / 20 = 0.35** | |
| far side on a collision, median (≤ 0.2) | 0.53 | 0.33 | — | 0.27 | |
| larger side moved | 0 of 30 | 10 of 24 | — | 14 of 20 | |
| eligible (≥ 0.5) | 0.95 | 0.95 | — | 0.97 | |
| > 45 km / < 30 km / median (label-free) | 0.230 / 0.115 / 38.8 | 0.171 / 0.130 / 38.5 | 0.286 / 0.103 / 37.9 | 0.258 / 0.136 / 37.6 | Earth 0.086 / 0.190 / 37.0 |
| land > 2 km of land | 0.255 | 0.212 | 0.302 | 0.312 | Earth 0.05 |
| largest > 45 km pieces (Mkm²) | — | 3.4 / 3.4 / 3.1 | 28.3 / 7.2 / 5.6 | 14.4 / 7.1 / 6.9 | 7.9 / 1.4 / 1.1 |
| extended by E, inland | 0.159 at 44.1 km | 0.216 at 44.3 | — | 0.187 at 43.4 | 0.05–0.15 at 29–33 |
| ends per run (median, guard-ended) | 9 (99 Myr, 4) | 5 (38 Myr, 2) | — | 8 (100 Myr, 5) | |
| volume / speed −225…0 / plates | −11.3% / 4.5 / 28 | −13.2% / 4.6 / 23 | −10.8% / 4.6 / 21 | −13.2% / 4.1 / 30 | ≥ −14% / ~3.5–4 / Bird 52 |
| area ledger residual | −0.011 | −0.014 | −0.007 | −0.019 | |

(The cloud's node-typing control for this baseline, thirty-first pass: > 45 km 0.299, < 30 km 0.068, depth 290 / 826,
slope 2.0 / 0.73, volume −11.2%, speed 3.7.) The elevation check needs ETOPO and did not run here.

**E4 on the old typing passed 2 of 4 E tests on this seed but lost more events (0.67), so I logged a rerun** (125 Myr,
every merge that touches a running event). **5 of E4's 9 lost events are the microplate docking onto its own parent,
each after exactly 40 Myr** (`merge_after_myr`); in E3 1 of 13 (E3's strips mostly dock onto or are consumed by the
third plate across their collision far side, which is what E4 removes). At n = 48 the parent–child contact during an
event is typed ridge on about half its nodes and collision or subduction on the other half, transform on 0–10%,
though the band's side edges are small circles about the relative pole: transforms by construction. The free rule
moves the larger side in 10 of 24 starts, whose long side edges carry > 1,000 km of this false collision.

**Cause, general: `classify_boundaries` types a node by its one neighbour on another plate with the largest normal
rate, so the boundary normal is a grid direction.** On the grid's staircase, a transform that does not follow the grid
shows normal rates of ~0.7 v against ~0.7 v of shear and fails the transform test (normal < ½ shear): ridge on one
step, collision or subduction on the next. Synthetic test, 12 random great circles with pure transform motion: median
30% of the boundary nodes typed transform, minimum 0%, at n = 48 and 217 alike; opening and closing boundaries typed
right. On the E3 run's present state, transforms are 11% of the node-typed boundary and 36% of the contour-typed length
(`scripts/boundary_lengths`, which types each segment of a smoothed contour by its true normal); collision node count
7.6k km against 1.4k km by contour. Every process reading the type was affected since step 7: collision thickening and
docking along oblique transforms, slab pull and trench flags along their "subduction" steps, ridge crust on their
"ridge" steps; probably part of the trench-length item (node count 1.4–2.1× Earth).

**Built, default off: PARAMS `boundary_normal` 'ring'** ('node' = as built, default until a sweep decides): the
normal toward a neighbouring plate b is the tangent direction of Σ (x_k − x_i) over the node's two-ring, +1 on b, −1 on
the node's own plate (`classify_boundaries_ring`; one plate type per node, the neighbouring plate with the largest
|normal rate|, as before). No new number. **Tests:** `test_ring_normal_types_transforms_at_any_orientation` (b1 / b2:
≥ 95% transform on 12 great circles and 12 small circles 400 km off them, n = 48 and 217; the node kernel's median
< 0.5 is asserted too, as the defect), `test_ring_normal_types_opening_and_closing_as_before` (b3: ≥ 98% ridge /
subduction / collision where the true rate exceeds twice the threshold; rate sign; trench on the lower-score side),
`test_node_normal_is_the_default_and_ring_keeps_the_ledgers` (b4 / b5). 'node' reproduces the code as built step by
step (defaults and E3, 30 steps). 65 planet-layer tests pass in my workspace.

**With 'ring' on the same state the simulator's own typing matches the contour typing:** node-counted shares ridge
0.31 / subduction 0.31 / collision 0.025 / transform 0.35 against contour-typed 0.31 / 0.27 / 0.024 / 0.35 (control).
Contour lengths (control): ridge 100k km (Bird 95k), subduction 86k (51k), collision 8k (Bird's collision and
convergent 41k), transform 115k (74k): **subduction is still 1.7× Bird's**, so the trench-length item is not only
typing. With 'ring' E4 loses 0.35 of its events (just over the 1/3 check), but on this seed passes 0 of 4 E tests:
5 of its 8 ends are the 100 Myr guard and its extended crust stays at 43 km (it does not thin to 30). The ring control
itself has the largest plateau so far (28 Mkm²; node control 17). One seed, a different trajectory each time: the
sweep decides.

**Sweep (`Typing E4 Sweep.cmd`, replaces the unused `Extension4 Sweep.cmd`):** tests first, then five combinations
× seeds 1001–1003 on the E3 sweep's baseline with E3's switches and `ext_side` free passed: node control (must equal
`sweep-20261025-141028/extension-False...` digit for digit, only new keys may differ), ring control, ring E4 45 km,
ring E4 50 km, node E4 50 km; the four reviews and `boundary_lengths` of every run's present snapshot in
`<sweep>/reviews/`. **Read in this order:** (1) node control identity; (2) **typing**: ring is adopted as the default
if its control passes the gates (elevation tests, export 4–8 km³/yr) and the guards (crust median 34–40 km, volume
≥ −14%); read beside it the boundary lengths by type against Bird, plate count and speed, the young-ocean offset, and
every crust number against the node control. A gate that fails with ring points to the next missing mechanism: stop
and read, do not revert the fix to pass it. (3) **E4** by the rule of the twenty-seventh pass, each candidate against
the control on its own typing, with the design checks (eligible ≥ 0.5; rift over the control ≤ own E area; lost ≤ 1/3;
far side on a collision median ≤ 0.2). If ring is adopted and E4 still loses > 1/3: read what ends the events first
(the logged-run tool: `merges` touching a running event, by role), then decide between keeping a pair from docking
while it runs and item C.

### Step 8, follow-up 1, thirty-fourth pass (26 Oct): Typing E4 sweep read: ring typing adopted, E4 not; C re-examined against Earth; sea level sits 600–900 m below the datum

**Sweep `sweep-20261025-170555`** (`Typing E4 Sweep.cmd`; reviews and `boundary_lengths` in `<sweep>/reviews/`).
pytest 65 pass. **(1) Node control identity holds:** every `stats.json` value equals `sweep-20261025-141028`'s
extension-False run except `seconds` and the two new parameter keys, and all 124 snapshots × 3 seeds are equal field
by field (NaN = NaN).

**(2) Typing: `boundary_normal` 'ring' adopted as the default** (PARAMS; the default assert, now
`test_ring_normal_is_the_default_and_keeps_the_ledgers`; audit 4c). All 65 tests pass in the device shell (Linux;
`test_coupling` run from a copy because the shell cannot delete its old temp folder); `Ring Defaults Tests.cmd` runs
them on Windows.

| 3-seed means | node control | ring control | Earth / band |
|---|---|---|---|
| elevation tests (seeds passing) | 1 of 3 (young-ocean offset −356 / −307 / −172 m) | **3 of 3** (−185 / −188 / −101) | ±300 m |
| export km³/yr / volume | 5.9 / −11.7% | 6.2 / −12.6% (seed 1003 −14.3%) | 4–8 / ≥ −14% |
| label-free crust: area / < 30 / > 45 / > 55 / median | 0.303 / 0.090 / 0.298 / 0.088 / 39.6 | 0.303 / 0.097 / 0.304 / 0.083 / 39.3 | 0.384 / 0.190 / 0.086 / 0.018 / 37.0 |
| margins at 1° (thin/km, abrupt, edge T, inner T) | 117, 16%, 26.1, 34.5: fails | 116, 13%, 25.4, 33.5: **passes** | |
| item-2 tests / E tests | 1 of 5 / 0 of 4 | 1 of 5 / 0 of 4 | |
| land > 2 km of land / young orogen on land | 0.342 / 0.422 | 0.381 / 0.425 | 0.05 / 0.16 |
| largest > 45 km pieces (Mkm², by seed) | 13.2 / 15.6 / 14.2 | 24.5 / 12.1 / 36.2 | Tibet 7.9 |
| contour lengths k km: ridge / subduction / collision / transform | 103.9 / 91.4 / 6.9 / 102.4 | 97.6 / 83.9 / 5.4 / 99.4 | Bird 94.8 / 51.3 / 40.5 (coll.+conv.) / 73.9 |
| node-counted shares ridge / sub / coll / transform | 0.44 / 0.42 / 0.03 / 0.11 | 0.32 / 0.32 / 0.02 / 0.34 | contour (ring) 0.34 / 0.29 / 0.02 / 0.35 |
| plates / area-weighted speed / ocean age (means −225…0) | 17.0 / 4.6 cm/yr / 62.9 Myr | 16.8 / 4.0 / 65.1 | Bird 52 / — / 64 |

The young-ocean watch-point eases with ring (no seed beyond −300 m): part of it was false collision / subduction typing
on transforms. Subduction is still 1.6× Bird's by contour (item 3). Land > 2 km and the largest plateau are higher
with ring (within the seed spread for > 45 km): the rule says read, not revert.

**(3) E4: not adopted.** Each candidate against the control on its own typing:

| 3-seed means | ring E4 45 km | ring E4 50 km | node E4 50 km |
|---|---|---|---|
| E tests (slope p50 / under 5 / depth p50 / p90) | 3.8 / 0.57 / 198 / 561: **0 of 4** | 3.1 / 0.62 / 177 / 467: 1 of 4 | 3.3 / 0.62 / 183 / 475: 1 of 4 |
| > 45 km / < 30 km / median | 0.236 / 0.157 / 38.7 | 0.198 / 0.118 / 38.4 | 0.236 / 0.103 / 39.1 |
| export / volume | 7.6 / −14.0% (seed 1002 −17.0%) | 6.1 / −12.4% (seed 1002 −16.9%) | 5.9 / −11.2% |
| area-ledger residual (control's) | −0.019 (−0.006): **fails** | −0.014 (−0.006): fails | −0.015 (−0.008): fails |
| eligible / rift over control ≤ own | 0.84 / +0.038 ≤ 0.041 ok | 0.96 / +0.015 ≤ 0.027 ok | 0.96 / — |
| lost / far side on a collision (≤ 1/3, ≤ 0.2) | **0.33 / 0.07 ok** | 0.47 / 0.23 x | 0.55 / 0.33 x |
| extended by E inland (Earth 0.09–0.11 at 31 km) | 0.275 at 39.0 km | 0.207 at 44.5 | 0.231 at 44.6 |

With ring typing at 45 km the docking problem is gone (lost 0.33, far side 0.07): it was the typing defect. What fails
now is the thinning. **Event ends (ring E4, from `extension_ends`):** every start hits the 400 km half-width cap (zone
800 km, divergence 1.6 cm/yr). 45 km: 39 ends over 3 seeds, 22 weld at 29.2–30.0 km (one at 9) after 6–98 Myr (median ~34), 17 end at the
100 Myr guard at 31–60 km (some zones thicker than at the start, some reduced to no area: shortened while extending);
50 km: 16 of 23 guard-ended. **Present E-marked crust** (`extension_age_myr`): p25 / p50 / p75 about 20 / 33–46 / 45 km
in every mark-age bin, and 30–99% of it (rising with mark age) carries a thickening clock younger than its mark: E
thins its core, marks two to three times Earth's extended share, and much of what it thinned is shortened again.

**C re-examined against Earth before designing it.** New `scripts/earth_lithosphere_residual.py`
(→ `outputs/planet-reference/earth-lithosphere-residual-n217.json`): per node, the Airy height of its own CRUST1.0
column (upper / middle / lower crust at the type's densities from `CNtype1_key.txt`, sediment at 2,400), and the
residual against ETOPO (water-unloaded), anchored to ocean ≤ 3 Myr (type A0, r = −3,139 m): dL = what the mantle
lithosphere adds.

| total T (km) | Earth z p50 | planet Airy | Earth crystalline ρ | dL p50 (m) | inside Bird's orogens: z / planet Airy | outside: z / planet Airy |
|---|---|---|---|---|---|---|
| 20–25 | −635 | −1,851 | 2,927 | −544 | −445 / −1,843 | −651 / −1,852 |
| 25–30 | −72 | −1,015 | 2,867 | −1,067 | −18 / −1,037 | −76 / −1,010 |
| 30–35 | 98 | −515 | 2,832 | −1,513 | 146 / −539 | 92 / −508 |
| 35–40 | 321 | 239 | 2,830 | −2,031 | 605 / 234 | 306 / 240 |
| 40–45 | 359 | 814 | 2,830 | −2,629 | 890 / 917 | 329 / 789 |
| 45–50 | 681 | 1,511 | 2,828 | −3,064 | **1,383 / 1,559** | 431 / 1,505 |
| 50–55 | 1,871 | 2,274 | 2,791 | −2,996 | **2,282 / 2,294** | 608 / 2,163 |
| 55–80 | 4,356 | 4,131 | 2,771 | −3,079 | 4,524 / 4,155 | 3,718 / 3,863 |

By group (dL p50): Archean −2,079, Proterozoic −2,242, platform −3,037, Phanerozoic −2,383, orogen −2,378, arc −1,892,
extended / rift −1,316, margin / shelf −1,179 m. **Readings:** (i) Earth's crust density does not rise with thickness
(2,830 in every bin from 30 to 50 km, lighter above): composition does not flatten Earth's relation. (ii) The mantle
lithosphere does: dL deepens ~110 m per km of crust from 25 to 48 km (part of that slope is regression dilution on
CRUST1.0's noisy thickness), and thin, rifted crust stands ~1 km higher than normal crust. (iii) **Inside Bird's
orogens Earth's 40–55 km crust stands on the planet's own Airy line** (1,383 vs 1,559 m, 2,282 vs 2,294 m); Earth's
thick crust is low only outside orogens (cold, old lithosphere: 431 m at 45–50 km). The planet's thick crust is almost
all young orogen (orogeny clock 0 on 94–100% of it). So **C would lift thin and rifted crust (shelves, shallow seas)
and lower thick old crust, but would not lower the planet's plateaus**: the thirty-second pass's "the slope pair is
for C" does not hold. The plateaus are a thickness problem (too much thick crust, spread wide), not an elevation-law
problem. This agrees with Lamb et al. 2020 (whole-lithosphere isostasy: the mantle lithosphere's negative buoyancy
counters the crust's; contrast with the asthenosphere 20–40 kg/m³; dynamic topography a few hundred m at most).

**Planet elevation by total thickness (ring control, 3 seeds, z relative to sea level):** 20–25 km −375, 25–30 −5,
30–35 483, 35–40 1,148, 40–45 1,694, 45–50 2,545, 50–55 3,160, 55–80 4,002 m. At 35–45 km the planet stands 0.8–1.3 km
above Earth: that is not the Airy law (238 / 814 m) but sea level.

**Sea level sits 590–890 m below the datum** (snapshots' `sea_level_m`; −330 to −353 m already at −250 Myr, about −700
m from −200 Myr on). The datum is Earth-calibrated (ridges at −2,600 m, 38 km crust at +300 m, Earth's 35–40 km bin at
321 m), so on Earth sea level is the datum. Water the basins hold below the datum (global equivalent layer): Earth
2,629 m; the planet 2,858–2,872 m at the start and **3,133–3,285 m at present**. Decomposition at present: extra
deep-ocean area (oceanic crust 0.66–0.70 of the surface vs Earth's 0.61 below −1,500 m; the continental area lost)
+200–360 m; a deeper floor (mean 4,448–4,545 m on the datum vs Earth's 4,205 m) +200–240 m. The deeper floor is not
the age distribution (GDH1 over the planet's ages 4,713–4,808 m, over Earth's triangular age–area distribution 4,844
m) and only partly sediment (planet's ocean sediment mean 394–727 m, 188–574 m beyond 200 km from continents; GlobSed,
Straume et al. 2019: 927 m whole ocean, ~404 m beyond 200 km, ~3 km on margins, 3.37e8 km³): Earth's floor stands
0.64 km above GDH1-over-its-ages, the planet's 0.20–0.27 km (missing: plateaus, seamounts, part of the margin
sediment). Shallow seas are missing too: ocean depth p90 −2.2 to −2.4 km vs Earth −0.5 km (shelves on thin crust,
which (ii) above says stands ~1 km higher on Earth).

**What sea level explains:** shifting the present land by the sea-level offset (static, no isostatic response) takes
land > 2 km from 0.34–0.45 of land to 0.24–0.32 (Earth 0.05), land p50 from 1.36–1.86 km to 1.06–1.18 km (0.38), and
land from 0.27–0.29 to 0.21–0.23 of the surface (0.28): about a quarter of the high-land problem. The rest is the crust
itself: too much thick crust (> 45 km 0.30 vs 0.086) on too little area (0.30 vs 0.384), with volume lost (−10 to −14%).

**Continental budget, measured from the run ledgers (ring control, km³/yr over 250 Myr):** continental erosion
6.8–8.8 (of it crystalline 4.2–5.5), redeposited on continents 3.5–4.5, to the ocean 3.3–4.4; sediment subducted
2.2–2.6, accreted 0.3–0.4; island-arc crust added 0.6–0.64, of which essentially none becomes continental (area-ledger
`arc` +0.000: arcs never reach `arc_to_continent_thick_m` 20 km); volcanic 0.1–0.15. Earth (Scholl & von Huene 2009):
arc addition 2.7 km³/yr (intra-oceanic 1.5, continental margins 1.0, collisional 0.2), losses 3.2 (ocean-margin
trenches 2.5, 60% of it subduction erosion; collision zones 0.7); net growth effectively nil in the Phanerozoic.
The planet's island arcs make 0.4× Earth's addition and never accrete; its sediment subduction is 1.3–2.6× Earth's (~1.0 from the 2009 split, 1.65 in older estimates).

**Proposal (to agree; nothing built):** the next item is the one that moves the most toward realism, and the order of
the handoff's list changes:
- **B. Continental budget and ocean basin (new, first).** Tests fixed now, 3-seed means at present: water below the
  datum within ±100 m of Earth's 2,629 m (sea level on the datum within about ±140 m of 0); continental volume change
  over 250 Myr ≥ −3% (Earth's net, −0.5 km³/yr, is −1.7%); label-free continental area 0.384 ± 0.03; budget terms in
  Earth's ranges (arc addition 2.0–3.5 km³/yr, of it intra-oceanic 1.0–2.0 reaching the continents by accretion;
  sediment subduction ≤ 1.7; losses at collisions counted); deep-ocean floor mean (below −1,500 m) within ±150 m of
  Earth's 4,205 m; the existing gates and guards. First step, no model change: a budget script (`scripts/crust_budget.py`,
  every term per run in km³/yr beside Scholl & von Huene) and a probe of where island-arc crust goes (thickness reached,
  subducted, collided, left as anomaly). Then candidates, each with its own test: arc growth rate from Earth's
  intra-oceanic arcs; terrane accretion of arcs and plateaus at trenches (docking without the 20 km threshold);
  collision-zone losses (lower-plate detachment, 0.7 km³/yr); oceanic plateaus / seamounts.
- **C. Lithosphere buoyancy (second, narrowed):** tests from the table above, read relative to the 35–40 km bin so the
  sea-level offset drops out (Earth: 20–25 −956, 25–30 −393, 30–35 −223, 40–45 +38, 45–50 +360 m; orogen bins on the
  Airy line), extended crust dL ~ +0.7 km over normal, cratons ~ −0.2 to −1.0 km. Expected to lift thin margins (shelves,
  shallow seas) and to lower nothing the planet has much of. Built alone it would raise flooded margins and push sea
  level lower still, so it follows B.
- **E (the thick crust itself) stays open:** what remains is thinning, not docking: marked area too wide, events
  guard-ended where the zone is shortened at the same time, thinned crust re-thickened later. Not to be tuned now.

My judgement: B first (Earth-measured fluxes, one integrated test that every land number depends on), then C, then E.
Alternative: E first (the plateaus are the largest single misfit).


### Step 8, follow-up 1, thirty-fifth pass (26 Oct): B's budget measured: the continent leaks at the sediment step, island arcs grow at a fifth of Earth's rate, the excess trenches are intra-oceanic from the start, sediment density is 40% of the floor misfit

User, 26 Oct: B first, as recommended. **Built, no model change:** `scripts/crust_budget.py` (reads `stats.json` series
and snapshots only; 2–3 s per run in the device shell) and two new keys in `scripts/boundary_lengths.lengths`
(`subduction_continental_km`: continental crust on either side; `subduction_oceanic_km`: oceanic on both;
`test_boundary_lengths` passes). Read on `sweep-20261025-170555`, all five combinations (`<sweep>/reviews/crust_budget.txt`,
`crust_budget.json`).

**Checks.** Earth from ETOPO on n = 217 reproduces the thirty-fourth pass (water below sea level 2,629 m GEL; floor
deeper than 1.5 km 4,205 m on 0.614 of the surface); the planet's water inventory reads back 2,617 m (1.335e9 km³);
sediment carried into trenches, measured on the 1 Myr snapshots (downgoing side × closing rate, scaled by contour /
node length, ×0.56), 4.10 km³/yr vs the ledger's subducted + accreted 4.04; arc crust carried into trenches 0.65 vs arc
crust lost by difference (growth − stock change − turned continental + eroded) 0.60.

**B's tests, ring control (3 seeds; start in brackets). All five fail, as do the node control and the three E4 runs:**

| | ring control (seeds) | Earth / test |
|---|---|---|
| water below the datum (GEL) | 3,206 m (3,133 / 3,201 / 3,285); start 2,863 | 2,629 ± 100 ✗ |
| sea level on the datum | −786 m (start −337) | 0 |
| deep floor (below −1.5 km on the datum): mean / share | 4,524 m / 0.698 (start 4,438 / 0.631) | 4,205 ± 150 / 0.614 ✗ |
| continental volume change (crust + sediment; crust) | −12.6% (−10.1 / −13.3 / −14.3); crust −15.2% | ≥ −3% ✗ |
| label-free continental area | 0.303 (start 0.380) | 0.384 ± 0.03 ✗ |
| sediment subducted, last 40 Myr / whole run | 3.52 / 2.49 km³/yr | ≤ 1.7 (Earth ~1.0) ✗ |

**Budget, ring control (km³/yr, whole run / last 40 Myr; the last 40 Myr spread between seeds is large, e.g. crust
net −1.96 to −6.58):**

| term | planet | Earth (Scholl & von Huene 2009) |
|---|---|---|
| continental crust, net (series) | −4.47 / −4.94 | net growth ≈ 0 (additions 2.7, losses 3.2) |
| crystalline erosion | −5.02 / −5.41 | (eroded crust stays in the system until it is subducted) |
| trench on continents, booked net (arc 750 m − subduction erosion 700 m per m of convergence) | +0.01 / +0.03 = +1.2 km³/Myr per km of continental-side trench | arc ~1.0 (~40 per km), subduction erosion ~1.5 (~36 per km) |
| sediment accreted at trenches / collision sediment folded into crust | +0.37 / +0.53; +0.20 / +0.16 | |
| volcanic (hotspots, LIPs; all nodes) | +0.13 / +0.16 | |
| island arcs turned continental (area ledger × 20 km) | +0.02 / +0.01 | intra-oceanic arcs 1.5 (accreted in time) |
| residual (crust ledger) | −0.18 / −0.42 | |
| continental surface eroded / redeposited on continents (of it submerged) | −8.01 / +4.05 (2.23); −9.03 / +4.28 (2.06) | river delivery 4–8 (sixteenth pass) |
| sediment reaching oceanic crust (of it turbidite fill, `deep_sea_sediment`) | +3.98 (2.77) / +4.76 (3.17) | |
| ocean-floor sediment subducted / accreted / stock change | −2.49 / −0.37 / +0.84; −3.52 / −0.53 / +0.07 (residual −0.27 / −0.66) | subducted ~1.0 (40% of 2.5; 1.65 older) |
| island arcs: growth net of forearc erosion / stock change / lost (subducted) | +0.62 / +0.18 / 0.41; +0.83 / +0.22 / 0.60 | 1.5 |
| area ledger per Myr: collision / rift | −4.7e-4 / +2.3e-4 of the surface; −3.5e-4 / +2.1e-4 | |

**Readings.**
1. **The continent does not lose volume at its trenches.** Per km of continental-side trench, arc addition and
   subduction erosion are Earth's by construction (750 / 700 m per m of convergence; net +1.2 km³/Myr/km, Earth
   ~+4), and the continental-side trench is Earth's length (26.4k km mean of −40..0; Earth ~25k, see 3). The loss is
   the crystalline erosion whose sediment leaves the continental crust: 4.0–4.8 km³/yr reaches oceanic crust and
   2.5–3.5 is subducted. In Scholl & von Huene's accounting (crust and its sediment until subducted) the planet's net is
   about +0.03 (trench, continents) + 0.01 (arcs) + 0.16 (volcanic) − 3.5 (sediment subducted) ≈ −3.3 km³/yr in the
   last 40 Myr, Earth −0.5.
2. **Why so much sediment reaches trenches (50 km³/Myr per km of trench incl. accreted; Earth ~27: ~1.0 subducted +
   13% accreted over ~42k km).** Trenches facing continental crust take 2.21 km³/yr (84 per km; seeds 0.96–3.20),
   intra-oceanic ones 1.89 (35 per km). (a) The margins on continental crust are full: sediment on submerged
   continental crust 2,562 m (Earth 2,398), on crust < 30 km 3,692 m (2,394), on half Earth's thin-crust area (< 30 km
   0.097 vs 0.190, `crust_review`), so the rest spills onto oceanic crust; 70% of what reaches it is the turbidite fill
   (`graph_landscape.fill_and_spill`: each parcel walks the steepest path to a low and ponds there; on a GDH1 floor that
   low is the oldest floor nearby, which on the planet borders trenches). (b) Too much trench: see 3. Where the fill
   lands has not been measured (it needs a probe inside a run; no model change).
3. **The excess trench is intra-oceanic and comes from the start.** Contour lengths (k km, continental side /
   intra-oceanic / ridge; ring control): seed 1001 −240 Myr 2 / 33 / 65, −150 14 / 44 / 79, −50 29 / 46 / 91, 0 23 / 56 / 96;
   seeds 1002 and 1003 alike (intra-oceanic 33–36k at −240, 54–58k at 0). Earth: intra-oceanic subduction is ~40% of
   the subduction margins (Leat & Larter 2003), ~17k of S&vH's ~42k km (2.5 AU at 60 km³/Myr/km). The random
   motions of the start's oceanic plates (7 of 12) make ocean–ocean convergence everywhere; on Earth oceanic plates
   mostly spread from each other and sink under continents. The thirty-fourth pass's "subduction 1.6× Bird" (item 4)
   is all this: 3× Earth's intra-oceanic length, an Earth-like continental side.
4. **Island arcs stay oceanic.** Growth net of forearc erosion 15.4 km³/Myr per intra-oceanic km (gross 300 m per m of
   convergence, ~16 at 54 mm/yr); Earth's western Pacific arcs 30–95 with crust 20–30 km thick (Dimalanta et al. 2002;
   Reymer & Schubert 1984: 20–40). The planet's arc stock (46 Mkm³ at present, from 0) stands p50 / p90 / max 8.9 / 11.5
   / 19.6 km, so none reaches `arc_to_continent_thick_m` (20 km); half of it lies within 300 km of a trench; 0.30
   km³/yr enters trenches facing continental crust (terranes that would dock on Earth), 0.35 facing ocean. Node typing
   had arc growth 1.3–1.5 km³/yr (transforms typed subduction), ring 0.65–0.93. At Earth's per-km rate the planet's
   54k km of intra-oceanic trench would add 1.6–5 km³/yr (Earth 1.5): the rate and the length (3) belong together.
5. **Sea floor: sediment density is ~40% of the misfit.** The planet's ocean sediment is a layer of `rho_sediment`
   2,400 (load and surface height); GlobSed's 927 m mean (Straume et al. 2019) is seismic, bulk thickness, and the
   basement correction it uses (Sykes 1996) implies ~2,015 kg/m³ near the top. The same solid mass at 2,015 (grain
   2,650; Airy, long wavelength) raises the floor 154 / 264 / 284 m (seeds): deep floor 4,524 → ~4,390 m (at 1,900:
   ~4,340), water below the datum −130 m (−180). The bulk-equivalent sediment would be 547–1,012 m (GlobSed 927 m whole
   ocean). The start already lacks 233 m of floor (no sediment, no plateaus or seamounts). The thirty-fourth pass
   compared thicknesses only and called it "partly".
6. **Area is the largest term in the water test.** The label-free continent goes 0.380 → 0.303; collisions take area
   at twice the rate rifts add it (above); on Earth arc accretion and extension add it back. After sediment density
   the remaining water excess (~420–450 m) is mostly this deep-ocean area (0.698 vs 0.614).

**Proposal (to agree; nothing built).** B splits into three candidates; all keep the fixed B tests, each adds its own:
- **B-a, sediment compaction.** The sediment field stays solid-equivalent (every ledger unchanged); the elevation reads
  bulk thickness and load from an Earth compaction law (Athy 1930; Sclater & Christie 1980: porosity φ0 e^(−cz),
  shale φ0 0.63, c 0.51 /km; sand 0.49, 0.27 /km), so a 0.5 km deep-sea cover stands at ~1,900–2,050 kg/m³ and a
  5 km basin at ~2,400–2,500 (today's value fits thick basins, not the thin deep-sea cover). Design point to agree:
  the erosion kernel moves surface heights, so within a plate step a deposit raises the surface by its solid height
  and the next elevation call by its bulk height; the alternative (deposit bulk = solid / (1 − φ_top) inside the
  kernel) touches `graph_landscape`, shared with the 11.53 km stage. Tests: floor mean ±150 m (expected to move
  ~130–180 m of 319); ocean-floor bulk sediment mean against GlobSed 927 m (± 25%); export and volume ledgers closed in
  solid volume as now; margins and elevation reviews pass.
- **B-b, intra-oceanic trenches and arcs (with item 4).** First step, no model change: why ocean–ocean convergence
  is so common (start plate motions, `_driving`, polarity rule; lengths by plate pair and over time). Then the plate-
  motion candidate it points to, with tests: intra-oceanic trench 12–25k km and continental side 20–35k (Earth ~17k /
  ~25k), ridge length kept (Bird 94.8k), plate speeds kept (`plate_speeds`). Only then arcs at Earth's rate:
  `arc_H_ocean_m` from Dimalanta et al. 2002 (30–95 km³/Myr/km ⇒ ~0.55–1.7 km of crust per m of convergence at
  55 mm/yr, vs 0.3), with tests: mature arc crust 20–30 km, intra-oceanic addition 1.0–2.0 km³/yr, arcs reaching
  continents (area ledger `arc` > 0) and docking by the existing collision rule.
- **B-c, sediment into trenches.** First a probe (no model change) of where the turbidite fill lands (distance to
  trenches and continents, floor age); a fan rule only if it lands at trenches out of proportion. Tests: sediment
  subducted ≤ 1.7 km³/yr; supply per km of continent-facing trench within Earth's range.
- E stays the main area lever (collision area loss); C follows B as planned.

**My judgement:** B-b's first step and B-c's probe first (both measure, change nothing; B-b also settles item 4),
then B-a (independent, quick to judge on the floor), then the arc rate. The volume test cannot pass on any one of
them: at −3% per 250 Myr the net may be ~−0.9 km³/yr, today −3.7 (crust + its sediment), and the pieces measured
are sediment subduction (−2.5 of excess), arcs (+1.5 missing) and parked ocean-floor sediment (−0.8).

### Step 8, follow-up 1, thirty-sixth pass (26 Oct): B's two measurements: the excess intra-oceanic trench is half plate fragments and half a start that converges too much; the offshore fill lands at the feet of active margins, and intra-oceanic trenches subduct the open-ocean stock

User, 26 Oct: the recommended route (B-b's diagnosis and B-c's probe first, then B-a, then the arcs). GlobSed v3 is
in `data/GlobSed/V3/GlobSed_package3/` (and v2 in `data/GlobSed/GlobSed_v2.nc`). **Built, no model change:**
`scripts/trench_origin.py` (B-b step 1), `scripts/earth_sediment.py` (GlobSed on n = 217, cached
`outputs/planet-reference/globsed-n217.npz`), `scripts/ocean_sediment_probe.py` (B-c), and
`boundary_lengths.typed_segments` (the per-segment typing `lengths()` already did, now returned as arrays; `lengths()`
output identical, `test_boundary_lengths` passes). Read on the ring control of `sweep-20261025-170555`
(`<sweep>/reviews/trench_origin.txt` / `.json`, `ocean_sediment_probe.txt` / `.json`).

**GlobSed format.** The v3 netCDF is HDF5 (the `.venv` has neither h5py nor netCDF4); the package's ArcGIS ASCII grid
writes missing values as `1.#QNAN`, which numpy cannot parse as is. `earth_sediment.read_asc` replaces the token and
reads the 5′ grid with numpy only. Checked with h5py in the device shell: ASCII = v3 netCDF within 0.006 m; v3 = v2
except v2's few negative values (min −69 m), which are 0 in v3. Area mean over the valid area (71.2% of the surface)
926.8 m, as published. On n = 217: 927.6 m over 0.712 of the surface.

**Earth references added (Bird 2003 PB2002_steps.dbf read directly; CRUST1.0 types 100 and 200 km to both sides of
each step, intra-oceanic = oceanic, island-arc or forearc types on both sides):**

| | length (k km) | closing / opening p10 / p50 / p90 (mm/yr) | < 20 mm/yr | flux (km²/yr) |
|---|---|---|---|---|
| SUB, intra-oceanic (200 km; 100 km: 25.8k) | 20.8 | 16 / 60 / 104 | 0.17 | 1.28 |
| SUB, continental side | 30.7 | 9 / 50 / 80 | 0.25 | 1.46 |
| OCB (oceanic convergence without a slab) | 17.4 | 2 / 7 / 33 | 0.81 | 0.23 |
| CCB | 23.0 | 5 / 13 / 46 | | 0.43 |
| OSR + CRB (production) | 67.3 + 27.5 | OSR 9 / 34 / 87 | | 3.37 (Rowley 2002: 3.4, steady since 180 Ma) |

Earth's ocean–ocean boundary (OSR, OTF, OCB, intra-oceanic SUB) is 153.4k km; intra-oceanic subduction is 0.14 of
it, 0.25 with OCB. Boundary total 260.8k km over 52 plates against 172.5k km of cap half-perimeters for the same
areas (shape ratio 1.51; big plates 1.3–1.8). Leat & Larter 2003: intra-oceanic systems ~40% of subduction margins
(the 20.8k above is 40% of 51.5k).

**B-b step 1: why ocean–ocean convergence is so common (3 seeds; seed 1001 / 1002 / 1003).**

| | −250 | −240 | −200 | −100 | −50 | 0 |
|---|---|---|---|---|---|---|
| intra-oceanic trench (k km) | 38 / 35 / 38 | 33 / 36 / 33 | 35 / 39 / 39 | 37 / 46 / 43 | 46 / 46 / 53 | 56 / 55 / 58 |
| of it around detached plate pieces | 0 | 0 | 2 / 0 / 1 | 12 / 7 / 15 | 20 / 14 / 34 | 24 / 28 / 27 |
| detached pieces (Mkm²) | 0 | 0 | 1 / 0 / 0 | 16 / 8 / 20 | 22 / 20 / 66 | 31 / 37 / 65 |
| shape ratio (Earth 1.51) | 1.21 / 1.19 / 1.24 | 1.22 / 1.20 / 1.25 | 1.34 / 1.31 / 1.27 | 1.82 / 1.66 / 1.78 | 2.10 / 1.81 / 1.95 | 2.19 / 1.98 / 2.19 |
| intra-oceanic / ocean–ocean boundary | 0.40 / 0.41 / 0.40 | 0.31 / 0.41 / 0.34 | 0.26 / 0.33 / 0.29 | 0.23 / 0.25 / 0.23 | 0.23 / 0.23 / 0.26 | 0.26 / 0.25 / 0.25 |
| ridge production (km²/yr; Earth 3.37) | 2.0 / 2.4 / 2.2 | 2.2 / 2.2 / 2.2 | 2.8 / 3.0 / 3.3 | 4.7 / 5.5 / 5.6 | 6.2 / 5.0 / 4.6 | 5.1 / 6.1 / 4.7 |
| consumption at intra-oceanic trenches (Earth 1.28) | 1.5 / 1.4 / 1.8 | 1.9 / 1.6 / 1.8 | 2.4 / 2.4 / 2.5 | 2.3 / 3.4 / 3.2 | 3.2 / 2.9 / 3.3 | 3.8 / 3.2 / 2.9 |
| consumption at continental-side trenches (Earth 1.46) | 0.0 | 0.1 / 0.2 / 0.1 | 0.2 / 0.3 / 0.4 | 2.0 / 1.5 / 2.1 | 2.6 / 1.8 / 1.2 | 1.2 / 2.4 / 1.3 |

Readings:
1. **The start converges too much.** At −250 and −240 Myr (12 plates, no fragments, shape ratio 1.2) the intra-oceanic
   trench is already 33–38k km, 1.6–1.8× Earth's 20.8k, and 0.31–0.41 of the ocean–ocean boundary converges (Earth:
   0.25 with OCB, 0.14 as trench). The seven oceanic start plates get random poles (`_init_supercontinent`:
   `axis = rng.normal(size=3)`, 3–7 cm/yr): random relative motion closes about as much ocean–ocean boundary as it
   opens. On Earth oceanic plates are driven by their own slabs and spread from each other.
2. **The growth after −200 Myr is plate fragmentation.** Plates end in 124–140 pieces; the pieces other than each
   plate's largest cover 31–65 Mkm² (6–13% of the surface, 97% oceanic), carry 99–127k km of boundary (35–53% of all)
   and 24–28k km of the intra-oceanic trench (43–50% of it). Without them the intra-oceanic trench is 27–32k km
   (1.3–1.5× Earth). A detached piece is not a plate: it keeps its parent's Euler pole thousands of km away. Seen in
   seed 1001, plate 24 between −18 and −17 Myr (`outputs/test-tmp/b36/frag_1001_24.png`): a plate inside it, bounded by
   ridges, transforms and trenches, grows (gap fill on its opening side, "younger ocean overrides" on its closing side)
   until it cuts its host in two in one step; in seed 1003 plate 3 (a continental start plate, 142 Mkm²) is spread over
   the globe in a dozen pieces (`pieces_1003.png`). The big plates' own perimeters are Earth-like (perimeter / cap
   perimeter, plates > 10 Mkm², median 1.49–1.75 vs Earth 1.56): the excess boundary is the pieces.
3. **Speeds follow the law, and the law is Earth's on Earth's geometry.** Planet speed / `_driving` target 0.98–1.00
   (p10–p90 0.75–1.27). The same law applied to Earth's plates with Bird's lengths (× 0.86, the planet's node-count
   factor: the law's trench length / contour trench length) gives Earth's speeds within 9% area-weighted (individual
   misses: AN 1.5 → 4.9, IN 5.6 → 0.9, EU 2.3 → 0.4 cm/yr). Planet oceanic plates are faster than Earth's at the same
   slab fraction of the perimeter (0.15–0.25: 6.5–8.1 vs AU 6.3; 0.05–0.15: 4.9–6.3 vs AN 1.5, IN 5.6) because the
   fragments add slab to their parents: the speed law is not the cause.
4. **Production and consumption.** Ridge production rises from 2.0–2.4 km²/yr at the start to 4.6–6.2 (Earth 3.37;
   Rowley 2002 3.4 since 180 Ma; reconstructions with a mid-Cretaceous peak reach ~5.5, Coltice et al. 2013): the
   excess is consumed at intra-oceanic trenches (2.9–3.8 vs 1.28); the continental side consumes Earth's 1.2–2.4 vs
   1.46. Ridge opening p50 43–52 mm/yr vs OSR 34.
5. **What the planet lacks is Earth's slow ocean–ocean convergence without a slab (OCB, 17.4k km, p50 7 mm/yr).** The
   planet's intra-oceanic trenches close at p50 50–65 mm/yr (Earth 60); only 0.06–0.18 of them close slower than
   20 mm/yr (Earth 0.17). Every oceanic convergence above 5 mm/yr is a trench with slab pull.
6. Downgoing crust at intra-oceanic trenches is p50 64–77 Myr old at 0 Myr (< 20 Myr: 0.10–0.18): not reversed ridges.
7. Pairs (0 Myr): the longest intra-oceanic trenches are 3–9k km; 5–13k km lie between two start plates; several of
   the longest are between the oceanic parts of mostly continental plates (continental share 0.4–0.65), formed in the
   last 20–100 Myr.
8. **Item 4 (trench length) is settled as:** 1.6–1.8× Earth's intra-oceanic length from the start's random poles,
   doubled to 2.6–2.8× by fragments; the continental side is Earth's.

**B-c probe: where the ocean's sediment lies and arrives (label-free ocean; 200 km classes).**

| class | Earth stock (GlobSed, bulk): area / mean / share | planet stock at 0 Myr (solid-equivalent): area / mean / share | planet net arrival, last 40 Myr: area / m/Myr / share |
|---|---|---|---|
| active-margin foot (continent and trench < 200 km) | 0.03 / 1,077 m / 0.05 | 0.02–0.04 / 1,603–2,296 m / 0.07–0.14 | 0.02–0.03 / 113–168 / 0.23–0.48 |
| passive-margin foot (continent < 200, trench ≥ 200 km) | 0.10 / 2,369 m / 0.39 | 0.04–0.05 / 2,223–3,096 m / 0.19–0.35 | 0.03–0.04 / 36–100 / 0.09–0.29 |
| intra-oceanic trench zone (continent ≥ 200, trench < 200 km) | 0.02 / 368 m / 0.01 | 0.13–0.14 / 227–510 m / 0.09–0.12 | 0.13–0.14 / 2.7–9.3 / 0.08 |
| open ocean | 0.85 / 390 m / 0.54 | 0.77–0.79 / 186–546 m / 0.43–0.63 | 0.80–0.82 / 1.0–9.1 / 0.16–0.60 |

Earth's ocean (label-free, 0.615 of the surface) holds a mean 610 m (bulk); the planet's (0.68–0.71) 337–687 m
(solid-equivalent at `rho_sediment` 2,400). Earth's sediment falls off with distance from continents 3.9 / 2.9 / 1.9 /
1.0 / 0.4 / 0.2× the mean (0–100 … > 1,600 km), the planet's 9.1 / 3.4 / 2.0 / 1.0 / 0.6 / 0.1× (seed 1001): steeper at
the edge and thinner far out (no pelagic sediment: `pelagic_m_myr` 0). Distance to trenches barely matters on Earth
(1.31× within 100 km). The probe's net arrival reproduces 70–84% of the ledger's ocean-floor sediment over the same
steps (1.58 / 5.11 / 4.29 vs 1.91 / 7.27 / 5.12 km³/yr); the gross carries resampling noise and is not read.

Readings:
1. **The fill lands at the feet of active margins out of proportion:** 23–48% of the net arrival on 2–3% of the
   ocean, 113–168 m/Myr, 1.1–4.6× the passive-margin feet's rate. `fill_and_spill` ponds each parcel at the nearest
   low, and beside an active margin that low is the trench. Earth's active-margin feet hold half the passive feet's
   stock (0.45); the planet's 0.6–0.8.
2. **Intra-oceanic trenches get little fresh sediment (8%) but subduct the open-ocean stock:** the thirty-fifth pass's
   1.89 km³/yr into intra-oceanic trenches is their consumption (2.9–3.8 km²/yr) × the stock the plates carry
   (~0.4–0.6 km). At Earth's intra-oceanic consumption (1.28) that term would be ~2.5× smaller: **B-b's fix is also
   about half of the sediment-subduction excess.**
3. A fan rule (B-c's candidate) would move sediment from the trench foot to the open ocean, where Earth holds 54% of
   its stock; it is not the first lever. The next levers on the sediment test are B-b (consumption) and the margin
   width (2b: the sediment spills off half Earth's thin-crust area).
4. Earth's ocean-floor age grid is not in `data/`; the age profile of GlobSed (e.g. against Müller et al. 2016's age
   grid, EarthByte) would give B-a and a pelagic rule an Earth reference by age.

**B-b candidate (proposed, not built): plates are connected.** A plate whose territory falls into separate pieces
becomes one plate per piece above a size floor (each piece keeps the parent's Euler vector at the split, then is
driven by its own boundaries; smaller pieces join the plate that surrounds most of their edge). On Earth a plate is
one rigid, connected piece of lithosphere; a piece cut off by another plate's growth moves on its own. Tests: detached
area ≤ 1% of the surface at every snapshot; intra-oceanic trench 12–25k km and continental side 20–35k at 0 Myr;
ridge production 2.5–4.5 km²/yr in the last 40 Myr; ridge length kept (Bird 94.8k ± 15%); plate count and size
distribution reported beside Bird's; the fixed B tests and the elevation and crust reviews. Second candidate, after
it, if the start still converges too much: start poles for the oceanic plates from their own slabs and ridges (the
`_driving` torque at t0) instead of random axes.

**B-a built (thirty-sixth pass, after the measurements), default off: `sediment_compaction`.** The route the user
agreed (measurements, then B-a) left one design point open: how deposits enter the erosion kernels. Taken: the kernels
are unchanged (they move solid heights within a plate step; the next elevation reads bulk), so `graph_landscape` and
the 11.53 km stage are untouched; the alternative (bulk heights inside the kernels) would need a porosity in every
routing kernel. With the flag:
- `sed` is the solid (grain) thickness, `rho_grain` 2,650 kg/m³. Every ledger stays in solid volume, as now; eroded
  crust becomes grain volume 1:1 (the grains are 6% lighter than the crust; before, a deposit was a layer of 2,400,
  i.e. 15% porous, so 15% of the eroded volume became pore space).
- `plates.bulk_thickness(S, phi0, c)` inverts Athy's law, S = h − φ0/c (1 − e^(−ch)) (Newton from S/(1 − φ0));
  `plates.sediment_column` gives every node's bulk height, dry mass (ρ_grain S + ρ_w (h − S), water-filled pores) and
  submerged excess mass ((ρ_grain − ρ_w) S). Laws (Sclater & Christie 1980): shale φ0 0.63, c 0.51 /km on oceanic
  crust; shaly sandstone 0.56, 0.39 /km on continental crust. A bulk 0.5 km deep-sea cover stands at ~1,750 kg/m³
  (solid 222 m), a 5 km basin at 2,250–2,280.
- `PlateSim.elevation` uses them: surface + bulk height, loads from the masses (the `above` test and the continental
  dry test on bulk height). Snapshots gain `sediment_bulk_m` (= `sediment_m` when off).
- Reviews that compare with CRUST1.0 or GlobSed read bulk (`crust_review.sediment_bulk_m`: total thickness, label-free
  continent, `margin_cross_section`); `elevation_check` takes the column from `plates.sediment_column`;
  `ocean_sediment_probe` reads the stock in bulk and the arrivals in solid volume. On the old runs every review
  printout is unchanged (`crust_review`, `elevation_check` diffed against `sweep-20261025-170555/reviews`).

Size per metre of cover: the deep floor stands h − 0.714 S above the uncovered floor, against 0.40 S before; for
S = 300 m (h 647 m) 433 m against 119 m. For thick piles h/S → 1 and the gain is small.

**Measured on the same states (new `scripts/compaction_effect.py`: a snapshot's crust, ages, sediment and dynamic
topography put back into a PlateSim and given both laws; the run's own law reproduces the snapshot's elevation within
p99 1.8–5.7 m).** Ring control of `sweep-20261025-170555` (`<sweep>/reviews/compaction_effect.txt`):

| seed, 0 Myr | deep floor (m) | water below the datum (m) | sea level on the datum (m) | ocean bulk sediment (m; GlobSed 610) |
|---|---|---|---|---|
| 1001 | 4,535 → 4,464 | 3,133 → 3,055 | −689 → −604 | 337 → 438 |
| 1002 | 4,474 → 4,359 | 3,202 → 3,088 | −774 → −636 | 588 → 776 |
| 1003 | 4,564 → 4,393 | 3,285 → 3,142 | −894 → −712 | 687 → 937 |

Mean: deep floor −119 m (37% of the 319 m gap to Earth's 4,205), water −112 m, sea level +135 m; at −20 Myr −81 /
−119 / −203 m. The thirty-fifth pass's ~40% holds. The ocean's sediment is mostly in thick piles at the margin feet
(B-c), where compaction gains little; the thin open-ocean cover, where it gains most, is thin on the planet (no
pelagic sediment). With compaction two seeds' bulk sediment exceeds GlobSed's 610 m by 27–54% (too much sediment
reaches the ocean floor: B-c, 2b).

**Pre-check (cloud, seed 1001, 250 Myr, `threads` 1; the cloud's trajectory equals the user's to at least −220 Myr,
then drifts):** B-a and its control both pass the elevation tests (young-ocean offset −48 vs −103 m, thin-crust
−184 vs −114 m), export 5.1 vs 5.0 km³/yr, volume −9.6% both, crust median 38.1 vs 37.8 km, label-free area 0.328 vs
0.326. Their deep floors (4,548 vs 4,510 m) differ the other way: the two runs' oceans differ in age (mean 53 vs 48
Myr), which outweighs the compaction gain. **So a sweep cannot read B-a's floor effect** (drift between runs >
effect); the floor test is the paired one above (passes: −119 m ≥ 100 m, not past 4,055 m), and `Compaction
Sweep.cmd` decides on the dynamics only. Its decision rule is written in the .cmd file (gates on 3 seeds, guards
against the control; floor, water, GlobSed, land > 2 km and export read beside). `crust_review`'s sediment read-out
now reads bulk too.

Tests (all pass in the device shell, 69 planet-layer tests): `test_bulk_thickness_inverts_athys_law`,
`test_compaction_is_off_by_default_and_reads_the_old_layer`, `test_compacted_sediment_is_a_porous_load` (n = 217, a
wide 500 m solid cover raises the deep floor and dry continental crust by the Airy values within 5%),
`test_compaction_run_keeps_both_ledgers`. Default path identical step by step to the pre-edit code at n = 48 (70
steps, `threads` 1, every snapshot field). `Compaction Sweep.cmd`: tests, GlobSed on n = 217, then the ring control's
settings with `sediment_compaction` False / True on 3 seeds and the reviews (elevation_check, crust_review,
margin_width, crust_budget, compaction_effect on the control, ocean_sediment_probe) into `<sweep>/reviews/`.

### Step 8, follow-up 1, thirty-seventh pass (26 Oct): B-a adopted; B-b built (plates are connected), default off; its pre-check; the start's convergence is a geometry question; `Plate Pieces Sweep.cmd` to run

User, 26 Oct: "Yes, you can do the next step" (B-b: detached pieces become plates; tests as in the thirty-sixth
pass). The user also added `data/EarthByte` (Seton et al. 2020 age and spreading grids, Müller et al. 2019 feature
files).

**Compaction sweep read (`sweep-20261025-201623`, reviews in `<sweep>/reviews/`): B-a adopted.** 69 tests passed on
Windows. The control equals `sweep-20261025-170555`'s ring control field by field (only the recorded overrides
differ). By the rule in `Compaction Sweep.cmd`: gates pass on 3 of 3 seeds for both (young-ocean offsets −194 / −137 /
−96 m with B-a); guards hold (label-free median 39.6 km; volume −12.5% vs the control's −12.6%; margins pass: thin per
km 120, abrupt 14%, edge T 25.5, inner T 33.8). Beside: B's deep-floor test now passes (4,317 m vs Earth 4,205 ± 150;
control 4,524), water below the datum 3,036 m (control 3,206; Earth 2,629), sea level −567 m on the datum (control
−786), ocean bulk sediment 494 / 497 / 717 m (mean 569; GlobSed 610), > 45 km 0.310 (control 0.304, within seed
spread), export 5.9 (6.2). Sediment subducted in the last 40 Myr rises 3.52 → 4.13 km³/yr (Earth ~1.0): more
sediment-borne mass leaves at trenches when the floor is lighter; B-b's consumption is the lever (thirty-sixth pass).
Changes: PARAMS `sediment_compaction` True; `test_compaction_is_the_default_and_off_reads_the_old_layer`;
`test_continental_sediment_is_a_sediment_load` pinned to the uncompacted layer (the compacting load has its own test).
**Reviews and old runs:** the reviews built a run's parameters as today's PARAMS plus the overrides in its
stats.json, so after a default change an old run would be reread with the new law. Now `simulate_plates` writes
`params_effective` (every parameter) into stats.json, and `crust_review.run_params` (used by `elevation_check` too)
reads old runs with `LEGACY_DEFAULTS` (`sediment_compaction` False). Checked: `elevation_check` on the B-a sweep's
control prints the saved review line for line.

**Earth's ocean floor by age (new `scripts/earth_seafloor_age.py`, numpy only: the 2′ `.grd` is classic netCDF, read
by a 60-line CDF1 reader; the `.nc` files are HDF5).** Same numbers as h5py on the `.nc`. Dated floor 292.8 Mkm² (0.574
of the surface), mean age 63.9 Myr, median 52.8. Production as area younger than A / A: 2.96 / 2.93 / 2.99 / 3.04 /
2.89 / 2.59 km²/yr for A = 2 / 5 / 10 / 20 / 40 / 60 Myr (Bird's 3.37 includes continental rifts; the test band 2.5–4.5
holds both). Area by 20 Myr bins 0.21 / 0.19 / 0.14 / 0.13 / 0.09 / 0.09 / 0.08 / 0.05 / 0.02 / 0.01. GlobSed (bulk) by
floor age on n = 217: median 61 m at 0–10 Myr, 100–125 m at 10–50, 200–290 m at 50–100, 300–480 m at 100–160 (means 147
→ ~1,100 m, margins included): ~3 m/Myr, the reference for a pelagic rule (`pelagic_m_myr` is 0). Cached in
`outputs/planet-reference/earth-seafloor-age-n217.npz` / `.json`. The planet's ocean floor is younger and has an old
tail: mean 50–55 Myr, 0.25–0.33 of it under 20 Myr (Earth 0.21), max 344 Myr (Earth ~180–280 in fragments).

**B-b built, default off: `piece_rule` 'plates', `piece_min_area_mkm2` 1.** After the rasters are updated each step
(`_separate_pieces`, before boundary typing and motion): every plate's world territory is split into connected pieces;
each piece other than the largest that has an interior node (all its neighbours on the same plate)
- becomes its own plate if it covers ≥ 1 Mkm² (`split` with zero relative speed: same frame, the parent's Euler
  vector at the split, then `_driving` on its own boundaries); Bird's 20 plates ≥ 1 Mkm² hold 98% of Earth's surface;
- else joins the plate around most of its edge (`_weld`: as `merge` for a part of a plate; the capturing plate's
  raster reads the piece's crust by backward lookup, replaces its own oceanic crust under it, stacks on its continental
  crust; seam cracks filled from a set neighbour; the capturing plate keeps its motion: microplate capture, e.g. the
  Monterey, Arguello and Magdalena microplates by the Pacific plate, Nicholson et al. 1994). Volume changes go to the
  `pieces` ledgers (crust and continental sediment);
- a piece with no interior node is left: staircase crumbs of 1–5 nodes on boundaries (measured on the B-a sweep, seed
  1001, −2 → −1 Myr: < 0.01 Mkm² pieces cover 0.36 Mkm², all on typed boundaries, 65% still there a step later;
  0.01–0.1 Mkm² pieces 91% and larger ones 99–100% persist: real lithosphere). No new number.
Pieces of plates in a running extension event are left to E. Runs record `detached_area_frac` and `piece_stats` every
step. Size of the problem it addresses on the B-a sweep at 0 Myr: 122–310 pieces per seed; ≥ 1 Mkm²: 8–11 pieces,
24–39 Mkm²; 0.1–1 Mkm²: 7–13 pieces, 2–6 Mkm²; < 0.1 Mkm²: 0.7–1.6 Mkm².

Tests (5 new; 74 planet-layer tests pass in the device shell): `test_piece_rule_shared_is_the_default`,
`test_large_detached_piece_becomes_a_plate_with_its_parents_motion`, `test_small_detached_piece_joins_the_plate_around_it`
(oceanic and continental pieces; Q = I so the volume moves node for node), `test_boundary_crumbs_are_left_alone`,
`test_plates_rule_run_keeps_the_ledgers_and_connected_plates` (60 steps, n = 48: two plates made, two welds, each weld
books exactly; detached ≤ 1% every step). Default path identical step by step to the code before B-b (n = 48, 70 steps,
four rifts, every snapshot field, ledgers). **Found on the way, not fixed (default path, before B-b):** the
continental-sediment ledger drifts by 1e-5–3e-4 of the stock over 60 steps at n = 48 on seeds 3 / 4 / 5 / 7 (in one
step after a rift on seed 5: 5.9e13 m³); it sits in `_update_rasters`, most likely raster nodes that are read (and get
increments) in the same step their crust is consumed, booked at their pre-increment value. The existing ledger tests
(15 steps, seed 3) do not reach it. To fix separately (it changes ledger numbers, not dynamics).

**Pre-check (cloud, seed 1001, 250 Myr, n = 217, `threads` 1, B-a on; ~12 min each; reviews: `plate_pieces_review`,
`trench_origin`, `crust_review`, `crust_budget`; no ETOPO there, so no elevation gates).**

| | −250 | −240 | −200 | −150 | −100 | −50 | −20 | 0 | window −100…0 mean |
|---|---|---|---|---|---|---|---|---|---|
| intra-oceanic trench, control (k km) | 38.1 | 32.9 | 36.0 | 49.3 | 46.5 | 46.5 | 31.5 | 25.1 | 39.1 |
| intra-oceanic trench, B-b | 38.1 | 32.9 | 33.7 | 37.8 | 34.4 | 32.2 | 36.7 | 34.5 | 33.7 |
| detached pieces control / B-b (Mkm²) | 0 / 0 | 0 / 0 | 1.1 / 0.1 | 3.5 / 0.1 | 7.8 / 0.8 | 5.6 / 0.4 | 4.5 / 0.4 | 7.1 / 0.6 | 6.3 / 0.6 |
| shape ratio control / B-b (Earth 1.51) | 1.21 | 1.22 | 1.34 / 1.29 | 1.50 / 1.34 | 1.71 / 1.48 | 1.72 / 1.48 | 1.71 / 1.54 | 1.64 / 1.53 | 1.71 / 1.51 |

B-b tests: T1 detached max 0.0034 of the surface (control 0.023) pass; on the window means T2 33.7k (control 39.1k),
T3 continental side 34.0k (27.1k; band 20–35), T4 production 4.72 (4.73), T5 ridge 83.6k (81.9k): passes the sweep rule
below on this seed. At 0 Myr alone: T2 34.5k vs the control's 25.1k, T3 46.7k vs 29.1k, T4 5.07 vs 4.48: one snapshot
swings by 10–20k km (the control fell from 46.5k to 25.1k in 50 Myr), hence the window. Plates 22 (≥ 1 Mkm²: 20; Bird
52 / 20), largest 63 Mkm² (control 86; Bird 104.6); 2 plates made, 207 welds (18 Mkm² over the run). Guards on one
seed: label-free median 39.3 km (control 38.1), continental volume −10.8% vs −9.6% (1.2 points: the guard's edge, one
seed), area 0.306 vs 0.328; deep floor 4,327 m (control 4,548). **Watch-point:** the continental-side trench is longer
with B-b (42.3k vs 29.2k km over the last 40 Myr; Earth 30.7k), collision 10.4k vs 4.8k.

**The start's convergence is not B-b's, and the plan's second candidate fails.** With B-b the intra-oceanic trench
stays at 32–38k km for the whole run, the level the start has at −250 Myr before any piece exists. Probed on the
start (device shell, 3 seeds, no model change): random poles give intra-oceanic trench 38.1 / 34.9 / 37.5k km, 0.40–0.41
of the ocean–ocean boundary (Earth 0.14 as trench); relaxing the oceanic plates' poles to the fixed point of the force
law (`_driving` on `classify` types, 80 iterations from rest) makes it worse, 46.9 / 42.3 / 42.2k (0.45–0.50): the
law pulls a plate toward its own ocean–ocean trench and that trench closes faster. Ridge push only (each oceanic plate
pushed away from its ocean–ocean boundaries, which the start's age field makes ridges) gives 30.5 / 34.6 / 32.2k (0.32–0.41).
The cause is the start's geometry: the supercontinent's plates are Voronoi cells that extend into the ocean, so its
coast is a passive margin inside plates (continental-side trench at −250 Myr: 0.2 / 0.7 / 0.7k km; Earth 30.7k), and a
closed sphere must consume what it opens, so all closing is ocean–ocean. Pangea was ringed by subduction (the
Panthalassa margin; e.g. Collins 2003). **Design question for the user (not built):** a start whose continental plates
end at the supercontinent's coast on most of its rim, with the ocean side subducting there (trench on the oceanic
plates' edge), instead of plates that reach into the ocean. Tests would be B-b's T2 / T3 at −240 Myr and over the run.

**`Plate Pieces Sweep.cmd`** (tests, `earth_seafloor_age`, `piece_rule` shared / plates on 3 seeds with the ring
control's settings and `sediment_compaction` True, then `plate_pieces_review`, `elevation_check`, `crust_review`,
`margin_width`, `crust_budget`, `trench_origin` and `ocean_sediment_probe` per combination into `<sweep>/reviews/`).
Rule fixed before the runs: B-b is adopted if T1 passes on every seed, T2–T5 as 3-seed window means (−100…0 Myr,
every 10 Myr) are each in their band or no farther from it than the control's, the gates pass on 3 seeds and no guard
fails (median 34–40 km, volume no worse than the control's by more than 1 point, margins as the control's). The
control must equal the B-a combination of `sweep-20261025-201623` field by field (new keys only: `params_effective`,
`detached_area_frac`, `piece_stats`). Why the window and "no farther": the start carries 33–38k km of intra-oceanic
trench that B-b cannot remove (above); the plan's forms are printed beside.

**`Plate Pieces Sweep.cmd`, first try (26 Oct, night): stopped at the tests, no run started.** On Windows
`test_plates_rule_run_keeps_the_ledgers_and_connected_plates` failed (73 of 74 passed): seed 3 at n = 48 made two
plates but welded no piece in 60 steps. In the device shell the same test welds at steps 47 and 57 only; first welds
come after 35 / 66 / 60 / 52 / 35 steps on seeds 3–7 (5–18 welds by 150 steps), so a fixed 60 steps sat at the edge
and another machine's trajectory (numpy 2.5.3 on Windows vs 2.2.6 there, ducc0 threads) missed them. Not a model
fault. **Test changed:** the run goes at least 60 steps and stops once a plate is made and two pieces are welded (at
most 150); every check is unchanged. Checked over 150 steps on seeds 3–7 in the device shell: each weld books exactly,
whole-run ledger 3–7e-4 (bound 2e-3), no pinholes; seed 3 detached max 0.0054. 74 tests pass there. **Watch-point
found on the way (n = 48 only):** seed 7 goes over the test's 1% detached bound at steps 109–149 (max 0.0142); all of
it is boundary crumbs (no interior node: one-node-wide slivers of 4–28 nodes, up to 1.1 Mkm² at n = 48, where a node
is ~200 km), which B-b leaves by design. The test's seed 3 stays under it. At n = 217 the pre-check's maximum was
0.0034; T1 in the sweep reads it on 3 seeds. If a sweep seed fails T1 by crumbs, the crumb rule (interior node) is
the thing to revisit, not the threshold. Rerun `Plate Pieces Sweep.cmd` as it is.

### Step 8, follow-up 1, thirty-eighth pass (26 Oct, evening): B-b adopted; the start ringed by subduction built (default off) with start poles from the force law; ledgers closed; `Coast Start Sweep.cmd` to run

User, 26 Oct evening: Claude may decide the start question itself ("You can decide the best course of action"), and
**decisions go to the user only when they are very important** (now a working rule in the handoff).

**Plate Pieces sweep read (`sweep-20261025-220434`, reviews in `<sweep>/reviews/`): B-b adopted.** 74 tests passed on
Windows. The control equals `sweep-20261025-201623\sediment_compaction-True` field by field on all three seeds (only
`params` and run time differ). By the rule in `Plate Pieces Sweep.cmd`: T1 passes on every seed (detached max 0.0018 /
0.0024 / 0.0033 of the surface; control 0.060 / 0.095 / 0.087); window means −100…0 Myr T2 41.4k (control 55.5k),
T3 30.7k (28.3k, both in band), T4 5.15 (5.72), T5 91.1k (97.7k): each in its band or closer to it than the control;
shape ratio 1.57 (Earth 1.51; control 1.98). Gates pass on 3 of 3 (young-ocean offsets −296 / −136 / −100 m); guards
hold: label-free median 39.2 km, continental volume −10.8% (control −12.5%), margins pass (thin per km 115, abrupt
13%, edge T 25.6, inner T 33.6). Beside: > 45 km 0.270 (control 0.310), area 0.318 (0.304), deep floor 4,411 m (4,317),
sediment subducted in the last 40 Myr 3.43 km³/yr (4.13; Earth ~1.0), plates 25–33 (≥ 1 Mkm²: 20–26; Bird 52 / 20),
11 plates made and 213 welds per seed on average. At 0 Myr alone T2 41.2k, T3 37.4k and T4 5.66 are out of band: the
start's excess is still there (below). Changes: PARAMS `piece_rule` 'plates';
`test_piece_rule_plates_is_the_default_and_shared_keeps_pieces`; `crust_review.LEGACY_DEFAULTS` reads runs that saved
only their overrides as 'shared'; audit row.

**Start geometry, built (default off): `start_plates` 'coast'.** The same plate seeds grow only within their own
domain (`nearest_label` with infinite distance across the coast): the five continental plates end at the coast, the
seven oceanic plates fill the ocean, and an ocean (or continent) component without a seed gets its own plate (none
on seeds 1001–1003). Same random draws in the same order as 'voronoi', so 'voronoi' is the old start node for node.
Probe on the start (device shell, n = 217, no ETOPO needed; contour typing as `trench_origin`):

| −240 Myr (seeds 1001 / 1002 / 1003) | intra-oceanic (k km) | continental side (k km) | coast typed subduction |
|---|---|---|---|
| voronoi (as now) | 32.9 / 35.9 / 33.0 | 2.0 / 4.1 / 3.3 | – (coast inside plates) |
| coast, random poles | 24.4 / 26.2 / 22.0 | 22.8 / 43.9 / 31.3 | 0.57 / 0.75 / 0.70 |
| coast, poles from forces | 9.3 / 23.1 / 7.9 | 39.3 / 42.4 / 42.3 | 0.76 / 0.77 / 0.89 |

**Cloud pre-check, seed 1001, baseline settings, 250 Myr (one seed; the cloud's trajectory is its own).** A first
pair ran on PARAMS defaults by mistake (K 0.75, θ 6: export 23–26, land 20%) and is not read. With the baseline:
'coast' with random poles keeps the intra-oceanic trench at 24–26k km until −150 Myr (control 33–38k), then the plate
count rises to 36 and it reaches 38–45k; window means T2 40.8k (control 33.7k), T3 35.3k (34.1k), T4 5.20 (4.72) are
all farther from Earth than the control's, and the young-ocean gate fails (−524 m): where an oceanic plate moves away
from the coast the rim opens as a ridge right at the continent's edge, young ocean against full-thickness crust.
That is not Pangea's rim either.

**`start_poles` 'forces' (built, default 'random').** The oceanic start plates' speed and pole come from the plate
layer's own force law (`_driving`, fully relaxed: `relax=1.`) on the start's geometry with the slabs Pangea had: ocean
nodes on the coast are trench, ocean-ocean boundary nodes are ridges (as the start's age field already has them).
Continental plates keep their dispersal motion. No random draws, no new number. Why this is not the fallback that
failed in the thirty-seventh pass: there the force law ran on the Voronoi geometry, whose trenches are ocean-ocean, so
it pulled plates into them; here the trenches are the coast. Pre-check (seed 1001, baseline):

| seed 1001, cloud | control (voronoi, random) | coast, random | coast, forces |
|---|---|---|---|
| start −240: continental side / intra-oceanic (k km) | 2.0 / 32.9 | 22.8 / 24.4 | 39.3 / 9.3 |
| window −100…0: T2 / T3 (k km) | 33.7 / 34.1 | 40.8 / 35.3 | **32.9 / 27.7** |
| window T4 production (km²/yr) / T5 ridge (k km) | 4.72 / 83.6 | 5.20 / 95.7 | **4.23** / 81.3 |
| young-ocean offset (m), gates | −218, pass | −524, FAIL | −147, pass |
| continental volume / label-free area / median | −10.8% / 0.306 / 39.3 | −5.1% / 0.321 / 40.2 | −7.0% / 0.318 / 38.9 |
| > 45 km (label-free) | 0.314 | 0.362 | 0.335 |
| ocean floor mean age (Earth 63.9 Myr) | 55 | 52 | 57 |
| floor area by age, 20 Myr bins (Earth 0.21 0.19 0.14 0.13 0.09 0.09 0.08 0.05 0.02 0.01) | 0.26 0.23 0.17 0.10 0.06 0.06 0.04 0.03 0.02 0.02 | 0.29 0.22 0.16 0.11 0.07 0.05 0.04 0.03 0.02 0.02 | 0.23 0.20 0.16 0.12 0.10 0.08 0.05 0.03 0.02 0.01 |
| plates at 0 Myr | 22 | 36 | 22 |

'coast' + 'forces' passes every B-b window test (T3 and T4 inside their bands; T4 4.23 against Earth's 2.9–3.4),
the gates, and the guards on this seed; its ocean-floor age distribution is the closest to Seton et al. (2020) of any
run so far. Beside: deep floor 4,420 m (control 4,327; the start's floor is old at the coast: 4,596 m at −250),
> 45 km 0.335 (0.314), margin review as the control's (both fail inner T on this seed), sediment subducted 4.0 km³/yr.

**The start test, restated before the sweep (and why).** The thirty-seventh pass proposed T2 / T3's present-day
bands at −240 Myr. They are not a measurement of Pangea's world: it was ringed by subduction (Collins 2003, EPSL 205;
Cawood 2005, Earth-Sci. Rev. 69: the Terra Australis orogen alone ~18,000 km) and held more subduction than today,
~65.7k km at ~235 Ma against ~42.4k today counted the same way from slab remnants (van der Meer et al. 2014, PNAS
111, 4380, Table S2); no measured split into intra-oceanic and continental-side exists for it. So the start test is
one-sided and about what the change is for: S1 continental-side trench ≥ 20k km (T3's lower edge) and S2 its share
of all subduction ≥ 0.60 (Earth today 30.7k of 51.5k), 3-seed means at −240 Myr. This was set after seeing seed
1001 (coast + forces: 39.3k, 0.81; coast + random: 22.8k, 0.48; control 2.0k, 0.06), so the run-wide tests, which keep
their present-day bands, carry the decision as much as the start does. `plate_pieces_review` prints it as the
"start" line under each Decision line (−250 Myr and the total beside).

**Ledgers closed (the small fix from the thirty-seventh pass).** Two causes, both in `_update_rasters`: (1) plate nodes
consumed this step were booked in pass 1 at their value then, but can still be read by a world node of the same plate
(the forward and backward nearest-node maps are not exact inverses) and take this step's increments, or turn
continental (arcs), before they leave: now the difference is booked when they leave (to `collision_out` or
`cont_subducted`, as in pass 1); (2) collision volume landing on new crust was booked as `cont_lost_other` although
its removal was already booked in `collision_out`: a double count, now counted in
`erosion_totals['collision_volume_unread_m3']` only. Both ledgers then close to round-off (n = 48, 60 steps: crust
1e-4 → 1e-16, sediment 2e-4 → 1e-15 of the stock on seeds 3 / 5 / 7); fields unchanged (the default path is identical
step by step to the code before this pass, 40 steps, every snapshot field). New `test_ledgers_close_over_a_long_run`.
Ledger-based review numbers (residuals, `crust_budget`) of new runs differ slightly from old ones.

Tests: 79 planet-layer tests pass in the device shell (new: `test_start_plates_voronoi_is_the_default`,
`test_coast_start_plates_end_at_the_coast`, `test_coast_start_rim_subducts_and_keeps_the_ledgers`,
`test_start_poles_forces_moves_ocean_plates_toward_the_ring`, `test_ledgers_close_over_a_long_run`).

**`Coast Start Sweep.cmd`** (tests, then `start_plates` × `start_poles` = voronoi:random (control), coast:random,
coast:forces on 3 seeds with the B-b settings, reviews into `<sweep>/reviews/`). Rule fixed before the runs: a
candidate passes if (a) the start passes S1 and S2, (b) T1 on every seed and T2–T5 window means each in band or no
farther than the control's, (c) the gates pass on 3 seeds, (d) no guard fails (median 34–40 km, volume no worse than
the control's by more than 1 point, margins as the control's). If both pass, the one with more of T2–T5 strictly in
band; tie: the smaller window T2. The control must equal `sweep-20261025-220434\piece_rule-plates` field by field
except the overrides, `params_effective`, run time and the ledger keys changed above. If adopted, both PARAMS change
(`start_plates` 'coast', and `start_poles` 'forces' if that one wins), with default tests and audit rows.

**Next after it:** arcs at Earth's rate (thirty-fifth pass) and the sediment-subduction excess (3.4–4.1 km³/yr vs
~1.0); then C; E; the rest of 2b. Watch-points: the ringed start's deep floor (older at the coast), > 45 km share,
and whether the plate count rises late in the run (coast + random reached 36).

### Investigation (26 Oct, night): the pattern shared within a seed is the dynamic-topography field; it is plate-blind and its spectrum is the inverse of Earth's

**User report:** a noise pattern repeats across runs of one seed with different settings (not across seeds), at the
same checkpoint, but not between the start and the end of one run; its height can differ. Suspected link: the
dynamic-topography panel is identical within a seed.

**Cause (by construction, not a stray RNG):** `Mantle` has its own stream, `default_rng([seed, 7])`, and draws every
knot field when it is built (`dyn_std_m` 500, `dyn_lmax` 30, `dyn_slope` 1; knots every `dt_knot_myr` 40 from t0 − 40:
−290, −250, …, −10, +30) plus the hotspots; `advance` draws plume births from the same stream, depending only on t and
dt. Nothing in it reads the plates, so the field is bit-identical in every run of a seed whatever the settings (max |Δ|
0 on the snapshots of `sweep-20261026-184745`), and `elevation()` adds it to every node unflexed (`z = z + self.dyn`).

**Measured** on `sweep-20261026-184745` (A voronoi:random vs C coast:forces, 3 seeds; high-pass = elevation minus a
mask-normalized smooth of ~550 km rms radius, land and ocean separately; scripts, tables and figures in
`outputs/claude-sync/dyn-probe/`):
- Ocean, −100 / −50 / 0 Myr: same-checkpoint correlation between settings 0.13–0.32; with the field subtracted
  −0.02…+0.05; cross-seed control −0.02…+0.01. Slope of elevation on the field 0.77–1.03 (it passes at full height);
  it is 20–30% of the ocean's high-pass variance. It is the whole shared pattern.
- Land: slope 0.83–1.01 too, but only 4–8% of land's high-pass variance (crust structure dominates): correlation
  0.00–0.09 raw, −0.03…+0.05 without it. On land the eye catches it where it moves near-sea-level platforms across
  the coast colours (the round inland seas and green patches of `maps.png`). Before ~−200 Myr the shared start
  dominates on land (0.70–0.91 at −240).
- Time (seed 1001, ocean, A at t1 vs C at t2): non-zero only on the diagonal (0.20–0.32 from −200 on; off-diagonal
  |r| ≤ 0.04), as the field's own lags (independent knots; −10 and 0 share a knot, r 0.98). Hence "same checkpoint
  across runs, not start vs end of one run".
- Amplitude pulses with the knots: global rms 500–514 m at a knot, 360–380 m halfway (two independent fields blended
  linearly keep √(f² + (1 − f)²) of the std), so the pattern's height changes on a 40 Myr cycle, and the blobs fade
  in and out in place instead of moving.

**Against Earth:**
- Spectrum: the code's coefficient variance ∝ l⁻¹ (flat power per degree) puts 12% of the variance in degrees 1–3
  and 67% in 10–30 (wavelengths ~1,300–3,900 km; a blob is half a degree-30 wavelength, ~650 km). Earth's residual
  topography peaks at degree 2 (0.46–0.76 km²), holds ~85% of its power in degrees 1–3 and falls by more than 10× to
  ~0.02 km² at degree 30 (Valentine & Davies 2020, G-cubed 21, e2020GC009240). No single power law fits both ends: a
  slope of ~3 gives 88% in degrees 1–3 but a degree-30 / degree-2 ratio of 0.004 vs Earth's ~0.03.
- Pattern: Earth's long-wavelength dynamic topography is set by subduction history: lows above subducted slabs
  (circum-Pacific, southern Eurasia), highs far from convergent margins (Africa, central Pacific), changing slowly
  (Rubey et al. 2017, Solid Earth 8, 899: e.g. ~−500 m over ~25 Myr in the Chukchi Sea, ~20 m/Myr). The planet's
  field has no relation to its trenches.
- History: degree 30 / slope 1 was chosen on 17–18 Oct because it raised the texture score; this doc already noted
  then that "the texture score was passed by round mantle swells". `dyn_std_m` 500 is a fitted value in the audit.

**Is the sharing harmful? Partly (user, 26 Oct night: not convinced it is a positive thing; agreed).** Pairing
(common random numbers) would only cut noise in paired differences, and the sweeps compare 3-seed means of runs that
drift apart from the first steps, so little is gained. Costs: (1) the field is the one input that stays locked while
everything else in a seed decorrelates, so every sweep since 17 Oct was decided against the same three mantle
histories: adopted values may partly fit where these blobs sit relative to these continents (reused realizations);
(2) each setting is tested against a forcing that cannot respond to it; (3) it shows as a fixed imprint across
previews. Remedies: D makes the field setting-dependent by mechanism; and the adopted defaults get one check on
held-out seeds (e.g. 1004–1006) before the next large change.

**Proposed, not built: D, dynamic topography from the slabs** (`dyn_rule` 'slabs'; default 'random' until a sweep):
1. Long wavelengths from subduction history: each step the volume consumed at each trench node enters a mantle-frame
   slab record at that position; the record sinks (rate from slab-remnant studies, to source before building) and
   its surface effect is negative with a kernel that widens and weakens with depth; the mean is removed (the
   complement is the broad highs far from slabs). Amplitude and spectral shape from Earth (Valentine & Davies 2020,
   full text, in their normalization), never fitted.
2. Short wavelengths: a random part at Earth's share (degree-30 power ~1/30 of degree 2), blended with constant
   variance (a·cos + b·sin, not linear), so nothing pulses.
3. Hotspots unchanged.
Tests, fixed before building: (a) share of the field's variance in degrees 1–3 ≥ 0.6 (Earth ~0.85), degree-30 power
≤ 0.1 × degree 2; (b) lows above slabs: area-weighted correlation of the field with distance to slabs subducted in
the last ~100 Myr positive (low near slabs); (c) rate at fixed points median ≤ 20 m/Myr; (d) global rms constant
within ±10% over a run; (e) gates and guards. Land share, flooded continental crust and texture reported, not
scored: they were tuned with the blobs, and lower window relief would be the honest number.
Cheaper control in the same sweep: `dyn_slope` ~3 (Earth's share in degrees 1–3, still plate-blind, no blobs).

**Where it goes:** after reading the Coast Start sweep (item 1), before C: C's elevation tests and the lowland /
flat-window checks read through this field (the 18 Oct staircase followed plates drifting across the degree-30 field).

### Held-out seeds check read (26 Oct, night): the crust and elevation numbers hold on new seeds, the plate geometry does not; Coast Start not adopted on either seed set; B-b to re-decide on six seeds

`Held Out Seeds Check.cmd` → `sweep-20261026-200857` (seeds 1004–1006, never used; baseline voronoi:random and
coast:forces, the Coast Start overrides; reference `sweep-20261026-184745`, same combinations on 1001–1003). Reviews in
`<sweep>/reviews/`; the flag table is `heldout_flags.txt` there. `trench_origin` crashed on its last table (a top pair
with < 500 km of trench at every listed time had no "since"); fixed (prints "-"), rerun, output complete.

**Rule (1), adopted gates and guards on the new seeds, baseline:** elevation tests pass on 1005 / 1006, **fail on
1004** (young-ocean offset −313 m vs the ±300 m band; −296 / −136 / −100 on the old seeds): 5 of 6 over six seeds.
Guards hold: label-free median 39.4 km, margins pass (thin per km 106, inner T 33.3). The gate sits at the edge of
the seed spread (six-seed offsets −100 … −313 m, mean −193): every "3 of 3 gates" decision since 24 Oct was a
three-draw sample of a gate one seed in six fails. It reopens no single adoption (ring, B-a and B-b all required it;
node typing failed it on 2 of 3), but "gates 3 of 3" is not a reliable selector on three seeds.

**Rule (2), headline numbers, baseline: 3 flags → decide again on six seeds.** In range (reference min–max widened by
half each side): > 45 km 0.254 (ref 0.235–0.313), young-ocean offset −209 m, water below the datum 3,067 m, volume
−8.9% (edge: allowed to −8.8), crust median 39.4, land 30.0%. **Flagged, all plate geometry (B-b's window tests,
−100…0 Myr):** T2 intra-oceanic trench 35.6k (ref 39.7–43.7k), T4 production 4.03 km²/yr (ref 5.01–5.36), T5 ridge
72.8k (ref 88.7–94.6k). Seed 1004 drives them (13 plates at 0 Myr, ridge 57.5k, production 3.1), but 1005 / 1006 sit
low too on T4 / T5. Reading: the old three seeds were unusually alike in plate geometry (production 5.01–5.36), so the
widened range is narrow; the plate layer's seed spread is larger than three seeds showed. **The flagged numbers are
not the dynamic-topography field's** (plate geometry comes from the plate stream and the force law); the crust and
elevation numbers, which the field does touch, all held. So the held-out check points at small samples, not at the
shared mantle.

**coast:forces on the new seeds: 0 flags** (every headline number and T2–T5 window inside the widened reference range);
the ringed start's runs are more alike across seeds than the voronoi start's.

**Rule (3) / item 1, Coast Start sweep (`sweep-20261026-184745`) read by its rule: neither candidate is adopted.**
coast:forces passes the start (S1 41.3k, S2 0.75; new seeds 43.7k, 0.76) and T1–T4 against the control, but **fails T5**
(ridge window 71.1k vs control 91.1k; band 80.6–109.0k), and fails the gates on 2 of 3 seeds (offsets −309 / −304 m).
coast:random fails the start (S2 0.57) and T5. On six seeds the same: coast:forces T5 70.9k vs control 81.9k (in band),
T2 28.5k vs 38.5k, T4 4.09 vs 4.59; gates 3 of 6 (offsets −309, −304, −336). It does move the B tests the right way
(volume −2.1 … −5.3% vs −5.5 … −11.5%, water below the datum 2,964 / 3,064 m vs 3,040 / 3,067, label-free area
0.343 / 0.350 vs 0.318 / 0.324) and its new-seed margins fail (inner T 35.4). Watch: its short ridge (fewer, larger
oceanic plates spread less) and its deep young-ocean offsets are the two things to fix if the ringed start comes back.
PARAMS unchanged (`start_plates` 'voronoi', `start_poles` 'random').

**Next:** `B-b Six Seeds Check.cmd` (piece_rule plates / shared on 1004–1006; plates must equal this sweep's control
field by field; rule in the .cmd, on six-seed means with 1001–1003 from `sweep-20261025-220434`). Proposed, for the
user (machine time): decide future sweeps on six seeds (1001–1006), about twice the run time.

### Step 8, follow-up 1, thirty-ninth pass (27 Oct): B-b kept on six seeds (the gate's one miss was a small-sample reading; the young-ocean and thin-crust offsets now pool the window's snapshots); B-d, island arcs at Earth's rate, tests fixed and value taken from Earth; pre-check; `Arc Rate Sweep.cmd`; sweeps decide on six seeds from now on

**B-b Six Seeds Check read** (`sweep-20261026-204140`, seeds 1004–1006, reviews in `<sweep>/reviews/`; 1001–1003 from
`sweep-20261025-220434`). Identity: `piece_rule-plates` equals `sweep-20261026-200857\start_plates-voronoi_start_poles-random`
on every field of every snapshot (62 snapshots × 18 fields per seed, NaN-aware; `stats.json` differs only in `params`
and `seconds`).

Six-seed means against the rule in the .cmd (plates / shared):

| | plates | shared | band / rule |
|---|---|---|---|
| T1 detached max per seed | 0.0013–0.0033 | 0.042–0.076 (1004–1006) | ≤ 0.01 on every seed: pass |
| T2 intra-oceanic trench, window | 38.5k km | 47.4k | 12–25k or closer than shared: pass |
| T3 continental-side trench | 29.4k | 31.2k | 20–35k: pass |
| T4 production | 4.59 km²/yr | 5.39 | 2.5–4.5 or closer: pass |
| T5 ridge | 81.9k km | 89.0k | 80.6–109.0k: pass |
| gates (elevation tests), as the .cmd read them | 5 of 6 (1004: young-ocean offset −313 m) | 6 of 6 | ≥ 5 and no fewer than shared: **FAIL** |
| crust median (label-free) | 39.2 / 39.4 km | 39.6 / 39.4 | 34–40: pass |
| continental volume | −9.9% | −10.4% | no worse by > 1 point: pass |
| margin review | passes on both seed sets | fails on 1004–1006 (inner T 35.8) | as shared's: pass |

So by the .cmd's letter B-b goes back to 'shared' on one clause: one seed's young-ocean offset, 13 m beyond the
±300 m band. **Looked at before applying it (total re-evaluation when something looks off):**
- The test's sample is young ocean (< 20 Myr, anomaly < 1 km, sediment < 500 m) within 100 km of a coast. On seed
  1004 with B-b it is 0.064% of the surface at −39 Myr and 0.23% on average (other runs 0.35–0.76%); the −39 Myr
  snapshot alone reads −575 m, the other eight −207 … −389 (mean −280). The statistic was the mean of per-snapshot
  medians, which gives that 0.33 Mkm² snapshot the weight of one ten times larger.
- Paired, B-b does deepen the offset on two seeds (1001 −296 vs −194, 1004 −313 vs −133; others within 20 m; mean
  −49 m pooled). Split by setting (`yo_probe`, seeds 1001 / 1004): near-trench young ocean (≤ 300 km of a
  subduction or collision boundary) sits ~−330 … −370 m below its own column in both arms on 1001 (flexure from the
  thick crust beside it; Te 30 km, flexural length ~75 km), and B-b puts more of the sample there (0.75 vs 0.41 of it);
  on 1004 the near-trench share is lower with B-b and the deeper reading comes from the small sample. It is a change
  in where young ocean forms, not a fault in the elevation.
- The user's standing rules: decide on seeds, not one; do not judge on a transient. B-b is not a fitted value: on
  Earth a plate is one connected rigid piece, and the shared rule leaves 4–8% of the surface in detached pieces with
  one pole.

**Decision: B-b stays adopted** (PARAMS unchanged, `piece_rule` 'plates'), an explicit exception to the .cmd's gate
clause for the reasons above, recorded here. **The two elevation offsets now pool the window** (`scripts/elevation_check`:
one area-weighted median over the nodes of all the window's snapshots; the old mean of medians printed beside it as
"snapshot mean"). Re-read on every sweep since 25 Oct (`<sweep>/reviews/elevation_check_pooled.txt`): within 6 m of
the old reading wherever the sample is large; two pass/fail readings change, baseline 1004 (−313 → −273) and
coast:forces 1001 (−309 → −267). Neither moves a decision: node typing still fails 2 of 3 (−332, −302) against ring's
3 of 3, and coast:forces still fails T5 (ridge 71k vs 91k) and, on 1004, the gate (−341). Watch-point: B-b's ~50 m
deeper young-ocean offset on average; the ±300 m band now has the six-seed baseline at −104 … −290 m.

**Six seeds from now on** (user, 27 Oct): sweeps decide on 1001–1006; three seeds understated the plate-geometry spread.

**B-d, island arcs at Earth's rate (thirty-fifth pass's item; B-b first, now adopted).** Earth values:
- Arc addition: Dimalanta et al. 2002 (EPSL 202): western Pacific arc crust volume / age since subduction began,
  30–95 km³/Myr per km of trench (twice Reymer & Schubert's 20–40), arc crust 20–30 km. Tetreault & Buiter 2014
  (Solid Earth 5, compilation): island arcs 26 ± 6 km thick, bulk density 2.79 g/cm³.
- Earth's intra-oceanic trenches (Bird 2003 + CRUST1.0, `earth-trenches.json`): 20,770 km, closing at a length-weighted
  mean of 61.5 mm/yr (p10 / p50 / p90 16.5 / 60.2 / 103.6), 1.28 km²/yr.
- ⇒ `arc_H_ocean_m` (arc crust per m of convergence, the model's form, kept): 30–95 / 61.5 = 490–1,540 m, midpoint
  **1,000 m** (as built 300). Check: 1,000 m × 1.28 km²/yr = 1.3 km³/yr, Scholl & von Huene 2009: 1.5. No planet
  output used.
- Docking: Cloos 1993 (via Tetreault & Buiter): island arcs thicker than ~15 km are buoyant enough to collide;
  Tetreault & Buiter: "only the upper 2–5 km of arc crust are accreted onto continents through thin-skinned
  thrusting", whole-crust underplating in some cases.

**Tests, fixed before the runs** (`crust_budget`, "Arc tests (B-d)"): A1 arc growth net of forearc erosion 30–95
km³/Myr per km of intra-oceanic trench (last 40 Myr); A2 arc-stock thickness p90 ≥ 15 km and arc crust reaching
continents (turned continental + docked) > 0 on every seed; A3 (design) arc crust lost by difference ≤ half of growth.
Baseline (six-seed check, 1004–1006): 14.5 / 11.9 km / 0.69 lost: all three fail.

**Pre-check 1 (cloud, seed 1001, rate only, 1,000 vs 300):** A1 passes (78.1 vs 15.3); arcs turn continental 0.51
km³/yr (0.01); volume −9.0% vs −10.8%, label-free area 0.328 vs 0.306, margins pass (control fails, inner T 34.2). But
A2 (p90 13.2 km) and A3 fail: **93% of the last 40 Myr's arc growth (2.7 of 2.9 km³/yr) is subducted**, 56% of it at
trenches facing continents (arc–continent collision on Earth). Arc crust arriving at trenches, flux-weighted p10 / p50
/ p90 7.8 / 9.5 / 16.0 km (`runs/arc_arrivals.py`): only 13% of it ≥ 15 km, so a buoyancy threshold alone docks
little. Gates failed on that seed: young-ocean offset −352 m (young ocean beside new arc continents), and subaerial
erosion of oceanic crust 0.12 km³/yr (> 0.1: the arcs stand above sea and erode, as islands do). T1 failed on a single
sample: plate 0 cut in two in the last step (0.049 of the surface), resolved by B-b on the next step.

**Built (default off) from it:** `arc_accretion` (consumed oceanic crust thicker than 7 km hands its excess to the
overriding plate's node as volume, as collision volume does: all of it where the column is ≥ `arc_buoyant_thick_m`
15 km, else its upper `arc_accrete_upper_m` 3.5 km; the 7 km basement subducts; on continental crust it marks orogeny,
an arc–continent collision); totals `arc_accreted_m3`, `arc_accreted_on_continent_m3`. Review corrections, fixed before
the sweep: the leak gate reads normal oceanic crust (< 8 km) eroded above sea (new total
`ocean_crust_eroded_subaerial_normal_m3`; older runs read the total); T1 reads detachment lasting two samples
(the min of consecutive samples; one-sample max printed beside); `crust_budget` books docked arcs. **Ledger bug found
and fixed:** an arc turning continental was booked per world node, and two world nodes can read one plate node, so
its crust was booked twice (1.1e-4 of the stock in one step at n = 48; rare while arcs never matured). Booked once
now; fields unchanged. Default path identical step by step to the pre-edit code (70 steps, n = 48, seed 5); 81
planet-layer tests pass (2 new: `test_arc_accretion_is_off_by_default`, `test_docked_arcs_keep_the_ledgers`).

**Pre-check 2 (seed 1001, rate + docking, crust placed on the receiving node):** the budget works: arc crust lost
falls from 0.61 to 0.01 of growth (whole run), continental volume −4.3% (control −10.8%), water below the datum 2,768 m
(3,040; Earth 2,629), deep floor 4,131 m (B test passes), arcs reaching continents 2.4 km³/yr (turned 1.4 + docked 1.0).
But the plate layer runs away: 40 plates at 0 Myr, window trench 100k km (intra-oceanic 48.5k), ridge 121k,
production 7.9 km²/yr, ocean mean age 42 Myr; T2–T5 all fail. Map: the ocean speckled with one-node continents
(isolated continental nodes 661 vs 153 control / 207 rate-only; oceanic one-node spikes > 12 km among thin neighbours
471 vs 16 / 60): docked crust put on one node pushes it past 20 km. **Fixed:** docked crust on an oceanic overriding
plate joins its arc band (`spread_to_band`: Gaussian at `arc_d0_km` ± `arc_sigma_km` from the nearest overriding
trench node, volume conserved, next step; booked as `arc_docked_spread_m3`, kept out of arc growth); on continental
crust it stays collision volume. New test `test_docked_volume_spreads_over_the_arc_band_and_is_conserved`; 82 tests
pass; default path identical step by step (70 steps).

**Pre-check 3 (same, spread):** spikes 319, isolated continental nodes 424 (fewer, not gone); budget still good (A2
p90 15.1 km and A3 0.06 pass; volume −7.3%; deep floor 4,247 m passes; water 2,926 m); A1 now fails high (114.7
km³/Myr/km: plates faster, 4.3 vs 3.4 cm/yr); gates fail (young-ocean offset −313 m; the leak gate passes, normal
crust 0.00); plate layer still runs away: 34 plates, window trench 85k (intra-oceanic 39.0k), ridge 111k, production
6.15; T2–T5 fail. The plate count grows by detach events (13–18 vs 2 control; welds 600+ vs 207): plates are cut, by
continental-flagged slivers only ~13 km thick (arcs turned continental, then thinned) as well as by plain oceanic
plates. Node-typed trench length parts from the other runs at −160 … −140 Myr (119k vs 82–91k) while plate count and
speed are still alike: the extra trench comes first, the plates follow. Where it sits is mixed (near thin continents
and far from them). One seed cannot separate drift (the three runs differ from −200 Myr) from effect, but both docking
runs end with 34–40 plates vs 20–22.

**Not sent:** `Arc Rate Sweep.cmd` (control / rate / rate + docking, six seeds, rule fixed; draft kept in
`outputs/claude-sync/s20261027a/`) would fail its geometry tests on the docking arm and its A2 on the rate arm.
**Next (B-d):** find why docking lengthens the trench and cuts plates, paired and cheaper than full runs (several seeds
at n = 96 or shorter windows; per-step trench-length ledger by cause: new subduction boundary, boundary roughening,
polarity flips). Candidates to look at: arc continents too thin to stand as continents (13 km continental crust:
erosion and collapse thinning arcs that turned continental at 20 km; Earth's arcs 26 ± 6 km), and docked crust
landing on nodes inside the hidden overlap region rather than the overriding plate's interior.


### Step 8, follow-up 1, fortieth pass (28 Oct): Arc Dock Probe read: the plate runaway was one seed; the trench fluxes follow node-typed trench, not consumed crust (arcs 1.8–2.7× too much); ocean–ocean polarity disagrees between typing and overlap at 22% of arc nodes; B-d2 built (`trench_flux` 'consumed', `trench_polarity` 'pair'), default off; two pre-checks; `Trench Flux Sweep.cmd`

**Arc Dock Probe read** (`sweep-20261027-210209`, reviews in `<sweep>/reviews/`). 82 tests pass on Windows; the control
equals `piece_rule-plates` of `sweep-20261025-220434` (1001–1003) and `sweep-20261026-204140` (1004–1006) on every field
of every snapshot (stats.json differs only in keys added since). Six-seed means, control (H 300) / rate (H 1000) /
rate + docking:

| | control | rate | rate + docking | Earth / band |
|---|---|---|---|---|
| plates at 0 Myr (≥ 1 Mkm²) | 27.8 (22.5) | 28.8 (22.3) | 32.2 (23.5) | 52 (20) |
| contour subduction, last 40 Myr: total / continental side / intra-oceanic (k km) | 73.3 / 33.5 / 39.8 | 76.9 / 37.9 / 39.0 | 78.9 / 41.7 / 37.2 | 51.3 / 30.7 / 20.8 |
| window T2 / T3 / T4 / T5 | 38.5 / 29.4 / 4.59 / 81.9 | 36.9 / 36.2 / 5.04 / 89.6 | 35.6 / 40.1 / 5.28 / 90.3 | 12–25 / 20–35 / 2.5–4.5 / 80.6–109 |
| small continental bodies (< 2 Mkm²) at 0; T p50; share < 20 km | 137; 12.7; 0.73 | 251; 14.7; 0.67 | 321; 21.7; 0.40 | |
| trench node-km at 0 near small bodies / thin continent / ocean arcs / far (k) | 19 / 86 / 9 / 137 | 49 / 98 / 40 / 91 | 60 / 114 / 46 / 74 | |
| arc growth last 40 Myr (km³/yr); A1 per io km | 0.65; 16.3 | 4.37; 111.5 | 4.44; 120.4 | 1.5; 30–95 |
| A2 stock p90; A3 lost share | 11.6; 0.70 | 13.7; 0.60 | 13.9; 0.03 | ≥ 15; ≤ 0.5 |
| water below datum / deep floor (m) | 3,054 / 4,454 | 2,996 / 4,427 | 2,925 / 4,367 | 2,629 / 4,205 |
| continental volume; label-free area | −9.9%; 0.321 | −7.6%; 0.329 | −3.8%; 0.335 | ≥ −3%; 0.384 |
| > 45 km / > 55 km; median | 0.262 / 0.075; 39.3 | 0.263 / 0.068; 39.6 | 0.321 / 0.101; 39.8 | 0.086 / 0.018; 37.0 |
| elevation gates (seeds passing) | 6 of 6 | 2 of 6 | 3 of 6 | |
| margin review | passes | passes | fails (inner T 34.5) | |

Readings. (1) **The runaway seen on the cloud's seed 1001 (34–40 plates) is not what six seeds show:** docking adds 4.4
plates, nearly all under 1 Mkm² (Earth has 32 such), and 5.6k km of contour subduction (+8%), all of it on the
continental side (+8.2k; intra-oceanic −2.6k): trenches now face arc continents and small continental bodies (their
count 137 → 321), as arc–continent collision zones do. Detach events in the last 50 Myr 18 → 28. The control's own
trench grows through the run (node-km 104k at −240 to 268k at 0, 12 → 28 plates): T2 fails there too. (2) **Arc growth
per km is twice Dimalanta's band in both arc arms** (111–120 vs 30–95) and four times Earth's intra-oceanic addition in
absolute terms (4.4 vs 1.5 km³/yr), so docking puts 2.2 km³/yr onto continents (Earth's collisional addition 0.2), and
> 45 km rises by 0.06. (3) The young-ocean gate fails beside the new arcs (−289 … −434 m pooled).

**Why the arc flux is 2× (new `tools/arc_growth_probe.py`, on the snapshots):** the band law adds closing rate × H ×
a Gaussian in trench distance on the overriding plate, i.e. H × v per m of trench, where "trench" is the set of
node-typed overriding nodes. On a straight great circle at n = 217 that integrates to 0.97–0.99× the true length (5
random circles), so the law is right in itself. On the planet's boundaries it is not: at 0 Myr on six rate-arm seeds the
oceanic band adds **1.84–2.69× H × the contour's intra-oceanic consumption**, and on three seeds (1002, 1005, 1006) more
arc crust than the whole planet's subduction consumed (add / H 5.9 / 4.5 / 5.0 vs all contour subduction 4.7 / 4.2 / 3.9
km²/yr). Split by the nearest contour segment (seeds 1001, 1003): 57–61% sits behind intra-oceanic subduction (itself
~1.5× its consumption: convex overriding sides, node vs contour rates), 13–15% behind transforms, 10–13% ridges, 11–12%
junctions / untyped, 3–6% continental subduction. The other band terms carry the same geometry in the control (H 300):
ocean arcs 1.74–1.89×, continental arcs 1.22–1.54×, subduction erosion 1.50–1.65× the contour's consumption. At n = 96
the band law gives 0.68× (bands one or two cells wide): the error depends on resolution, which a law should not.

**Why a fifth of the arcs sit on the wrong plate:** the node typing (`classify_boundaries_ring`) makes the older ocean
the trench side node by node, while the overlap resolution consumes crust by one polarity per plate pair, set once and
remembered (`_ocean_polarity`). In a run (n = 217, seed 1001, −240 … −178 Myr), **22% of oceanic overriding-typed nodes
facing ocean (5,308 of 23,697) are on the plate the overlap resolution subducts**: arcs grow there and are consumed
(part of A3's "lost"), and `_driving` gives slab pull to that plate's nodes instead of the downgoing plate's. A defect
since step 7, in the default path.

**B-d2 built, both default off:**
- `trench_flux` 'consumed' ('band' as built): the oceanic area each trench consumed (`gone & ~cont` in pass 1 of
  `_update_rasters`, at the world node it disappears under) goes next step to the nearest overriding trench node of that
  plate that owns a band (Gaussian integral at 150 ± 50 km over its band ≥ half a cell; on a staircase the outer row
  owns almost none, and routing to it lost half the volume in the first test); each trench term (continental arc,
  island arc, subduction erosion) adds consumed area × its H over that node's band with its own Gaussian, normalized
  over the whole band (`trench_flux_consumed`). A straight trench gets exactly the band law with v = consumed area /
  length; Earth's H values (per m of convergence per m of trench) are per m² consumed. Consumption with no overriding
  trench of the receiving plate within 1,000 km (overlap churn at transforms and ridges) feeds nothing; booked as
  `trench_flux_routed_m2` / `trench_flux_unrouted_m2`, printed by `crust_budget` (design check: routed ≥ 0.9). At
  n = 217 the consumed area equals the contour's subduction consumption (2.14 vs 2.34 km²/yr, seed 1001, −235 … −230);
  at n = 96 it is ~3× (churn), so n = 96 cannot test this.
- `trench_polarity` 'pair' ('node' as built): ocean–ocean subduction nodes facing an oceanic node of another plate take
  the polarity the overlap resolution used for that pair this step (`polarity_now`, set in `_ocean_polarity`); pairs that
  did not overlap keep the node rule. Disagreement 22% → 0.03%; unrouted consumption over the same 77 steps 8.0% → 5.8%
  (60% of what is left is churn at transforms and ridges; 24% beside the receiving plate's own trench-typed nodes).
- Tests (4 new): `test_trench_flux_band_is_the_default`, `test_consumed_trench_flux_on_a_straight_trench_is_the_band_law`
  (volume = H × routed area exactly; within 5% of the band law; profile median within 25%),
  `test_consumed_trench_flux_keeps_the_ledgers` (with pair and docking: both ledgers to 1e-9, growth ≤ H × routed),
  `test_pair_polarity_follows_the_overlap_resolution`. 86 planet-layer tests pass in the device shell (`test_coupling`
  from a copy). Default path identical step by step to the pre-edit code (70 steps, n = 48, seed 5, threads 1, with
  `arc_accretion` and H 1000 so the edited code runs), after each of the two edits.
- Reviews: `crust_budget` prints the routed share; `plate_pieces_review`'s "3-seed mean" label now reads "seed mean"
  (it always used all seeds given).

**Pre-checks** (device shell, n = 217, seeds 1001 and 1004, chunked with the new `tools/device_runs/chunk_run.py`:
the run is pickled between 180 s calls, resume bit-identical; ~11–12 min per run, two in parallel; trajectories differ
from Windows; reviews in `outputs/planet-calibration/precheck-20261028-trenchflux/reviews/`):

| | consumed, node polarity, docking | consumed, pair, docking | Windows band + docking (six seeds) |
|---|---|---|---|
| routed share | 0.875 (fails ≥ 0.9) | 0.921 | – |
| A1 / A2 p90 / A3 | 31.2 / 16.5 / −0.12: pass | 35.0 / 15.9 / −0.16: pass | 120.4 / 13.9 / 0.03 |
| arc growth / docked on continents, last 40 Myr (km³/yr) | 1.05 / 0.59 | 1.12 / 0.79 | 4.44 / 2.16 |
| water below datum / deep floor / volume / area | 2,967 / 4,445 / −6.5% / 0.340 | 3,001 / 4,452 / −6.0% / 0.335 | 2,925 / 4,367 / −3.8% / 0.335 |
| > 45 / > 55 km; median | 0.285 / 0.076; 39.5 | 0.286 / 0.114; 38.7 | 0.321 / 0.101; 39.8 |
| gates | 1 of 2 (1004: −329 m) | 2 of 2 (−80 / −62 m) | 3 of 6 |
| window T2 / T3 / T4 / T5 | 33.3 / 32.5 / 4.31 / 76.5 | 33.6 / 38.3 / 5.23 / 79.1 | 35.6 / 40.1 / 5.28 / 90.3 |
| plates at 0 (1001 / 1004); speed | 31 / 20; 3.5 cm/yr | 28 / 15; 4.9 cm/yr | 35 / 24; 3.6 |
| margin review | – | passes | fails |

The corrected fluxes bring arc growth to Earth's order (1.1 vs 1.5 km³/yr, on 1.5× Earth's intra-oceanic trench) and
pass all three arc tests; docking onto continents falls from 2.2 to 0.6–0.8 km³/yr. Pair polarity puts arcs on the
overriding plate: the young-ocean offsets beside them come back inside the band on both seeds. Watch-points (two seeds,
another machine): plates faster with pair (4.9 cm/yr; slab pull now on the downgoing plate, on a planet with 1.5× Earth's
intra-oceanic trench), continental-side trench and production above their bands, sediment subducted per km 64 vs 37,
> 55 km 0.114 on seed 1001. The volume gain is smaller than band + docking's (−6.0% vs −3.8%): that gain came from the
over-counted arcs.

**Decision: send `Trench Flux Sweep.cmd`** (in the repo root; rule fixed in the file before the runs): A control
(today's model, must equal the probe's control), B `trench_polarity` pair alone, C pair + consumed + H 1,000 + docking;
six seeds, 18 runs. Guards for B and C against A: T1; T2–T5 window means in band or no farther than A; gates ≥ 5 of 6
and no fewer than A; median 34–40; volume no worse than A by > 1 point; margins pass if A's do. C also: routed ≥ 0.9,
A1–A3, each B test no farther from Earth than A, > 45 km ≤ A + 0.02. C adopted if it passes, else B if it passes.
Expected risks by the pre-check: C's T3 / T4 (continental-side trench and production rise with arc continents and
faster plates).
