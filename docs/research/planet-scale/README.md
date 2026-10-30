# Planet-scale layer: research survey (16 Oct 2026)

**Question.** The chain works from 23 km down: the meso (23 → 2.88 km, detail-aware fine-tune) renders credible
terrain from credible 23 km input, and a regional landscape-evolution model (`planet_training/landscape_evolution.py`)
turns tectonic uplift into credible 23 km structure. Nothing yet decides *what sits where on a whole planet*:
continents, plate boundaries, which margins get ranges, where old worn ranges, cratons, rifts and basins are. What
existing work can supply or inform that layer, and how should it hand off to the landscape model on the cube sphere?

This is research only; nothing here has been built.

## Summary

1. **Phenomenological plate simulation is a solved-enough problem for graphics.** Cortial et al. 2019 (*Procedural
   Tectonic Planets*) is the reference design: ~35–100 km crust samples on the sphere, rigid plate rotations,
   rule-based subduction uplift, collisions, ridge crust generation, rifting, and per-sample crust attributes
   (type, thickness, age, orogeny age and type, fold direction). ~250 Myr of history in a few hundred 2 Myr steps,
   ~2 s/step at 35 km on a 2019 desktop. It is not physical, and it does **not** model drainage, passive margins,
   shelves or hotspots: exactly what our landscape model adds. A hobby re-implementation (second-system.de, 2022)
   confirms it is doable but fiddly (mesh repair, self-intersection, collision handling).
2. **Landscape evolution on a sphere exists and matches our process set.** goSPL (Salles et al.; used for the
   2023 *Science* 100 Myr global reconstruction) runs stream-power erosion, depression deposition, marine
   deposition, hillslope processes and **horizontal advection by plate motion** on an unstructured spherical mesh at
   5–100 km. It is GPL-3, PETSc/MPI, Docker-oriented: a reference design more than a drop-in dependency on a
   Windows/RTX 3070 setup. Fastscapelib has a `HealpixGrid` and a planetary example (stream power + MST sink
   resolution at ~65 km), but HEALPix support ships only for Linux/macOS (conda-forge).
3. **Graphics work on uplift + erosion at regional scale** (Cordonnier et al. 2016; analytical stream-power erosion,
   Tzathas et al. 2024) supports our regional findings (uplift gradients matter more than shape; implicit
   tree-ordered solvers; slope limits) but none couples to plates on a sphere.
4. **The closest whole-planet ML work** is Borg et al. 2026 (*Authoring Terrestrial Planets with Diffusion Models*,
   Eurographics 2026): a diffusion model on a quadrilateralised spherical cube at 2.4 km, conditioned on 78 km
   sketches plus a derived river network. Its stated limitations are the gap this project targets: no tectonic or
   erosion history (inconsistent large-scale structure), blocky coastlines from a 32× jump, endorheic basins not
   handled. They name coupling with physics-based plate tectonics and erosion as future work.
5. **Plate-reconstruction tools (GPlates/pyGPlates/gplately)** are built for Earth's real reconstructions. Writing
   synthetic topological plate networks for invented planets is possible but laborious; useful for concepts and
   for validating against Earth models (e.g. Merdith et al. 2021, 1 Ga), not as the engine.
6. **Mantle-convection models with emergent plates** (StagYY, ASPECT) are far too expensive and not steerable.
7. **Earth reference statistics exist for validating a generated layout:** plate count/size distribution
   (Bird 2003: 52 plates, 8,117 km² to 104 × 10⁶ km²; broken-sheet fit), Hasterok et al. 2022 global geological
   provinces and plates (polygons with tectonic setting and age; Zenodo), hypsometry, and our own 368 km land-relief
   and 40 km arrangement statistics.

## What this means for the design (recommendation to evaluate, not decided)

**Two clocks, two resolutions.** Plates need ~100–250 Myr of history; erosion at 11.5 km needs 50 kyr steps.
A global 11.52 km cube sphere is ~3.8 M cells (59× the 256² test window, which takes ~30 s per 30 Myr with numba),
so 250 Myr at 11.5 km is out of reach on a desktop, but the young phase is not:

1. **Plate layer (coarse, long):** Cortial-style kinematics on ~50–100 km spherical samples for ~100–250 Myr.
   Outputs per time step: plate id, velocity, crust type/thickness/age, boundary type, rock-uplift and subsidence
   rates, orogeny age and type, and simple erosion so old orogens decay (as Cortial does). This replaces the
   hand-made tectonic templates and the "inherited history" phase of the prototype.
2. **Landscape layer (fine, short):** the existing numba landscape model ported to the cube sphere (cube charts
   already exist in `planet_diffusion/cube.py`), run for the last ~20–40 Myr at 11.52 km (or 23 km first), driven
   by the plate layer's uplift and starting from its coarse elevation. Needs: routing across face edges and
   corners, sea connectivity on the sphere, flexure without a periodic FFT (local kernel per face with halo, or
   spherical harmonics), and a decision on horizontal advection (goSPL advects; over 20–40 Myr plates move
   200–2,000 km, so it matters at plate boundaries).
3. **Hand-off to the meso:** area-average to 23.04 km on the cube (as now), plus a **detail map** for the detail-aware
   meso derived from the plate/landscape state (rock age and erodibility, recent uplift, relief), since the meso is
   close to Earth when told the detail and leans rough when not.

**Alternatives to compare honestly before building:**
- adopt goSPL as the landscape layer (proven, has advection; cost: GPL, PETSc/Docker, HPC-oriented);
- fastscapelib HEALPix (WSL/Linux only; would replace the cube grid for the simulation stage);
- a learned layout model on PaleoDEMs (the earlier "L" plan in `docs/training-data-plan.md`): already judged weak
  (15–30 distinct layouts, smooth reconstructions); usable at most as a validation reference.

## Validation targets (from Earth, fixed before building)

- Land fraction ~29%; bimodal hypsometry; ocean depth distribution (ridges ~2.5 km, abyss 4–6 km).
- Plate count and size distribution against Bird 2003 / broken-sheet statistics.
- Fraction of land within ~500 km of an active boundary; proportions of cratons / old orogens / active orogens /
  rifts / basins against Hasterok et al. 2022 provinces.
- 368 km land-relief distribution: Earth p10/p25/p50/p75 = 133/237/456/872 m, 34% of windows under 300 m
  (`scripts/compare_plain_inputs.py`).
- 40 km arrangement metrics (`planet_training/arrangement_metrics.py`): Alps/California/Plains references.
- 23 km cell-to-cell curvature per relief class against the meso's validation inputs (see
  `docs/tectonic-prototype.md`, version 3 meso check).
- End-to-end: `scripts/diagnose_meso_plains.py` on planet windows.

## Sources

| Source | What it gives us |
|---|---|
| [Cortial et al. 2019, Procedural Tectonic Planets (PDF)](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf) ([EG library](https://diglib.eg.org/handle/10.1111/cgf13614), [video](https://www.youtube.com/watch?v=GJQVl6Xld0w)) | Full plate-layer recipe: 500k spherical Fibonacci samples (~35 km), rigid rotations s = ωw×p, slab-pull axis drift, subduction uplift u = u0·f(d)·g(v)·h(z) out to 1,800 km, terrane collision at 300 km overlap (radius rc·√(v/v0·A/A0), rc 4,200 km), ridge crust from a template profile, Voronoi rifting with Poisson timing, erosion εc 0.03 mm/yr, δt 2 Myr, resampling every 10–60 steps, ~1.9 s/step at 35 km. No drainage, shelves, passive margins or hotspots. |
| [second-system.de, Plate Tectonics 1](https://second-system.de/2022/03/01/tectonics_1) and [part 3](https://second-system.de/2022/09/04/tectonics_3) | Independent re-implementation of Cortial on one spherical Delaunay mesh with spring/Verlet plates; practical pitfalls (self-intersections, edge flips, debugging tools first). |
| [goSPL documentation](https://goSPL.readthedocs.io), [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02804.pdf), [CSDMS page](https://csdms.colorado.edu/wiki/Model:Gospl) | Global landscape evolution on an unstructured spherical mesh, 5–100 km; stream power with multiple-flow-direction, priority-flood depression deposition, marine dual-lithology deposition, hillslope, compaction; vertical and **horizontal** tectonic forcing from plate velocities; PETSc/MPI; GPL-3. |
| [Salles et al. 2023 (Science) summary, EarthByte](https://www.earthbyte.org/science-hundred-million-years-of-landscape-dynamics-from-catchment-to-global-scale/); [Eos article](https://eos.org/articles/one-surface-model-to-rule-them-all) | goSPL run over 100 Myr of Earth with paleo-elevation and paleo-climate forcing: global sediment flux and physiography; shows the approach works at planet scale (on HPC). |
| [Fastscapelib grids guide](https://fastscapelib.readthedocs.io/en/latest/guide_grids.html), [planetary example](https://fastscapelib.readthedocs.io/en/latest/examples/planetary_py.html) | C++/Python LEM building blocks; `HealpixGrid` on the sphere (nside 100 ≈ 65 km), SingleFlowRouter + MSTSinkResolver + SPLEroder; HEALPix only on Linux/macOS via conda-forge. |
| [Cordonnier et al. 2016, Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion](https://www.cs.purdue.edu/homes/bbenes/papers/Cordonier16CGF.pdf) | Uplift map + stream power on a Poisson-disk graph; implicit O(N) tree solver; lake super-graph; uplift gradients control the result; fluvial only, uplift painted by hand. |
| [Tzathas et al. 2024, Physically-based Analytical Erosion for fast Terrain Generation](https://diglib7.eg.org/items/0a412929-2e29-4bcb-a597-1a4b2cc558e2/full) | Analytical stream-power solutions with time as a slider (multigrid); possible fast path for the young phase. |
| [Borg et al. 2026, Authoring Terrestrial Planets with Diffusion Models](https://diglib.eg.org/bitstreams/0b159ae6-4104-4325-be37-0ce97149537c/download) | Closest whole-planet ML system: quadsphere, 2.4 km, DDPM conditioned on 78 km sketches + river network, tiled parallel denoising without seams, ~1.5–5 h per planet on an H100. Limitations: no tectonics/erosion history, blocky coasts, endorheic basins. |
| [PyPlatec / platec](https://pypi.org/project/PyPlatec/1.2.9) | Older C++ plate simulation (Viitanen) on a flat wrapping heightmap; historical reference, not suitable (not spherical, unmaintained). |
| [Worldengine](https://pypi.org/project/worldengine/) | Game world generator built on platec; same limitations. |
| [pyGPlates docs](https://www.gplates.org/docs/pygplates), [gplately](https://pypi.org/project/gplately) | Plate reconstruction tooling for Earth; topological models, rotations; possible validation/reference, laborious for synthetic planets. |
| [Merdith et al. 2021 plate model (Zenodo)](https://zenodo.org/records/10910449) | 1 Ga full-plate Earth model (GPlates): reference for realistic plate motions, boundary lengths, supercontinent cycle. |
| [GSA Today: plate size distribution and fragmentation](https://www.geosociety.org/gsatoday/science/G358A/article.htm) | Bird 2003: 52 plates, 229 boundary segments, smallest 8,117 km², largest 104 × 10⁶ km²; broken-sheet fit R² 0.87 (Bird) / 0.93 (20-plate model). |
| [Hasterok et al. 2022 global provinces and plates (Zenodo)](https://zenodo.org/records/6586972) | Global geological province and plate polygons with tectonic setting and age: Earth proportions of cratons, orogens of each age, rifts, basins; candidate source for detail-map statistics. |
| [Dynamic topography / tectonic-rules mantle frame (EarthByte)](https://www.earthbyte.org/solid-earth-a-tectonic-rules-based-mantle-reference-frame-since-1-billion-years-ago-implications-for-supercontinent-cycles-and-plate-mantle-system-evolution/) | Background for long-wavelength mantle swells (the prototype found 22–40 m/Myr swell rates necessary for Earth-like interior relief). |
