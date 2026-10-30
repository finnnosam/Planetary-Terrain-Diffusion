# Planet layer: current state

**As of 29 Oct 2026, after the forty-first pass.** Rewrite this file at the end of every pass; don't append to it.
History lives in `journal/` (one file per pass) and `decisions.md` (one line per decision).

## 1. Right now

**The repository migration is under way** (Claude Doc "Planet generator: repository migration plan"). Phase 0
commits everything and tags `pre-split-2026-10-29` in this repository. Phase A then creates the new repository
`E:\projects\Planetary_Diffusion`, proven by an identity run. No model change happens during the migration. Model work
resumes in the new repository afterwards.

**The Trench Flux sweep has been read** (`sweep-20261028-200644`, [pass 41](journal/59_2026-10-29_pass-41.md)).
Neither candidate was adopted, so PARAMS are unchanged:
- B (pair polarity) fails T4, T5 and the margin guard against the control.
- C (pair + consumed fluxes + arc H 1,000 + docking) fails T2, T3, T4, the > 45 km guard and the margins. It does
  pass the arc tests and moves the volume, water and floor toward Earth.

## 2. The chain and how to run it

```
P  plate layer      planet_layer/plates.py     n = 217 (46 km), 1 Myr steps, 250 Myr from a supercontinent
L  landscape        coupling.py + graph_landscape.py   n = 868 (11.53 km), 50 kyr, last 30 Myr, θ 5
H  hand-off         scripts/planet_meso_check.py       exact area mean onto the meso's 23.04 km cells
M  meso             toy_meso/planet_tiled.py           outputs/meso/detail-ft/step10000-passed.pt (detail unknown)
```

Commands, scripts and data folders are listed in `docs/project-map.md`. Every value is labelled measured / assumed /
fitted in `docs/planet-parameter-audit.md`.

## 3. The model as it stands

> **Warning: the working baseline is not all in PARAMS.** Four values are still the old fitted ones in
> `planet_layer/plates.py` PARAMS. Every sweep `.cmd` since 23 Oct overrides them explicitly. A bare
> `simulate_plates` run without these `--param`s is **not** the current model:
> `erosion_K_factor=1.0 erosion_theta=200 erosion_Sc=0.036 margin_shortening=True sediment_accretion=0.13`
> (PARAMS has 0.75 / 6 / 0.05 / False / 0.75). The .cmd files also pass `continental_sediment`,
> `crust_kappa_normal_m2_yr`, `sediment_compaction` and `piece_rule`, which already match PARAMS.
> Moving the five values into PARAMS is planned for the code cleanup after the Trench Flux sweep.

| Area | Setting (current) | Since |
|---|---|---|
| 46 km erosion | K × 1.0, θ 200, Sc 0.036, Kd × 4 (Earth-calibrated, OCTOPUS ¹⁰Be): **working baseline** | 23 Oct |
| margins | `margin_shortening` True (Oncken rollback 17 mm/yr, arc 750 m), `sediment_accretion` 0.13: **working baseline** | 23 Oct |
| rifting | `rift_rule` pure_shear, `breakup_thick_m` 10 km, `rift_width_km` 100 | 24 Oct |
| isostasy | `thinning_local_isostasy` True; `ocean_crust_erosion` True | 24 Oct |
| crust flow | `crust_kappa_normal_m2_yr` 30; collapse diffusion above 45 km unchanged (Tibet-tuned, audit F/A) | 24 Oct |
| sediment | `continental_sediment` True (2,400 kg/m³ layer); `sediment_compaction` True (Athy); `keep_deep_sediment` True (fill and spill) | 24 / 26 / 21 Oct |
| start | supercontinent, `start_crust` earth (CRUST1.0 stable crust), `start_plates` voronoi, `start_poles` random | 25 Oct |
| orogeny clock | `orogen_rule` belt | 25 Oct |
| boundaries | `boundary_normal` ring | 26 Oct |
| plate pieces | `piece_rule` plates (detached pieces ≥ 1 Mkm² become plates, smaller ones weld) | 26 Oct |
| arcs | `arc_H_ocean_m` 300, `trench_flux` band, `trench_polarity` node, `arc_accretion` False (B-d2 not adopted, pass 41; the polarity defect remains in the default path) | 29 Oct |
| shelves | κ 3·10⁴ m²/yr, cliff retreat 0.003 m/yr | 22 Oct |
| dynamic topography | random field, `dyn_std_m` 500, degree 30, slope 1 (fitted; plate-blind; see D below) | 18 Oct |
| landscape stage | θ 5 at 11.53 km (`run_planet_landscape.THETA`); its law's form does not match Earth's rate–slope relation | 23 Oct |
| built but off | extension E–E4 (`extension`, `ext_*`), `collapse_edge`, coast start / force poles, `pelagic_m_myr` | rejected |

## 4. Scorecard against Earth

Six-seed means of today's model. Source: the Arc Dock Probe's control (`sweep-20261027-210209`, H 300, no
docking), reproduced exactly by the Trench Flux sweep's arm A (`sweep-20261028-200644`). Earth values are on the same grid. A ✓ means the value
is inside its band.

| Measure | Planet | Earth / band | |
|---|---|---|---|
| elevation gates (young-ocean / thin-crust offset, pooled) | 6 of 6 seeds; −187 / −183 m (−104 … −290) | ±300 m | ✓ (young ocean is a watch-point) |
| land fraction | 29.6% | 28.1% | ✓ |
| land p25 / p50 / p90 | 858 / 1,444 / 3,317 m | 166 / 384 / 1,461 m | ✗ **land stands ~1 km too high** |
| continental crust drowned | 12.8% | ~29% (20–35%) | ✗ |
| export | 5.1 km³/yr | 4–8 | ✓ |
| continental volume over 250 Myr | −9.9% | ≥ −3% | ✗ |
| label-free continental area | 0.321 | 0.384 (0.35–0.42) | ✗ |
| crust < 30 km / > 45 km / > 55 km | 0.089 / 0.262 / 0.075 | 0.190 / 0.086 / 0.018 | ✗ thin crust short, thick tail 3× |
| crust median | 39.3 km | 37.0 (34–40) | ✓ |
| margins (thin per km of edge / abrupt / edge T / inner T, 1°) | 110 km / 14% / 25.8 / 33.4 km | 155 / 11% / 25.2 / 31.6 | ✓ passes |
| water below the datum / deep floor | 3,054 / 4,454 m | 2,629 / 4,205 m | ✗ (sea level ~600 m below the datum) |
| plates at 0 Myr (≥ 1 Mkm²) | 27.8 (22.5) | 52 (20) | microplates missing |
| contour subduction, last 40 Myr: total / continental side / intra-oceanic | 73.3 / 33.5 / 39.8k km | 51.3 / 30.7 / 20.8k | ✗ intra-oceanic ~2× |
| window T2 / T3 / T4 / T5 | 38.5k / 29.4k / 4.59 / 81.9k | 12–25k / 20–35k / 2.5–4.5 / 80.6–109k | T2 ✗, T4 at the edge |
| sediment subducted, last 40 Myr | 2.8 km³/yr | ~1.0–1.65 | ✗ |
| island-arc growth (A1, per km of intra-oceanic trench) | 16.3 km³/Myr | 30–95 | ✗ (B-d2 addresses it) |

Last measured elsewhere (older sweeps, same model except as noted):
- Land above 2 km is 0.34–0.45 of land (Earth 0.05). The slope of land above 2 km is 2.0–2.5 m/km (Earth 11.1): the
  high land is plateau. Source: thirty-fourth and twenty-sixth passes, `sweep-20261025-170555`, `sweep-20261024-193840`.
- Passive edge is 60–69k km (Earth 173k). Edge in small enclosed basins is 0–7k km (Earth 34k). Source: twenty-second pass.
- Lowland-plain windows are 5.7% of land (band ≥ 8%). The meso over-textures flat windows (4.9× Earth's power, with
  the detail value unknown). Source: ninth and tenth passes, measured on the old 22 Oct baseline.
- Boundary lengths (contour, ring control): ridge 97.6k, subduction 83.9k, collision 5.4k, transform 99.4k km
  (Bird 94.8 / 51.3 / 40.5 / 73.9). Source: thirty-fourth pass.

## 5. Open problems, in the order they will be worked

1. **Finish the repository migration** (section 1).
2. **The rest of B (continental budget and ocean basin):**
   - The intra-oceanic trench excess. T2 fails in every arm: 36–41k vs 12–25k km. The control's trench grows through
     the run (node-km 104k at −240 to 268k at 0; plates 12 → 28).
   - Take the polarity fix (`trench_polarity` pair) along with whatever moves T2.
   - Then retry C (consumed fluxes, Earth arc rate, docking) once docking no longer lengthens the continental-side
     trench and thickens margins.
   - The sediment-subduction excess (2.1–2.8 km³/yr vs ~1).
3. **D, dynamic topography from the slabs.** It comes before C, because C's tests read through the current field.
   The current field is plate-blind, flat in spectrum to degree 30 (12% of its variance in degrees 1–3, Earth ~85%),
   and identical in every run of a seed. Design and tests: [investigation entry](journal/55_2026-10-26_investigation-dynamic-topography.md).
4. **C, lithosphere buoyancy.** It is narrowed: thin and rifted crust stand higher, old thick crust stands lower.
   It does not explain the plateaus ([pass 34](journal/50_2026-10-26_pass-34.md)).
5. **E, the thick crust itself.** It covers the plateaus, the > 45 km tail and the area lost at collisions. E–E4
   failed because thinning does not finish ([pass 34](journal/50_2026-10-26_pass-34.md)).
6. **The rest of 2b:** back-arc basins, microcontinents, passive edge in small basins.
7. **The detail value for the meso**, which is independent of the rest: `planet_meso_check` with fixed detail values
   on flat windows first.
8. **Freeze and look:** GeoTIFF, `toy_meso.planet_tiled --bounds`, drainage vs RiverATLAS.

Small items:
- A pelagic rule from GlobSed by floor age (~3 m/Myr bulk median).
- The 11.53 km erosion law's form (n > 1 or a sub-cell channel–hillslope split; fifteenth pass).
- Endorheic basins (needs a water balance).

## 6. Watch-points and known defects

- The young-ocean offset is the gate closest to its edge (six-seed control −104 … −290 m). New arcs and B-b deepen it.
- Three seeds understate the plate-geometry spread (held-out check, 26 Oct). Six seeds from 27 Oct.
- Runs on the user's machine are deterministic. The cloud workspace and the device shell have their own
  trajectories, so use them for pre-checks only.
- The trench-flux band law depends on resolution (0.7× at n = 96). Use n = 96 for plumbing only.
- Boundary crumbs (pieces with no interior node) are left by B-b.
- `graph_landscape.erode_and_deposit` is off by ~2e-5 of the gross now and then. This is negligible.
- Plate-raster nearest-node sampling loses ~17% of continental sediment increments and adds ~0.4–0.6 km per 40 Myr
  to quiet margins (twentieth pass).
- The polarity defect: arcs on the downgoing plate at 22% of oceanic arc nodes (default path; fix built, not adopted).
- A stale `_state.pkl` sits in `outputs/planet-calibration/precheck-20261028-trenchflux/`. It is harmless.

## 7. Where things are

| What | Where |
|---|---|
| this file, the decision log, the journal, the original design | `docs/planet-layer/` |
| handoff for a new chat (working rules) | `docs/next-chat-prompt.md` |
| code, scripts, data, outputs, commands | `docs/project-map.md` |
| every value: measured / assumed / fitted, replacement order | `docs/planet-parameter-audit.md` |
| long-term target for all stages | `docs/planet-ideal-flow.md` |
| the old single plan doc (frozen 28 Oct; code comments and `.cmd` headers cite its passes) | `docs/planet-layer-plan.md` = `design-2026-10-17.md` + `journal/` |
| old handoff (verbatim) | `docs/planet-layer/archive/next-chat-prompt-2026-10-28.md` |

When code or a `.cmd` cites "plan doc, Nth pass", open `journal/` and find the file named `pass-NN`.
