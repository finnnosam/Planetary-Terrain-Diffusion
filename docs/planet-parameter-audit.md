# Planet layer: parameter audit (23 Oct 2026; updated after the margin sweep)

Every tunable value in the planet chain, labelled by where it comes from, so that values fitted to the planet's own
output can be replaced by Earth measurements or by mechanism (the user's direction of 23 Oct: realism; no value
that exists only to make a band pass).

Kinds:
- **M**: measured on Earth (a value or range from data; source given).
- **P**: physics, a conservation law or a mechanism with no free value.
- **A**: assumed: plausible, not fitted, but without a source. To be sourced or checked.
- **F**: fitted: chosen because the planet's output then passed a band, or picked from a range by a sweep score.
  To be replaced.
- **N**: numerical (resolution, stability, step size).

Files: `planet_layer/plates.py` (PARAMS), `planet_layer/mantle.py` (PARAMS), `planet_training/landscape_evolution.py`
(DEFAULTS, used by the graph kernels at both grids), `planet_layer/coupling.py`, `scripts/run_planet_landscape.py`.

## 1. Fitted values (F): the replacement list

| Parameter | Value | How it was set | Earth reference / replacement |
|---|---|---|---|
| `sediment_accretion` | 0.75 | matched Earth's *absolute* sediment subduction (1.65 km³/yr) while the deep-sea sink deleted half the export (18 Oct) | **M available, conflicts:** 0.13 length-weighted (von Huene & Scholl 1991). In the running margin sweep |
| `subduction_H_cont_m` | 2,500 m | sweep-20261018-144253 (ranges for the arrangement score) | **conflicts:** adds 5–8 km³/yr of new crust vs Earth's ~1.0 at continental margins. Replaced by `margin_shortening` (shortening rate from Oncken 2006 + arc addition 750 m). In the running sweep |
| `breakup_thick_m` | 17,500 m | shelf gate, sweep-20261022-194148 (bracketed 12.5–20 km, picked by the bands) | **conflicts:** hyper-extended margins break up at ~5–15 km of crust. Tested 23 Oct at Earth's middle values (10 km, `rift_width_km` 100): Earth's thin-crust share and median thickness (CRUST1.0), but 48% continental area and +23% volume (seventeenth pass) |
| `erosion_K_factor` (46 km) | 0.75 | sweep-20261018-164500 score | **tested 23 Oct against OCTOPUS ¹⁰Be rates (fifteenth pass): the current law is ×8 too fast; Earth-calibrated pooled fits: (K × 1.0, θ 200) or (K × 0.1, θ 100), Kd × 4, Sc 0.036. In `Earth Erosion Law Sweep.cmd`** |
| `erosion_theta` (46 km) | 6 | shelf gate (sweep-20261022-194148); was 8 from step 6 | **Earth data: 100–200** (fifteenth pass) |
| `erosion_Kd_factor` | 4 | calibrated against the 11.53 km model (step 6), i.e. against the model, not Earth | same steady-state test |
| `dyn_std_m` | 500 m | sweep-20261018-144253 (score), between 400 and 800 tried | **M range:** Earth's residual topography ±~1 km to degree ~30 (Hoggard et al. 2016) or smaller (debated). Take a published global estimate's amplitude and spectrum as is |
| `shelf_kappa_m2_yr` | 3·10⁴ | shelf gate (from 0, 3·10⁴, 10⁵) | stratigraphic-model values (Kaufman 1991, Rivenæs 1992, Granjeon & Joseph 1999) as published, or shelf-profile data (topset gradient ~1–2 m/km) |
| `cliff_retreat_m_yr` | 0.003 | shelf gate (from 0.003, 0.01) | measured cliff retreat rates by rock type (Earth: mm/yr to dm/yr) |
| landscape `THETA` (11.53 km) | 5 | 3-seed rule on denudation and 23 km texture of the planet's output | **tested against OCTOPUS (fifteenth pass): the law is ×12.5 too fast and its form does not match Earth's rate–slope relation at 11.5 km; needs a mechanism change (n > 1 or a sub-cell channel–hillslope split)** |
| coupling `smooth_km` | 80 km | removes beaded collision belts (one-node pulses at 46 km) | a belt shape from mechanism (collision width from convergence and crust strength); for now a numerical fix, kind N/F |

## 2. Assumed values (A): to source or check

| Parameter | Value | Note / Earth data to check against |
|---|---|---|
| plate speeds: `v_min_cm_yr` 1.5, `v_slab_cm_yr` 9, `slab_ref_km` 6000, `v_ridge_cm_yr` 1.5, `ridge_ref_km` 8000, `ridge_torque` 0.3, `cont_drag` 0.3, `collision_ref_km` 1500, `tau_speed_myr` 10, `tau_axis_myr` 20 | | Forsyth & Uyeda (1975) qualitatively. **Checked 23 Oct against Earth** (`scripts/plate_speeds.py`, Bird 2003, no-net-rotation frame): continental plates 0.5–5.3 cm/yr (Earth 1.2–2.9), oceanic 1.8–11.3 (Earth 1.5–7.5). Earth-like; the fastest oceanic plates ~1.5× Earth's. Kept |
| `arc_H_ocean_m` | 300 m | Computed (thirty-fifth pass): the planet's arcs grow 15 km³/Myr per km of intra-oceanic trench (net of forearc erosion) vs Earth's 30–95 (Dimalanta et al. 2002; crust 20–30 km) ⇒ 0.55–1.7 km per m of convergence at 55 mm/yr. Not to change before the planet's intra-oceanic trench length (54–58k km vs Earth ~17k, inherited from the start) is fixed (candidate B-b). **Thirty-ninth pass: candidate 1,000 m (B-d)**: Dimalanta's 30–95 over Earth's intra-oceanic closing rate (Bird 2003 + CRUST1.0: 20.8k km, length-weighted 61.5 mm/yr) = 490–1,540 m, midpoint 1,000; × Earth's 1.28 km²/yr = 1.3 km³/yr (Scholl & von Huene 2009: 1.5). No planet output used. Tests A1–A3 in `crust_budget`. Pre-check (seed 1001): alone, 93% of new arc crust is subducted; not swept yet. **Fortieth pass:** the six-seed probe (`sweep-20261027-210209`) gives 111–120 km³/Myr per km at 1,000 m because the band law counts node-typed trench (1.8–2.7× H × the consumed area); with `trench_flux` 'consumed' the pre-checks read 31–35 (A1 passes). To be decided by `Trench Flux Sweep.cmd` |
| `arc_to_continent_thick_m` | 20 km | arc crust becomes "continental" (island arcs 15–35 km thick on Earth) |
| `pelagic_m_myr` | 0 (off) | **conflicts with M:** Earth's deep sea 1–10 m/Myr (plan doc), GlobSed mean ~0.5 km. Off because the planet's floor already carries 1–2 km of exported sediment; revisit after the export is Earth-like. 26 Oct (thirty-sixth pass, GlobSed v3 on n = 217): Earth's label-free ocean holds 610 m (bulk; open ocean, > 200 km from continents and trenches, 390 m), the planet's open ocean 186–546 m (layer of 2,400) with nothing from pelagic rain; an Earth rate by floor age needs an age grid (not in `data/`) |
| `rift_thinning` | 0.6 | stretching factor of a rift gap; Earth β factors to source |
| `rift_width_km` | 0 | **conflicts:** Earth necking zones 50–200 km; rejected in step 8 under the old regime |
| `stable_erodibility` 0.15, `orogen_erodibility_tau_myr` 50 (both grids) | | from the test bed. Erodibility from orogeny age only; Earth's erodibility depends mostly on lithology and climate. Bird 2003 orogens give a measured "young belt" map for tests. **25 Oct (twenty-third pass): the 46 km law was calibrated with erodibility 1 in Bird's orogens (16% of Earth's land, 76% of land > 2 km), but the planet resets the orogeny clock only at collision contacts: young-orogen erodibility on 1–2% of its land and 2–4% of land > 2 km; shortening belts and collapsed crust erode at 0.15 (proposal B). Built as `orogen_rule` (default 'collision'; 'belt'; 'strain' with `orogen_strain_rate` 3.5e-9 /yr, M: GSRM v2.1 areal dilatation at which Earth's flagged land equals Bird's orogen land, `scripts/earth_deformation`). 25 Oct (twenty-fifth pass): 'belt' adopted as default by the interior sweep's rule (`sweep-20261024-193840`); it marks 0.40–0.47 of land (Bird 0.16) and thins nothing: the high land is plateau, which a thresholded law cannot lower** |
| `subduction_d0_cont_km` 150, `subduction_sigma_cont_km` 60, `arc_d0_km` 150, `arc_sigma_km` 50 | | arc–trench distance on Earth ~100–300 km: in range, to source |
| `foreland_d0_km` 350, `foreland_sigma_km` 100 (new) | | geometry of the underthrust foreland; Andean foreland widths to source |
| `crust_kappa_normal_m2_yr` 1000, `crust_kappa_m2_yr` 1000, `collapse_thick_m` 45 km, `collapse_scale_m` 6 km, `collapse_max_factor` 200 | | collapse tuned to hold plateaus near ~70 km (Tibet, Altiplano: an Earth target, kind M-target); `kappa_normal` erased 100 km structure in 10 Myr (step 8), so cold-crust flow should come from lithosphere strength. **24 Oct (twentieth pass): `kappa_normal` 1000 is the main thickener of old passive margins (44–109 m/Myr into thin margins); Earth value ~30–40 (10 km lower-crust channel at ~10²¹ Pa s; surviving 180 Myr margins need ≪ 125). Adopted as default 30 (`sweep-20261024-143735`). 25 Oct (twenty-fifth pass): the collapse terms make the plateaus: κ 1,300 / 4,300 / 11,000 m²/yr at 50 / 55 / 60 km of crust, 30 below 45 km, so thick crust spreads until just above 45 km and stalls (0.19 of the continent at 45–55 km vs Earth 0.068; slope of land above 2 km median 1.7–3.0 m/km vs 11.1). M-target (Tibet) → to replace: proposal E, collapse as extension that regains area (replacement order, step 4b)** |
| events: `split_rate_per_myr` .006, `split_min_area_mkm2` 12, `split_speed_cm_yr` 2, `suture_weakness` .6, `init_age_myr` 170, `init_rate_per_myr` .004, `init_speed_cm_yr` 3, `init_area_ref_mkm2` 2, merge (`merge_contact_km` 1000, `merge_after_myr` 40, ratios) | | `init_age_myr` 170 is close to Earth's oldest ocean floor (~180 Myr, M). Rates checkable against Earth's Wilson cycle and plate-size statistics (Bird 2003: 52 plates, 32 under 1 Mkm²; the planet has ~3 microplates) |
| `boundary_min_mm_yr` | 5 | transform threshold |
| start: 12 plates, 5 continental, continental fraction 0.38, craton ages 1.5–3.5 Ga, sutures 0.3–1.5 Ga, crust 38 km ± 1.5 km field | | supercontinent assembly (Pangaea-like). Continental fraction 0.38 vs Earth's ~0.40 (M-ish). **25 Oct: the interior is 37.5 km + 2 km beyond 300 km from sutures + 4 km near sutures + field: p50 39.2–40.1 km, p10–p90 3–4 km wide, and it survives 250 Myr untouched (95–99% of present interior crust). Earth's stable crust (CRUST1.0 Precambrian / platform types, > 500 km from plate boundaries): crystalline p10 / 50 / 90 = 32.9 / 38.2 / 43.0 km (proposal A). Built as `start_crust` 'earth' (M: `EARTH_STABLE_CRUST_KM`); adopted as default 25 Oct (twenty-fifth pass, `sweep-20261024-193840`): median 39.5–39.7 km, < 30 km share up, p10 −1 to −3 km** |
| mantle: `dt_knot_myr` 40, hotspot birth .05/Myr, life 60–180 Myr, flux 2,000–15,000 km³/Myr, swell 800 m / 500 km, LIP 0.5–4·10⁶ km³ | | Earth-like ranges (Hawaii flux; LIP volumes) but unsourced in the code; `dyn_lmax` 30 and `dyn_slope` 1 follow Hoggard et al. 2016 (M) |

## 3. Measured, physical or numerical (M / P / N): keep

| Parameter | Value | Kind and source |
|---|---|---|
| `rho_crust` 2830, `rho_mantle` 3300, `rho_water` 1030, `rho_sediment` 2400 kg/m³ | | M (standard values); `rho_sediment` fits thick compacted basins, not the thin deep-sea cover (bulk ~1,900–2,050 near the top: Sykes 1996; 40% of the floor misfit, thirty-fifth pass; candidate B-a: compaction law); built 26 Oct, thirty-sixth pass, default off) |
| `sediment_compaction` (**default True since the thirty-seventh pass**), `rho_grain` 2650, `phi0_ocean` 0.63 / `compaction_c_ocean_per_km` 0.51, `phi0_continent` 0.56 / `compaction_c_continent_per_km` 0.39 | | M: grain density of quartz–clay sediment; Athy's law with Sclater & Christie (1980) shale (oceanic crust) and shaly sandstone (continental crust); pores water-filled. With it `sed` is solid volume and the elevation reads bulk height and mass (B-a). Adopted by `sweep-20261025-201623`: gates pass on 3 seeds, guards hold; deep floor 4,317 m (Earth 4,205 ± 150; control 4,524), ocean bulk sediment 569 m (GlobSed 610). Reviews read runs that saved only overrides with `crust_review.LEGACY_DEFAULTS` (compaction off); runs save `params_effective` since then |
| `piece_rule` (thirty-seventh pass; **default 'plates' since the thirty-eighth pass**), `piece_min_area_mkm2` 1 | | P / M: on Earth a plate is one connected rigid piece; 'plates' makes each detached piece with an interior node its own plate (≥ 1 Mkm²; Bird 2003: the 20 plates ≥ 1 Mkm² hold 98% of the surface) or welds it to the plate around most of its edge (microplate capture, Nicholson et al. 1994). Boundary crumbs (no interior node) are left: no new value. Adopted by `sweep-20261025-220434` (B-b): detached max 0.0018–0.0033 of the surface (control 0.060–0.095); window −100…0 Myr intra-oceanic trench 41.4k (control 55.5k), continental side 30.7k, production 5.15 (5.72), ridge 91.1k, shape ratio 1.57 (Earth 1.51); gates 3 of 3, median 39.2 km, volume −10.8% (control −12.5%), margins pass. `LEGACY_DEFAULTS` reads older runs as 'shared'. Thirty-ninth pass, six seeds (1004–1006 in `sweep-20261026-204140`): T1–T5 pass on six-seed means, guards pass; the .cmd's gate clause failed on seed 1004 (−313 m, one snapshot with 0.064% of the surface in the sample); kept as a recorded exception; the elevation offsets now pool the window (1004: −273 m) |
| `start_plates` (new, thirty-eighth pass; default 'voronoi') | | M: 'coast' grows the start's plate seeds only within their own domain, so the supercontinent's plates end at its coast and its rim is a plate boundary (Pangea ringed by subduction: Collins 2003; Cawood 2005). No new value. Start tests S1 continental-side trench ≥ 20k km (T3's lower edge) and S2 its share of subduction ≥ 0.60 (Earth today 30.7k / 51.5k), one-sided because the Triassic held ~1.55× today's subduction (van der Meer et al. 2014). `Coast Start Sweep.cmd` decides |
| `start_poles` (new, thirty-eighth pass; default 'random') | | M: 'forces' sets the oceanic start plates' speed and pole from the plate layer's own force law (`_driving`, fully relaxed) with the coast as trench and ocean-ocean boundaries as ridges; continental plates keep their dispersal motion. No new value. Pre-check seed 1001 with 'coast': window T2 32.9k, T3 27.7k, T4 4.23 km²/yr, gates pass; `Coast Start Sweep.cmd` decides |
| `water_volume_km3` | 1.335·10⁹ | M (Charette & Smith 2010) |
| GDH1 ocean depth | 2,600 + 365√age … | M (Stein & Stein 1992) |
| `ocean_thick_m` | 7 km | M |
| `cont_ref_thick_m` 38 km at `cont_ref_elev_m` 300 m | | M-ish (Earth's mean continental crust ~38–40 km); to source |
| `te_m` / `Te` | 30 km | M range (Earth 10–100 km; constant is a simplification, variable Te planned) |
| `subduction_erosion_H_m` | 700 m × v | M: ~38 km³ per km of trench per Myr at 54 mm/yr vs Earth 25–50, conservative 30 (von Huene & Scholl 1991) |
| `rollback_mm_yr` (new) | 17 | M, one region (Central Andes, Oncken et al. 2006) |
| `arc_accretion` (thirty-ninth pass, default off), `arc_buoyant_thick_m` 15 km, `arc_accrete_upper_m` 3.5 km | | P / M: consumed arc crust docks on the overriding plate instead of subducting: whole excess from 15 km (Cloos 1993: arcs > ~15 km collide), else its upper 3.5 km (Tetreault & Buiter 2014: upper 2–5 km accreted); on oceanic overriding plates spread over the arc band. Pre-checks (seed 1001): arcs lost 0.61 → 0.01–0.06, volume −10.8% → −4.3 / −7.3%, deep floor passes; but the plate layer runs away (34–40 plates, trench 85–100k km): cause open, not swept. Fortieth pass: six seeds show no runaway (32.2 plates vs 27.8, +5.6k km contour subduction, all continental side); arcs lost 0.03 of growth, but 2.2 km³/yr docked on continents from over-counted arcs |
| `trench_flux` (fortieth pass, default 'band'; candidate 'consumed') | **Forty-first pass: not adopted** (`sweep-20261028-200644`: arm C passes A1–A3 and routed 0.901 but fails T2–T4, > 45 km and margins). | P: Earth's trench fluxes (arc addition, subduction erosion) are per m of convergence per m of trench, i.e. per m² consumed; 'consumed' feeds each term with the oceanic area the trench actually consumed (routed to the overriding plate's nearest band-owning trench node, spread over its band), so it no longer depends on how rough the node-typed boundary is (band law: ocean arcs 1.7–2.7×, continental arcs 1.2–1.5×, subduction erosion 1.5–1.7× the contour's consumption at n = 217; 0.7× at n = 96). No new value. Design check: routed share ≥ 0.9 (pre-check 0.92 with pair polarity) |
| `trench_polarity` (fortieth pass, default 'node'; candidate 'pair') | **Forty-first pass: not adopted** (arm B fails T4, T5 and the margin guard by small margins; the defect remains in the default path; to return with the next plate-geometry change). | P: one subduction polarity per plate pair, as the overlap resolution already consumes crust (`_ocean_polarity`); the node rule (older ocean subducts, node by node) disagreed at 22% of oceanic overriding-typed nodes, putting arcs on the downgoing plate and slab pull on the overriding one. A correctness fix, not a value |
| `arc_H_cont_m` (new) | 750 m | M-derived: Earth's ~1.0 km³/yr at continental margins (Scholl & von Huene 2009) over ~25,000 km of trench (estimate) |
| `keep_deep_sediment` True, `basin_to_continent` False, water inventory, ledger | | P (mass conservation fixes) |
| `continental_sediment` (new, 24 Oct) | True (`sweep-20261024-143735`) | P/M: deposits on continents are sediment of `rho_sediment`, not crust (twentieth pass) |
| `dyn_lmax` 30, `dyn_slope` 1 | | M (Hoggard et al. 2016 spectrum) |
| `m` 0.4 (n = 1) | | M range (m/n 0.35–0.6 in channel profiles) |
| `Sc` 0.05 | | **contradicted at 11.53 km:** Earth's surface on that grid exceeds 0.05 on 3.1% of land (p99 0.076); at 46 km p99 is 0.036 (fifteenth pass) |
| `shelf_depth` 10 m, `shelf_gradient` 1.2 m/km, `shelf_wave_base_m` 50 | | M-ish (Earth shelves 1–2 m/km) |
| `K` 8·10⁻⁵, `Kd` 50, `Kd_marine` 2·10³ (11.53 km) | | tuned in the test bed at 11.5 km cells (how, to be checked in `docs/tectonic-prototype.md`; scale-specific effective values); the 11.53 km law gets the same steady-state test |
| `dt_myr` 1, `erosion_substeps` 4, `dt` 50 kyr, `hidden_steps` 3, `threads`, 1,000 km margin reach, 30% caps per step, sea seeds below −1,500 m | | N |
| `density_scale` 0.78 (coupling) | | P (matches broad isostatic uplift between the two crust densities) |
| L window 30 Myr | | decision (sect. 8; 20 / 40 Myr to test) |

## 4. Replacement order (proposed)

The F values interact, so they are replaced in an order where each step has an Earth test that does not depend on
the next ones:

1. **Mass budget at margins**: built and swept (`sweep-20261023-192601`, thirteenth pass). The realistic values
   (`margin_shortening`, accretion 0.13) cut the export to 11–13 km³/yr but lose 16% of continental volume and a
   third of the high ground, because the export is still 2–3× Earth's. Not adopted yet; they return after step 3.
   Still to do with Earth numbers: `arc_H_ocean_m` (intra-oceanic arcs), `pelagic_m_myr` (deep-sea rates).
2. ~~**Plate speeds**~~ checked 23 Oct: Earth-like (thirteenth pass); nothing to change. 26 Oct (thirty-sixth pass, `scripts/trench_origin.py`): the speed law applied to Earth's plates with Bird's
   boundary lengths gives Earth's speeds within 9% area-weighted (single plates miss: AN, IN, EU); the planet's fast
   oceanic plates come from slab added by detached plate pieces, not from the law.
3. **Erosion law at steady state** (46 and 11.53 km): K, θ, Kd from measured erosion rates and relief of Earth's belts
   and cratons, not from the planet's bands. Replaces `erosion_K_factor`, `erosion_theta`, `erosion_Kd_factor` and the
   landscape `THETA`.
4. **Rifted margins**: breakup at Earth's 5–15 km with the mechanism that keeps thin margins (necking width,
   sediment as its own layer of density 2,400), replacing the fitted 17.5 km.
4b. **Thick-crust collapse** (next, twenty-fifth pass): the collapse diffusion (`crust_kappa_m2_yr`,
   `collapse_thick_m`, `collapse_scale_m`, `collapse_max_factor`, tuned to Tibet-like plateaus) replaced by
   extension of thick, non-converging crust that thins it to Earth's post-extension thickness and regains area
   (plate split + pure-shear rifting, no breakup). Earth references and tests first (plan doc, twenty-fifth pass).
   25 Oct late (twenty-sixth pass): references measured, tests proposed, onset / end / rate sourced (onset ~45–50 km
   with no active convergence, end ~30 km, strain rate ~1–5e-8 /yr). The diffusion is the fixed-boundary mode of
   collapse (Rey et al. 2001) and its κ at 50–60 km matches a 15 km channel at 10¹⁹–10²⁰ Pa s: replace its threshold
   and e-fold with a channel law, or keep, is a design question; the missing part is the free-boundary mode (E).
   Twenty-seventh pass: E built, default off (`extension`, `ext_*`; values sourced, none fitted); the diffusion is kept
   for the first sweep so E is measured alone (`Extension Sweep.cmd`).
   Twenty-eighth pass: E not adopted (0 of 4 E tests; the "held" rule shields 81–100% of the thick crust).
   Twenty-ninth pass: E2 built, default off: `ext_held_rule` 'direction' (held pieces are cut across their long axis,
   not exempt; mechanism, Tibet), `ext_hot_myr` 100 Myr (measured/mechanism: delay of 0–100 Myr between thickening and
   extension, Sonder et al. 1987; observed 0 / 20–40 / ~70 Myr), the thickening clock's threshold is the measured
   `orogen_strain_rate` (no new value). Sweep: `Extension2 Sweep.cmd`.
   Thirtieth pass: E2 not adopted (trigger reach 0.01–0.04). Thirty-first pass: E3 built, default off:
   `ext_clock_flow` (the clock carried by the collapse flow), `ext_cut` 'band' (a microplate between two transforms),
   `ext_pair_motion` 'union' (the pair moves under the forces on it as one plate); no new value.
   Thirty-second pass: E3 not adopted (1 of 4 E tests; 60–70% of band events lost when the microplate docks).
   Thirty-third pass: E4 built, default off: `ext_side` 'free' (the band side whose far edge faces fewer collision
   nodes moves; mechanism, Rey et al. 2001; no new value). Its pre-check found a general defect: boundary typing
   (below, `boundary_normal`). Sweep with both: `Typing E4 Sweep.cmd`.
   Thirty-fourth pass: E4 not adopted (0–1 of 4 E tests; ledger guard fails; on ring typing at 45 km its design checks
   pass: lost 0.33, far side 0.07). Next: item C (thickness–elevation from the lithosphere's thermal state).
4c. **Boundary typing** (thirty-third pass): `classify_boundaries` typed a node by one grid neighbour's direction, so
   transforms that do not follow the grid were typed ridge / collision / subduction (pure transforms on random great
   circles: median 30% typed transform). `boundary_normal` 'ring' takes the normal from the node's two-ring
   (96-100%). A correctness fix, not a value. **Default 'ring' since the thirty-fourth pass** (`sweep-20261025-170555`:
   its control passes the gates and guards on 3 seeds; node typing failed the elevation tests on 2 of 3 by the
   young-ocean offset). Every band is reread after it (control numbers in the plan doc's thirty-fourth pass).
5. **Dynamic topography**: amplitude and spectrum from one published global estimate, used as is.
6. **Shelves**: κ and cliff retreat from published values.
7. **Coupling**: belt shapes from mechanism instead of the 80 km smoothing.

After each step the bands are read, not fitted. A band that then fails points to a missing mechanism, which is the
next item, as with the shelves and the deep-sea sink.
