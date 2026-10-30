# Planet generator: ideal flow (long-term target, 17 Oct 2026)

This is where the chain could end up, not what gets built first. The first build is `docs/planet-layer-plan.md`.
Each stage lists:
- its spacing and clock;
- the fields it outputs and which stage consumes them;
- the statistics to report against Earth.

● = in the first build, ○ = later.

## Overview

```
S0 Planet constants ─┐
S1 Mantle (absolute frame) ──► S2 Plates & crust history ──► S3 Paleoclimate & sea level
                                       │                              │
                                       └──────────► S4 Landscape evolution ◄──┘  (climate ↔ erosion coupling)
                                                           │
                                                   S5 Present climate & hydrology
                                                           │
                                                   S6 23 km conditioning bundle (same operator on Earth and planet)
                                                           │
                                                   S7 Meso 2.88 km (+ river / lake / coast consistency)
                                                           │
                                                   S8 Regional zoom 720 m → 90 m (on demand)
                                                           │
                                                   S9 Products, QA report, provenance
```

| Stage | Spacing / grid | Clock | Cost (*est.*) |
|---|---|---|---|
| S0 planet constants | scalars | — | — |
| S1 mantle | spherical harmonics, degree ≤ 30 (~650 km) | 1 Myr | seconds |
| S2 plates | cube n=217, 46 km | 1 Myr over 250+ Myr | ~5 min |
| S3 paleoclimate | ~200 km | every 1–5 Myr | minutes |
| S4 landscape | cube n=868, 11.53 km | 50 kyr over the last 30–40 Myr | ~45–60 min |
| S5 present climate | 23 km (11.53 km for hydrology) | steady state | minutes |
| S6 bundle | cube n=434, 23.06 km | — | seconds |
| S7 meso | cube N=3,472, 2.88 km | — | GPU, global |
| S8 zoom | 720 m / 90 m, regional | — | GPU, per region |

## Cross-cutting rules

1. **Same operator on both sides.** Every channel a learned model reads must be computed on Earth (training) by
   the same code path that computes it on the planet (inference). For example, the 23 km river channels come from
   running *our* router on Earth's 11.53 km DEM, not from a different product. This is the general form of the
   `bridge-adapted` lesson.
2. **Optional channels carry a known/unknown mask,** as the detail map already does. A model must work when a
   channel is missing.
3. **Each stage runs at the spacing it was built or trained at.**
4. **Mechanism above ~100 km, learning below.** That holds for climate too: physics at large scale, an
   Earth-trained downscaler below.
5. **Metres, years, m³/s throughout.** Every statistic has an Earth reference and a pass band fixed before the run.
6. **Physical bookkeeping is allowed; projections are not.** For example, a fixed ocean-water volume is
   physics. The present day only defines the 0 m datum.

---

## S0 Planet constants

| Output | Consumer | Notes |
|---|---|---|
| radius, gravity ● | all | Earth by default |
| rotation period ○ | S3, S5 | Coriolis; Hadley cell width; wind belts |
| obliquity, solar constant, greenhouse level ○ | S3, S5 | seasonality, ice, global mean temperature |
| ocean water volume ○ | S2, S3 | sea level from basin volume + ice (eustasy) |
| continental crust volume ● | S2 | with water volume, sets land fraction |
| mantle heat flow / potential temperature ○ | S1, S2 | plate speeds, ridge depth, Te law |
| densities, Te law ● | S2, S4 | crust, mantle, sediment, water |

## S1 Mantle (absolute frame) ○

| Output | Consumer |
|---|---|
| dynamic-topography field and rate (m/Myr; **non-load** motion) | S2, S4 |
| hotspots: position, buoyancy flux, start age; plume heads → LIP events | S2 |
| upwelling / downwelling pattern (biases where rifts and subduction start) | S2 |

**Stats:** residual-topography amplitude and spectrum (Hoggard et al. 2016); hotspot count against catalogues.

## S2 Plates and crust history

**Per node, every 1 Myr:**

| Output | Consumer |
|---|---|
| plate id, velocity vector ● | S4, S9 |
| boundary type and segment id; signed distance to each type ● | S4 forcing |
| convergence, divergence and shear rates; convergence obliquity ● | S4 forcing |
| subduction polarity, age of subducting slab (arc distance, back-arc extension) ○ | S4 |
| crust type, thickness, density, formation age ● | S4, S6 |
| tectonothermal age (last event); orogeny type (collisional, accretionary, arc) and age ● | S4 erodibility, S6 |
| **structural grain:** fold/thrust strike angle and strength ○ | S6 (anisotropic texture: Appalachian ridges, Zagros folds) |
| lithology class (sedimentary basin, carbonate, crystalline basement, volcanic, ophiolite; GLiM-like) ○ | S4 K, S6 |
| sediment thickness ● | S4 |
| oceanic lithosphere age, heat flow, Te from thermal age ●/○ | S4 (depth, flexure) |
| coarse isostatic elevation ● | S4 initial surface |
| province label (Hasterok classes) ● | S6, S9, validation |
| hotspot-track / LIP flags; sutures; failed rifts (aulacogens); terrane ids ○ | S4, S9 |

**History products:**
- parcel back-trace table (position per Myr for every present-day node) ●;
- thickening and non-load uplift history ●;
- event log of rifts, collisions and terrane transfers, with times and places ●;
- boundary polylines per Myr ○;
- paleo-maps every 10 Myr ●.

**Stats:**

| Statistic | Earth reference |
|---|---|
| plate count and size distribution | Bird 2003 (52 plates, 8,117 km² to 104 Mkm²) |
| boundary length by type | ridges ~65,000 km; subduction ~55,000 km |
| plate speeds | MORVEL |
| seafloor age distribution | Seton et al. 2020 (oldest ~180 Ma) |
| continental crust age histogram | — |
| province shares | Hasterok 2022 |
| land fraction | 29% |
| hypsometry | ETOPO 2022 |
| land within 500 km of an active boundary | — |
| passive vs active margin length | — |
| number of continents; largest continent's share over time (supercontinent index) | — |

## S3 Paleoclimate and sea level ○

An energy-balance atmosphere with latitude, continentality and lapse rate, simple ocean heat transport, moisture
carried along the prevailing wind belts, and orographic precipitation. It runs along S2's history every 1–5 Myr.

| Output (per epoch) | Consumer |
|---|---|
| mean temperature, seasonal range | S4 (frost, glaciers) |
| annual precipitation, runoff, aridity | S4 (discharge, water balance) |
| ice-sheet extent and thickness | S4 (glacial erosion, ice load) |
| eustatic sea-level offset (basin volume + ice) | S4 (shoreline migration, shelves) |

**Stats:** latitudinal temperature gradient; latitudes of the desert belts (Earth ~20–30°); glaciated area through
time.

## S4 Landscape evolution

**Processes added later (○):**
- discharge-based stream power (runoff from S3);
- precipitation–evaporation water balance, giving real lake levels, endorheic basins, playas and evaporites;
- glacial erosion;
- lithology-dependent K and erosion thresholds;
- sediment grain size (gravel / sand / mud) and deposit age;
- carbonate platforms and reefs in warm shallow seas;
- sea-level cycles;
- volcanic construction from S2's magmatic flux;
- variable-Te flexure;
- per-plate remapping, only if needed.

**Per node at present day:**

| Output | Consumer |
|---|---|
| elevation ●, bedrock elevation ○, sediment thickness ● | S6, S9 |
| surface deposit type and age ○ | S6 (roughness type) |
| filled surface ●, lake depth ●, lake type (open fresh / closed saline / playa) ○ | S5, S6 |
| drainage area ●, mean discharge (m³/s) ○ | S5, S6 |
| D8 receiver ●, multiple-flow fractions ○ | S6, S7 |
| river flag ●, Strahler order ○, channel width/depth from hydraulic geometry ○ | S6, S7, S8 |
| basin id, outlet ●; drains to sea / endorheic ● | S5, S9 |
| distance to outlet; normalised channel steepness ksn; χ (divide migration) ○ | S6 (tectonic activity cue), QA |
| local relief at 23 / 46 / 92 / 184 / 368 km ● | S6, QA |
| erosion rate and rock-uplift rate over the last 1 Myr ●/○ | S6 (detail predictor) |
| flexural deflection ● | QA |
| integrated glacial-erosion intensity; ice thickness ○ | S6 (glacial roughness) |
| coast type (delta, cliff, fjord, barrier, reef) ○; shelf width ● | S6, S9 |
| volcanic-edifice flag ○ | S6 |

**Vectors (○):** river polylines with discharge and order; basin polygons; lakes; coastline.

**Stats:**

| Statistic | Earth reference |
|---|---|
| 368 km land relief p10/p25/p50/p75 | 133/237/456/872 m |
| 40 km arrangement metrics | Alps / California / Plains |
| 23 km curvature per relief class | meso validation inputs |
| hypsometry per province | — |
| drainage density | — |
| Hack's law exponent | ~0.5–0.6 |
| basin-area distribution; largest basins | HydroBASINS (Amazon ~7 Mkm²) |
| endorheic share of land | ~18–20% |
| lake size distribution | HydroLAKES |
| sediment flux to the ocean | Milliman & Farnsworth 2011 (~19 Gt/yr) |
| denudation rate by setting | cosmogenic compilation, Portenga & Bierman 2011 |
| shelf width distribution | — |
| delta count | — |

## S5 Present climate and hydrology ○

Physics at large scale, plus an Earth-trained downscaler: WorldClim as target, conditioned on topography,
land/sea and latitude.

| Output | Consumer |
|---|---|
| bio1 (annual mean T), bio4 (T seasonality), bio12 (annual P), bio15 (P seasonality): the coarse model's climate channels | S6, S8 |
| monthly temperature and precipitation, PET, aridity index | S6, S9 |
| snowline / glacier equilibrium-line altitude; permafrost | S6, S8 |
| prevailing wind (rain shadows, dune orientation) | S6, S8 |
| final discharge and lake levels from the water balance | S6, S7, S9 |
| Köppen class, biome, vegetation-cover proxy | S8, S9 |

**Stats:** Köppen class shares (Beck et al. 2018); dryland share; latitudinal profiles; rain-shadow contrast
across ranges; precipitation–elevation relation.

## S6 23 km conditioning bundle (cube n=434)

Every channel lists its Earth source (training) and planet source (inference). Both go through the same operator.

| Channel | Earth source | Planet source |
|---|---|---|
| mean elevation (area mean) ● | ETOPO | S4 |
| sub-cell detail (std about the 23 km surface) ○ | meso dataset | predictor (province, age, relief, recent uplift) |
| **land fraction per cell** (sub-cell coast position; aimed at the 23 km coast stair-steps) ○ | Earth DEM at 11.53 km → same operator | S4 |
| **river channels:** log discharge / area of the main channel, flow direction, entry and exit edges ○ | our router on Earth's 11.53 km DEM | S4 |
| lake fraction, lake level, closed-basin flag ○ | same router, plus HydroLAKES as a check | S4 / S5 |
| **roughness type** probabilities: fluvial-dissected, glacial, aeolian, karst, volcanic, structural ridge-and-valley, layered plateau, shield / scoured ○ | landform classifications, LGM ice extent, karst map (WOKAM), erg maps, GLiM | S2 + S4 + S5 |
| structural grain (cos 2θ, sin 2θ, strength) ○ | fold/fault-trend maps, or orientation of 2.88 km ridges | S2 |
| lithology / hardness class, rock age ○ | GLiM, Hasterok | S2 |
| tectonic activity (recent uplift or ksn) ○ | ksn from Earth's DEM (same code) | S4 |
| climate bio1/bio4/bio12/bio15, ELA ○ | WorldClim | S5 |
| latitude ○ | — | — |

**Masks:** each optional channel gets known/unknown.

**Stats:** for every channel, the planet-vs-Earth distribution by relief class (domain-gap check) before any
model uses it.

## S7 Meso 2.88 km (N=3,472, ~72 M nodes)

| Output | Consumer |
|---|---|
| elevation ● | S8, S9 |
| river-channel probability or centreline ○ (multi-input meso) | S8 |
| lake and coast masks ○ | S8, S9 |
| derived: slope, flow accumulation, drainage density ●/○ | QA, S9 |

**Consistency step (○).** Measure first: does every S4 river continue as a connected 2.88 km channel, and are lakes
level? Correct only if that test fails.

**Stats:**
- band power at 5.8–46 km;
- channel incision and fine roughness per relief class (`diagnose_meso_plains`);
- coastline fractal dimension;
- river continuity: the share of S4 river length that shows up as a continuous channel;
- lake flatness.

## S8 Regional zoom, 720 m latent → 90 m decoder (on demand)

**Inputs:** the S7 field as a low-frequency guide ○, rivers, climate, roughness type and grain.

| Output | Consumer |
|---|---|
| 90 m elevation | S9 |
| slope, aspect, curvature; flow accumulation | S9 |
| river vectors with width; lake and coast vectors | S9 |
| snow and ice; rock-outcrop mask (steep + hard) | S9 (rendering) |
| land cover / biome material weights; regolith depth proxy | S9 (rendering) |

**Stats:**
- 90 m averaged to 2.88 km must equal S7 within a set tolerance, so zooms match the global map;
- slope distribution and spectrum against Copernicus GLO-90.

## S9 Products

- global equirectangular GeoTIFFs per stage, and a COG tile pyramid;
- GeoPackage vectors: rivers (order, discharge), basins, lakes, coastline, plate boundaries, sutures, provinces;
- climate rasters; biome map;
- paleo-map timeline (S2, every 10 Myr);
- a per-stage QA report (HTML) with every statistic against its Earth reference;
- a provenance manifest: seeds, code versions, checkpoint hashes, parameters;
- deterministic regional regeneration.

## Storage (float32 unless noted)

| Grid | Nodes | Per field | Typical |
|---|---|---|---|
| n=217 (46 km) | 0.28 M | 1.1 MB | 20 fields x 250 Myr ≈ 5.6 GB; float16, or 5 Myr spacing before the S4 window |
| n=434 (23 km) | 1.13 M | 4.5 MB | 30-channel bundle ≈ 135 MB |
| n=868 (11.5 km) | 4.52 M | 18 MB | ~30 fields ≈ 540 MB |
| N=3,472 (2.88 km) | 72 M | 290 MB (145 MB float16) | elevation + 3 masks ≈ 0.6 GB |
| 90 m | regional only | — | — |

## Earth reference datasets

| Dataset | Use |
|---|---|
| ETOPO 2022 / GEBCO | elevation, bathymetry |
| HydroSHEDS / HydroBASINS / HydroRIVERS, HydroLAKES | rivers, basins, lakes (check against our router on Earth) |
| WorldClim 2.1 | climate |
| Beck et al. 2018 | Köppen classes |
| GLiM (Hartmann & Moosdorf 2012) | lithology |
| Hasterok et al. 2022 | provinces |
| Seton et al. 2020 | seafloor age |
| CRUST1.0 | crustal thickness |
| Bird 2003 | plates |
| Hoggard et al. 2016 | dynamic topography |
| Lucazeau 2019 | heat flow |
| Portenga & Bierman 2011 | erosion rates |
| Milliman & Farnsworth 2011 | sediment flux |
| Randolph Glacier Inventory, LGM ice-extent compilations | ice |
| WOKAM | karst |
| global landform classifications | roughness types |
| Copernicus GLO-90 | 90 m checks |

## Rough order beyond the first build

1. Rivers in the 23 km bundle (S6), and the multi-input meso trained with them.
2. Water balance and S5 climate. Closed basins stay closed, which also fixes dead-flat fills.
3. Roughness type and structural grain (S2, S6).
4. Glacial processes (S3, S4).
5. S1 mantle and hotspots.
6. Zoom-stage conditioning (S8).
7. Products (S9).

Each step is adopted only if it beats the current chain on a held-out test fixed in advance.
