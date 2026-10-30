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

