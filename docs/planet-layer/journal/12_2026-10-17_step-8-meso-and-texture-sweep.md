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

