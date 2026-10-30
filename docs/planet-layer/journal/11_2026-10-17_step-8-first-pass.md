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

