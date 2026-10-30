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

