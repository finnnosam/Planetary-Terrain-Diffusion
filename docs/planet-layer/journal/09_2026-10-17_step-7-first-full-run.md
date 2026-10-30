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

