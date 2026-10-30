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

