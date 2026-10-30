### Investigation (26 Oct, night): the pattern shared within a seed is the dynamic-topography field; it is plate-blind and its spectrum is the inverse of Earth's

**User report:** a noise pattern repeats across runs of one seed with different settings (not across seeds), at the
same checkpoint, but not between the start and the end of one run; its height can differ. Suspected link: the
dynamic-topography panel is identical within a seed.

**Cause (by construction, not a stray RNG):** `Mantle` has its own stream, `default_rng([seed, 7])`, and draws every
knot field when it is built (`dyn_std_m` 500, `dyn_lmax` 30, `dyn_slope` 1; knots every `dt_knot_myr` 40 from t0 − 40:
−290, −250, …, −10, +30) plus the hotspots; `advance` draws plume births from the same stream, depending only on t and
dt. Nothing in it reads the plates, so the field is bit-identical in every run of a seed whatever the settings (max |Δ|
0 on the snapshots of `sweep-20261026-184745`), and `elevation()` adds it to every node unflexed (`z = z + self.dyn`).

**Measured** on `sweep-20261026-184745` (A voronoi:random vs C coast:forces, 3 seeds; high-pass = elevation minus a
mask-normalized smooth of ~550 km rms radius, land and ocean separately; scripts, tables and figures in
`outputs/claude-sync/dyn-probe/`):
- Ocean, −100 / −50 / 0 Myr: same-checkpoint correlation between settings 0.13–0.32; with the field subtracted
  −0.02…+0.05; cross-seed control −0.02…+0.01. Slope of elevation on the field 0.77–1.03 (it passes at full height);
  it is 20–30% of the ocean's high-pass variance. It is the whole shared pattern.
- Land: slope 0.83–1.01 too, but only 4–8% of land's high-pass variance (crust structure dominates): correlation
  0.00–0.09 raw, −0.03…+0.05 without it. On land the eye catches it where it moves near-sea-level platforms across
  the coast colours (the round inland seas and green patches of `maps.png`). Before ~−200 Myr the shared start
  dominates on land (0.70–0.91 at −240).
- Time (seed 1001, ocean, A at t1 vs C at t2): non-zero only on the diagonal (0.20–0.32 from −200 on; off-diagonal
  |r| ≤ 0.04), as the field's own lags (independent knots; −10 and 0 share a knot, r 0.98). Hence "same checkpoint
  across runs, not start vs end of one run".
- Amplitude pulses with the knots: global rms 500–514 m at a knot, 360–380 m halfway (two independent fields blended
  linearly keep √(f² + (1 − f)²) of the std), so the pattern's height changes on a 40 Myr cycle, and the blobs fade
  in and out in place instead of moving.

**Against Earth:**
- Spectrum: the code's coefficient variance ∝ l⁻¹ (flat power per degree) puts 12% of the variance in degrees 1–3
  and 67% in 10–30 (wavelengths ~1,300–3,900 km; a blob is half a degree-30 wavelength, ~650 km). Earth's residual
  topography peaks at degree 2 (0.46–0.76 km²), holds ~85% of its power in degrees 1–3 and falls by more than 10× to
  ~0.02 km² at degree 30 (Valentine & Davies 2020, G-cubed 21, e2020GC009240). No single power law fits both ends: a
  slope of ~3 gives 88% in degrees 1–3 but a degree-30 / degree-2 ratio of 0.004 vs Earth's ~0.03.
- Pattern: Earth's long-wavelength dynamic topography is set by subduction history: lows above subducted slabs
  (circum-Pacific, southern Eurasia), highs far from convergent margins (Africa, central Pacific), changing slowly
  (Rubey et al. 2017, Solid Earth 8, 899: e.g. ~−500 m over ~25 Myr in the Chukchi Sea, ~20 m/Myr). The planet's
  field has no relation to its trenches.
- History: degree 30 / slope 1 was chosen on 17–18 Oct because it raised the texture score; this doc already noted
  then that "the texture score was passed by round mantle swells". `dyn_std_m` 500 is a fitted value in the audit.

**Is the sharing harmful? Partly (user, 26 Oct night: not convinced it is a positive thing; agreed).** Pairing
(common random numbers) would only cut noise in paired differences, and the sweeps compare 3-seed means of runs that
drift apart from the first steps, so little is gained. Costs: (1) the field is the one input that stays locked while
everything else in a seed decorrelates, so every sweep since 17 Oct was decided against the same three mantle
histories: adopted values may partly fit where these blobs sit relative to these continents (reused realizations);
(2) each setting is tested against a forcing that cannot respond to it; (3) it shows as a fixed imprint across
previews. Remedies: D makes the field setting-dependent by mechanism; and the adopted defaults get one check on
held-out seeds (e.g. 1004–1006) before the next large change.

**Proposed, not built: D, dynamic topography from the slabs** (`dyn_rule` 'slabs'; default 'random' until a sweep):
1. Long wavelengths from subduction history: each step the volume consumed at each trench node enters a mantle-frame
   slab record at that position; the record sinks (rate from slab-remnant studies, to source before building) and
   its surface effect is negative with a kernel that widens and weakens with depth; the mean is removed (the
   complement is the broad highs far from slabs). Amplitude and spectral shape from Earth (Valentine & Davies 2020,
   full text, in their normalization), never fitted.
2. Short wavelengths: a random part at Earth's share (degree-30 power ~1/30 of degree 2), blended with constant
   variance (a·cos + b·sin, not linear), so nothing pulses.
3. Hotspots unchanged.
Tests, fixed before building: (a) share of the field's variance in degrees 1–3 ≥ 0.6 (Earth ~0.85), degree-30 power
≤ 0.1 × degree 2; (b) lows above slabs: area-weighted correlation of the field with distance to slabs subducted in
the last ~100 Myr positive (low near slabs); (c) rate at fixed points median ≤ 20 m/Myr; (d) global rms constant
within ±10% over a run; (e) gates and guards. Land share, flooded continental crust and texture reported, not
scored: they were tuned with the blobs, and lower window relief would be the honest number.
Cheaper control in the same sweep: `dyn_slope` ~3 (Earth's share in degrees 1–3, still plate-blind, no blobs).

**Where it goes:** after reading the Coast Start sweep (item 1), before C: C's elevation tests and the lowland /
flat-window checks read through this field (the 18 Oct staircase followed plates drifting across the degree-30 field).

