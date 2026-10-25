"""Plate layer, part 1 (docs/planet-layer-plan.md, step 5): rigid plates on the cube sphere, each keeping its crust in
its own rotating frame; ridges, subduction, collision thickening and ocean-floor ages; a supercontinent start.

Representation
--------------
Every plate owns a full raster on the same CubeSphere node set, interpreted in the plate's own frame, plus a
rotation Q (plate frame -> world). Each step (default 1 Myr):

1. rotate every plate about its Euler pole;
2. world node p is *claimed* by plate k if k's raster is set at the node nearest Q_k^T p;
3. no claim = divergence: new oceanic crust (7 km, age 0) joins the nearest plate (ridge);
   several claims = convergence: continental crust beats oceanic, younger ocean beats older; the losing ocean is
   subducted; losing continental crust under continental crust is added to the winner's thickness (collision);
4. thick crust spreads laterally (diffusion of thickness within continents: gravitational spreading of belts);
5. each plate's raster is updated by *forward* lookup (plate node j -> world node nearest Q_k j): nodes the plate
   still owns keep their values and receive only *increments* (collision, spreading); newly owned nodes are
   initialised from the world; nodes it lost are cleared. Crust is never resampled from itself, so 250 Myr of
   motion adds no interpolation blur.
6. plate motion: slab pull turns each plate's pole so that it moves toward its trenches and speeds it up with
   trench length (Forsyth & Uyeda 1975); continental area and collisions slow it.

Elevation here is isostatic only (ocean depth from age, GDH1; continents from Airy crustal thickness with water
loading below 0 m). Erosion, rifting, terrane transfer, hotspots and mantle swells are part 2 (step 6).
Times are in Myr (negative = before present), distances in m, speeds in cm/yr.
"""
import heapq

import numpy as np

from planet_training.landscape_evolution import njit
from planet_layer import fields, graph_landscape as gl
from planet_layer.sphere_flexure import SphereFlexure
from planet_layer.mantle import Mantle
from planet_layer import shelf as shelf_mod

PARAMS = {
    'dt_myr': 1.,
    'ocean_thick_m': 7000.,
    'cont_ref_thick_m': 38000.,    # continental crust standing at cont_ref_elev_m
    'cont_ref_elev_m': 300.,
    'rho_crust': 2830., 'rho_mantle': 3300., 'rho_water': 1030., 'rho_sediment': 2400.,
    # sea level from a fixed water inventory (Earth's ocean, Charette & Smith 2010): each elevation is returned relative
    # to the level that holds this volume, so a deeper or wider ocean lowers the sea instead of drowning continents.
    # Elevations are first computed on the fixed reference datum (ridges at -2,600 m, cont_ref_elev_m). None = sea
    # level fixed at the datum (before step 8)
    'water_volume_km3': 1.335e9,
    'te_m': 30e3,                  # elastic thickness for the elevation (flexural isostasy of crust and sediment)
    # continental crust thinner than cont_ref_thick_m stands at its own (local, Airy) depth instead of loading the
    # elastic plate with a negative load (eighteenth pass). Crust thinned by stretching is compensated locally (the
    # McKenzie 1978 stretching model assumes local isostasy; young rifted lithosphere is weak, Te of a few km, Watts
    # 2001). As a flexural load a narrow strip thinned to 10-20 km lifted the young ocean beside it to +2-12 km (Te 30
    # km spreads its rebound over ~70 km) and sat 2.5 km below its own depth. Thickening and the erosion of thick belts
    # stay flexural. On since 24 Oct (sweep-20261024-114002: passes the elevation tests on 3 seeds, 0% of young
    # coastal ocean above sea level vs 31-33% without)
    'thinning_local_isostasy': True,
    # ocean floor eroded below its sediment loses the rest from its crust (eighteenth pass): before, the clamp
    # dsed >= -sed dropped it from the ledger while the material was deposited elsewhere (8 km^3/yr created from
    # nothing with Earth rifting). False = the old leak, for reproducing earlier runs only
    'ocean_crust_erosion': True,
    # sediment on continental crust as its own layer (twentieth pass): deposits on continental nodes go to `sed`
    # (rho_sediment, a flexural load like ocean-floor sediment: dry where its top stands above sea level, displacing
    # water below), and erosion takes that sediment first, then the crust. Before, deposits became crust of rho_crust:
    # filling 1 m of water on a submerged margin took (rm-rw)/(rm-rc) = 4.8 m of material instead of
    # (rm-rw)/(rm-rs) = 2.5 m, so margins trapped ~2x Earth's volume and old rifted margins thickened into abrupt
    # edges (nineteenth pass, margin_age_probe). Sediment on continental crust consumed in a collision joins the
    # receiving belt's crust (sediments are folded and metamorphosed into orogens). CRUST1.0's total thickness counts
    # crust + sediment, so the reviews (crust_review, margin_width) compare that sum. False = deposits booked as crust
    # (all runs before 24 Oct). On since 24 Oct (sweep-20261024-143735: passes the elevation tests on 3 seeds)
    'continental_sediment': True,
    'threads': None,              # threads for the spherical-harmonic flexure (None: all cores)
    # coarse erosion: the landscape model's surface processes at the plate grid (calibrated: scripts/calibrate_coarse_erosion)
    'erosion': True, 'erosion_substeps': 4,
    'erosion_K_factor': .75, 'erosion_Sc': .05, 'erosion_Kd_factor': 4.,   # K 0.75: sweep-20261018-164500
    # incision threshold (A^m S at 46 km). 8 from step 6 (calibrated with the deep-sea sediment sink and one-cell rifted
    # margins); 6 since 22 Oct (sweep-20261022-194148, with breakup 17.5 km and shelves)
    'erosion_theta': 6.,
    'stable_erodibility': .15,     # old crust relative to fresh orogens (as the test bed)
    'orogen_erodibility_tau_myr': 50.,
    # which crust resets the orogeny clock (erodibility 1, decaying with orogen_erodibility_tau_myr) (twenty-fourth
    # pass). The 46 km erosion law was calibrated with erodibility 1 inside Bird's (2003) orogens (16% of Earth's land,
    # 76% of its land above 2 km; scripts/erosion_law_check). 'collision' (until now): only nodes receiving collision
    # volume, which left shortening belts and collapsed crust at the stable erodibility (young-orogen erodibility on
    # 1-2% of the planet's land, twenty-third pass). 'belt': also the subduction belt on overriding continental crust
    # (within 2 sigma of subduction_d0_cont_km, where 95% of the belt's added volume goes, while the trench
    # converges) and crust in the collapse regime (thicker than collapse_thick_m: hot, weak crust). 'strain': also
    # every continental node whose crust thickens or thins tectonically faster than orogen_strain_rate
    # (|tectonic thickening| / thickness: the areal dilatation of incompressible crust). 3.5e-9 /yr is measured:
    # the threshold at which GSRM v2.1's areal dilatation (Kreemer et al. 2014) flags as much of Earth's land as
    # Bird's orogens cover (17%; it flags 59% of orogen land, 8% of the rest, 67% of land above 2 km;
    # scripts/earth_deformation). Default 'belt' since the interior sweep (sweep-20261024-193840, twenty-fifth pass;
    # 'collision' before)
    'orogen_rule': 'belt', 'orogen_strain_rate': 3.5e-9,
    # subduction zones (overriding plate): thickening flux per unit trench length = convergence rate x H, spread as a
    # Gaussian band at distance d0 from the trench; oceanic overriding plates: island arcs, which become continental
    # crust once thick enough.
    # Continental margins, two modes:
    #  * margin_shortening False (the 18 Oct Andes-type belts, sweep-20261018-144253): H_cont 2,500 m of *new* crust per
    #    m of convergence. It stood in for Andean shortening + arc magmatism but added 5-8 km^3/yr of volume (Earth's
    #    arc addition at continental margins ~1.0; Scholl & von Huene 2009), which the trenches recycled (step 8,
    #    follow-up 1, tenth pass).
    #  * margin_shortening True (23 Oct): upper-plate shortening moves existing crust into the belt, volume conserved
    #    per trench segment: rate s = max(0, upper plate's trenchward speed - rollback_mm_yr) (Oncken et al. 2006:
    #    Central Andes shortening ~ upper-plate westward drift - slab rollback, which held at ~17 mm/yr; present
    #    shortening 9-12 mm/yr). The volume s x T (T: local crust thickness) per unit trench length comes from a
    #    foreland band at foreland_d0 +- foreland_sigma behind the belt (underthrust foreland; lateral crust flow
    #    refills it from the interior; the plate raster cannot lose area inside a rigid plate) and is added to the
    #    belt at d0 +- sigma. New volume is only arc magmatism, arc_H_cont_m x convergence (750 m: ~40 km^3 per km of
    #    trench per Myr at 54 mm/yr, Earth's ~1.0 km^3/yr over ~25,000 km of continental-margin trench)
    'subduction_H_cont_m': 2500., 'subduction_d0_cont_km': 150., 'subduction_sigma_cont_km': 60.,
    # (default False until the 3-seed sweep decides: one seed in the cloud workspace scored 0.64 vs 0.37, see the plan
    # doc's eleventh pass)
    'margin_shortening': False, 'rollback_mm_yr': 17., 'arc_H_cont_m': 750.,
    'foreland_d0_km': 350., 'foreland_sigma_km': 100.,
    'arc_H_ocean_m': 300., 'arc_d0_km': 150., 'arc_sigma_km': 50.,
    'arc_to_continent_thick_m': 20000.,
    # subduction erosion: the trench removes crust from the base of the overriding plate near it (forearc), the
    # main continental loss balancing arc growth on Earth (~1.3-1.7 km^3/yr); flux = convergence x H_se. On oceanic
    # crust only arc crust above ocean_thick_m can be removed
    'subduction_erosion_H_m': 700., 'subduction_erosion_d0_km': 50., 'subduction_erosion_sigma_km': 40.,
    # share of trench sediment scraped onto the overriding plate. Earth (von Huene & Scholl 1991): ~20% at small and
    # medium prisms (16,300 km of trench), ~30% at large prisms (8,200 km), none at nonaccretionary margins
    # (19,000 km): 0.13 length-weighted. 0.75 (18 Oct) matched Earth's *absolute* sediment subduction (~1.65
    # km^3/yr) while the deep-sea sink deleted half the export; with the sink closed it returned three quarters of
    # all exported sediment to the continents (a loop of gain up to 4; tenth pass). Default kept at 0.75 until the
    # 3-seed sweep decides (Margin Shortening Sweep.cmd)
    'sediment_accretion': .75,
    # continental rifting: a gap opening between continental crust is stretched continental crust (rift_thinning x its
    # neighbours' thickness, volume taken from them) until it would be thinner than breakup_thick_m; then ocean floor
    # breakup 17.5 km since 22 Oct (sweep-20261022-194148; 20 km left rifted margins one cell wide and only ~15% of the
    # continental crust under water, Earth ~29%)
    # 10 km since 24 Oct (twenty-first pass, with rift_rule pure_shear; Earth's hyper-extended margins break up at
    # ~5-15 km); 17.5 km was fitted with the node rule (sweep-20261022-194148)
    'rift_thinning': .6, 'breakup_thick_m': 10000.,
    # the volume for a rift gap comes from continental crust within rift_width_km of it (linear taper), not only
    # from its direct neighbours, so rifted margins thin over a necking zone (Earth: ~50-200 km) instead of one cell.
    # 100 km since 24 Oct (the realistic-margin value used in every sweep since 23 Oct; 0 before)
    'rift_width_km': 100.,
    # how a rift gap's crust is set (twenty-first pass). 'node': rift_thinning x its continental neighbours' mean
    # thickness, a factor per grid cell (38 -> 22.8 -> 13.7 km -> breakup: ~2 cells of stretched crust per side at any
    # resolution, 45-70 km of thin crust per km of passive edge against Earth's 140). 'pure_shear': the gap is the new
    # area of a stretching zone of half-width rift_width_km (linear taper of the strain rate from the gap cell's edge,
    # through continental crust only); every zone node thins as pure shear, T exp(-c) with c its share of the opened
    # area, and the gap receives exactly the volume the zone loses, i.e. the strain-weighted mean thickness of the
    # zone. The gap breaks up (ocean floor) when that mean is below breakup_thick_m. Stated in km, so a margin's profile
    # does not depend on the cell size; a stretched zone keeps thinning while the rift stays open, which gives the
    # hyper-extended domain (crust ~10 km over 70-200 km, Brune et al. 2014) and a necking zone at its inner edge.
    # pure_shear since 24 Oct (sweep-20261024-173200, 3 seeds: passes the elevation tests; thin crust per km of passive
    # edge 130-144 km at 1 degree vs 58-83 with 'node', Earth 140). 'node' reproduces earlier runs
    'rift_rule': 'pure_shear',
    # pelagic (biogenic and wind-blown) sediment on the ocean floor, m/Myr (Earth's deep sea ~1-10; GlobSed mean
    # thickness ~0.5 km); without it the planet's ocean floor carried ~0.1 km of sediment and stood 150-300 m too deep
    'pelagic_m_myr': 0.,
    # continental shelves (planet_layer.shelf): wave transport in shallow water, kappa_wave * exp(-depth / wave_base),
    # and coastal retreat at cliff_retreat_m_yr damped by the water in front of the coast; run after every coarse
    # erosion substep. shelf_kappa_m2_yr = 0 turns it off. On since 22 Oct (sweep-20261022-194148: 3e4 / 0.003 with
    # breakup 17.5 km; on one-cell margins it had no effect, sweep-20261021-182714)
    'shelf_kappa_m2_yr': 3e4, 'shelf_wave_base_m': 50., 'cliff_retreat_m_yr': .003,
    # deep-sea sediment: the landscape kernels' marine step places sediment only on shelves (within 150 km of land) and
    # drops the rest ("to the deep sea", right for a regional window). On the planet that deleted ~7 km^3/yr, more
    # than the continents' net loss (step 8 follow-up 1). Kept, the rest ponds where its path ended and fills the
    # sea-floor lows from the bottom up (graph_landscape.fill_and_spill: turbidites, abyssal plains), then rides the
    # ocean floor to the trenches like any other sediment. (A first version spread it by diffusion over the connected
    # open sea; enclosed pockets could not pass it on and grew 20-250 km towers.) False = the old sink
    'keep_deep_sediment': True,
    'mantle': True,                # dynamic topography and hotspots (planet_layer.mantle)
    # dynamic topography (planet_layer.mantle): std 800 m to spherical-harmonic degree 30 with a flat (slope 1) spectrum,
    # chosen by the step 8 sweep (sweep-20261017-223508: texture 0.23 vs 0.77 for the old 400 m / degree 12 / slope 2.5;
    # residual topography on Earth is +-~1 km with power to degree ~30, Hoggard et al. 2016, the high end of estimates)
    # (sweep-20261018-144253, scored with arrangement and budget too, then preferred 500 m)
    'dyn_std_m': 500., 'dyn_lmax': 30, 'dyn_slope': 1., 'dt_knot_myr': 40.,
    # plate splitting (Poisson events). Rifting: rate per Myr for a 100 Mkm^2 plate, scaled by area and doubled for
    # mostly continental plates; the split line prefers old sutures; halves diverge at split_speed_cm_yr.
    'split_rate_per_myr': .006, 'split_min_area_mkm2': 12., 'split_speed_cm_yr': 2., 'suture_weakness': .6,
    # subduction initiation: ocean older than init_age_myr on a plate breaks away and converges on the rest
    'init_age_myr': 170., 'init_rate_per_myr': .004, 'init_speed_cm_yr': 3.,
    'init_area_ref_mkm2': 2.,      # initiation rate scales with old-ocean area / this (capped at 5x)
    # suturing: two plates in continental collision along more than merge_contact_km for merge_after_myr in a row
    # become one plate (as India is joining Eurasia); the smaller plate's crust moves into the larger plate's frame
    'merge_contact_km': 1000., 'merge_after_myr': 40.,
    'merge_max_ratio': .35, 'merge_max_area_mkm2': 150.,   # docking: a smaller block onto a bigger one, size-capped
    # sediment-filled ocean floor becomes continental crust (margin prisms, filled remnant basins), keeping its
    # elevation: oceanic nodes with more than basin_to_continent_sed_m of sediment, or any sediment above sea level
    'basin_to_continent_sed_m': 10000.,
    # off since step 8: the conversion set the new continental thickness to match the node's elevation, not its mass
    # (1 km of sediment could become ~30 km of crust); over 250 Myr it added continent over 12.8% of the surface,
    # thinned the mean continental crust from 38.5 to 34.9 km and hid a ~30% continental volume loss. Thick fills now
    # stay sediment on oceanic crust, as in the South Caspian or the Gulf of Mexico
    'basin_to_continent': False,
    # lateral spreading of continental crust (m^2/yr): kappa_normal everywhere, plus gravitational collapse of thick
    # (hot, weak) crust, crust_kappa * (exp((T - collapse_thick) / collapse_scale) - 1) up to collapse_max_factor x
    # crust_kappa, which keeps plateaus near ~70 km as Tibet and the Altiplano. Cold crust of normal thickness barely
    # flows: kappa_normal = 1000 erased all 100 km crustal structure within ~10 Myr (step 8 texture check).
    # kappa_normal 30 since 24 Oct (twentieth pass, sweep-20261024-143735: passes the elevation tests on 3 seeds):
    # kappa ~ rho_c g (1 - rho_c/rho_m) h^3 / (3 eta) for a 10 km lower-crust channel is ~40 m^2/yr at eta = 1e21 Pa s
    # (stable continental lower crust; 1000 is ~4e19, hot crust), and Earth's 180 Myr Atlantic margins survive,
    # which needs kappa << 125. At 1000 the flow thickened old passive margins by 5-10 km per 100 Myr
    'crust_kappa_normal_m2_yr': 30.,
    'crust_kappa_m2_yr': 1000.,
    'collapse_thick_m': 45000.,
    'collapse_scale_m': 6000.,
    'collapse_max_factor': 200.,
    # collapse past the continent edge (seventeenth pass): the collapse part of the crust flow (crust thicker than
    # collapse_thick_m) may also cross the coastline into oceanic nodes of the same plate that are not on a plate
    # boundary, volume-conserving; such nodes become continental crust through the arc rule (arc_to_continent_thick_m).
    # On Earth thick orogens collapse and extend, adding continental area (Basin and Range, Aegean); before this, crust
    # flow stopped at the coastline, so collisions only ever removed continental area. Normal-thickness crust does not
    # cross (cold crust is strong). Default False until the 3-seed sweep decides
    'collapse_edge': False,
    # E, collapse of thick crust as continental extension (twenty-sixth / twenty-seventh passes). Crust thicker than
    # ext_onset_thick_m that no convergence holds any more (no collision boundary or subduction belt in or next to
    # the thick piece, which covers at least ext_min_area_mkm2) extends: its plate splits along the piece's long axis,
    # the halves diverge at ext_strain_rate_per_yr x the zone's full width, the rift gaps between them fill by pure
    # shear of continental crust within the zone's half-width (the piece's p90 distance from the cut, between
    # rift_width_km and ext_max_half_width_km; same kernel and breakup rule as rifting; only gaps within that
    # half-width of the thick piece, which is tracked in both halves' rasters, are the event's: the rest of the cut
    # rifts as usual), and once the crust E thinned or made (marked by time in 'ext') has a mean thickness at or below
    # ext_end_thick_m, the halves stop and weld
    # (ext_max_myr is only a guard). This is the free-boundary mode of collapse (Rey, Vanderhaeghe & Teyssier 2001);
    # the collapse diffusion above is the fixed-boundary mode and is kept. Earth values, not fitted: onset 45-50 km
    # (Basin and Range thickened to >= 45-50 km, Colgan et al. 2006; Aegean 40-50 km, Zhu et al. 2006); end ~30 km
    # (28-31 / 25-28 km); strain rate 1-5e-8 /yr (Basin and Range 1-2e-8, Bennett et al. 2003; Aegean 3-5e-8, Zhu
    # et al. 2006); zones 500-800 km wide (Basin and Range ~750 km at 40 N); the Aegean and the Basin and Range cover
    # ~0.5-1 Mkm^2. Default False until the 3-seed sweep decides. This paragraph describes E as built (twenty-seventh
    # pass) = ext_held_rule 'exempt', ext_hot_myr 0, ext_clock_flow False, ext_cut 'whole', ext_pair_motion 'mean';
    # the defaults of those switches below are E3 (held pieces cut across, hot crust only, carried clock, band cut,
    # pair moved as one plate)
    'extension': False, 'ext_onset_thick_m': 45000., 'ext_end_thick_m': 30000., 'ext_strain_rate_per_yr': 2e-8,
    'ext_min_area_mkm2': .5, 'ext_max_half_width_km': 400., 'ext_max_myr': 100.,
    # E2 (twenty-ninth pass), two changes to the trigger and one to the cut (used only with extension True):
    #  * ext_held_rule 'direction' (E2): convergence sets the direction of extension, not whether it happens. A thick
    #    piece touching a collision boundary or subduction belt is cut across its long axis (great circle through its
    #    centroid along its short axis: orogen-parallel stretching, as Tibet's east-west extension with north-south
    #    grabens while India converges); a free piece is cut along its long axis (Basin and Range, Aegean: stretching
    #    across the former belt). 'exempt': E as built (twenty-seventh pass), held pieces do not extend.
    #  * ext_hot_myr: only hot crust extends. A piece starts an event only if more than half of its area (area-weighted)
    #    was tectonically thickened within ext_hot_myr (the thickening clock 'thk': collision volume, margin shortening,
    #    arc and subduction thickening, at a rate above orogen_strain_rate; not collapse diffusion, not erosion). Crust
    #    never thickened in the run (the start's) is cold: Earth's thick Precambrian crust does not extend. 100 Myr:
    #    lithosphere thickened in the Sevier and Laramide orogenies becomes weak enough to extend (Moho > ~700 C) after
    #    a delay of 0 (initial Moho > ~700 C) to ~100 Myr (initial Moho < 450 C), and the observed gaps between
    #    compression and extension are ~0 (Pacific Northwest), 20-40 Myr (northern Great Basin) and ~70 Myr (southern
    #    Great Basin) (Sonder, England, Wernicke & Christiansen 1987, Geol. Soc. Spec. Publ. 28); crust thickened
    #    longer ago than that has had time to heat and, if it did not extend, to cool and strengthen (McKenzie 1978's
    #    lithosphere time constant is ~63 Myr). 0 switches the hot rule off (E as built)
    'ext_held_rule': 'direction', 'ext_hot_myr': 100.,
    # E3 (thirtieth pass), two changes to E2 (used only with extension True):
    #  * ext_clock_flow: the thickening clock is carried by the collapse flow. In each substep of the crust diffusion
    #    a continental node thickened by the flow faster than orogen_strain_rate (gain / thickness / substep) takes the
    #    most recent time among its neighbours that were thicker before the substep and carry a time (the volume's
    #    source), if more recent than its own (carry_clock). The flow never sets the present time, so the start's crust
    #    spreading into the start's crust stays untimed (cold). Crust flows in the collapse diffusion because it is hot
    #    and weak (kappa at 50-60 km ~ a 15 km channel at 1e19-1e20 Pa s, twenty-sixth pass), so its thermal age is its
    #    source's. No new number. False: E2 as built (in E2 64-89% of the > 45 km crust, nearly all of it thickened in
    #    the run, carried no time, and the trigger reached 1-4% of it).
    #  * ext_cut 'band': the cut ends at the zone. The relative pole u is the point on the cut's great circle 90
    #    degrees from the piece's centroid c (rotation about u moves c straight across the cut); the moving part is the
    #    plate's nodes on one side of the cut inside the band |asin(x.u)| R <= L (L: the piece's p90 distance from c
    #    along u, at least rift_width_km), from the cut to the plate's edge (the part connected to the piece); the
    #    smaller side moves. The band's edges are small circles about u, i.e. flow lines of the relative motion:
    #    transforms. A microplate between two transforms, so the cut opens only inside the band. The pole is carried
    #    in the parent's frame. 'whole': E2 as built (the cut across the whole plate added rift area elsewhere 2-5x
    #    E's own extension area, thirtieth pass)
    'ext_clock_flow': True, 'ext_cut': 'band',
    #  * ext_pair_motion 'union' (E3): while an event runs, the two halves' mean motion follows the usual forces on the
    #    pair as one plate (slab pull, ridge push, continental drag and collision resistance of both halves together,
    #    relaxed from the speed and axis the plate had when it split), so cutting a plate does not by itself change how
    #    fast it moves. 'mean' (as built): the area-weighted mean of each half's own force result; a band microplate cut
    #    along a collision belt takes the collision boundary with it, its large parent loses the collision resistance
    #    and speeds up, and the mean follows the parent (pre-check, thirty-first pass: area-weighted plate speed 7-9
    #    cm/yr against the control's 3-4, collisions consumed 2x the continent)
    'ext_pair_motion': 'union',
    # thickness of the supercontinent's crust at the start (twenty-fourth pass). 'assumed' (until now): 37.5 km + 2 km
    # beyond 300 km from old sutures + 4 km near them + a +-1.5 km field (interior p50 39.2-40.1 km, p10-p90 3-4 km
    # wide); it survives 250 Myr almost untouched (95-99% of the present interior, twenty-third pass). 'earth': the
    # same field quantile-mapped onto Earth's stable continental crust (EARTH_STABLE_CRUST_KM: CRUST1.0 crystalline
    # thickness of Precambrian and platform crust types more than 500 km from plate boundaries, p10 / 50 / 90 =
    # 32.9 / 38.2 / 43.0 km), keeping its spatial structure (thicker near old sutures); the thinned rim at the coast
    # as before. Default 'earth' since the interior sweep (sweep-20261024-193840, twenty-fifth pass; 'assumed' before)
    'start_crust': 'earth',
    'v_min_cm_yr': 1.5, 'v_slab_cm_yr': 9.,  # plate speed without / with a full slab
    'slab_ref_km': 6000.,          # trench length giving the full slab-pull speed
    'v_ridge_cm_yr': 1.5,          # extra speed from ridge push with ridge_ref_km of ridge
    'ridge_ref_km': 8000.,
    'ridge_torque': .3,            # weight of a ridge node relative to a trench node in turning the pole
    'cont_drag': .3,               # speed factor (1 - cont_drag * continental fraction)
    'collision_ref_km': 1500.,     # collision length that halves the target speed
    'tau_speed_myr': 10., 'tau_axis_myr': 20.,
    'boundary_min_mm_yr': 5.,      # below this, a boundary is a transform
    'hidden_steps': 3,             # overlapped crust is consumed only after this many steps under another plate
}

RIDGE, SUBDUCTION, COLLISION, TRANSFORM = 1, 2, 3, 4

# Earth's stable continental crust (start_crust 'earth'): crystalline thickness (CRUST1.0 crsthk - sedthk, Laske et
# al. 2013) of Precambrian and platform crust types (D-, E-, F-, G1, G2, H1, H2, I1, I2) more than 500 km from Bird's
# (2003) plate boundaries, area-weighted on the n = 217 graph, at the quantiles EARTH_STABLE_CRUST_Q (twenty-fourth
# pass; tests/test_plates.py recomputes it)
EARTH_STABLE_CRUST_Q = (.01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .99)
EARTH_STABLE_CRUST_KM = (25.65, 30.54, 32.91, 34.36, 35.2, 35.9, 36.43, 36.9, 37.19, 37.7, 38.17, 38.7, 38.99, 39.44,
                         39.99, 40.51, 41.02, 42.0, 43.0, 44.9, 48.79)


def quantile_map(values, weights, q=EARTH_STABLE_CRUST_Q, target=EARTH_STABLE_CRUST_KM):
    """Map values onto a target distribution by their area-weighted rank (mid-rank of each node's area), keeping their
    order; ranks beyond the outermost quantiles take the end values."""
    o = np.argsort(values, kind='stable')
    c = np.cumsum(weights[o]); u = (c-.5*weights[o])/c[-1]
    out = np.empty(len(values))
    out[o] = np.interp(u, q, target)
    return out


def gdh1_depth(age_myr):
    """Ocean depth (m, positive) from lithosphere age (Stein & Stein 1992)."""
    a = np.clip(age_myr, 0, None)
    return np.where(a < 20, 2600.+365.*np.sqrt(a), 5651.-2473.*np.exp(-.0278*a))


def isostatic_elevation(cont, thick, ocean_age, p=PARAMS):
    """Elevation (m): oceanic crust from age; continental crust from Airy thickness, water-loaded below 0 m."""
    dry = p['cont_ref_elev_m']+(thick-p['cont_ref_thick_m'])*(p['rho_mantle']-p['rho_crust'])/p['rho_mantle']
    wet = np.where(dry < 0, dry*p['rho_mantle']/(p['rho_mantle']-p['rho_water']), dry)
    return np.where(cont, wet, -gdh1_depth(ocean_age))


def sea_level_for_volume(z, area, volume_m3):
    """Level s (m) at which the water below it, sum of (s - z) * area over z < s, equals volume_m3."""
    order = np.argsort(z)
    zs, a = z[order], area[order]
    ca = np.cumsum(a); cza = np.cumsum(zs*a)
    held = zs*ca-cza                                  # volume below level zs[k] (the k+1 lowest nodes flooded)
    k = int(np.searchsorted(held, volume_m3))
    if k == 0:
        return float(zs[0])
    k = min(k, len(zs))
    return float((volume_m3+cza[k-1])/ca[k-1])


@njit(cache=True)
def nearest_label(source_label, nbr, dist):
    """Label of the nearest labelled node (label >= 0) for every node (multi-source Dijkstra)."""
    n = source_label.shape[0]
    d = np.full(n, 1e30)
    lab = source_label.copy()
    heap = [(0.0, -1)]
    heap.pop()
    for i in range(n):
        if source_label[i] >= 0:
            d[i] = 0.
            heapq.heappush(heap, (0., i))
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        if h > d[i]:
            continue
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0:
                continue
            v = h+dist[i, k]
            if v < d[j]:
                d[j] = v; lab[j] = lab[i]
                heapq.heappush(heap, (v, j))
    return lab, d


def margin_shortening(near, d, dom, s, thick, area, dt, d_belt, sg_belt, d_fore, sg_fore, cap=.3):
    """Volume-conserving upper-plate shortening (m of crust per node over dt). For every trench segment (the nodes
    whose nearest overriding trench node is the same, `near`), crust is taken from the foreland band (Gaussian in
    trench distance d at d_fore +- sg_fore) at s * thick per unit trench length, at most `cap` of a column, and the
    same volume is added to the belt (Gaussian at d_belt +- sg_belt) of that segment. dom: continental nodes of the
    overriding plate within reach. Segments without belt nodes move nothing."""
    gf = np.exp(-.5*((d-d_fore)/sg_fore)**2)/(sg_fore*np.sqrt(2*np.pi))
    gb = np.exp(-.5*((d-d_belt)/sg_belt)**2)/(sg_belt*np.sqrt(2*np.pi))
    seg = np.where(dom, near, -1)
    m = seg >= 0
    nseg = int(near.max())+1 if m.any() else 0
    out = np.zeros(d.shape[0])
    if nseg == 0:
        return out
    bw = np.where(m, gb*area, 0.)
    B = np.bincount(seg[m], weights=bw[m], minlength=nseg)
    sink = np.where(m, -np.minimum(s*dt*thick*gf, cap*thick), 0.)
    sink = np.where(m & (B[np.maximum(seg, 0)] > 0), sink, 0.)
    Rv = np.bincount(seg[m], weights=(sink*area)[m], minlength=nseg)            # m^3 removed per segment (<= 0)
    add = np.where(m, -Rv[np.maximum(seg, 0)]*bw/np.maximum(B[np.maximum(seg, 0)], 1e-300)/area, 0.)
    return sink+add


@njit(cache=True)
def edge_spread(th, cont, kap, excess, owner, ocean_ok, h, nbr, dist, axis, area):
    """Collapse flux of thick continental crust across the coastline (one explicit substep of length h, years): from a
    continental node i with excess thickness E_i over the collapse threshold into each axis neighbour j that is
    oceanic, owned by the same plate and not on a plate boundary: kap_i * E_i / d * (sqrt(a_i a_j) / d) * h m^3.
    Conservative; never takes more than the excess."""
    n = th.shape[0]
    out = th.copy()
    for i in range(n):
        if not cont[i] or excess[i] <= 0. or kap[i] <= 0.:
            continue
        tot = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not axis[i, k] or cont[j] or not ocean_ok[j] or owner[j] != owner[i]:
                continue
            d = dist[i, k]
            tot += kap[i]*excess[i]/d*(np.sqrt(area[i]*area[j])/d)*h
        if tot <= 0.:
            continue
        scale = min(1., excess[i]*area[i]/tot)
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not axis[i, k] or cont[j] or not ocean_ok[j] or owner[j] != owner[i]:
                continue
            d = dist[i, k]
            f = kap[i]*excess[i]/d*(np.sqrt(area[i]*area[j])/d)*h*scale
            out[i] -= f/area[i]
            out[j] += f/area[j]
    return out


@njit(cache=True)
def carry_clock(before, after, dom, src_ok, tk, nbr, axis, h_yr, rate):
    """E3 (thirtieth pass): the thickening clock carried by the collapse flow over one substep of h_yr years. A node of
    dom whose thickness gain, divided by its thickness and h_yr, exceeds rate takes the most recent time among its axis
    neighbours in src_ok that were thicker before the substep and carry a time (the volume's source), if that is more
    recent than its own (an untimed node, NaN, takes it). Only existing times move, at most one node per substep; the
    present time is never set here. Returns the new clock."""
    out = tk.copy()
    for i in range(before.shape[0]):
        if not dom[i]:
            continue
        if (after[i]-before[i])/max(before[i], 1e3)/h_yr <= rate:
            continue
        best = -np.inf
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not axis[i, k] or not src_ok[j]:
                continue
            if before[j] > before[i] and tk[j] == tk[j] and tk[j] > best:
                best = tk[j]
        if best > -np.inf and not (tk[i] >= best):
            out[i] = best
    return out


@njit(cache=True)
def _zone_dijkstra(g, zone, nbr, dist, cutoff, dd, touched):
    """Distances from node g to the zone nodes within cutoff, through zone nodes only. Fills dd (inf elsewhere) and
    returns the number of touched nodes listed in `touched` (g first); the caller resets dd[touched[:m]] = inf."""
    heap = [(0.0, g)]
    dd[g] = 0.
    touched[0] = g; m = 1
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        if h > dd[i]:
            continue
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not zone[j]:
                continue
            v = h+dist[i, k]
            if v < dd[j] and v <= cutoff:
                if dd[j] == np.inf:
                    touched[m] = j; m += 1
                dd[j] = v
                heapq.heappush(heap, (v, j))
    return m


@njit(cache=True)
def rift_pure_shear(rows, area, thick, zone, nbr, dist, W, Tb):
    """Pure-shear stretching of the zone around rift gap nodes (rift_rule 'pure_shear', see PARAMS).
    rows: gap nodes; zone: continental nodes that may stretch (not gaps); W: half-width (m) of the linear strain taper,
    measured from the gap cell's edge (half its spacing). Returns per row the strain-weighted mean zone thickness Tbar
    (before stretching), the gap crust Tg (0 where it would be thinner than Tb: breakup, or no zone), and per node the
    thinning dT (<= 0).
    Zone node i takes c_i = sum over rifting gaps g of a_g k_ig / K_g (K_g = sum_j a_j k_jg) and thins to
    T_i exp(-c_i); gap g receives its share a_g k_ig / (K_g c_i) of every removed volume, so the volume is conserved
    exactly; Tg = Tbar for small strain, less for a gap that is wide against W (a 46 km cell opening into a 100 km
    zone is a finite stretch)."""
    n = area.shape[0]
    R = rows.shape[0]
    dd = np.full(n, np.inf)
    touched = np.empty(n, np.int64)
    K = np.zeros(R); S = np.zeros(R)
    for r in range(R):
        g = rows[r]
        rg = .5*np.sqrt(area[g])
        m = _zone_dijkstra(g, zone, nbr, dist, W+rg, dd, touched)
        for q in range(1, m):
            j = touched[q]
            k = 1.-max(dd[j]-rg, 0.)/W
            if k > 0.:
                K[r] += area[j]*k; S[r] += area[j]*k*thick[j]
        for q in range(m):
            dd[touched[q]] = np.inf
    Tbar = np.where(K > 0., S/np.maximum(K, 1e-300), 0.)
    ok = (K > 0.) & (Tbar >= Tb)
    # finite strain makes Tg < Tbar (the gap is part of the stretched zone); gaps whose Tg falls below Tb break up,
    # and the rest is recomputed without them (a few rounds at most)
    for it in range(20):
        c, dT, Tg = _pure_shear_apply(rows, ok, K, area, thick, zone, nbr, dist, W, dd, touched)
        bad = ok & (Tg < Tb)
        if not bad.any():
            break
        ok &= ~bad
    return Tbar, Tg, dT


@njit(cache=True)
def _pure_shear_apply(rows, ok, K, area, thick, zone, nbr, dist, W, dd, touched):
    n = area.shape[0]
    R = rows.shape[0]
    c = np.zeros(n)
    for r in range(R):
        if not ok[r]:
            continue
        g = rows[r]
        rg = .5*np.sqrt(area[g])
        m = _zone_dijkstra(g, zone, nbr, dist, W+rg, dd, touched)
        for q in range(1, m):
            j = touched[q]
            k = 1.-max(dd[j]-rg, 0.)/W
            if k > 0.:
                c[j] += area[g]*k/K[r]
        for q in range(m):
            dd[touched[q]] = np.inf
    dT = np.zeros(n)
    for i in range(n):
        if c[i] > 0.:
            dT[i] = -thick[i]*(1.-np.exp(-c[i]))
    Tg = np.zeros(R)
    for r in range(R):
        if not ok[r]:
            continue
        g = rows[r]
        rg = .5*np.sqrt(area[g])
        m = _zone_dijkstra(g, zone, nbr, dist, W+rg, dd, touched)
        v = 0.
        for q in range(1, m):
            j = touched[q]
            k = 1.-max(dd[j]-rg, 0.)/W
            if k > 0.:
                v += -dT[j]*area[j]*(area[g]*k/K[r])/c[j]
        for q in range(m):
            dd[touched[q]] = np.inf
        Tg[r] = v/area[g]
    return c, dT, Tg


@njit(cache=True)
def components(mask, nbr):
    """Connected-component label of every mask node (-1 elsewhere)."""
    n = mask.shape[0]
    lab = np.full(n, -1, np.int64)
    work = np.empty(n, np.int64)
    c = 0
    for s in range(n):
        if not mask[s] or lab[s] >= 0:
            continue
        lab[s] = c; work[0] = s; w = 1
        while w > 0:
            w -= 1; i = work[w]
            for k in range(nbr.shape[1]):
                j = nbr[i, k]
                if j >= 0 and mask[j] and lab[j] < 0:
                    lab[j] = c; work[w] = j; w += 1
        c += 1
    return lab


@njit(cache=True)
def plate_holes(mask, nbr):
    """Unset raster nodes whose every neighbour is set, and the neighbour to copy from (sampling holes left by the
    nearest-node forward/backward maps not being exact inverses; real plate edges always touch unset nodes)."""
    n = mask.shape[0]
    hole = np.zeros(n, np.bool_)
    src = np.full(n, -1, np.int64)
    for i in range(n):
        if mask[i]:
            continue
        ok = True; first = -1
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0:
                continue
            if not mask[j]:
                ok = False; break
            if first < 0:
                first = j
        if ok and first >= 0:
            hole[i] = True; src[i] = first
    return hole, src


@njit(cache=True)
def classify_boundaries(owner, slot, omega, xyz, nbr, score, cont, radius, vmin):
    """Boundary type, convergence rate (m/yr, + = closing) and trench flag of every node. A boundary whose normal
    (closing or opening) rate is under half its shear rate, or under vmin, is a transform.

    omega: (K, 3) angular velocity vectors (rad/yr) by plate slot; slot[owner] gives the row.
    A node is a trench node when it would subduct under its converging neighbour (lower score)."""
    n = owner.shape[0]
    btype = np.zeros(n, np.int8)
    rate = np.zeros(n)
    trench = np.zeros(n, np.bool_)
    for i in range(n):
        a = slot[owner[i]]
        best = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or owner[j] == owner[i]:
                continue
            b = slot[owner[j]]
            w0 = omega[a, 0]-omega[b, 0]; w1 = omega[a, 1]-omega[b, 1]; w2 = omega[a, 2]-omega[b, 2]
            x0, x1, x2 = xyz[i, 0], xyz[i, 1], xyz[i, 2]
            v0 = (w1*x2-w2*x1)*radius; v1 = (w2*x0-w0*x2)*radius; v2 = (w0*x1-w1*x0)*radius
            d0 = xyz[j, 0]-x0; d1 = xyz[j, 1]-x1; d2 = xyz[j, 2]-x2
            dn = d0*x0+d1*x1+d2*x2
            d0 -= dn*x0; d1 -= dn*x1; d2 -= dn*x2
            L = np.sqrt(d0*d0+d1*d1+d2*d2)
            conv = (v0*d0+v1*d1+v2*d2)/L
            vt = np.sqrt(max(v0*v0+v1*v1+v2*v2-conv*conv, 0.))   # boundary-parallel (shear) component
            if abs(conv) >= abs(best) or btype[i] == 0:
                best = conv
                if abs(conv) < max(vmin, .5*vt):
                    btype[i] = 4; trench[i] = False
                elif conv > 0:
                    if cont[i] and cont[j]:
                        btype[i] = 3
                        trench[i] = False
                    else:
                        btype[i] = 2
                        trench[i] = score[i] < score[j]
                else:
                    btype[i] = 1; trench[i] = False
        rate[i] = best
    return btype, rate, trench


class Plate:
    def __init__(self, pid, size, axis, speed):
        self.id = pid
        self.Q = np.eye(3)
        self.axis = np.asarray(axis, np.float64)/np.linalg.norm(axis)
        self.speed = float(speed)                 # cm/yr at 90 degrees from the pole
        self.mask = np.zeros(size, bool)
        self.cont = np.zeros(size, bool)
        self.thick = np.zeros(size, np.float32)
        self.born = np.zeros(size, np.float32)    # crust formation time, Myr
        self.orog = np.full(size, np.nan, np.float32)  # last orogeny (collision) time, Myr
        self.under = np.zeros(size, np.int16)           # consecutive steps hidden under another plate
        self.sed = np.zeros(size, np.float32)            # sediment, m (on continental crust too with continental_sediment)
        self.ext = np.full(size, np.nan, np.float32)     # last thinned or made by extension E, Myr
        self.thk = np.full(size, np.nan, np.float32)     # last tectonic thickening (E2's thickening clock), Myr

    def omega(self, radius):
        """Angular velocity vector, rad/yr."""
        return self.axis*self.speed*1e-2/radius


class PlateSim:
    """Plates on a planet_layer.sphere_graph.CubeSphere (n = 217 for 46 km)."""

    def __init__(self, graph, seed=1, n_plates=12, n_cont_plates=5, cont_fraction=.38, t0=-250., start='supercontinent',
                 params=None):
        self.g = graph
        self.p = {**PARAMS, **(params or {})}
        self.t = float(t0)
        self.rng = np.random.default_rng(seed)
        self.bound = graph.diffusion_bound()
        self.h_min = graph.spacing_min()
        # loads are passed as mass per area (kg/m^2): subaerial loads displace mantle against air (rho_m), submarine
        # loads replace water and the depression fills with water (rho_m - rho_w)
        self.flex_dry = SphereFlexure.for_graph(graph, self.p['te_m'], rho_load=1., rho_mantle=self.p['rho_mantle'],
                                                nthreads=self.p['threads'])
        self.flex_wet = SphereFlexure.for_graph(graph, self.p['te_m'], rho_load=1.,
                                                rho_mantle=self.p['rho_mantle']-self.p['rho_water'], nthreads=self.p['threads'])
        mp = {k: self.p[k] for k in ('dyn_std_m', 'dyn_lmax', 'dyn_slope', 'dt_knot_myr')}
        self.mantle = Mantle(graph, np.random.default_rng([seed, 7]), t0, params=mp) if self.p['mantle'] else None
        self.dyn = self.mantle.topography(t0) if self.mantle else np.zeros(graph.size)
        self.next_id = 0
        self.plates = []
        self.events = []
        self.ledger = dict.fromkeys(('new_nodes', 'holes', 'increments', 'collision_out', 'cont_subducted', 'cont_lost_other'), 0.)
        # sediment on continental crust (continental_sediment_volume), by cause; with continental_sediment its
        # change over a run equals the sum of these (tests/test_plates.py)
        self.sed_ledger = dict.fromkeys(('increments', 'arc_continent', 'new_nodes', 'holes', 'collision_out',
                                         'cont_subducted', 'merge'), 0.)
        self.erosion_totals = {'continental_eroded_m3': 0., 'continental_deposited_m3': 0., 'ocean_sediment_m3': 0.}
        # continental area (flag) gained and lost by mechanism, m^2 (the area ledger; the rest of the net change is
        # nearest-node jitter, holes and merges, read as the residual)
        self.area_ledger = dict.fromkeys(('collision', 'subduction', 'rift', 'extension', 'arc'), 0.)
        self.ext_events = []                             # running extension events (E)
        self.ext_marked = np.zeros(graph.size, bool)
        # crust thicker than ext_onset_thick_m summed over steps (m^2 x steps): all of it, in pieces eligible to start
        # an event, in hot pieces, in held pieces, and hot nodes (E2's trigger reach = eligible / all)
        self.ext_reach = dict.fromkeys(('gt_onset', 'eligible', 'hot_pieces', 'held_pieces', 'hot_nodes'), 0.)
        if start != 'supercontinent':
            raise NotImplementedError('only the supercontinent start exists so far (dispersed start: later)')
        self._init_supercontinent(n_plates, n_cont_plates, cont_fraction)
        self.compose_from_plates()
        self.area0 = float(graph.area[self.world['cont']].sum())

    # ------------------------------------------------------------------ start
    def _new_plate(self, axis, speed):
        pl = Plate(self.next_id, self.g.size, axis, speed)
        self.next_id += 1
        self.plates.append(pl)
        return pl

    def _init_supercontinent(self, n_plates, n_cont_plates, frac):
        g, rng, p = self.g, self.rng, self.p
        xyz = g.xyz
        s = rng.normal(size=3); s /= np.linalg.norm(s)
        self.sc_centre = s
        ang = np.arccos(np.clip(xyz@s, -1, 1))
        score = ang+.35*fields.random_field(xyz, 12, 2.5, rng)
        cont = score <= np.quantile(score, frac)
        lab = components(cont, g.nbr)
        sizes = np.bincount(lab[lab >= 0], weights=g.area[lab >= 0])
        cont = lab == np.argmax(sizes)
        ocean_lab = components(~cont, g.nbr)            # fill small enclosed seas
        osz = np.bincount(ocean_lab[ocean_lab >= 0], weights=g.area[ocean_lab >= 0])
        small = np.flatnonzero(osz < .02*g.area.sum())
        cont |= np.isin(ocean_lab, small)

        # cratons (Voronoi cells inside the continent) welded by old orogens
        cids = np.flatnonzero(cont)
        n_cr = int(rng.integers(7, 12))
        seeds = np.full(g.size, -1, np.int64)
        seeds[rng.choice(cids, n_cr, replace=False)] = np.arange(n_cr)
        clab, _ = nearest_label(np.where(cont, seeds, -2), g.nbr, np.where(cont[g.nbr] & cont[:, None], g.dist, 1e30))
        boundary = self._label_boundary(cont, clab)
        d_suture = gl.distance_to(boundary, g.nbr, g.dist)
        d_coast = gl.distance_to(~cont, g.nbr, g.dist)
        craton_age = rng.uniform(1500, 3500, n_cr)
        pair_age = rng.uniform(300, 1500, (n_cr, n_cr)); pair_age = np.triu(pair_age)+np.triu(pair_age, 1).T
        thick = (p['cont_ref_thick_m']-500.+1500.*fields.random_field(xyz, 24, 2., rng)
                 + 4000.*np.exp(-d_suture/150e3)+2000.*np.clip((d_suture-300e3)/300e3, 0, 1))
        if p['start_crust'] == 'earth':                 # Earth's stable crust distribution, the field's order kept
            thick[cont] = quantile_map(thick[cont], g.area[cont], target=tuple(1e3*k for k in EARTH_STABLE_CRUST_KM))
        elif p['start_crust'] != 'assumed':
            raise ValueError(f"start_crust {p['start_crust']!r}: 'assumed' or 'earth'")
        thick = thick-(p['cont_ref_thick_m']-28000.)*np.clip(1-d_coast/200e3, 0, 1)   # thinned rim at the margins
        born_c = self.t-craton_age[np.clip(clab, 0, None)]
        # old orogen age: from the two cratons it welds (nearest other-craton label across the suture)
        other = self._other_label(cont, clab)
        orog_c = np.where(d_suture < 250e3, self.t-pair_age[np.clip(clab, 0, None), np.clip(other, 0, None)], np.nan)

        # plates: Voronoi with wiggled distances; n_cont_plates seeds inside the continent
        oids = np.flatnonzero(~cont)
        pseeds = np.concatenate([rng.choice(cids, n_cont_plates, replace=False), rng.choice(oids, n_plates-n_cont_plates, replace=False)])
        lab0 = np.full(g.size, -1, np.int64); lab0[pseeds] = np.arange(n_plates)
        wig = 1+.35*fields.random_field(xyz, 10, 2., rng)
        w = np.clip(.5*(wig[:, None]+wig[np.clip(g.nbr, 0, None)]), .3, None)
        plab, _ = nearest_label(lab0, g.nbr, g.dist*w)

        # ocean ages: distance from initial ridges (ocean-ocean plate boundaries), half rate 3 cm/yr, cap 180 Myr
        ridge = (~cont) & self._label_boundary(~cont, plab)
        d_ridge = gl.distance_to(ridge, g.nbr, g.dist)
        age_o = np.clip(d_ridge/30e3*(1+.2*fields.random_field(xyz, 16, 2., rng)), 0, 180)

        for k in range(n_plates):
            own = plab == k
            c = (g.area[own, None]*xyz[own]).sum(0); c /= np.linalg.norm(c)
            if k < n_cont_plates:
                axis = np.cross(s, c)+.3*rng.normal(size=3)*np.linalg.norm(np.cross(s, c))
                speed = rng.uniform(2., 4.)
            else:
                axis = rng.normal(size=3)
                speed = rng.uniform(3., 7.)
            pl = self._new_plate(axis, speed)
            pl.mask[:] = own
            pl.cont[:] = own & cont
            pl.thick[:] = np.where(cont, thick, p['ocean_thick_m'])
            pl.born[:] = np.where(cont, born_c, self.t-age_o)
            pl.orog[:] = np.where(cont, orog_c, np.nan)

    def _label_boundary(self, mask, lab):
        g = self.g
        nb = np.where(g.nbr >= 0, g.nbr, np.arange(g.size)[:, None])
        return mask & ((mask[nb] & (lab[nb] != lab[:, None])).any(1))

    def _other_label(self, mask, lab):
        """For each node, the label of the nearest node of a different label within the mask."""
        g = self.g
        nb = np.where(g.nbr >= 0, g.nbr, np.arange(g.size)[:, None])
        diff = mask[nb] & (lab[nb] != lab[:, None]) & mask[:, None]
        src = np.where(diff.any(1), np.take_along_axis(lab[nb], np.argmax(diff, 1)[:, None], 1)[:, 0], -1)
        out, _ = nearest_label(np.where(mask, src, -1), g.nbr, g.dist)
        return out

    # ------------------------------------------------------------------ composition
    def _claims(self):
        g = self.g
        K = len(self.plates)
        idx = np.empty((K, g.size), np.int64)
        claimed = np.empty((K, g.size), bool)
        for k, pl in enumerate(self.plates):
            idx[k] = fields.nearest_nodes(g, g.xyz@pl.Q)          # Q^T p, as row vectors
            claimed[k] = pl.mask[idx[k]]
        return idx, claimed

    def _score(self, cont, age, cont_share):
        """Overlap priority: continental crust (bigger continent first) beats ocean; younger ocean beats older."""
        return np.where(cont, 1e6+1e5*cont_share, -age)

    def _ocean_polarity(self, claimed, cont_k, age_k, nclaim, top):
        """Ocean-ocean overlaps: one subduction polarity per pair of plates, not per node. The plate whose crust along
        the contact is younger (lighter) overrides; the polarity is remembered while the pair stays in contact, as
        real subduction zones keep their polarity."""
        ocean_only = (nclaim >= 2) & ~(claimed & cont_k).any(0)
        if not ocean_only.any():
            return top
        pid = np.array([pl.id for pl in self.plates])
        pol = getattr(self, 'polarity', {})
        cols = np.flatnonzero(ocean_only)
        cl = claimed[:, cols]
        K = len(self.plates)
        wins = np.zeros((K, cols.size), np.int32)
        for a in range(K):
            for b in range(a+1, K):
                both = cl[a] & cl[b]
                if not both.any():
                    continue
                key = (pid[a], pid[b])
                if key not in pol:
                    ya = age_k[a, cols[both]].mean(); yb = age_k[b, cols[both]].mean()
                    pol[key] = pid[a] if ya <= yb else pid[b]
                over = a if pol[key] == pid[a] else b
                wins[over] += both
        self.polarity = pol
        best = np.argmax(np.where(cl, wins, -1), 0)
        top = top.copy()
        top[cols] = best
        return top

    def compose_from_plates(self):
        """World fields from the plate rasters without moving anything (start, and after events)."""
        idx, claimed = self._claims()
        self._world_from(idx, claimed, record=False)

    def _world_from(self, idx, claimed, record=True):
        g, p = self.g, self.p
        K = len(self.plates)
        cols = np.arange(g.size)
        cont_k = np.stack([pl.cont[idx[k]] for k, pl in enumerate(self.plates)])
        age_k = np.stack([self.t-pl.born[idx[k]] for k, pl in enumerate(self.plates)])
        share = np.array([(g.area*(pl.mask & pl.cont)).sum() for pl in self.plates])/g.area.sum()
        sc = np.where(claimed, self._score(cont_k, age_k, share[:, None]), -np.inf)
        nclaim = claimed.sum(0)
        top = np.argmax(sc, 0)
        top = self._ocean_polarity(claimed, cont_k, age_k, nclaim, top)
        gap = nclaim == 0
        # gaps join the nearest plate (ridge); label by slot
        lab = np.where(gap, -1, top)
        lab, _ = nearest_label(lab.astype(np.int64), g.nbr, g.dist)
        slot = lab
        pid = np.array([pl.id for pl in self.plates])
        self.owner_slot = slot
        self.owner = pid[slot]
        wcont = np.where(gap, False, cont_k[slot, cols])
        wthick = np.where(gap, p['ocean_thick_m'], np.stack([pl.thick[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        wborn = np.where(gap, self.t, np.stack([pl.born[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        worog = np.where(gap, np.nan, np.stack([pl.orog[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        wsed = np.where(gap, 0., np.stack([pl.sed[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        wext = np.where(gap, np.nan, np.stack([pl.ext[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        wthk = np.where(gap, np.nan, np.stack([pl.thk[idx[k]] for k, pl in enumerate(self.plates)])[slot, cols]).astype(np.float64)
        self.new_crust = gap
        self.world = {'cont': wcont, 'thick': wthick, 'born': wborn, 'orog': worog, 'sed': wsed, 'ext': wext, 'thk': wthk}

    # ------------------------------------------------------------------ stepping
    def step(self):
        g, p = self.g, self.p
        dt = p['dt_myr']
        for pl in self.plates:
            pl.Q = fields.rotation(pl.axis, pl.speed*1e-2*dt*1e6/g.radius)@pl.Q
        self.t += dt
        if self.mantle is not None:
            self.mantle.advance(self.t, dt)
            self.dyn = self.mantle.topography(self.t)
        idx, claimed = self._claims()
        self._world_from(idx, claimed, record=True)
        w = self.world
        rift_dthick = self._rift()
        # lateral spreading of thick crust inside continents (conservative; increments only)
        before = w['thick'].copy()
        th = before
        T = dt*1e6
        carry = bool(p['extension'] and p['ext_clock_flow'])
        tk = w['thk'].copy() if carry else None          # E3: the clock carried by the flow (carry_clock)
        while T > 0:                                     # nonlinear: substeps follow the current largest diffusivity
            kap = p['crust_kappa_normal_m2_yr']+p['crust_kappa_m2_yr']*np.minimum(
                np.expm1(np.clip(th-p['collapse_thick_m'], 0, None)/p['collapse_scale_m']), p['collapse_max_factor'])
            kmax = kap[w['cont']].max() if w['cont'].any() else 0.
            if kmax <= 0:
                break
            h = min(T, .9/(kmax*self.bound))
            th0 = th
            th = gl.diffuse_variable(th, w['cont'], kap, h, g.nbr, g.dist, g.axis, g.area)
            if carry:
                tk = carry_clock(th0, th, w['cont'], w['cont'], tk, g.nbr, g.axis, h, p['orogen_strain_rate'])
            if p['collapse_edge'] and hasattr(self, 'btype'):
                kc = p['crust_kappa_m2_yr']*np.minimum(np.expm1(np.clip(th-p['collapse_thick_m'], 0, None)/p['collapse_scale_m']),
                                                     p['collapse_max_factor'])
                th1 = edge_spread(th, w['cont'], kc, np.clip(th-p['collapse_thick_m'], 0, None), self.owner,
                                  (~w['cont']) & (self.btype == 0), h, g.nbr, g.dist, g.axis, g.area)
                self.erosion_totals['edge_collapse_m3'] = self.erosion_totals.get('edge_collapse_m3', 0.)+float(
                    (np.maximum(th1-th, 0)*g.area)[~w['cont']].sum())
                if carry:
                    tk = carry_clock(th, th1, w['cont'] | ((~w['cont']) & (self.btype == 0)), w['cont'], tk, g.nbr, g.axis,
                                     h, p['orogen_strain_rate'])
                th = th1
            T -= h
        self.thk_flow = tk
        dthick = th-before+rift_dthick
        th = th+rift_dthick
        dsed = np.zeros(g.size)
        sub = self._subduction_thickening()
        self.thickening = np.array(sub, np.float64)      # the thickening clock's source; + collision volume below
        if self.mantle is not None:
            vol = self.mantle.volcanic(self.t, dt)
            self.erosion_totals['volcanic_added_m3'] = self.erosion_totals.get('volcanic_added_m3', 0.)+float((vol*g.area).sum())
            sub = sub+vol
        dthick = dthick+sub
        th = th+sub
        self.tectonic = dthick.copy()                    # tectonic thickening this step, m (no erosion); + collisions below
        arc = (~w['cont']) & (th >= p['arc_to_continent_thick_m'])
        if arc.any():                                    # island arcs grown thick enough are continental crust
            self.area_ledger['arc'] += float(g.area[arc].sum())
            w['cont'] = w['cont'] | arc
            self.new_continent = arc
        else:
            self.new_continent = np.zeros(g.size, bool)
        if p['erosion']:
            e_cont, dsed = self._erode(th)
            dthick = dthick+e_cont
            th = th+e_cont
        w['thick'] = th
        if p['pelagic_m_myr'] > 0:
            pel = np.where(w['cont'], 0., p['pelagic_m_myr']*dt)
            dsed = dsed+pel
            self.erosion_totals['pelagic_m3'] = self.erosion_totals.get('pelagic_m3', 0.)+float((pel*g.area).sum())
        w['sed'] = w['sed']+dsed
        conv_t, conv_s = self._sediment_basins_to_continent()
        dthick = dthick+conv_t
        dsed = dsed+conv_s
        self._update_rasters(dthick, idx, dsed)
        self.plates = [pl for pl in self.plates if pl.mask.any()]
        self._classify()
        self._update_motion()
        if p['extension']:
            self._extension_motion()
        self._merge_events()
        if p['extension']:
            self._extension_events()
        self._split_events()

    def _rift(self):
        """Gaps opening between continental crust become stretched continental crust (see PARAMS). Returns the
        thinning of the continental neighbours that supply the volume (m, <= 0)."""
        g, p, w = self.g, self.p, self.world
        out = np.zeros(g.size)
        self.ext_marked = np.zeros(g.size, bool)
        gap = self.new_crust
        if not gap.any():
            return out
        rows = np.flatnonzero(gap)
        nb = g.nbr[rows]
        ok = nb >= 0
        nbs = np.where(ok, nb, 0)
        solid = ok & ~gap[nbs]
        cn = solid & w['cont'][nbs]
        ncont = cn.sum(1)
        if p['rift_rule'] == 'pure_shear':
            return self._rift_pure_shear(rows, ncont, solid, cn, nbs)
        meanT = np.where(cn, w['thick'][nbs], 0.).sum(1)/np.maximum(ncont, 1)
        Tg = p['rift_thinning']*meanT
        rift = (ncont > 0) & (ncont*2 >= solid.sum(1)) & (Tg >= p['breakup_thick_m'])
        if not rift.any():
            return out
        r, Tr, cr = rows[rift], Tg[rift], cn[rift]
        a = g.area
        if p['rift_width_km'] > 0:                       # necking zone: the volume comes from a tapered band
            src = np.zeros(g.size, bool); src[r] = True
            D = max(p['rift_width_km']*1e3, 1.5*float(g.dist[g.nbr >= 0].max()))   # at least the direct neighbours
            d = gl.distance_to(src, g.nbr, g.dist, D)
            band = w['cont'] & ~gap & (d < D)
            wt = np.where(band, (1-d/D)*a, 0.)
            take = (Tr*a[r]).sum()*wt/max(wt.sum(), 1e-9)
        else:
            share = (Tr*a[r]/ncont[rift])[:, None]*cr      # volume each continental neighbour gives (m^3)
            take = np.zeros(g.size)
            np.add.at(take, nbs[rift][cr], share[cr])
        out = -np.minimum(take/a, .5*w['thick'])
        w['cont'][r] = True
        w['thick'][r] = Tr
        w['born'][r] = np.where(cr, w['born'][nbs[rift]], np.inf).min(1)   # stretched old crust keeps its age
        self.erosion_totals['rift_nodes'] = self.erosion_totals.get('rift_nodes', 0)+int(r.size)
        return out

    def _rift_pure_shear(self, rows, ncont, solid, cn, nbs):
        """rift_rule 'pure_shear' (see PARAMS): gaps between continental crust are the new area of a stretching zone."""
        g, p, w = self.g, self.p, self.world
        out = np.zeros(g.size)
        cand = (ncont > 0) & (ncont*2 >= solid.sum(1))
        if not cand.any():
            return out
        rc = rows[cand]
        zone = w['cont'] & ~self.new_crust
        if p['rift_width_km'] <= 0:
            raise ValueError("rift_rule 'pure_shear' needs rift_width_km > 0 (the stretching zone's half-width)")
        W = p['rift_width_km']*1e3
        # gaps between the two halves of a running extension event (E) stretch that event's zone (its own half-width,
        # its two plates' continental crust); the rest as before
        ev_of = np.full(rc.size, -1)
        near = {}
        if p['extension'] and self.ext_events:
            own_r = self.owner[rc]
            nb_own = self.owner[nbs[cand]]
            sol = solid[cand]
            for e, ev in enumerate(self.ext_events):
                a_, b_ = ev['parent'], ev['child']
                near[e] = self._near_piece(ev)
                other = np.where(own_r == a_, b_, np.where(own_r == b_, a_, -1))
                sel = (other >= 0) & (sol & (nb_own == other[:, None])).any(1) & (ev_of < 0) & near[e][rc]
                ev_of[sel] = e
        Tg = np.zeros(rc.size)
        dT = np.zeros(g.size)
        th = w['thick'].astype(np.float64)
        groups = [(ev_of < 0, W, zone)]+[(ev_of == e, ev['W'], zone & near[e] & np.isin(self.owner, (ev['parent'], ev['child'])))
                                          for e, ev in enumerate(self.ext_events) if (ev_of == e).any()]
        for sel, Wg, zg in groups:
            if not sel.any():
                continue
            _, Tg_s, dT_s = rift_pure_shear(rc[sel].astype(np.int64), g.area, th+dT, zg, g.nbr, g.dist, Wg,
                                            p['breakup_thick_m'])
            Tg[sel] = Tg_s
            if Wg is not W or zg is not zone:            # an extension group: mark the crust it thinned
                self.ext_marked |= dT_s < 0
            dT = dT+dT_s
        rift = Tg > 0.
        if not rift.any():
            return dT
        isE = rift & (ev_of >= 0)
        if isE.any():
            self.ext_marked[rc[isE]] = True
            self.area_ledger['extension'] += float(g.area[rc[isE]].sum())
            self.erosion_totals['ext_nodes'] = self.erosion_totals.get('ext_nodes', 0)+int(isE.sum())
        self.area_ledger['rift'] += float(g.area[rc[rift & (ev_of < 0)]].sum())
        w['ext'] = np.where(self.ext_marked, self.t, w['ext'])
        r, cr = rc[rift], cn[cand][rift]
        w['cont'][r] = True
        w['thick'][r] = Tg[rift]
        w['born'][r] = np.where(cr, w['born'][nbs[cand][rift]], np.inf).min(1)   # stretched old crust keeps its age
        self.erosion_totals['rift_nodes'] = self.erosion_totals.get('rift_nodes', 0)+int(r.size)
        self.erosion_totals['rift_stretched_m3'] = self.erosion_totals.get('rift_stretched_m3', 0.)+float((Tg[rift]*g.area[r]).sum())
        return dT

    def _sediment_basins_to_continent(self):
        """Oceanic nodes carrying more than basin_to_continent_sed_m of sediment, or sediment above sea level, become
        continental crust. Its thickness is set so the node keeps its present elevation (the isostatic equivalent)."""
        g, p, w = self.g, self.p, self.world
        dT = np.zeros(g.size); dS = np.zeros(g.size)
        if not p['basin_to_continent']:
            return dT, dS
        z = self.elevation()
        conv = (~w['cont']) & ((w['sed'] > p['basin_to_continent_sed_m']) | ((w['sed'] > 500.) & (z > 0)))
        if not conv.any():
            return dT, dS
        rm, rc, rw = p['rho_mantle'], p['rho_crust'], p['rho_water']
        zr = z-self.dyn
        dry = np.where(zr > 0, zr, zr*(rm-rw)/rm)
        T_new = p['cont_ref_thick_m']+(dry-p['cont_ref_elev_m'])*rm/(rm-rc)
        T_new = np.maximum(T_new, w['thick'])
        dT[conv] = T_new[conv]-w['thick'][conv]
        dS[conv] = -w['sed'][conv]
        a = g.area
        self.erosion_totals['basin_to_continent_m3'] = self.erosion_totals.get('basin_to_continent_m3', 0.)+float((w['sed'][conv]*a[conv]).sum())
        w['cont'] = w['cont'] | conv
        w['thick'] = w['thick']+dT
        w['sed'] = w['sed']+dS
        self.new_continent = self.new_continent | conv
        return dT, dS

    def _subduction_thickening(self):
        """Thickening (m over this step) of the overriding plate near trenches, from the previous step's boundaries."""
        g, p, w = self.g, self.p, self.world
        out = np.zeros(g.size)
        self.belt = np.zeros(g.size, bool)
        if not hasattr(self, 'btype'):
            return out
        src = (self.btype == SUBDUCTION) & ~self.trench          # overriding side of a subduction boundary
        if not src.any():
            return out
        lab = np.where(src, np.arange(g.size), -1).astype(np.int64)
        near, d = nearest_label(lab, g.nbr, g.dist)
        ok = (near >= 0) & (d < 1000e3)
        ok[ok] &= self.owner[ok] == self.owner[near[ok]]           # stay on the overriding plate
        v = np.clip(self.conv_rate[np.clip(near, 0, None)], 0, None)  # m/yr closing
        dt = p['dt_myr']*1e6
        cont = w['cont']
        shorten = p['margin_shortening']
        H_cont = p['arc_H_cont_m'] if shorten else p['subduction_H_cont_m']
        for mask, H, d0, sg in ((ok & cont, H_cont, p['subduction_d0_cont_km']*1e3, p['subduction_sigma_cont_km']*1e3),
                                (ok & ~cont, p['arc_H_ocean_m'], p['arc_d0_km']*1e3, p['arc_sigma_km']*1e3),
                                (ok, -p['subduction_erosion_H_m'], p['subduction_erosion_d0_km']*1e3, p['subduction_erosion_sigma_km']*1e3)):
            if H == 0:
                continue
            shape = np.exp(-.5*((d-d0)/sg)**2)/(sg*np.sqrt(2*np.pi))
            out = out+np.where(mask, v*H*shape*dt, 0.)
        # erosion takes at most 30% of a continental column per step; on oceanic crust it removes only arc crust built
        # above the normal 7 km (an intra-oceanic forearc), never the oceanic basement: unbounded, it dug pits to
        # -10 km that trailed behind moving trenches (step 7 run)
        floor = np.where(cont, -.3*w['thick'], -np.clip(w['thick']-p['ocean_thick_m'], 0, None))
        out = np.maximum(out, floor)
        # the subduction belt on continental crust (orogen_rule 'belt'): within 2 sigma of d0, trench converging
        d0c, sgc = p['subduction_d0_cont_km']*1e3, p['subduction_sigma_cont_km']*1e3
        self.belt = ok & cont & (np.abs(d-d0c) <= 2*sgc) & (v > 0)
        for key, m in (('subduction_on_continents_m3', cont), ('subduction_on_ocean_arcs_m3', ~cont)):
            self.erosion_totals[key] = self.erosion_totals.get(key, 0.)+float((out*g.area)[m].sum())
        self.erosion_totals['subduction_added_m3'] = self.erosion_totals.get('subduction_added_m3', 0.)+float((out*g.area).sum())
        if shorten:
            s_rate = self._shortening_rate(src)                     # m/yr per overriding trench node
            moved = margin_shortening(near, d, ok & cont, s_rate[np.clip(near, 0, None)], w['thick']+out, g.area, dt,
                                      p['subduction_d0_cont_km']*1e3, p['subduction_sigma_cont_km']*1e3,
                                      p['foreland_d0_km']*1e3, p['foreland_sigma_km']*1e3)
            out = out+moved
            self.erosion_totals['margin_shortening_moved_m3'] = self.erosion_totals.get('margin_shortening_moved_m3', 0.)+float(
                (np.maximum(moved, 0)*g.area).sum())
        return out

    def _orogen_marks(self):
        """World nodes (continental crust) besides collision contacts whose orogeny clock is reset this step
        (orogen_rule; collision contacts are always marked)."""
        g, p, w = self.g, self.p, self.world
        rule = p['orogen_rule']
        none = np.zeros(g.size, bool)
        if rule == 'collision':
            return none
        cont = w['cont']
        if rule == 'belt':
            return cont & (getattr(self, 'belt', none) | (w['thick'] > p['collapse_thick_m']))
        if rule == 'strain':
            if not hasattr(self, 'tectonic'):
                return none
            rate = np.abs(self.tectonic)/np.maximum(w['thick'], 1e3)/(p['dt_myr']*1e6)
            return cont & (rate > p['orogen_strain_rate'])
        raise ValueError(f"orogen_rule {rule!r}: 'collision', 'belt' or 'strain'")

    def _shortening_rate(self, src):
        """Upper-plate shortening rate (m/yr) at every overriding trench node: max(0, u - rollback), u the overriding
        plate's speed toward the trench in the mantle (world) frame (Oncken et al. 2006). The trench direction is the
        mean direction to the node's neighbours on other plates."""
        g, p = self.g, self.p
        out = np.zeros(g.size)
        idx = np.flatnonzero(src)
        nb = g.nbr[idx]
        other = (nb >= 0) & (self.owner[np.maximum(nb, 0)] != self.owner[idx][:, None])
        x = g.xyz[idx]
        disp = np.where(other[..., None], g.xyz[np.maximum(nb, 0)]-x[:, None], 0.).sum(1)
        disp -= np.einsum('ij,ij->i', disp, x)[:, None]*x
        n = disp/np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-15)
        om = {pl.id: pl.omega(g.radius) for pl in self.plates}
        w = np.stack([om.get(int(k), np.zeros(3)) for k in self.owner[idx]])
        u = np.einsum('ij,ij->i', np.cross(w, x)*g.radius, n)             # m/yr toward the trench
        out[idx] = np.clip(u-p['rollback_mm_yr']*1e-3, 0, None)*other.any(1)
        return out

    def _erode(self, thick):
        """Coarse erosion over one plate step: the landscape model's surface processes (graph_landscape.surface_step)
        on the flexural-isostatic elevation, in erosion_substeps steps. Returns the change of crust thickness
        (continental: erosion < 0, and without continental_sediment sediment deposited on continental crust > 0;
        oceanic, with ocean_crust_erosion: erosion below the sediment, <= 0) and of sediment (ocean floor, and with
        continental_sediment continental crust too: deposits > 0, erosion of the sediment first). Their sum times area
        is 0 (surface_step is closed)."""
        g, p, w = self.g, self.p, self.world
        z0 = self.elevation(thick, w['sed'])
        lp = {**gl.DEFAULTS, 'K': gl.DEFAULTS['K']*p['erosion_K_factor'], 'Sc': p['erosion_Sc'],
              'Kd': gl.DEFAULTS['Kd']*p['erosion_Kd_factor'], 'open_edges': False, 'theta': p['erosion_theta']}
        since = np.where(np.isnan(w['orog']), np.inf, self.t-w['orog'])
        erod = p['stable_erodibility']+(1-p['stable_erodibility'])*np.exp(-np.clip(since, 0, None)/p['orogen_erodibility_tau_myr'])
        K = lp['K']*np.where(w['cont'], erod, 1.)
        seed = z0 < -1500.
        dt = p['dt_myr']*1e6/p['erosion_substeps']
        z = z0.copy()
        fallback = np.ones(g.size, bool)
        shelf_on = p['shelf_kappa_m2_yr'] > 0
        keep = p['keep_deep_sediment']
        deep = np.zeros(g.size)
        for _ in range(p['erosion_substeps']):
            r = gl.surface_step(g, z, seed, None, fallback, K, lp, dt, self.h_min, self.bound, keep_deep=keep)
            z = r['z']
            if keep and r['deep_m3'].any():          # turbidites: fill the sea-floor lows from the bottom up
                deep += r['deep_m3']
                z = gl.fill_and_spill(z, r['deep_m3'], g.nbr, g.area)
            if shelf_on:                             # wave planation and coastal retreat (planet_layer.shelf)
                ocean = gl.connected_ocean(z, seed, g.nbr)
                avail = np.where(w['cont'], 1e9, np.clip(w['sed']+z-z0, 0, None))
                z, _, off = shelf_mod.shelf_step(g, z, ocean, avail, p['shelf_kappa_m2_yr'], p['shelf_wave_base_m'],
                                                 p['cliff_retreat_m_yr'], dt)
                self.erosion_totals['shelf_off_land_m3'] = self.erosion_totals.get('shelf_off_land_m3', 0.)+off
        self.erosion_totals['deep_sea_sediment_m3'] = self.erosion_totals.get('deep_sea_sediment_m3', 0.)+float(deep.sum())
        dz = z-z0
        cont = w['cont']
        a = g.area
        # surface change on continental nodes (sediment and crust together): the totals keep their meaning
        dz_cont = np.where(cont, dz, 0.)
        self.erosion_totals['continental_eroded_m3'] += float((np.minimum(dz_cont, 0)*a).sum())
        self.erosion_totals['continental_deposited_m3'] += float((np.maximum(dz_cont, 0)*a).sum())
        # the part deposited on continental crust under the sea (shelves, drowned margins): delivered to the sea in
        # Earth's sense (river delivery, 4-8 km^3/yr), so scripts.sweep_plates counts it as export
        self.erosion_totals['continental_deposited_submerged_m3'] = self.erosion_totals.get(
            'continental_deposited_submerged_m3', 0.)+float((np.maximum(dz_cont, 0)*a)[z0 <= 0].sum())
        if p['continental_sediment']:                # deposits are sediment everywhere; erosion takes sediment first
            dsed = np.maximum(dz, -w['sed'])
            e_cont = np.where(cont, dz-dsed, 0.)
            self.erosion_totals['continental_sediment_m3'] = self.erosion_totals.get('continental_sediment_m3', 0.)+float(
                (dsed*a)[cont].sum())
            dsed_oc = np.where(cont, 0., dsed)
        else:                                        # deposits on continental crust become crust
            e_cont = dz_cont
            dsed = dsed_oc = np.where(cont, 0., np.maximum(dz, -w['sed']))
        self.erosion_totals['continental_crust_eroded_m3'] = self.erosion_totals.get('continental_crust_eroded_m3', 0.)+float(
            (np.minimum(e_cont, 0)*a).sum())
        self.erosion_totals['ocean_sediment_m3'] += float((dsed_oc*a).sum())
        if p['ocean_crust_erosion']:                 # cut below the sediment: from the oceanic crust (closes the ledger)
            e_oc = np.where(cont, 0., np.maximum(np.minimum(dz+w['sed'], 0.), -thick))
            self.erosion_totals['ocean_crust_eroded_m3'] = self.erosion_totals.get('ocean_crust_eroded_m3', 0.)+float(
                (e_oc*a).sum())
            self.erosion_totals['ocean_crust_eroded_subaerial_m3'] = self.erosion_totals.get(
                'ocean_crust_eroded_subaerial_m3', 0.)+float((e_oc*a)[z0 > 0].sum())
            e_cont = e_cont+e_oc
        return e_cont, dsed

    def elevation(self, thick=None, sed=None):
        """Elevation (m) with flexural isostasy on an elastic plate (Te).

        Continental crust: its thickness anomaly (relative to cont_ref_thick_m, which stands at cont_ref_elev_m) is a
        load of density rho_crust against the mantle. Ocean floor: thermal subsidence with age (GDH1); its crust
        anomaly (arcs, plateaus) and sediment are submarine loads of density (rho - rho_water) against
        (rho_mantle - rho_water), so a basin filled to sea level holds ~2.5x its depth of sediment, as on Earth.
        Long wavelengths reduce to Airy isostasy; valleys and single belts are partly held up by the plate.
        A thinned continental column below sea level is water-loaded (its own Airy depth only, not its neighbours').
        With thinning_local_isostasy, continental crust thinner than cont_ref_thick_m stands at that local Airy depth
        (plus the deflection from other loads) and is no negative load on the plate. With continental_sediment, the
        sediment on continental crust adds its thickness and loads the plate (rho_sediment; against water where its
        top is below sea level).
        """
        p, w = self.p, self.world
        thick = w['thick'] if thick is None else thick
        sed = w['sed'] if sed is None else sed
        cont = w['cont']
        rc, rm, rw, rs = p['rho_crust'], p['rho_mantle'], p['rho_water'], p['rho_sediment']
        anomaly = np.where(cont, thick-p['cont_ref_thick_m'], thick-p['ocean_thick_m'])
        ocean_age = np.where(cont, 0., self.t-w['born'])
        s0 = getattr(self, 'sea_level', 0.)                     # previous sea level on the datum (loads use it)
        above = (-gdh1_depth(ocean_age)+anomaly+sed) > s0       # oceanic crust built above sea level (arcs, fills)
        # thinning_local_isostasy: the thinned part (negative anomaly) of continental crust stands at its local Airy
        # level and is no load on the plate; the thickened part loads it as before
        thin = np.where(cont, np.minimum(anomaly, 0.), 0.) if p['thinning_local_isostasy'] else 0.
        load = anomaly-thin
        own_airy = p['cont_ref_elev_m']+anomaly*(rm-rc)/rm-s0
        # continental_sediment: sediment on continental crust is a load like ocean-floor sediment, dry where its top
        # (on the crust's own water-loaded Airy level, the ocean rule's counterpart) stands above sea level
        cs = np.where(cont, sed, 0.) if p['continental_sediment'] else np.zeros_like(anomaly)
        c_dry = cont & (np.where(own_airy < 0, own_airy*rm/(rm-rw), own_airy)+cs > 0)
        dry = np.where(cont, rc*load+np.where(c_dry, rs*cs, 0.), np.where(above, rc*anomaly+rs*sed, 0.))
        wet = np.where(cont, np.where(c_dry, 0., (rs-rw)*cs), np.where(above, 0., (rc-rw)*anomaly+(rs-rw)*sed))
        defl = self.flex_dry.deflection(dry)+self.flex_wet.deflection(wet)
        z = np.where(cont, p['cont_ref_elev_m']+load+thin*(rm-rc)/rm+cs, -gdh1_depth(ocean_age)+anomaly+sed)-defl
        z = z+np.where(cont & (own_airy < 0), own_airy*rw/(rm-rw), 0.)
        z = z+self.dyn                                    # dynamic topography: non-load surface motion
        if p['water_volume_km3'] is None:
            return z
        self.sea_level = sea_level_for_volume(z, self.g.area, p['water_volume_km3']*1e9)
        return z-self.sea_level

    def _update_rasters(self, dthick, idx, dsed=None):
        """Update every plate raster after a step.

        Ownership: forward lookup (plate node j -> world node). Overlapped crust stays hidden for hidden_steps before
        it is consumed (subducted ocean, or continental crust whose volume is added to the overriding continent:
        collision). Thickness changes go to the plate nodes the world actually reads (backward samples), so every
        change feeds back: diffusion increments (per area) are averaged over the world nodes reading a plate node;
        collision volume is added as volume. Plate nodes the world does not read keep their values."""
        g, w, p = self.g, self.world, self.p
        a = g.area
        wns, owns, consumed = [], [], []
        dVc = np.zeros(g.size)                          # collision volume arriving at world nodes (m^3)
        collided = np.zeros(g.size, bool)
        subducted = np.zeros(g.size, bool)
        for k, pl in enumerate(self.plates):            # pass 1: hidden / consumed crust
            wn = fields.nearest_nodes(g, g.xyz@pl.Q.T)
            own = self.owner_slot[wn] == k
            hidden = pl.mask & ~own
            pl.under[own] = 0
            pl.under[hidden] += 1
            gone = hidden & (pl.under > p['hidden_steps'])
            cc = gone & pl.cont & w['cont'][wn]
            np.add.at(dVc, wn[cc], pl.thick[cc].astype(np.float64)*a[cc])
            collided[wn[cc]] = True
            SL = self.sed_ledger
            if p['continental_sediment']:                # its sediment is folded into the receiving belt's crust
                np.add.at(dVc, wn[cc], pl.sed[cc].astype(np.float64)*a[cc])
                self.erosion_totals['collision_sediment_to_crust_m3'] = self.erosion_totals.get(
                    'collision_sediment_to_crust_m3', 0.)+float((pl.sed[cc]*a[cc]).sum())
            SL['collision_out'] -= float((pl.sed[cc]*a[cc]).sum())
            SL['cont_subducted'] -= float((pl.sed[gone & pl.cont & ~w['cont'][wn]]*a[gone & pl.cont & ~w['cont'][wn]]).sum())
            subducted[wn[gone & ~pl.cont]] = True
            sg = gone & ~pl.cont & (pl.sed > 0)          # trench sediment: part is scraped onto the overriding plate
            np.add.at(dVc, wn[sg], p['sediment_accretion']*pl.sed[sg].astype(np.float64)*a[sg])
            self.erosion_totals['sediment_accreted_m3'] = self.erosion_totals.get('sediment_accreted_m3', 0.)+float(
                (p['sediment_accretion']*pl.sed[sg]*a[sg]).sum())
            self.erosion_totals['sediment_subducted_m3'] = self.erosion_totals.get('sediment_subducted_m3', 0.)+float(
                ((1-p['sediment_accretion'])*pl.sed[sg]*a[sg]).sum())
            L = self.ledger
            L['collision_out'] -= float((pl.thick[cc]*a[cc]).sum())
            lost = gone & pl.cont & ~w['cont'][wn]
            self.area_ledger['collision'] -= float(a[cc].sum())
            self.area_ledger['subduction'] -= float(a[lost].sum())
            L['cont_subducted'] -= float((pl.thick[lost]*a[lost]).sum())
            wns.append(wn); owns.append(own); consumed.append(gone)
        self.collided, self.subducted = collided, subducted
        if hasattr(self, 'tectonic'):
            self.tectonic = self.tectonic+dVc/a          # collision volume, as thickness at the receiving world node
        marked = collided | self._orogen_marks()
        self.orogen_marked = marked
        w['orog'] = np.where(marked, self.t, w['orog'])
        # the thickening clock (E2): nodes thickened tectonically (collision volume, margin shortening, arc and
        # subduction thickening; net of subduction erosion) faster than orogen_strain_rate this step
        flow = getattr(self, 'thk_flow', None)           # E3: times the collapse flow carried this step
        self.thk_flow = None
        self.thk_carried = None
        if flow is not None:
            moved = np.isfinite(flow) & ~(w['thk'] >= flow)
            w['thk'] = np.where(moved, flow, w['thk'])
            self.thk_carried = np.where(moved, flow, np.nan)
        th_src = getattr(self, 'thickening', np.zeros(g.size))+dVc/a
        self.thk_marked = th_src/np.maximum(w['thick'], 1e3)/(p['dt_myr']*1e6) > p['orogen_strain_rate']
        w['thk'] = np.where(self.thk_marked, self.t, w['thk'])
        for k, pl in enumerate(self.plates):            # pass 2: ownership, new crust, thickness changes
            wn, own, gone = wns[k], owns[k], consumed[k]
            # a plate node switches on for genuinely new crust (a ridge gap), or for oceanic crust the plate already
            # covers through a neighbouring node (filling the staircase behind a spreading ridge). Continental crust
            # is never duplicated that way (the forward and backward nearest-node maps are not exact inverses), so
            # the continental inventory is conserved.
            new = own & ~pl.mask & (self.new_crust[wn] | ~w['cont'][wn])
            ws = np.flatnonzero((self.owner_slot == k) & ~self.new_crust)   # world nodes read from this plate
            js = idx[k, ws]
            nread = np.bincount(js, minlength=g.size)
            read = nread > 0
            before_v = float((pl.thick*a*(pl.mask & pl.cont)).sum())
            vs = self._plate_cont_sed(pl)
            d_int = np.bincount(js, weights=dthick[ws], minlength=g.size)/np.maximum(nread, 1)
            d_vol = np.bincount(js, weights=dVc[ws], minlength=g.size)/a
            pl.thick[read] += (d_int[read]+d_vol[read]).astype(np.float32)
            if dsed is not None:
                d_sed = np.bincount(js, weights=dsed[ws], minlength=g.size)/np.maximum(nread, 1)
                pl.sed[read] = np.maximum(pl.sed[read]+d_sed[read], 0).astype(np.float32)
            self.ledger['increments'] += float((pl.thick*a*(pl.mask & pl.cont)).sum())-before_v
            vs1 = self._plate_cont_sed(pl)
            self.sed_ledger['increments'] += vs1-vs
            conv = self.new_continent[ws] if hasattr(self, 'new_continent') else np.zeros(ws.size, bool)
            if conv.any():
                pl.cont[js[conv]] = True
                pl.orog[js[conv]] = self.t
                self.ledger['arc_continent'] = self.ledger.get('arc_continent', 0.)+float((pl.thick[js[conv]]*a[js[conv]]).sum())
                self.sed_ledger['arc_continent'] += self._plate_cont_sed(pl)-vs1
            hit = np.zeros(g.size, bool); hit[js[marked[ws]]] = True
            pl.orog[hit] = self.t
            hit = np.zeros(g.size, bool); hit[js[self.ext_marked[ws]]] = True
            pl.ext[hit] = self.t
            if self.thk_carried is not None:             # E3: carried times first, tectonic marking after
                cv = self.thk_carried[ws]; f = np.isfinite(cv)
                if f.any():
                    best = np.full(g.size, -np.inf); np.maximum.at(best, js[f], cv[f])
                    up = np.isfinite(best) & ~(pl.thk >= best)
                    pl.thk[up] = best[up]
            hit = np.zeros(g.size, bool); hit[js[self.thk_marked[ws]]] = True
            pl.thk[hit] = self.t
            pl.cont[new] = w['cont'][wn[new]]
            pl.thick[new] = w['thick'][wn[new]]
            pl.born[new] = w['born'][wn[new]]
            pl.orog[new] = w['orog'][wn[new]]
            pl.ext[new] = w['ext'][wn[new]]
            pl.thk[new] = w['thk'][wn[new]]
            pl.sed[new] = w['sed'][wn[new]]
            pl.under[new] = 0
            self.ledger['new_nodes'] += float((pl.thick[new]*a[new]*pl.cont[new]).sum())
            self.sed_ledger['new_nodes'] += float((pl.sed[new]*a[new]*pl.cont[new]).sum())
            pl.mask[gone] = False
            pl.mask[new] = True
            pl.cont[~pl.mask] = False
            hole, src = plate_holes(pl.mask, g.nbr)
            if hole.any():
                hs = src[hole]
                pl.mask[hole] = True
                for arr in (pl.cont, pl.thick, pl.born, pl.orog, pl.sed, pl.ext, pl.thk):
                    arr[hole] = arr[hs]
                self.ledger['holes'] += float((pl.thick[hole]*a[hole]*pl.cont[hole]).sum())
                self.area_ledger['holes'] = self.area_ledger.get('holes', 0.)+float(a[hole & pl.cont].sum())
                self.sed_ledger['holes'] += float((pl.sed[hole]*a[hole]*pl.cont[hole]).sum())
            self.holes_filled = getattr(self, 'holes_filled', 0)+int(hole.sum())
        # collision volume that reached a world node no plate node reads is lost (counted)
        self.ledger['cont_lost_other'] -= float(dVc[self.new_crust].sum())

    def _classify(self):
        g = self.g
        K = len(self.plates)
        pid = np.array([pl.id for pl in self.plates])
        slot = np.zeros(pid.max()+1, np.int64); slot[pid] = np.arange(K)
        omega = np.stack([pl.omega(g.radius) for pl in self.plates])
        share = np.array([(g.area*(pl.mask & pl.cont)).sum() for pl in self.plates])/g.area.sum()
        age = self.t-self.world['born']
        score = self._score(self.world['cont'], age, share[slot[self.owner]])
        # owner may refer to plates removed this step only if they own nothing, so slot lookup is safe
        bt, rate, trench = classify_boundaries(self.owner, slot, omega, g.xyz, g.nbr, score, self.world['cont'], g.radius,
                                               self.p['boundary_min_mm_yr']*1e-3)
        self.btype, self.conv_rate, self.trench = bt, rate, trench

    def _driving(self, own, speed, axis):
        """The usual forces on the plate made of the world nodes own, one step from its speed and axis: the speed
        relaxes toward the target set by slab pull, ridge push, continental drag and collision resistance, and the pole
        turns so the plate moves toward its trenches (slab pull) and away from its ridges (ridge push). Returns the new
        speed and axis."""
        g, p = self.g, self.p
        dt = p['dt_myr']
        h = np.sqrt(g.area.mean())
        c = (g.area[own, None]*g.xyz[own]).sum(0); c /= np.linalg.norm(c)
        tr = own & self.trench
        ri = own & (self.btype == RIDGE)
        coll = own & (self.btype == COLLISION)
        L_slab = tr.sum()*h/2e3                          # boundary nodes sit on both sides of a line: halve
        L_ridge = ri.sum()*h/2e3
        L_coll = coll.sum()*h/2e3
        cf = (g.area*(own & self.world['cont'])).sum()/(g.area*own).sum()
        target = (p['v_min_cm_yr']+(p['v_slab_cm_yr']-p['v_min_cm_yr'])*min(1., L_slab/p['slab_ref_km'])
                  + p['v_ridge_cm_yr']*min(1., L_ridge/p['ridge_ref_km']))
        target *= (1-p['cont_drag']*cf)/(1+L_coll/p['collision_ref_km'])
        speed = speed+(target-speed)*min(1., dt/p['tau_speed_myr'])
        pull = np.cross(c, g.xyz[tr]).sum(0)+p['ridge_torque']*np.cross(g.xyz[ri], c).sum(0)
        nrm = np.linalg.norm(pull)
        if nrm > 0:
            b = min(1., dt/p['tau_axis_myr'])
            a = (1-b)*axis+b*pull/nrm
            axis = a/np.linalg.norm(a)
        return speed, axis

    def _update_motion(self):
        for pl in self.plates:
            own = self.owner == pl.id
            if not own.any():
                continue
            pl.speed, pl.axis = self._driving(own, pl.speed, pl.axis)

    # ------------------------------------------------------------------ plate splitting
    def _omega_vec(self, pl):
        return pl.axis*pl.speed

    def _set_omega(self, pl, v):
        sp = float(np.linalg.norm(v))
        pl.speed = max(sp, 1e-3)
        if sp > 1e-9:
            pl.axis = v/sp

    def split(self, pl, part_world, rel_speed_cm_yr, kind, pole=None):
        """Split plate `pl`: the world nodes in part_world (owned by pl) become a new plate in the same frame (no
        resampling). The two parts get Euler vectors differing by rel_speed (positive: diverge, negative: converge),
        about the pole 90 degrees from both parts' centroids, or about `pole` (E3; signed: the new part moves as a
        rotation about +pole relative to the rest)."""
        g = self.g
        own = self.owner == pl.id
        part = part_world & own
        rest = own & ~part
        if part.sum() < 4 or rest.sum() < 4:
            return None
        wn = fields.nearest_nodes(g, g.xyz@pl.Q.T)
        lab = part[wn] & pl.mask
        new = self._new_plate(pl.axis, pl.speed)
        new.Q = pl.Q.copy()
        for name in ('mask', 'cont', 'thick', 'born', 'orog', 'under', 'sed', 'ext', 'thk'):
            arr = getattr(pl, name)
            na = getattr(new, name)
            na[lab] = arr[lab]
        pl.mask[lab] = False; pl.cont[lab] = False
        c1 = (g.area[rest, None]*g.xyz[rest]).sum(0); c1 /= np.linalg.norm(c1)
        c2 = (g.area[part, None]*g.xyz[part]).sum(0); c2 /= np.linalg.norm(c2)
        d = np.cross(c1, c2) if pole is None else np.asarray(pole, np.float64)
        if np.linalg.norm(d) < 1e-9:
            d = self.rng.normal(size=3)
        d = d/np.linalg.norm(d)*rel_speed_cm_yr
        base = self._omega_vec(pl)
        self._set_omega(pl, base-d/2)
        self._set_omega(new, base+d/2)
        self.owner = np.where(part, new.id, self.owner)
        self.events.append({'t': self.t, 'kind': kind, 'plate': int(pl.id), 'new_plate': int(new.id),
                            'area_mkm2': float(g.area[part].sum()/1e12)})
        return new

    def _merge_events(self):
        g, p = self.g, self.p
        coll = np.flatnonzero(self.btype == COLLISION)
        contact = {}
        if coll.size:
            nb = g.nbr[coll]
            ok = nb >= 0
            a = np.repeat(self.owner[coll][:, None], nb.shape[1], 1)
            b = self.owner[np.where(ok, nb, 0)]
            m = ok & (a != b) & (self.btype[np.where(ok, nb, 0)] == COLLISION)
            pairs, counts = np.unique(np.stack((np.minimum(a[m], b[m]), np.maximum(a[m], b[m])), 1), axis=0, return_counts=True)
            h = np.sqrt(g.area.mean())/1e3
            for (x, y), c in zip(pairs, counts):
                contact[(int(x), int(y))] = c*h/2          # each contact appears from both sides
        prev = getattr(self, 'contact_myr', {})
        self.contact_myr = {k: prev.get(k, 0.)+p['dt_myr'] for k, L in contact.items() if L >= p['merge_contact_km']}
        for (x, y), myr in list(self.contact_myr.items()):
            if myr < p['merge_after_myr']:
                continue
            pa = [q for q in self.plates if q.id == x]; pb = [q for q in self.plates if q.id == y]
            if not pa or not pb:
                continue
            ax, ay = g.area[self.owner == x].sum(), g.area[self.owner == y].sum()
            big, small = (pa[0], pb[0]) if ax >= ay else (pb[0], pa[0])
            if min(ax, ay) > p['merge_max_ratio']*max(ax, ay) or (ax+ay)/1e12 > p['merge_max_area_mkm2']:
                continue
            self.merge(big, small)
            self.contact_myr = {k: v for k, v in self.contact_myr.items() if small.id not in k}

    def merge(self, big, small):
        """Weld `small` onto `big`. Every node of big's raster whose world position lies in small's territory reads
        small's crust there (backward lookup, so the region is filled without holes); where big already holds crust,
        small's continental volume is added to big's thickness. Euler vectors are averaged by area."""
        g = self.g
        v0 = self.continental_volume()
        vs0 = self.continental_sediment_volume()
        wk = fields.nearest_nodes(g, g.xyz@big.Q.T)                  # big node -> world node
        take = self.owner[wk] == small.id
        k = np.flatnonzero(take)
        j = fields.nearest_nodes(g, (g.xyz[k]@big.Q.T)@small.Q)      # -> small's raster
        ok = small.mask[j]
        k, j = k[ok], j[ok]
        free = ~big.mask[k]
        for name in ('cont', 'thick', 'born', 'orog', 'sed', 'ext', 'thk'):
            getattr(big, name)[k[free]] = getattr(small, name)[j[free]]
        big.under[k[free]] = 0
        big.mask[k[free]] = True
        # the old boundary is now inside one plate: every big node whose world position the welded plate owns must
        # hold crust, else later sampling finds cracks along the seam; fill them from the nearest set node
        need = ((self.owner[wk] == big.id) | take) & ~big.mask
        if need.any():
            src_lab = np.where(big.mask, np.arange(g.size), -1).astype(np.int64)
            near, _ = nearest_label(src_lab, g.nbr, g.dist)
            src = near[need]
            big.mask[need] = True
            for name in ('cont', 'thick', 'born', 'orog', 'sed', 'ext', 'thk'):
                arr = getattr(big, name); arr[need] = arr[src]
        both = ~free & small.cont[j] & big.cont[k]
        self.area_ledger['merge'] = self.area_ledger.get('merge', 0.)-float(g.area[k[both]].sum())
        np.add.at(big.thick, k[both], small.thick[j[both]])
        if self.p['continental_sediment']:
            np.add.at(big.sed, k[both], small.sed[j[both]])
        a_big, a_small = g.area[self.owner == big.id].sum(), g.area[self.owner == small.id].sum()
        self._set_omega(big, (self._omega_vec(big)*a_big+self._omega_vec(small)*a_small)/(a_big+a_small))
        self.owner = np.where(self.owner == small.id, big.id, self.owner)
        self.plates = [q for q in self.plates if q is not small]
        self.ledger['merge'] = self.ledger.get('merge', 0.)+self.continental_volume()-v0
        self.sed_ledger['merge'] += self.continental_sediment_volume()-vs0
        self.events.append({'t': self.t, 'kind': 'merge', 'plate': int(big.id), 'merged': int(small.id),
                            'area_mkm2': float(a_small/1e12)})

    # ------------------------------------------------------------------ extension of thick crust (E)
    def _pair(self, ev):
        a = [q for q in self.plates if q.id == ev['parent']]; b = [q for q in self.plates if q.id == ev['child']]
        return (a[0], b[0]) if a and b else (None, None)

    def _near_piece(self, ev):
        """World nodes within the event's half-width of its thick piece (tracked as raster nodes of both halves)."""
        g = self.g
        src = np.zeros(g.size, bool)
        for q in self.plates:
            r = ev['piece'].get(q.id)
            if r is not None and r.size:
                src[fields.nearest_nodes(g, g.xyz[r]@q.Q.T)] = True
        if not src.any():
            return src
        return gl.distance_to(src, g.nbr, g.dist, ev['W']*1.5) <= ev['W']

    def _extension_motion(self):
        """Running events: the two halves diverge at the event's rate (cm/yr at 90 degrees from the relative pole). The
        pole is the event's own (band cut, E3: carried in the parent's frame) or else lies 90 degrees from both halves'
        centroids, as in split. Their mean motion: with ext_pair_motion 'union' (E3) the usual forces on the pair as one
        plate (_driving on both halves, from the motion the plate had when it split); with 'mean' (E, E2 as built) the
        area-weighted mean of each half's own force result."""
        g = self.g
        for ev in self.ext_events:
            pa, pb = self._pair(ev)
            if pa is None:
                continue
            oa, ob = self.owner == pa.id, self.owner == pb.id
            if not oa.any() or not ob.any():
                continue
            if ev.get('pole_body') is not None:          # E3 band cut: the pole carried in the parent's frame
                d = np.asarray(ev['pole_body'])@pa.Q.T
            else:
                c1 = (g.area[oa, None]*g.xyz[oa]).sum(0); c1 /= np.linalg.norm(c1)
                c2 = (g.area[ob, None]*g.xyz[ob]).sum(0); c2 /= np.linalg.norm(c2)
                d = np.cross(c1, c2)
            if np.linalg.norm(d) < 1e-9:
                continue
            d = d/np.linalg.norm(d)*ev['v']
            if 'base_speed' in ev:                       # E3 'union': the forces on the pair as one plate
                ev['base_speed'], ev['base_axis'] = self._driving(oa | ob, ev['base_speed'], ev['base_axis'])
                base = ev['base_axis']*ev['base_speed']
            else:
                A1, A2 = g.area[oa].sum(), g.area[ob].sum()
                base = (self._omega_vec(pa)*A1+self._omega_vec(pb)*A2)/(A1+A2)
            self._set_omega(pa, base-d/2)
            self._set_omega(pb, base+d/2)

    def _extension_events(self):
        """End running events whose extended crust has thinned to ext_end_thick_m (weld the halves), then start
        events on thick pieces (see PARAMS 'extension', 'ext_held_rule', 'ext_hot_myr')."""
        g, p, w = self.g, self.p, self.world
        a = g.area
        for ev in list(self.ext_events):
            pa, pb = self._pair(ev)
            if pa is None:                               # a half was consumed or merged away: the event ends unwelded
                self.ext_events.remove(ev)
                self.events.append({'t': self.t, 'kind': 'extension lost', 'plate': int(ev['parent']), 'new_plate': int(ev['child'])})
                continue
            zone = w['cont'] & (w['ext'] >= ev['t0']) & np.isin(self.owner, (pa.id, pb.id))
            T = float(np.average(w['thick'][zone], weights=a[zone])) if zone.any() else np.inf
            ev['T_zone_m'] = T
            if T <= p['ext_end_thick_m'] or self.t-ev['t0'] >= p['ext_max_myr']:
                A1, A2 = a[self.owner == pa.id].sum(), a[self.owner == pb.id].sum()
                big, small = (pa, pb) if A1 >= A2 else (pb, pa)
                self.merge(big, small)
                self.ext_events.remove(ev)
                self.events.append({'t': self.t, 'kind': 'extension end', 'plate': int(big.id), 'merged': int(small.id),
                                    'myr': float(self.t-ev['t0']), 'zone_thick_km': T/1e3,
                                    'zone_area_mkm2': float(a[zone].sum()/1e12), 'guard': bool(T > p['ext_end_thick_m'])})
        busy = {i for ev in self.ext_events for i in (ev['parent'], ev['child'])}
        thick = w['cont'] & (w['thick'] > p['ext_onset_thick_m'])
        if not thick.any():
            return
        held = (self.btype == COLLISION) | getattr(self, 'belt', np.zeros(g.size, bool))
        nb = g.nbr; ok = nb >= 0
        held = held | np.any(np.where(ok, held[np.where(ok, nb, 0)], False), axis=1)
        lab = components(thick, nb)
        nlab = lab.max()+1
        area = np.bincount(lab[thick], weights=a[thick], minlength=nlab)
        heldp = np.bincount(lab[thick], weights=held[thick].astype(float), minlength=nlab) > 0
        rule = p['ext_held_rule']
        if rule not in ('direction', 'exempt'):
            raise ValueError(f"ext_held_rule {rule!r}: 'direction' or 'exempt'")
        hot = thick & (self.t-w['thk'] <= p['ext_hot_myr'])          # never thickened (NaN) is cold
        hotp = np.bincount(lab[thick], weights=(a*hot)[thick], minlength=nlab) > .5*area
        elig = (area >= p['ext_min_area_mkm2']*1e12) & (hotp if p['ext_hot_myr'] > 0 else True)
        if rule == 'exempt':
            elig &= ~heldp
        R = self.ext_reach
        R['gt_onset'] += float(area.sum()); R['eligible'] += float(area[elig].sum())
        R['hot_pieces'] += float(area[hotp].sum()); R['held_pieces'] += float(area[heldp].sum())
        R['hot_nodes'] += float(a[hot].sum())
        for c in np.argsort(-area):
            if area[c] < p['ext_min_area_mkm2']*1e12:
                break
            if not elig[c]:
                continue
            P_ = lab == c
            ids, cnt = np.unique(self.owner[P_], return_counts=True)
            pid = int(ids[np.argmax(cnt)])
            if pid in busy:
                continue
            pl = [q for q in self.plates if q.id == pid][0]
            ev = self._start_extension(pl, P_, across=bool(heldp[c]) and rule == 'direction')
            if ev is not None:
                busy |= {ev['parent'], ev['child']}

    def _start_extension(self, pl, piece, across=False):
        """Split pl along the thick piece's long axis (great circle through its centroid), or across it (along its
        short axis) for a piece convergence holds (E2, ext_held_rule 'direction'); the smaller side becomes the new
        plate. The zone's half-width is the piece's p90 distance from the cut."""
        g, p, w = self.g, self.p, self.world
        a = g.area[piece]; X = g.xyz[piece]
        c = (a[:, None]*X).sum(0); c /= np.linalg.norm(c)
        Tn = X-(X@c)[:, None]*c
        C = (a[:, None, None]*Tn[:, :, None]*Tn[:, None, :]).sum(0)
        u = np.linalg.eigh(C)[1][:, -2 if across else -1]   # ascending: [-1] long axis, [-2] short (tangent plane)
        u = u-(u@c)*c
        if np.linalg.norm(u) < 1e-9:
            return None
        u /= np.linalg.norm(u)
        m = np.cross(c, u); m /= np.linalg.norm(m)
        s = np.abs(np.arcsin(np.clip(X@m, -1, 1)))*g.radius
        o = np.argsort(s); cw = np.cumsum(a[o]); cw /= cw[-1]
        W = float(np.clip(s[o][np.searchsorted(cw, .9)], p['rift_width_km']*1e3, p['ext_max_half_width_km']*1e3))
        own = self.owner == pl.id
        pole = None; L = None
        if p['ext_cut'] == 'whole':
            side = own & (g.xyz@m > 0)
            part = side if g.area[side].sum() <= g.area[own & ~side].sum() else own & ~side
        elif p['ext_cut'] == 'band':                     # E3: a microplate between two transforms
            su = np.abs(np.arcsin(np.clip(X@u, -1, 1)))*g.radius
            o2 = np.argsort(su); cw2 = np.cumsum(a[o2]); cw2 /= cw2[-1]
            L = max(float(su[o2][np.searchsorted(cw2, .9)]), p['rift_width_km']*1e3)
            band = own & (np.abs(np.arcsin(np.clip(g.xyz@u, -1, 1)))*g.radius <= L)
            sides = []
            for sg in (1., -1.):
                cand = band & (sg*(g.xyz@m) > 0)
                lab = components(cand, g.nbr)
                keep = np.unique(lab[piece & cand]); keep = keep[keep >= 0]
                sides.append(cand & np.isin(lab, keep))
            A_ = [g.area[q].sum() if q.sum() >= 4 else np.inf for q in sides]
            k = int(A_[1] < A_[0])
            if not np.isfinite(A_[k]):
                return None
            part = sides[k]
            out = (1., -1.)[k]*m                         # the part's side of the cut
            pole = u if np.cross(u, c)@out > 0 else -u   # rotation about pole moves c into the part's side
        else:
            raise ValueError(f"ext_cut {p['ext_cut']!r}: 'band' or 'whole'")
        v = p['ext_strain_rate_per_yr']*2*W*100.          # cm/yr
        if p['ext_pair_motion'] not in ('union', 'mean'):
            raise ValueError(f"ext_pair_motion {p['ext_pair_motion']!r}: 'union' or 'mean'")
        base0 = (float(pl.speed), np.array(pl.axis, np.float64))   # the plate's motion before the cut
        new = self.split(pl, part, v, 'extension', pole=pole)
        if new is None:
            return None
        far = {}
        if pole is not None and hasattr(self, 'btype'):  # where the microplate meets other plates (far side, edges)
            nb = g.nbr; okn = nb >= 0
            on = self.owner[np.where(okn, nb, 0)]
            edge = part & np.any(okn & (on != pl.id) & (on != new.id), axis=1)
            far = {'far_nodes': int(edge.sum()),
                   'far_collision': float(np.mean(self.btype[edge] == COLLISION)) if edge.any() else float('nan')}
        track = {}
        for q in (pl, new):                              # the piece as raster nodes of each half (same frame)
            wn = fields.nearest_nodes(g, g.xyz@q.Q.T)
            track[q.id] = np.flatnonzero(q.mask & piece[wn] & (self.owner[wn] == q.id))
        hot = (self.t-w['thk'][piece] <= p['ext_hot_myr']).astype(float)
        ev = {'parent': int(pl.id), 'child': int(new.id), 't0': float(self.t), 'W': W, 'v': v, 'piece': track,
              'piece_mkm2': float(a.sum()/1e12), 'piece_thick_km': float(np.average(w['thick'][piece], weights=a)/1e3),
              'across': bool(across), 'hot_share': float(np.average(hot, weights=a)), 'cut_normal': m.tolist(),
              'cut': p['ext_cut'], 'L': L, 'pole': None if pole is None else pole.tolist(),
              'pole_body': None if pole is None else (pole@pl.Q).tolist(), **far}
        if p['ext_pair_motion'] == 'union':
            ev['base_speed'], ev['base_axis'] = base0
        self.ext_events.append(ev)
        self.events[-1].update({k: ev[k] for k in ('W', 'v', 'piece_mkm2', 'piece_thick_km', 'across', 'hot_share', 'cut', 'L')})
        self.events[-1].update(far)
        return ev

    def _split_events(self):
        g, p, rng, w = self.g, self.p, self.rng, self.world
        dt = p['dt_myr']
        A = g.area
        for pl in list(self.plates):
            own = self.owner == pl.id
            area = A[own].sum()/1e12
            if area < p['split_min_area_mkm2']:
                continue
            cf = A[own & w['cont']].sum()/A[own].sum()
            rate = p['split_rate_per_myr']*area/100.*(2. if cf > .5 else 1.)
            if rng.uniform() < 1-np.exp(-rate*dt):
                self._rift_plate(pl, own)
                continue
            old = own & ~w['cont'] & (self.t-w['born'] > p['init_age_myr'])
            old_area = A[old].sum()/1e12
            irate = p['init_rate_per_myr']*min(5., old_area/p['init_area_ref_mkm2'])
            if old.sum() >= 4 and rng.uniform() < 1-np.exp(-irate*dt):
                self._initiate_subduction(pl, own, old)

    def _rift_plate(self, pl, own):
        """Two seeds far apart inside the plate; the split follows a wiggled Voronoi line that prefers old sutures."""
        g, p, rng, w = self.g, self.p, self.rng, self.world
        ids = np.flatnonzero(own)
        a = ids[rng.integers(ids.size)]
        dist_a = np.arccos(np.clip(g.xyz[ids]@g.xyz[a], -1, 1))
        b = ids[rng.choice(np.argsort(dist_a)[-max(1, ids.size//10):])]
        weak = np.where(np.isnan(w['orog']), 0., p['suture_weakness'])
        wig = 1+.3*rng.standard_normal(g.size)
        cost = np.clip(.5*(wig[:, None]+wig[np.clip(g.nbr, 0, None)]), .3, None)*(1-.5*(weak[:, None]+weak[np.clip(g.nbr, 0, None)]))
        cost = np.where(own[:, None] & own[np.clip(g.nbr, 0, None)], g.dist*cost, 1e30)
        lab0 = np.full(g.size, -1, np.int64); lab0[a] = 0; lab0[b] = 1
        lab, _ = nearest_label(lab0, g.nbr, cost)
        self.split(pl, lab == 1, p['split_speed_cm_yr'], 'rift')

    def _initiate_subduction(self, pl, own, old):
        """Old ocean breaks away from the rest of its plate and converges on it."""
        g, p, w = self.g, self.p, self.world
        ocean = own & ~w['cont']
        lab = components(ocean, g.nbr)
        target = np.bincount(lab[old & (lab >= 0)]).argmax()
        part = lab == target
        if (own & ~part).sum() == 0:                     # all-ocean plate: split through the old region instead
            c = (g.area[old, None]*g.xyz[old]).sum(0); c /= np.linalg.norm(c)
            n = np.cross(c, self.rng.normal(size=3)); n /= np.linalg.norm(n)
            part = own & (g.xyz@n > 0)
        self.split(pl, part, -p['init_speed_cm_yr'], 'subduction initiation')

    def run(self, years_myr, callback=None, every=None):
        steps = int(round(years_myr/self.p['dt_myr']))
        for s in range(steps):
            self.step()
            if callback is not None and every and (s+1) % every == 0:
                callback(self)
        return self

    # ------------------------------------------------------------------ outputs
    def state(self):
        """World fields at the current time (node vectors)."""
        w = self.world
        ocean_age = np.where(w['cont'], np.nan, self.t-w['born'])
        elev = self.elevation()
        return {'time_myr': self.t, 'plate_id': self.owner.astype(np.int32), 'continental': w['cont'],
                'crust_thickness_m': w['thick'].astype(np.float32), 'crust_age_myr': (self.t-w['born']).astype(np.float32),
                'ocean_age_myr': ocean_age.astype(np.float32), 'orogeny_age_myr': (self.t-w['orog']).astype(np.float32),
                'boundary_type': self.btype if hasattr(self, 'btype') else np.zeros(self.g.size, np.int8),
                'convergence_mm_yr': (self.conv_rate*1e3).astype(np.float32) if hasattr(self, 'conv_rate') else np.zeros(self.g.size, np.float32),
                'elevation_m': elev.astype(np.float32), 'sediment_m': w['sed'].astype(np.float32),
                'dynamic_topography_m': self.dyn.astype(np.float32), 'new_crust': self.new_crust,
                'sea_level_m': np.float32(getattr(self, 'sea_level', 0.)),
                'tectonic_thickening_m': getattr(self, 'tectonic', np.zeros(self.g.size)).astype(np.float32),
                'extension_age_myr': (self.t-w['ext']).astype(np.float32),
                'thickening_age_myr': (self.t-w['thk']).astype(np.float32)}

    def plate_table(self):
        g = self.g
        rows = []
        for pl in self.plates:
            own = self.owner == pl.id
            rows.append({'id': pl.id, 'area_km2': float(g.area[own].sum()/1e6), 'continental_km2': float(g.area[own & self.world['cont']].sum()/1e6),
                         'speed_cm_yr': pl.speed, 'axis': pl.axis.tolist(), 'Q': pl.Q.tolist()})
        return rows

    def continental_volume(self):
        """Continental crust volume held by the plates (m^3), including crust hidden under another plate."""
        a = self.g.area
        return float(sum((a*pl.thick*(pl.mask & pl.cont)).sum() for pl in self.plates))

    def _plate_cont_sed(self, pl):
        return float((self.g.area*pl.sed.astype(np.float64)*(pl.mask & pl.cont)).sum())

    def continental_sediment_volume(self):
        """Sediment on continental crust held by the plates (m^3), including crust hidden under another plate. Without
        continental_sediment this is only ocean-floor sediment carried onto crust that became continental (arcs)."""
        return float(sum(self._plate_cont_sed(pl) for pl in self.plates))

    def pinholes(self):
        """New-crust nodes more than 2.5 cells from any plate boundary: holes inside a plate (should be ~0). Ridge crust
        is always next to the boundary it opened at."""
        g = self.g
        nb = np.where(g.nbr >= 0, g.nbr, np.arange(g.size)[:, None])
        edge = (self.owner[nb] != self.owner[:, None]).any(1)
        d = gl.distance_to(edge, g.nbr, g.dist, 3*np.sqrt(g.area.mean()))
        return int((self.new_crust & (d > 2.5*np.sqrt(g.area.mean()))).sum())
