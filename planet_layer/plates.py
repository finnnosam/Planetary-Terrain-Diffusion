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

PARAMS = {
    'dt_myr': 1.,
    'ocean_thick_m': 7000.,
    'cont_ref_thick_m': 38000.,    # continental crust standing at cont_ref_elev_m
    'cont_ref_elev_m': 300.,
    'rho_crust': 2830., 'rho_mantle': 3300., 'rho_water': 1030., 'rho_sediment': 2400.,
    'te_m': 30e3,                  # elastic thickness for the elevation (flexural isostasy of crust and sediment)
    'threads': None,               # threads for the spherical-harmonic flexure (None: all cores)
    # coarse erosion: the landscape model's surface processes at the plate grid (calibrated: scripts/calibrate_coarse_erosion)
    'erosion': True, 'erosion_substeps': 4,
    'erosion_K_factor': 1., 'erosion_Sc': .05, 'erosion_Kd_factor': 4.,
    'erosion_theta': 8.,           # incision threshold (A^m S at 46 km), calibrated at n=217 against ETOPO (plan doc)
    'stable_erodibility': .15,     # old crust relative to fresh orogens (as the test bed)
    'orogen_erodibility_tau_myr': 50.,
    # subduction zones (overriding plate): thickening flux per unit trench length = convergence rate x H, spread as a
    # Gaussian band at distance d0 from the trench. Continental margins: shortening + arc magmatism (Andes ~3 km x v,
    # quieter margins much less); oceanic: island arcs, which become continental crust once thick enough.
    'subduction_H_cont_m': 1000., 'subduction_d0_cont_km': 250., 'subduction_sigma_cont_km': 120.,
    'arc_H_ocean_m': 300., 'arc_d0_km': 150., 'arc_sigma_km': 50.,
    'arc_to_continent_thick_m': 20000.,
    # subduction erosion: the trench removes crust from the base of an overriding continent near it (forearc), the
    # main continental loss balancing arc growth on Earth (~1.3-1.7 km^3/yr); flux = convergence x H_se
    'subduction_erosion_H_m': 700., 'subduction_erosion_d0_km': 50., 'subduction_erosion_sigma_km': 40.,
    'sediment_accretion': .3,      # share of trench sediment scraped onto the overriding plate
    # continental rifting: a gap opening between continental crust is stretched continental crust (rift_thinning x its
    # neighbours' thickness, volume taken from them) until it would be thinner than breakup_thick_m; then ocean floor
    'rift_thinning': .6, 'breakup_thick_m': 20000.,
    'mantle': True,                # dynamic topography and hotspots (planet_layer.mantle)
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
    'crust_kappa_m2_yr': 1000.,    # lateral spreading of continental crust, m^2/yr, at normal thickness ...
    'collapse_thick_m': 45000.,    # ... rising e-fold per collapse_scale_m above this (gravitational collapse of thick
    'collapse_scale_m': 6000.,     #     crust keeps plateaus near ~70 km, as Tibet and the Altiplano)
    'collapse_max_factor': 200.,
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


def gdh1_depth(age_myr):
    """Ocean depth (m, positive) from lithosphere age (Stein & Stein 1992)."""
    a = np.clip(age_myr, 0, None)
    return np.where(a < 20, 2600.+365.*np.sqrt(a), 5651.-2473.*np.exp(-.0278*a))


def isostatic_elevation(cont, thick, ocean_age, p=PARAMS):
    """Elevation (m): oceanic crust from age; continental crust from Airy thickness, water-loaded below 0 m."""
    dry = p['cont_ref_elev_m']+(thick-p['cont_ref_thick_m'])*(p['rho_mantle']-p['rho_crust'])/p['rho_mantle']
    wet = np.where(dry < 0, dry*p['rho_mantle']/(p['rho_mantle']-p['rho_water']), dry)
    return np.where(cont, wet, -gdh1_depth(ocean_age))


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
        self.sed = np.zeros(size, np.float32)            # sediment on oceanic crust, m

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
        self.mantle = Mantle(graph, np.random.default_rng([seed, 7]), t0) if self.p['mantle'] else None
        self.dyn = self.mantle.topography(t0) if self.mantle else np.zeros(graph.size)
        self.next_id = 0
        self.plates = []
        self.events = []
        self.ledger = dict.fromkeys(('new_nodes', 'holes', 'increments', 'collision_out', 'cont_subducted', 'cont_lost_other'), 0.)
        self.erosion_totals = {'continental_eroded_m3': 0., 'continental_deposited_m3': 0., 'ocean_sediment_m3': 0.}
        if start != 'supercontinent':
            raise NotImplementedError('only the supercontinent start exists so far (dispersed start: later)')
        self._init_supercontinent(n_plates, n_cont_plates, cont_fraction)
        self.compose_from_plates()

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
        self.new_crust = gap
        self.world = {'cont': wcont, 'thick': wthick, 'born': wborn, 'orog': worog, 'sed': wsed}

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
        while T > 0:                                     # nonlinear: substeps follow the current largest diffusivity
            kap = p['crust_kappa_m2_yr']*np.minimum(np.exp(np.clip(th-p['collapse_thick_m'], 0, None)/p['collapse_scale_m']),
                                                    p['collapse_max_factor'])
            h = min(T, .9/(kap[w['cont']].max()*self.bound) if w['cont'].any() else T)
            th = gl.diffuse_variable(th, w['cont'], kap, h, g.nbr, g.dist, g.axis, g.area)
            T -= h
        dthick = th-before+rift_dthick
        th = th+rift_dthick
        dsed = np.zeros(g.size)
        sub = self._subduction_thickening()
        if self.mantle is not None:
            vol = self.mantle.volcanic(self.t, dt)
            self.erosion_totals['volcanic_added_m3'] = self.erosion_totals.get('volcanic_added_m3', 0.)+float((vol*g.area).sum())
            sub = sub+vol
        dthick = dthick+sub
        th = th+sub
        self.tectonic = dthick.copy()                    # tectonic thickening this step, m (no erosion); + collisions below
        arc = (~w['cont']) & (th >= p['arc_to_continent_thick_m'])
        if arc.any():                                    # island arcs grown thick enough are continental crust
            w['cont'] = w['cont'] | arc
            self.new_continent = arc
        else:
            self.new_continent = np.zeros(g.size, bool)
        if p['erosion']:
            e_cont, dsed = self._erode(th)
            dthick = dthick+e_cont
            th = th+e_cont
        w['thick'] = th
        w['sed'] = w['sed']+dsed
        conv_t, conv_s = self._sediment_basins_to_continent()
        dthick = dthick+conv_t
        dsed = dsed+conv_s
        self._update_rasters(dthick, idx, dsed)
        self.plates = [pl for pl in self.plates if pl.mask.any()]
        self._classify()
        self._update_motion()
        self._merge_events()
        self._split_events()

    def _rift(self):
        """Gaps opening between continental crust become stretched continental crust (see PARAMS). Returns the
        thinning of the continental neighbours that supply the volume (m, <= 0)."""
        g, p, w = self.g, self.p, self.world
        out = np.zeros(g.size)
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
        meanT = np.where(cn, w['thick'][nbs], 0.).sum(1)/np.maximum(ncont, 1)
        Tg = p['rift_thinning']*meanT
        rift = (ncont > 0) & (ncont*2 >= solid.sum(1)) & (Tg >= p['breakup_thick_m'])
        if not rift.any():
            return out
        r, Tr, cr = rows[rift], Tg[rift], cn[rift]
        a = g.area
        share = (Tr*a[r]/ncont[rift])[:, None]*cr          # volume each continental neighbour gives (m^3)
        take = np.zeros(g.size)
        np.add.at(take, nbs[rift][cr], share[cr])
        out = -np.minimum(take/a, .5*w['thick'])
        w['cont'][r] = True
        w['thick'][r] = Tr
        w['born'][r] = np.where(cr, w['born'][nbs[rift]], np.inf).min(1)   # stretched old crust keeps its age
        self.erosion_totals['rift_nodes'] = self.erosion_totals.get('rift_nodes', 0)+int(r.size)
        return out

    def _sediment_basins_to_continent(self):
        """Oceanic nodes carrying more than basin_to_continent_sed_m of sediment, or sediment above sea level, become
        continental crust. Its thickness is set so the node keeps its present elevation (the isostatic equivalent)."""
        g, p, w = self.g, self.p, self.world
        z = self.elevation()
        conv = (~w['cont']) & ((w['sed'] > p['basin_to_continent_sed_m']) | ((w['sed'] > 500.) & (z > 0)))
        dT = np.zeros(g.size); dS = np.zeros(g.size)
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
        for mask, H, d0, sg in ((ok & cont, p['subduction_H_cont_m'], p['subduction_d0_cont_km']*1e3, p['subduction_sigma_cont_km']*1e3),
                                (ok & ~cont, p['arc_H_ocean_m'], p['arc_d0_km']*1e3, p['arc_sigma_km']*1e3),
                                (ok, -p['subduction_erosion_H_m'], p['subduction_erosion_d0_km']*1e3, p['subduction_erosion_sigma_km']*1e3)):
            if H == 0:
                continue
            shape = np.exp(-.5*((d-d0)/sg)**2)/(sg*np.sqrt(2*np.pi))
            out = out+np.where(mask, v*H*shape*dt, 0.)
        out = np.maximum(out, -.3*w['thick'])
        for key, m in (('subduction_on_continents_m3', cont), ('subduction_on_ocean_arcs_m3', ~cont)):
            self.erosion_totals[key] = self.erosion_totals.get(key, 0.)+float((out*g.area)[m].sum())
        self.erosion_totals['subduction_added_m3'] = self.erosion_totals.get('subduction_added_m3', 0.)+float((out*g.area).sum())
        return out

    def _erode(self, thick):
        """Coarse erosion over one plate step: the landscape model's surface processes (graph_landscape.surface_step)
        on the flexural-isostatic elevation, in erosion_substeps steps. Returns the change of continental crust
        thickness (erosion < 0, sediment deposited on land > 0) and of ocean-floor sediment."""
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
        for _ in range(p['erosion_substeps']):
            z = gl.surface_step(g, z, seed, None, fallback, K, lp, dt, self.h_min, self.bound)['z']
        dz = z-z0
        cont = w['cont']
        e_cont = np.where(cont, dz, 0.)
        dsed = np.where(cont, 0., np.maximum(dz, -w['sed']))
        a = g.area
        self.erosion_totals['continental_eroded_m3'] += float((np.minimum(e_cont, 0)*a).sum())
        self.erosion_totals['continental_deposited_m3'] += float((np.maximum(e_cont, 0)*a).sum())
        self.erosion_totals['ocean_sediment_m3'] += float((dsed*a).sum())
        return e_cont, dsed

    def elevation(self, thick=None, sed=None):
        """Elevation (m) with flexural isostasy on an elastic plate (Te).

        Continental crust: its thickness anomaly (relative to cont_ref_thick_m, which stands at cont_ref_elev_m) is a
        load of density rho_crust against the mantle. Ocean floor: thermal subsidence with age (GDH1); its crust
        anomaly (arcs, plateaus) and sediment are submarine loads of density (rho - rho_water) against
        (rho_mantle - rho_water), so a basin filled to sea level holds ~2.5x its depth of sediment, as on Earth.
        Long wavelengths reduce to Airy isostasy; valleys and single belts are partly held up by the plate.
        A thinned continental column below sea level is water-loaded (its own Airy depth only, not its neighbours').
        """
        p, w = self.p, self.world
        thick = w['thick'] if thick is None else thick
        sed = w['sed'] if sed is None else sed
        cont = w['cont']
        rc, rm, rw, rs = p['rho_crust'], p['rho_mantle'], p['rho_water'], p['rho_sediment']
        anomaly = np.where(cont, thick-p['cont_ref_thick_m'], thick-p['ocean_thick_m'])
        ocean_age = np.where(cont, 0., self.t-w['born'])
        above = (-gdh1_depth(ocean_age)+anomaly+sed) > 0        # oceanic crust built above sea level (arcs, fills)
        dry = np.where(cont, rc*anomaly, np.where(above, rc*anomaly+rs*sed, 0.))
        wet = np.where(cont | above, 0., (rc-rw)*anomaly+(rs-rw)*sed)
        defl = self.flex_dry.deflection(dry)+self.flex_wet.deflection(wet)
        z = np.where(cont, p['cont_ref_elev_m']+anomaly, -gdh1_depth(ocean_age)+anomaly+sed)-defl
        own_airy = p['cont_ref_elev_m']+anomaly*(rm-rc)/rm
        z = z+np.where(cont & (own_airy < 0), own_airy*rw/(rm-rw), 0.)
        return z+self.dyn                                 # dynamic topography: non-load surface motion

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
            subducted[wn[gone & ~pl.cont]] = True
            sg = gone & ~pl.cont & (pl.sed > 0)          # trench sediment: part is scraped onto the overriding plate
            np.add.at(dVc, wn[sg], p['sediment_accretion']*pl.sed[sg].astype(np.float64)*a[sg])
            self.erosion_totals['sediment_accreted_m3'] = self.erosion_totals.get('sediment_accreted_m3', 0.)+float(
                (p['sediment_accretion']*pl.sed[sg]*a[sg]).sum())
            L = self.ledger
            L['collision_out'] -= float((pl.thick[cc]*a[cc]).sum())
            lost = gone & pl.cont & ~w['cont'][wn]
            L['cont_subducted'] -= float((pl.thick[lost]*a[lost]).sum())
            wns.append(wn); owns.append(own); consumed.append(gone)
        self.collided, self.subducted = collided, subducted
        if hasattr(self, 'tectonic'):
            self.tectonic = self.tectonic+dVc/a          # collision volume, as thickness at the receiving world node
        w['orog'] = np.where(collided, self.t, w['orog'])
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
            d_int = np.bincount(js, weights=dthick[ws], minlength=g.size)/np.maximum(nread, 1)
            d_vol = np.bincount(js, weights=dVc[ws], minlength=g.size)/a
            pl.thick[read] += (d_int[read]+d_vol[read]).astype(np.float32)
            if dsed is not None:
                d_sed = np.bincount(js, weights=dsed[ws], minlength=g.size)/np.maximum(nread, 1)
                pl.sed[read] = np.maximum(pl.sed[read]+d_sed[read], 0).astype(np.float32)
            self.ledger['increments'] += float((pl.thick*a*(pl.mask & pl.cont)).sum())-before_v
            conv = self.new_continent[ws] if hasattr(self, 'new_continent') else np.zeros(ws.size, bool)
            if conv.any():
                pl.cont[js[conv]] = True
                pl.orog[js[conv]] = self.t
                self.ledger['arc_continent'] = self.ledger.get('arc_continent', 0.)+float((pl.thick[js[conv]]*a[js[conv]]).sum())
            hit = np.zeros(g.size, bool); hit[js[collided[ws]]] = True
            pl.orog[hit] = self.t
            pl.cont[new] = w['cont'][wn[new]]
            pl.thick[new] = w['thick'][wn[new]]
            pl.born[new] = w['born'][wn[new]]
            pl.orog[new] = w['orog'][wn[new]]
            pl.sed[new] = w['sed'][wn[new]]
            pl.under[new] = 0
            self.ledger['new_nodes'] += float((pl.thick[new]*a[new]*pl.cont[new]).sum())
            pl.mask[gone] = False
            pl.mask[new] = True
            pl.cont[~pl.mask] = False
            hole, src = plate_holes(pl.mask, g.nbr)
            if hole.any():
                hs = src[hole]
                pl.mask[hole] = True
                for arr in (pl.cont, pl.thick, pl.born, pl.orog, pl.sed):
                    arr[hole] = arr[hs]
                self.ledger['holes'] += float((pl.thick[hole]*a[hole]*pl.cont[hole]).sum())
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

    def _update_motion(self):
        g, p = self.g, self.p
        dt = p['dt_myr']
        h = np.sqrt(g.area.mean())
        for pl in self.plates:
            own = self.owner == pl.id
            if not own.any():
                continue
            c = (g.area[own, None]*g.xyz[own]).sum(0); c /= np.linalg.norm(c)
            tr = own & self.trench
            ri = own & (self.btype == RIDGE)
            coll = own & (self.btype == COLLISION)
            L_slab = tr.sum()*h/2e3                      # boundary nodes sit on both sides of a line: halve
            L_ridge = ri.sum()*h/2e3
            L_coll = coll.sum()*h/2e3
            cf = (g.area*(own & self.world['cont'])).sum()/(g.area*own).sum()
            target = (p['v_min_cm_yr']+(p['v_slab_cm_yr']-p['v_min_cm_yr'])*min(1., L_slab/p['slab_ref_km'])
                      + p['v_ridge_cm_yr']*min(1., L_ridge/p['ridge_ref_km']))
            target *= (1-p['cont_drag']*cf)/(1+L_coll/p['collision_ref_km'])
            pl.speed += (target-pl.speed)*min(1., dt/p['tau_speed_myr'])
            # pole turns so the plate moves toward its trenches (slab pull) and away from its ridges (ridge push)
            pull = np.cross(c, g.xyz[tr]).sum(0)+p['ridge_torque']*np.cross(g.xyz[ri], c).sum(0)
            nrm = np.linalg.norm(pull)
            if nrm > 0:
                b = min(1., dt/p['tau_axis_myr'])
                a = (1-b)*pl.axis+b*pull/nrm
                pl.axis = a/np.linalg.norm(a)

    # ------------------------------------------------------------------ plate splitting
    def _omega_vec(self, pl):
        return pl.axis*pl.speed

    def _set_omega(self, pl, v):
        sp = float(np.linalg.norm(v))
        pl.speed = max(sp, 1e-3)
        if sp > 1e-9:
            pl.axis = v/sp

    def split(self, pl, part_world, rel_speed_cm_yr, kind):
        """Split plate `pl`: the world nodes in part_world (owned by pl) become a new plate in the same frame (no
        resampling). The two parts get Euler vectors differing by rel_speed (positive: diverge, negative: converge)."""
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
        for name in ('mask', 'cont', 'thick', 'born', 'orog', 'under', 'sed'):
            arr = getattr(pl, name)
            na = getattr(new, name)
            na[lab] = arr[lab]
        pl.mask[lab] = False; pl.cont[lab] = False
        c1 = (g.area[rest, None]*g.xyz[rest]).sum(0); c1 /= np.linalg.norm(c1)
        c2 = (g.area[part, None]*g.xyz[part]).sum(0); c2 /= np.linalg.norm(c2)
        d = np.cross(c1, c2)
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
        wk = fields.nearest_nodes(g, g.xyz@big.Q.T)                  # big node -> world node
        take = self.owner[wk] == small.id
        k = np.flatnonzero(take)
        j = fields.nearest_nodes(g, (g.xyz[k]@big.Q.T)@small.Q)      # -> small's raster
        ok = small.mask[j]
        k, j = k[ok], j[ok]
        free = ~big.mask[k]
        for name in ('cont', 'thick', 'born', 'orog', 'sed'):
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
            for name in ('cont', 'thick', 'born', 'orog', 'sed'):
                arr = getattr(big, name); arr[need] = arr[src]
        both = ~free & small.cont[j] & big.cont[k]
        np.add.at(big.thick, k[both], small.thick[j[both]])
        a_big, a_small = g.area[self.owner == big.id].sum(), g.area[self.owner == small.id].sum()
        self._set_omega(big, (self._omega_vec(big)*a_big+self._omega_vec(small)*a_small)/(a_big+a_small))
        self.owner = np.where(self.owner == small.id, big.id, self.owner)
        self.plates = [q for q in self.plates if q is not small]
        self.ledger['merge'] = self.ledger.get('merge', 0.)+self.continental_volume()-v0
        self.events.append({'t': self.t, 'kind': 'merge', 'plate': int(big.id), 'merged': int(small.id),
                            'area_mkm2': float(a_small/1e12)})

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
                'tectonic_thickening_m': getattr(self, 'tectonic', np.zeros(self.g.size)).astype(np.float32)}

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

    def pinholes(self):
        """New-crust nodes more than 2.5 cells from any plate boundary: holes inside a plate (should be ~0). Ridge crust
        is always next to the boundary it opened at."""
        g = self.g
        nb = np.where(g.nbr >= 0, g.nbr, np.arange(g.size)[:, None])
        edge = (self.owner[nb] != self.owner[:, None]).any(1)
        d = gl.distance_to(edge, g.nbr, g.dist, 3*np.sqrt(g.area.mean()))
        return int((self.new_crust & (d > 2.5*np.sqrt(g.area.mean()))).sum())
