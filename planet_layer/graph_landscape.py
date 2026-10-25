"""Landscape evolution on a node graph: the processes of planet_training.landscape_evolution, unchanged, on any
planet_layer.sphere_graph.Graph (the cube sphere or a flat test window).

Each 50 kyr step:
  1. rock uplift (thickening, m/yr: a flexural load) and non-load vertical motion (m/yr: dynamic topography,
     thermal subsidence; moves the surface without loading the plate);
  2. open sea = water connected to the open-sea seeds; priority flood to spill levels; D8 steepest descent on true
     neighbour distances; drainage area from node cell areas;
  3. Yuan et al. 2019 implicit erosion-deposition (n = 1), slope cap Sc; lake filling;
  4. sediment entering the sea builds deltas and graded shelves (distance to land along the graph);
  5. hillslope diffusion on land and marine diffusion on shelves (finite volume over axis neighbours);
  6. flexure of the rock load (FlatFlexure for the window, SphereFlexure on the sphere).

On flat.Graph the results match landscape_evolution.Landscape (tests/test_graph_landscape.py). Sea level is the 0 m
datum only; nothing is frozen, conserved or projected.
"""
import heapq
import time

import numpy as np

from planet_training import landscape_evolution as le
from planet_training.landscape_evolution import njit, RHO_CRUST, RHO_SEDIMENT

DEFAULTS = dict(le.DEFAULTS)


@njit(cache=True)
def connected_ocean(z, seed, nbr):
    n = z.shape[0]
    ocean = np.zeros(n, np.bool_)
    work = np.empty(n, np.int64); w = 0
    for i in range(n):
        if seed[i] and z[i] <= 0.:
            ocean[i] = True; work[w] = i; w += 1
    while w > 0:
        w -= 1; i = work[w]
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0:
                continue
            if not ocean[j] and z[j] <= 0.:
                ocean[j] = True; work[w] = j; w += 1
    return ocean


@njit(cache=True)
def priority_flood(z, outlet, fallback, nbr, eps):
    """Depression-filled surface seeded at outlets; with no outlet, at the lowest fallback node."""
    n = z.shape[0]
    zf = z.copy()
    done = np.zeros(n, np.bool_)
    heap = [(0.0, -1)]
    heap.pop()
    count = 0
    for i in range(n):
        if outlet[i]:
            done[i] = True
            count += 1
    for i in range(n):
        # only outlets touching a non-outlet can spread the flood; skipping the others leaves the order unchanged
        if outlet[i]:
            for k in range(nbr.shape[1]):
                j = nbr[i, k]
                if j >= 0 and not outlet[j]:
                    heapq.heappush(heap, (z[i], i))
                    break
    if count == 0:
        best = -1
        for i in range(n):
            if fallback[i] and (best < 0 or z[i] < z[best]):
                best = i
        done[best] = True
        heapq.heappush(heap, (z[best], best))
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or done[j]:
                continue
            done[j] = True
            if zf[j] <= h+eps:
                zf[j] = h+eps
            heapq.heappush(heap, (zf[j], j))
    return zf


@njit(cache=True)
def route(zf, outlet, nbr, dist):
    """D8 receivers on the filled surface, topological stack (outlets first) and receiver distances."""
    n = zf.shape[0]
    rec = np.arange(n)
    rdist = np.empty(n)
    for i in range(n):
        rdist[i] = dist[i, 0] if nbr[i, 0] >= 0 else dist[i, 1]
        if outlet[i]:
            continue
        best_s = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0:
                continue
            d = dist[i, k]
            s = (zf[i]-zf[j])/d
            if s > best_s:
                best_s = s; rec[i] = j; rdist[i] = d
    ndon = np.zeros(n, np.int64)
    for i in range(n):
        if rec[i] != i:
            ndon[rec[i]] += 1
    start = np.zeros(n+1, np.int64)
    for i in range(n):
        start[i+1] = start[i]+ndon[i]
    fill = start[:-1].copy()
    donors = np.zeros(max(start[n], 1), np.int64)
    for i in range(n):
        j = rec[i]
        if j != i:
            donors[fill[j]] = i; fill[j] += 1
    stack = np.empty(n, np.int64)
    top = 0
    work = np.empty(n, np.int64)
    for o in range(n):
        if rec[o] != o:
            continue
        work[0] = o; w = 1
        while w > 0:
            w -= 1; i = work[w]
            stack[top] = i; top += 1
            for q in range(start[i], start[i+1]):
                work[w] = donors[q]; w += 1
    return rec, rdist, stack[:top]


@njit(cache=True)
def accumulate(rec, stack, cell_area):
    a = cell_area.copy()
    for k in range(stack.shape[0]-1, -1, -1):
        i = stack[k]; j = rec[i]
        if j != i:
            a[j] += a[i]
    return a


@njit(cache=True)
def erode_and_deposit(z, zf, ocean, rec, rdist, stack, area, K, m, G, Sc, dt, cell_area, lake_tol, iterations=6, theta=0.):
    """Yuan et al. 2019 coupled erosion-deposition (n = 1), then lake / river-mouth deposition (see landscape_evolution).

    theta > 0 adds an incision threshold, E = K (A^m S - theta) where positive, else no fluvial erosion: rivers need
    enough stream power to cut bedrock, so low-gradient lowlands stop eroding (cratons and platforms erode at only
    ~1-10 m/Myr on Earth). theta = 0 is exactly the regional model."""
    n = z.shape[0]
    lake = (zf-z) > lake_tol
    zn = z.copy()
    qs = np.zeros(n)
    for it in range(iterations):
        qs[:] = 0.
        for k in range(stack.shape[0]-1, -1, -1):
            i = stack[k]; j = rec[i]
            if j != i:
                qs[j] += max(0., qs[i]+(z[i]-zn[i])*cell_area[i])
        for k in range(stack.shape[0]):
            i = stack[k]; j = rec[i]
            if j == i or ocean[i] or lake[i]:
                continue
            if ocean[j]:
                base = 0.
            elif lake[j]:
                base = max(zf[j], zn[j])
            else:
                base = zn[j]
            f = K[i]*area[i]**m*dt/rdist[i]
            v = (z[i]+G*qs[i]/area[i]+f*base+K[i]*dt*theta)/(1.+f)
            if theta > 0. and area[i]**m*(v-base)/rdist[i] <= theta:
                v = z[i]+G*qs[i]/area[i]                 # below threshold: deposition only
            v = min(v, base+Sc*rdist[i])
            zn[i] = v
    eroded = np.maximum(z-zn, 0.)
    deposited = np.maximum(zn-z, 0.)
    out = np.zeros(n)
    for k in range(stack.shape[0]-1, -1, -1):
        i = stack[k]; j = rec[i]
        out[i] += (z[i]-zn[i])*cell_area[i]
        if out[i] > 0.:
            d = 0.
            if lake[i]:
                d = min(out[i], (zf[i]-zn[i])*cell_area[i])
            zn[i] += d/cell_area[i]; deposited[i] += d/cell_area[i]; out[i] -= d
        else:
            out[i] = 0.
        if j != i:
            out[j] += out[i]
    sea_in = np.where(ocean, out, 0.)
    return zn, eroded, deposited, sea_in


@njit(cache=True)
def distance_to(source, nbr, dist, cutoff=np.inf):
    """Graph distance (m) from every node to the nearest source node (multi-source Dijkstra). Nodes farther than
    cutoff keep 1e15 (exact below the cutoff: every prefix of a shortest path is shorter than the path)."""
    n = source.shape[0]
    d = np.full(n, 1e15)
    heap = [(0.0, -1)]
    heap.pop()
    for i in range(n):
        if source[i]:
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
            if v < d[j] and v <= cutoff:
                d[j] = v
                heapq.heappush(heap, (v, j))
    return d


def marine_sediment(z, ocean, sea_in, nbr, dist, cell_area, shelf_depth, shelf_gradient, dist_to_land, shelf_width):
    """Sediment entering the sea moves downslope underwater to a graded shelf surface (see landscape_evolution).
    What finds no room on the shelf leaves the domain ("goes to the deep sea"): right for an open regional window,
    a mass sink on a closed planet (see marine_sediment_kept)."""
    return marine_sediment_kept(z, ocean, sea_in, nbr, dist, cell_area, shelf_depth, shelf_gradient, dist_to_land,
                                shelf_width)[0]


@njit(cache=True)
def marine_sediment_kept(z, ocean, sea_in, nbr, dist, cell_area, shelf_depth, shelf_gradient, dist_to_land, shelf_width):
    """marine_sediment, also returning the volume (m^3) that found no room, at the node where its path ended (a
    local low of the sea floor beyond the shelf)."""
    n = z.shape[0]
    flux = sea_in.copy()
    dep = np.zeros(n)
    left = np.zeros(n)
    heap = [(0.0, -1)]
    heap.pop()
    for i in range(n):
        if ocean[i] and flux[i] > 0.:
            heapq.heappush(heap, (-(z[i]), i))
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        if flux[i] <= 0.:
            continue
        if sea_in[i] > 0.:
            top = 2.
        elif dist_to_land[i] <= shelf_width:
            top = -(shelf_depth+shelf_gradient*dist_to_land[i])
        else:
            top = -1e9
        cap = max(0., top-z[i]-dep[i])*cell_area[i]
        d = min(flux[i], cap)
        dep[i] += d/cell_area[i]; flux[i] -= d
        if flux[i] <= 0.:
            continue
        best = -1; best_s = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not ocean[j]:
                continue
            s = ((z[i]+dep[i])-(z[j]+dep[j]))/dist[i, k]
            if s > best_s:
                best_s = s; best = j
        if best >= 0:
            # volume moves to the receiving node
            flux[best] += flux[i]
            heapq.heappush(heap, (-(z[best]+dep[best]), best))
        else:
            left[i] += flux[i]
        flux[i] = 0.
    return dep, left


@njit(cache=True)
def fill_and_spill(z, vol, nbr, cell_area):
    """Place volumes vol (m^3, at sea-floor lows: marine_sediment_kept's leftover) by filling depressions from the
    bottom up: the deposit ponds at the lowest open level (a priority flood from the node), raising the pond to the
    next lowest neighbour until the volume is used; where the pond reaches a lower neighbour it spills, and the rest
    runs down the steepest path from there and ponds again. Turbidites fill abyssal plains and basins this way; the
    surface stays continuous and every m^3 is placed. Returns the new surface."""
    n = z.shape[0]
    z = z.copy()
    stamp = np.zeros(n, np.int64)
    lake = np.empty(n, np.int64)
    run = 0
    order = np.argsort(-vol)
    for k in range(n):
        p = order[k]
        V = vol[p]
        if V <= 0.:
            break
        guard = 0
        while V > 0. and guard < 10000:
            guard += 1
            # walk down to a local low
            moved = True
            while moved:
                moved = False
                best = p
                for m in range(nbr.shape[1]):
                    j = nbr[p, m]
                    if j >= 0 and z[j] < z[best]:
                        best = j
                if best != p:
                    p = best
                    moved = True
            run += 1
            stamp[p] = run
            nl = 1
            lake[0] = p
            A = cell_area[p]
            level = z[p]
            heap = [(0.0, -1)]
            heap.pop()
            for m in range(nbr.shape[1]):
                j = nbr[p, m]
                if j >= 0 and stamp[j] != run:
                    stamp[j] = run
                    heapq.heappush(heap, (z[j], j))
            spill = -1
            while V > 0. and len(heap) > 0:
                hq, q = heapq.heappop(heap)
                if hq < level:                       # overflow point: the rest runs down from q
                    spill = q
                    break
                need = (hq-level)*A
                if V <= need:
                    level += V/A
                    V = 0.
                    break
                V -= need
                level = hq
                lake[nl] = q
                nl += 1
                A += cell_area[q]
                for m in range(nbr.shape[1]):
                    j = nbr[q, m]
                    if j >= 0 and stamp[j] != run:
                        stamp[j] = run
                        heapq.heappush(heap, (z[j], j))
            if V > 0. and spill < 0:                 # the whole graph is one pond (not on a planet)
                level += V/A
                V = 0.
            for i in range(nl):
                z[lake[i]] = level
            if spill >= 0:
                p = spill
    return z


@njit(cache=True)
def basins(rec, stack):
    out = np.arange(rec.shape[0])
    for k in range(stack.shape[0]):
        i = stack[k]; j = rec[i]
        if j != i:
            out[i] = out[j]
    return out


@njit(cache=True)
def diffuse(z, domain, kd, dt, nbr, dist, axis, cell_area):
    """Explicit finite-volume diffusion among axis neighbours that are both in the domain (zero flux elsewhere)."""
    n = z.shape[0]
    out = z.copy()
    for i in range(n):
        if not domain[i]:
            continue
        s = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not axis[i, k] or not domain[j]:
                continue
            d = dist[i, k]
            s += kd*(z[j]-z[i])/d*(np.sqrt(cell_area[i]*cell_area[j])/d)
        out[i] = z[i]+dt*s/cell_area[i]
    return out


@njit(cache=True)
def diffuse_variable(z, domain, kappa, dt, nbr, dist, axis, cell_area):
    """diffuse() with a per-node diffusivity (edge value: mean of its two nodes); conservative."""
    n = z.shape[0]
    out = z.copy()
    for i in range(n):
        if not domain[i]:
            continue
        s = 0.
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j < 0 or not axis[i, k] or not domain[j]:
                continue
            d = dist[i, k]
            s += .5*(kappa[i]+kappa[j])*(z[j]-z[i])/d*(np.sqrt(cell_area[i]*cell_area[j])/d)
        out[i] = z[i]+dt*s/cell_area[i]
    return out


def surface_step(g, z, sea_seed, edge_outlets, fallback, K, p, dt, h_min, bound, tick=None, start=None, keep_deep=False):
    """One step of the surface processes on a graph, without tectonic forcing or flexure: open sea, priority flood,
    D8 routing, Yuan erosion-deposition with lakes, marine sediment (deltas, graded shelves), hillslope and marine
    diffusion. Returns the new surface and the routing state. Used by GraphLandscape and the plate layer.
    keep_deep: also return 'deep_m3', the sediment that found no room on the shelves, at the sea-floor low where its
    path ended (the regional model drops it; a closed planet must place it: the plate layer spreads it as fans)."""
    c = time.perf_counter() if start is None else start
    tick = tick or (lambda key, c0: time.perf_counter())
    ocean = connected_ocean(z, sea_seed, g.nbr)
    outlets = ocean | edge_outlets if edge_outlets is not None else ocean
    zf = priority_flood(z, outlets, fallback, g.nbr, 1e-3)
    rec, dist, stack = route(zf, outlets, g.nbr, g.dist)
    area = accumulate(rec, stack, g.area)
    c = tick('routing', c)
    zn, eroded, deposited, sea_in = erode_and_deposit(z, zf, ocean, rec, dist, stack, area, K, p['m'], p['G'], p['Sc'], dt,
                                                      g.area, p['lake_tolerance'], 6, p.get('theta', 0.))
    c = tick('erosion', c)
    marine, deep = marine_sediment_kept(zn, ocean, sea_in, g.nbr, g.dist, g.area, p['shelf_depth'], p['shelf_gradient'],
                                        distance_to(~ocean, g.nbr, g.dist, p['shelf_width']), p['shelf_width'])
    zn = zn+marine; deposited = deposited+marine
    c = tick('marine', c)
    land = ~ocean
    zn = diffuse(zn, land, p['Kd'], dt, g.nbr, g.dist, g.axis, g.area)
    before = zn.copy()
    sea = (~land) & (zn > -1000)
    # substeps: the regional model's rule (kept so the flat window matches it exactly), and never fewer than the
    # graph's own stability bound needs
    sub = max(1, int(np.ceil(p['Kd_marine']*dt/h_min**2/.2)), int(np.ceil(p['Kd_marine']*dt*bound/.9)))
    for _ in range(sub):
        zn = diffuse(zn, sea, p['Kd_marine'], dt/sub, g.nbr, g.dist, g.axis, g.area)
    moved = zn-before
    deposited = deposited+np.maximum(moved, 0.); eroded = eroded+np.maximum(-moved, 0.)
    c = tick('diffusion', c)
    out = {'z': zn, 'eroded': eroded, 'deposited': deposited, 'ocean': ocean, 'outlet': outlets, 'zf': zf, 'rec': rec,
           'dist': dist, 'stack': stack, 'area': area, 'clock': c}
    if keep_deep:
        out['deep_m3'] = deep
    return out


class FlatFlexure:
    """landscape_evolution.Flexure for a flat ny x nx window, on node vectors."""

    def __init__(self, ny, nx, dx, te, young=7e10, poisson=.25):
        self.f = le.Flexure(ny, nx, dx, te, young, poisson)
        self.shape = (ny, nx)
        self.alpha_m = self.f.alpha_m

    def deflection(self, load):
        return self.f.deflection(load.reshape(self.shape)).ravel()


class GraphLandscape:
    """State and stepping on a graph. uplift / surface_rate: callables of t (years) -> (N,) m/yr, arrays, or None."""

    def __init__(self, graph, z0, uplift, flexure, params=None, erodibility=None, open_sea=None, surface_rate=None):
        self.g = graph
        self.p = {**DEFAULTS, **(params or {})}
        self.z = np.asarray(z0, np.float64).ravel().copy()
        n = self.z.shape[0]
        if open_sea is None:
            if graph.edge.any():
                seed = graph.edge & (self.z <= 0)
            else:
                seed = np.zeros(n, bool); seed[int(np.argmin(self.z))] = True
            open_sea = connected_ocean(self.z, seed, graph.nbr)
        self.sea_seed = np.asarray(open_sea, bool).ravel()
        self.edge = graph.edge
        self.fallback = graph.edge if graph.edge.any() else np.ones(n, bool)
        self.uplift = uplift
        self.surface_rate = surface_rate
        self.K = (self.p['K']*(np.ones(n) if erodibility is None else np.asarray(erodibility, np.float64).ravel()))
        self.load = np.zeros(n)
        self.flex = flexure
        self.w = np.zeros(n)
        self.t = 0.
        self.h_min = graph.spacing_min()
        self.bound = graph.diffusion_bound()
        if self.p['Kd']*self.p['dt']*self.bound > .9:
            raise ValueError('hillslope diffusion unstable at this dt and spacing (Kd dt bound > 0.9)')
        self.totals = {'uplift_m3': 0., 'surface_motion_m3': 0., 'eroded_m3': 0., 'deposited_m3': 0.}
        self.timing = dict.fromkeys(('forcing', 'routing', 'erosion', 'marine', 'diffusion', 'flexure'), 0.)
        self._route()

    def _rate(self, f):
        if f is None:
            return 0.
        return np.asarray(f(self.t) if callable(f) else f, np.float64).ravel()

    def _route(self):
        g = self.g
        self.ocean = connected_ocean(self.z, self.sea_seed, g.nbr)
        outlets = self.ocean | self.edge if self.p['open_edges'] else self.ocean
        self.outlet_mask = outlets
        self.zf = priority_flood(self.z, outlets, self.fallback, g.nbr, 1e-3)
        self.rec, self.dist, self.stack = route(self.zf, outlets, g.nbr, g.dist)
        self.area = accumulate(self.rec, self.stack, g.area)

    def step(self):
        p, g, tm = self.p, self.g, self.timing
        dt = p['dt']
        c = time.perf_counter()
        du = self._rate(self.uplift)*dt+np.zeros_like(self.z)
        ds = self._rate(self.surface_rate)*dt+np.zeros_like(self.z)
        if p['open_edges'] and self.edge.any():
            hold = self.edge & ~self.ocean
            du = np.where(hold, 0., du); ds = np.where(hold, 0., ds)
        self.z += du+ds
        c = self._tick('forcing', c)
        r = surface_step(g, self.z, self.sea_seed, self.edge if p['open_edges'] else None, self.fallback, self.K, p, dt,
                         self.h_min, self.bound, tick=lambda key, c0: self._tick(key, c0), start=c)
        self.ocean, self.outlet_mask, self.zf, self.rec, self.dist, self.stack, self.area = (
            r['ocean'], r['outlet'], r['zf'], r['rec'], r['dist'], r['stack'], r['area'])
        zn, eroded, deposited = r['z'], r['eroded'], r['deposited']
        c = r['clock']
        self.load += du-eroded+deposited*RHO_SEDIMENT/RHO_CRUST
        w = self.flex.deflection(self.load)
        self.z = zn-(w-self.w)
        self.w = w
        c = self._tick('flexure', c)
        self.t += dt
        a = g.area
        self.totals['uplift_m3'] += float((du*a).sum())
        self.totals['surface_motion_m3'] += float((ds*a).sum())
        self.totals['eroded_m3'] += float((eroded*a).sum())
        self.totals['deposited_m3'] += float((deposited*a).sum())

    def _tick(self, key, start):
        now = time.perf_counter()
        self.timing[key] += now-start
        return now

    def run(self, years, callback=None, every=None):
        steps = int(round(years/self.p['dt']))
        for s in range(steps):
            self.step()
            if callback and every and (s+1) % every == 0:
                callback(self)
        self._route()
        return self

    def outputs(self, river_area_m2=2e10):
        """Node vectors: elevation, filled surface, drainage area, receiver, outlet, rivers, lakes, sea, deflection."""
        outlet = basins(self.rec, self.stack)
        land = ~self.ocean
        return {'elevation_m': self.z.astype(np.float32), 'filled_m': self.zf.astype(np.float32),
                'drainage_area_m2': self.area.astype(np.float32), 'receiver': self.rec.astype(np.int64),
                'outlet': outlet.astype(np.int64), 'river': land & (self.area >= river_area_m2),
                'lake_depth_m': np.where(land, self.zf-self.z, 0).astype(np.float32),
                'drains_to_sea': self.ocean[outlet] & land, 'ocean': self.ocean.copy(),
                'deflection_m': self.w.astype(np.float32), 'rock_load_m': self.load.astype(np.float32)}
