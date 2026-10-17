"""Regional landscape evolution: tectonic uplift, rivers, sediment and flexure, in metres and years.

One step does, in order:
  1. tectonic rock uplift / subsidence (from a template field, m/yr);
  2. flow routing on the depression-filled surface (priority flood), D8 steepest descent,
     drainage area;
  3. fluvial erosion, stream-power law E = K A^m S (n = 1, implicit, unconditionally stable),
     with lakes and the sea as local base level, plus a critical slope Sc standing in for
     landsliding (cell-averaged slopes cannot exceed it);
  4. sediment: erosion and river deposition are solved together (Yuan et al. 2019 implicit
     erosion-deposition, deposition = G * Qs / A), so large lowland rivers become
     transport-limited; leftover sediment fills closed depressions (lakes, landlocked
     forelands, intermontane basins) towards their spill level; sediment reaching the sea moves
     downslope underwater, building deltas and shelves and filling shallow seas;
  5. hillslope diffusion on land;
  6. flexural isostasy: the lithosphere (elastic thickness Te) bends under the changing rock
     load, which is what creates foredeeps such as the Po plain beside the Alps.

Sea level is fixed at 0 m. The ocean is water connected to the open sea; a landlocked cell
below 0 m is a closed basin, not ocean. Ocean cells are base level. By default land on the
domain edge is also an outlet held at its initial height, standing in for the continent that
continues beyond the window (open_edges=False makes every drop reach the sea or a closed
basin). Drainage basins, rivers and closed basins are exact by-products, not post-processing.

Numba makes a step take milliseconds; without it the same code runs in plain Python
(slow but correct).
"""
import heapq

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

G = 9.81
RHO_CRUST = 2700.
RHO_MANTLE = 3300.
RHO_SEDIMENT = 2400.

DEFAULTS = {
    'K': 8.0e-5,          # stream-power erodibility, m^(1-2m)/yr (tuned at 11.52 km cells)
    'm': .4,              # drainage-area exponent (n = 1)
    'G': 1.0,             # deposition coefficient (Yuan et al. 2019); 0 = detachment-limited
    'Sc': .05,            # critical cell-to-cell slope at ~11.5 km cells (Alps / Himalaya fronts ~.05)
    'Kd': 50.,            # effective hillslope diffusivity at this cell size, m^2/yr
    'Te': 30e3,           # elastic thickness, m
    'E_young': 7e10, 'poisson': .25,
    'shelf_depth': 10.,   # marine fill reaches this depth at the shoreline ...
    'shelf_gradient': 1.2e-3,  # ... and deepens by this much per metre offshore (~1.2 m/km)
    'shelf_width': 150e3, # m; no shelf fill beyond this distance from land (shelf break); the rest goes to the deep sea
    'Kd_marine': 2e3,     # m^2/yr; spreads delta sediment into shelves and fills shallow seas
    'lake_tolerance': .5, # m; filled surface this far above ground counts as lake / closed basin
    'dt': 5e4,            # years per step
    'open_edges': True,   # land edge cells drain out of the window at fixed height
}

_OFF = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], np.int64)


@njit(cache=True)
def connected_ocean(z, seed, ny, nx):
    """Cells at or below sea level connected (8-neighbour) to the open-sea seed cells."""
    n = ny*nx
    ocean = np.zeros(n, np.bool_)
    work = np.empty(n, np.int64); w = 0
    for i in range(n):
        if seed[i] and z[i] <= 0.:
            ocean[i] = True; work[w] = i; w += 1
    while w > 0:
        w -= 1; i = work[w]
        r, c = i//nx, i % nx
        for k in range(8):
            rr, cc = r+_OFF[k, 0], c+_OFF[k, 1]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx:
                continue
            j = rr*nx+cc
            if not ocean[j] and z[j] <= 0.:
                ocean[j] = True; work[w] = j; w += 1
    return ocean


@njit(cache=True)
def priority_flood(z, ocean, ny, nx, eps):
    """Depression-filled surface; seeds are outlet cells (open ocean and, if used, open edges)."""
    n = ny*nx
    zf = z.copy()
    done = np.zeros(n, np.bool_)
    heap = [(0.0, -1)]
    heap.pop()
    count = 0
    for i in range(n):
        if ocean[i]:
            done[i] = True
            heapq.heappush(heap, (z[i], i))
            count += 1
    if count == 0:  # no ocean: drain through the lowest edge cell
        best = 0
        for i in range(n):
            r, c = i//nx, i % nx
            if (r == 0 or c == 0 or r == ny-1 or c == nx-1) and z[i] < z[best]:
                best = i
        done[best] = True
        heapq.heappush(heap, (z[best], best))
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        r, c = i//nx, i % nx
        for k in range(8):
            rr, cc = r+_OFF[k, 0], c+_OFF[k, 1]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx:
                continue
            j = rr*nx+cc
            if done[j]:
                continue
            done[j] = True
            if zf[j] <= h+eps:
                zf[j] = h+eps
            heapq.heappush(heap, (zf[j], j))
    return zf


@njit(cache=True)
def route(zf, ocean, ny, nx, dx):
    """D8 receivers on the filled surface, topological stack (outlets first) and distances."""
    n = ny*nx
    rec = np.arange(n)
    dist = np.full(n, dx)
    for i in range(n):
        if ocean[i]:
            continue  # open sea: outlet
        r, c = i//nx, i % nx
        best_s = 0.
        for k in range(8):
            rr, cc = r+_OFF[k, 0], c+_OFF[k, 1]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx:
                continue
            j = rr*nx+cc
            d = dx*(1.4142135623730951 if _OFF[k, 0] != 0 and _OFF[k, 1] != 0 else 1.)
            s = (zf[i]-zf[j])/d
            if s > best_s:
                best_s = s; rec[i] = j; dist[i] = d
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
        w = 0; work[0] = o; w = 1
        while w > 0:
            w -= 1; i = work[w]
            stack[top] = i; top += 1
            for q in range(start[i], start[i+1]):
                work[w] = donors[q]; w += 1
    return rec, dist, stack[:top]


@njit(cache=True)
def accumulate(rec, stack, cell_area):
    a = np.full(rec.shape[0], cell_area)
    for k in range(stack.shape[0]-1, -1, -1):
        i = stack[k]; j = rec[i]
        if j != i:
            a[j] += a[i]
    return a


@njit(cache=True)
def erode_and_deposit(z, zf, ocean, rec, dist, stack, area, K, m, G, Sc, dt, cell_area, lake_tol, shelf_depth, iterations=6):
    """Coupled erosion-deposition (Yuan et al. 2019, n = 1), then lake / delta deposition.

    River nodes solve dh/dt = -K A^m S + (G/A) Qs implicitly, where Qs is the net sediment
    flux arriving from upstream; Gauss-Seidel iterations update Qs from the previous estimate.
    G = 0 is pure detachment-limited stream power; G ~ 1 makes large lowland rivers
    transport-limited (they aggrade instead of incising). Cell-to-cell slopes are capped at Sc.
    Sediment reaching a closed depression fills it towards its spill level; sediment reaching
    the sea builds a delta up to shelf_depth below sea level; the rest leaves the window.

    Returns new z, eroded thickness and deposited thickness (both >= 0)."""
    n = z.shape[0]
    lake = (zf-z) > lake_tol
    zn = z.copy()
    qs = np.zeros(n)
    for it in range(iterations):
        # upstream net sediment flux (m^3 per step) from the current estimate, excluding the node itself
        qs[:] = 0.
        for k in range(stack.shape[0]-1, -1, -1):
            i = stack[k]; j = rec[i]
            if j != i:
                qs[j] += max(0., qs[i]+(z[i]-zn[i])*cell_area)
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
            f = K[i]*area[i]**m*dt/dist[i]
            v = (z[i]+G*qs[i]/area[i]+f*base)/(1.+f)
            v = min(v, base+Sc*dist[i])
            zn[i] = v
    eroded = np.maximum(z-zn, 0.)
    deposited = np.maximum(zn-z, 0.)
    out = np.zeros(n)
    for k in range(stack.shape[0]-1, -1, -1):
        i = stack[k]; j = rec[i]
        out[i] += (z[i]-zn[i])*cell_area
        if out[i] > 0.:
            d = 0.
            if lake[i]:
                d = min(out[i], (zf[i]-zn[i])*cell_area)
            zn[i] += d/cell_area; deposited[i] += d/cell_area; out[i] -= d
        else:
            out[i] = 0.
        if j != i:
            out[j] += out[i]
    sea_in = np.where(ocean, out, 0.)  # sediment volume delivered to the sea at each river mouth
    return zn, eroded, deposited, sea_in


@njit(cache=True)
def coast_distance(ocean, ny, nx, dx):
    """Approximate distance (m) from each sea cell to the nearest land cell (two-pass chamfer)."""
    n = ny*nx
    big = 1e15
    d = np.where(ocean, big, 0.)
    diag = dx*1.4142135623730951
    for i in range(n):
        r, c = i//nx, i % nx
        if r > 0:
            d[i] = min(d[i], d[i-nx]+dx)
            if c > 0:
                d[i] = min(d[i], d[i-nx-1]+diag)
            if c < nx-1:
                d[i] = min(d[i], d[i-nx+1]+diag)
        if c > 0:
            d[i] = min(d[i], d[i-1]+dx)
    for i in range(n-1, -1, -1):
        r, c = i//nx, i % nx
        if r < ny-1:
            d[i] = min(d[i], d[i+nx]+dx)
            if c > 0:
                d[i] = min(d[i], d[i+nx-1]+diag)
            if c < nx-1:
                d[i] = min(d[i], d[i+nx+1]+diag)
        if c < nx-1:
            d[i] = min(d[i], d[i+1]+dx)
    return d


@njit(cache=True)
def marine_sediment(z, ocean, sea_in, ny, nx, cell_area, shelf_depth, shelf_gradient, dist_to_land, shelf_width):
    """Sediment entering the sea moves downslope underwater, filling each cell up to a graded
    shelf surface before passing on: deltas prograde, shelves build, straits and forearc seas fill.
    River-mouth cells aggrade to 2 m above sea level (the delta plain, so the coast progrades);
    sea cells within shelf_width of land fill to -(shelf_depth + shelf_gradient * distance to land);
    beyond the shelf break sediment only passes downslope; sediment with nowhere
    lower to go leaves for the deep sea (out of the model)."""
    n = ny*nx
    flux = sea_in.copy()
    dep = np.zeros(n)
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
            top = -1e9   # beyond the shelf break: pass on downslope; sediment leaving the window is lost
        cap = max(0., top-z[i]-dep[i])*cell_area
        d = min(flux[i], cap)
        dep[i] += d/cell_area; flux[i] -= d
        if flux[i] <= 0.:
            continue
        r, c = i//nx, i % nx
        best = -1; best_s = 0.
        for k in range(8):
            rr, cc = r+_OFF[k, 0], c+_OFF[k, 1]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx:
                continue
            j = rr*nx+cc
            if not ocean[j]:
                continue
            s = ((z[i]+dep[i])-(z[j]+dep[j]))/(1.4142135623730951 if _OFF[k, 0] != 0 and _OFF[k, 1] != 0 else 1.)
            if s > best_s:
                best_s = s; best = j
        if best >= 0:
            flux[best] += flux[i]
            heapq.heappush(heap, (-(z[best]+dep[best]), best))
        flux[i] = 0.
    return dep


@njit(cache=True)
def basins(rec, stack):
    """Outlet index for every cell (cells sharing an outlet form one drainage basin)."""
    out = np.arange(rec.shape[0])
    for k in range(stack.shape[0]):
        i = stack[k]; j = rec[i]
        if j != i:
            out[i] = out[j]
    return out


def diffuse(z, land, kd, dt, dx):
    """Explicit diffusion among cells of one domain (land, or sea); zero flux across its boundary and edges."""
    zz = z.reshape(land.shape)
    flux = np.zeros_like(zz)
    for axis in (0, 1):
        d = np.diff(zz, axis=axis)
        both = np.logical_and(np.take(land, range(land.shape[axis]-1), axis), np.take(land, range(1, land.shape[axis]), axis))
        q = kd*d/dx*both
        sl_lo = [slice(None)]*2; sl_hi = [slice(None)]*2
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        flux[tuple(sl_lo)] += q; flux[tuple(sl_hi)] -= q
    return (zz+dt*flux/dx).ravel()


class Flexure:
    """Thin elastic plate deflection under a rock load, by FFT on a zero-padded grid."""

    def __init__(self, ny, nx, dx, te, young=7e10, poisson=.25):
        self.ny, self.nx = ny, nx
        py, px = 2*ny, 2*nx
        ky = 2*np.pi*np.fft.fftfreq(py, dx); kx = 2*np.pi*np.fft.rfftfreq(px, dx)
        k4 = (ky[:, None]**2+kx[None, :]**2)**2
        d = young*te**3/(12*(1-poisson**2))
        self.response = RHO_CRUST*G/(RHO_MANTLE*G+d*k4)
        self.shape = (py, px)
        self.alpha_m = (4*d/(RHO_MANTLE*G))**.25

    def deflection(self, load_thickness):
        pad = np.zeros(self.shape); pad[:self.ny, :self.nx] = load_thickness
        w = np.fft.irfft2(np.fft.rfft2(pad)*self.response, s=self.shape)
        return w[:self.ny, :self.nx]


class Landscape:
    """State and stepping for one regional simulation."""

    def __init__(self, z0, dx, uplift, params=None, erodibility=None, open_sea=None):
        self.p = {**DEFAULTS, **(params or {})}
        self.ny, self.nx = z0.shape
        self.dx = float(dx); self.cell_area = self.dx**2
        self.z = z0.astype(np.float64).ravel().copy()
        # Open-sea seeds: the initial ocean (or a supplied mask); the sea can flood any
        # connected cell that later drops below 0 m, but never a landlocked depression.
        if open_sea is None:
            edge = np.zeros(z0.shape, bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
            open_sea = connected_ocean(self.z, (edge & (z0 <= 0)).ravel(), self.ny, self.nx)
        self.sea_seed = np.asarray(open_sea, bool).ravel()
        edge = np.zeros(z0.shape, bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
        self.edge = edge.ravel()
        self.edge_height = self.z.copy()
        self.uplift = uplift  # callable(t_years) -> (ny, nx) m/yr, or an array
        self.K = (self.p['K']*(np.ones(z0.shape) if erodibility is None else erodibility)).ravel()
        self.load = np.zeros(z0.shape)
        self.flex = Flexure(self.ny, self.nx, self.dx, self.p['Te'], self.p['E_young'], self.p['poisson'])
        self.w = np.zeros(z0.shape)
        self.t = 0.
        self.totals = {'uplift_m3': 0., 'eroded_m3': 0., 'deposited_m3': 0.}
        self._route()

    def _route(self):
        self.ocean = connected_ocean(self.z, self.sea_seed, self.ny, self.nx)
        outlets = self.ocean | self.edge if self.p['open_edges'] else self.ocean
        self.outlet_mask = outlets
        self.zf = priority_flood(self.z, outlets, self.ny, self.nx, 1e-3)
        self.rec, self.dist, self.stack = route(self.zf, outlets, self.ny, self.nx, self.dx)
        self.area = accumulate(self.rec, self.stack, self.cell_area)

    def step(self):
        p = self.p; dt = p['dt']
        u = self.uplift(self.t) if callable(self.uplift) else self.uplift
        du = (u*dt).ravel()
        if p['open_edges']:
            du = np.where(self.edge & ~self.ocean, 0., du)  # edge outlets stay at their height
        self.z += du
        self._route()
        zn, eroded, deposited, sea_in = erode_and_deposit(self.z, self.zf, self.ocean, self.rec, self.dist, self.stack, self.area, self.K,
                                                          p['m'], p['G'], p['Sc'], dt, self.cell_area, p['lake_tolerance'], p['shelf_depth'])
        marine = marine_sediment(zn, self.ocean, sea_in, self.ny, self.nx, self.cell_area, p['shelf_depth'], p['shelf_gradient'],
                                 coast_distance(self.ocean, self.ny, self.nx, self.dx), p['shelf_width'])
        zn = zn+marine; deposited = deposited+marine
        land = (~self.ocean).reshape(self.ny, self.nx)
        zn = diffuse(zn, land, p['Kd'], dt, self.dx)
        before = zn.copy()
        sea = (~land) & (zn.reshape(land.shape) > -1000)   # shelf / shallow sea only; deep basins untouched
        sub = max(1, int(np.ceil(p['Kd_marine']*dt/self.dx**2/.2)))
        for _ in range(sub):
            zn = diffuse(zn, sea, p['Kd_marine'], dt/sub, self.dx)
        moved = zn-before
        deposited = deposited+np.maximum(moved, 0.); eroded = eroded+np.maximum(-moved, 0.)
        self.load += (du-eroded+deposited*RHO_SEDIMENT/RHO_CRUST).reshape(self.ny, self.nx)
        w = self.flex.deflection(self.load)
        self.z = zn-(w-self.w).ravel()
        self.w = w
        self.t += dt
        self.totals['uplift_m3'] += du.sum()*self.cell_area
        self.totals['eroded_m3'] += eroded.sum()*self.cell_area
        self.totals['deposited_m3'] += deposited.sum()*self.cell_area

    def run(self, years, callback=None, every=None):
        steps = int(round(years/self.p['dt']))
        for s in range(steps):
            self.step()
            if callback and every and (s+1) % every == 0:
                callback(self)
        self._route()
        return self

    def outputs(self, river_area_m2=2e10):
        """Elevation, drainage area, rivers, closed basins / lakes and drainage-basin labels."""
        shape = (self.ny, self.nx)
        outlet = basins(self.rec, self.stack)
        z = self.z.reshape(shape)
        land = ~self.ocean.reshape(shape)
        to_sea = self.ocean[outlet].reshape(shape)
        lake_depth = (self.zf-self.z).reshape(shape)
        return {'elevation_m': z.astype(np.float32), 'filled_m': self.zf.reshape(shape).astype(np.float32),
                'drainage_area_m2': self.area.reshape(shape).astype(np.float32),
                'receiver': self.rec.reshape(shape).astype(np.int32), 'outlet': outlet.reshape(shape).astype(np.int32),
                'river': (land & (self.area.reshape(shape) >= river_area_m2)),
                'lake_depth_m': np.where(land, lake_depth, 0).astype(np.float32),
                'drains_to_sea': to_sea & land, 'ocean': ~land, 'deflection_m': self.w.astype(np.float32),
                'rock_load_m': self.load.astype(np.float32)}
