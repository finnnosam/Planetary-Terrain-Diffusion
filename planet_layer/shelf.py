"""Continental shelves for the plate layer (docs/planet-layer-plan.md, step 8 follow-up 1): wave-driven transport in
shallow water and wave attack on the coast, on the 46 km plate grid.

Earth's shelves are flat surfaces at ~50-130 m below sea level with a break to the slope. Two processes make them,
neither of which the landscape kernels have:

* **Wave-base planation.** Waves move sediment on the sea floor where the water is shallower than wave base, and
  the shoreline sweeps across the shelf as sea level rises and falls. Modelled, as in stratigraphic forward models
  (Kaufman et al. 1991; Rivenaes 1992; Granjeon & Joseph 1999), as diffusion with a diffusivity that decays with
  water depth: kappa = kappa_wave * exp(-depth / wave_base), depth taken at the edge midpoint. Highs above wave base
  are planed, hollows filled; the transport fades at the shelf edge.
* **Shelf-edge progradation** follows from the same law: sediment carried across the topset stops where the water
  deepens and kappa collapses, so the shelf edge builds out over the slope as a clinoform. The topset gradient is
  set by supply / kappa (50 m^2/yr per m of coast over kappa 3e4 m^2/yr is ~1.7 m/km, Earth's ~1-2 m/km).
* **Coastal retreat (cliffs).** A land node next to the open sea loses material into it at a cliff retreat rate R:
  volume flux R x (cliff face: land height plus the water in front, down to wave base) per unit coast length,
  deposited at the foot (the sea neighbour). The attack is damped by the
  water in front of the coast: factor 1 - exp(-depth / wave_base) of the sea neighbour, so a coast fronted by a
  wide shallow shelf stops retreating (waves break and lose their energy on the shelf; Anderson et al. 1999,
  Trenhaile 2000), which keeps planation from eating whole continents.

Conservation: every exchange is a flux between two nodes (finite volume over axis neighbours, as
graph_landscape.diffuse). A donor can only give what it has: continental nodes give crust; oceanic nodes give only
their sediment (the oceanic basement is never eroded), so the flux out of a node is scaled to what it holds.
"""
import numpy as np

from planet_training.landscape_evolution import njit


@njit(cache=True)
def _edges(domain, nbr, dist, axis, cell_area):
    """Axis edges with both ends in the domain, each once (i < j): ends, width sqrt(a_i a_j) / d, length d."""
    n = nbr.shape[0]
    cnt = 0
    for i in range(n):
        if not domain[i]:
            continue
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j > i and axis[i, k] and domain[j]:
                cnt += 1
    ei = np.empty(cnt, np.int64); ej = np.empty(cnt, np.int64); ew = np.empty(cnt); ed = np.empty(cnt)
    c = 0
    for i in range(n):
        if not domain[i]:
            continue
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            if j > i and axis[i, k] and domain[j]:
                d = dist[i, k]
                ei[c] = i; ej[c] = j; ed[c] = d; ew[c] = np.sqrt(cell_area[i]*cell_area[j])/d
                c += 1
    return ei, ej, ew, ed


@njit(cache=True)
def _transport(z, avail, sea, ei, ej, ew, ed, cell_area, kw, hwb, rc, dt):
    """Explicit substeps of the wave transport over the edge list. Returns the new z, the new avail and the volume
    moved off land (m^3)."""
    n = z.shape[0]
    m = ei.shape[0]
    # stability: a node's total coefficient sum(kappa_max w / (d a)) times the substep must stay below 0.9
    csum = np.zeros(n)
    for e in range(m):
        i = ei[e]; j = ej[e]
        kmax = kw if (sea[i] and sea[j]) else rc*ed[e]
        c = kmax*ew[e]/ed[e]
        csum[i] += c/cell_area[i]; csum[j] += c/cell_area[j]
    cmax = 0.
    for i in range(n):
        if csum[i] > cmax:
            cmax = csum[i]
    nsub = max(1, int(np.ceil(dt*cmax/.9)))
    h = dt/nsub
    q = np.empty(m)
    out = np.zeros(n)
    off_land = 0.
    for _ in range(nsub):
        for i in range(n):
            out[i] = 0.
        for e in range(m):
            i = ei[e]; j = ej[e]
            if sea[i] and sea[j]:
                depth = -.5*(z[i]+z[j])
                if depth < 0.:
                    depth = 0.
                kap = kw*np.exp(-depth/hwb)
            elif sea[i] or sea[j]:
                s = i if sea[i] else j
                l = j if sea[i] else i
                if z[l] <= z[s]:
                    q[e] = 0.
                    continue
                ds = -z[s]
                if ds < 0.:
                    ds = 0.
                # cliff: retreat R over the face above wave base (the water in front of it, down to wave base)
                face = z[l]+min(ds, hwb)
                f = rc*(1.-np.exp(-ds/hwb))*face*ew[e]*h
                if l == j:
                    f = -f
                q[e] = f
                if f > 0.:
                    out[i] += f
                else:
                    out[j] -= f
                continue
            else:
                q[e] = 0.
                continue
            f = kap*ew[e]*(z[i]-z[j])/ed[e]*h          # m^3 from i to j over the substep (negative: j to i)
            q[e] = f
            if f > 0.:
                out[i] += f
            else:
                out[j] -= f
        for e in range(m):                                # donors give at most what they hold
            f = q[e]
            if f == 0.:
                continue
            i = ei[e]; j = ej[e]
            dn = i if f > 0. else j
            have = avail[dn]*cell_area[dn]
            if out[dn] > have:
                f = f*have/out[dn]
            q[e] = f
        for e in range(m):
            f = q[e]
            if f == 0.:
                continue
            i = ei[e]; j = ej[e]
            z[i] -= f/cell_area[i]; z[j] += f/cell_area[j]
            avail[i] -= f/cell_area[i]; avail[j] += f/cell_area[j]
            if f > 0. and not sea[i]:
                off_land += f
            elif f < 0. and not sea[j]:
                off_land -= f
    return z, avail, off_land


def shelf_step(g, z, ocean, avail, kappa_wave, wave_base, cliff_retreat, dt, max_depth=None):
    """One step (dt years) of wave transport on the graph g.

    z: elevation relative to sea level (m); ocean: connected open sea (bool); avail: erodible thickness per node (m;
    a large number for continental crust, the sediment for ocean floor). The domain is the open sea shallower than
    max_depth (default 12 wave bases: kappa is below 1e-5 kappa_wave there) plus the land nodes touching it.
    Returns (z, avail, volume moved off land in m^3). Arrays are copied."""
    z = np.asarray(z, np.float64).copy()
    avail = np.asarray(avail, np.float64).copy()
    max_depth = 12.*wave_base if max_depth is None else max_depth
    nb = g.nbr
    ok = (nb >= 0) & g.axis
    nbs = np.where(ok, nb, 0)

    def touching(mask):
        return (ok & mask[nbs]).any(1)
    coast = ~ocean & touching(ocean)                     # land facing the open sea
    core = (ocean & (z > -max_depth)) | coast
    if not core.any():
        return z, avail, 0.
    # sea nodes next to the core (deeper water) are included, so sediment can leave the shelf and cliffs can shed
    domain = core | (ocean & touching(core))
    ei, ej, ew, ed = _edges(domain, nb, g.dist, g.axis, g.area)
    return _transport(z, avail, ocean.astype(np.bool_), ei, ej, ew, ed, g.area, float(kappa_wave), float(wave_base),
                      float(cliff_retreat), float(dt))
