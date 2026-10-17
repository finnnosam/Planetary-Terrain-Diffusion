"""Coupling the plate layer to the fine landscape stage (docs/planet-layer-plan.md, sect. 4 and step 7).

The landscape stage runs on one fixed cube grid in the *present-day* plate geometry. Every fine node is a crust
parcel; its position at earlier times is found by stepping back through the plate layer's saved rotations, one Myr
at a time (the parcel follows whichever plate owned it at each step). Its forcing at time t is the plate layer's
forcing at the position it occupied then, so rigid plate interiors are exact and mantle-frame fields (swells,
hotspots) are felt as the parcel drifts over them.

Forcing per Myr interval (t-1, t], sampled bilinearly from the 46 km plate fields:
  * load (m of rock, flexurally compensated by the landscape model): the plate layer's tectonic thickening
    (subduction, collision, rifting, volcanism, crust spreading; not its erosion), on continental crust, scaled by
    `density_scale` so that broad isostatic uplift matches between the two models (crust 2,830 vs 2,700 kg/m^3);
  * non-load (m, surface motion): on continents the change of dynamic topography along the parcel path; on ocean
    floor the change of the plate layer's own ocean-floor elevation (thermal subsidence, its sediment).
Initial surface: the plate layer's elevation at the start of the window, at each parcel's position then.
"""
import json
from pathlib import Path

import numpy as np

from planet_diffusion import cube
from planet_layer import fields, graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere


class PlateHistory:
    """Snapshots written by scripts.simulate_plates (t<time>.npz plus t<time>-plates.json)."""

    def __init__(self, snapshot_dir):
        self.dir = Path(snapshot_dir)
        self.times = sorted(float(f.name[1:8]) for f in self.dir.glob('t*.npz'))
        sample = np.load(self._npz(self.times[0]))
        n_nodes = sample['plate_id'].shape[0]
        n = int(round(np.sqrt((n_nodes-2)/6)))
        if 6*n*n+2 != n_nodes:
            raise ValueError('snapshots are not on a cube sphere graph')
        self.g = CubeSphere(n)
        self._cache = {}

    def _npz(self, t):
        return self.dir/f't{t:+07.1f}.npz'

    def has(self, t):
        return self._npz(t).exists()

    def fields(self, t):
        if t not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            z = np.load(self._npz(t))
            plates = json.loads((self.dir/f't{t:+07.1f}-plates.json').read_text())
            self._cache[t] = ({k: z[k] for k in z.files}, {int(r['id']): np.asarray(r['Q']) for r in plates})
        return self._cache[t]

    def sample(self, t, name, p, smooth_km=0.):
        """Bilinear sample of a node field at directions p (M, 3), optionally after conservative smoothing."""
        f, _ = self.fields(t)
        v = np.nan_to_num(np.asarray(f[name], np.float64))
        if smooth_km > 0:
            v = self.smooth(v, smooth_km*1e3)
        return cube.sample(self.g.to_cube(v.astype(np.float32))[None], p)[0]

    def smooth(self, v, length_m):
        """Volume-conserving graph diffusion with kappa*t = length^2 / 2 (Gaussian of std ~length)."""
        g = self.g
        if not hasattr(self, '_bound'):
            self._bound = g.diffusion_bound()
        total = length_m**2/2
        steps = max(1, int(np.ceil(total*self._bound/.9)))
        dom = np.ones(g.size, bool)
        for _ in range(steps):
            v = gl.diffuse(v, dom, total/steps, 1., g.nbr, g.dist, g.axis, g.area)
        return v

    def owner(self, t, p):
        f, _ = self.fields(t)
        return f['plate_id'][fields.nearest_nodes(self.g, p)]


def step_back(hist, t, p):
    """Parcel positions at t-1 from positions p at t (row vectors): p(t-1) = Q_k(t-1) Q_k(t)^T p(t) for the plate k
    that owns the parcel at t; if that plate did not exist at t-1 (it split off during the step), the plate owning the
    position at t-1 is used."""
    _, Qt = hist.fields(t)
    _, Qp = hist.fields(t-1)
    own_t = hist.owner(t, p)
    out = p.copy()
    missing = np.zeros(p.shape[0], bool)
    for k in np.unique(own_t):
        sel = own_t == k
        if k in Qt and k in Qp:
            out[sel] = p[sel]@Qt[k]@Qp[k].T
        else:
            missing |= sel
    if missing.any():
        own_p = hist.owner(t-1, p[missing])
        idx = np.flatnonzero(missing)
        for k in np.unique(own_p):
            sel = idx[own_p == k]
            if k in Qt and k in Qp:
                out[sel] = p[sel]@Qt[k]@Qp[k].T
    return out/np.linalg.norm(out, axis=1, keepdims=True)


def build_forcing(hist, fine, window_myr, out_dir, density_scale=(1-2830./3300.)/(1-2700./3300.), smooth_km=80., progress=print):
    """Write the landscape forcing for the last window_myr Myr to out_dir (memory-mapped float32 arrays):
    load_m[i], surface_m[i] for interval i = (t0+i, t0+i+1], plus z0_m, erodibility, continental, and positions_start.
    Returns a dict of the arrays (memmaps)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    W = int(round(window_myr))
    t_end = 0.
    t0 = t_end-W
    for t in np.arange(t0, t_end+1):
        if not hist.has(float(t)):
            raise FileNotFoundError(f'plate snapshot at {t:+.0f} Myr missing (simulate_plates --save-last >= {W})')
    N = fine.size
    load = np.lib.format.open_memmap(out/'load_m.npy', 'w+', np.float32, (W, N))
    surf = np.lib.format.open_memmap(out/'surface_m.npy', 'w+', np.float32, (W, N))
    p = fine.xyz.copy()
    cont_now = hist.sample(t_end, 'continental', p)
    for i in range(W-1, -1, -1):
        t = t0+i+1                                     # interval (t-1, t]
        q = step_back(hist, float(t), p)
        c = np.clip(hist.sample(float(t), 'continental', p), 0, 1)
        # collision thickening arrives as one-node pulses at the plate grid: spread over a belt core (conservative)
        thick = hist.sample(float(t), 'tectonic_thickening_m', p, smooth_km)
        dyn = hist.sample(float(t), 'dynamic_topography_m', p)-hist.sample(float(t-1), 'dynamic_topography_m', q)
        zoc = hist.sample(float(t), 'elevation_m', p)-hist.sample(float(t-1), 'elevation_m', q)
        load[i] = (c*thick*density_scale).astype(np.float32)
        surf[i] = (c*dyn+(1-c)*zoc).astype(np.float32)
        p = q
        progress(f'  forcing {t-1:+.0f} to {t:+.0f} Myr')
    z0 = hist.sample(float(t0), 'elevation_m', p).astype(np.float32)
    orog = hist.sample(t_end, 'orogeny_age_myr', fine.xyz)
    since = np.where(orog > 0, orog, 1e4)
    np.save(out/'z0_m.npy', z0)
    np.save(out/'orogeny_age_myr.npy', since.astype(np.float32))
    np.save(out/'continental_now.npy', cont_now.astype(np.float32))
    np.save(out/'positions_start.npy', p.astype(np.float32))
    load.flush(); surf.flush()
    meta = {'window_myr': W, 't0_myr': t0, 'fine_n': fine.n, 'coarse_n': hist.g.n, 'density_scale': density_scale,
            'smooth_km': smooth_km}
    (out/'forcing.json').write_text(json.dumps(meta, indent=1))
    return {'load_m': load, 'surface_m': surf, 'z0_m': z0, 'orogeny_age_myr': since, 'continental_now': cont_now, 'meta': meta}


def erodibility(orogeny_age_myr, stable=.15, tau_myr=50.):
    """Relative erodibility from time since the last orogeny, as the test bed: young belts 1, old crust `stable`."""
    return stable+(1-stable)*np.exp(-np.clip(orogeny_age_myr, 0, None)/tau_myr)
