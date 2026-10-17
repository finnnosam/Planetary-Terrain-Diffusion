"""Mantle-frame forcing for the plate layer (docs/planet-layer-plan.md, sect. 3): dynamic topography and hotspots.

The world frame of PlateSim is the mantle frame, so plates drift over these fields and each crust parcel feels them
as it passes (which is what the landscape stage's parcel histories need).

Dynamic topography: a long-wavelength field (spherical-harmonic degrees 1-12, red spectrum) that evolves by blending
independent random fields every `dt_knot_myr`; standard deviation `dyn_std_m`. It is *non-load* vertical motion: it
moves the surface without loading the plate (residual topography on Earth is ~ +-1 km, changing at ~10-100 m/Myr;
Hoggard et al. 2016). Hotspots add a swell (part of the dynamic topography) and volcanic crust to whatever plate
passes over them; a new plume starts with a large igneous province (a burst of crust over a wide disc).
"""
import numpy as np

from planet_layer import fields

PARAMS = {
    'dyn_std_m': 400., 'dyn_lmax': 12, 'dyn_slope': 2.5, 'dt_knot_myr': 40.,
    'hotspots_initial': 10, 'hotspot_birth_per_myr': .05,
    'hotspot_life_myr': (60., 180.),
    'hotspot_flux_km3_myr': (2000., 15000.),      # steady volcanic output (Hawaii ~1e6 km3 over ~70 Myr)
    'hotspot_radius_km': 60., 'hotspot_swell_m': 800., 'hotspot_swell_radius_km': 500.,
    'lip_volume_km3': (5e5, 4e6), 'lip_radius_km': 600., 'lip_duration_myr': 2.,
}


class Mantle:
    def __init__(self, graph, rng, t0, t_end=0., params=None):
        self.g, self.rng = graph, rng
        self.p = {**PARAMS, **(params or {})}
        p = self.p
        self.knots_t = np.arange(t0-p['dt_knot_myr'], t_end+2*p['dt_knot_myr'], p['dt_knot_myr'])
        self.knots = [p['dyn_std_m']*fields.random_field(graph.xyz, p['dyn_lmax'], p['dyn_slope'], rng)
                      for _ in self.knots_t]
        self.hotspots = []
        for _ in range(p['hotspots_initial']):
            self._new_hotspot(t0, lip=False, age=rng.uniform(0, p['hotspot_life_myr'][0]))

    def _new_hotspot(self, t, lip=True, age=0.):
        p, rng = self.p, self.rng
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        life = rng.uniform(*p['hotspot_life_myr'])
        self.hotspots.append({'pos': v, 'start': t-age, 'end': t-age+life,
                              'flux_m3_yr': np.exp(rng.uniform(*np.log(p['hotspot_flux_km3_myr'])))*1e9/1e6,
                              'lip_m3': np.exp(rng.uniform(*np.log(p['lip_volume_km3'])))*1e9 if lip else 0.})

    def advance(self, t, dt):
        """Births and deaths of plumes over (t - dt, t]."""
        p = self.p
        for _ in range(self.rng.poisson(p['hotspot_birth_per_myr']*dt)):
            self._new_hotspot(t)
        self.hotspots = [h for h in self.hotspots if h['end'] > t]

    def _disc(self, pos, radius_m):
        ang = np.arccos(np.clip(self.g.xyz@pos, -1, 1))*self.g.radius
        return np.exp(-.5*(ang/radius_m)**2)

    def topography(self, t):
        """Dynamic topography (m) at time t: blended random fields plus hotspot swells."""
        k = np.clip(np.searchsorted(self.knots_t, t)-1, 0, len(self.knots_t)-2)
        f = np.clip((t-self.knots_t[k])/(self.knots_t[k+1]-self.knots_t[k]), 0, 1)
        f = f*f*(3-2*f)
        out = (1-f)*self.knots[k]+f*self.knots[k+1]
        for h in self.hotspots:
            if h['start'] <= t < h['end']:
                out = out+self.p['hotspot_swell_m']*self._disc(h['pos'], self.p['hotspot_swell_radius_km']*1e3)
        return out

    def volcanic(self, t, dt):
        """Volcanic crust added over (t - dt, t] (m of crust per node): steady plume output and LIP bursts."""
        p, a = self.p, self.g.area
        out = np.zeros(self.g.size)
        for h in self.hotspots:
            if not (h['start'] < t and h['end'] > t-dt):
                continue
            w = self._disc(h['pos'], p['hotspot_radius_km']*1e3)
            out += h['flux_m3_yr']*dt*1e6*w/(w*a).sum()
            if h['lip_m3'] > 0 and t-dt < h['start']+p['lip_duration_myr']:
                wl = self._disc(h['pos'], p['lip_radius_km']*1e3)
                out += h['lip_m3']*min(dt, p['lip_duration_myr'])/p['lip_duration_myr']*wl/(wl*a).sum()
        return out
