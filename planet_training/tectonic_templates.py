"""Regional tectonic templates: initial land/sea and a time-dependent rock-uplift history (m/yr).

These encode *mechanisms*, not finished landforms. Ranges, valleys, forelands, drainage divides
and coastlines come from erosion, sediment and flexure in landscape_evolution.

Every template has two phases:

  inherited (0 - 20 Myr, default on)  1-2 older orogens that rise and then stop and wear down,
                                      usually a rift (subsiding trough with raised shoulders), and
                                      long-wavelength mantle-driven swells and basins (active the whole run). This gives
                                      the young events an aged, varied continent instead of a flat plain.
  young (last 10 Myr)
    collision  continent-continent convergence on a curved suture (Alps type), built from offset
               (en-echelon) segments, each with outer and inner sub-ranges so longitudinal valleys
               can form; usually a younger oblique belt rises across the foreland (Apennine type).
    margin     ocean-continent subduction (California / Andes type): segmented coast range, slowly
               subsiding forearc, segmented tilted arc block with a steep back side, back-arc plateau
               with basin-and-range blocks.

All distances are metres, times years. Every template is seeded; seeds give different geometry
within plausible ranges. These are prototypes for checking that the processes work, not a
plate-tectonic model (see docs/tectonic-prototype.md).
"""
import numpy as np

SCENARIOS = ['collision', 'margin']
OLD_PHASE_YR = 20e6
YOUNG_PHASE_YR = 10e6
# Continental interiors, set by comparing 368 km land-relief statistics with modern Earth (docs/tectonic-prototype.md):
SWELL_RATE_YR = (22.5e-6, 40.5e-6)  # mantle swell / basin uplift-rate amplitude, 22-40 m/Myr (observed dynamic
                                    # topography rates are ~10-100 m/Myr); was 5-9 m/Myr, which let interiors wear flat
STABLE_ERODIBILITY = .15            # old crystalline crust vs young belts (was .3)


def smooth_noise(shape, rng, wavelength_cells):
    # fixed wavelengths in this module are written for the 256-cell, 11.52 km window and scaled by n/256
    """Unit-variance Gaussian random field with a Gaussian spectrum of the given wavelength."""
    ky = np.fft.fftfreq(shape[0]); kx = np.fft.rfftfreq(shape[1])
    k = np.sqrt(ky[:, None]**2+kx[None, :]**2)
    spec = np.exp(-.5*(k*wavelength_cells)**2)
    f = np.fft.irfft2(np.fft.rfft2(rng.normal(size=shape))*spec, s=shape)
    return (f-f.mean())/max(f.std(), 1e-12)


def fractal_noise(shape, rng, beta=2.2):
    ky = np.fft.fftfreq(shape[0]); kx = np.fft.rfftfreq(shape[1])
    k = np.sqrt(ky[:, None]**2+kx[None, :]**2); k[0, 0] = 1
    f = np.fft.irfft2(np.fft.rfft2(rng.normal(size=shape))*k**(-beta/2), s=shape)
    return (f-f.mean())/max(f.std(), 1e-12)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)


class Curve:
    """Parabolic centre line: through `centre` (m), direction `angle`, curvature (1/m), length (m)."""

    def __init__(self, centre, angle, length, curvature, samples=400):
        u = np.array([np.cos(angle), np.sin(angle)]); v = np.array([-u[1], u[0]])
        s = np.linspace(-length/2, length/2, samples)
        self.points = np.asarray(centre)[None]+s[:, None]*u[None]+(curvature*s**2/2)[:, None]*v[None]
        tangent = np.gradient(self.points, axis=0); tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self.normals = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
        self.length = length
        self.angle = angle

    def field(self, x, y):
        """Signed distance (m, + along normal), along-strike fraction (0-1) at grid points."""
        pts = np.stack((x.ravel(), y.ravel()), 1)
        dist = np.empty(len(pts)); side = np.empty(len(pts)); frac = np.empty(len(pts))
        for a in range(0, len(pts), 8192):
            d = pts[a:a+8192, None, :]-self.points[None]
            d2 = (d**2).sum(-1); i = d2.argmin(1); r = np.arange(len(i))
            dist[a:a+8192] = np.sqrt(d2[r, i])
            side[a:a+8192] = np.sign((d[r, i]*self.normals[i]).sum(-1))
            frac[a:a+8192] = i/(len(self.points)-1)
        return (dist*side).reshape(x.shape), frac.reshape(x.shape)


def _taper(frac, end=.12):
    return smoothstep(np.minimum(frac, 1-frac)/end)


def _along(rng, frac, amplitude, knots):
    k = rng.uniform(-1, 1, knots)
    return amplitude*np.interp(frac, np.linspace(0, 1, knots), k)


def belt_field(x, y, curve, rng, u0, sig_front, sig_back, front_sign, zones=True):
    """Segmented orogenic belt (m/yr): 3-5 offset segments with overlapping tips; each segment has
    the main range plus optional outer (foreland-side) and inner sub-ranges."""
    across, frac = curve.field(x, y)
    n_seg = int(rng.integers(3, 6))
    bounds = np.concatenate(([0.], np.sort(rng.uniform(.12, .88, n_seg-1)), [1.]))
    ov = .05
    total = np.zeros_like(across)
    for s in range(n_seg):
        a, b = bounds[s], bounds[s+1]
        w = smoothstep((frac-(a-ov))/(2*ov))*smoothstep(((b+ov)-frac)/(2*ov))
        if s == 0:
            w = smoothstep(((b+ov)-frac)/(2*ov))
        if s == n_seg-1:
            w = smoothstep((frac-(a-ov))/(2*ov))
        acr = across-rng.uniform(-.9, .9)*sig_back
        sig = np.where(acr*front_sign > 0, sig_front, sig_back)
        prof = np.exp(-.5*(acr/sig)**2)
        if zones:
            if rng.uniform() < .7:   # outer sub-range towards the foreland
                off = front_sign*rng.uniform(1.8, 2.4)*sig_front
                prof = prof+rng.uniform(.35, .6)*np.exp(-.5*((acr-off)/(.5*sig_front))**2)
            if rng.uniform() < .6:   # inner sub-range towards the hinterland
                off = -front_sign*rng.uniform(1.4, 1.9)*sig_back
                prof = prof+rng.uniform(.3, .5)*np.exp(-.5*((acr-off)/(.45*sig_back))**2)
        total += w*rng.uniform(.7, 1.15)*prof
    mod = 1+_along(rng, frac, .25, 14)
    return u0*total*_taper(frac)*mod


class Timeline:
    """Rock-uplift history: a sum of (start, end, field) events with 0.5 Myr ramps in and out."""

    def __init__(self, events, ramp=5e5):
        self.events = events; self.ramp = ramp

    def __call__(self, t):
        total = 0.
        for a, b, f in self.events:
            w = min(1., max(0., (t-a)/self.ramp))*min(1., max(0., (b-t)/self.ramp))
            if w > 0:
                total = total+w*f
        return total


def erodibility(rng, young_uplift, n):
    """Relative erodibility: stable (old, crystalline) crust erodes ~7x slower than young belts, with
    patchy hardness at ~140 km and ~460 km so differential erosion leaves uplands and escarpments."""
    belt = smoothstep(np.abs(young_uplift)/.5e-3)
    patches = .6*(.7*smooth_noise((n, n), rng, 12*n/256)+.7*smooth_noise((n, n), rng, 40*n/256))
    return (STABLE_ERODIBILITY+(1-STABLE_ERODIBILITY)*belt)*np.exp(patches)


def grid(n, dx):
    c = (np.arange(n)-(n-1)/2)*dx
    x, y = np.meshgrid(c, c)  # x east, y north (row 0 = south; plot with origin='lower')
    return x, y


def continent(n, rng):
    """Starting continental surface: a 500 m platform with small-scale roughness."""
    return np.clip(500+150*fractal_noise((n, n), rng), 60, None)


def passive_ocean(z, ocean_distance, rng):
    """Ocean depths from distance offshore (m): shelf break ~60-120 km, abyss ~ -4,200 m."""
    d = np.clip(ocean_distance, 0, None)
    shelf = rng.uniform(60e3, 120e3)
    depth = -120-4100*(1-np.exp(-np.clip(d-shelf, 0, None)/150e3))-80*d/shelf*(d < shelf)
    return np.where(ocean_distance > 0, depth, z)


def inherited_events(x, y, rng, span, old_end):
    """Older history: mantle swells/basins (whole run), 1-2 old orogens, usually a rift."""
    n = x.shape[0]
    events = [(-1., np.inf, rng.uniform(*SWELL_RATE_YR)*smooth_noise(x.shape, rng, n/4.5))]
    for _ in range(int(rng.integers(1, 3))):
        c = Curve(rng.uniform(-.35, .35, 2)*span, rng.uniform(0, np.pi), span*rng.uniform(.4, .7), rng.uniform(-1, 1)/span)
        f = belt_field(x, y, c, rng, rng.uniform(3e-3, 5e-3), rng.uniform(30e3, 45e3), rng.uniform(40e3, 60e3), rng.choice([-1, 1]))
        start = rng.uniform(0, 4e6)
        events.append((start, min(start+rng.uniform(8e6, 12e6), old_end-3e6), f))
    if rng.uniform() < .7:
        c = Curve(rng.uniform(-.3, .3, 2)*span, rng.uniform(0, np.pi), span*rng.uniform(.3, .5), rng.uniform(-.5, .5)/span)
        across, frac = c.field(x, y)
        shoulder = rng.uniform(70e3, 90e3)
        rift = _taper(frac, .2)*(-rng.uniform(.25e-3, .45e-3)*np.exp(-.5*(across/rng.uniform(35e3, 50e3))**2)
                                 + rng.uniform(.08e-3, .15e-3)*np.exp(-.5*((np.abs(across)-shoulder)/25e3)**2))
        start = rng.uniform(4e6, 10e6)
        events.append((start, min(start+rng.uniform(5e6, 8e6), old_end), rift))
    return events


def collision(n, dx, seed, inherited=True):
    rng = np.random.default_rng([seed, 1])
    x, y = grid(n, dx)
    span = n*dx
    t0 = OLD_PHASE_YR if inherited else 0.
    z = continent(n, rng)
    coast_y = -span*rng.uniform(.30, .38)+span*.04*smooth_noise((1, n), rng, n/6)[0][None, :]
    z = passive_ocean(z, coast_y-y, rng)
    events = inherited_events(x, y, np.random.default_rng([seed, 11]), span, t0) if inherited else []
    angle = np.radians(rng.uniform(-15, 15))
    curvature = rng.choice([-1, 1])*rng.uniform(.6, 1.4)/span
    belt = Curve((rng.uniform(-.05, .05)*span, span*rng.uniform(.05, .14)), angle, span*rng.uniform(.62, .78), curvature)
    front = -1 if belt.normals[len(belt.normals)//2, 1] > 0 else 1   # sign of 'across' pointing south (foreland)
    u0 = rng.uniform(5.5e-3, 7.5e-3)
    events.append((t0, np.inf, belt_field(x, y, belt, rng, u0, rng.uniform(26e3, 34e3), rng.uniform(40e3, 55e3), front)))
    second_start = None
    if rng.uniform() < .75:
        gap = rng.uniform(260e3, 340e3)
        mid = belt.points[len(belt.points)//2]
        b2 = Curve(np.array([mid[0]+rng.uniform(.0, .12)*span, mid[1]-gap]), angle+np.radians(rng.uniform(25, 45))*rng.choice([-1, 1]),
                   span*rng.uniform(.35, .5), rng.uniform(-.8, .8)/span)
        second_start = t0+rng.uniform(3e6, 5e6)
        events.append((second_start, np.inf, belt_field(x, y, b2, rng, rng.uniform(2.5e-3, 3.5e-3), rng.uniform(20e3, 26e3), rng.uniform(24e3, 32e3), rng.choice([-1, 1]))))
    erod = erodibility(rng, sum(f for a, b, f in events if a >= t0), n)
    meta = {'scenario': 'collision', 'seed': seed, 'inherited': inherited, 'years': t0+YOUNG_PHASE_YR, 'young_start_yr': t0,
            'main_peak_uplift_mm_yr': u0*1e3, 'swell_rate_yr': list(SWELL_RATE_YR), 'stable_erodibility': STABLE_ERODIBILITY, 'second_belt': second_start is not None, 'second_belt_start_yr': second_start,
            'events': len(events)}
    return z, Timeline(events), erod, meta


def margin(n, dx, seed, inherited=True):
    rng = np.random.default_rng([seed, 2])
    x, y = grid(n, dx)
    span = n*dx
    t0 = OLD_PHASE_YR if inherited else 0.
    tilt = np.radians(rng.uniform(-12, 12))
    xr = x*np.cos(tilt)+y*np.sin(tilt)   # across-margin coordinate (east positive)
    yr = -x*np.sin(tilt)+y*np.cos(tilt)  # along-margin coordinate
    coast = -span*rng.uniform(.28, .34)+span*.03*smooth_noise((n, n), rng, n/5)
    inland = xr-coast                    # metres inland from the initial coast
    z = continent(n, rng)
    z = passive_ocean(z, -inland, rng)
    z = np.where(inland < 0, z-1800*np.exp(-.5*((-inland-110e3)/35e3)**2), z)
    events = inherited_events(x, y, np.random.default_rng([seed, 12]), span, t0) if inherited else []
    # inherited events act on the continent only; the ocean side is the subducting plate
    events = [(a, b, f*smoothstep(inland/80e3+.5)) for a, b, f in events]
    along = (yr-yr.min())/(yr.max()-yr.min())
    end = smoothstep(np.minimum(along, 1-along)/.08)
    seg = np.clip(1.3*smooth_noise((n, n), rng, 18*n/256)+.9, 0, 1)
    coast_range = rng.uniform(2.5e-3, 3.5e-3)*seg*np.exp(-.5*((inland-rng.uniform(45e3, 70e3))/rng.uniform(20e3, 28e3))**2)
    forearc = -rng.uniform(.05e-3, .15e-3)*np.exp(-.5*((inland-rng.uniform(150e3, 180e3))/45e3)**2)
    # tilted arc block, segmented: the block's position steps along strike, its height varies
    shift = np.zeros_like(along)
    cuts = np.sort(rng.uniform(.1, .9, int(rng.integers(2, 5))))
    for c in cuts:
        shift += rng.uniform(-30e3, 30e3)*smoothstep((along-c)/.03+.5)
    a0, a1 = rng.uniform(220e3, 250e3), rng.uniform(320e3, 360e3)
    ia = inland-shift
    ramp = np.clip((ia-a0)/(a1-a0), 0, 1)**1.3
    back = np.exp(-np.clip(ia-a1, 0, None)/rng.uniform(18e3, 26e3))
    arc = rng.uniform(7e-3, 9e-3)*ramp*back*(1+_along(rng, along, .3, 12))
    width = rng.uniform(350e3, 550e3)
    beyond = np.clip((ia-a1-40e3)/60e3, 0, 1)*np.clip((a1+40e3+width-ia)/120e3, 0, 1)
    blocks = np.cos(2*np.pi*(ia-a1)/rng.uniform(90e3, 130e3)+1.2*smooth_noise((n, n), rng, 10*n/256))
    backarc = beyond*(rng.uniform(.2e-3, .4e-3)+rng.uniform(.9e-3, 1.3e-3)*blocks*np.clip(smooth_noise((n, n), rng, 14*n/256)+.6, 0, 1))
    events.append((t0, np.inf, (coast_range+forearc+arc+backarc)*end))
    erod = erodibility(rng, events[-1][2], n)
    meta = {'scenario': 'margin', 'seed': seed, 'inherited': inherited, 'years': t0+YOUNG_PHASE_YR, 'young_start_yr': t0,
            'arc_peak_uplift_mm_yr': float(arc.max()*1e3), 'swell_rate_yr': list(SWELL_RATE_YR), 'stable_erodibility': STABLE_ERODIBILITY, 'events': len(events)}
    return z, Timeline(events), erod, meta


def build(scenario, n, dx, seed, inherited=True):
    if scenario == 'collision':
        return collision(n, dx, seed, inherited)
    if scenario == 'margin':
        return margin(n, dx, seed, inherited)
    raise ValueError(f'Unknown scenario {scenario!r}; choose {SCENARIOS}')
