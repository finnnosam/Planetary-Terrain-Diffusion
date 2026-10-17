"""Rule-based structural plans for the frozen continuous renderer (plan source B).

A layout draws a *relief residual* in metres on the 64x64 / 40.32 km grid:
belts are Gaussian ridges along gently curved centre lines, basins and
forelands are negative bands beside them, plateaus are flat-topped highs.
The residual is added to the coarse baseline and converted to the renderer's
four continuous channels with exactly the same function that made its
training labels (terrain_spatial_plan.continuous_fields). So the plan format
is in-distribution even though its arrangement is authored.

Layouts are placed on the guide's own upland: the main belt runs along the
principal axis of baseline elevation above 200 m, with seeded jitter. Ocean
and policy-frozen cells receive no residual.

This is a plan *source*, not a terrain stamp: the renderer still decides the
final 40 km heights under the existing conservation/sign constraints, and meso
adds detail. Sea level stays 0 m.
"""
import numpy as np
import torch

from planet_training.structure_plan_v2 import active_for

LAYOUTS = ['arc-foreland', 'paired-ranges', 'plateau-margin', 'branching', 'basin-range', 'composite']
SPEC = {'version': 'procedural-plan-v1', 'layouts': LAYOUTS, 'grid_cells': 64, 'spacing_km': 40.32,
        'placement': 'main axis = principal axis of (baseline-200 m)+ over land in the central 48x48 cells; centre = that '
                     'centroid +/-3 cells, clamped to cells 16-48; angle +/-20 deg; length 0.7-0.95 x upland extent, 10-40 cells',
        'balance': 'residual minus its eligibility-weighted 161.28 km (4-cell) Gaussian mean, reducing block steps from 645 km conservation',
        'profiles': 'Gaussian cross-sections, smooth along-strike amplitude, tapered ends',
        'conversion': 'baseline + residual -> continuous_fields -> codec (same as training labels)'}


def _smooth_noise(rng, n, scale):
    """1-D smooth random modulation in [-1, 1] over n samples."""
    knots = rng.uniform(-1, 1, max(3, int(n/scale)+2))
    return np.interp(np.linspace(0, len(knots)-1, n), np.arange(len(knots)), knots)


class Curve:
    """Parabolic centre line through `centre`, along unit `axis`, curvature k (1/cells)."""

    def __init__(self, centre, angle, length, curvature, samples=240):
        self.u = np.array([np.cos(angle), np.sin(angle)]); self.v = np.array([-self.u[1], self.u[0]])
        s = np.linspace(-length/2, length/2, samples)
        self.s = s
        self.points = centre[None]+s[:, None]*self.u[None]+(curvature*s**2/2)[:, None]*self.v[None]
        tangent = np.gradient(self.points, axis=0); tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self.normals = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
        self.length = length

    def offset(self, distance):
        c = Curve.__new__(Curve)
        c.u, c.v, c.s, c.length = self.u, self.v, self.s, self.length
        c.points = self.points+distance*self.normals; c.normals = self.normals
        return c

    def field(self, grid):
        """Distance (cells), signed side (+ along normal) and along-strike fraction in [0, 1]."""
        diff = grid[:, None, :]-self.points[None]
        d2 = (diff**2).sum(-1); i = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(grid)), i])
        side = np.sign((diff[np.arange(len(grid)), i]*self.normals[i]).sum(-1))
        return dist, side, i/(len(self.points)-1)


def _band(curve, grid, rng, amplitude, width, taper=.18, width_far=None, along=(0., 1.)):
    """Gaussian band along a curve; optional asymmetric width (front vs back) and partial extent."""
    dist, side, t = curve.field(grid)
    w = np.where(side >= 0, width, width if width_far is None else width_far)
    lo, hi = along
    inside = np.clip((t-lo)/max(hi-lo, 1e-6), 0, 1)
    ends = np.clip(np.minimum(inside, 1-inside)/taper, 0, 1)
    ends = ends*ends*(3-2*ends)*((t >= lo) & (t <= hi))
    mod = 1+.25*_smooth_noise(rng, len(curve.points), 30)[(t*(len(curve.points)-1)).astype(int)]
    return amplitude*mod*ends*np.exp(-.5*(dist/w)**2)


def _basin_range(centre, angle, length, grid, rng):
    r = np.zeros(len(grid))
    count = int(rng.integers(3, 6))
    normal = np.array([-np.sin(angle), np.cos(angle)]); along = np.array([np.cos(angle), np.sin(angle)])
    offsets = np.cumsum(rng.uniform(3.2, 5.0, count)); offsets -= offsets.mean()
    for n, off in enumerate(offsets):
        seg = Curve(centre+off*normal+rng.uniform(-3, 3)*along, angle+np.radians(rng.uniform(-10, 10)),
                    length*rng.uniform(.35, .7), rng.uniform(-.03, .03))
        r += _band(seg, grid, rng, rng.uniform(800, 1500), rng.uniform(.8, 1.2), taper=rng.uniform(.15, .35))
        if n < count-1:
            gap = offsets[n+1]-off
            r += _band(seg.offset(gap/2), grid, rng, -rng.uniform(250, 500), rng.uniform(.8, 1.1), along=(rng.uniform(0, .15), rng.uniform(.85, 1)))
    return r


def upland_frame(base, rng, size=64):
    b = np.asarray(base, np.float64).squeeze()
    w = np.clip(b-200, 0, None)*(b > 0)
    ys, xs = np.mgrid[:size, :size]
    inner = (xs >= size//8) & (xs < size-size//8) & (ys >= size//8) & (ys < size-size//8)
    w = w*inner  # the reviewed / handed-off region, so belts sit on the upland that is actually shown
    if w.sum() < 1e3:
        centre = np.array([size/2, size/2]); angle = rng.uniform(0, np.pi); extent = size*.5
    else:
        centre = np.array([(xs*w).sum(), (ys*w).sum()])/w.sum()
        dx, dy = xs-centre[0], ys-centre[1]
        cov = np.array([[(dx*dx*w).sum(), (dx*dy*w).sum()], [(dx*dy*w).sum(), (dy*dy*w).sum()]])/w.sum()
        lam, vec = np.linalg.eigh(cov)
        angle = float(np.arctan2(vec[1, 1], vec[0, 1])); extent = float(np.sqrt(12*max(lam[1], 1)))
    centre = np.clip(centre+rng.uniform(-3, 3, 2), size//4, size-size//4)
    angle += np.radians(rng.uniform(-20, 20))
    length = float(np.clip(extent*rng.uniform(.7, .95), 10, 40))
    return centre, angle, length


def residual(layout, base, seed, size=64):
    """Relief residual (metres, float32, size x size) for one layout on one baseline."""
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown layout {layout!r}; choose {LAYOUTS}')
    rng = np.random.default_rng([seed, LAYOUTS.index(layout)])
    centre, angle, length = upland_frame(base, rng, size)
    ys, xs = np.mgrid[:size, :size]; grid = np.stack((xs.ravel(), ys.ravel()), 1).astype(float)
    side = rng.choice([-1, 1])
    r = np.zeros(len(grid))
    if layout == 'arc-foreland':
        # Alps + Po type: curved belt, connected foreland basin on its concave side, gentle outer foreland.
        k = side*rng.uniform(.035, .06)
        belt = Curve(centre, angle, length, k)
        r += _band(belt, grid, rng, rng.uniform(1600, 2400), rng.uniform(1.4, 1.9), width_far=rng.uniform(2.0, 2.6))
        r += _band(belt.offset(np.sign(k)*rng.uniform(3.2, 4.0)), grid, rng, -rng.uniform(400, 650), rng.uniform(1.3, 1.7), along=(.1, .9))
        r += _band(belt.offset(-np.sign(k)*rng.uniform(3.5, 4.5)), grid, rng, -rng.uniform(120, 220), 1.8, along=(.15, .85))
    elif layout == 'paired-ranges':
        # California type: primary range, connected trough, lower secondary range.
        k = rng.uniform(-.02, .02)
        primary = Curve(centre, angle, length, k)
        gap = rng.uniform(5.0, 6.5)
        r += _band(primary, grid, rng, rng.uniform(1800, 2500), rng.uniform(1.5, 2.0), width_far=rng.uniform(2.2, 3.0))
        r += _band(primary.offset(side*gap/2), grid, rng, -rng.uniform(450, 750), rng.uniform(1.2, 1.5), along=(.05, .95))
        r += _band(primary.offset(side*gap), grid, rng, rng.uniform(700, 1100), rng.uniform(1.0, 1.3), along=(.1, .9))
    elif layout == 'plateau-margin':
        # Quiet high interior with a marginal range and an outer foreland low.
        plateau = Curve(centre, angle, length*.8, 0.)
        dist, _, t = plateau.field(grid)
        width = rng.uniform(4.5, 6.5)
        ends = np.clip(np.minimum(t, 1-t)/.2, 0, 1)
        r += rng.uniform(600, 1000)*ends*np.exp(-.5*(dist/width)**6)
        margin = plateau.offset(side*width)
        r += _band(margin, grid, rng, rng.uniform(1100, 1600), rng.uniform(1.2, 1.6))
        r += _band(margin.offset(side*rng.uniform(3.0, 3.8)), grid, rng, -rng.uniform(300, 500), 1.4, along=(.1, .9))
    elif layout == 'basin-range':
        # Zagros / Basin-and-Range type: 3-5 short, en-echelon ranges with connected basins between.
        r += _basin_range(centre, angle, length, grid, rng)
    elif layout == 'composite':
        # Main belt (Sierra / Andes type) with a basin-and-range hinterland behind it and a trough in front.
        k = rng.uniform(-.025, .025)
        belt = Curve(centre, angle, length, k)
        r += _band(belt, grid, rng, rng.uniform(1800, 2500), rng.uniform(1.5, 2.0), width_far=rng.uniform(2.0, 2.8))
        r += _band(belt.offset(-side*rng.uniform(3.0, 3.8)), grid, rng, -rng.uniform(400, 700), rng.uniform(1.2, 1.5), along=(.05, .95))
        behind = centre+side*rng.uniform(8, 11)*belt.v
        r += .8*_basin_range(behind, angle+np.radians(rng.uniform(-15, 15)), length*rng.uniform(.6, .85), grid, rng)
    else:  # branching
        main = Curve(centre, angle, length, side*rng.uniform(.0, .03))
        r += _band(main, grid, rng, rng.uniform(1700, 2300), rng.uniform(1.4, 1.8))
        j = int(rng.uniform(.3, .5)*len(main.points))
        b_angle = angle+side*np.radians(rng.uniform(30, 50))
        b_len = length*rng.uniform(.4, .55)
        branch = Curve(main.points[j]+b_len/2*np.array([np.cos(b_angle), np.sin(b_angle)]), b_angle, b_len, 0.)
        r += _band(branch, grid, rng, rng.uniform(900, 1400), rng.uniform(1.1, 1.4), taper=.25)
        mid_angle = (angle+b_angle)/2
        wedge = Curve(main.points[j]+b_len/2*np.array([np.cos(mid_angle), np.sin(mid_angle)]), mid_angle, b_len*.8, 0.)
        r += _band(wedge, grid, rng, -rng.uniform(350, 550), 1.2, taper=.3)
    return r.reshape(size, size).astype(np.float32)


def balance(r, weight, sigma_cells=4.):
    """Remove the eligibility-weighted local mean so regional conservation does not create block steps."""
    from planet_training.terrain_structure import blur
    den = blur(weight, sigma_cells).clamp_min(1e-6)
    return (r-blur(r*weight, sigma_cells)/den)*weight


@torch.no_grad()
def plan(layout, base, codec, seed):
    """Encoded continuous plan (1x4x64x64) and the authored residual, for a baseline tensor (1x1x64x64)."""
    r = torch.from_numpy(residual(layout, base.cpu().numpy(), seed, base.shape[-1])).to(base)[None, None]
    w = active_for(base, codec.policy).to(base)
    r = balance(r*w, w)
    z = (base+r).clamp_min(torch.where(base > 0, torch.ones_like(base), base))
    return codec.encode(z, base), r
