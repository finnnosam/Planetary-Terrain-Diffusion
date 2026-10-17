"""Flexural isostasy on the sphere: the thin elastic plate response of landscape_evolution.Flexure, per
spherical-harmonic degree (k^2 = l(l+1)/R^2), evaluated directly at graph nodes.

The rock load (m of crust-density rock per node) is integrated onto spherical harmonics with the node cell areas
as quadrature weights (ducc0 adjoint_synthesis_general, a non-uniform transform), multiplied by
    rho_c g / (rho_m g + D k^4),
and synthesised back at the same nodes. There is no intermediate grid, so nothing is interpolated. Degrees above
lmax are dropped; lmax is chosen where the response has fallen below ~0.5% of full compensation (l ~ 5 R / alpha),
so the truncation is invisible. Constant elastic thickness, as in the regional model (variable Te later).

Needs ducc0 (pip install ducc0; wheels for Windows, Python 3.10-3.13).
"""
import os

import numpy as np

from planet_training.landscape_evolution import G, RHO_CRUST, RHO_MANTLE


class SphereFlexure:
    def __init__(self, xyz, cell_area, radius, te, young=7e10, poisson=.25, lmax=None, epsilon=1e-7, nthreads=None,
                 rho_load=RHO_CRUST, rho_mantle=RHO_MANTLE):
        import ducc0
        self.sht = ducc0.sht.experimental
        x, y, z = np.asarray(xyz, np.float64).T
        self.loc = np.ascontiguousarray(np.stack((np.arccos(np.clip(z, -1, 1)), np.mod(np.arctan2(y, x), 2*np.pi)), 1))
        self.weight = np.asarray(cell_area, np.float64)/radius**2      # quadrature weights (unit sphere)
        d = young*te**3/(12*(1-poisson**2))
        self.alpha_m = (4*d/(rho_mantle*G))**.25
        if lmax is None:
            lmax = int(np.ceil(5*radius/self.alpha_m/16)*16)
        self.lmax = int(lmax)
        l = np.arange(self.lmax+1, dtype=np.float64)
        k4 = (l*(l+1)/radius**2)**2
        resp_l = rho_load*G/(rho_mantle*G+d*k4)        # load given in metres of material of density rho_load
        # healpy alm ordering: index(l, m) = m (2 lmax + 1 - m) / 2 + l
        ls = np.concatenate([np.arange(m, self.lmax+1) for m in range(self.lmax+1)])
        self.response = resp_l[ls]
        self.epsilon = float(epsilon)
        self.nthreads = os.cpu_count() if nthreads is None else int(nthreads)

    def deflection(self, load):
        """Downward deflection (m, positive down) of the plate under the rock load (m), at every node."""
        q = np.ascontiguousarray((np.asarray(load, np.float64)*self.weight)[None])
        alm = self.sht.adjoint_synthesis_general(map=q, spin=0, lmax=self.lmax, loc=self.loc, epsilon=self.epsilon,
                                                 nthreads=self.nthreads)
        alm *= self.response[None]
        return self.sht.synthesis_general(alm=alm, spin=0, lmax=self.lmax, loc=self.loc, epsilon=self.epsilon,
                                          nthreads=self.nthreads)[0]

    @classmethod
    def for_graph(cls, graph, te, **kw):
        return cls(graph.xyz, graph.area, graph.radius, te, **kw)
