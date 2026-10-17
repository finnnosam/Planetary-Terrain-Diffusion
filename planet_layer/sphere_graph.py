"""Node graphs for landscape kernels: the cube sphere (unique nodes) and a flat test window.

Both expose the same arrays, so every kernel in planet_layer.graph_landscape runs on either:

  nbr   (N, 8) int64   neighbour node ids, -1 = none
  dist  (N, 8) float64 great-circle (or planar) distance to each neighbour, m (0 where none)
  axis  (N, 8) bool    neighbour lies along a chart axis (used by finite-volume diffusion)
  area  (N,)   float64 node cell area, m^2 (cells tile the domain exactly)
  edge  (N,)   bool    domain boundary (flat window only; all False on the sphere)

The sphere graph follows planet_diffusion.cube: shared-node equi-angular charts with n intervals per face,
6 n^2 + 2 unique nodes. A node's cell is a quarter of each chart quad touching it, so cells tile the sphere.
Interior nodes have 8 neighbours, edge nodes 8 (from two faces), the 8 cube corners 6 (3 axis, 3 diagonal).
The flat graph lists neighbours in landscape_evolution._OFF order, so results can be compared cell for cell.
"""
from dataclasses import dataclass

import numpy as np

from planet_diffusion import cube

EARTH_RADIUS_M = 6371000.0
OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], np.int64)


@dataclass
class Graph:
    nbr: np.ndarray
    dist: np.ndarray
    axis: np.ndarray
    area: np.ndarray
    edge: np.ndarray

    @property
    def size(self):
        return self.area.shape[0]

    def spacing_min(self):
        """Smallest axis-neighbour distance."""
        return float(self.dist[self.axis & (self.nbr >= 0)].min())

    def diffusion_bound(self):
        """max over nodes of sum_j sqrt(a_i a_j) / (a_i d_ij^2) over axis neighbours (1/m^2). Explicit finite-volume
        diffusion (planet_layer.graph_landscape.diffuse) is stable for kappa dt * bound <= 1; 4/dx^2 on a flat grid."""
        ok = self.axis & (self.nbr >= 0)
        j = np.where(ok, self.nbr, 0)
        d = np.where(ok, self.dist, 1.)
        c = np.where(ok, np.sqrt(self.area[:, None]*self.area[j])/(self.area[:, None]*d*d), 0.)
        return float(c.sum(1).max())


def flat(ny, nx, dx):
    """Regular ny x nx window, row-major, 8 neighbours in landscape_evolution._OFF order."""
    r, c = np.divmod(np.arange(ny*nx), nx)
    rr, cc = r[:, None]+OFFSETS[None, :, 0], c[:, None]+OFFSETS[None, :, 1]
    ok = (rr >= 0) & (cc >= 0) & (rr < ny) & (cc < nx)
    nbr = np.where(ok, rr*nx+cc, -1).astype(np.int64)
    diag = (OFFSETS[:, 0] != 0) & (OFFSETS[:, 1] != 0)
    dist = np.where(ok, np.where(diag, dx*1.4142135623730951, dx)[None, :], 0.)
    axis = ok & ~diag[None, :]
    edge = (r == 0) | (c == 0) | (r == ny-1) | (c == nx-1)
    return Graph(nbr, dist, axis, np.full(ny*nx, float(dx)**2), edge)


def _quad_areas(p):
    """Spherical areas (unit sphere) of the chart quads of one face; p is (n+1, n+1, 3) unit vectors."""
    def tri(a, b, c):
        num = np.abs(np.einsum('...i,...i', a, np.cross(b, c)))
        den = 1+np.einsum('...i,...i', a, b)+np.einsum('...i,...i', b, c)+np.einsum('...i,...i', c, a)
        return 2*np.arctan2(num, den)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    return tri(a, b, c)+tri(a, c, d)


class CubeSphere(Graph):
    """Unique-node graph of the shared-node cube with n intervals per face, radius in metres."""

    def __init__(self, n, radius=EARTH_RADIUS_M):
        self.n, self.radius = int(n), float(radius)
        n = self.n
        side = n+1
        entries = 6*side*side
        members, ids, owners, _ = cube.edge_groups(n)
        canonical = np.arange(entries, dtype=np.int64)
        canonical[members] = owners[ids]
        unique = np.unique(canonical)
        node_of_entry = np.searchsorted(unique, canonical)
        self.node_of_entry = node_of_entry          # shared-layout flat entry -> node id
        self.entry_of_node = unique                 # node id -> owning entry
        N = unique.shape[0]
        assert N == 6*n*n+2, (N, 6*n*n+2)

        yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
        xyz = np.empty((N, 3))
        area = np.zeros(N)
        for f in range(6):
            p = cube.directions(f, yy, xx, n)
            base = f*side*side
            xyz[node_of_entry[base+yy*side+xx].ravel()] = p.reshape(-1, 3)
            q = _quad_areas(p)
            for dy in (0, 1):
                for dx in (0, 1):
                    e = base+(yy[:-1, :-1]+dy)*side+(xx[:-1, :-1]+dx)
                    np.add.at(area, node_of_entry[e].ravel(), q.ravel()/4)
        self.xyz = xyz
        f_, r_ = np.divmod(unique, side*side)
        self.face_yx = np.stack((f_,)+np.divmod(r_, side), 1)   # owning (face, y, x) of each node

        nbr = np.full((N, 8), -1, np.int64)
        axis = np.zeros((N, 8), bool)
        # face-interior nodes: one chart, 8 neighbours in OFFSETS order (written in place, no pair lists)
        iy, ix = yy[1:-1, 1:-1], xx[1:-1, 1:-1]
        for f in range(6):
            base = f*side*side
            ids = node_of_entry[base+iy*side+ix].ravel()
            for k, (oy, ox) in enumerate(OFFSETS):
                nbr[ids, k] = node_of_entry[base+(iy+oy)*side+(ix+ox)].ravel()
                axis[ids, k] = oy == 0 or ox == 0
        # face-edge and corner nodes: union of their neighbours over the 2-3 charts that contain them
        by, bx = np.nonzero((yy == 0) | (yy == n) | (xx == 0) | (xx == n))
        src, dst, isaxis = [], [], []
        for f in range(6):
            base = f*side*side
            for k, (oy, ox) in enumerate(OFFSETS):
                ok = (by+oy >= 0) & (by+oy <= n) & (bx+ox >= 0) & (bx+ox <= n)
                src.append(node_of_entry[base+by[ok]*side+bx[ok]])
                dst.append(node_of_entry[base+(by[ok]+oy)*side+(bx[ok]+ox)])
                isaxis.append(np.full(ok.sum(), oy == 0 or ox == 0))
        src, dst, isaxis = np.concatenate(src), np.concatenate(dst), np.concatenate(isaxis)
        _, first = np.unique(src*N+dst, return_index=True)      # sorted by (src, dst)
        src, dst, isaxis = src[first], dst[first], isaxis[first]
        counts = np.bincount(src, minlength=N)
        if counts.max() > 8:
            raise RuntimeError('more than 8 neighbours')
        start = np.concatenate(([0], np.cumsum(counts)[:-1]))
        slot = np.arange(src.shape[0])-start[src]
        nbr[src, slot] = dst
        axis[src, slot] = isaxis
        dist = np.zeros((N, 8))
        for k in range(8):
            ok = nbr[:, k] >= 0
            a, b = xyz[ok], xyz[nbr[ok, k]]
            dist[ok, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))*self.radius
        super().__init__(nbr, dist, axis, area*self.radius**2, np.zeros(N, bool))

    # ---- field conversion between the shared-node cube layout (C, 6, n+1, n+1) and node vectors ----
    def to_nodes(self, a):
        """(…, 6, n+1, n+1) cube array -> (…, N) node values (owning entry of each shared node)."""
        a = np.asarray(a)
        flat = a.reshape(a.shape[:-3]+(-1,))
        return flat[..., self.entry_of_node]

    def to_cube(self, v):
        """(…, N) node values -> (…, 6, n+1, n+1) shared-node cube array (duplicates identical)."""
        v = np.asarray(v)
        side = self.n+1
        return v[..., self.node_of_entry].reshape(v.shape[:-1]+(6, side, side))
