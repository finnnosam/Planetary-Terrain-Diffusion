"""Small helpers shared by the planet layer: node locations, random spherical fields, rotations, fast cube lookup."""
import numpy as np

from planet_diffusion import cube
from planet_training.landscape_evolution import njit


def node_loc(xyz):
    """(N, 3) unit vectors -> (N, 2) colatitude, longitude in [0, 2 pi) (ducc0 convention)."""
    x, y, z = np.asarray(xyz, np.float64).T
    return np.ascontiguousarray(np.stack((np.arccos(np.clip(z, -1, 1)), np.mod(np.arctan2(y, x), 2*np.pi)), 1))


def random_field(xyz, lmax, slope, rng):
    """Zero-mean, unit-variance random field with angular power ~ l^-slope (1 <= l <= lmax), at the given points."""
    import ducc0
    alm = np.zeros((lmax+1)*(lmax+2)//2, np.complex128)
    for m in range(lmax+1):
        for l in range(max(m, 1), lmax+1):
            i = m*(2*lmax+1-m)//2+l
            a = rng.normal()*l**(-slope/2)
            alm[i] = a if m == 0 else (a+1j*rng.normal()*l**(-slope/2))/np.sqrt(2)
    f = ducc0.sht.experimental.synthesis_general(alm=alm[None], spin=0, lmax=lmax, loc=node_loc(xyz), epsilon=1e-8)[0]
    return (f-f.mean())/f.std()


def rotation(axis, angle):
    """Rotation matrix about a unit axis (right-handed, radians)."""
    a = np.asarray(axis, np.float64); a = a/np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


_NORMAL, _RIGHT, _DOWN = cube.NORMAL.copy(), cube.RIGHT.copy(), cube.DOWN.copy()
_FACE_POS = np.array([0, 1, 4]); _FACE_NEG = np.array([2, 3, 5])


@njit(cache=True)
def nearest_entries(p, n, normal, right, down, face_pos, face_neg):
    """Shared-layout cube entry of the node nearest (in chart index) to each direction p (M, 3); as cube.coordinates."""
    m = p.shape[0]
    out = np.empty(m, np.int64)
    q = np.pi/4
    for i in range(m):
        ax = 0
        if abs(p[i, 1]) > abs(p[i, ax]):
            ax = 1
        if abs(p[i, 2]) > abs(p[i, ax]):
            ax = 2
        f = face_pos[ax] if p[i, ax] >= 0 else face_neg[ax]
        den = p[i, 0]*normal[f, 0]+p[i, 1]*normal[f, 1]+p[i, 2]*normal[f, 2]
        r = p[i, 0]*right[f, 0]+p[i, 1]*right[f, 1]+p[i, 2]*right[f, 2]
        d = p[i, 0]*down[f, 0]+p[i, 1]*down[f, 1]+p[i, 2]*down[f, 2]
        x = n*(1+np.arctan2(r, den)/q)/2
        y = n*(1+np.arctan2(d, den)/q)/2
        xi = min(max(int(np.floor(x+.5)), 0), n)
        yi = min(max(int(np.floor(y+.5)), 0), n)
        out[i] = (f*(n+1)+yi)*(n+1)+xi
    return out


def nearest_nodes(graph, p):
    """Graph node nearest to each direction p (M, 3), for a planet_layer.sphere_graph.CubeSphere."""
    e = nearest_entries(np.ascontiguousarray(p, np.float64), graph.n, _NORMAL, _RIGHT, _DOWN, _FACE_POS, _FACE_NEG)
    return graph.node_of_entry[e]


def equirect_nodes(graph, height):
    """Node index for every pixel of a (height, 2*height) equirectangular grid (nearest node; categorical-safe)."""
    lat = np.pi/2-np.pi*(np.arange(height)+.5)/height
    lon = -np.pi+2*np.pi*(np.arange(2*height)+.5)/(2*height)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    p = np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), -1).reshape(-1, 3)
    return nearest_nodes(graph, p).reshape(height, 2*height)
