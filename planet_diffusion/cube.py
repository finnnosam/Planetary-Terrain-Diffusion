"""Shared-node equi-angular cubed sphere; poles are ordinary face-interior points.

Fields have shape (channels, 6, N+1, N+1). Model halos cross faces through
3-D directions. Duplicate edge/corner entries identify one physical node.

Node indices are equally spaced in angle along each chart axis, so neighbour
spacing varies by at most sqrt(2) over a face (gnomonic charts vary by about
2.1x, and 4.5x in cell area). The average spacing is unchanged: n intervals
per quarter great circle.
"""
from functools import lru_cache
import numpy as np

NORMAL = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
RIGHT = np.array([[0,1,0],[-1,0,0],[0,-1,0],[1,0,0],[0,1,0],[0,1,0]], float)
DOWN = np.array([[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[1,0,0],[-1,0,0]], float)


QUARTER = np.pi/4


def _angles(index, n):
    """Chart angle of a (possibly fractional or out-of-face) node index."""
    return QUARTER*(2*np.asarray(index, dtype=np.float64)/n-1)


def directions(face, y, x, n, normalize=True):
    """Directions of chart nodes; normalize=False returns unit-cube surface points.

    Inside a face this is normalize(N + tan(a)R + tan(b)D). The equivalent
    cos(a)cos(b)N + sin(a)cos(b)R + cos(a)sin(b)D form also extends halo
    indices beyond an edge: along a chart axis they continue the same great
    circle at the same angular step instead of compressing toward a horizon.
    """
    y, x = np.broadcast_arrays(y, x)
    a, b = _angles(x, n)[..., None], _angles(y, n)[..., None]
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    p = ca*cb*NORMAL[face] + sa*cb*RIGHT[face] + ca*sb*DOWN[face]
    norm = np.linalg.norm(p, axis=-1, keepdims=True)
    degenerate = norm < 1e-12
    if degenerate.any():
        # Only reachable when both extended angles are exactly 90 degrees.
        p = np.where(degenerate, sa*RIGHT[face] + sb*DOWN[face], p)
        norm = np.linalg.norm(p, axis=-1, keepdims=True)
    if normalize:
        return p/norm
    return p/np.max(np.abs(p), axis=-1, keepdims=True)


def chart_indices(p, face, n):
    """Fractional (y, x) of directions p on one face chart; p must face it."""
    den = np.sum(p*NORMAL[face], axis=-1)
    x = n*(1+np.arctan2(np.sum(p*RIGHT[face], axis=-1), den)/QUARTER)/2
    y = n*(1+np.arctan2(np.sum(p*DOWN[face], axis=-1), den)/QUARTER)/2
    return y, x


def coordinates(p, n):
    axis = np.argmax(np.abs(p), axis=-1)
    positive = np.take_along_axis(p, axis[..., None], -1)[..., 0] >= 0
    face = np.where(positive, np.array([0,1,4])[axis], np.array([2,3,5])[axis])
    y, x = chart_indices(p, face, n)
    return face, np.clip(y, 0, n), np.clip(x, 0, n)


def area_weight(y, x, n):
    """Relative solid angle per unit chart-index area (equi-angular Jacobian)."""
    y, x = np.broadcast_arrays(y, x)
    a, b = _angles(x, n), _angles(y, n)
    ca, cb, sb = np.cos(a), np.cos(b), np.sin(b)
    return ca*cb/(cb*cb+ca*ca*sb*sb)**1.5


def _sample_indices(a, face, y, x, nearest=False):
    n = a.shape[-1]-1
    if nearest:
        return a[:, face, np.floor(y+.5).astype(int), np.floor(x+.5).astype(int)]
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    jy, jx = np.minimum(iy+1,n), np.minimum(ix+1,n)
    fy, fx = y-iy, x-ix
    return ((1-fy)*((1-fx)*a[:,face,iy,ix]+fx*a[:,face,iy,jx])
            + fy*((1-fx)*a[:,face,jy,ix]+fx*a[:,face,jy,jx])).astype(np.float32)


def sample(a, p, nearest=False):
    return _sample_indices(a, *coordinates(p, a.shape[-1]-1), nearest=nearest)


def read(a, face, y, x, nearest=False):
    n = a.shape[-1]-1
    y, x = np.broadcast_arrays(y, x)
    if y.min() >= 0 and x.min() >= 0 and y.max() <= n and x.max() <= n:
        return _sample_indices(a, face, y, x, nearest)
    return sample(a, directions(face, y, x, n, normalize=False), nearest)


@lru_cache(maxsize=12)
def edge_groups(n):
    groups = {}
    for f in range(6):
        boundary = {(y,x) for y in (0,n) for x in range(n+1)}
        boundary |= {(y,x) for x in (0,n) for y in range(n+1)}
        for y,x in sorted(boundary):
            # Topological key: the chart warp is odd and shared by all faces,
            # so a physical edge node has the same index lattice as before.
            key = tuple((n*NORMAL[f]+(2*x-n)*RIGHT[f]+(2*y-n)*DOWN[f]).astype(int))
            groups.setdefault(key, []).append(f*(n+1)**2+y*(n+1)+x)
    members, ids, owners, counts = [], [], [], []
    for i, group in enumerate(groups.values()):
        members.extend(group)
        ids.extend([i]*len(group))
        owners.append(group[0])
        counts.append(len(group))
    return tuple(np.array(a, dtype=np.int64) for a in (members, ids, owners, counts))


def identify(a, noise=False):
    a = np.array(a, dtype=np.float32, copy=True)
    c, faces, h, w = a.shape
    if faces != 6 or h != w or h < 3:
        raise ValueError("Expected C x 6 x (N+1) x (N+1), N >= 2")
    members, ids, owners, counts = edge_groups(h-1)
    flat = a.reshape(c, -1)
    if noise:
        # Copy one Gaussian draw per shared node; averaging would reduce variance.
        flat[:, members] = flat[:, owners[ids]]
    else:
        sums = np.zeros((c, len(owners)), np.float64)
        np.add.at(sums, (np.arange(c)[:,None], ids[None,:]), flat[:,members])
        flat[:,members] = (sums/counts)[:,ids]
    return a


def noise(seed, stream, channels, n):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,stream])))
    return identify(rng.standard_normal((channels,6,n+1,n+1), dtype=np.float32), noise=True)


def conditioning(seed, n, frequency_mult=None, drop_water_pct=.5, raw=False):
    """Source procedural channels sampled on one continuous 3-D sphere.

    Radius gives n coarse cells per quarter great circle, the exact spacing
    along every equi-angular chart axis. Neither face edges nor poles split the
    noise field.
    Raw mode is for source-compatible imported conditioning before encoding.
    """
    from . import procedural
    frequency = procedural.DEFAULT_FREQUENCY if frequency_mult is None else frequency_mult
    y, x = np.arange(n+1)[:,None], np.arange(n+1)[None,:]
    out = []
    for f in range(6):
        p = directions(f,y,x,n)
        out.append(procedural.sample_raw(seed, p, 2*n/np.pi, frequency, drop_water_pct))
    values = identify(np.stack(out,axis=1))
    return values if raw else identify(procedural.encode(procedural.finalize(values)))


def linear_weight_window(size):
    """WorldPipeline's float32 linear ramp, including epsilon throughout."""
    mid = (size-1)/2
    w = 1-(1-1e-3)*np.clip(np.abs(np.arange(size,dtype=np.float32)-mid)/mid,0,1)
    return w[:,None]*w[None,:]


def tile_positions(n, size, stride, include_endpoint=False):
    """All stride-aligned tiles intersecting the requested node interval."""
    return range(-((size-1)//stride)*stride,n+int(include_endpoint),stride)


def consensus(a, predict, size, stride, progress=None, batch_size=1, predict_batch=None,
              weight_window=None, shared_weights=False, include_endpoint=False):
    """Blend in stable patch order, optionally evaluating bounded groups together."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    n = a.shape[-1]-1
    positions = list(tile_positions(n,size,stride,include_endpoint))
    w1 = np.maximum(1e-3, 1-np.abs(np.linspace(-1,1,size)))
    weight = w1[:,None]*w1[None,:]
    if weight_window is not None:
        weight = np.asarray(weight_window, dtype=np.float64)
        if weight.shape != (size,size) or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("Weight window must be finite, positive and match the patch size")
    norm = np.zeros((6,n+1,n+1),np.float64)
    total = None
    count, patches = 0, 6*len(positions)**2
    locations = [(f,y,x) for f in range(6) for y in positions for x in positions]
    for start in range(0, patches, batch_size):
        batch_locations = locations[start:start+batch_size]
        contexts = []
        for f,y,x in batch_locations:
            # Nearest shared-node reads preserve Gaussian innovation variance.
            context = read(a,f,np.arange(y,y+size)[:,None],np.arange(x,x+size)[None,:],nearest=True)
            contexts.append(context)
        if predict_batch is None:
            predictions = np.stack([predict(context,*location)
                                    for context,location in zip(contexts,batch_locations)])
        else:
            predictions = np.asarray(predict_batch(np.stack(contexts),batch_locations),dtype=np.float32)
        if (predictions.ndim != 4 or predictions.shape[0] != len(batch_locations)
                or predictions.shape[-2:] != (size,size) or not np.isfinite(predictions).all()):
            raise ValueError("Invalid batched cube predictions")
        for pred,(f,y,x) in zip(predictions,batch_locations):
            pred = np.asarray(pred,dtype=np.float32)
            if total is None:
                total = np.zeros((pred.shape[0],6,n+1,n+1),np.float64)
            y0,y1,x0,x1 = max(0,y),min(n+1,y+size),max(0,x),min(n+1,x+size)
            crop = (...,slice(y0-y,y1-y),slice(x0-x,x1-x))
            weights = weight[crop[-2:]]
            total[:,f,y0:y1,x0:x1] += pred[crop]*weights
            norm[f,y0:y1,x0:x1] += weights
            count += 1
            if progress is not None and (count%16 == 0 or count == patches):
                progress(count,patches)
    if np.any(norm == 0):
        raise RuntimeError("Incomplete cube prediction coverage")
    if shared_weights:
        # Duplicate face entries represent a single node. Combine their weighted
        # numerators and denominators before dividing, not equal face averages.
        members, ids, owners, _ = edge_groups(n)
        for values in (total, norm[None]):
            flat = values.reshape(values.shape[0], -1)
            sums = np.zeros((values.shape[0],len(owners)),np.float64)
            np.add.at(sums,(np.arange(values.shape[0])[:,None],ids[None,:]),flat[:,members])
            flat[:,members] = sums[:,ids]
    return identify(total/norm)


def resize(a, n):
    old_n = a.shape[-1]-1
    coords = np.linspace(0,old_n,n+1)
    return identify(np.stack([read(a,f,coords[:,None],coords[None,:]) for f in range(6)],axis=1))


def filter_axis(a, weights, axis):
    n, radius = a.shape[-1]-1, len(weights)//2
    normal = np.arange(n+1)
    extended = np.arange(-radius,n+radius+1)
    out = np.empty_like(a)
    for f in range(6):
        ys,xs = (extended,normal) if axis == 0 else (normal,extended)
        halo = read(a,f,ys[:,None],xs[None,:])
        value = np.zeros_like(a[:,f],dtype=np.float64)
        for i,weight in enumerate(weights):
            value += weight*(halo[:,i:i+n+1,:] if axis == 0 else halo[:,:,i:i+n+1])
        out[:,f] = value
    return identify(out)


def reconstruct(residual, lowfreq):
    n = residual.shape[-1]-1
    scale = n//(lowfreq.shape[-1]-1)
    provisional = residual+resize(lowfreq,n)
    offsets = np.arange(1-scale,scale)
    weights = (1-np.abs(offsets)/scale)/scale
    reduced = filter_axis(filter_axis(provisional,weights,1),weights,0)[:,:,::scale,::scale]
    offsets = np.arange(-5,6)
    weights = np.exp(-.5*(offsets/5)**2)
    weights /= weights.sum()
    low = filter_axis(filter_axis(reduced,weights,1),weights,0)
    return identify(residual+resize(low,n))


def to_equirectangular(a, height):
    """Reproject a shared sphere field; no output seam blending or pole repair."""
    out = np.empty((a.shape[0],height+1,2*height),np.float32)
    lon = np.linspace(-np.pi,np.pi,2*height,endpoint=False)[None,:]
    for start in range(0,height+1,64):
        rows = np.arange(start,min(height+1,start+64))
        theta = (np.pi*rows/height)[:,None]
        p = np.stack(np.broadcast_arrays(np.sin(theta)*np.cos(lon),np.sin(theta)*np.sin(lon),np.cos(theta)),axis=-1)
        if start == 0:
            p[0] = (0,0,1)
        if rows[-1] == height:
            p[-1] = (0,0,-1)
        out[:,start:start+len(rows)] = sample(a,p)
    return out
