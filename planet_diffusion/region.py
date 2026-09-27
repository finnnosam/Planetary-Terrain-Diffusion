"""On-demand cube detail, with spherical halos and bounded in-memory tile caches.

Like upstream WorldPipeline, queries pull overlapping decoder patches. Dense
generation can provide global coarse and latent guides; sparse generation
supplies the same interface with guides evaluated only on demand.
"""
from functools import lru_cache
import math
import time
import numpy as np
from . import cube
from .cache import PredictionCache
from .detail import decoder_conditioning, DECODER_SIZE, DECODER_STRIDE


def validate_region(bounds, width, height, native_height=None):
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("Bounds must be finite")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("West and east must be between -180 and 180 degrees")
    if not -90 <= south < north <= 90 or west == east:
        raise ValueError("Require south < north within [-90, 90] and distinct west/east")
    if east < west:
        east += 360
    if east <= west:
        raise ValueError("Longitude span must be positive")
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (width, height)):
        raise ValueError("Output width and height must be positive integers")
    if native_height is not None:
        needed = max(width*180/(east-west), height*180/(north-south))
        if needed > native_height + 1e-8:
            coarse = max(4, 4*math.ceil(needed/1024))
            raise ValueError(f"Regional output exceeds native detail; increase coarse height to at least {coarse}, or reduce output resolution")
    return [west, south, east, north]


class Field:
    """Lazy scalar shared-node field; weighted raw() returns (sum, weight).

    Weighted fields combine all chart contributions before normalization at
    shared nodes, matching dense consensus rather than averaging face means.
    """
    def __init__(self, n, raw, tile=32, noise=False, cache_bytes=16*1024**2, weighted=False):
        self.n, self.raw, self.tile, self.noise = n, raw, tile, noise
        self.weighted = weighted
        # A fixed 128 entries held only 32 KiB at the low-frequency level and
        # thrashed when a latitude scan revisited adjacent cube tiles. Budget
        # each stage in bytes so reconstruction halos survive those scans.
        self.block = lru_cache(maxsize=max(1,cache_bytes//(tile*tile*4)))(self._block)

    def _block(self, face, ty, tx):
        y = np.arange(ty*self.tile, min((ty+1)*self.tile, self.n+1))[:, None]
        x = np.arange(tx*self.tile, min((tx+1)*self.tile, self.n+1))[None, :]
        raw = self.raw(face,y,x)
        values = raw[0]/raw[1] if self.weighted else raw
        value = np.broadcast_to(values, (len(y), x.shape[1])).astype(np.float32).copy()
        yy, xx = np.broadcast_arrays(y, x)
        boundary = (yy == 0) | (yy == self.n) | (xx == 0) | (xx == self.n)
        if boundary.any():
            # All charts representing this exact physical edge/corner participate.
            p = cube.directions(face, yy[boundary], xx[boundary], self.n, normalize=False)
            sums = np.zeros(len(p), np.float64)
            counts = np.zeros(len(p), np.float64)
            for other in range(6):
                den = p @ cube.NORMAL[other]
                use = np.isclose(den, 1, atol=1e-12, rtol=0)
                if self.noise:
                    use &= counts == 0  # One Gaussian draw, never averaged.
                if use.any():
                    oy = np.rint(self.n*(1+p[use] @ cube.DOWN[other])/2).astype(int)
                    ox = np.rint(self.n*(1+p[use] @ cube.RIGHT[other])/2).astype(int)
                    raw = self.raw(other,oy,ox)
                    sums[use] += raw[0] if self.weighted else raw
                    counts[use] += raw[1] if self.weighted else 1
            value[boundary] = sums/counts
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite regional cube field")
        return value

    def nodes(self, f, y, x):
        # Filtering asks for rectangular grids. Slice their cube tiles directly
        # instead of sorting and masking every pixel for every filter tap.
        y, x = np.broadcast_arrays(y, x)
        if (np.ndim(f) == 0 and y.ndim == 2
                and np.array_equal(y, np.broadcast_to(y[:, :1], y.shape))
                and np.array_equal(x, np.broadcast_to(x[:1, :], x.shape))):
            ys, xs = y[:, 0], x[0, :]
            result = np.empty(y.shape, np.float32)
            for ty in np.unique(ys//self.tile):
                rows = np.flatnonzero(ys//self.tile == ty)
                for tx in np.unique(xs//self.tile):
                    cols = np.flatnonzero(xs//self.tile == tx)
                    block = self.block(int(f), int(ty), int(tx))
                    result[np.ix_(rows, cols)] = block[np.ix_(ys[rows]%self.tile, xs[cols]%self.tile)]
            return result
        f, y, x = np.broadcast_arrays(f, y, x)
        shape = y.shape
        f, y, x = f.ravel(), y.ravel(), x.ravel()
        result = np.empty(len(y), np.float32)
        tiles = self.n//self.tile+1
        keys = (f*tiles+y//self.tile)*tiles+x//self.tile
        unique, inverse = np.unique(keys, return_inverse=True)
        for i, key in enumerate(unique):
            face, remainder = divmod(int(key),tiles*tiles)
            ty, tx = divmod(remainder,tiles)
            use = inverse == i
            result[use] = self.block(int(face), int(ty), int(tx))[y[use]%self.tile, x[use]%self.tile]
        return result.reshape(shape)

    def indices(self, f, y, x, nearest=False):
        if nearest:
            return self.nodes(f, np.floor(y+.5).astype(int), np.floor(x+.5).astype(int))
        iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
        if np.array_equal(y,iy) and np.array_equal(x,ix):
            return self.nodes(f,iy,ix)
        jy, jx = np.minimum(iy+1, self.n), np.minimum(ix+1, self.n)
        fy, fx = y-iy, x-ix
        return ((1-fy)*((1-fx)*self.nodes(f, iy, ix)+fx*self.nodes(f, iy, jx))
                + fy*((1-fx)*self.nodes(f, jy, ix)+fx*self.nodes(f, jy, jx))).astype(np.float32)

    def read(self, f, y, x, nearest=False):
        y, x = np.broadcast_arrays(y, x)
        if y.min() >= 0 and x.min() >= 0 and y.max() <= self.n and x.max() <= self.n:
            return self.indices(f, y, x, nearest)
        return self.sample(cube.directions(f, y, x, self.n, normalize=False), nearest)

    def sample(self, p, nearest=False):
        return self.indices(*cube.coordinates(p, self.n), nearest)


def noise_field(seed, n):
    # Fixed global tile coordinates make separate region requests agree exactly.
    @lru_cache(maxsize=64)
    def tile(f, ty, tx):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 6819, f, ty, tx]))
        return rng.standard_normal((64, 64), dtype=np.float32)

    def raw(f, y, x):
        y, x = np.broadcast_arrays(y, x)
        out = np.empty(y.shape, np.float32)
        tiles = n//64+1
        keys = (y//64)*tiles+x//64
        for key in np.unique(keys):
            ty, tx = divmod(int(key),tiles)
            use = keys == key
            out[use] = tile(f, int(ty), int(tx))[y[use]%64, x[use]%64]
        return out
    return Field(n, raw, tile=64, noise=True)


def filtered(field, weights, axis, output_n=None):
    n = field.n if output_n is None else output_n
    scale = field.n//n
    def raw(f, y, x):
        y, x = y*scale, x*scale
        out = np.zeros(np.broadcast_shapes(y.shape, x.shape), np.float64)
        radius = len(weights)//2
        if y.ndim == x.ndim == 2 and y.shape[1] == x.shape[0] == 1:
            # Fetch the rectangular halo once, then apply the separable filter
            # with strided views. Edge/corner point queries use the path below.
            if axis == 0:
                extended = np.arange(y[0,0]-radius,y[-1,0]+radius+1)[:,None]
                halo = field.read(f,extended,x)
                for i,w in enumerate(weights):
                    out += w*halo[i:i+len(y)*scale:scale,:]
            else:
                extended = np.arange(x[0,0]-radius,x[0,-1]+radius+1)[None,:]
                halo = field.read(f,y,extended)
                for i,w in enumerate(weights):
                    out += w*halo[:,i:i+x.shape[1]*scale:scale]
            return out.astype(np.float32)
        for i, w in enumerate(weights):
            offset = i-len(weights)//2
            out += w*field.read(f, y+offset if axis == 0 else y, x+offset if axis == 1 else x)
        return out.astype(np.float32)
    return Field(n, raw, tile=8 if output_n else field.tile)


def reconstruct(residual, lowfreq):
    n, nl = residual.n, (lowfreq.n if hasattr(lowfreq,"read") else lowfreq.shape[-1]-1)
    scale = n//nl
    def low_read(f,y,x):
        return (lowfreq.read(f,y,x) if hasattr(lowfreq,"read") else
                cube.read(lowfreq,f,y,x)[0])
    provisional = Field(n, lambda f,y,x: residual.read(f,y,x)
                        + low_read(f,y/scale,x/scale))
    offsets = np.arange(1-scale, scale)
    weights = (1-np.abs(offsets)/scale)/scale
    reduced = filtered(filtered(provisional, weights, 1), weights, 0, nl)
    offsets = np.arange(-5, 6)
    weights = np.exp(-.5*(offsets/5)**2)
    weights /= weights.sum()
    low = filtered(filtered(reduced, weights, 1), weights, 0)
    return Field(n, lambda f,y,x: residual.read(f,y,x)+low.read(f,y/scale,x/scale))


def generate_region(backend, latent, seed, bounds, width, height, metadata, directory, progress):
    started = time.perf_counter()
    n = metadata['face_native_intervals']
    noise = noise_field(seed, n)
    t = float(np.arctan(160))
    seen = set()
    model_calls, model_seconds = 0, 0.

    def predict(a, f, y, x):
        nonlocal model_calls, model_seconds
        cond = decoder_conditioning(latent,f,y,x)
        before = time.perf_counter()
        prediction = backend.predict('decoder', a, cond, t)
        model_seconds += time.perf_counter()-before
        model_calls += 1
        return prediction
    if directory is not None:
        import hashlib
        latent_identity = (latent.identity if hasattr(latent,"identity") else
                           hashlib.sha256(latent.tobytes()).hexdigest())
        predict = PredictionCache(directory/'regional-decoder',
            {'generation':metadata, 'latent_sha256':latent_identity}, predict)

    @lru_cache(maxsize=64)
    def patch(f, y, x):
        def make_input():
            return noise.read(f, np.arange(y,y+512)[:,None], np.arange(x,x+512)[None,:], nearest=True)[None]*np.sin(t)
        if isinstance(predict, PredictionCache):
            prediction = predict.get_or_compute(make_input, (1,512,512), f,y,x)[0]
        else:
            prediction = predict(make_input(), f,y,x)[0]
        if (f,y,x) not in seen:
            seen.add((f,y,x))
            if len(seen) == 1 or len(seen)%16 == 0:
                progress(f"regional decoder: {len(seen)} unique patches used (global would use {6*len(cube.tile_positions(n,DECODER_SIZE,DECODER_STRIDE,True))**2})")
        return prediction

    weights = cube.linear_weight_window(DECODER_SIZE)
    def residual_raw(f, y, x):
        y, x = np.broadcast_arrays(y, x)
        total, norm = np.zeros(y.shape, np.float64), np.zeros(y.shape, np.float64)
        stride = DECODER_STRIDE
        for by, bx in np.unique(np.stack([y.ravel()//stride, x.ravel()//stride], axis=1), axis=0):
            group = (y//stride == by) & (x//stride == bx)
            for py in (int(by-1)*stride, int(by)*stride):
                for px in (int(bx-1)*stride, int(bx)*stride):
                    if py > n or px > n:
                        continue
                    use = group & (y >= py) & (y < py+DECODER_SIZE) & (x >= px) & (x < px+DECODER_SIZE)
                    if not use.any():
                        continue
                    dy, dx = y[use]-py, x[use]-px
                    weight = weights[dy,dx]
                    total[use] += patch(f,py,px)[dy,dx]*weight
                    norm[use] += weight
        value = np.cos(t)*np.sin(t)*noise.read(f,y,x)*norm+np.sin(t)*total
        return value*backend.residual_std+backend.residual_mean*norm, norm

    if hasattr(latent,"channel"):
        lowfreq = Field(latent.n,lambda f,y,x:latent.channel(4).read(f,y,x)*38.6-31.4)
    else:
        lowfreq = latent[4:5]*38.6-31.4
    elevation = reconstruct(Field(n, residual_raw,weighted=True), lowfreq)
    # Square nodes before interpolation, as in the global generator.
    square = Field(n, lambda f,y,x: np.sign(z := elevation.read(f,y,x))*z*z)
    west, south, east, north = bounds
    out = np.empty((height,width), np.float32)
    # Small blocks keep dependencies local, and tile caches bounded.
    for row in range(0,height,32):
        for col in range(0,width,128):
            lon = np.deg2rad(west+(np.arange(col,min(col+128,width))+.5)*(east-west)/width)[None,:]
            lat = np.deg2rad(north-(np.arange(row,min(row+32,height))+.5)*(north-south)/height)[:,None]
            p = np.stack(np.broadcast_arrays(np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
            out[row:row+len(lat),col:col+lon.shape[1]] = square.sample(p)
        progress(f"regional export: {min(row+32,height)}/{height} rows")
    timing = {'seconds':time.perf_counter()-started, 'decoder_model_seconds':model_seconds,
              'decoder_model_calls':model_calls,
              'prediction_disk_reads':predict.hits if isinstance(predict,PredictionCache) else 0}
    progress(f"Regional detail finished in {timing['seconds']:.1f}s: {len(seen)} unique patches, "
             f"{model_calls} decoder evaluations, {timing['prediction_disk_reads']} saved predictions loaded")
    metadata = {**metadata, 'coverage':'region', 'bounds':bounds, 'output_width':width,
                'output_height':height, 'decoder_patches':len(seen),
                'regional_execution':timing,
                'detail_noise':'cube-tile-seeded-v1', 'state_grid':'regional pixel centres'}
    return out, metadata
