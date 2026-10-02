"""Demand-driven coarse and latent generation on a fixed cubed-sphere grid."""
import copy
import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cube, procedural
from .cache import ArrayCache, PredictionCache
from .climate import local_baseline
from .region import Field, generate_region, _coordinate_groups

VERSION = "sparse-cube-v1"


class Channels:
    """A shared set of scalar Field views with channel-first patch reads."""
    def __init__(self, n, channels, raw, *, weighted=False, noise=False, tile=32):
        self.n = n
        self.fields = [Field(n, lambda f,y,x,c=c: raw(c,f,y,x), tile=tile,
                             cache_bytes=4*1024**2, weighted=weighted, noise=noise)
                       for c in range(channels)]

    def read(self, f, y, x, nearest=False):
        y,x = np.broadcast_arrays(y,x)
        if y.min() < 0 or x.min() < 0 or y.max() > self.n or x.max() > self.n:
            return self.sample(cube.directions(f,y,x,self.n,normalize=False),nearest)
        return self.indices(f,y,x,nearest)

    def sample(self, p, nearest=False):
        return self.indices(*cube.coordinates(p,self.n),nearest)

    def nodes(self, f, y, x):
        f,y,x = np.broadcast_arrays(f,y,x)
        shape = y.shape
        f,y,x = f.ravel(),y.ravel(),x.ravel()
        tile = self.fields[0].tile
        tiles = self.n//tile+1
        result = np.empty((len(self.fields),len(y)),np.float32)
        for key,use in _coordinate_groups((f*tiles+y//tile)*tiles+x//tile):
            face,remainder = divmod(key,tiles*tiles)
            ty,tx = divmod(remainder,tiles)
            dy,dx = y[use]%tile,x[use]%tile
            for channel,field in enumerate(self.fields):
                result[channel,use] = field.block(face,ty,tx)[dy,dx]
        return result.reshape((len(self.fields),*shape))

    def indices(self, f, y, x, nearest=False):
        if nearest:
            return self.nodes(f,np.floor(y+.5).astype(int),np.floor(x+.5).astype(int))
        iy,ix = np.floor(y).astype(int),np.floor(x).astype(int)
        if np.array_equal(y,iy) and np.array_equal(x,ix):
            return self.nodes(f,iy,ix)
        jy,jx = np.minimum(iy+1,self.n),np.minimum(ix+1,self.n)
        fy,fx = y-iy,x-ix
        return ((1-fy)*((1-fx)*self.nodes(f,iy,ix)+fx*self.nodes(f,iy,jx))
                +fy*((1-fx)*self.nodes(f,jy,ix)+fx*self.nodes(f,jy,jx))).astype(np.float32)

    def channel(self, index):
        return self.fields[index]


class Noise(Channels):
    """Tile-seeded Gaussian values with one owner draw per physical cube node."""
    def __init__(self, seed, stream, channels, n):
        @lru_cache(maxsize=128)
        def tile(channel, face, ty, tx):
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, stream, channel, face, ty, tx]))
            return rng.standard_normal((64,64),dtype=np.float32)

        def raw(channel, face, y, x):
            y,x = np.broadcast_arrays(y,x)
            result = np.empty(y.shape,np.float32)
            keys = np.stack([y.ravel()//64,x.ravel()//64],axis=-1)
            for ty,tx in np.unique(keys,axis=0):
                use = (y//64 == ty) & (x//64 == tx)
                result[use] = tile(channel,face,int(ty),int(tx))[y[use]%64,x[use]%64]
            return result

        super().__init__(n,channels,raw,noise=True,tile=64)


class Guide(Channels):
    def __init__(self, seed, n, source, frequency, drop_water):
        options = dict(frequency_mult=frequency,drop_water_pct=drop_water)

        @lru_cache(maxsize=128)
        def rectangle(face, y0, y1, x0, x1):
            return calculate(face,np.arange(y0,y1)[:,None],np.arange(x0,x1)[None,:])

        def calculate(face,y,x):
            if source is not None:
                return source.conditioning_nodes(seed,face,y,x,n,**options)
            p = cube.directions(face,y,x,n)
            return procedural.encode(procedural.finalize(
                procedural.sample_raw(seed,p,2*n/np.pi,**options)))

        def raw(channel,face,y,x):
            y,x = np.broadcast_arrays(y,x)
            regular = (y.ndim == 2 and np.array_equal(y,np.broadcast_to(y[:,:1],y.shape))
                       and np.array_equal(x,np.broadcast_to(x[:1,:],x.shape))
                       and np.array_equal(y[:,0],np.arange(y[0,0],y[-1,0]+1))
                       and np.array_equal(x[0],np.arange(x[0,0],x[0,-1]+1)))
            values = (rectangle(face,int(y[0,0]),int(y[-1,0])+1,int(x[0,0]),int(x[0,-1])+1)
                      if regular else calculate(face,y,x))
            return values[channel]

        super().__init__(n,5,raw,tile=32)


class Stage(Channels):
    """Completed model tiles blended at nodes, including other face owners."""
    def __init__(self, directory, identity, name, n, channels, size, stride,
                 calculate, *, include_endpoint=False, tile=32, calculate_batch=None, batch_size=1,
                 patch_cache_bytes=64*1024**2, blend_cache_bytes=8*1024**2):
        self.patch_memory = ArrayCache(patch_cache_bytes)
        self.blend_memory = ArrayCache(blend_cache_bytes)
        self.identity = {"generation":identity,"stage":name}
        self.size,self.stride,self.include_endpoint = size,stride,include_endpoint
        self.weight = cube.linear_weight_window(size)
        self.maximum = ((n if include_endpoint else n-1)//stride)*stride
        self.minimum = -((size-1)//stride)*stride
        self.cache = PredictionCache(directory/name,self.identity,
                                     lambda _,f,y,x:calculate(f,y,x))

        memory = self.patch_memory
        shape = (channels,size,size)

        def patches(locations):
            values, missing = {}, []
            for location in locations:
                value = memory.get(location)
                if value is not None:
                    values[location] = value
                else:
                    value = self.cache.load(shape,*location)
                    if value is None:
                        missing.append(location)
                    else:
                        values[location] = value
            # Only batch dependencies of this read; never generate speculative tiles.
            for start in range(0,len(missing),batch_size):
                batch = missing[start:start+batch_size]
                predicted = (calculate_batch(batch) if calculate_batch is not None else
                             np.stack([calculate(*location) for location in batch]))
                if np.shape(predicted) != (len(batch),*shape):
                    raise ValueError(f"Invalid sparse {name} batch shape")
                for location,value in zip(batch,predicted):
                    values[location] = self.cache.save(value,shape,*location)
            for location in locations:
                memory.put(location, values[location])
            return values

        self.patch = lambda f,y,x: patches([(f,y,x)])[(f,y,x)]

        def raw(channel,face,y,x):
            # Field views ask for identical blocks/edge coordinates across
            # channels. Keep the shared weighted sums, including the norm, so
            # each view still performs the original shared-node consensus.
            key = (face,y.shape,y.dtype.str,y.tobytes(),x.shape,x.dtype.str,x.tobytes())
            blend = self.blend_memory.get(key)
            if blend is None:
                blend = blend_nodes(face,y,x)
                self.blend_memory.put(key,blend)
            return blend[channel],blend[-1]

        def dependencies(face,y,x):
            y,x = np.broadcast_arrays(y,x)
            first_y = max(self.minimum,-((size-1-int(y.min()))//stride)*stride)
            first_x = max(self.minimum,-((size-1-int(x.min()))//stride)*stride)
            last_y = min(self.maximum,(int(y.max())//stride)*stride)
            last_x = min(self.maximum,(int(x.max())//stride)*stride)
            locations = []
            for py in range(first_y,last_y+1,stride):
                for px in range(first_x,last_x+1,stride):
                    use = (y>=py)&(y<py+size)&(x>=px)&(x<px+size)
                    if use.any():
                        locations.append((face,py,px))
            return locations

        def blend_nodes(face,y,x):
            y,x = np.broadcast_arrays(y,x)
            blend = np.zeros((channels+1,*y.shape),np.float64)
            total, norm = blend[:-1], blend[-1]
            locations = dependencies(face,y,x)
            values = patches(locations)
            for location in locations:
                _,py,px = location
                use = (y>=py)&(y<py+size)&(x>=px)&(x<px+size)
                dy,dx = y[use]-py,x[use]-px
                weight = self.weight[dy,dx]
                total[:,use] += values[location][:,dy,dx]*weight
                norm[use] += weight
            if (norm == 0).any():
                raise RuntimeError(f"Incomplete sparse {name} tile coverage")
            return blend

        def prepare_reads(requests):
            """Batch exact block and shared-edge dependencies of nearest reads."""
            needed = dict()
            blocks = set()
            tiles = n//tile+1
            for face,y,x in requests:
                y,x = np.broadcast_arrays(y,x)
                if y.min() < 0 or x.min() < 0 or y.max() > n or x.max() > n:
                    face,y,x = cube.coordinates(cube.directions(face,y,x,n,normalize=False),n)
                face,y,x = np.broadcast_arrays(face,np.floor(y+.5).astype(int),np.floor(x+.5).astype(int))
                for key,_ in _coordinate_groups((face*tiles+y//tile)*tiles+x//tile):
                    if key in blocks:
                        continue
                    blocks.add(key)
                    f,remainder = divmod(key,tiles*tiles)
                    ty,tx = divmod(remainder,tiles)
                    yy = np.arange(ty*tile,min((ty+1)*tile,n+1))[:,None]
                    xx = np.arange(tx*tile,min((tx+1)*tile,n+1))[None,:]
                    needed.update(dict.fromkeys(dependencies(f,yy,xx)))
                    by,bx = np.broadcast_arrays(yy,xx)
                    edge = (by==0)|(by==n)|(bx==0)|(bx==n)
                    if edge.any():
                        p = cube.directions(f,by[edge],bx[edge],n,normalize=False)
                        for other in range(6):
                            use = np.isclose(p@cube.NORMAL[other],1,atol=1e-12,rtol=0)
                            if use.any():
                                oy,ox = (np.rint(v).astype(int) for v in cube.chart_indices(p[use],other,n))
                                needed.update(dict.fromkeys(dependencies(other,oy,ox)))
            patches(list(needed))

        self.prepare_reads = prepare_reads

        super().__init__(n,channels,raw,weighted=True,tile=tile)


class SparseClimate(Channels):
    """Evaluate the source's 15-cell climate regression near requested nodes."""
    def __init__(self, coarse):
        n = coarse.n

        @lru_cache(maxsize=128)
        def rectangle(face,y0,y1,x0,x1):
            yy = np.arange(y0-7,y1+7)[:,None]
            xx = np.arange(x0-7,x1+7)[None,:]
            patch = coarse.read(face,yy,xx)
            baseline,beta = local_baseline(patch[2],np.maximum(patch[0],0.)**2)
            return np.stack([baseline,patch[3,7:-7,7:-7],
                             patch[4,7:-7,7:-7],patch[5,7:-7,7:-7],beta])

        def raw(channel,face,y,x):
            y,x = np.broadcast_arrays(y,x)
            regular = (y.ndim == 2 and np.array_equal(y,np.broadcast_to(y[:,:1],y.shape))
                       and np.array_equal(x,np.broadcast_to(x[:1,:],x.shape))
                       and np.array_equal(y[:,0],np.arange(y[0,0],y[-1,0]+1))
                       and np.array_equal(x[0],np.arange(x[0,0],x[0,-1]+1)))
            if regular:
                return rectangle(face,int(y[0,0]),int(y[-1,0])+1,
                                 int(x[0,0]),int(x[0,-1])+1)[channel]
            result = np.empty(y.shape,np.float32)
            for index in np.ndindex(y.shape):
                result[index] = rectangle(face,int(y[index]),int(y[index])+1,
                                          int(x[index]),int(x[index])+1)[channel,0,0]
            return result

        super().__init__(n,5,raw,tile=16)


class SparseWorld:
    def __init__(self, backend, seed, guide_height, coarse_steps, radius_metres,
                 checkpoint_dir, source=None, identity=None, latent_batch_size=1,
                 patch_cache_bytes=64*1024**2, blend_cache_bytes=8*1024**2,
                 decoder_cache_bytes=64*1024**2):
        self.decoder_cache_bytes = decoder_cache_bytes
        budgets = dict(patch_cache_bytes=patch_cache_bytes,blend_cache_bytes=blend_cache_bytes)
        if guide_height < 4 or guide_height % 4 or coarse_steps < 2:
            raise ValueError("Sparse guide height must be divisible by four and >= 4")
        if (isinstance(latent_batch_size,bool) or not isinstance(latent_batch_size,(int,np.integer))
                or latent_batch_size < 1):
            raise ValueError("latent_batch_size must be a positive integer")
        if source is not None:
            backend = copy.copy(backend)
            backend.cond_snr = (source.cond_snr.copy() if hasattr(source,"cond_snr")
                                else np.array([source.refinement,.2,1.,.2,1.],np.float32))
        self.backend,self.seed,self.n = backend,seed,guide_height//2
        self.steps,self.directory = coarse_steps,Path(checkpoint_dir)
        frequency,drop_water = procedural.validate(
            getattr(backend,"frequency_mult",procedural.DEFAULT_FREQUENCY),
            getattr(backend,"drop_water_pct",procedural.DEFAULT_DROP_WATER))
        digest = hashlib.sha256()
        root = Path(__file__).parent
        for name in ("sparse.py","region.py","cube.py","backends.py","procedural.py",
                     "climate.py","spherical_raster.py","draft.py","conditioning.py","detail.py"):
            digest.update((root/name).read_bytes())
        for name in ("synthetic_map_stats.json","elevation_reference.npz"):
            digest.update((procedural.DATA/name).read_bytes())
        metadata = {**backend.metadata,"algorithm":VERSION,"seed":seed,
                    "coarse_steps":coarse_steps,"coarse_height":guide_height,
                    "latent_batch_size":int(latent_batch_size),
                    "face_coarse_intervals":self.n,"face_native_intervals":self.n*256,
                    "native_height":guide_height*256,"radius_metres":radius_metres,
                    "units":"m","source_sha256":digest.hexdigest(),
                    "geometry":"six equi-angular charts; poles are regular face interiors",
                    "noise":"tile-seeded per-channel Gaussian, shared cube nodes",
                    "procedural_conditioning":{"frequency_mult":list(frequency),
                                               "drop_water_pct":drop_water,
                                               "pyfastnoiselite":importlib.metadata.version("pyfastnoiselite")}}
        if source is not None:
            metadata["conditioning_noise"] = backend.cond_snr.tolist()
            metadata["conditioning_tiffs" if hasattr(source,"cond_snr") else "draft"] = source.metadata
        self.metadata = metadata
        if identity is not None and identity != metadata:
            raise ValueError("Sparse generation identity does not match saved state")
        self.directory.mkdir(parents=True,exist_ok=True)
        manifest = self.directory/"identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Sparse checkpoints belong to different generation settings")
        else:
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
            temporary.replace(manifest)
        if source is not None:
            if hasattr(source,"cond_snr"):
                source.snapshot(self.directory)
            else:
                (self.directory/"draft.png").write_bytes(source.png_bytes)

        n = self.n
        self.guide = Guide(seed,n,source,frequency,drop_water)
        self.noises = {stream:Noise(seed,stream,channels,size) for stream,channels,size in
                       ((0,5,n),(1,6,n),(5819,5,n*32),(5820,5,n*32))}
        # Latent conditioning needs 4x4 cells. A 32-cell sampling block can
        # trigger unrelated 64-cell coarse trajectories (20 model calls each).
        self.coarse = Stage(self.directory,metadata,"sparse-coarse",n,6,64,48,
                            self._coarse_tile,tile=4,**budgets)
        self.latent_first = Stage(self.directory,metadata,"sparse-latent-0",n*32,5,64,32,
                                  lambda f,y,x:self._latent_tiles(0,[(f,y,x)])[0],include_endpoint=True,
                                  calculate_batch=lambda locations:self._latent_tiles(0,locations),
                                  batch_size=latent_batch_size,**budgets)
        self.latent = Stage(self.directory,metadata,"sparse-latent-1",n*32,5,64,32,
                            lambda f,y,x:self._latent_tiles(1,[(f,y,x)])[0],include_endpoint=True,
                            calculate_batch=lambda locations:self._latent_tiles(1,locations),
                            batch_size=latent_batch_size,**budgets)
        self.climate = SparseClimate(self.coarse)

    def _coarse_tile(self,face,y,x):
        yy,xx = np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:]
        raw = self.guide.read(face,yy,xx,nearest=True)
        b = self.backend
        channels = [0,2,3,4,5]
        scaled = (raw-b.means[channels,None,None])/b.stds[channels,None,None]
        angles = np.arctan(b.cond_snr)[:,None,None]
        cond = (np.cos(angles)*scaled+np.sin(angles)*self.noises[0].read(face,yy,xx,nearest=True)).astype(np.float32)
        noise = self.noises[1].read(face,yy,xx,nearest=True)
        result = b.sample_coarse_tile(noise,cond,self.steps)*b.stds[:,None,None]+b.means[:,None,None]
        result[1] = result[0]-result[1]
        return result.astype(np.float32)

    def _latent_tiles(self,step,locations):
        t = float(np.arctan(160 if step == 0 else .7))
        if step == 1 and len(locations) > 1:
            self.latent_first.prepare_reads([(face,np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:])
                                             for face,y,x in locations])
        samples,conditions = [],[]
        for face,y,x in locations:
            yy,xx = np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:]
            previous = (0 if step == 0 else self.latent_first.read(face,yy,xx,nearest=True))
            innovation = self.noises[5819+step].read(face,yy,xx,nearest=True)
            samples.append((np.cos(t)*previous+np.sin(t)*innovation).astype(np.float32))
            cy,cx = y/32-1+np.arange(4),x/32-1+np.arange(4)
            conditions.append(self.coarse.read(face,cy[:,None],cx[None,:]))
        xt,cond = np.stack(samples),np.stack(conditions)
        prediction = self.backend.predict_batch("latent",xt,cond,t)
        return (np.cos(t)*xt+np.sin(t)*prediction).astype(np.float32)

    def region(self,bounds,width,height,with_climate=False,progress=print):
        elevation,metadata = generate_region(self.backend,self.latent,self.seed,
            bounds,width,height,self.metadata,self.directory,progress,
            decoder_cache_bytes=self.decoder_cache_bytes)
        return elevation,metadata,(self.climate if with_climate else None)
