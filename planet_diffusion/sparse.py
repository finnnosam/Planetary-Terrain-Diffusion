"""Demand-driven coarse and latent generation on a fixed cubed-sphere grid."""
import copy
import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cube, procedural
from .cache import PredictionCache
from .climate import local_baseline
from .region import Field, generate_region

VERSION = "sparse-cube-v1"


class Channels:
    """A shared set of scalar Field views with channel-first patch reads."""
    def __init__(self, n, channels, raw, *, weighted=False, noise=False, tile=32):
        self.n = n
        self.fields = [Field(n, lambda f,y,x,c=c: raw(c,f,y,x), tile=tile,
                             cache_bytes=4*1024**2, weighted=weighted, noise=noise)
                       for c in range(channels)]

    def read(self, f, y, x, nearest=False):
        return np.stack([field.read(f,y,x,nearest) for field in self.fields])

    def sample(self, p, nearest=False):
        return np.stack([field.sample(p,nearest) for field in self.fields])

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
                 calculate, *, include_endpoint=False):
        self.identity = {"generation":identity,"stage":name}
        self.size,self.stride,self.include_endpoint = size,stride,include_endpoint
        self.weight = cube.linear_weight_window(size)
        self.maximum = ((n if include_endpoint else n-1)//stride)*stride
        self.minimum = -((size-1)//stride)*stride
        self.cache = PredictionCache(directory/name,self.identity,
                                     lambda _,f,y,x:calculate(f,y,x))

        @lru_cache(maxsize=48)
        def patch(f,y,x):
            return self.cache.get_or_compute(lambda:None,(channels,size,size),f,y,x)
        self.patch = patch

        def raw(channel,face,y,x):
            y,x = np.broadcast_arrays(y,x)
            total = np.zeros(y.shape,np.float64)
            norm = np.zeros(y.shape,np.float64)
            first_y = max(self.minimum,-((size-1-int(y.min()))//stride)*stride)
            first_x = max(self.minimum,-((size-1-int(x.min()))//stride)*stride)
            last_y = min(self.maximum,(int(y.max())//stride)*stride)
            last_x = min(self.maximum,(int(x.max())//stride)*stride)
            for py in range(first_y,last_y+1,stride):
                for px in range(first_x,last_x+1,stride):
                    use = (y>=py)&(y<py+size)&(x>=px)&(x<px+size)
                    if not use.any():
                        continue
                    dy,dx = y[use]-py,x[use]-px
                    weight = self.weight[dy,dx]
                    total[use] += patch(face,py,px)[channel,dy,dx]*weight
                    norm[use] += weight
            if (norm == 0).any():
                raise RuntimeError(f"Incomplete sparse {name} tile coverage")
            return total,norm

        super().__init__(n,channels,raw,weighted=True,tile=32)


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
                 checkpoint_dir, source=None, identity=None):
        if guide_height < 4 or guide_height % 4 or coarse_steps < 2:
            raise ValueError("Sparse guide height must be divisible by four and >= 4")
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
                    "face_coarse_intervals":self.n,"face_native_intervals":self.n*256,
                    "native_height":guide_height*256,"radius_metres":radius_metres,
                    "units":"m","source_sha256":digest.hexdigest(),
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
        self.coarse = Stage(self.directory,metadata,"sparse-coarse",n,6,64,48,
                            self._coarse_tile)
        self.latent_first = Stage(self.directory,metadata,"sparse-latent-0",n*32,5,64,32,
                                  lambda f,y,x:self._latent_tile(0,f,y,x),include_endpoint=True)
        self.latent = Stage(self.directory,metadata,"sparse-latent-1",n*32,5,64,32,
                            lambda f,y,x:self._latent_tile(1,f,y,x),include_endpoint=True)
        self.climate = SparseClimate(self.coarse)

    def _coarse_tile(self,face,y,x):
        yy,xx = np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:]
        raw = self.guide.read(face,yy,xx,nearest=True)
        b = self.backend
        channels = [0,2,3,4,5]
        scaled = (raw-b.means[channels,None,None])/b.stds[channels,None,None]
        angles = np.arctan(b.cond_snr)[:,None,None]
        cond = (np.cos(angles)*scaled+np.sin(angles)*self.noises[0].read(face,yy,xx,nearest=True)).astype(np.float32)
        state = self.noises[1].read(face,yy,xx,nearest=True)*b.start_coarse(self.steps)
        for step in range(self.steps):
            prediction = b.coarse_predict(state,cond,step)
            state = b.coarse_advance(prediction,state,step)
        result = b.coarse_finish(state)*b.stds[:,None,None]+b.means[:,None,None]
        result[1] = result[0]-result[1]
        return result.astype(np.float32)

    def _latent_tile(self,step,face,y,x):
        t = float(np.arctan(160 if step == 0 else .7))
        yy,xx = np.arange(y,y+64)[:,None],np.arange(x,x+64)[None,:]
        previous = (0 if step == 0 else self.latent_first.read(face,yy,xx,nearest=True))
        innovation = self.noises[5819+step].read(face,yy,xx,nearest=True)
        xt = (np.cos(t)*previous+np.sin(t)*innovation).astype(np.float32)
        cy,cx = y/32-1+np.arange(4),x/32-1+np.arange(4)
        cond = self.coarse.read(face,cy[:,None],cx[None,:])
        prediction = self.backend.predict("latent",xt,cond,t)
        return (np.cos(t)*xt+np.sin(t)*prediction).astype(np.float32)

    def region(self,bounds,width,height,with_climate=False,progress=print):
        elevation,metadata = generate_region(self.backend,self.latent,self.seed,
            bounds,width,height,self.metadata,self.directory,progress)
        return elevation,metadata,(self.climate if with_climate else None)
