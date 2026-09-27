"""Source T=2 latent passes and decoder tile conditioning on a shared sphere."""
import numpy as np
from . import cube

VERSION = "source-detail-assembly-v1"
LATENT_SIZE, LATENT_STRIDE = 64, 32
DECODER_SIZE, DECODER_STRIDE = 512, 384
COMPRESSION = 8


def decoder_conditioning(latent, face, y, x, size=DECODER_SIZE):
    """Read completed low-resolution nodes, then nearest-upsample as upstream.

    Nearest cross-face reads retain the globe's detail distribution. Bilinear
    halo transport was tested but reduced residual detail without consistent
    edge improvement. Each sampled node becomes one 8 x 8 block.
    """
    if y % COMPRESSION or x % COMPRESSION or size % COMPRESSION:
        raise ValueError("Decoder tile coordinates and size must align to latent compression")
    cond = cube.read(latent[:4],face,(y//COMPRESSION+np.arange(size//COMPRESSION))[:,None],
                     (x//COMPRESSION+np.arange(size//COMPRESSION))[None,:],nearest=True)
    return np.repeat(np.repeat(cond,COMPRESSION,-2),COMPRESSION,-1)


def sample_latents(backend, coarse, seed, progress=print, record=None, batch_size=1):
    def emit(name,value):
        return record(name,value) if record is not None else value
    n = (coarse.shape[-1]-1)*32
    state = np.zeros((5,6,n+1,n+1),np.float32)
    for step,t in enumerate([float(np.arctan(160)),float(np.arctan(.7))]):
        progress(f"cube latent {step+1}/2")
        innovation = cube.noise(seed,5819+step,5,n)
        emit(f"latent-noisy-{step}",(np.cos(t)*state+np.sin(t)*innovation).astype(np.float32))
        def predict_batch(noise,locations):
            previous = np.stack([cube.read(state,f,np.arange(y,y+64)[:,None],
                                           np.arange(x,x+64)[None,:],nearest=True) for f,y,x in locations])
            xt = (np.cos(t)*previous+np.sin(t)*noise).astype(np.float32)
            cond = np.stack([cube.read(coarse,f,(y/32-1+np.arange(4))[:,None],
                                      (x/32-1+np.arange(4))[None,:]) for f,y,x in locations])
            if hasattr(backend,"predict_batch"):
                prediction = backend.predict_batch("latent",xt,cond,t)
            else:
                prediction = np.stack([backend.predict("latent",a,c,t) for a,c in zip(xt,cond)])
            # Blend completed denoised samples. The next pass reads their
            # normalized shared field, matching WorldPipeline's default T=2.
            return np.cos(t)*xt+np.sin(t)*prediction
        state = cube.consensus(innovation,None,LATENT_SIZE,LATENT_STRIDE,
            predict_batch=predict_batch,batch_size=batch_size,
            weight_window=cube.linear_weight_window(LATENT_SIZE),shared_weights=True,include_endpoint=True,
            progress=lambda d,n:progress(f"cube latent {step+1}/2: {d}/{n} patches"))
        state = emit(f"latent-{step}",state)
    return state
