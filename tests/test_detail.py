"""Source tile geometry, noise preservation, and numerical inference parity."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from planet_diffusion import cube, detail
from planet_diffusion.backends import TerrainBackend


def source_methods():
    torch = pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1]/"upstream/terrain_diffusion/inference/world_pipeline.py"
    if not path.exists():
        pytest.skip("Upstream reference unavailable")
    wanted = {"_latent_inference","_process_latent_conditioning","_decoder_inference"}
    nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name in wanted]
    env = {"torch":torch,"np":np,"MOCK":False,"mp_concat":lambda xs,dim:torch.cat(xs,dim=dim)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),env)
    return env


def test_tile_positions_include_every_intersecting_source_window():
    for size,stride,n in [(64,32,64),(512,384,512),(512,384,768)]:
        positions = list(cube.tile_positions(n,size,stride,True))
        expected = [i*stride for i in range(-10,10) if i*stride <= n and i*stride+size > 0]
        assert positions == expected


def test_decoder_reads_latent_nodes_then_repeats_interior_and_halos():
    n=64
    p = np.stack([cube.directions(f,np.arange(n+1)[:,None],np.arange(n+1)[None,:],n) for f in range(6)])
    latent = cube.identify(np.moveaxis(p,-1,0))
    for face,y,x in [(0,0,0),(4,-384,0),(2,384,384)]:
        actual = detail.decoder_conditioning(latent,face,y,x)
        nodes = cube.read(latent,face,(y//8+np.arange(64))[:,None],(x//8+np.arange(64))[None,:],nearest=True)
        np.testing.assert_array_equal(actual[:,::8,::8],nodes)
        np.testing.assert_array_equal(actual,np.repeat(np.repeat(nodes,8,-2),8,-1))


def test_latent_preserves_nearest_state_and_unfiltered_noise(monkeypatch):
    seen=[]
    class Backend:
        def predict_batch(self,stage,a,cond,t):
            seen.append(a.copy())
            return np.zeros_like(a)
    n=64
    first=cube.noise(10,0,5,n)
    innovation=cube.noise(42,1,5,n)
    monkeypatch.setattr(cube,"noise",lambda *args:innovation)
    def consensus(noise,predict,*args,**kwargs):
        a=cube.read(noise,0,np.arange(-32,32)[:,None],np.arange(64)[None,:],nearest=True)
        kwargs['predict_batch'](a[None],[(0,-32,0)])
        return first
    monkeypatch.setattr(cube,"consensus",consensus)
    detail.sample_latents(Backend(),np.zeros((6,6,3,3),np.float32),42,lambda _:None)
    t=float(np.arctan(.7))
    old=cube.read(first,0,np.arange(-32,32)[:,None],np.arange(64)[None,:],nearest=True)
    noise=cube.read(innovation,0,np.arange(-32,32)[:,None],np.arange(64)[None,:],nearest=True)
    np.testing.assert_array_equal(seen[1][0],(np.cos(t)*old+np.sin(t)*noise).astype(np.float32))


def test_latent_and_decoder_samples_match_source_methods():
    env=source_methods(); torch=env['torch']
    rng=np.random.default_rng(8)
    backend=TerrainBackend.__new__(TerrainBackend)
    backend.torch,backend.device,backend.histogram=torch,'cpu',np.zeros(5,np.float32)
    backend.mp_concat=env['mp_concat']
    captured=[]
    def model(a,noise_labels,conditional_inputs):
        captured.append([x.clone() for x in conditional_inputs])
        return .2*a[:,:5]+.1*torch.roll(a[:,:5],1,-1)
    backend.models={'latent':model}
    pipe=SimpleNamespace(_dtype=None,device='cpu',log_mode='none',seed=42,torch_compile=False,
                         base_model=model,latent_compression=8)
    pipe._process_latent_conditioning=lambda *a,**kw:env['_process_latent_conditioning'](pipe,*a,**kw)
    scheduler=SimpleNamespace(config=SimpleNamespace(sigma_data=.5))
    cond=rng.normal(size=(6,4,4)).astype(np.float32)*10
    previous=rng.normal(size=(5,64,64)).astype(np.float32)
    noise=rng.normal(size=previous.shape).astype(np.float32)
    env['gaussian_noise_patch']=lambda *a,**kw:noise.copy()
    weight=torch.from_numpy(cube.linear_weight_window(64))
    means=torch.tensor([14.99,11.65,15.87,619.26,833.12,69.40,.66])
    stds=torch.tensor([21.72,21.78,10.40,452.29,738.09,34.59,.47])
    for t,sample in [(float(np.arctan(160)),None),(float(np.arctan(.7)),previous)]:
        packed=None if sample is None else [torch.from_numpy(np.concatenate([sample*weight.numpy(),weight.numpy()[None]]))]
        inputs=[torch.from_numpy(np.concatenate([cond,np.ones((1,4,4),np.float32)]))]
        result=env['_latent_inference'](pipe,[(0,0,0)],packed,inputs,t,scheduler,weight,torch.zeros(1,5),means,stds)[0]
        expected=(result[:5]/result[5:]).numpy()
        xt=(np.cos(t)*(0 if sample is None else sample)+np.sin(t)*noise).astype(np.float32)
        actual=np.cos(t)*xt+np.sin(t)*backend.predict('latent',xt,cond,t)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(captured[-1][0].numpy(),captured[-2][0].numpy(),rtol=2e-6,atol=2e-6)
    low=rng.normal(size=(5,64,64)).astype(np.float32)
    noise=rng.normal(size=(1,512,512)).astype(np.float32)
    pipe.decoder_model=lambda a,**kw:.2*a[:,:1]+.1*a[:,1:2]
    backend.models['decoder']=pipe.decoder_model
    weight=torch.from_numpy(cube.linear_weight_window(512))
    packed=torch.from_numpy(np.concatenate([low,np.ones((1,64,64),np.float32)]))
    t=float(np.arctan(160))
    result=env['_decoder_inference'](pipe,(0,0,0),packed,scheduler,weight,[torch.tensor(t)],512,384)
    cond=np.repeat(np.repeat(low[:4],8,-2),8,-1)
    xt=np.sin(t)*noise
    actual=np.cos(t)*xt+np.sin(t)*backend.predict('decoder',xt,cond,t)
    np.testing.assert_allclose(actual,(result[:1]/result[1:]).numpy(),rtol=2e-5,atol=2e-6)
