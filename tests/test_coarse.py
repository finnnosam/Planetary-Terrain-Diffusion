"""Independent scheduler trajectories, source parity and spherical tile blending."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from planet_diffusion import cube
from planet_diffusion.coarse import sample_coarse, linear_weight_window
from planet_diffusion.backends import DiagnosticBackend, TerrainBackend, MEANS, STDS


def test_tile_histories_reset_and_completed_outputs_are_repeatable():
    class Backend(DiagnosticBackend):
        def start_coarse(self,steps):
            self.next_step = 0
            self.starts = getattr(self,"starts",0)+1
            return 1.
        def coarse_predict(self,a,cond,step):
            assert step == self.next_step
            assert a.shape == (6,64,64) and cond.shape == (5,64,64)
            return super().coarse_predict(a,cond,step)
        def coarse_advance(self,prediction,state,step):
            self.next_step += 1
            return super().coarse_advance(prediction,state,step)
    backend = Backend()
    raw = cube.conditioning(42,8)
    actual = sample_coarse(backend,42,raw,3,lambda _:None)
    assert backend.starts == 25  # one audit initialization plus 24 full tiles
    np.testing.assert_array_equal(actual,sample_coarse(Backend(),42,raw,3,lambda _:None))
    members,ids,owners,_ = cube.edge_groups(8)
    flat = actual.reshape(6,-1)
    np.testing.assert_array_equal(flat[:,members],flat[:,owners[ids]])


def test_shared_nodes_combine_weighted_sums_before_division():
    # Only the central tile on face zero differs. Its confidence at shared
    # nodes differs from the other overlapping tiles, exposing face averaging.
    n = 8
    a = np.zeros((1,6,n+1,n+1),np.float32)
    def predict(context,f,y,x):
        return np.full_like(context,10 if (f,y,x)==(0,0,0) else 0)
    result = cube.consensus(a,predict,64,48,weight_window=linear_weight_window(),shared_weights=True)
    # Compute the independent numerator/denominator for one shared edge node.
    members,ids,owners,_ = cube.edge_groups(n)
    group = ids[np.where(members==4*(n+1)+n)[0][0]]
    aliases = members[ids==group]
    total = norm = 0.
    weight = linear_weight_window()
    for index in aliases:
        f,rem = divmod(int(index),(n+1)**2)
        yy,xx = divmod(rem,n+1)
        for y in (-48,0):
            for x in (-48,0):
                w = weight[yy-y,xx-x]
                total += w*(10 if (f,y,x)==(0,0,0) else 0)
                norm += w
    np.testing.assert_allclose(result.reshape(1,-1)[0,aliases],total/norm,rtol=1e-6)


def test_single_tile_matches_actual_worldpipeline_method(monkeypatch):
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]/"upstream"
    source = root/"terrain_diffusion/inference/world_pipeline.py"
    if not source.exists():
        pytest.skip("Upstream checkout is unavailable")
    sys.path.insert(0,str(root))
    from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
    # Execute the actual two source functions without WorldPipeline's unrelated
    # optional imports (infinite_tensor, h5py and plotting dependencies).
    tree = ast.parse(source.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                 and node.name in ("_coarse_inference","linear_weight_window")]
    namespace = {"torch":torch,"np":np,"MOCK":False}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),namespace)
    np.testing.assert_array_equal(linear_weight_window(),namespace["linear_weight_window"](64,"cpu",torch.float32).numpy())
    seed,n = 42,64
    raw = cube.conditioning(seed,n)
    noises = {seed+i:cube.noise(seed,i,5 if i==0 else 6,n)[:,0,:64,:64] for i in (0,1)}
    namespace["gaussian_noise_patch"] = lambda base_seed,*args,**kwargs:noises[base_seed].copy()
    def model(x,noise_labels,conditional_inputs):
        return torch.tanh(x[:,:6]*.15+x[:,6:7]*.3+torch.roll(x[:,:6],1,-1)*.1)
    def scheduler():
        return EDMDPMSolverMultistepScheduler(sigma_min=.002,sigma_max=80,sigma_data=.5)
    snr = np.array([.3,.1,1,.1,1],np.float32)
    pipeline = SimpleNamespace(kwargs={"coarse_means":MEANS,"coarse_stds":STDS},
        log_mode="none",_dtype=None,device="cpu",seed=seed,coarse_model=model,
        _conditioning_model_input=lambda *args:torch.from_numpy(raw[:,0,:64,:64].copy()))
    weight = namespace["linear_weight_window"](64,"cpu",torch.float32)
    angle = torch.atan(torch.tensor(snr))
    inputs = [v.view(-1) for v in torch.log(torch.tan(angle)/8)]
    expected = namespace["_coarse_inference"](pipeline,(0,0,0),scheduler(),weight,angle,inputs)
    expected = (expected[:6]/expected[6:]).numpy()
    backend = TerrainBackend.__new__(TerrainBackend)
    backend.torch,backend.device,backend.models = torch,"cpu",{"coarse":model}
    backend.scheduler,backend.cond_snr,backend.means,backend.stds = scheduler(),snr,MEANS,STDS
    monkeypatch.setattr(cube,"consensus",lambda a,predict,**kw:predict(a[:,0,:64,:64],0,0,0))
    actual = sample_coarse(backend,seed,raw,20,lambda _:None)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=.001)
