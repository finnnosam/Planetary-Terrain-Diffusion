"""Exercise the real upstream API with tiny untrained weights, no downloads."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_real_upstream_adapter_contract(tmp_path,device):
    torch = pytest.importorskip("torch")
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    pytest.importorskip("diffusers")
    source = Path(__file__).resolve().parents[1]/"upstream"
    if not source.is_dir():
        pytest.skip("Pinned upstream checkout is not installed")
    sys.path.insert(0, str(source))
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from planet_diffusion.backends import TerrainBackend
    torch.set_num_threads(2)
    torch.manual_seed(123)
    (tmp_path/"config.json").write_text(json.dumps({"latent_compression": 8, "coarse_pooling": 1}))
    specs = [("coarse_model", 11, 6, [["float", 8, .2]]*5),
             ("base_model", 5, 5, [["tensor", 58, 1.]]), ("decoder_model", 5, 1, [])]
    for folder, inputs, outputs, cond in specs:
        model = EDMUnet2D(image_size=64, in_channels=inputs, out_channels=outputs,
            model_channels=8, model_channel_mults=[1], layers_per_block=1,
            midblock_attention=False, attn_resolutions=[], conditional_inputs=cond,
            disable_out_gain=True)
        model.save_pretrained(tmp_path/folder)
    backend = TerrainBackend(source, str(tmp_path), "unused", device,precision='float32')
    assert not backend.models  # Identity/config validation must not load weights.
    noise = np.random.default_rng(42).standard_normal((6,64,64),dtype=np.float32)
    cond = np.zeros((5,64,64),np.float32)
    state = noise*backend.start_coarse(3)
    for step in range(3):
        state = backend.coarse_advance(backend.coarse_predict(state,cond,step),state,step)
    expected = backend.coarse_finish(state)
    # The fast path uploads noise, conditioning and SNR once, regardless of steps.
    uploads = []
    original_tensor = backend.tensor
    def tensor(value):
        uploads.append(np.shape(value))
        return original_tensor(value)
    backend.tensor = tensor
    actual = backend.sample_coarse_tile(noise,cond,3)
    assert uploads == [(6,64,64),(5,64,64),(5,)]
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
    np.testing.assert_array_equal(actual,backend.sample_coarse_tile(noise,cond,3))
    backend.tensor = original_tensor
    assert set(backend.models) == {'coarse'}
    backend.start_coarse(3)
    a = np.ones((6, 64, 64), np.float32)
    cond = np.zeros((5, 64, 64), np.float32)
    for step in range(3):
        pred = backend.coarse_predict(a, cond, step)
        assert pred.shape == a.shape and np.isfinite(pred).all()
        a = backend.coarse_advance(pred, a, step)
        assert np.isfinite(a).all()
    for stage, c, condition in [("latent", 5, np.zeros((6, 4, 4), np.float32)),
                                ("decoder", 1, np.zeros((4, 64, 64), np.float32))]:
        x = np.ones((c, 64, 64), np.float32)
        out = backend.predict(stage, x, condition, .5)
        assert out.shape == x.shape and np.isfinite(out).all()
        np.testing.assert_array_equal(out, backend.predict(stage, x, condition, .5))
    # A cube solver update carries six faces in one shared state/history.
    backend.start_coarse(3)
    state = np.ones((6,6,3,3),np.float32)
    for step in range(3):
        state = backend.coarse_advance(np.zeros_like(state),state,step)
        assert state.shape == (6,6,3,3) and np.isfinite(state).all()
    # A fully cached terrain query validates identity without loading any stage.
    from planet_diffusion.sparse import SparseWorld
    from planet_diffusion.storage import save_state
    from planet_diffusion.query import query_region
    run = tmp_path.parent/(tmp_path.name+'-cached-run')
    checkpoints = tmp_path.parent/(tmp_path.name+'-cached-checkpoints')
    world = SparseWorld(backend,42,4,2,6371000.,checkpoints)
    first,metadata,_ = world.region((-2,-2,2,2),2,2,progress=lambda _:None)
    save_state(run,first,metadata)
    lazy = TerrainBackend(source,str(tmp_path),'unused',device,precision='float32')
    second,_,_ = query_region(lazy,run,(-2,-2,2,2),2,2,checkpoint_dir=checkpoints,progress=lambda _:None)
    assert not lazy.models
    np.testing.assert_array_equal(second,first)
