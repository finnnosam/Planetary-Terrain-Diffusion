"""Scientific-data alignment, spherical topology and training safety checks."""
import json
import numpy as np
import pytest
import torch

from planet_diffusion import cube
from planet_training.data import (PaleoRaster, age_split, geographic_split,
                                  bridge_patch, layout_target, SPACING_M, FACTOR)
from planet_training.layout import SphereGrid, LayoutDenoiser, layout_loss
from planet_training.prepare import prepare_bridge
from planet_training.train import (BridgeData, LayoutData, padded_target,
                                   main as train_main, sample_layout)
from planet_training.data import VERSION
from toy_meso.sphere import decode, random_frames


def test_paleo_point_coordinates_and_dateline_are_not_pixel_shifted():
    lat = np.arange(90, -91, -1)
    lon = np.arange(-180, 181)
    a = lat[:, None]*10 + np.cos(np.radians(lon))[None, :]*100
    raster = PaleoRaster(a)
    np.testing.assert_allclose(raster.sample([90, -90, 0], [0, 0, 180]), [1000, -800, -100], atol=1e-4)
    np.testing.assert_allclose(raster.sample(25.5, 179.9), raster.sample(25.5, -180.1), atol=1e-4)
    with pytest.raises(ValueError, match="endpoints"):
        a[:, -1] += 10
        PaleoRaster(a)


def test_time_splits_keep_validation_neighbors_out_of_training():
    for age in (0, 5, 20):
        assert age_split(age) == "test"
    for age in (100, 120, 140, 280, 320):
        assert age_split(age) == "val"
    for age in (25, 35, 80, 95, 145, 160, 260, 275, 325, 340):
        assert age_split(age) == "buffer"
    for age in (45, 175, 240, 385.2, 540):
        assert age_split(age) == "train"


def test_geographic_split_checks_all_condition_footprint_not_just_centre():
    def p(lat, lon):
        la, lo = np.radians([lat, lon])
        return [np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
    inside, outside = p(55, 20), p(-45, 110)
    assert geographic_split(np.array([inside, inside])) == "val"
    assert geographic_split(np.array([outside, outside])) == "train"
    assert geographic_split(np.array([inside, outside])) is None


def test_weighted_averaging_preserves_constant_metres_at_both_scales():
    class Constant:
        def sample(self, lat, lon):
            return np.full(np.shape(lat), -1234., np.float32)
    frames = random_frames(np.random.default_rng(0), 1)
    frame = tuple(f[0] for f in frames)
    target, cond = bridge_patch(Constant(), frame, size=56, sub=3)
    np.testing.assert_allclose(decode(target.astype(np.float32)), -1234, atol=1)
    np.testing.assert_allclose(decode(cond), -1234, atol=.001)
    assert target.shape == (56, 56) and cond.shape == (10, 10)
    globe = layout_target(Constant(), np.eye(3), n=6, sub=3)
    np.testing.assert_allclose(decode(globe), -1234, atol=.001)
    assert SPACING_M == 23040 and FACTOR == 7


def test_layout_projection_shared_edges_halos_gradients_and_noise():
    grid = SphereGrid(n=6)
    x = torch.randn(2, 6, grid.side, grid.side, requires_grad=True)
    out = grid.project(x)
    core = out[..., :7, :7].detach().numpy()
    np.testing.assert_allclose(core, cube.identify(core), atol=1e-7)
    torch.testing.assert_close(grid.project(out), out)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    noise = grid.noise(1024, "cpu", torch.Generator().manual_seed(3))
    flat = noise[..., :7, :7].reshape(1024, -1)
    members, ids, owners, _ = cube.edge_groups(6)
    torch.testing.assert_close(flat[:, members], flat[:, owners[ids]])
    assert .94 < float(flat[:, owners].std()) < 1.06


def test_layout_edm_has_finite_gradient_and_spherical_samples():
    torch.set_num_threads(2)
    model = LayoutDenoiser(n=6, ch=8, mults=(1, 2), blocks=1)
    x = model.grid.noise(2, "cpu")*.5
    loss = layout_loss(model, x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.net.out[-1].weight.grad).all()
    result = sample_layout(model.eval(), {"mean": 0., "std": 1.}, steps=2)
    assert result.shape == (1, 6, 7, 7) and np.isfinite(result).all()
    np.testing.assert_allclose(result, cube.identify(result), atol=1e-7)


def test_layout_warm_start_separate_run_preserves_source_and_resume(tmp_path):
    lat=np.radians(np.arange(90,-91,-1));lon=np.radians(np.arange(-180,181))
    values=(2500*np.cos(lat[:,None])*np.cos(lon[None])).astype(np.float32)
    folder=tmp_path/'layout';folder.mkdir()
    np.save(folder/'maps.npy',np.stack((values,values*.8,values*.9)))
    meta=dict(kind='layout',version=VERSION,face_intervals=6,supersample=2,
              spacing_m=161280.,radius_m=6371000.,
              records=[dict(split=s) for s in ('train','val','test')])
    (folder/'manifest.json').write_text(json.dumps(meta))
    run0=tmp_path/'initial'
    args=['layout','--data',str(folder),'--run',str(run0),'--device','cpu','--batch','2',
          '--ch','8','--mults','1,2','--blocks','1','--threads','2','--preview-steps','2','--eval-every','1']
    train_main(args+['--steps','1'])
    original=(run0/'best.pt').read_bytes()
    run=tmp_path/'continued'
    args=['layout','--data',str(folder),'--run',str(run),'--device','cpu','--batch','2',
          '--init-from',str(run0/'best.pt'),'--threads','2','--preview-steps','2','--eval-every','1']
    train_main(args+['--steps','1']);train_main(args+['--steps','2'])
    ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    assert ck['initialized_from']['step']==1 and ck['step']==2 and ck['config']['ch']==8
    assert (run0/'best.pt').read_bytes()==original
    bad=dict(meta,face_intervals=7)
    (folder/'manifest.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Warm-start'):
        train_main(args+['--run',str(tmp_path/'bad'),'--steps','1'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA target integration")
def test_cuda_layout_targets_preserve_sampling_and_match_cpu(tmp_path):
    lat = np.radians(np.arange(90, -91, -1))
    lon = np.radians(np.arange(-180, 181))
    values = (2000*np.sin(lat[:, None]) + 3000*np.cos(lat[:, None])*np.cos(lon[None])).astype(np.float32)
    np.save(tmp_path/"maps.npy", np.stack((values, values*.8, values*.9)))
    meta = {"kind": "layout", "version": VERSION, "face_intervals": 6, "supersample": 4,
            "records": [{"split": "train"}, {"split": "val"}, {"split": "test"}]}
    (tmp_path/"manifest.json").write_text(json.dumps(meta))
    cpu, gpu = LayoutData(tmp_path, 17), LayoutData(tmp_path, 17)
    grid = SphereGrid(n=6).cuda()
    stats = {"mean": -10., "std": 40.}
    expected = padded_target(cpu.batch(4), type("GridModel", (), {"grid": grid})(), stats, "cuda")
    actual = gpu.cuda_batch(4, grid, stats, "cuda")
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-4)
    assert cpu.rng.bit_generator.state == gpu.rng.bit_generator.state
    assert not actual.requires_grad
    torch.testing.assert_close(grid.project(actual), actual)


def test_bridge_preparation_resume_and_checkpoint_dataset_guard(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    dem = tmp_path/"earth.tif"
    a = np.random.default_rng(6).uniform(-4000, 4000, (180, 360)).astype(np.float32)
    with rasterio.open(dem, "w", driver="GTiff", height=180, width=360,
                       count=1, dtype="float32", transform=from_origin(-180, 90, 1, 1), crs="EPSG:4326") as dst:
        dst.write(a, 1)
    data = tmp_path/"data"
    prepare_bridge(dem, data, tmp_path/"cache", train=8, val=2, size=56, sub=2)
    target = np.load(data/"train_target.npy").copy()
    prepare_bridge(dem, data, tmp_path/"cache", train=8, val=2, size=56, sub=2)
    np.testing.assert_array_equal(target, np.load(data/"train_target.npy"))
    other = tmp_path/"other-data"
    prepare_bridge(dem, other, tmp_path/"cache", train=8, val=2, size=56, sub=2, workers=1)
    np.testing.assert_array_equal(target, np.load(other/"train_target.npy"))
    np.testing.assert_array_equal(np.load(data/"val_target.npy"), np.load(other/"val_target.npy"))
    dataset = BridgeData(data, 0)
    assert dataset.batch(2)[0].shape == (2, 1, 56, 56)
    run = tmp_path/"run"
    args = ["bridge", "--data", str(data), "--run", str(run), "--device", "cpu",
            "--batch", "2", "--ch", "8", "--mults", "1,2", "--blocks", "1",
            "--threads", "2", "--preview-steps", "2", "--eval-every", "1"]
    train_main(args + ["--steps", "1"])
    train_main(args + ["--steps", "2"])
    ck = torch.load(run/"latest.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and (run/"best.pt").exists()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr("planet_training.train.edm_loss", interrupt)
        assert train_main(args + ["--steps", "3"]) == 130
    assert torch.load(run/"latest.pt", map_location="cpu", weights_only=False)["step"] == 2
    meta = json.loads((data/"manifest.json").read_text())
    meta["seed"] += 1
    (data/"manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different dataset"):
        train_main(args + ["--steps", "3"])
