import numpy as np
import pytest
import rasterio
from PIL import Image

from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cli import main
from planet_diffusion.draft import Draft
from planet_diffusion.gui import build_command
from planet_diffusion.model_presets import (MODEL_30M, MODEL_90M, REVISION_30M,
                                            default_revision, sparse_guide_height)
from planet_diffusion.sparse import SparseWorld, Stage, Guide


def test_sparse_patch_budget_keeps_working_sets_larger_than_48(tmp_path):
    calls = []
    def predict(face,y,x):
        calls.append((face,y,x))
        return np.full((5,64,64),face+y+x,np.float32)
    stage = Stage(tmp_path,{},'latent',512,5,64,32,predict)
    locations = [(0,32*(i//10),32*(i%10)) for i in range(80)]
    expected = [stage.patch(*location).copy() for location in locations]
    for location, value in zip(locations,expected):
        np.testing.assert_array_equal(stage.patch(*location),value)
    assert len(calls) == 80 and stage.cache.hits == 0
    assert stage.patch_memory.bytes == 80*5*64*64*4
    assert stage.patch_memory.bytes <= stage.patch_memory.max_bytes


def test_sparse_patch_eviction_reloads_without_recomputing(tmp_path):
    calls = []
    def predict(face,y,x):
        calls.append((face,y,x))
        return np.full((1,8,8),y+x,np.float32)
    stage = Stage(tmp_path,{},'small',32,1,8,4,predict,patch_cache_bytes=2*8*8*4)
    first = stage.patch(0,0,0).copy()
    stage.patch(0,0,4)
    np.testing.assert_array_equal(stage.patch(0,0,0),first)
    stage.patch(0,0,8)
    assert (0,0,4) not in stage.patch_memory.entries
    np.testing.assert_array_equal(stage.patch(0,0,4),np.full((1,8,8),4,np.float32))
    assert len(calls) == 3 and stage.cache.hits == 1
    assert stage.patch_memory.bytes <= stage.patch_memory.max_bytes


def test_shared_channel_blends_preserve_weighted_seams_and_interpolation(tmp_path):
    def predict(face,y,x):
        yy,xx = np.meshgrid(np.arange(64),np.arange(64),indexing='ij')
        return np.stack([face*10+y+yy*.3+x+xx*.1+c for c in range(3)]).astype(np.float32)
    shared = Stage(tmp_path/'shared',{},'blend',48,3,64,32,predict,include_endpoint=True)
    scalar = Stage(tmp_path/'scalar',{},'blend',48,3,64,32,predict,
                   include_endpoint=True,blend_cache_bytes=0)
    y,x = np.arange(-1,50)[:,None]+.25,np.arange(-1,50)[None,:]+.5
    for face in range(6):
        np.testing.assert_array_equal(shared.read(face,y,x),scalar.read(face,y,x))
    assert shared.blend_memory.hits > 0
    assert shared.blend_memory.misses < scalar.blend_memory.misses
    assert shared.blend_memory.bytes <= shared.blend_memory.max_bytes
    assert shared.cache.misses == scalar.cache.misses


def test_coarse_small_reads_do_not_generate_unused_tiles(tmp_path):
    calls = []
    def predict(face,y,x):
        calls.append((face,y,x))
        return np.full((6,64,64),face+y+x,np.float32)
    stage = Stage(tmp_path,{},"coarse",512,6,64,48,predict,tile=4)
    actual = stage.read(0,np.arange(310,314)[:,None],np.arange(310,314)[None,:])
    assert calls == [(0,288,288)]
    np.testing.assert_array_equal(actual,np.full((6,4,4),576,np.float32))


def test_sparse_batches_skip_partially_saved_predictions(tmp_path):
    def predict(face,y,x):
        return np.full((1,64,64),y+x,np.float32)
    first = Stage(tmp_path,{},'latent',512,1,64,32,predict)
    first.patch(0,32,32)
    calls = []
    def batch(locations):
        calls.append(locations)
        return np.stack([predict(*location) for location in locations])
    resumed = Stage(tmp_path,{},'latent',512,1,64,32,predict,
                    calculate_batch=batch,batch_size=2)
    actual = resumed.read(0,np.arange(64,96)[:,None],np.arange(64,96)[None,:])
    assert calls == [[(0,32,64),(0,64,32)],[(0,64,64)]]
    assert resumed.cache.hits == 1 and resumed.cache.misses == 3
    reference = Stage(tmp_path/'reference',{},'latent',512,1,64,32,predict)
    np.testing.assert_array_equal(actual,reference.read(
        0,np.arange(64,96)[:,None],np.arange(64,96)[None,:]))


@pytest.mark.parametrize('bounds', [(12,-12.04,12.04,-12), (44.98,-.02,45.02,.02)])
def test_sparse_batches_preserve_output_dependencies_and_resume(tmp_path,bounds):
    class Counting(DiagnosticBackend):
        def __init__(self):
            self.batches = []
        def predict_batch(self,stage,a,cond,t):
            if stage == 'latent':
                self.batches.append(len(a))
            return super().predict_batch(stage,a,cond,t)
    results,counts,calls = [],[],[]
    for batch in (1,3):
        backend = Counting()
        world = SparseWorld(backend,42,1024,2,6371000.,tmp_path/str(batch),latent_batch_size=batch)
        out,metadata,_ = world.region(bounds,8,8,progress=lambda _:None)
        results.append(out)
        counts.append([s.cache.misses for s in (world.coarse,world.latent_first,world.latent)])
        calls.append(len(backend.batches))
        assert metadata['latent_batch_size'] == batch
        assert max(backend.batches) <= batch
        if batch == 3:
            assert 3 in backend.batches and 1 in backend.batches
        resumed = SparseWorld(backend,42,1024,2,6371000.,tmp_path/str(batch),latent_batch_size=batch)
        replay,_,_ = resumed.region(bounds,8,8,progress=lambda _:None)
        np.testing.assert_array_equal(out,replay)
        assert len(backend.batches) == calls[-1]
        with pytest.raises(ValueError,match='different generation settings'):
            SparseWorld(backend,42,1024,2,6371000.,tmp_path/str(batch),latent_batch_size=batch+1)
    np.testing.assert_array_equal(*results)
    assert counts[0] == counts[1]
    assert calls[1] < calls[0]


@pytest.mark.parametrize('batch', [0,-1,1.5,True])
def test_sparse_invalid_batch_size(tmp_path,batch):
    with pytest.raises(ValueError,match='positive integer'):
        SparseWorld(DiagnosticBackend(),42,1024,2,6371000.,tmp_path,latent_batch_size=batch)


@pytest.mark.parametrize('channels',[1,3,6])
def test_sparse_weighted_field_matches_dense_cube_blend(tmp_path,channels):
    n = 8
    def predict(face,y,x):
        yy,xx = np.meshgrid(np.arange(64),np.arange(64),indexing="ij")
        return np.array([face*100+y+x+yy*.1+xx*.01+c for c in range(channels)],np.float32)
    sparse = Stage(tmp_path,{"seed":1},"blend",n,channels,64,48,predict)
    dense = cube.consensus(np.zeros((channels,6,n+1,n+1),np.float32),
                           lambda _,f,y,x:predict(f,y,x),size=64,stride=48,
                           weight_window=cube.linear_weight_window(64),shared_weights=True)
    y,x = np.arange(n+1)[:,None],np.arange(n+1)[None,:]
    for face in range(6):
        np.testing.assert_allclose(sparse.read(face,y,x),dense[:,face],rtol=2e-6,atol=2e-5)


def test_sparse_regional_queries_reuse_fixed_guides(tmp_path):
    backend = DiagnosticBackend()
    world = SparseWorld(backend,42,128,2,6371000.,tmp_path/"checkpoints")
    whole,_,_ = world.region((-2,-1,2,1),4,2,progress=lambda _:None)
    counts = (world.coarse.cache.misses,world.latent_first.cache.misses,
              world.latent.cache.misses)
    assert counts[0] < 54  # Full six-face coarse schedule at guide height 128.
    assert counts[1] < 100 and counts[2] < 100
    left,_,_ = world.region((-2,-1,0,1),2,2,progress=lambda _:None)
    np.testing.assert_array_equal(left,whole[:,:2])
    assert counts == (world.coarse.cache.misses,world.latent_first.cache.misses,
                      world.latent.cache.misses)


def test_sparse_cli_can_query_and_export_climate(tmp_path):
    state = tmp_path/"state"
    output = tmp_path/"first.tif"
    climate = tmp_path/"climate.tif"
    assert main(["generate","--backend","diagnostic","--regional-only",
                 "--latent-batch-size","3",
                 "--seed","42","--coarse-height","4","--coarse-steps","2",
                 "--bounds","-2","-2","2","2","--width","2","--height","2",
                 "--state",str(state),"--output",str(output),
                 "--climate-output",str(climate)]) == 0
    second = tmp_path/"second.tif"
    assert main(["query","--state",str(state),"--bounds","-2","-2","2","2",
                 "--width","2","--height","2","--output",str(second)]) == 0
    with rasterio.open(output) as first, rasterio.open(second) as repeated, rasterio.open(climate) as weather:
        np.testing.assert_array_equal(first.read(1),repeated.read(1))
        assert weather.count == 5 and np.isfinite(weather.read()).all()


def test_sparse_draft_guide_matches_dense_nodes(tmp_path):
    pixels = np.full((4,8,4),255,np.uint8)
    pixels[:,:4,:3] = 0
    path = tmp_path/"draft.png"
    Image.fromarray(pixels,"RGBA").save(path)
    draft = Draft(path)
    n = 4
    sparse = Guide(42,n,draft,(1.5,3,3,3,3),.5)
    dense = draft.conditioning(42,n)
    y,x = np.arange(n+1)[:,None],np.arange(n+1)[None,:]
    for face in range(6):
        # Dense import averages shared-chart inputs before the elevation square
        # root; sparse sampling averages the completed per-chart guide.
        np.testing.assert_allclose(sparse.read(face,y,x),dense[:,face],atol=.01)
    command,_,_ = build_command(tmp_path/"run",42,coarse_height=1024,
                                 bounds=(-2,-2,2,2),export_width=512,export_height=512,
                                 regional_only=True)
    assert "--regional-only" in command


def test_sparse_imported_guide_survives_new_query(tmp_path):
    folder = tmp_path/"inputs"
    folder.mkdir()
    from rasterio.transform import from_bounds
    with rasterio.open(folder/"heightmap.tif","w",driver="GTiff",width=8,height=4,
                       count=1,dtype="float32",crs="EPSG:4326",
                       transform=from_bounds(-180,-90,180,90,8,4)) as dataset:
        dataset.write(np.ones((4,8),np.float32)*1000,1)
    state = tmp_path/"state"
    assert main(["generate","--backend","diagnostic","--regional-only",
                 "--conditioning-dir",str(folder),"--coarse-height","4",
                 "--coarse-steps","2","--seed","42","--bounds","-2","-2","2","2",
                 "--width","2","--height","2","--state",str(state),
                 "--output",str(tmp_path/"first.tif")]) == 0
    assert main(["query","--state",str(state),"--bounds","1","-2","3","2",
                 "--width","2","--height","2","--output",str(tmp_path/"next.tif")]) == 0


def test_30m_regional_model_selection(tmp_path):
    assert sparse_guide_height(MODEL_90M) == 1024
    assert sparse_guide_height(MODEL_30M) == 2560
    assert sparse_guide_height("models/terrain-diffusion-30m") == 2560
    assert default_revision(MODEL_30M) == REVISION_30M
    command,_,_ = build_command(tmp_path/"run30",42,coarse_height=2560,
                                 bounds=(-.1,-.1,.1,.1),export_width=512,
                                 export_height=512,regional_only=True,
                                 model_choice="30 m")
    assert command[command.index("--model")+1].endswith("terrain-diffusion-30m")
    assert command[command.index("--revision")+1] == REVISION_30M
    assert "--regional-only" in command
