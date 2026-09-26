import numpy as np
import pytest
import rasterio
from planet_diffusion import cube
from planet_diffusion.region import Field, reconstruct, noise_field, validate_region, generate_region
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.storage import save_state, load_state, export_tiff, verify_state
from planet_diffusion.gui import build_command
from planet_diffusion.cache import PredictionCache


def materialize(field):
    y,x = np.arange(field.n+1)[:,None],np.arange(field.n+1)[None,:]
    return np.stack([field.read(f,y,x) for f in range(6)])[None]


@pytest.mark.parametrize('n',[16,48])
def test_lazy_geometry_and_reconstruction_match_dense(n):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(1,6,n+1,n+1)).astype(np.float32)
    field = Field(n,lambda f,y,x:a[0,f,y,x],tile=4)
    dense = cube.identify(a)
    np.testing.assert_array_equal(materialize(field),dense)
    for f in range(6):
        y,x = np.arange(-2,n+3)[:,None], np.arange(-2,n+3)[None,:]
        np.testing.assert_allclose(field.read(f,y,x),cube.read(dense,f,y,x)[0],atol=1e-6)
    low = cube.identify(rng.normal(size=(1,6,n//8+1,n//8+1)).astype(np.float32))
    np.testing.assert_allclose(materialize(reconstruct(field,low)),cube.reconstruct(dense,low),atol=2e-6)


def test_noise_is_shared_and_query_order_independent():
    a,b = noise_field(42,128),noise_field(42,128)
    b.read(5,np.array([0,128]),np.array([128,0]))
    np.testing.assert_array_equal(materialize(a),materialize(b))
    dense = materialize(a)
    np.testing.assert_array_equal(dense,cube.identify(dense,noise=True))


@pytest.mark.parametrize('bounds',[(170,-10,-170,10),(-180,80,180,90),(-45,-10,45,10)])
def test_bounds(bounds):
    result = validate_region(bounds,8,8,2048)
    assert result[2] > result[0]


@pytest.mark.parametrize('bounds,width,height',[
    ((0,0,0,10),1,1),((0,10,20,-10),1,1),((0,-91,20,0),1,1),
    ((0,0,float('nan'),10),1,1),((0,0,1,1),1000,1000),
    ((0,0,20,20),0,10),((0,0,20,20),1,None)])
def test_invalid_region(bounds,width,height):
    with pytest.raises(ValueError): validate_region(bounds,width,height,2048)


def test_regional_generation_resume_storage_and_patch_savings(tmp_path):
    class Counting(DiagnosticBackend):
        calls = 0
        def predict(self,stage,*args):
            if stage == 'decoder': self.calls += 1
            return super().predict(stage,*args)
    backend = Counting()
    messages = []
    args = dict(seed=42,face_coarse=4,coarse_steps=2,progress=messages.append,
                checkpoint_dir=tmp_path/'checkpoints',region=((-5,-5,5,5),8,8))
    a,meta = generate_cube(backend,**args)
    assert backend.calls < 6*25
    assert meta['decoder_patches'] == backend.calls
    assert meta['regional_execution']['decoder_model_calls'] == backend.calls
    progress = [m for m in messages if m.startswith('regional decoder:')]
    assert len(progress) == len(set(progress))
    calls = backend.calls
    b,resumed = generate_cube(backend,**args)
    assert backend.calls == calls
    assert resumed['regional_execution']['decoder_model_calls'] == 0
    assert resumed['regional_execution']['prediction_disk_reads'] > 0
    np.testing.assert_array_equal(a,b)
    assert not (tmp_path/'checkpoints'/'residual.npy').exists()
    assert not (tmp_path/'checkpoints'/'cube-elevation.npy').exists()
    meta['radius_metres'] = 6371000
    save_state(tmp_path/'state',a,meta)
    saved,meta = load_state(tmp_path/'state')
    assert verify_state(saved,meta)['passed']
    export_tiff(tmp_path/'region.tif',saved,meta)
    with rasterio.open(tmp_path/'region.tif') as ds:
        assert ds.shape == (8,8)
        assert tuple(ds.bounds) == (-5,-5,5,5)
        np.testing.assert_array_equal(ds.read(1),a)
        assert ds.units == ('m',)
    with pytest.raises(ValueError,match='Regional states'):
        export_tiff(tmp_path/'wrong.tif',saved,meta,height=4)


def test_launcher_passes_region(tmp_path):
    cmd,_,_ = build_command(tmp_path/'run',42,coarse_height=16,device='cuda',
                            export_height=128,export_width=256,bounds=(170,-10,-170,10))
    assert cmd[cmd.index('--bounds')+1:cmd.index('--bounds')+5] == ['170','-10','-170','10']
    assert cmd[cmd.index('--width')+1] == '256'
    assert cmd[cmd.index('--height')+1] == '128'


def test_regional_decoder_matches_dense_at_seams_poles_and_overlaps(tmp_path):
    backend = DiagnosticBackend()
    n = 512
    latent = cube.noise(123,17,5,n//8)
    noise = materialize(noise_field(42,n))
    t = float(np.arctan(160))
    xt = noise*np.sin(t)
    def predict(a,f,y,x):
        cond = cube.read(latent[:4],f,np.floor_divide(np.arange(y,y+512),8)[:,None],
                         np.floor_divide(np.arange(x,x+512),8)[None,:],nearest=True)
        return backend.predict('decoder',a,cond,t)
    pred = cube.consensus(xt,predict,512,256)
    residual = cube.identify(np.cos(t)*xt+np.sin(t)*pred)
    z = cube.reconstruct(residual*backend.residual_std+backend.residual_mean,latent[4:5]*38.6-31.4)
    dense = np.sign(z)*z*z
    meta = {'face_native_intervals':n}
    for bounds in ([170,-2,190,2],[-60,86,60,90],[-60,-90,60,-86],[40,30,50,40]):
        west,south,east,north = bounds
        result,_ = generate_region(backend,latent,42,bounds,4,2,meta,tmp_path,lambda _:None)
        lon = np.deg2rad(west+(np.arange(4)+.5)*(east-west)/4)[None,:]
        lat = np.deg2rad(north-(np.arange(2)+.5)*(north-south)/2)[:,None]
        p = np.stack(np.broadcast_arrays(np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
        np.testing.assert_allclose(result,cube.sample(dense,p)[0],atol=.005,rtol=2e-6)
        # Independent query with different bounds, same centres, and saved patches.
        half,_ = generate_region(backend,latent,42,[west,south,(west+east)/2,north],2,2,
                               meta,tmp_path,lambda _:None)
        np.testing.assert_array_equal(half,result[:,:2])


def test_saved_prediction_does_not_prepare_noise_again(tmp_path):
    calls = []
    def make_input():
        calls.append('input')
        return np.ones((1,4,4),np.float32)
    def predict(a,*location):
        calls.append('model')
        return 2*a
    cache = PredictionCache(tmp_path,{'test':'lazy-input'},predict)
    first = cache.get_or_compute(make_input,(1,4,4),0,0,0)
    # A new in-memory cache simulates eviction or restarting the generator.
    cache = PredictionCache(tmp_path,{'test':'lazy-input'},predict)
    second = cache.get_or_compute(make_input,(1,4,4),0,0,0)
    assert calls == ['input','model']
    np.testing.assert_array_equal(first,second)
    assert cache.hits == 1 and cache.misses == 0
    np.save(tmp_path/'0_0_0.npy',np.zeros((1,2,2),np.float32))
    with pytest.raises(ValueError,match='Invalid decoder prediction'):
        cache.get_or_compute(make_input,(1,4,4),0,0,0)
    assert calls == ['input','model']


def test_latitude_scan_keeps_nearby_reconstruction_tiles():
    calls = []
    def raw(f,y,x):
        calls.append(1)
        return (y+x).astype(np.float32)
    field = Field(1024,raw)
    # More than 128 adjacent blocks, as encountered across a broad latitude
    # strip. The next row must reuse them, rather than recompute the hierarchy.
    y,x = np.meshgrid(np.arange(1,9)*32+16,np.arange(1,21)*32+16,indexing='ij')
    first = field.read(0,y,x)
    count = len(calls)
    second = field.read(0,y+1,x)
    assert len(calls) == count
    np.testing.assert_array_equal(second,first+1)
