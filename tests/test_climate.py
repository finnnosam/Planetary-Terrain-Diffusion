import ast
from pathlib import Path
import numpy as np
import pytest
import rasterio

from planet_diffusion import cube
from planet_diffusion.climate import local_baseline, from_coarse, evaluate, BANDS
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.storage import save_state, load_state, load_climate, export_tiff
from planet_diffusion.cli import main
from planet_diffusion.gui import build_command


def features(n=2):
    a = np.empty((5,6,n+1,n+1),np.float32)
    a[:] = np.array([20,350,1000,50,-.0065],np.float32)[:,None,None,None]
    return a


def test_regression_known_lapse_rate_and_fallbacks():
    e = np.arange(19*21,dtype=np.float32).reshape(19,21)*10
    baseline,beta = local_baseline(25-.0065*e,e)
    np.testing.assert_allclose(baseline,25,atol=.0002)
    np.testing.assert_allclose(beta,-.0065,atol=1e-7)
    for h in (0,10):
        baseline,beta = local_baseline(np.full((17,17),20.),np.full((17,17),h))
        np.testing.assert_allclose(beta,-.0065)
        np.testing.assert_allclose(baseline,20+.0065*h)
    _,beta = local_baseline(20+.01*e,e)
    np.testing.assert_array_equal(beta,0)
    _,beta = local_baseline(20-.02*e,e)
    np.testing.assert_allclose(beta,-.012)


def test_regression_matches_pinned_upstream():
    torch = pytest.importorskip('torch')
    path = Path(__file__).resolve().parents[1]/'upstream/terrain_diffusion/inference/postprocessing.py'
    if not path.exists(): pytest.skip('Pinned upstream checkout unavailable')
    # Load only the pure regression function, avoiding unrelated plotting dependencies.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'local_baseline_temperature_torch')
    namespace = {'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    rng = np.random.default_rng(42)
    e = np.maximum(rng.normal(500,1200,(29,31)),0).astype(np.float32)
    t = (25-.007*e+rng.normal(0,2,e.shape)).astype(np.float32)
    actual = local_baseline(t,e)
    expected = namespace[fn.name](torch.from_numpy(t),torch.from_numpy(e),win=15,fallback_threshold=.02)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b.numpy()[0],atol=1e-4,rtol=1e-5)


def test_sphere_features_share_edges_and_adjust_land_only():
    coarse = np.zeros((6,6,5,5),np.float32)
    coarse[0] = 10  # Signed square-root elevation: 100 metres.
    coarse[2] = 20
    coarse[3:6] = np.array([350,1000,50])[:,None,None,None]
    climate = from_coarse(coarse)
    np.testing.assert_array_equal(climate,cube.identify(climate))
    np.testing.assert_allclose(climate[0],20.65,atol=1e-5)
    z = np.array([[-500.,0.,100.,1100.]])
    a = evaluate(climate,np.array([[-180,-90,0,180]]),0,z)
    np.testing.assert_allclose(a[0],[[20.65,20.65,20,13.5]],atol=1e-5)
    np.testing.assert_allclose(a[1],3.5)
    np.testing.assert_allclose(a[2],1000)
    np.testing.assert_allclose(a[3],50)
    # Longitude is irrelevant at poles; date-line sides identify the same place.
    for lat in (-90,0,90):
        a = evaluate(climate,np.array([-180,180]),lat,np.array([0,0]))
        np.testing.assert_array_equal(a[:,0],a[:,1])


def test_global_storage_and_tiled_climate_exports(tmp_path):
    a = np.full((17,32),1000,np.float32)
    meta = {'native_height':16,'face_coarse_intervals':2,'radius_metres':6371000}
    save_state(tmp_path/'state',a,meta,climate=features())
    saved,meta = load_state(tmp_path/'state')
    climate = load_climate(tmp_path/'state',meta)
    export_tiff(tmp_path/'elev.tif',saved,meta,height=8)
    export_tiff(tmp_path/'climate.tif',saved,meta,height=8,climate=climate)
    with rasterio.open(tmp_path/'climate.tif') as ds, rasterio.open(tmp_path/'elev.tif') as elev:
        assert ds.count == 5
        assert ds.descriptions == tuple(name for name,_ in BANDS)
        assert ds.units == tuple(unit for _,unit in BANDS)
        assert ds.transform == elev.transform and ds.crs == elev.crs
        np.testing.assert_allclose(ds.read()[:,0,0],[13.5,3.5,1000,50,-.0065],atol=1e-6)
        assert ds.tags()['units'] == 'per-band'
    export_tiff(tmp_path/'tile.tif',saved,meta,height=8,window=(15,1,3,2),climate=climate)
    with rasterio.open(tmp_path/'tile.tif') as ds:
        assert ds.bounds.right > 180
        np.testing.assert_allclose(ds.read(1),13.5,atol=1e-6)
    climate[0,0,0,0] += 1
    np.save(tmp_path/'state/climate.npy',climate)
    with pytest.raises(ValueError,match='checksum'):
        load_state(tmp_path/'state')


def test_regional_climate_resume_and_export(tmp_path):
    args = dict(seed=42,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                region=((170,-10,-170,10),4,4),checkpoint_dir=tmp_path/'checkpoints',with_climate=True)
    a,meta,climate = generate_cube(DiagnosticBackend(),**args)
    b,_,resumed = generate_cube(DiagnosticBackend(),**args)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(climate,resumed)
    meta['radius_metres'] = 6371000
    save_state(tmp_path/'state',a,meta,climate=climate)
    a,meta = load_state(tmp_path/'state')
    export_tiff(tmp_path/'climate.tif',a,meta,climate=load_climate(tmp_path/'state',meta))
    with rasterio.open(tmp_path/'climate.tif') as ds:
        assert ds.count == 5 and ds.shape == (4,4)
        assert tuple(ds.bounds) == (170,-10,190,10)
        assert np.isfinite(ds.read()).all()
    (tmp_path/'checkpoints/coarse.npy').unlink()
    with pytest.raises(ValueError,match='coarse checkpoint'):
        generate_cube(DiagnosticBackend(),**args)


def test_cli_generation_and_later_climate_export(tmp_path):
    state,output,climate = [str(tmp_path/n) for n in ('state','elev.tif','climate.tif')]
    assert main(['generate','--backend','diagnostic','--seed','42','--coarse-height','4',
                 '--coarse-steps','2','--bounds','-5','-5','5','5','--width','4','--height','4',
                 '--state',state,'--output',output,'--climate-output',climate]) == 0
    assert main(['export','--state',state,'--output',str(tmp_path/'export.tif'),
                 '--climate-output',str(tmp_path/'export-climate.tif')]) == 0
    with rasterio.open(climate) as a, rasterio.open(tmp_path/'export-climate.tif') as b:
        np.testing.assert_array_equal(a.read(),b.read())
    assert main(['verify','--state',state]) == 0
    cmd,_,_ = build_command(tmp_path/'run',42,export_climate=True)
    assert cmd[cmd.index('--climate-output')+1] == str((tmp_path/'run/planet-climate.tif').resolve())


def test_old_states_and_output_collisions(tmp_path):
    a = np.zeros((17,32),np.float32)
    meta = {'native_height':16,'radius_metres':6371000}
    save_state(tmp_path/'old',a,meta)
    _,meta = load_state(tmp_path/'old')
    with pytest.raises(ValueError,match='no saved climate'):
        load_climate(tmp_path/'old',meta)
    with pytest.raises(SystemExit):
        main(['export','--state',str(tmp_path/'old'),'--output',str(tmp_path/'same.tif'),
              '--climate-output',str(tmp_path/'same.tif')])
    assert not (tmp_path/'same.tif').exists()
