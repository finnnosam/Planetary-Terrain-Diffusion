"""Physical units, spherical boundaries, and end-to-end TIFF conditioning."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.conditioning import CHANNEL_FILES, TiffConditioning, parse_snr
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.cli import main
from planet_diffusion.gui import build_command
from planet_diffusion.storage import load_state
from planet_diffusion.procedural import encode


def raw_import_fallback(seed, n):
    return encode(cube.conditioning(seed,n,raw=True))


def write_map(folder, name, values, bounds=(-180,-90,180,90), crs="EPSG:4326"):
    values = np.broadcast_to(np.asarray(values,np.float32),(8,16)).copy()
    path = folder/name
    with rasterio.open(path,"w",driver="GTiff",width=16,height=8,count=1,
                       dtype="float32",crs=crs,transform=from_bounds(*bounds,16,8),nodata=-9999) as ds:
        ds.write(values,1)
    return path


def test_units_missing_channels_and_backend_unchanged(tmp_path):
    for name,value in zip(CHANNEL_FILES,[-1600,18,3.5,1200,55]):
        write_map(tmp_path,name,value)
    source = TiffConditioning(tmp_path)
    guide = source.conditioning(42,4)
    for channel,expected in enumerate([-40,18,350,1200,55]):
        np.testing.assert_allclose(guide[channel],expected,rtol=1e-6)
    for name in CHANNEL_FILES[1:]:
        (tmp_path/name).unlink()
    guide = TiffConditioning(tmp_path).conditioning(42,4)
    np.testing.assert_array_equal(guide[1:],raw_import_fallback(42,4)[1:])


def test_orientation_seam_poles_and_nodata(tmp_path):
    pixels = np.zeros((8,16),np.float32)
    pixels[:4,8:] = 20
    write_map(tmp_path,"temperature.tif",pixels)
    source = TiffConditioning(tmp_path)
    raster = source.rasters[1]
    p = np.array([[.5,.5,2**-.5],[.5,.5,-2**-.5],[.5,-.5,2**-.5]])
    np.testing.assert_allclose(raster.evaluate(p)[0],[20,0,0],atol=1e-5)
    p = np.array([[-1,1e-9,0],[-1,-1e-9,0]])
    assert np.ptp(raster.evaluate(p)[0]) < 1e-5
    p = np.array([[1e-9,0,1],[-1e-9,0,1],[0,1e-9,1]])
    assert np.ptp(raster.evaluate(p)[0]) < 1e-5
    guide = source.conditioning(42,4).reshape(5,-1)
    members,ids,owners,_ = cube.edge_groups(4)
    np.testing.assert_array_equal(guide[:,members],guide[:,owners[ids]])
    for value in [-9999,np.nan,np.inf]:
        write_map(tmp_path,"temperature.tif",value)
        np.testing.assert_array_equal(TiffConditioning(tmp_path).conditioning(42,4),raw_import_fallback(42,4))
    # Half the source area is missing, so value and coverage stay premultiplied.
    pixels[:,::2],pixels[:,1::2] = 20,-9999
    write_map(tmp_path,"temperature.tif",pixels)
    expected = 10+.5*raw_import_fallback(42,4)[1]
    np.testing.assert_allclose(TiffConditioning(tmp_path).conditioning(42,4)[1],expected,rtol=1e-6)


def test_validation_and_scaled_values(tmp_path):
    with pytest.raises(ValueError,match="No conditioning"):
        TiffConditioning(tmp_path)
    for bounds,crs in [((0,0,360,180),"EPSG:4326"),((-180,-90,180,90),"EPSG:3857")]:
        write_map(tmp_path,"heightmap.tif",1,bounds,crs)
        with pytest.raises(ValueError,match="global geographic"):
            TiffConditioning(tmp_path)
    path = write_map(tmp_path,"heightmap.tif",10)
    with rasterio.open(path,"r+") as ds:
        ds.scales = (2,)
        ds.offsets = (-120,)
    np.testing.assert_allclose(TiffConditioning(tmp_path).conditioning(42,2)[0],-10)
    write_map(tmp_path,"precipitation.tif",-1)
    with pytest.raises(ValueError,match="nonnegative"):
        TiffConditioning(tmp_path)
    for snr in ["1,2", "a,b,c,d,e", "nan,1,1,1,1", "0,1,1,1,1", "5,1,1,1,1"]:
        with pytest.raises(ValueError,match="five numbers"):
            parse_snr(snr)


def test_generation_resume_identity_and_snapshot(tmp_path):
    write_map(tmp_path,"temperature.tif",18)
    source = TiffConditioning(tmp_path,".1,.2,.3,.4,.5")
    backend = DiagnosticBackend()
    original = backend.cond_snr.copy()
    seen = {}
    args = dict(seed=42,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                conditioning=source,checkpoint_dir=tmp_path/"checkpoints",with_climate=True,
                region=((170,-10,-170,10),8,8))
    a,meta,climate = generate_cube(backend,**args,audit=lambda k,v:seen.update({k:v.copy()}) if k == "raw-conditioning" else None)
    np.testing.assert_array_equal(seen["raw-conditioning"],source.conditioning(42,2))
    np.testing.assert_array_equal(backend.cond_snr,original)
    b,_,other = generate_cube(backend,**args)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(climate,other)
    assert meta["conditioning_tiffs"] == source.metadata
    assert (tmp_path/"checkpoints/conditioning/temperature.tif").read_bytes() == source.sources["temperature.tif"]
    write_map(tmp_path,"temperature.tif",19)
    args["conditioning"] = TiffConditioning(tmp_path,".1,.2,.3,.4,.5")
    with pytest.raises(ValueError,match="different parameters"):
        generate_cube(backend,**args)
    write_map(tmp_path,"temperature.tif",18)
    args["conditioning"] = TiffConditioning(tmp_path,".2,.2,.3,.4,.5")
    with pytest.raises(ValueError,match="different parameters"):
        generate_cube(backend,**args)


def test_cli_global_export_and_launcher(tmp_path):
    write_map(tmp_path,"temperature.tif",18)
    args = ["generate","--backend","diagnostic","--seed","42","--coarse-height","4",
            "--coarse-steps","2","--height","16","--state",str(tmp_path/"state"),
            "--output",str(tmp_path/"map.tif"),"--climate-output",str(tmp_path/"climate.tif"),
            "--conditioning-dir",str(tmp_path),"--snr",".2,.2,1,.2,1"]
    assert main(args) == 0
    _,meta = load_state(tmp_path/"state")
    assert "conditioning_tiffs" in meta
    assert (tmp_path/"state/conditioning/temperature.tif").exists()
    with rasterio.open(tmp_path/"climate.tif") as ds:
        assert ds.count == 5 and ds.shape == (16,32)
        assert np.isfinite(ds.read()).all()
    cmd,_,_ = build_command(tmp_path/"run",42,conditioning_dir=str(tmp_path))
    assert cmd[cmd.index("--conditioning-dir")+1] == str(tmp_path.resolve())
    with pytest.raises(ValueError,match="either"):
        build_command(tmp_path/"run",42,draft="draft.png",conditioning_dir=str(tmp_path))


@pytest.mark.parametrize("extra",[["--snr","1,1,1,1,1"],
                                  ["--conditioning-dir","maps","--draft","map.png"],
                                  ["--conditioning-dir","maps","--geometry","equirectangular"]])
def test_cli_rejects_incompatible_options_before_model_loading(tmp_path,extra):
    with pytest.raises(SystemExit) as exc:
        main(["generate","--state",str(tmp_path/"state"),"--output",str(tmp_path/"out.tif"),*extra])
    assert exc.value.code == 2
