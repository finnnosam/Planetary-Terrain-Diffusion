import json
import numpy as np
from PIL import Image
import pytest
from planet_diffusion.draft import Draft
from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.cli import main
from planet_diffusion.seeds import resolve_seed
from planet_diffusion.storage import load_state
from planet_diffusion.gui import build_command


def png(tmp_path,values,name="draft.png"):
    path = tmp_path/name
    Image.fromarray(values).save(path)
    return path


def test_linear_elevation_mapping_and_16bit(tmp_path):
    for code,expected in [(0,-2000),(64,-2000+6000*64/255),(128,-2000+6000*128/255),(255,4000)]:
        draft = Draft(png(tmp_path,np.full((4,8),code,np.uint8)))
        guide = draft.conditioning(42,2)[0]
        np.testing.assert_allclose(np.sign(guide)*guide**2,expected,rtol=1e-6)
    path = png(tmp_path,np.full((4,8),32768,np.uint16))
    value = Draft(path).evaluate(np.array([1.,0.,0.]))[0]
    assert np.isclose(value,-2000+6000*32768/65535)
    path = png(tmp_path,np.zeros((4,8),np.uint8))
    assert np.all(Draft(path,ocean_depth=0).conditioning(42,2)[0] < 0)


def test_orientation_wrap_poles_and_alpha(tmp_path):
    image = np.zeros((32,64),np.uint8)
    image[:16,32:] = 255  # NE quadrant, east of Greenwich.
    draft = Draft(png(tmp_path,image))
    p = np.array([[.5,.5,2**-.5],[.5,.5,-2**-.5],[.5,-.5,2**-.5]])
    np.testing.assert_allclose(draft.evaluate(p)[0],[4000,-2000,-2000],atol=.01)
    seam = np.array([[-1.,1e-9,0.],[-1.,-1e-9,0.]])
    assert np.ptp(draft.evaluate(seam)[0]) < .01
    poles = np.array([[1e-9,0.,1.],[-1e-9,0.,1.],[0.,1e-9,1.]])
    assert np.ptp(draft.evaluate(poles)[0]) < .01
    guide = draft.conditioning(42,4)
    members,ids,owners,_ = cube.edge_groups(4)
    flat = guide.reshape(5,-1)
    np.testing.assert_array_equal(flat[:,members],flat[:,owners[ids]])
    transparent = np.full((4,8,4),255,np.uint8)
    transparent[...,3] = 0
    guide = Draft(png(tmp_path,transparent)).conditioning(42,4)
    np.testing.assert_allclose(guide,cube.conditioning(42,4),atol=1e-5)


def test_invalid_draft(tmp_path):
    path = png(tmp_path,np.zeros((4,4),np.uint8))
    with pytest.raises(ValueError,match="2:1"):
        Draft(path)
    path = png(tmp_path,np.zeros((4,8),np.uint8))
    for kwargs in ({"ocean_depth":-1},{"maximum":-2000},{"ocean_depth":float("nan")}):
        with pytest.raises(ValueError): Draft(path,**kwargs)
    path.write_bytes(b"not a PNG")
    with pytest.raises(ValueError,match="Cannot read"): Draft(path)


def test_draft_reaches_generation_and_repeatability(tmp_path):
    path = png(tmp_path,np.full((8,16),180,np.uint8))
    draft = Draft(path)
    seen = {}
    def audit(name,value):
        if name == "raw-conditioning": seen[name] = value.copy()
    args = dict(seed=12,face_coarse=2,coarse_steps=2,progress=lambda _:None,draft=draft)
    a,meta = generate_cube(DiagnosticBackend(),**args,audit=audit,checkpoint_dir=tmp_path/"checkpoints")
    np.testing.assert_array_equal(seen["raw-conditioning"],draft.conditioning(12,2))
    b,_ = generate_cube(DiagnosticBackend(),**args)
    np.testing.assert_array_equal(a,b)
    c,_ = generate_cube(DiagnosticBackend(),12,2,2,progress=lambda _:None)
    assert not np.array_equal(a,c)
    assert meta["draft"]["sha256"] == draft.metadata["sha256"]
    assert (tmp_path/"checkpoints"/"draft.png").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError,match="different parameters"):
        generate_cube(DiagnosticBackend(),12,2,2,progress=lambda _:None,
                      draft=Draft(path,ocean_depth=3000),checkpoint_dir=tmp_path/"checkpoints")


def test_random_seed_cli_and_saved_draft(tmp_path,monkeypatch):
    monkeypatch.setattr("planet_diffusion.seeds.secrets.randbits",lambda bits:123456)
    assert resolve_seed("random") == 123456
    assert resolve_seed("18446744073709551615") == 2**64-1
    for bad in ("-1",str(2**64),"oops"):
        with pytest.raises(ValueError): resolve_seed(bad)
    path = png(tmp_path,np.full((4,8),255,np.uint8))
    assert main(["generate","--backend","diagnostic","--draft",str(path),
                 "--coarse-height","4","--coarse-steps","2","--height","32","--keep-checkpoints",
                 "--state",str(tmp_path/"state"),"--output",str(tmp_path/"map.tif")]) == 0
    _,meta = load_state(tmp_path/"state")
    assert meta["seed"] == 123456
    assert (tmp_path/"state"/"draft.png").read_bytes() == path.read_bytes()
    monkeypatch.setattr("planet_diffusion.seeds.secrets.randbits",lambda bits:789)
    assert resolve_seed("random",tmp_path/"state.checkpoints") == 123456


def test_launcher_passes_paths_and_resumes_seed(tmp_path):
    path = png(tmp_path,np.ones((4,8),np.uint8),"a draft.png")
    folder = tmp_path/"a run"
    (folder/"checkpoints").mkdir(parents=True)
    (folder/"checkpoints"/"identity.json").write_text(json.dumps({"seed":99}))
    cmd,seed,_ = build_command(folder,"random",str(path),8,1500,6000)
    assert seed == 99
    assert cmd[cmd.index("--draft")+1] == str(path.resolve())
    assert cmd[cmd.index("--draft-ocean-depth")+1] == "1500"
    assert cmd[cmd.index("--draft-white-metres")+1] == "6000"
    (folder/"state").mkdir()
    with pytest.raises(ValueError,match="already has output"):
        build_command(folder,99)


def test_native_guide_does_not_overwrite_model_outputs(tmp_path):
    class IgnoreGuide(DiagnosticBackend):
        def coarse_predict(self, a, cond, step):
            return .3*a
    dark = Draft(png(tmp_path,np.zeros((4,8),np.uint8),"dark.png"))
    light = Draft(png(tmp_path,np.full((4,8),255,np.uint8),"light.png"))
    backend = IgnoreGuide()
    original = backend.cond_snr.copy()
    a,meta = generate_cube(backend,42,2,2,progress=lambda _:None,draft=dark)
    b,_ = generate_cube(backend,42,2,2,progress=lambda _:None,draft=light)
    # If the model ignores conditioning, no extra guide can replace its output.
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(backend.cond_snr,original)
    np.testing.assert_allclose(meta["conditioning_noise"],[.2,.2,1,.2,1])


def test_transparent_draft_does_not_change_generated_terrain(tmp_path):
    rgba = np.zeros((4,8,4),np.uint8)
    draft = Draft(png(tmp_path,rgba))
    a,_ = generate_cube(DiagnosticBackend(),42,2,2,progress=lambda _:None,draft=draft)
    equivalent = DiagnosticBackend()
    equivalent.cond_snr = np.array([.2,.2,1.,.2,1.],np.float32)
    b,_ = generate_cube(equivalent,42,2,2,progress=lambda _:None)
    np.testing.assert_array_equal(a,b)



def test_explicit_endpoints_and_saved_mapping(tmp_path):
    path = png(tmp_path,np.full((4,8),64,np.uint8))
    draft = Draft(path,black_metres=-6000,maximum=6000)
    np.testing.assert_allclose(draft.evaluate(np.array([1.,0.,0.]))[0],
                               -6000+12000*64/255,rtol=1e-6)
    restored = Draft.from_metadata(path,json.loads(json.dumps(draft.metadata)))
    assert restored.metadata == draft.metadata
    np.testing.assert_array_equal(restored.conditioning(42,2),draft.conditioning(42,2))
    legacy = Draft(path,_legacy_mapping=True)
    restored = Draft.from_metadata(path,legacy.metadata)
    assert restored.metadata == legacy.metadata
    np.testing.assert_allclose(restored.evaluate(np.array([1.,0.,0.]))[0],4000*64/255,rtol=1e-6)
    for black,white in [(0,0),(100,0),(float('nan'),100),(0,float('inf'))]:
        with pytest.raises(ValueError,match="Black and white"):
            Draft(path,black_metres=black,maximum=white)
    cmd,_,_ = build_command(tmp_path/'linear-run',42,str(path),black_metres=-6000,white_metres=6000)
    assert cmd[cmd.index('--draft-black-metres')+1] == '-6000'
    assert '--draft-ocean-depth' not in cmd


def test_linear_png_sparse_cli_query(tmp_path):
    import rasterio
    path = png(tmp_path,np.full((4,8),128,np.uint8))
    state = tmp_path/'state'
    first,second = tmp_path/'first.tif',tmp_path/'second.tif'
    assert main(['generate','--backend','diagnostic','--regional-only',
                 '--draft',str(path),'--draft-black-metres','-6000',
                 '--draft-white-metres','6000','--seed','42','--coarse-height','4',
                 '--coarse-steps','2','--bounds','-2','-2','2','2',
                 '--width','2','--height','2','--state',str(state),'--output',str(first)]) == 0
    info = json.loads((state/'planet.json').read_text())['draft']
    assert info['mapping_version'] == 'linear-v1'
    assert info['black_metres'] == -6000
    assert main(['query','--state',str(state),'--bounds','-2','-2','2','2',
                 '--width','2','--height','2','--output',str(second)]) == 0
    with rasterio.open(first) as a,rasterio.open(second) as b:
        np.testing.assert_array_equal(a.read(),b.read())
