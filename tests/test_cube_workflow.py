import numpy as np
import pytest
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.storage import save_state, load_state, verify_state, export_tiff
from planet_diffusion import cube
from planet_diffusion.cli import main


def test_all_stages_use_shared_cube_nodes_and_resume(tmp_path):
    stages = set()
    def audit(name,a):
        members,ids,owners,counts = cube.edge_groups(a.shape[-1]-1)
        flat = a.reshape(a.shape[0],-1)
        np.testing.assert_array_equal(flat[:,members],flat[:,owners[ids]])
        stages.add(name)
    backend = DiagnosticBackend()
    a,meta = generate_cube(backend,42,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                          audit=audit,checkpoint_dir=tmp_path/"checkpoints")
    assert {"conditioning","coarse","latent","decoder-noise","residual","cube-elevation"} <= stages
    b,_ = generate_cube(backend,42,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                        checkpoint_dir=tmp_path/"checkpoints")
    np.testing.assert_array_equal(a,b)
    independent,_ = generate_cube(DiagnosticBackend(),42,face_coarse=2,coarse_steps=2,
                                   progress=lambda _:None)
    np.testing.assert_array_equal(a,independent)
    with pytest.raises(ValueError,match="different parameters"):
        generate_cube(backend,43,face_coarse=2,coarse_steps=2,progress=lambda _:None,
                      checkpoint_dir=tmp_path/"checkpoints")
    meta["radius_metres"] = 6371000.
    save_state(tmp_path/"state",a,meta)
    a,meta = load_state(tmp_path/"state")
    assert verify_state(a,meta)["passed"]
    export_tiff(tmp_path/"planet.tif",a,meta,height=64)


def test_cli_defaults_to_cube(tmp_path):
    assert main(["generate", "--backend", "diagnostic", "--seed", "42",
                 "--coarse-height", "4", "--coarse-steps", "2", "--height", "64", "--keep-checkpoints",
                 "--state", str(tmp_path/"state"), "--output", str(tmp_path/"map.tif")]) == 0
    _, meta = load_state(tmp_path/"state")
    assert meta["algorithm"] == "cubed-sphere-v3"
    assert (tmp_path/"state.checkpoints"/"cube-elevation.npy").exists()
