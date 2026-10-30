import math

import numpy as np
import pytest

from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cli import main
from planet_diffusion.coarse import pool_coarse
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.gui import build_command, load_launch_settings
from planet_diffusion.model_presets import auto_coarse_pooling, parse_coarse_pooling
from planet_diffusion.storage import load_state


def smooth(n, channels=6):
    p = np.stack([cube.directions(f, np.arange(n+1)[:, None], np.arange(n+1)[None, :], n)
                  for f in range(6)])
    base = 1000*p[..., 0]+400*p[..., 2]
    return cube.identify(np.stack([base+10*c for c in range(channels)]).astype(np.float32))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_pooling_preserves_constants_and_shared_nodes(k):
    a = np.full((6, 6, 8*k+1, 8*k+1), 5., np.float32)
    for mode in ("avg", "max"):
        pooled = pool_coarse(a, k, mode)
        assert pooled.shape == (6, 6, 9, 9)
        np.testing.assert_allclose(pooled, 5, atol=1e-4)
    pooled = pool_coarse(smooth(8*k), k)
    members, ids, owners, _ = cube.edge_groups(8)
    flat = pooled.reshape(6, -1)
    np.testing.assert_array_equal(flat[:, members], flat[:, owners[ids]])


def test_avg_is_area_mean_and_max_keeps_peaks_and_low_p5():
    k, n = 4, 32
    fine = smooth(n)
    avg = pool_coarse(fine, k, "avg")
    extreme = pool_coarse(fine, k, "max")
    # A smooth field's footprint mean stays near its centre value.
    np.testing.assert_allclose(avg, fine[:, :, ::k, ::k], atol=30)
    assert (extreme[0] >= avg[0]-1e-3).all() and (extreme[1] <= avg[1]+1e-3).all()
    np.testing.assert_allclose(extreme[2:], avg[2:])
    spike = np.zeros((6, 6, n+1, n+1), np.float32)
    spike[0, 0, 13, 17] = 4000.
    assert pool_coarse(spike, k, "max")[0].max() == 4000.
    assert pool_coarse(spike, k, "avg")[0].max() < 4000./k


def test_pooling_rejects_nondividing_factor():
    with pytest.raises(ValueError, match="divide"):
        pool_coarse(np.zeros((6, 6, 11, 11), np.float32), 3)


def test_auto_factor_matches_model_coarse_cell():
    assert parse_coarse_pooling("AUTO") == "auto" and parse_coarse_pooling(" 3") == 3
    for bad in ("0", "-2", "1.5", "x"):
        with pytest.raises(ValueError):
            parse_coarse_pooling(bad)
    assert auto_coarse_pooling(6371000., 72, 90.) == 12
    assert auto_coarse_pooling(6371000., 868, 90.) == 1
    assert auto_coarse_pooling(1000., 16, 90.) == 1
    k = auto_coarse_pooling(6371000., 32, 30.)
    assert abs(math.pi*6371000./(32*k)/7680.-1) < .02


def test_pooled_generation_runs_coarse_on_finer_grid_and_resumes(tmp_path):
    backend = DiagnosticBackend()
    stages = {}
    audit = lambda name, a: stages.setdefault(name, a.shape)
    run = dict(face_coarse=2, coarse_steps=2, progress=lambda _: None)
    a, meta = generate_cube(backend, 42, audit=audit, checkpoint_dir=tmp_path/"ck",
                            coarse_pooling=3, **run)
    assert stages["raw-conditioning"][-1] == 2*3+1
    assert stages["coarse-model"] == (6, 6, 7, 7)
    assert stages["coarse"] == (6, 6, 3, 3)
    assert stages["latent"][-1] == 2*32+1
    assert meta["coarse_pooling"] == {"version": "cube-coarse-pooling-v1", "factor": 3, "mode": "avg",
                                      "model_coarse_height": 12, "model_face_coarse_intervals": 6}
    assert meta["procedural_conditioning"]["radius_coarse_cells"] == pytest.approx(12/np.pi)
    assert (tmp_path/"ck"/"coarse-model.npy").exists()
    b, _ = generate_cube(backend, 42, checkpoint_dir=tmp_path/"ck", coarse_pooling=3, **run)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="different parameters"):
        generate_cube(backend, 42, checkpoint_dir=tmp_path/"ck", coarse_pooling=3,
                      coarse_pool_mode="max", **run)
    unpooled, meta1 = generate_cube(backend, 42, **run)
    assert meta1["coarse_pooling"]["factor"] == 1
    assert a.shape == unpooled.shape and not np.array_equal(a, unpooled)


def test_cli_auto_pooling_and_regional_rejection(tmp_path, capsys):
    assert main(["generate", "--backend", "diagnostic", "--seed", "7", "--coarse-height", "4",
                 "--coarse-steps", "2", "--height", "32", "--radius-metres", "100000",
                 "--coarse-pooling", "auto", "--state", str(tmp_path/"state"),
                 "--output", str(tmp_path/"map.tif")]) == 0
    _, meta = load_state(tmp_path/"state")
    expected = auto_coarse_pooling(100000., 4, 90.)
    assert expected == 3 and meta["coarse_pooling"]["factor"] == expected
    assert "Coarse pooling 3 (avg)" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["generate", "--backend", "diagnostic", "--coarse-height", "8", "--regional-only",
              "--bounds", "0", "0", "10", "10", "--width", "8", "--height", "8",
              "--coarse-pooling", "2", "--state", str(tmp_path/"r"), "--output", str(tmp_path/"r.tif")])


def test_scale_note_for_unpooled_earth(tmp_path, capsys):
    assert main(["generate", "--backend", "diagnostic", "--seed", "7", "--coarse-height", "4",
                 "--coarse-steps", "2", "--height", "16", "--state", str(tmp_path/"state"),
                 "--output", str(tmp_path/"map.tif")]) == 0
    assert "Scale note" in capsys.readouterr().err


def test_launcher_round_trips_pooling(tmp_path):
    cmd, seed, _ = build_command(tmp_path, 42, coarse_height=72, coarse_pooling="auto", coarse_pool_mode="max")
    assert cmd[cmd.index("--coarse-pooling")+1] == "auto" and cmd[cmd.index("--coarse-pool-mode")+1] == "max"
    path = tmp_path/"launch.json"
    import json
    path.write_text(json.dumps({"seed": seed, "command": cmd}), encoding="utf-8")
    settings = load_launch_settings(path)
    assert settings["coarse_pooling"] == "auto" and settings["coarse_pool_mode"] == "max"
    plain, _, _ = build_command(tmp_path/"plain", 42)
    assert "--coarse-pooling" not in plain
    with pytest.raises(ValueError, match="pooling"):
        build_command(tmp_path/"r", 42, coarse_height=8, bounds=(0, 0, 10, 10), export_width=8,
                      export_height=8, regional_only=True, coarse_pooling="2")
