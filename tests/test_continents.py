import json
import subprocess
import sys

import numpy as np
import pytest

from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cli import main
from planet_diffusion.continents import ContinentGuide, cascade_heights, gaussian_blur, to_metres, upsample
from planet_diffusion.cube_generate import generate_cube, remove_checkpoints
from planet_diffusion.gui import build_command, load_launch_settings
from planet_diffusion.storage import load_state, verify_state

quiet = lambda _: None


def test_blur_preserves_constants_and_shared_nodes():
    a = np.full((2, 6, 17, 17), 3., np.float32)
    np.testing.assert_allclose(gaussian_blur(a, 2.5), 3, atol=1e-5)
    b = gaussian_blur(cube.noise(1, 0, 1, 16), 2.)
    members, ids, owners, _ = cube.edge_groups(16)
    flat = b.reshape(1, -1)
    np.testing.assert_array_equal(flat[:, members], flat[:, owners[ids]])
    assert b.std() < .5


def test_upsample_interpolates_and_keeps_pass_relief():
    n, N = 32, 96
    def field(m):
        p = np.stack([cube.directions(f, np.arange(m+1)[:, None], np.arange(m+1)[None, :], m) for f in range(6)])
        return (100*p[..., 0]*p[..., 2])[None].astype(np.float32)
    coarse = cube.identify(field(n))
    fine = upsample(coarse, N)
    np.testing.assert_allclose(fine, field(N), atol=.01)            # bilinear+blur (v3) erred by ~0.07+
    np.testing.assert_allclose(upsample(coarse, 2*n)[:, :, ::2, ::2], coarse, atol=1e-5)
    members, ids, owners, _ = cube.edge_groups(N)
    flat = fine.reshape(1, -1)
    np.testing.assert_array_equal(flat[:, members], flat[:, owners[ids]])
    np.testing.assert_allclose(upsample(np.full((2, 6, 9, 9), 4., np.float32), 20), 4, atol=1e-5)
    # Cell-scale ridges keep far more amplitude than a blurred bilinear resize.
    ridges = cube.noise(3, 0, 1, 24)
    def relief(a):
        return float((a-gaussian_blur(a, 2.)).std())
    blurred = gaussian_blur(cube.resize(ridges, 42), .5*42/24)
    assert relief(upsample(ridges, 42)) > 1.4*relief(blurred)


def test_cascade_heights_are_geometric_multiples_of_four():
    assert cascade_heights(48, 864, 3) == [48, 124, 328]
    assert cascade_heights(48, 864, 5) == [48, 204]
    assert cascade_heights(48, 96, 3) == [48]
    assert cascade_heights(48, 48, 3) == [48]
    for start, final, step in ((36, 864, 3), (8, 864, 2), (72, 2604, 3)):
        h = cascade_heights(start, final, step)
        ratios = np.array(h[1:]+[final])/np.array(h)
        assert all(v % 4 == 0 for v in h) and h == sorted(set(h)) and h[0] == start
        assert ratios.max() <= step*1.15


def test_each_pass_is_guided_by_the_previous_one():
    backend = DiagnosticBackend()
    seen = []
    class Recording(ContinentGuide):
        def _pass(self, index, backend, raw, label):
            seen.append((index, raw.shape[-1]-1, np.asarray(backend.cond_snr).copy(), raw.copy()))
            return super()._pass(index, backend, raw, label)
    g = Recording(backend, guide_height=8, step=2., steps=2, refinement=1.5, progress=quiet)
    guide = g.conditioning(42, 24)
    assert [h for h, _ in g.levels] == [8, 16, 28]
    assert [n for _, n, _, _ in seen] == [4, 8, 14]
    np.testing.assert_allclose(seen[0][2], backend.cond_snr)  # rough pass: procedural guide settings
    assert seen[1][2][0] == np.float32(1.5) and seen[2][2][0] == np.float32(1.5)
    for (_, out), (_, _, _, raw) in zip(g.levels[:-1], seen[1:]):
        np.testing.assert_allclose(raw, upsample(out, raw.shape[-1]-1))
    np.testing.assert_allclose(guide, upsample(g.levels[-1][1], 24), atol=1e-5)
    assert guide.shape == (5, 6, 25, 25) and np.isfinite(guide).all()
    again = ContinentGuide(backend, 8, step=2., steps=2, refinement=1.5, progress=quiet).conditioning(42, 24)
    np.testing.assert_array_equal(guide, again)
    flat = to_metres(ContinentGuide(backend, 8, .25, 2., 2, 1.5, progress=quiet).conditioning(42, 24))[0]
    metres = to_metres(guide)[0]
    land = metres > 0
    np.testing.assert_allclose(flat[land], .25*metres[land], rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(flat[~land], metres[~land], rtol=1e-4, atol=1e-2)


def test_guide_validation():
    backend = DiagnosticBackend()
    for kwargs in (dict(guide_height=6), dict(guide_height=50), dict(relief=-1), dict(step=1.2),
                   dict(refinement=0)):
        with pytest.raises(ValueError):
            ContinentGuide(backend, **kwargs)
    with pytest.raises(ValueError, match="exceeds"):
        ContinentGuide(backend, 48, progress=quiet).conditioning(1, 8)


def test_continent_coarse_preview_then_full_run_resumes(tmp_path):
    backend = DiagnosticBackend()
    checkpoints = tmp_path/"ck"
    def run(**extra):
        return generate_cube(backend, 7, face_coarse=2, coarse_steps=2, progress=quiet, checkpoint_dir=checkpoints,
                             coarse_pooling=8, continents=ContinentGuide(backend, 8, steps=2, progress=quiet), **extra)
    preview, meta = run(coarse_only=True)
    assert preview.shape == (33, 64) and meta["output"] == "coarse-only"
    assert meta["continent_guide"]["version"] == "continent-cascade-v4"
    for name in ("continent-pass-8", "continent-pass-16", "continent-guide", "coarse-model", "coarse"):
        assert (checkpoints/f"{name}.npy").exists(), name
    full, _ = run(keep_checkpoints=False)
    assert full.shape == (2*512+1, 4*512) and (checkpoints/"latent.npy").exists()
    assert not (checkpoints/"residual.npy").exists() and not (checkpoints/"cube-elevation.npy").exists()
    independent, _ = generate_cube(backend, 7, face_coarse=2, coarse_steps=2, progress=quiet, coarse_pooling=8,
                                   continents=ContinentGuide(backend, 8, steps=2, progress=quiet))
    np.testing.assert_array_equal(full, independent)
    with pytest.raises(ValueError, match="only one"):
        generate_cube(backend, 7, face_coarse=2, coarse_steps=2, progress=quiet, draft=object(),
                      continents=ContinentGuide(backend, 8, progress=quiet))


def test_remove_checkpoints_only_touches_generator_files(tmp_path):
    backend = DiagnosticBackend()
    ck = tmp_path/"ck"
    generate_cube(backend, 3, face_coarse=2, coarse_steps=2, progress=quiet, checkpoint_dir=ck)
    (ck/"notes.txt").write_text("mine")
    (ck/"mine").mkdir()
    assert remove_checkpoints(ck) > 0
    assert sorted(p.name for p in ck.iterdir()) == ["mine", "notes.txt"]
    assert remove_checkpoints(tmp_path/"nothing") == 0
    plain = tmp_path/"plain"; plain.mkdir(); (plain/"x.npy").write_bytes(b"1")
    assert remove_checkpoints(plain) == 0 and (plain/"x.npy").exists()


def test_cli_removes_checkpoints_unless_kept_and_preview_keeps_them(tmp_path, capsys):
    args = ["generate", "--backend", "diagnostic", "--seed", "5", "--coarse-height", "4", "--coarse-steps", "2",
            "--radius-metres", "300000", "--coarse-pooling", "auto", "--continents", "--continent-guide-height", "8"]
    assert main(args+["--coarse-only", "--state", str(tmp_path/"cs"), "--output", str(tmp_path/"c.tif"),
                      "--checkpoint-dir", str(tmp_path/"ck")]) == 0
    a, meta = load_state(tmp_path/"cs")
    assert meta["coarse_pooling"]["factor"] == 10 and a.shape == (41, 80) and verify_state(a, meta)["passed"]
    assert (tmp_path/"ck"/"coarse-model.npy").exists()
    err = capsys.readouterr().err
    assert "Continent passes: rough guide 8" in err and "continent passes: guide 8 -> 16 -> 40" in err
    assert main(args+["--height", "64", "--state", str(tmp_path/"s"), "--output", str(tmp_path/"p.tif"),
                      "--checkpoint-dir", str(tmp_path/"ck")]) == 0
    err = capsys.readouterr().err
    assert "resumed coarse" in err and "Removed checkpoints" in err and not (tmp_path/"ck").exists()
    assert main(args+["--height", "64", "--keep-checkpoints", "--state", str(tmp_path/"k"),
                      "--output", str(tmp_path/"k.tif")]) == 0
    assert (tmp_path/"k.checkpoints"/"cube-elevation.npy").exists()
    for extra in (["--coarse-pool-mode", "max"], ["--draft", "x.png"], ["--continent-detail", "0.5"],
                  ["--regional-only", "--bounds", "0", "0", "1", "1", "--width", "4", "--height", "4"]):
        with pytest.raises(SystemExit):
            main(args+extra+["--state", str(tmp_path/"bad"), "--output", str(tmp_path/"bad.tif")])
    assert not (tmp_path/"bad").exists()


def test_launcher_continent_preview_and_checkpoint_settings(tmp_path):
    cmd, seed, folder = build_command(tmp_path, 3, coarse_height=72, export_height=4096, export_width=8192,
                                      coarse_pooling="auto", continents=True, continent_guide_height=36,
                                      continent_relief=.8, continent_step=2.5, coarse_only=True)
    assert cmd[cmd.index("--state")+1].endswith("coarse-state") and "--height" not in cmd
    assert "--coarse-only" in cmd and "--keep-checkpoints" not in cmd and "--continent-detail" not in cmd
    path = tmp_path/"launch.json"
    path.write_text(json.dumps({"seed": seed, "command": cmd}), encoding="utf-8")
    settings = load_launch_settings(path)
    assert settings["continents"] and settings["coarse_only"] and not settings["keep_checkpoints"]
    assert settings["continent_refinement"] == "" and "--continent-refinement" not in cmd
    assert (settings["continent_guide"], settings["continent_relief"], settings["continent_step"]) == ("36", "0.8", "2.5")
    refined, seed2, _ = build_command(tmp_path/"r", 3, continents=True, continent_refinement="1")
    assert refined[refined.index("--continent-refinement")+1] == "1.0"
    path.write_text(json.dumps({"seed": seed2, "command": refined}), encoding="utf-8")
    assert load_launch_settings(path)["continent_refinement"] == "1.0"
    assert load_launch_settings(path)["warnings"] == []
    with pytest.raises(ValueError):
        build_command(tmp_path/"r2", 3, continents=True, continent_refinement="5")
    with pytest.raises(ValueError):
        build_command(tmp_path/"r3", 3, continents=True, continent_refinement="abc")
    kept, _, _ = build_command(tmp_path/"k", 3, keep_checkpoints=True)
    assert "--keep-checkpoints" in kept
    old = cmd+["--continent-detail", "0.6"]
    path.write_text(json.dumps({"seed": seed, "command": old}), encoding="utf-8")
    assert "--continent-detail 0.6" in load_launch_settings(path)["warnings"]
    with pytest.raises(ValueError, match="Max pooling"):
        build_command(tmp_path/"m", 3, coarse_pooling="12", coarse_pool_mode="max")
    with pytest.raises(ValueError):
        build_command(tmp_path/"g", 3, continents=True, continent_step=1.)


def test_preview_tool_renders_all_passes(tmp_path):
    backend = DiagnosticBackend()
    generate_cube(backend, 9, face_coarse=2, coarse_steps=2, progress=quiet, checkpoint_dir=tmp_path/"checkpoints",
                  coarse_pooling=8, continents=ContinentGuide(backend, 8, steps=2, progress=quiet), coarse_only=True)
    result = subprocess.run([sys.executable, "tools/preview_layout.py", str(tmp_path), "--width", "256"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/"checkpoints"/"layout-preview.png").exists()
