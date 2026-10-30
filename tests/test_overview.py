import numpy as np
import pytest

from planet_diffusion import cube, overview as ov
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cli import main
from planet_diffusion.cube_generate import generate_cube, remove_checkpoints
from planet_diffusion.detail import sample_latents
from planet_diffusion.sparse import Noise
from planet_diffusion.storage import load_state, verify_state

quiet = lambda *_: None


def coarse_field(nc, seed=1):
    rng = np.random.default_rng(seed)
    scale = np.array([30, 30, 10, 300, 800, 30], np.float32)[:, None, None, None]
    mean = np.array([0, 0, 15, 300, 1300, 50], np.float32)[:, None, None, None]
    return cube.identify(rng.normal(size=(6, 6, nc+1, nc+1)).astype(np.float32)*scale+mean)


def test_streamed_passes_reproduce_dense_latent_stage(tmp_path):
    backend, coarse, seed = DiagnosticBackend(), coarse_field(4), 42
    n = 4*32
    dense = sample_latents(backend, coarse, seed, progress=quiet, batch_size=3)
    sources = [ov.ArraySource(cube.noise(seed, 5819, 5, n)), ov.ArraySource(cube.noise(seed, 5820, 5, n))]
    streamed = ov.stream_latents(backend, coarse, seed, tmp_path, quiet, batch_size=5,
                                 final_channels=range(5), noise_sources=sources)
    np.testing.assert_allclose(np.asarray(streamed), dense, atol=1e-5, rtol=1e-5)


def test_tile_noise_matches_sparse_noise_including_halos_and_corners():
    n = 128
    noise, reference = ov.TileNoise(7, 5819, 5, n, ov.EdgeOwners(n)), Noise(7, 5819, 5, n)
    for f in range(6):
        for y in (-32, 0, 32, n-32, n):
            band = ov.extended_band(noise, f, y, n)
            for x in (-32, 0, 32, n-32, n):
                expected = reference.read(f, np.arange(y, y+64)[:, None], np.arange(x, x+64)[None, :], nearest=True)
                np.testing.assert_array_equal(band[:, :, x+32:x+96], expected)


def test_overview_map_area_averages_cube_field():
    n, height = 96, 32
    p = np.stack([cube.directions(f, np.arange(n+1)[:, None], np.arange(n+1)[None, :], n) for f in range(6)])
    metres = 3000*p[..., 0]+1500*p[..., 2]
    z = np.sign(metres)*np.sqrt(np.abs(metres))
    lowfreq = ((z-ov.LOWFREQ_MEAN)/ov.LOWFREQ_STD).astype(np.float32)
    grid = ov.to_overview_map(cube.identify(lowfreq[None])[0], height)
    lon = np.linspace(-np.pi, np.pi, 2*height, endpoint=False)[None, :]
    theta = (np.pi*np.arange(height+1)/height)[:, None]
    expected = 3000*np.sin(theta)*np.cos(lon)+1500*np.cos(theta)
    assert grid.shape == (height+1, 2*height)
    assert np.abs(grid-expected)[2:-2].max() < 60          # node-cell averaging of a smooth field
    assert np.ptp(grid[0]) == 0 and np.ptp(grid[-1]) == 0


def test_interrupted_overview_resumes_to_identical_map(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "CHECKPOINT_SECONDS", 0)
    coarse, seed = coarse_field(4, 3), 11

    class Failing(DiagnosticBackend):
        calls = 0
        def predict_batch(self, stage, a, cond, t):
            Failing.calls += 1
            if Failing.calls == 40:
                raise RuntimeError("stop")
            return super().predict_batch(stage, a, cond, t)

    with pytest.raises(RuntimeError, match="stop"):
        ov.generate_overview(Failing(), coarse, seed, tmp_path/"a", 40, quiet, batch_size=7)
    assert (tmp_path/"a"/"progress.npz").exists()
    messages = []
    resumed, info = ov.generate_overview(DiagnosticBackend(), coarse, seed, tmp_path/"a", 40, messages.append, 7)
    assert any("resuming latent pass" in m for m in messages)
    straight, _ = ov.generate_overview(DiagnosticBackend(), coarse, seed, tmp_path/"b", 40, quiet, 7)
    np.testing.assert_array_equal(resumed, straight)
    assert info["face_latent_intervals"] == 128 and info["height"] == 40
    with pytest.raises(ValueError, match="different settings"):
        ov.generate_overview(DiagnosticBackend(), coarse, seed, tmp_path/"a", 42, quiet, 7)


def test_cli_overview_writes_whole_globe_map_and_cleans_up(tmp_path, capsys):
    args = ["generate", "--backend", "diagnostic", "--seed", "5", "--coarse-height", "4", "--coarse-steps", "2",
            "--detail", "overview", "--latent-batch-size", "8"]
    assert main(args+["--state", str(tmp_path/"s"), "--output", str(tmp_path/"o.tif")]) == 0
    a, meta = load_state(tmp_path/"s")
    assert meta["output"] == "latent-overview" and a.shape == (23, 44) and verify_state(a, meta)["passed"]
    assert meta["overview"]["latent_cells_per_pixel_equator"] == pytest.approx(256/44)
    err = capsys.readouterr().err
    assert "overview latent 2/2" in err and "Removed checkpoints" in err and not (tmp_path/"s.checkpoints").exists()
    for extra in (["--height", "80"], ["--coarse-only"], ["--regional-only", "--bounds", "0", "0", "1", "1",
                                                          "--width", "4", "--height", "4"]):
        with pytest.raises(SystemExit):
            main(args+extra+["--state", str(tmp_path/"bad"), "--output", str(tmp_path/"bad.tif")])


def test_coarse_preview_continues_to_overview_in_same_folder(tmp_path, capsys):
    backend = DiagnosticBackend()
    ck = tmp_path/"ck"
    generate_cube(backend, 9, face_coarse=2, coarse_steps=2, progress=quiet, checkpoint_dir=ck, coarse_only=True)
    grid, meta = generate_cube(backend, 9, face_coarse=2, coarse_steps=2, progress=print, checkpoint_dir=ck,
                               detail="overview", overview_height=20)
    assert "resumed coarse" in capsys.readouterr().out and grid.shape == (21, 40)
    assert (ck/"overview"/"latent-pass-1.npy").exists()
    remove_checkpoints(ck)
    assert not ck.exists()


def test_launcher_overview_settings(tmp_path):
    import json
    from planet_diffusion.gui import build_command, load_launch_settings
    cmd, seed, _ = build_command(tmp_path, 3, coarse_height=864, export_height=4096, export_width=8192, detail="overview")
    assert cmd[cmd.index("--detail")+1] == "overview" and cmd[cmd.index("--height")+1] == "4096"
    path = tmp_path/"launch.json"
    path.write_text(json.dumps({"seed": seed, "command": cmd}), encoding="utf-8")
    assert load_launch_settings(path)["detail"] == "overview"
    assert "--detail" not in build_command(tmp_path/"f", 3)[0]
    with pytest.raises(ValueError, match="at most 1152"):
        build_command(tmp_path/"h", 3, coarse_height=72, export_height=4096, export_width=8192, detail="overview")
    with pytest.raises(ValueError, match="whole-globe"):
        build_command(tmp_path/"c", 3, coarse_height=864, detail="overview", coarse_only=True)
