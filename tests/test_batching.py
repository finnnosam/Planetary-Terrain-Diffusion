import numpy as np
import pytest

from planet_diffusion import cube
from planet_diffusion.backends import DiagnosticBackend, TerrainBackend
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.gui import build_command
from planet_diffusion.cli import main


@pytest.mark.parametrize('platform,device,existing,expected', [
    ('win32', 'cuda', 1., .65), ('win32', 'cuda:1', .4, .4),
    ('win32', 'cpu', 1., None), ('linux', 'cuda', 1., None),
])
def test_windows_cuda_workspace_budget(monkeypatch, platform, device, existing, expected):
    from types import SimpleNamespace
    from planet_diffusion import backends
    calls = []
    fake_torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name.split(':')[0],
                                            index=int(name.split(':')[1]) if ':' in name else None),
        cuda=SimpleNamespace(current_device=lambda: 0, get_per_process_memory_fraction=lambda dev: existing,
                             set_per_process_memory_fraction=lambda *args: calls.append(args)))
    monkeypatch.setattr(backends.sys, 'platform', platform)
    assert backends._limit_windows_cuda_memory(fake_torch, device) == expected
    assert calls == ([] if expected is None else [(expected, 1 if device == 'cuda:1' else 0)])


def test_windows_cuda_budget_supports_older_torch(monkeypatch):
    from types import SimpleNamespace
    from planet_diffusion import backends
    calls = []
    fake_torch = SimpleNamespace(device=lambda _: SimpleNamespace(type='cuda', index=None),
        cuda=SimpleNamespace(current_device=lambda: 0, set_per_process_memory_fraction=lambda *args: calls.append(args)))
    monkeypatch.setattr(backends.sys, 'platform', 'win32')
    assert backends._limit_windows_cuda_memory(fake_torch, 'cuda') == .65
    assert calls == [(.65, 0)]


@pytest.mark.parametrize('batch_size', [1, 7, 100])
def test_consensus_batch_preserves_order_halos_and_progress(batch_size):
    a = cube.noise(42, 1, 2, 8)
    locations, batches, progress = [], [], []
    def predict(a, f, y, x):
        locations.append((f, y, x))
        return a * .25 + f + y * .1 + x * .01
    expected = cube.consensus(a, predict, 8, 4, progress=lambda *v: progress.append(v))
    actual_locations, actual_progress = [], []
    def predict_batch(a, locs):
        batches.append(len(a))
        actual_locations.extend(locs)
        return np.stack([v * .25 + f + y * .1 + x * .01 for v, (f, y, x) in zip(a, locs)])
    actual = cube.consensus(a, None, 8, 4, batch_size=batch_size, predict_batch=predict_batch,
                            progress=lambda *v: actual_progress.append(v))
    np.testing.assert_array_equal(actual, expected)
    assert locations == actual_locations
    assert progress == actual_progress
    assert max(batches) <= batch_size
    assert len(batches) == (len(locations) + batch_size - 1) // batch_size


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_batch_size(value):
    with pytest.raises(ValueError, match='positive integer'):
        generate_cube(DiagnosticBackend(), 1, latent_batch_size=value)
    with pytest.raises(ValueError, match='positive integer'):
        cube.consensus(cube.noise(1, 0, 1, 2), None, 4, 2, batch_size=value)


def test_regional_batching_and_resume(tmp_path):
    class Counting(DiagnosticBackend):
        batches = 0
        def predict_batch(self, *args):
            self.batches += 1
            return super().predict_batch(*args)
    args = dict(seed=42, face_coarse=2, coarse_steps=2, progress=lambda _: None,
                region=((-5,-5,5,5), 4, 4))
    single, _ = generate_cube(DiagnosticBackend(), **args)
    backend = Counting()
    batched, metadata = generate_cube(backend, **args, latent_batch_size=7, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(single, batched)
    assert backend.batches == 28  # 96 patches per pass including terminal nodes; two passes.
    assert metadata['latent_batch_size'] == 7
    resumed, _ = generate_cube(backend, **args, latent_batch_size=7, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(batched, resumed)
    assert backend.batches == 28
    with pytest.raises(ValueError, match='different parameters'):
        generate_cube(backend, **args, latent_batch_size=1, checkpoint_dir=tmp_path)


def test_terrain_batch_conditioning_matches_single_patch_formula():
    torch = pytest.importorskip('torch')
    backend = TerrainBackend.__new__(TerrainBackend)
    backend.torch, backend.device = torch, 'cpu'
    backend.histogram = np.arange(5, dtype=np.float32)
    backend.mp_concat = lambda values, dim: torch.cat(values, dim)
    calls = []
    def model(a, noise_labels, conditional_inputs):
        calls.append((a.clone(), noise_labels.clone(), conditional_inputs[0].clone()))
        return a + conditional_inputs[0].mean(1)[:, None, None, None]
    backend.models = {'latent': model}
    rng = np.random.default_rng(42)
    a = rng.normal(size=(3, 5, 64, 64)).astype(np.float32)
    cond = rng.normal(size=(3, 6, 4, 4)).astype(np.float32)
    actual = backend.predict_batch('latent', a, cond, .7)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][1], [.7]*3)
    for i in range(3):
        raw = np.concatenate([cond[i], np.ones((1,4,4))])
        c = ((raw-np.array([14.99,11.65,15.87,619.26,833.12,69.40,.66])[:,None,None]) /
             np.array([21.72,21.78,10.40,452.29,738.09,34.59,.47])[:,None,None]).astype(np.float32)
        expected = np.concatenate([c[:1].ravel(), c[1:2].ravel(), c[2:6,1:3,1:3].mean((1,2)),
                                   c[6:7].ravel(), backend.histogram, [-np.sqrt(3)]])
        np.testing.assert_allclose(calls[0][2][i], expected, rtol=1e-6)
        np.testing.assert_allclose(actual[i], a[i]+expected.astype(np.float32).mean(), atol=1e-6)


def test_launcher_and_cli_validation(tmp_path):
    cmd, _, _ = build_command(tmp_path, 42, latent_batch_size=8)
    assert cmd[cmd.index('--latent-batch-size')+1] == '8'
    with pytest.raises(ValueError, match='positive integer'):
        build_command(tmp_path, 42, latent_batch_size=0)
    with pytest.raises(SystemExit) as exc:
        main(['generate', '--state', str(tmp_path/'state'), '--output', str(tmp_path/'out.tif'),
              '--backend', 'diagnostic', '--latent-batch-size', '0'])
    assert exc.value.code == 2
