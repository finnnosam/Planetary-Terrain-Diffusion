"""Input approval stops before inference and only approves unchanged layouts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toy_meso import planet_tiled as tiled, gui
from toy_meso.common import Normalizer
from toy_meso.conditioning import LEGACY_RANDOM_VERSION, RANDOM_VERSION, source_settings, settings_differences


@pytest.fixture
def run(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

        def __call__(self, x, sigma, condition):
            return torch.zeros_like(x)

    network = Model()
    checkpoint = tmp_path / 'run/latest.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'model')
    folder = tmp_path / 'planet'
    checkpoint_meta = dict(step=1, meta=dict(spacing_m=np.pi * 6371000 / 2 / 64, factor=4, size=16))
    norm = Normalizer(dict(cond_mean=0, cond_std=1, res_std=1), 4)
    requested_devices = []

    def load(path, device):
        requested_devices.append(device)
        return network, norm, checkpoint_meta

    monkeypatch.setattr(tiled, 'load_checkpoint', load)
    argv = ['--run', str(checkpoint.parent), '--out', str(folder), '--source', 'random',
            '--bounds', '0', '0', '5', '5', '--width', '16', '--height', '16',
            '--steps', '8', '--batch', '2', '--seed', '42', '--device', 'cpu']
    return argv, folder, network, requested_devices


def test_pause_before_gpu_inference_then_approve_reuses_input(run, monkeypatch):
    argv, folder, model, requested_devices = run
    with monkeypatch.context() as m:
        m.setattr(tiled, 'compute_tiles', lambda *a: pytest.fail('preview must not run the model'))
        assert tiled.main(argv + ['--device', 'cuda', '--pause-for-input']) == 42
    assert requested_devices == ['cpu'] and model.moves == []
    assert not (folder / 'tiles').exists() and not (folder / 'region.tif').exists()
    assert (folder / 'input.preview.png').exists() and (folder / 'random.preview.png').exists()
    assert not json.loads((folder / 'input-review.json').read_text())['approved']
    coarse_stamp = (folder / 'coarse.npy').stat().st_mtime_ns
    tiled.main(argv + ['--pause-for-input', '--approve-input'])
    assert json.loads((folder / 'input-review.json').read_text())['approved']
    assert (folder / 'coarse.npy').stat().st_mtime_ns == coarse_stamp
    assert (folder / 'region.tif').exists()
    # An approved layout can resume/export another filename without another gate.
    tiled.main(argv + ['--pause-for-input', '--name', 'repeat.tif'])
    import rasterio
    with rasterio.open(folder / 'region.tif') as a, rasterio.open(folder / 'repeat.tif') as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))


def test_changed_preview_or_output_scope_requires_fresh_approval(run):
    argv, folder, model, _ = run
    tiled.main(argv + ['--pause-for-input'])
    with (folder / 'input.preview.png').open('ab') as image:
        image.write(b'changed')
    with pytest.raises(ValueError, match='preview changed'):
        tiled.main(argv + ['--approve-input'])
    assert model.moves == []
    assert tiled.main(argv + ['--pause-for-input']) == 42
    with pytest.raises(ValueError, match='preview changed'):
        tiled.main(argv + ['--height', '8', '--approve-input'])
    assert not json.loads((folder / 'input-review.json').read_text())['approved']


def test_approval_cannot_skip_a_missing_preview(run):
    argv, folder, model, _ = run
    with pytest.raises(ValueError, match='missing'):
        tiled.main(argv + ['--approve-input'])
    assert model.moves == [] and not (folder / 'region.tif').exists()


def test_old_random_identity_is_preserved_and_new_runs_use_multiscale():
    legacy = source_settings('random', random_version=LEGACY_RANDOM_VERSION)
    # This identity is the one stored in the user's completed MESO1 run.
    assert legacy['random_source_sha256'] == 'a0ea39532cb94a9ff6354ac7141d8f4194687b56bfcf28c42cccafaf84dc411f'
    current = source_settings('random')
    assert current['random_source_version'] == RANDOM_VERSION
    assert 'random_source_version' in settings_differences(legacy, current)


def test_gui_review_flags_are_optional_and_approval_is_explicit(tmp_path):
    checkpoint = tmp_path / 'latest.pt'
    checkpoint.write_bytes(b'model')
    values = dict(gui.default_settings(), checkpoint=str(checkpoint), folder=str(tmp_path / 'planet'),
                  source='Random terrain', review_input='1')
    cmd, _, _ = gui.prepare_command(values)
    assert '--pause-for-input' in cmd and '--approve-input' not in cmd
    cmd, _, _ = gui.prepare_command(values, approve=True)
    assert '--approve-input' in cmd
    cmd, _, _ = gui.prepare_command(dict(values, review_input='0'), approve=True)
    assert '--pause-for-input' not in cmd and '--approve-input' not in cmd


