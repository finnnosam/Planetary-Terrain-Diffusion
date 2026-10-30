import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from planet_training.adaptation import RECIPE
from planet_training.data import VERSION as DATA_VERSION, HOLDOUT_CAPS
from planet_training.prepare_terrain_structure import prepare
from planet_training.terrain_structure import (StructureData, StructureNorm, area_weights,
    overlap_matrix, conservative_resample, encode, decode, project, handoff,
    target_for, build, loss, constraint_metrics, OBJECTIVE, CONSERVE_FACTOR, HANDOFF_FACTOR)
from planet_training.train_terrain_structure import main as train
from planet_training.run_lock import training_lock, RunAlreadyActive


def source_fixture(folder):
    folder.mkdir()
    q = torch.linspace(-1, 1, 112)
    z = 1200+1000*torch.sin(6*q[None, :])*torch.cos(3*q[:, None])
    z[:, :24] = -1000; z[:, 24:29] = 80
    meta = {'kind': 'bridge', 'version': DATA_VERSION, 'size': 112, 'factor': 7,
        'spacing_m': 23040., 'train_count': 12, 'val_count': 3, 'seed': 42,
        'guide_recipe': RECIPE, 'holdout_caps_lat_lon_radius_m': HOLDOUT_CAPS}
    for split, count in [('train', 12), ('val', 3)]:
        t = torch.stack([z*(.85+.025*i) for i in range(count)])[:, None]
        c = encode(F.pad(F.avg_pool2d(t, 7), (1, 1, 1, 1), mode='replicate'))
        ctx = F.pad(c, (5, 5, 5, 5), mode='replicate')
        for name, a in [('target', encode(t).numpy()[:, 0].astype(np.float16)), ('cond', c.numpy()[:, 0]), ('context', ctx.numpy()[:, 0])]:
            np.save(folder/f'{split}_{name}.npy', a)
    (folder/'manifest.json').write_text(json.dumps(meta))
    (folder/'progress.json').write_text(json.dumps({'done': {'train': 12, 'val': 3}}))


def test_noninteger_conversion_conserves_physical_area_and_never_encoded_values():
    a = overlap_matrix(112, 64)
    torch.testing.assert_close(a.sum(0), torch.ones(112))
    torch.testing.assert_close(a.sum(1), torch.full((64,), 1.75))
    z = torch.arange(112*112).float().reshape(1, 112, 112)-4000
    converted = conservative_resample(z, 64)
    w = area_weights(112, 23040.)[0, 0]; weights40 = a@w@a.T
    torch.testing.assert_close((converted*weights40).sum(), (z*w).sum(), rtol=1e-6, atol=.5)
    torch.testing.assert_close(conservative_resample(torch.full((2, 112, 112), -1234.), 64), torch.full((2, 64, 64), -1234.), rtol=1e-6, atol=.001)
    wrong = decode(conservative_resample(encode(z), 64))
    assert (wrong-converted).abs().max() > 10


def test_projection_preserves_shelves_sign_region_means_and_gradients():
    torch.manual_seed(1)
    base = torch.full((2, 1, 64, 64), 1100.)
    base[..., :12] = -100.; base[..., 12:16] = 100.
    raw = (base+torch.randn_like(base)*3000).requires_grad_()
    z = project(raw, base)
    result = constraint_metrics(z, base, CONSERVE_FACTOR, 40320.)
    assert result['protected_height_change_max_m'] == 0
    assert result['sign_disagreement_fraction'] == 0
    assert result['added_footprint_mean_drift_max_m'] < .002
    assert z[..., 16:].min() >= .99
    torch.testing.assert_close(project(z, base), z, atol=.003, rtol=1e-5)
    z.square().mean().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def test_handoff_carries_belt_valley_structure_instead_of_only_coarse_averages():
    q = torch.arange(64).float()
    ridge = 1500*torch.exp(-((q-34)/3)**2)-700*torch.exp(-((q-25)/4)**2)
    base = torch.full((1, 1, 64, 64), 1000.)
    structure = project(base+ridge[None, None, None, :], base)
    coarse = encode(torch.full((1, 1, 18, 18), 1000.))
    frozen = encode(torch.full((1, 1, 112, 112), 1000.))
    candidate, control, base23 = handoff(structure, base, frozen, coarse)
    assert candidate.shape == (1, 1, 112, 112)
    torch.testing.assert_close(control, base23)
    assert candidate[0, 0, 56].max() > 2000 and candidate[0, 0, 56].min() < 600
    interpolated = F.interpolate(structure, size=(112, 112), mode='bilinear', align_corners=False)
    assert (candidate-interpolated).square().mean().sqrt() < 20
    result = constraint_metrics(candidate, base23, HANDOFF_FACTOR, 23040.)
    assert result['added_footprint_mean_drift_max_m'] < .002 and result['sign_disagreement_fraction'] == 0


def test_preparation_source_is_read_only_and_identity_detects_changes(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path/'source'; source_fixture(source)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    out = tmp_path/'new'; prepare(source, out, chunk=3); prepare(source, out, chunk=4)
    data = StructureData(out)
    assert data.batch(2)[0].shape == (2, 1, 64, 64)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == before
    progress = json.loads((out/'progress.json').read_text())
    # A committed prefix can be safely regenerated from the immutable source.
    original = np.load(out/'train_metres.npy').copy()
    data.arrays['train']._mmap.close(); data.arrays['val']._mmap.close(); del data
    progress['done']['train'] = 6; (out/'progress.json').write_text(json.dumps(progress))
    prepare(source, out, chunk=2)
    np.testing.assert_array_equal(original, np.load(out/'train_metres.npy'))
    import os
    changed = source/'train_cond.npy'; stamp = changed.stat()
    os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+10000000))
    with pytest.raises(ValueError, match='source array changed'): StructureData(out)


def test_shape_objective_training_resume_and_lock(tmp_path):
    source = tmp_path/'source'; source_fixture(source)
    data = tmp_path/'data'; prepare(source, data)
    dataset = StructureData(data); raw, c, _ = dataset.batch(2)
    norm = StructureNorm(dataset.stats()); model = build({'ch': 8, 'mults': [1, 2, 4], 'blocks': 1, 'dropout': 0.})
    value = loss(model, norm, raw, c); value.backward()
    assert torch.isfinite(value) and torch.isfinite(model.net.out[-1].weight.grad).all()
    run = tmp_path/'run'
    args = ['--data', str(data), '--run', str(run), '--device', 'cpu', '--ch', '8', '--blocks', '1',
        '--batch', '2', '--threads', '2', '--eval-every', '2', '--preview-steps', '2', '--log-every', '1']
    assert train(args+['--steps', '2']) == 0
    assert train(args+['--steps', '3']) == 0
    ck = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
    assert ck['step'] == 3 and ck['objective'] == OBJECTIVE
    with pytest.raises(ValueError, match='Resume identity'): train(args+['--steps', '4', '--lr', '.001'])
    with training_lock(run):
        with pytest.raises(RunAlreadyActive): train(args+['--steps', '4'])
