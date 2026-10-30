import json
from pathlib import Path

import numpy as np
import pytest
import torch

from planet_training.adaptation import (HALO, RECIPE, PROFILE, coarse_context,
    degrade_guides, paired_degradation, frame_matches_target, recovered_frames)
from planet_training.data import bridge_patch, VERSION, manifest_id, geographic_split, SPACING_M, FACTOR
from planet_training.train import main as train_main
from planet_training.prepare import prepare_bridge
from planet_training.prepare_adaptation import prepare
from toy_meso.common import Normalizer
from toy_meso.sphere import encode, random_frames, patch_points


def test_meso_crop_alignment_legacy_warm_start_and_resume(tmp_path):
    from planet_training.adaptation import MESO_RECIPE, MESO_PROFILE
    from planet_training.prepare_meso_adaptation import prepare as prepare_meso
    from planet_training.train import BridgeData
    from toy_meso.common import build_model, compute_stats
    torch.set_num_threads(2)
    rng = np.random.default_rng(2)
    target = rng.normal(15, 5, (8, 64, 64)).astype(np.float16)
    cond = rng.normal(15, 5, (8, 10, 10)).astype(np.float32)
    parent = dict(size=64, factor=8, spacing_m=2880.)
    source = tmp_path/'original.npz'
    np.savez(source, train_target=target, train_cond=cond, val_target=target[:4],
             val_cond=cond[:4], meta=json.dumps(parent))
    out = tmp_path/'adapted-meso'
    prepare_meso(source, out, train=8, val=4)
    prepare_meso(source, out, train=8, val=4)
    data = BridgeData(out, 0)
    np.testing.assert_array_equal(data.arrays['train_target'], target[:,24:-24,24:-24])
    np.testing.assert_array_equal(data.arrays['train_context'], cond)
    np.testing.assert_array_equal(data.arrays['train_cond'], cond[:,3:-3,3:-3])
    stats = compute_stats(target, cond, 8)
    norm = Normalizer(stats, 8)
    full_up, _ = norm.condition(torch.from_numpy(cond)[:, None])
    crop_up, _ = norm.condition(torch.from_numpy(np.array(data.arrays['train_cond']))[:, None])
    torch.testing.assert_close(crop_up, full_up[...,24:-24,24:-24])
    t, c, context = data.batch(2, include_context=True)
    modified, smooth = paired_degradation(norm, t, c, context, recipe=MESO_RECIPE)
    old_up, _ = norm.condition(c); new_up, _ = norm.condition(smooth)
    torch.testing.assert_close(modified-new_up, t-old_up)
    config = dict(ch=8, mults=[1,2], blocks=1, dropout=0.)
    model = build_model(config)
    checkpoint = tmp_path/'original.pt'
    torch.save(dict(model=model.state_dict(), ema=model.state_dict(), config=config,
                    stats=stats, meta=parent, step=222000), checkpoint)
    original_bytes = checkpoint.read_bytes()
    run = tmp_path/'meso-run'
    args = ['bridge','--data',str(out),'--run',str(run),'--device','cpu','--batch','2',
            '--init-from',str(checkpoint),'--guide-profile',MESO_PROFILE,'--threads','2',
            '--preview-steps','2','--eval-every','1']
    train_main(args+['--steps','1']); train_main(args+['--steps','2'])
    ck = torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    assert ck['meta']['size']==64 and ck['meta']['training_crop_size']==16
    assert ck['stats']==stats and ck['step']==2 and ck['initialized_from']['step']==222000
    assert checkpoint.read_bytes()==original_bytes


def test_degradation_uses_real_halo_and_preserves_detail_residual():
    torch.set_num_threads(2)
    cond = torch.full((16, 1, 10, 10), 20.)
    context = torch.full((16, 1, 20, 20), 20.)
    norm = Normalizer({"cond_mean": 0., "cond_std": 20., "res_std": 5.}, 7)
    target = torch.randn(16, 1, 56, 56)+20
    a, b = paired_degradation(norm, target, cond, context, torch.Generator().manual_seed(1))
    torch.testing.assert_close(b, cond)
    torch.testing.assert_close(a, target)
    # Only surrounding terrain changes, so this also tests that genuine context
    # contributes to the filter rather than reflecting a flat central guide.
    context[:, :, :HALO] = 60
    a, b = paired_degradation(norm, target, cond, context, torch.Generator().manual_seed(1))
    assert (b[:, :, :2] > cond[:, :, :2]).any()
    assert (b==cond).flatten(1).all(1).any()  # clean mixture is exact
    old_up, _ = norm.condition(cond); new_up, _ = norm.condition(b)
    torch.testing.assert_close(a-new_up, target-old_up)
    b2 = degrade_guides(context, cond, torch.Generator().manual_seed(1))
    torch.testing.assert_close(b, b2)
    with pytest.raises(ValueError, match="five extra"):
        degrade_guides(context[:, :, 1:], cond, None)


def test_orientation_probe_rejects_wrong_rotation_and_context_matches_constants():
    class Raster:
        def sample(self, lat, lon):
            return (1200+np.asarray(lat)*20+np.sin(np.radians(lon))*900).astype(np.float32)
    frame = tuple(a[0] for a in random_frames(np.random.default_rng(12), 1))
    target, _ = bridge_patch(Raster(), frame, size=56, sub=3)
    assert frame_matches_target(Raster(), frame, target, 56, sub=3)
    wrong = (frame[0], frame[2], -frame[1])
    assert not frame_matches_target(Raster(), wrong, target, 56, sub=3)
    class Constant:
        def sample(self, lat, lon): return np.full(np.shape(lat), 400., dtype=np.float32)
    np.testing.assert_allclose(coarse_context(Constant(), frame, size=56), 20., atol=1e-5)


def test_preparation_and_warm_start_preserve_original_and_resume(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    dem = tmp_path/'earth.tif'
    values = (800+np.random.default_rng(5).uniform(-400,400,(180,360))).astype(np.float32)
    with rasterio.open(dem, 'w', driver='GTiff', height=180, width=360, count=1,
                       dtype='float32', transform=from_origin(-180,90,1,1), crs='EPSG:4326') as ds:
        ds.write(values,1)
    clean = tmp_path/'clean'
    prepare_bridge(dem, clean, tmp_path/'cache', train=40, val=16, size=56, sub=2)
    frames = recovered_frames(clean)
    assert all(f is not None for s in frames.values() for f in s)
    before = (clean/'manifest.json').read_bytes()
    adapted = tmp_path/'adapted'
    prepare(clean, adapted, tmp_path/'cache', workers=2, sub=4)
    assert (clean/'manifest.json').read_bytes()==before
    meta = json.loads((adapted/'manifest.json').read_text())
    for split in ('train','val'):
        original = np.load(clean/f'{split}_cond.npy')[meta['parent_source_indices'][split]]
        context = np.load(adapted/f'{split}_context.npy')
        np.testing.assert_array_equal(original, context[:, HALO:-HALO, HALO:-HALO])
        for index in meta['parent_source_indices'][split]:
            footprint=patch_points(frames[split][index], 56+2*(HALO+1)*FACTOR, SPACING_M)
            assert geographic_split(footprint)==split
    run0 = tmp_path/'initial'
    args = ['bridge','--data',str(clean),'--run',str(run0),'--device','cpu','--batch','2',
            '--ch','8','--mults','1,2','--blocks','1','--threads','2','--preview-steps','2','--eval-every','1']
    train_main(args+['--steps','1'])
    run = tmp_path/'run'
    args = ['bridge','--data',str(adapted),'--run',str(run),'--device','cpu','--batch','2',
            '--init-from',str(run0/'best.pt'),'--guide-profile',PROFILE,'--threads','2',
            '--preview-steps','2','--eval-every','1']
    train_main(args+['--steps','1']); train_main(args+['--steps','2'])
    ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    assert ck['step']==2 and ck['initialized_from']['step']==1
    assert ck['config']['ch']==8 and ck['guide_profile']==PROFILE
    assert ck['dataset_id']==manifest_id(meta)
    with pytest.raises(ValueError, match='requires'):
        train_main(['bridge','--data',str(adapted),'--run',str(run),'--device','cpu'])
    stopped = tmp_path/'stopped'
    stop_args = args.copy()
    stop_args[stop_args.index('--run')+1] = str(stopped)
    with monkeypatch.context() as patch:
        patch.setattr('planet_training.train.validation_loss', lambda *args, **kwargs: 1.)
        train_main(stop_args+['--steps','5','--validation-patience','1'])
    latest=torch.load(stopped/'latest.pt',map_location='cpu',weights_only=False)
    best=torch.load(stopped/'best.pt',map_location='cpu',weights_only=False)
    assert latest['step']==1 and latest['stale_evaluations']==1 and best['step']==0
