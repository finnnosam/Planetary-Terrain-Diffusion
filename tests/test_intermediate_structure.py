import json
from pathlib import Path

import numpy as np
import pytest
import torch

from planet_training.adaptation import RECIPE, PROFILE, STRUCTURE_RECIPE, STRUCTURE_PROFILE, paired_degradation
from planet_training.data import VERSION, manifest_id
from planet_training.prepare_structure import prepare
from planet_training.run_lock import training_lock, RunAlreadyActive
from planet_training.train import BridgeData, main as train
from toy_meso.common import Normalizer


def data_folder(tmp_path):
    folder=tmp_path/'adapted';folder.mkdir()
    rng=np.random.default_rng(8)
    for split,count in [('train',4),('val',2)]:
        target=rng.normal(5,12,(count,28,28)).astype(np.float16)
        context=rng.normal(5,15,(count,16,16)).astype(np.float32)
        np.save(folder/f'{split}_target.npy',target)
        np.save(folder/f'{split}_context.npy',context)
        np.save(folder/f'{split}_cond.npy',context[:,5:-5,5:-5])
    meta=dict(version=VERSION,kind='bridge',size=28,factor=7,spacing_m=23040.,
              train_count=4,val_count=2,guide_recipe=RECIPE,parent_dataset_id='clean-parent')
    (folder/'manifest.json').write_text(json.dumps(meta))
    (folder/'progress.json').write_text(json.dumps({'done':{'train':4,'val':2}}))
    return folder


def test_structure_keeps_signed_target_and_clean_guides_exact():
    torch.set_num_threads(2)
    target=torch.linspace(-50,60,28*28).reshape(1,1,28,28)
    context=torch.linspace(-70,90,16*16).reshape(1,1,16,16)
    cond=context[...,5:-5,5:-5].clone()
    originals=[t.clone() for t in (target,cond,context)]
    norm=Normalizer({'cond_mean':0.,'cond_std':20.,'res_std':5.},7)
    actual,smooth=paired_degradation(norm,target,cond,context,
                                    sigma_cells=1.,clean_fraction=0.,recipe=STRUCTURE_RECIPE)
    assert actual is target
    assert not torch.equal(smooth,cond)
    assert torch.equal(actual>0,target>0)
    legacy,legacy_guide=paired_degradation(norm,target,cond,context,
                                          sigma_cells=1.,clean_fraction=0.,recipe=RECIPE)
    torch.testing.assert_close(smooth,legacy_guide)
    assert not torch.equal(legacy,target)
    actual,clean=paired_degradation(norm,target,cond,context,
                                   sigma_cells=1.,clean_fraction=1.,recipe=STRUCTURE_RECIPE)
    torch.testing.assert_close(actual,target,rtol=0,atol=0)
    torch.testing.assert_close(clean,cond,rtol=0,atol=0)
    for value,original in zip((target,cond,context),originals):
        torch.testing.assert_close(value,original,rtol=0,atol=0)


def test_structure_references_are_read_only_and_detect_changed_source(tmp_path):
    source=data_folder(tmp_path);before={p.name:p.read_bytes() for p in source.iterdir()}
    out=tmp_path/'structure';meta=prepare(source,out);prepare(source,out)
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before
    assert meta['parent_dataset_id']==manifest_id(json.loads(before['manifest.json']))
    assert not list(out.glob('*.npy'))
    data=BridgeData(out,0)
    assert not data.arrays['train_target'].flags.writeable
    np.testing.assert_array_equal(data.arrays['train_target'],np.load(source/'train_target.npy'))
    for array in data.arrays.values():array._mmap.close()
    del data
    np.save(source/'val_context.npy',np.zeros((2,16,16),np.float32))
    with pytest.raises(ValueError,match='array changed'):BridgeData(out,0)


def test_structure_warm_start_resume_and_objective_guard(tmp_path):
    torch.set_num_threads(2)
    source=data_folder(tmp_path)
    legacy=tmp_path/'legacy-run'
    common=['bridge','--device','cpu','--batch','2','--threads','2','--warmup','1',
            '--preview-steps','2','--eval-every','1','--ch','8','--mults','1,2','--blocks','1']
    train(common+['--data',str(source),'--run',str(legacy),'--guide-profile',PROFILE,'--steps','1'])
    frozen=(legacy/'best.pt').read_bytes()
    out=tmp_path/'structure';prepare(source,out)
    run=tmp_path/'structure-run'
    args=common+['--data',str(out),'--run',str(run),'--guide-profile',STRUCTURE_PROFILE,
                 '--init-from',str(legacy/'best.pt')]
    train(args+['--steps','1']);train(args+['--steps','2'])
    ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    assert ck['step']==2 and ck['guide_profile']==STRUCTURE_PROFILE
    assert ck['dataset_id']==manifest_id(json.loads((out/'manifest.json').read_text()))
    assert ck['initialized_from']['step']==1
    assert (legacy/'best.pt').read_bytes()==frozen
    with pytest.raises(ValueError,match='original learning rate'):
        train(args+['--steps','3','--lr','0.00002'])
    wrong=args.copy();wrong[wrong.index(STRUCTURE_PROFILE)]=PROFILE
    with pytest.raises(ValueError,match='Guide profile'):train(wrong+['--steps','3'])
    with training_lock(run):
        with pytest.raises(RunAlreadyActive):train(args+['--steps','3'])


def test_review_measures_physical_coarse_drift_and_separate_belts():
    from scripts.review_intermediate_structure import measurements,widths
    from toy_meso.sphere import encode
    result=widths(np.array([0,1200,1400,100,1300,0]),1000,23.04)
    assert result=={'segments':2,'longest_km':46.08}
    z=np.full((28,28),400.,np.float32)
    coarse=encode(np.full((6,6),400.,np.float32))
    m=measurements(z,z,23040.,coarse,7)
    assert m['coarse_metre_average_drift_rms_m']==0
    assert m['output_minus_guide_rms_m']==0
    assert m['broad_smoothing_sigma_m']==46080
