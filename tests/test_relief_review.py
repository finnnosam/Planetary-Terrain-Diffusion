import hashlib
import json
from pathlib import Path
import pytest

from planet_training.data import manifest_id
from planet_training.structured_data import OBJECTIVE,STRUCTURED_VERSION
from scripts import train_and_review_relief as pipeline


def setup_review(tmp_path,monkeypatch):
    coast=tmp_path/'coast.pt';coast.write_bytes(b'frozen selected coast')
    digest=hashlib.sha256(coast.read_bytes()).hexdigest()
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(coast=dict(checkpoint=str(coast),checkpoint_sha256=digest))))
    data=tmp_path/'data';data.mkdir();meta=dict(source='fixture')
    (data/'manifest.json').write_text(json.dumps(meta))
    checkpoint=dict(objective=OBJECTIVE,dataset_id=manifest_id(dict(source=meta,representation=STRUCTURED_VERSION)))
    monkeypatch.setattr(pipeline,'load',lambda *args:(None,checkpoint,digest))
    run=tmp_path/'run'
    args=['--selection',str(selection),'--run',str(run),'--data',str(data),'--steps','1']
    return coast,run,args,checkpoint,digest


def test_relief_pipeline_freezes_review_weights_and_requires_visual_review(tmp_path,monkeypatch):
    coast,run,args,checkpoint,digest=setup_review(tmp_path,monkeypatch)
    def train(argv):
        assert argv[0]=='relief'
        run.mkdir();(run/'best.pt').write_bytes(b'finished relief weights')
        return 0
    def evaluate(argv):
        snapshot=Path(argv[argv.index('--relief')+1]);out=Path(argv[argv.index('--out')+1])
        assert snapshot.read_bytes()==b'finished relief weights'
        assert argv[argv.index('--coast')+1]==str(coast)
        (run/'best.pt').write_bytes(b'later training weights')
        assert snapshot.read_bytes()==b'finished relief weights'
        report=dict(coast_sha256=digest,relief_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    coast_step=10000,relief_step=10000)
        (out/'evaluation.json').write_text(json.dumps(report))
    monkeypatch.setattr(pipeline,'train',train);monkeypatch.setattr(pipeline,'evaluate',evaluate)
    assert pipeline.main(args)==0
    status=json.loads((run/'relief-quality-status.json').read_text())
    assert status['visual_review_required'] and status['coast_sha256']==digest


def test_interrupted_relief_training_cannot_trigger_generation(tmp_path,monkeypatch):
    _,run,args,_,_=setup_review(tmp_path,monkeypatch)
    monkeypatch.setattr(pipeline,'train',lambda argv:130)
    def unexpected(argv): raise AssertionError('Review must not start after interruption')
    monkeypatch.setattr(pipeline,'evaluate',unexpected)
    assert pipeline.main(args)==130
    assert not (run/'relief-quality-status.json').exists()


@pytest.mark.parametrize('change',['checkpoint','dataset','objective'])
def test_invalid_selected_coast_blocks_expensive_training(tmp_path,monkeypatch,change):
    coast,_,args,checkpoint,_=setup_review(tmp_path,monkeypatch)
    if change=='checkpoint': coast.write_bytes(b'changed')
    elif change=='dataset': checkpoint['dataset_id']='different'
    else: checkpoint['objective']='old'
    def unexpected(argv): raise AssertionError('Training must not start with invalid selection')
    monkeypatch.setattr(pipeline,'train',unexpected)
    with pytest.raises(ValueError): pipeline.main(args)
