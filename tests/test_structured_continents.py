import json
import numpy as np
import pytest
import torch

from planet_diffusion import cube
from planet_training.spherical_net import Chart,CubeConv,CoastReliefDenoiser
from planet_training.structured_data import split_encoded,decode_relief


def test_halos_sample_neighbour_charts_and_preserve_constants():
    n=12
    chart=Chart(n)
    values=np.stack([cube.directions(f,np.arange(n+1)[:,None],np.arange(n+1)[None],n)[...,0] for f in range(6)]).astype(np.float32)
    x=torch.from_numpy(values)[None,None]
    out=chart.halo(x)[0,0].numpy()
    q=np.arange(-1,n+2)
    expected=np.stack([cube.sample(values[None],cube.directions(f,q[:,None],q[None,:],n))[0] for f in range(6)])
    np.testing.assert_allclose(out,expected,atol=2e-6)
    np.testing.assert_allclose(chart.halo(torch.ones_like(x)),1,atol=1e-6)


def test_convolution_sends_gradients_across_geographic_face_edges():
    torch.set_num_threads(2)
    conv=CubeConv(1,1,12)
    with torch.no_grad(): conv.conv.weight.fill_(1);conv.conv.bias.zero_()
    x=torch.zeros(1,1,6,13,13,requires_grad=True)
    conv(x)[0,0,0,6,-1].backward()
    assert x.grad[0,0,0].abs().sum()>0
    assert x.grad[0,0,1:].abs().sum()>0


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_chart_projection_and_halos_accept_mixed_precision_features(dtype):
    chart=Chart(8)
    x=torch.randn(2,4,6,9,9).to(dtype).requires_grad_(True)
    y=chart.halo(x)
    assert y.dtype==torch.float32 and torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_geometry_is_independent_of_height_magnitudes_and_fixed_zero_datum():
    encoded=np.array([[[-50.,-2.,3.,40.]]]*6,dtype=np.float32)
    coast,magnitude=split_encoded(encoded)
    np.testing.assert_array_equal(decode_relief(coast,magnitude),np.sign(encoded)*encoded**2)
    taller=decode_relief(coast,magnitude*1.5)
    np.testing.assert_array_equal(taller>0,coast>.5)
    np.testing.assert_array_equal(taller<0,coast<.5)
    with pytest.raises(ValueError): decode_relief(coast,np.full_like(magnitude,np.nan))


@pytest.mark.parametrize('stage',['coast','relief'])
def test_models_use_whole_sphere_noise_seams_and_finite_training_gradients(stage):
    from planet_training.train_structured import loss_for,sample
    torch.set_num_threads(2)
    model=CoastReliefDenoiser(stage,n=8,ch=8,dropout=0)
    noise=model.grid.noise(2,1,'cpu')
    np.testing.assert_allclose(noise,model.grid.project(noise),atol=1e-6)
    coast=(noise>0).float();magnitude=torch.randn_like(noise)
    stats=dict(mean=30.,std=12.)
    loss=loss_for(model,coast,magnitude,stats)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    result=sample(model,stats,[.3,.4],steps=2,coast=coast)
    assert result.shape==noise.shape and torch.isfinite(result).all()
    np.testing.assert_allclose(result,model.grid.project(result),atol=1e-5)
    if stage=='coast': assert result.min()>=0 and result.max()<=1


def tiny_data(tmp_path):
    from planet_training.data import VERSION
    folder=tmp_path/'data';folder.mkdir()
    lat=np.radians(np.arange(90,-91,-1));lon=np.radians(np.arange(-180,181))
    values=(1500*np.cos(lat[:,None])*np.cos(lon[None])+200).astype(np.float32)
    np.save(folder/'maps.npy',np.stack([values,values*.8,values*1.2]))
    meta=dict(kind='layout',version=VERSION,face_intervals=8,supersample=2,spacing_m=161280.,radius_m=6371000.,
              records=[dict(split=s,age_ma=i*100) for i,s in enumerate(['train','val','test'])])
    (folder/'manifest.json').write_text(json.dumps(meta))
    return folder


def test_training_resume_versions_data_and_preview_only_export(tmp_path):
    from planet_training.train_structured import main as train
    from planet_training.generate_structured import main as generate
    from planet_training.structured_data import STRUCTURED_VERSION
    torch.set_num_threads(2)
    data=tiny_data(tmp_path)
    runs={s:tmp_path/s for s in ('coast','relief')}
    common=['--data',str(data),'--device','cpu','--ch','8','--batch','2','--warmup','1','--threads','2',
            '--preview-steps','2','--eval-every','2','--save-every','1','--log-every','1']
    for stage in runs:
        assert train([stage,'--run',str(runs[stage]),'--steps','1',*common])==0
        before=torch.load(runs[stage]/'latest.pt',weights_only=False)
        assert train([stage,'--run',str(runs[stage]),'--steps','2',*common])==0
        after=torch.load(runs[stage]/'latest.pt',weights_only=False)
        assert before['step']==1 and after['step']==2 and after['version']==STRUCTURED_VERSION
        assert (runs[stage]/'preview-000002.png').exists()
    # One training map means exactly one supported coverage; derive it from
    # the saved normalization so the support guard itself is exercised.
    ck=torch.load(runs['coast']/'best.pt',weights_only=False)
    percent=ck['stats']['training_coverage_min']*100
    out=tmp_path/'sample'
    args=['--coast',str(runs['coast']/'best.pt'),'--relief',str(runs['relief']/'best.pt'),
          '--out',str(out),'--land-percent',str(percent),'--steps','2','--height','32','--device','cpu']
    generate(args)
    meta=json.loads((out/'layout.json').read_text())
    assert meta['sea_level_m']==0 and meta['requested_land_percent']==pytest.approx(percent)
    assert meta['geography']['seam_max_error_m']<1e-3
    coast=np.load(out/'cube-coast.npy');z=np.load(out/'cube-elevation.npy')
    np.testing.assert_array_equal(z>0,coast>.5)
    assert not (out/'tiles').exists()
    with pytest.raises(ValueError,match='fresh'): generate(args)
    with pytest.raises(ValueError,match='Resume'): train(['coast','--run',str(runs['coast']),'--steps','3','--lr','.002',*common])
    with pytest.raises(ValueError,match='architecture'): train(['relief','--run',str(runs['coast']),'--steps','3',*common])
    from planet_training.evaluate_structured import main as evaluate
    evaluation=tmp_path/'evaluation'
    evaluate(['--coast',str(runs['coast']/'best.pt'),'--relief',str(runs['relief']/'best.pt'),
              '--data',str(data),'--out',str(evaluation),'--land-percent',str(percent),'--steps','2','--seeds','42','--device','cpu'])
    checked=json.loads((evaluation/'evaluation.json').read_text())
    assert checked['sea_level_m']==0 and 'test' not in checked['references']
    assert len(checked['samples'])==1 and (evaluation/'comparison.png').exists()


@pytest.mark.parametrize('stage',['coast','relief'])
def test_only_known_relief_condition_is_balanced_and_absent_domains_are_finite(stage):
    from planet_training.train_structured import spatial_weights
    model=CoastReliefDenoiser(stage,n=8,ch=8,dropout=0)
    coast=torch.ones(2,1,6,9,9)
    coast[0,:,:,0:2]=0
    magnitude=torch.full_like(coast,2.)
    weights=spatial_weights(model,coast,magnitude,dict(mean=30.,std=12.))
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(model.grid.mean(weights),torch.ones(2,1),atol=1e-5,rtol=1e-5)
    if stage=='coast':
        torch.testing.assert_close(weights,torch.ones_like(coast))
        torch.testing.assert_close(model.grid.mean(weights*(1-coast)),model.grid.mean(1-coast))
    else:
        torch.testing.assert_close(model.grid.mean(weights*(1-coast))[0],torch.tensor([.5]),atol=1e-5,rtol=1e-5)
        # Unknown target heights cannot change conditional posterior odds.
        torch.testing.assert_close(weights,spatial_weights(model,coast,magnitude*100,dict(mean=30.,std=12.)))


def test_ambiguous_coastline_loss_preserves_true_land_odds():
    from planet_training.train_structured import spatial_weights
    model=CoastReliefDenoiser('coast',n=8,ch=8,dropout=0)
    coast=(model.grid.noise(1,1,'cpu',torch.Generator().manual_seed(3))>1).float()
    weights=spatial_weights(model,coast,torch.zeros_like(coast),{})
    clean=coast-.5
    # At zero information the optimum constant prediction must be p(land)-.5,
    # rather than zero (the old class-balanced objective's 50/50 answer).
    prediction=torch.zeros((),requires_grad=True)
    model.grid.mean(weights*(prediction-clean).square()).mean().backward()
    optimum=model.grid.mean(weights*clean)/model.grid.mean(weights)
    torch.testing.assert_close(optimum,model.grid.mean(coast)-.5)
    assert optimum.item()<-.1 and prediction.grad.item()>0


def test_fresh_coast_model_uses_known_high_noise_prior_and_legacy_default_is_exact():
    torch.set_num_threads(2)
    corrected=CoastReliefDenoiser('coast',n=8,ch=8,dropout=0,coverage_prior=True)
    legacy=CoastReliefDenoiser('coast',n=8,ch=8,dropout=0)
    legacy.load_state_dict(corrected.state_dict())
    x=torch.zeros(2,1,6,9,9);coverage=torch.tensor([.2,.35]);sigma=torch.full((2,),10000.)
    with torch.no_grad():
        result=corrected(x,sigma,coverage)
        torch.testing.assert_close(corrected.grid.mean(result)[:,0],coverage-.5,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(legacy(x,sigma,coverage),torch.zeros_like(x))
        torch.testing.assert_close(corrected(x,torch.full_like(sigma,1e-6),coverage),x,atol=1e-5,rtol=1e-5)


def test_warm_start_resets_training_and_rejects_legacy_resume(tmp_path):
    from planet_training.train_structured import main as train
    from planet_training.structured_data import OBJECTIVE
    import hashlib
    data=tiny_data(tmp_path);source=tmp_path/'source';warm=tmp_path/'warm'
    common=['--data',str(data),'--device','cpu','--ch','8','--batch','2','--warmup','1','--threads','2',
            '--preview-steps','2','--eval-every','1','--save-every','1','--log-every','1']
    train(['coast','--run',str(source),'--steps','1',*common])
    legacy=torch.load(source/'latest.pt',weights_only=False);legacy.pop('objective')
    torch.save(legacy,source/'latest.pt')
    digest=hashlib.sha256((source/'latest.pt').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Old biased objective'):
        train(['coast','--run',str(source),'--steps','2',*common])
    train(['coast','--run',str(warm),'--init-from',str(source/'latest.pt'),'--steps','1',*common])
    result=torch.load(warm/'latest.pt',weights_only=False)
    assert result['objective']==OBJECTIVE and result['step']==1
    assert result['initialized_from']['step']==1
    assert hashlib.sha256((source/'latest.pt').read_bytes()).hexdigest()==digest
    assert (warm/'preview-000000.json').exists()
    with pytest.raises(ValueError,match='Warm-start checkpoint'):
        train(['relief','--run',str(tmp_path/'wrong'),'--init-from',str(source/'latest.pt'),'--steps','1',*common])


def test_single_trainer_lock_is_released_and_blocks_a_second_launch(tmp_path):
    from planet_training.run_lock import training_lock,RunAlreadyActive
    from planet_training.train_structured import main as train
    run=tmp_path/'run';data=tiny_data(tmp_path)
    args=['coast','--run',str(run),'--data',str(data),'--device','cpu','--ch','8','--check-only']
    with training_lock(run):
        with pytest.raises(RunAlreadyActive): train(args)
    assert train(args)==0


def test_review_pipeline_stops_after_training_interrupt(tmp_path,monkeypatch):
    from scripts import train_and_review_coast as pipeline
    monkeypatch.setattr(pipeline,'train',lambda argv:130)
    def unexpected_review(argv): raise AssertionError('Interrupted training cannot trigger sampling')
    monkeypatch.setattr(pipeline,'review',unexpected_review)
    assert pipeline.main(['--run',str(tmp_path/'run')])==130


def test_review_pipeline_reports_coverage_without_approving_terrain(tmp_path,monkeypatch):
    from scripts import train_and_review_coast as pipeline
    run=tmp_path/'run';run.mkdir()
    monkeypatch.setattr(pipeline,'train',lambda argv:0)
    def review(argv):
        out=tmp_path/Path(argv[argv.index('--out')+1]);out.mkdir()
        report=dict(step=10000,checkpoint_sha256='test',samples=[dict(actual_land_percent=28,requested_land_percent=30)],
                    coverage_sweep_same_seed=[dict(requested=20,achieved=20),dict(requested=35,achieved=35)])
        (out/'review.json').write_text(json.dumps(report))
    from pathlib import Path
    monkeypatch.setattr(pipeline,'review',review)
    assert pipeline.main(['--run',str(run)])==0
    status=json.loads((run/'coast-quality-status.json').read_text())
    assert status['preliminary_coverage_gate_passed'] and status['visual_review_required']
    assert status['mean_30_percent_error_pp']==2
