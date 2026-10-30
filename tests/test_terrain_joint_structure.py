import inspect
import pytest
import torch

from planet_training.terrain_structure import StructureNorm, encode, project
from planet_training.terrain_plan_oracle import build as source_build
from planet_training import terrain_joint_structure as joint


CONFIG={'ch':8,'mults':[1,2,4],'blocks':1,'dropout':0.}


def fixture():
    torch.set_num_threads(2);torch.manual_seed(8)
    source=source_build(CONFIG).eval()
    with torch.no_grad():
        source.net.inp.weight[:,3:].zero_()
        source.net.out[-1].weight.normal_(std=.02)
    model=joint.build(CONFIG).eval();joint.initialize_from_control(model,source.state_dict())
    norm=StructureNorm({'cond_mean':0.,'cond_std':40.,'res_std':2.})
    coarse=encode(torch.full((1,1,18,18),1200.));coarse[...,:3]=-10
    base=norm.condition(coarse)[0]
    q=torch.arange(64).float()
    delta=1000*torch.exp(-((q-36)/3)**2)-700*torch.exp(-((q-24)/4)**2)
    raw=project(base+delta[None,None,None],base)
    return source,model,norm,coarse,raw


def test_source_height_response_preserved_without_receiving_plan():
    source,model,norm,coarse,raw=fixture()
    _,_,cin=norm.condition(coarse)
    x=torch.randn(1,5,64,64);s=torch.tensor([.5])
    old=source(x[:,:1],s,torch.cat((cin,torch.zeros(1,4,64,64)),1))
    torch.testing.assert_close(model(x,s,cin)[:,:1],old,atol=1e-6,rtol=1e-5)
    assert model.net.inp.weight[:,1:5].abs().max()==0
    assert model.net.out[-1].weight[1:].abs().max()==0
    assert model.net.out[-1].bias[1:].abs().max()==0


def test_targets_change_outputs_only_not_conditions_and_height_loss_scale():
    _,model,norm,coarse,raw=fixture()
    control=joint.targets(norm,raw,coarse,'control')
    new=joint.targets(norm,raw,coarse,'joint')
    torch.testing.assert_close(control[0][:,:1],new[0][:,:1],atol=0,rtol=0)
    torch.testing.assert_close(control[1],new[1],atol=0,rtol=0)
    assert new[1].shape[1]==2 and control[0][:,1:].abs().max()==0
    assert new[0][:,1:].abs().max()>.1
    changed=joint.targets(norm,raw+500,coarse,'joint')
    torch.testing.assert_close(new[1],changed[1],atol=0,rtol=0)
    parts=[]
    for mode in ['control','joint']:
        parts.append(joint.loss_parts(model,norm,raw,coarse,mode,torch.Generator().manual_seed(5)))
    torch.testing.assert_close(parts[0]['height'],parts[1]['height'],atol=0,rtol=0)
    torch.testing.assert_close(parts[0]['shape'],parts[1]['shape'],atol=0,rtol=0)
    parts[1]['total'].backward()
    assert model.net.out[-1].weight.grad[1:].abs().sum()>0
    assert torch.isfinite(model.net.out[-1].weight.grad).all()


def test_generation_needs_only_coarse_input_and_seed_is_reproducible(monkeypatch):
    _,model,norm,coarse,_=fixture()
    assert 'plan' not in inspect.signature(joint.sample).parameters
    def forbidden(*args,**kwargs):raise AssertionError('Sampling tried to inspect a target plan')
    monkeypatch.setattr(joint,'plan_for',forbidden)
    first=joint.sample(model,norm,coarse,steps=4,seed=19)
    again=joint.sample(model,norm,coarse,steps=4,seed=19)
    other=joint.sample(model,norm,coarse,steps=4,seed=23)
    for a,b in zip(first,again):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.equal(first[2],other[2])
    assert all(torch.isfinite(v).all() for v in first)
    protected=first[1]<=200
    torch.testing.assert_close(first[0][protected],first[1][protected],atol=0,rtol=0)
    assert first[3].masked_select(protected.expand_as(first[3])).abs().max()==0


def test_no_active_cells_have_finite_loss_and_locked_output():
    _,model,norm,_,_=fixture()
    coarse=encode(torch.full((1,1,18,18),100.));raw=torch.full((1,1,64,64),2000.)
    parts=joint.loss_parts(model,norm,raw,coarse,'joint',torch.Generator().manual_seed(5))
    assert parts['plan']==0 and all(torch.isfinite(v) for v in parts.values())
    z,base,_,p=joint.sample(model,norm,coarse,steps=4)
    torch.testing.assert_close(z,base,atol=0,rtol=0);assert p.abs().max()==0


def test_generated_plan_interpretation_does_not_rewrite_heights():
    base=torch.full((1,1,8,8),1200.);base[...,:2]=100.
    p=torch.zeros(1,4,8,8);p[:,0]=.8;p[:,1]=.9;p[:,2]=3.;p[:,3]=4.
    result=joint.interpret_plan(p,base)
    assert result[...,:2].abs().max()==0
    torch.testing.assert_close(result[:,2:,2:,2:],torch.tensor([.6,.8])[None,:,None,None].expand(1,2,6,6))
    assert p[:,2].min()==3. and base.min()==100.
    with pytest.raises(ValueError):joint.targets(StructureNorm({'cond_mean':0,'cond_std':1,'res_std':1}),torch.zeros(1,1,64,64),torch.zeros(1,1,18,18),'oracle')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA mixed precision check')
def test_joint_training_cuda_geometry_and_gradients_stay_finite():
    _,model,norm,coarse,raw=fixture();model=model.cuda();coarse,raw=coarse.cuda(),raw.cuda()
    expected=joint.targets(norm,raw,coarse,'joint')
    with torch.autocast('cuda',dtype=torch.float16):
        actual=joint.targets(norm,raw,coarse,'joint')
        parts=joint.loss_parts(model,norm,raw,coarse,'joint')
    torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
    parts['total'].backward()
    assert all(torch.isfinite(v) for v in parts.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
