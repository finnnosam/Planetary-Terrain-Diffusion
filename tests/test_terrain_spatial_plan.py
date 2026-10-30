import pytest
import torch

from planet_training.terrain_structure import StructureNorm,encode
from planet_training.terrain_plan_oracle import build as old_build
from planet_training.terrain_spatial_plan import (
    PlanCodec,continuous_fields,build,initialize,loss_parts,sample_planner,
)
from scripts.audit_terrain_plan_targets import continuous_candidate


def codec():
    return PlanCodec({m:{'mean':[.1,-.2,.3,.2],'std':[.4,.6,.8,.5]} for m in ['categorical','continuous']})


def fields():
    base=torch.full((1,1,64,64),1000.);base[...,:12]=-1000.;base[...,12:16]=100.
    z=base+400*torch.sin(torch.arange(64).float()/4)[None,None,None,:]*(base>200)
    return z,base


def test_continuous_codec_matches_audit_and_roundtrips_physical_units():
    torch.set_num_threads(2);z,b=fields();p=continuous_fields(z,b)
    torch.testing.assert_close(p,continuous_candidate(z,b))
    c=codec();encoded=c.encode_fields(p,b,'continuous');decoded=c.decode(encoded,b,'continuous')
    torch.testing.assert_close(decoded,p*(b>200),atol=.001,rtol=1e-4)
    assert encoded[...,:16].abs().max()==0
    cat=c.encode(z,b,'categorical')
    torch.testing.assert_close(c.canonical_generated(cat,b,'categorical'),cat,atol=1e-6,rtol=1e-5)


def test_new_plan_channels_preserve_identical_height_source_and_receive_gradients():
    torch.set_num_threads(2);torch.manual_seed(42)
    config={'ch':8,'mults':[1,2,4],'blocks':1,'dropout':0.}
    source=old_build(config);source.net.inp.weight.data[:,3:]=0
    torch.nn.init.normal_(source.net.out[-1].weight,std=.03)
    a=build('renderer',config);b=build('renderer',config)
    initialize(a,source.state_dict(),'renderer');initialize(b,source.state_dict(),'renderer')
    x=torch.randn(1,1,64,64);cin=torch.randn(1,2,64,64);s=torch.tensor([.5])
    zero=torch.zeros(1,4,64,64);p=torch.randn_like(zero)
    expected=source(x,s,torch.cat((cin,zero),1))
    actual=a(x,s,torch.cat((cin,p),1));torch.testing.assert_close(actual,expected)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    actual.square().mean().backward();assert a.net.inp.weight.grad[:,3:].abs().sum()>0


def test_planner_has_no_noisy_height_shortcut_and_unused_height_head_has_zero_gradient():
    torch.set_num_threads(2);torch.manual_seed(5)
    config={'ch':8,'mults':[1,2,4],'blocks':1,'dropout':0.}
    model=build('planner',config);torch.nn.init.normal_(model.net.out[-1].weight,std=.03)
    norm=StructureNorm({'cond_mean':0.,'cond_std':40.,'res_std':2.})
    coarse=encode(torch.full((1,1,18,18),1000.));base,up,cin=norm.condition(coarse)
    target=encode(base);plan=codec().encode(base,base,'continuous');s=torch.tensor([.5]);noise=torch.randn(1,5,64,64)
    a=loss_parts(model,'planner',norm,target,base,up,cin,plan,s,noise)['total']
    changed=noise.clone();changed[:,:1]+=1000
    b=loss_parts(model,'planner',norm,target+100,base,up,cin,plan,s,changed)['total']
    torch.testing.assert_close(a,b,atol=0,rtol=0);a.backward()
    assert model.net.inp.weight.grad[:,0].abs().max()==0
    assert model.net.out[-1].weight.grad[:1].abs().max()==0
    assert model.net.out[-1].weight.grad[1:].abs().sum()>0


def test_planner_sampling_starts_and_stays_height_zero_and_is_deterministic():
    class Spy(torch.nn.Module):
        def forward(self,x,sigma,cin):
            assert x[:,:1].abs().max()==0
            return torch.zeros_like(x)
    norm=StructureNorm({'cond_mean':0.,'cond_std':40.,'res_std':2.})
    c=encode(torch.full((1,1,18,18),1000.));a=sample_planner(Spy(),norm,c,steps=2,device='cpu')
    b=sample_planner(Spy(),norm,c,steps=2,device='cpu');torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA autocast regression')
def test_continuous_physics_ignores_mixed_precision():
    z,b=fields();z,b=z.cuda(),b.cuda();expected=continuous_fields(z,b)
    with torch.autocast('cuda',dtype=torch.float16):actual=continuous_fields(z,b)
    assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
