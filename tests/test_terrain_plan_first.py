import torch

from planet_training.terrain_joint_structure import build,targets,loss_parts
from planet_training.terrain_structure import StructureNorm,encode
from planet_training.terrain_plan_first import parts,sample_plan


def fixture():
    torch.set_num_threads(2);torch.manual_seed(3)
    model=build({'ch':8,'mults':[1,2,4],'blocks':1,'dropout':0.})
    torch.nn.init.normal_(model.net.out[-1].weight,std=.02)
    norm=StructureNorm({'cond_mean':0.,'cond_std':40.,'res_std':2.})
    c=encode(torch.full((1,1,18,18),1200.));raw=torch.full((1,1,64,64),1200.)
    raw[...,25:33]+=900;raw[...,38:46]-=500
    packet=targets(norm,raw,c,'joint');sigma=torch.tensor([.5]);noise=torch.randn_like(packet[0])
    return model,norm,c,raw,packet,sigma,noise


def test_planner_loss_cannot_use_target_or_noisy_height_and_trains_plan_head():
    model,norm,c,raw,packet,sigma,noise=fixture()
    x,cin,target,base,up=packet
    first=parts(model,'planner',x,cin,target,base,up,norm,sigma,noise)
    altered=x.clone();altered[:,:1]+=100
    other_noise=noise.clone();other_noise[:,:1]*=-19
    again=parts(model,'planner',altered,cin,target+100,base,up,norm,sigma,other_noise)
    torch.testing.assert_close(first['total'],again['total'],atol=0,rtol=0)
    first['total'].backward()
    assert model.net.out[-1].weight.grad[:1].abs().max()==0
    assert model.net.out[-1].weight.grad[1:].abs().sum()>0
    assert model.net.inp.weight.grad[:,0].abs().max()==0


def test_coupled_loss_matches_existing_joint_objective_with_shared_noise():
    model,norm,c,raw,packet,sigma,_=fixture()
    g=torch.Generator().manual_seed(777);noise=torch.randn(packet[0].shape,generator=g)
    actual=parts(model,'coupled',packet[0],packet[1],packet[2],packet[3],packet[4],norm,sigma,noise)
    previous=loss_parts(model,norm,raw,c,'joint',torch.Generator().manual_seed(777),sigma)
    for k in actual:torch.testing.assert_close(actual[k],previous[k],atol=0,rtol=0)


def test_dedicated_sampling_uses_coarse_guide_and_no_height_state():
    _,norm,c,_,_,_,_=fixture();c[...,:3]=10.
    class Recorder(torch.nn.Module):
        def forward(self,x,sigma,cin):
            assert x[:,:1].abs().max()==0 and cin.shape[1]==2
            return .5*x
    first=sample_plan(Recorder(),norm,c,'planner',steps=4,seed=42,device='cpu')
    again=sample_plan(Recorder(),norm,c,'planner',steps=4,seed=42,device='cpu')
    torch.testing.assert_close(first,again,atol=0,rtol=0)
    assert first.shape==(1,4,64,64) and torch.isfinite(first).all()
    assert first[...,:4].abs().max()==0
