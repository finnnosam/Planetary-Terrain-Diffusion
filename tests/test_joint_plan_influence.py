import torch

from scripts.probe_joint_plan_influence import states_from_anchor, rotate_plan, response_record
from planet_training.terrain_structure import StructureNorm, encode


def test_plan_interventions_preserve_height_noise_and_protected_mask():
    torch.manual_seed(3)
    base=torch.full((1,1,64,64),1200.);base[...,:9]=100.
    clean=torch.randn(1,5,64,64);clean[:,1:]*=base>200
    donor=torch.randn_like(clean);donor[:,1:]*=base>200
    noise=torch.randn_like(clean)
    for sigma in [.05,.5,10.]:
        variants=states_from_anchor(clean,donor,noise,sigma,base)
        for k,x in variants.items():
            if k!='height_noise_control':torch.testing.assert_close(x[:,:1],variants['same'][:,:1],atol=0,rtol=0)
            assert x[:,1:,:,:9].abs().max()==0
        assert not torch.equal(variants['height_noise_control'][:,:1],variants['same'][:,:1])
        torch.testing.assert_close(variants['same'],variants['repeat'],atol=0,rtol=0)


def test_rotation_transforms_axial_vectors_with_scalar_maps():
    p=torch.zeros(1,4,8,8);p[:,0,:,2]=1.;p[:,2,:,2]=1.;p[:,1,1:3,2:4]=1.
    result=rotate_plan(p)
    torch.testing.assert_close(result[:,:2],torch.rot90(p[:,:2],1,(-2,-1)))
    torch.testing.assert_close(result[:,2:],-torch.rot90(p[:,2:],1,(-2,-1)))


def test_raw_and_projected_sensitivity_separate_locked_cells():
    base=torch.full((1,1,64,64),100.)
    norm=StructureNorm({'cond_mean':0.,'cond_std':40.,'res_std':2.})
    reference=torch.zeros(1,5,64,64);changed=reference.clone();changed[:,:1]=.2
    row=response_record(changed,changed,reference,reference,base,encode(base),norm)
    assert row['raw_height_response_active_rms_m'] is None
    assert row['projected_height_response_active_rms_m'] is None
    base.fill_(1200.)
    row=response_record(changed,changed,reference,reference,base,encode(base),norm)
    assert row['raw_height_response_active_rms_m']>0
    assert row['projected_height_response_active_rms_m']<.01
