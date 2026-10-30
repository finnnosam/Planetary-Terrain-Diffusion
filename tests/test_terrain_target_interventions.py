import numpy as np
import torch

from scripts.audit_terrain_plan_targets import intervene, objects, continuous_candidate


def test_shuffle_preserves_joint_tuples_within_permission_bands_and_is_deterministic():
    torch.set_num_threads(2)
    base=torch.linspace(-100,1200,64*64).reshape(1,1,64,64)
    p=torch.randn(1,4,64,64)*(base>200)
    shuffled=intervene(p,base,'shuffle')
    torch.testing.assert_close(shuffled,intervene(p,base,'shuffle'),atol=0,rtol=0)
    assert not torch.equal(p,shuffled)
    assert shuffled.masked_select((base<=200).expand_as(shuffled)).abs().max()==0
    for mask in [(base>200)&(base<=400),(base>400)&(base<=600),base>600]:
        a=p[0,:,mask[0,0]].T.numpy();b=shuffled[0,:,mask[0,0]].T.numpy()
        np.testing.assert_array_equal(a[np.lexsort(a.T)],b[np.lexsort(b.T)])


def test_coarsening_preserves_uniform_corridor_and_never_activates_locked_cells():
    p=torch.zeros(1,4,64,64);p[:,0]=1;p[:,2]=1
    base=torch.full((1,1,64,64),1000.);base[...,:5]=100
    p*=base>200
    for mode in ['coarse80','coarse160']:
        q=intervene(p,base,mode)
        assert q[...,:5].abs().max()==0
        torch.testing.assert_close(q[...,8:],p[...,8:])
        assert q.abs().max()<=1
    q=intervene(p,base,'no-axis')
    torch.testing.assert_close(q[:,:2],p[:,:2]);assert q[:,2:].abs().max()==0


def test_connectivity_distinguishes_connected_belt_from_scattered_equal_area():
    belt=np.zeros((16,16),bool);belt[3,2:10]=True
    scattered=np.zeros_like(belt);scattered[::4,::8]=True
    a,b=objects(belt),objects(scattered)
    assert a['cells']==b['cells']==8
    assert a['components']==1 and b['components']==8
    assert a['largest_fraction']==1 and b['largest_fraction']==1/8


def test_candidate_does_not_turn_flat_coastal_land_into_mountains_and_is_scalar_rotation_equivariant():
    torch.set_num_threads(2)
    base=torch.full((1,1,64,64),1200.);base[...,:20]=-3000.
    flat=continuous_candidate(base,base)
    assert flat.abs().max()<2.  # float32 variance cancellation, sub-metre relative positions
    q=torch.arange(64).float();z=base+300*torch.exp(-((q-40)/3).square())[None,None,None,:]*(base>0)
    p=continuous_candidate(z,base)
    assert p.shape==(1,4,64,64) and torch.isfinite(p).all()
    assert p[...,:20].abs().max()==0
    rotated=continuous_candidate(torch.rot90(z,1,(-2,-1)),torch.rot90(base,1,(-2,-1)))
    torch.testing.assert_close(rotated,torch.rot90(p,1,(-2,-1)),atol=1.,rtol=.002)
