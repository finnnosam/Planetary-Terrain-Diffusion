import numpy as np
import pytest
import torch
from toy_meso.model import upsample_condition
from planet_training.terrain_structure import encode, decode, project, handoff
from scripts.probe_structure_stage_roles import build_branches, component_summary, describe, compare_meso


def test_direct_route_really_removes_intermediate_detail_and_keeps_structure():
    torch.set_num_threads(2)
    coarse = encode(torch.full((1, 1, 18, 18), 1200.))
    up = upsample_condition(coarse, 7)
    base40 = torch.full((1, 1, 64, 64), 1200.)
    q = torch.arange(64)
    ridge = 1400*torch.exp(-((q-35)/3)**2)-650*torch.exp(-((q-25)/4)**2)
    structure = project(base40+ridge[None, None, None], base40)
    q23 = torch.arange(112)
    detail = 130*torch.sin(q23*1.3)[None, None, None].expand_as(up)
    sampled = encode(decode(up)+detail)
    branches, baseline = build_branches(coarse, up, sampled, structure, base40, structure, up, decode(up))
    direct, _, _ = handoff(structure, base40, up, coarse)
    torch.testing.assert_close(branches['oracle-direct'], direct)
    torch.testing.assert_close(branches['oracle-target-detail'], direct)
    torch.testing.assert_close(branches['learned-direct'], direct)
    assert (direct-baseline).abs().max() > 800
    assert (branches['oracle-sampled-detail']-direct).abs().max() > 50
    torch.testing.assert_close(branches['learned-sampled-detail'], branches['oracle-sampled-detail'])


def test_quiet_high_plateau_and_connected_versus_fragmented_objects():
    plateau = describe(np.full((52, 52), 3000.), 23040.)
    assert plateau['gentle_fraction_of_land'] == 1
    assert plateau['positive_prominence_objects']['count'] == 0
    assert plateau['negative_prominence_objects']['count'] == 0
    connected = np.zeros((20, 20), bool); connected[8:10, 2:18] = True
    fragmented = np.zeros_like(connected)
    fragmented[2:4, 2:6] = True; fragmented[2:4, 10:14] = True
    fragmented[10:12, 2:6] = True; fragmented[10:12, 10:14] = True
    a = component_summary(connected, 10000.)
    b = component_summary(fragmented, 10000.)
    assert connected.sum() == fragmented.sum()
    assert a['count'] == 1 and b['count'] == 4
    assert a['largest_fraction_of_mask'] == 1 and b['largest_fraction_of_mask'] == .25
    assert a['largest_objects'][0]['elongation'] > 5


def test_local_edges_are_not_wrapped_and_censored_objects_are_reported():
    z = np.zeros((12, 12), bool); z[3:6, :2] = True; z[3:6, -2:] = True
    result = component_summary(z, 10000.)
    assert result['count'] == 2
    assert all(r['edge_censored'] for r in result['largest_objects'])


def test_meso_preservation_uses_actual_supplied_centre_and_physical_means():
    raw = torch.full((1, 1, 128, 128), 1100.)
    guide = torch.full((1, 1, 112, 112), 900.)
    guide[..., 48:64, 48:64] = 1100.
    a = compare_meso(raw, guide)
    assert a['input_footprint_mean_drift_rms_m'] < .001
    guide[..., 48:64, 48:64] += 125
    b = compare_meso(raw, guide)
    assert abs(b['input_footprint_mean_drift_rms_m']-125) < .001


@pytest.mark.parametrize('spacing', [0, -1, np.nan])
def test_invalid_physical_spacing(spacing):
    with pytest.raises(ValueError):
        describe(np.ones((8, 8)), spacing)
