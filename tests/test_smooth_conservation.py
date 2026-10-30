import torch

from planet_training.terrain_structure import constraint_metrics, project
from planet_training.smooth_conservation import project_smooth, grid_jumps


def test_smooth_correction_preserves_shelves_means_and_does_not_add_block_jumps():
    torch.set_num_threads(2)
    q = torch.arange(128).float()
    base = torch.full((1, 1, 128, 128), 1000.)
    raw = base+80*torch.sin(q[None, None, None, :]/37)+40*torch.cos(q[None, None, :, None]/23)
    old = project(raw, base, 8, 2880.)
    smooth, info = project_smooth(raw, base, return_info=True)
    values = constraint_metrics(smooth, base, 8, 2880.)
    assert values['added_footprint_mean_drift_max_m'] < .002
    assert values['sign_disagreement_fraction'] == 0 and info[0]['positivity_scale'] == 1.
    assert grid_jumps(old)['boundary_to_interior_ratio'] > 2
    assert grid_jumps(smooth)['boundary_to_interior_ratio'] < 1.4


def test_mixed_coasts_and_positivity_preserve_constraints_and_batch():
    torch.manual_seed(42)
    base = torch.full((2, 1, 64, 64), 800.)
    base[..., :20] = -100.; base[..., 20:24] = 150.; base[..., 24:26] = 210.
    raw = base+torch.randn_like(base)*3000
    smooth, info = project_smooth(raw, base, return_info=True)
    values = constraint_metrics(smooth, base, 8, 2880.)
    assert len(info) == 2 and any(v['positivity_scale'] < 1 for v in info)
    assert values['protected_height_change_max_m'] == 0
    assert values['sign_disagreement_fraction'] == 0
    assert values['added_footprint_mean_drift_max_m'] < .002
    protected = base <= 200
    torch.testing.assert_close(smooth[protected], base[protected])
    assert smooth[~protected].min() >= .99
    sea = torch.full((1, 1, 64, 64), -300.)
    torch.testing.assert_close(project_smooth(torch.zeros_like(sea), sea), sea)
