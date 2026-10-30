"""Structure-plan v2, arrangement metrics and rule-based plans. CPU only; no data files or checkpoints."""
import numpy as np
import pytest
import torch

from planet_training import arrangement_metrics as am
from planet_training import procedural_plan as pp
from planet_training import structure_plan_v2 as v2
from planet_training.terrain_joint_structure import build as build_joint
from planet_training.terrain_spatial_plan import PlanCodec as V1Codec, continuous_fields
from planet_training.terrain_structure import area_weights, blocks, project

STATS = {'mean': [.1, -.2, .3, .2], 'std': [.4, .6, .8, .5]}


def guide():
    """Ocean strip, a shallow coastal strip (<=200 m) and an upland ridge."""
    base = torch.full((1, 1, 64, 64), 900.)
    base[..., :10] = -2000.; base[..., 10:16] = 120.
    x = torch.arange(64).float()
    base = base+(base > 200)*600*torch.exp(-.5*((x-40)/8)**2)[None, None, None, :]
    return base


def test_locked_policy_reproduces_existing_projection_exactly():
    torch.manual_seed(0); base = guide(); raw = base+300*torch.randn_like(base)
    for feather in (False, True):
        torch.testing.assert_close(v2.project_with(raw, base, 'locked-200', feather=feather), project(raw, base, feather=feather), atol=0, rtol=0)


def test_coast_only_policy_keeps_ocean_sign_and_regional_means():
    torch.manual_seed(1); base = guide(); raw = base+600*torch.randn_like(base)
    z = v2.project_with(raw, base, 'coast-only')
    ocean = base <= 0
    assert torch.equal(z[ocean], base[ocean])
    assert (z[~ocean] >= 1-1e-4).all()
    land_shallow = (base > 0) & (base <= 200)
    assert (z[land_shallow]-base[land_shallow]).abs().max() > 1, 'coast-only must let low land change'
    check = v2.constraint_check(z, base, 'coast-only', 16, 40320.)
    assert check['sign_disagreement_fraction'] == 0
    locked = v2.project_with(raw, base, 'locked-200')
    assert torch.equal(locked[land_shallow], base[land_shallow])


def test_v2_codec_matches_v1_continuous_for_locked_policy():
    base = guide(); z = base+200*torch.sin(torch.arange(64).float()/3)[None, None, None, :]*(base > 0)
    old = V1Codec({'categorical': STATS, 'continuous': STATS}); new = v2.PlanCodec(STATS, 'locked-200')
    torch.testing.assert_close(new.encode(z, base), old.encode(z, base, 'continuous'))
    torch.testing.assert_close(new.decode(new.encode(z, base), base), continuous_fields(z, base)*(base > 200), atol=1e-3, rtol=1e-4)


def test_planner_copy_from_v1_is_functionally_exact():
    torch.manual_seed(2); config = {'ch': 8, 'mults': [1, 2, 4], 'blocks': 1, 'dropout': 0.}
    old = build_joint(config)
    torch.nn.init.normal_(old.net.out[-1].weight, std=.05); torch.nn.init.normal_(old.net.out[-1].bias, std=.05)
    new = v2.build_planner(config); v2.planner_from_v1(old.state_dict(), new)
    plan = torch.randn(2, 4, 64, 64); cin = torch.randn(2, 2, 64, 64); sigma = torch.tensor([.3, 2.])
    a = old(torch.cat((torch.zeros(2, 1, 64, 64), plan), 1), sigma, cin)[:, 1:]
    torch.testing.assert_close(new(plan, sigma, cin), a, atol=1e-5, rtol=1e-5)


def test_wider_planner_builds_and_trains_one_step():
    torch.manual_seed(3); config = v2.unet_config({'ch': 8, 'mults': [1, 2, 4], 'blocks': 1, 'dropout': 0.}, width=16)
    model = v2.build_planner(config); assert model.net.inp.out_channels == 16
    base = guide(); active = v2.active_for(base, 'locked-200').float()
    loss = v2.planner_loss(model, torch.randn(1, 4, 64, 64)*active, torch.randn(1, 2, 64, 64), active, torch.tensor([.5]), torch.randn(1, 4, 64, 64))
    loss.backward(); assert torch.isfinite(loss)


@pytest.mark.parametrize('layout', pp.LAYOUTS)
def test_rule_based_layouts_are_deterministic_finite_and_masked(layout):
    base = guide(); codec = v2.PlanCodec(STATS, 'locked-200')
    p1, r1 = pp.plan(layout, base, codec, 7); p2, r2 = pp.plan(layout, base, codec, 7)
    assert torch.equal(p1, p2) and torch.isfinite(p1).all()
    frozen = ~v2.active_for(base, 'locked-200')
    assert r1[frozen].abs().max() == 0 and p1.expand(-1, 4, -1, -1)[frozen.expand(-1, 4, -1, -1)].abs().max() == 0
    assert r1.abs().max() > 300, 'layout should author real relief'
    assert not torch.equal(pp.plan(layout, base, codec, 8)[0], p1)


def two_ranges_and_a_valley(angle=0.):
    """Synthetic California-like 40 km field: two parallel ranges with a connected trough."""
    y, x = np.mgrid[:64, :64].astype(float)
    u = (x-32)*np.cos(angle)+(y-32)*np.sin(angle); v = -(x-32)*np.sin(angle)+(y-32)*np.cos(angle)
    along = np.clip(1-(np.abs(u)/22)**4, 0, 1)
    z = 800+along*(2000*np.exp(-.5*((v+4)/1.5)**2)-600*np.exp(-.5*(v/1.3)**2)+1000*np.exp(-.5*((v-4)/1.2)**2))
    return z.astype(np.float32), np.full((64, 64), 800, np.float32)


def test_metrics_detect_paired_ranges_and_ignore_a_ramp():
    z, base = two_ranges_and_a_valley()
    m = am.arrangement(z, base)
    assert m['long_ridge_fraction'] > .5 and m['flanked_low_fraction'] > .02 and m['connected_low_fraction'] > .5
    ramp = (np.linspace(200, 2500, 64)[None, :]*np.ones((64, 1))).astype(np.float32)
    r = am.arrangement(ramp, base)
    assert r['long_ridge_fraction'] == 0 and r['flanked_low_fraction'] == 0
    assert am.structured(m) and not am.structured(r)
    assert am.compare(m, m)['pass'] and not am.compare(r, m)['pass']


def test_metrics_are_invariant_to_grid_rotation_and_reflection():
    z, base = two_ranges_and_a_valley(.4)
    a = am.arrangement(z, base)
    for k in range(1, 4):
        b = am.arrangement(np.ascontiguousarray(np.rot90(z, k)), base)
        for key in am.METRICS:
            assert b[key] == pytest.approx(a[key], rel=1e-4, abs=1e-6), key
    c = am.arrangement(np.ascontiguousarray(z[:, ::-1]), base)
    assert c['flanked_low_fraction'] == pytest.approx(a['flanked_low_fraction'], rel=1e-4)


def test_components_and_nearest_training_correlation():
    mask = np.zeros((10, 10), bool); mask[1, 1:4] = True; mask[6:9, 6] = True; mask[2, 4] = True
    labels, n = am.components(mask); assert n == 2
    rng = np.random.default_rng(0); fields = rng.normal(size=(5, 64, 64)).astype(np.float32)
    library = am.plan_signature(fields)
    assert am.nearest_training_correlation(np.rot90(fields[3], 1), library) == pytest.approx(1, abs=1e-5)
    assert am.nearest_training_correlation(rng.normal(size=(64, 64)), library) < .3
