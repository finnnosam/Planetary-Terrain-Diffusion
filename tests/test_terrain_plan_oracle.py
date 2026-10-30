import numpy as np
import pytest
import torch

from planet_training.terrain_structure import build as old_build, StructureNorm, encode, project
from planet_training.terrain_plan_oracle import build, initialize_from_source, plan_for, condition, loss


def fields():
    q = torch.arange(64).float()
    ridge = 1300*torch.exp(-((q-36)/3)**2)-700*torch.exp(-((q-25)/4)**2)
    base = torch.full((1, 1, 64, 64), 1200.)
    base[..., :8] = -300.; base[..., 8:12] = 100.
    return project(base+ridge[None, None, None], base), base


def test_plan_has_physical_classes_directions_and_no_forbidden_information():
    torch.set_num_threads(2)
    target, base = fields(); plan = plan_for(target, base)
    assert plan.shape == (1, 4, 64, 64) and torch.isfinite(plan).all()
    assert plan.abs().max() <= 1.000001
    assert plan[..., :12].abs().max() == 0
    assert (plan[:, 0] == 1).any() and (plan[:, 0] == -1).any()
    assert plan[:, 2:4].abs().max() > .5
    flat = plan_for(torch.full_like(base, 3000.), torch.full_like(base, 3000.))
    assert flat[:, 0].abs().max() == 0 and flat[:, 2:4].abs().max() == 0
    assert flat[:, 1].min() == 1
    rotated = plan_for(torch.rot90(target, 1, (-2, -1)), torch.rot90(base, 1, (-2, -1)))
    torch.testing.assert_close(rotated[:, :2], torch.rot90(plan[:, :2], 1, (-2, -1)))
    torch.testing.assert_close(rotated[:, 2:4], -torch.rot90(plan[:, 2:4], 1, (-2, -1)), atol=1e-4, rtol=1e-4)


def test_initialization_preserves_source_response_even_with_nonzero_oracle_plan():
    torch.manual_seed(1)
    config = {'ch': 8, 'mults': [1, 2, 4], 'blocks': 1, 'dropout': 0.}
    old = old_build(config).eval()
    # Avoid a trivially zero output layer in this initialization check.
    torch.nn.init.normal_(old.net.out[-1].weight, std=.05)
    new = build(config).eval(); initialize_from_source(new, old.state_dict())
    x = torch.randn(1, 1, 64, 64); height_mask = torch.randn(1, 2, 64, 64)
    plan = torch.randn(1, 4, 64, 64).clamp(-1, 1); sigma = torch.tensor([.5])
    torch.testing.assert_close(new(x, sigma, torch.cat((height_mask, plan), 1)), old(x, sigma, height_mask), atol=1e-6, rtol=1e-5)
    assert new.net.inp.weight[:, 3:].abs().max() == 0


def test_control_and_oracle_are_matched_before_learning_and_plan_weights_get_gradients():
    torch.set_num_threads(2)
    config = {'ch': 8, 'mults': [1, 2, 4], 'blocks': 1, 'dropout': 0.}
    old = old_build(config)
    torch.nn.init.normal_(old.net.out[-1].weight, std=.05)
    model = build(config); initialize_from_source(model, old.state_dict())
    norm = StructureNorm({'cond_mean': 0., 'cond_std': 40., 'res_std': 2.})
    raw, _ = fields(); coarse = encode(torch.full((1, 1, 18, 18), 1200.))
    torch.manual_seed(7); control = loss(model, norm, raw, coarse, 'control')
    torch.manual_seed(7); oracle = loss(model, norm, raw, coarse, 'oracle')
    torch.testing.assert_close(control, oracle)
    oracle.backward()
    assert torch.isfinite(model.net.inp.weight.grad).all()
    assert model.net.inp.weight.grad[:, 3:].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA precision regression')
def test_training_autocast_keeps_the_exact_review_plan_and_high_plains():
    target, base = fields(); target, base = target.cuda(), base.cuda()
    expected = plan_for(target, base)
    with torch.autocast('cuda', dtype=torch.float16):
        actual = plan_for(target, base)
        elevated_plain = plan_for(torch.full_like(base, 3500.), torch.full_like(base, 3500.))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    assert elevated_plain[:, 1].min() == 1
    assert elevated_plain[:, 0].abs().max() == 0
