# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.losses import ReversedHuberLoss


def _paper_reference(err: torch.Tensor, delta: float, sharpness: float) -> torch.Tensor:
    """Eqs. B.3-B.4 of Pereira et al. (2026), written independently."""
    w = 1.0 / (1.0 + torch.exp(-sharpness * (err.abs() - delta)))
    return (1.0 - w) * delta * err.abs() + w * 0.5 * err**2


@pytest.fixture
def tensors() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    shape = (2, 1, 1, 128, 6)  # (bs, time, ens, grid, vars)
    return torch.randn(shape) * 3.0, torch.randn(shape)


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_matches_paper_formula(tensors, delta):
    pred, target = tensors
    loss = ReversedHuberLoss(delta=delta)
    out = loss.calculate_difference(pred, target)
    torch.testing.assert_close(out, _paper_reference(pred - target, delta, 2.0))


def test_asymptotic_regimes():
    # Large sharpness makes the blend close to a hard switch.
    loss = ReversedHuberLoss(delta=1.0, sharpness=50.0)
    zero = torch.zeros(1)
    small, large = torch.tensor([0.1]), torch.tensor([10.0])
    torch.testing.assert_close(loss.calculate_difference(small, zero), 1.0 * small.abs())
    torch.testing.assert_close(loss.calculate_difference(large, zero), 0.5 * large**2)


def test_piecewise_continuous_at_delta():
    delta = 1.5
    loss = ReversedHuberLoss(delta=delta, smooth=False)
    zero = torch.zeros(1)
    below = loss.calculate_difference(torch.tensor([delta - 1e-6]), zero)
    above = loss.calculate_difference(torch.tensor([delta + 1e-6]), zero)
    torch.testing.assert_close(below, above, atol=1e-4, rtol=0)


def test_symmetric_nonnegative_and_zero_at_zero(tensors):
    pred, target = tensors
    loss = ReversedHuberLoss()
    a = loss.calculate_difference(pred, target)
    b = loss.calculate_difference(target, pred)
    torch.testing.assert_close(a, b)
    assert (a >= 0).all()
    assert loss.calculate_difference(target, target).abs().max() == 0


def test_gradients_finite_under_bf16(tensors):
    pred, target = (t.to(torch.bfloat16) for t in tensors)
    pred.requires_grad_(True)
    out = ReversedHuberLoss().calculate_difference(pred, target).sum()
    out.backward()
    assert torch.isfinite(pred.grad.float()).all()


def test_forward_returns_scalar(tensors):
    pred, target = tensors
    out = ReversedHuberLoss()(pred, target)
    assert out.ndim == 0
    assert torch.isfinite(out)


@pytest.mark.parametrize("kwargs", [{"delta": 0.0}, {"delta": -1.0}, {"sharpness": 0.0}])
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        ReversedHuberLoss(**kwargs)
