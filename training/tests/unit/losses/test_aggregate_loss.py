# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.losses.aggregate import TimeAggregateLossWrapper
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS
from anemoi.training.utils.enums import TensorDim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MAELossFn(FunctionalLoss):
    """Minimal MAE-style functional loss for testing."""

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target)


def _make_loss() -> FunctionalLoss:
    """Return an MAE loss with a unit grid scaler (4 grid points)."""
    loss = MAELossFn()
    loss.add_scaler(TensorDim.GRID, torch.ones(4), name="unit_grid")
    return loss


def _make_crps_loss() -> AlmostFairKernelCRPS:
    """Return an AlmostFairKernelCRPS loss with a unit grid scaler (4 grid points)."""
    loss = AlmostFairKernelCRPS(no_autocast=False)
    loss.add_scaler(TensorDim.GRID, torch.ones(4), name="unit_grid")
    return loss


# Shapes used throughout: (bs=1, time=3, ens=1, latlon=4, nvar=2)
BS, TIME, ENS, LATLON, NVAR = 1, 3, 1, 4, 2
# CRPS requires ens > 1
ENS_CRPS = 3


@pytest.fixture
def pred() -> torch.Tensor:
    return torch.rand(BS, TIME, ENS, LATLON, NVAR)


@pytest.fixture
def target() -> torch.Tensor:
    return torch.rand(BS, TIME, LATLON, NVAR)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_is_base_loss() -> None:
    wrapper = TimeAggregateLossWrapper(["mean"], _make_loss())
    assert isinstance(wrapper, BaseLoss)


def test_stores_loss_fn_and_agg_types() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean", "diff"], inner)
    assert wrapper.loss_fn is inner
    assert wrapper.time_aggregation_types == ["mean", "diff"]


# ---------------------------------------------------------------------------
# Output shape / type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_op", ["mean", "min", "max", "diff"])
def test_returns_scalar_tensor(agg_op: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    wrapper = TimeAggregateLossWrapper([agg_op], _make_loss())
    result = wrapper(pred, target)
    assert isinstance(result, torch.Tensor)
    assert result.numel() == 1


def test_multiple_agg_ops_return_scalar(pred: torch.Tensor, target: torch.Tensor) -> None:
    wrapper = TimeAggregateLossWrapper(["mean", "max", "diff"], _make_loss())
    result = wrapper(pred, target)
    assert result.numel() == 1


# ---------------------------------------------------------------------------
# Empty aggregation list
# ---------------------------------------------------------------------------


def test_empty_aggregation_returns_zero(pred: torch.Tensor, target: torch.Tensor) -> None:
    wrapper = TimeAggregateLossWrapper([], _make_loss())
    result = wrapper(pred, target)
    assert torch.allclose(result, torch.zeros(1))


# ---------------------------------------------------------------------------
# Correctness: perfect predictions yield zero loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_op", ["mean", "min", "max", "diff"])
def test_zero_loss_for_perfect_predictions(agg_op: str) -> None:
    x = torch.rand(BS, TIME, ENS, LATLON, NVAR)
    # target matches pred (broadcast ens dimension away)
    perfect_target = x[:, :, 0, :, :]  # (bs, time, latlon, nvar)
    wrapper = TimeAggregateLossWrapper([agg_op], _make_loss())
    result = wrapper(x, perfect_target)
    assert torch.allclose(result, torch.zeros(1), atol=1e-6), f"{agg_op}: expected zero loss for perfect predictions"


# ---------------------------------------------------------------------------
# Correctness: accumulation across multiple  time aggregation types
# ---------------------------------------------------------------------------


def test_loss_accumulates_across_agg_ops(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Combined wrapper loss equals sum of individual wrapper losses."""
    inner = _make_loss()

    wrapper_mean = TimeAggregateLossWrapper(["mean"], inner)
    wrapper_diff = TimeAggregateLossWrapper(["diff"], inner)
    wrapper_both = TimeAggregateLossWrapper(["mean", "diff"], inner)

    loss_mean = wrapper_mean(pred, target)
    loss_diff = wrapper_diff(pred, target)
    loss_both = wrapper_both(pred, target)

    assert torch.allclose(loss_both, loss_mean + loss_diff, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: "diff" aggregation uses temporal differences
# ---------------------------------------------------------------------------


def test_diff_aggregation_computes_temporal_differences() -> None:
    """The diff wrapper should apply loss on (pred[:,1:]-pred[:,:-1]) vs (target[:,1:]-target[:,:-1])."""
    inner = _make_loss()

    pred = torch.rand(BS, TIME, ENS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    pred_diff = pred[:, 1:, ...] - pred[:, :-1, ...]
    target_diff = target[:, 1:, ...] - target[:, :-1, ...]

    wrapper_diff = TimeAggregateLossWrapper(["diff"], inner)
    expected = inner(pred_diff, target_diff)
    result = wrapper_diff(pred, target)

    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: "mean"/"min"/"max" aggregation reduces over time dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_op", ["mean", "min", "max"])
def test_reduction_aggregation_reduces_time_dim(agg_op: str) -> None:
    inner = _make_loss()
    pred = torch.rand(BS, TIME, ENS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    agg_fn = getattr(torch, agg_op)
    pred_result = agg_fn(pred, dim=1, keepdim=True)
    target_result = agg_fn(target, dim=1, keepdim=True)
    if agg_op in {"min", "max"}:
        pred_agg = pred_result.values
        target_agg = target_result.values
    else:
        pred_agg = pred_result
        target_agg = target_result

    expected = inner(pred_agg, target_agg)
    result = TimeAggregateLossWrapper([agg_op], inner)(pred, target)

    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# CRPS tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_op", ["mean", "min", "max", "diff"])
def test_crps_returns_scalar_tensor(agg_op: str) -> None:
    """TimeAggregateLossWrapper with AlmostFairKernelCRPS should return a scalar for each agg type."""
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)
    wrapper = TimeAggregateLossWrapper([agg_op], _make_crps_loss())
    result = wrapper(pred, target)
    assert isinstance(result, torch.Tensor)
    assert result.numel() == 1


def test_crps_multiple_agg_ops_return_scalar() -> None:
    """Multiple aggregation types should accumulate into a single scalar."""
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)
    wrapper = TimeAggregateLossWrapper(["mean", "diff"], _make_crps_loss())
    result = wrapper(pred, target)
    assert result.numel() == 1


def test_crps_loss_accumulates_across_agg_ops() -> None:
    """Combined CRPS wrapper loss equals sum of individual wrapper losses."""
    inner = _make_crps_loss()
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    loss_mean = TimeAggregateLossWrapper(["mean"], inner)(pred, target)
    loss_diff = TimeAggregateLossWrapper(["diff"], inner)(pred, target)
    loss_both = TimeAggregateLossWrapper(["mean", "diff"], inner)(pred, target)

    assert torch.allclose(loss_both, loss_mean + loss_diff, atol=1e-6)


@pytest.mark.parametrize("agg_op", ["mean", "min", "max"])
def test_crps_reduction_reduces_time_dim(agg_op: str) -> None:
    """CRPS wrapper with time-reduction passes keepdim=True aggregated tensors to inner loss."""
    inner = _make_crps_loss()
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    agg_fn = getattr(torch, agg_op)
    pred_result = agg_fn(pred, dim=1, keepdim=True)
    target_result = agg_fn(target, dim=1, keepdim=True)
    if agg_op in {"min", "max"}:
        pred_agg = pred_result.values
        target_agg = target_result.values
    else:
        pred_agg = pred_result
        target_agg = target_result

    expected = inner(pred_agg, target_agg, squash_mode="avg")
    result = TimeAggregateLossWrapper([agg_op], inner)(pred, target)

    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Unknown aggregation type raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_agg_op_raises(pred: torch.Tensor, target: torch.Tensor) -> None:
    wrapper = TimeAggregateLossWrapper(["sum"], _make_loss())
    with pytest.raises(ValueError, match="Unknown aggregation type"):
        wrapper(pred, target)


# ---------------------------------------------------------------------------
# ignore_nans flag is forwarded to BaseLoss
# ---------------------------------------------------------------------------


def test_ignore_nans_flag() -> None:
    wrapper =  TimeAggregateLossWrapper(["mean"], _make_loss(), ignore_nans=True)
    assert wrapper.avg_function is torch.nanmean
    assert wrapper.sum_function is torch.nansum


def test_default_no_ignore_nans() -> None:
    wrapper = TimeAggregateLossWrapper(["mean"], _make_loss())
    assert wrapper.avg_function is torch.mean
    assert wrapper.sum_function is torch.sum
