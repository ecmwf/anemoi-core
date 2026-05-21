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
from anemoi.training.losses.kcrps import CRPS
from anemoi.training.losses.mae import MAELoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.utils.enums import TensorDim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loss() -> MAELoss:
    """Return an MAE loss with a unit grid scaler (4 grid points)."""
    loss = MAELoss()
    loss.add_scaler(TensorDim.GRID, torch.ones(4), name="unit_grid")
    return loss


def _make_crps_loss() -> CRPS:
    """Return a CRPS loss with a unit grid scaler (4 grid points)."""
    loss = CRPS(no_autocast=False)
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


def test_stores_loss_and_agg_types() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean", "diff"], inner)
    assert wrapper.loss is inner
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
# Correctness: accumulation across multiple  time aggregation types
# ---------------------------------------------------------------------------


def test_loss_accumulates_across_agg_ops(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Combined wrapper loss equals average of individual wrapper losses."""
    inner = _make_loss()

    wrapper_mean = TimeAggregateLossWrapper(["mean"], inner)
    wrapper_diff = TimeAggregateLossWrapper(["diff"], inner)
    wrapper_both = TimeAggregateLossWrapper(["mean", "diff"], inner)

    loss_mean = wrapper_mean(pred, target)
    loss_diff = wrapper_diff(pred, target)
    loss_both = wrapper_both(pred, target)

    assert torch.allclose(loss_both, (loss_mean + loss_diff) / 2, atol=1e-6)


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
    # The wrapper iterates per diff-step to handle time scalers correctly.
    expected = torch.tensor(0.0)
    for step in range(pred_diff.shape[1]):
        expected = expected + inner(pred_diff[:, step : step + 1, ...], target_diff[:, step : step + 1, ...])
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

    if agg_op == "min":
        pred_agg = torch.amin(pred, dim=1, keepdim=True)
        target_agg = torch.amin(target, dim=1, keepdim=True)
    elif agg_op == "max":
        pred_agg = torch.amax(pred, dim=1, keepdim=True)
        target_agg = torch.amax(target, dim=1, keepdim=True)
    else:
        pred_agg = torch.mean(pred, dim=1, keepdim=True)
        target_agg = torch.mean(target, dim=1, keepdim=True)

    expected = inner(pred_agg, target_agg)
    result = TimeAggregateLossWrapper([agg_op], inner)(pred, target)

    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# CRPS tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_op", ["mean", "min", "max", "diff"])
def test_crps_returns_scalar_tensor(agg_op: str) -> None:
    """TimeAggregateLossWrapper with CRPS should return a scalar for each agg type."""
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
    """Combined CRPS wrapper loss equals average of individual wrapper losses."""
    inner = _make_crps_loss()
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    loss_mean = TimeAggregateLossWrapper(["mean"], inner)(pred, target)
    loss_diff = TimeAggregateLossWrapper(["diff"], inner)(pred, target)
    loss_both = TimeAggregateLossWrapper(["mean", "diff"], inner)(pred, target)

    assert torch.allclose(loss_both, (loss_mean + loss_diff) / 2, atol=1e-6)


@pytest.mark.parametrize("agg_op", ["mean", "min", "max"])
def test_crps_reduction_reduces_time_dim(agg_op: str) -> None:
    """CRPS wrapper with time-reduction passes keepdim=True aggregated tensors to inner loss."""
    inner = _make_crps_loss()
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)

    if agg_op == "min":
        pred_agg = torch.amin(pred, dim=1, keepdim=True)
        target_agg = torch.amin(target, dim=1, keepdim=True)
    elif agg_op == "max":
        pred_agg = torch.amax(pred, dim=1, keepdim=True)
        target_agg = torch.amax(target, dim=1, keepdim=True)
    else:
        pred_agg = torch.mean(pred, dim=1, keepdim=True)
        target_agg = torch.mean(target, dim=1, keepdim=True)

    expected = inner(pred_agg, target_agg)
    result = TimeAggregateLossWrapper([agg_op], inner)(pred, target)

    assert torch.allclose(result, expected, atol=1e-6)


def test_crps_wrapper_forwards_explicit_squash_mode() -> None:
    inner = _make_crps_loss()
    pred = torch.rand(BS, TIME, ENS_CRPS, LATLON, NVAR)
    target = torch.rand(BS, TIME, LATLON, NVAR)
    pred_mean = torch.mean(pred, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=1, keepdim=True)

    expected = inner(pred_mean, target_mean, squash_mode="avg")
    result = TimeAggregateLossWrapper(["mean"], inner)(pred, target, squash_mode="avg")

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
    wrapper = TimeAggregateLossWrapper(["mean"], _make_loss(), ignore_nans=True)
    assert wrapper.avg_function is torch.nanmean
    assert wrapper.sum_function is torch.nansum


def test_default_no_ignore_nans() -> None:
    wrapper = TimeAggregateLossWrapper(["mean"], _make_loss())
    assert wrapper.avg_function is torch.mean
    assert wrapper.sum_function is torch.sum


# ---------------------------------------------------------------------------
# Transparent wrapper: scaler delegation
# ---------------------------------------------------------------------------


def test_scaler_is_shared_with_inner_loss() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    assert wrapper.scaler is inner.scaler


def test_add_scaler_reaches_inner_loss() -> None:
    inner = MAELoss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    scaler = torch.ones(NVAR)
    wrapper.add_scaler(TensorDim.VARIABLE, scaler, name="var_scaler")
    assert inner.scaler.has_scaler_for_dim(TensorDim.VARIABLE)


def test_update_scaler_delegates_to_inner_loss() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    new_grid = torch.ones(4) * 2.0
    wrapper.update_scaler("unit_grid", new_grid, override=True)
    assert torch.allclose(inner.scaler.unit_grid, new_grid)


def test_has_scaler_for_dim_delegates() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    assert wrapper.has_scaler_for_dim(TensorDim.GRID) is True
    assert wrapper.has_scaler_for_dim(TensorDim.VARIABLE) is False


# ---------------------------------------------------------------------------
# Transparent wrapper: metadata delegation
# ---------------------------------------------------------------------------


def test_supports_sharding_matches_inner() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    assert wrapper.supports_sharding == inner.supports_sharding


def test_supports_sharding_propagates_false() -> None:
    inner = _make_loss()
    inner.supports_sharding = False
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    assert wrapper.supports_sharding is False


def test_needs_shard_layout_info_default_false() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    assert wrapper.needs_shard_layout_info is False


def test_iter_leaf_losses_yields_inner_leaves() -> None:
    inner = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner)
    leaves = list(wrapper.iter_leaf_losses())
    assert leaves == [inner]
    assert wrapper not in leaves


# ---------------------------------------------------------------------------
# Nested composition: TimeAggregateLossWrapper(MultiscaleLossWrapper(...))
# ---------------------------------------------------------------------------


def _make_multiscale_wrapper(inner: BaseLoss | None = None) -> "MultiscaleLossWrapper":
    """Build a single-scale MultiscaleLossWrapper (no smoothing matrices)."""
    if inner is None:
        inner = _make_loss()
    return MultiscaleLossWrapper(
        per_scale_loss=inner,
        weights=[1.0],
    )


def test_nested_scaler_shared_through_chain() -> None:
    leaf = _make_loss()
    ms = _make_multiscale_wrapper(leaf)
    wrapper = TimeAggregateLossWrapper(["mean"], ms)
    # All three should share the same scaler
    assert wrapper.scaler is ms.scaler
    assert ms.scaler is leaf.scaler


def test_nested_add_scaler_reaches_leaf() -> None:
    leaf = MAELoss()
    ms = _make_multiscale_wrapper(leaf)
    wrapper = TimeAggregateLossWrapper(["mean"], ms)
    wrapper.add_scaler(TensorDim.GRID, torch.ones(4), name="grid_w")
    assert leaf.scaler.has_scaler_for_dim(TensorDim.GRID)


def test_nested_iter_leaf_losses_reaches_innermost() -> None:
    leaf = _make_loss()
    ms = _make_multiscale_wrapper(leaf)
    wrapper = TimeAggregateLossWrapper(["mean"], ms)
    # MultiscaleLossWrapper inherits default iter_leaf_losses (yields self),
    # so the leaf list should be [ms], not [wrapper]
    leaves = list(wrapper.iter_leaf_losses())
    assert wrapper not in leaves
    assert ms in leaves


# ---------------------------------------------------------------------------
# CombinedLoss integration
# ---------------------------------------------------------------------------


def test_combined_loss_scaler_reaches_wrapped_inner() -> None:
    inner1 = MAELoss()
    inner2 = MAELoss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner2)

    # Verify that add_scaler on the wrapper propagates to the inner loss
    grid_scaler = torch.ones(4)
    wrapper.add_scaler(TensorDim.GRID, grid_scaler, name="node_weights")
    inner1.add_scaler(TensorDim.GRID, grid_scaler, name="node_weights")

    # Both leaf losses should have the scaler
    assert inner1.scaler.has_scaler_for_dim(TensorDim.GRID)
    assert inner2.scaler.has_scaler_for_dim(TensorDim.GRID)


def test_combined_loss_iter_leaf_losses_includes_wrapped() -> None:
    from anemoi.training.losses.combined import CombinedLoss

    inner1 = _make_loss()
    inner2 = _make_loss()
    wrapper = TimeAggregateLossWrapper(["mean"], inner2)

    combined = CombinedLoss(losses=[inner1, wrapper])
    leaves = list(combined.iter_leaf_losses())
    assert inner1 in leaves
    assert inner2 in leaves
    assert wrapper not in leaves
