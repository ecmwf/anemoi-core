# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
from collections.abc import Callable

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pydantic import ValidationError
from pytest_mock import MockerFixture

from anemoi.models.data import Source
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CRPS
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import TargetWeightedLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

GRID = 6
W0 = 0.01
LAYOUTS = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_FULL}
TARGET = "anemoi.training.losses.TargetWeightedLoss"
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR_LAYOUT = TensorLayout(grid=0, variables=1)

# Data (target) columns:       a=0, f=1 (forcing), tc=2 (diagnostic), b=3
# Model output (pred) columns: a=0, tc=1, b=2
NAME_TO_INDEX = {"a": 0, "f": 1, "tc": 2, "b": 3}
OUTPUT_COLUMN = {"a": 0, "tc": 1, "b": 2}
DATA_COLUMN = {"a": 0, "tc": 2, "b": 3}
OUTPUT_NAMES = ["a", "tc", "b"]
DATA_NAMES = ["a", "f", "tc", "b"]

# Point-wise error of the inner losses
ERROR_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "MSELoss": torch.square,
    "MAELoss": torch.abs,
    "HuberLoss": lambda d: torch.where(d.abs() < 1.0, 0.5 * d**2, 1.0 * (d.abs() - 0.5)),
}
INNER_LOSSES = list(ERROR_FNS)


def _view(data: torch.Tensor, variables: list[str]) -> Source:
    """Gridded source of a five-dimensional tensor with named variables."""
    assert data.shape[-1] == len(variables)
    return GriddedSource(
        name="data",
        values=data,
        variables=variables,
        statistics={},
        coordinates=torch.zeros(data.shape[GRIDDED_LAYOUT.grid], 2, device=data.device),
        layout=GRIDDED_LAYOUT,
    )


def _pred_view(pred: torch.Tensor) -> Source:
    return _view(pred, OUTPUT_NAMES)


def _target_view(target: torch.Tensor) -> Source:
    return _view(target, DATA_NAMES)


def _tabular_view(data: list[torch.Tensor], variables: list[str]) -> Source:
    return TabularSource(
        name="obs",
        values=data,
        variables=variables,
        statistics={},
        coordinates=[torch.zeros(sample.shape[0], 2) for sample in data],
        layout=TABULAR_LAYOUT,
        boundaries=[(slice(0, sample.shape[0]),) for sample in data],
        timedeltas=[torch.zeros(sample.shape[0]) for sample in data],
    )


@pytest.fixture
def data_indices() -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": ["f"], "diagnostic": ["tc"], "target": []}), NAME_TO_INDEX)


@pytest.fixture
def pred() -> torch.Tensor:
    torch.manual_seed(0)
    return 3 * torch.rand(2, 1, 1, GRID, 3)


@pytest.fixture
def target() -> torch.Tensor:
    torch.manual_seed(1)
    target = torch.rand(2, 1, 1, GRID, 4)
    # tc target: a mix of positive values and exact zeros
    target[..., DATA_COLUMN["tc"]] = torch.tensor([1.0, 0.0, 0.5, 0.0, 0.0, 1.0])
    target[1, ..., DATA_COLUMN["tc"]] = 0.0
    return target


def _config(
    zero_target_weights: dict | None = None,
    inner: str = "MSELoss",
    *,
    scalers: list[str] | None = None,
    wrapper_scalers: list[str] | None = None,
    **inner_kwargs: object,
) -> DictConfig:
    """Wrapper config; ``inner_kwargs`` (predicted_variables, ignore_nans, ...) go to the inner loss."""
    inner_config = {"_target_": f"anemoi.training.losses.{inner}", "scalers": scalers or [], **inner_kwargs}
    config = {
        "_target_": TARGET,
        "zero_target_weights": {"tc": W0} if zero_target_weights is None else zero_target_weights,
        "loss_fn": inner_config,
    }
    if wrapper_scalers is not None:
        config["scalers"] = wrapper_scalers
    return DictConfig(config)


def _inner_config(inner: str = "MSELoss", **kwargs: object) -> DictConfig:
    return DictConfig({"_target_": f"anemoi.training.losses.{inner}", "scalers": [], **kwargs})


def _reference_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    variables: list[str],
    weights: dict[str, float],
    inner: str = "MSELoss",
) -> torch.Tensor:
    """Point-wise error of ``variables`` (columns in this order), down-weighted at zero targets."""
    columns = []
    for name in variables:
        error = ERROR_FNS[inner](pred[..., OUTPUT_COLUMN[name]] - target[..., DATA_COLUMN[name]])
        if name in weights:
            error = error * torch.where(target[..., DATA_COLUMN[name]] > 0, 1.0, weights[name])
        columns.append(error)
    return torch.stack(columns, dim=-1)


def _reduce(error: torch.Tensor) -> torch.Tensor:
    """Sum over grid, mean over batch and variables (no scalers)."""
    return error.sum(dim=-2).mean()


def test_factory_builds_mapper_around_wrapper(data_indices: IndexCollection) -> None:
    loss = get_loss_function(_config(), data_indices=data_indices)

    # The variable filter stays outermost, as for the other wrappers
    assert isinstance(loss, LossVariableMapper)
    assert isinstance(loss.loss, TargetWeightedLoss)
    assert isinstance(loss.loss.loss, MSELoss)
    # The scalers of the wrapper are those of the inner loss
    assert loss.loss.scaler is loss.loss.loss.scaler


@pytest.mark.parametrize("inner", INNER_LOSSES)
def test_only_named_column_is_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    inner: str,
) -> None:
    loss = get_loss_function(_config(inner=inner), data_indices=data_indices)
    unweighted = get_loss_function(_inner_config(inner), data_indices=data_indices)

    per_variable = loss(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS)
    plain = unweighted(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS)

    # The mapper puts per-variable values at their model output positions
    torch.testing.assert_close(per_variable[OUTPUT_COLUMN["a"]], plain[OUTPUT_COLUMN["a"]])
    torch.testing.assert_close(per_variable[OUTPUT_COLUMN["b"]], plain[OUTPUT_COLUMN["b"]])
    assert per_variable[OUTPUT_COLUMN["tc"]] < plain[OUTPUT_COLUMN["tc"]]

    expected = _reference_error(pred, target, OUTPUT_NAMES, {"tc": W0}, inner).sum(dim=-2).mean(dim=(0, 1, 2))
    torch.testing.assert_close(per_variable, expected)
    torch.testing.assert_close(loss(_pred_view(pred), _target_view(target), **LAYOUTS), expected.mean())


def test_positive_targets_have_weight_one_and_zero_targets_the_configured_weight() -> None:
    loss = TargetWeightedLoss(loss_fn=MAELoss(), zero_target_weights={"x": 0.25})
    pred = torch.full((1, 1, 1, 4, 2), 2.0)
    target = torch.tensor([3.0, 0.0, 0.5, -1.0]).reshape(1, 1, 1, 4, 1).repeat(1, 1, 1, 1, 2)

    result = loss(_view(pred, ["y", "x"]), _view(target, ["y", "x"]), squash=False)

    # target 3.0 and 0.5 are positive, 0.0 and -1.0 are not
    error = (pred - target).abs()
    expected_x = (error[..., 1].flatten() * torch.tensor([1.0, 0.25, 1.0, 0.25])).sum()
    torch.testing.assert_close(result, torch.stack([error[..., 0].sum(), expected_x]))


@pytest.mark.parametrize("inner", INNER_LOSSES)
@pytest.mark.parametrize("zero_target_weights", [None, {}])
def test_equals_inner_loss_when_unconfigured(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    inner: str,
    zero_target_weights: dict | None,
) -> None:
    config = _config({}, inner=inner)
    if zero_target_weights is None:
        del config["zero_target_weights"]
    loss = get_loss_function(config, data_indices=data_indices)
    plain = get_loss_function(_inner_config(inner), data_indices=data_indices)

    torch.testing.assert_close(
        loss(_pred_view(pred), _target_view(target), **LAYOUTS),
        plain(_pred_view(pred), _target_view(target), **LAYOUTS),
    )
    torch.testing.assert_close(
        loss(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS),
        plain(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS),
    )


@pytest.mark.parametrize("inner", INNER_LOSSES)
def test_equals_inner_loss_when_all_targets_positive(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    inner: str,
) -> None:
    target[..., DATA_COLUMN["tc"]] = 0.3
    loss = get_loss_function(_config(inner=inner), data_indices=data_indices)
    plain = get_loss_function(_inner_config(inner), data_indices=data_indices)

    torch.testing.assert_close(
        loss(_pred_view(pred), _target_view(target), **LAYOUTS),
        plain(_pred_view(pred), _target_view(target), **LAYOUTS),
    )


def test_squash_mode_sum_matches_inner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    target[..., DATA_COLUMN["tc"]] = 0.3
    loss = get_loss_function(_config(), data_indices=data_indices)
    plain = get_loss_function(_inner_config(), data_indices=data_indices)

    torch.testing.assert_close(
        loss(_pred_view(pred), _target_view(target), squash_mode="sum", **LAYOUTS),
        plain(_pred_view(pred), _target_view(target), squash_mode="sum", **LAYOUTS),
    )


@pytest.mark.parametrize("inner", ["MSELoss", "MAELoss"])
def test_nan_targets_with_ignore_nans(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    inner: str,
) -> None:
    target = target.clone()
    nan_points = [(0, 0), (0, 1), (1, 3)]
    for batch, point in nan_points:
        target[batch, 0, 0, point, DATA_COLUMN["tc"]] = float("nan")
    loss = get_loss_function(_config(inner=inner, ignore_nans=True), data_indices=data_indices)

    result = loss(_pred_view(pred), _target_view(target), **LAYOUTS)

    assert torch.isfinite(result)
    # NaN targets and the matching predictions are zeroed: no error, whatever the weight
    pred_ref, target_ref = pred.clone(), target.clone()
    for batch, point in nan_points:
        pred_ref[batch, 0, 0, point, OUTPUT_COLUMN["tc"]] = 0.0
        target_ref[batch, 0, 0, point, DATA_COLUMN["tc"]] = 0.0
    expected = _reduce(_reference_error(pred_ref, target_ref, OUTPUT_NAMES, {"tc": W0}, inner))
    torch.testing.assert_close(result, expected)


def test_nan_targets_without_ignore_nans_propagate(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    target = target.clone()
    target[0, 0, 0, 0, DATA_COLUMN["tc"]] = float("nan")
    loss = get_loss_function(_config(), data_indices=data_indices)

    assert torch.isnan(loss(_pred_view(pred), _target_view(target), **LAYOUTS))


def test_ignore_nans_is_taken_from_the_inner_loss(data_indices: IndexCollection) -> None:
    loss = get_loss_function(_config(ignore_nans=True), data_indices=data_indices)

    assert loss.ignore_nans
    assert loss.loss.ignore_nans


@pytest.mark.parametrize("inner", ["MSELoss", "MAELoss"])
@pytest.mark.parametrize("variables", [["tc"], ["b", "tc"], ["tc", "a"], ["b", "tc", "a"], ["a", "b"]])
def test_variable_subsets_and_reordering_via_mapper(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    variables: list[str],
    inner: str,
) -> None:
    weights = {"tc": W0} if "tc" in variables else {}
    loss = get_loss_function(
        _config(weights, inner, predicted_variables=variables, target_variables=variables),
        data_indices=data_indices,
    )

    result = loss(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS)

    expected = _reference_error(pred, target, variables, weights, inner).sum(dim=-2).mean(dim=(0, 1, 2))
    torch.testing.assert_close(result[[OUTPUT_COLUMN[v] for v in variables]], expected)
    torch.testing.assert_close(loss(_pred_view(pred), _target_view(target), **LAYOUTS), expected.mean())


def test_subset_with_different_target_variable_names(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    """Weights are keyed by the predicted variable, the target is read from the paired target variable."""
    loss = get_loss_function(
        _config({"tc": W0}, predicted_variables=["tc"], target_variables=["a"]),
        data_indices=data_indices,
    )

    result = loss(_pred_view(pred), _target_view(target), **LAYOUTS)

    error = (pred[..., OUTPUT_COLUMN["tc"]] - target[..., DATA_COLUMN["a"]]) ** 2
    error = error * torch.where(target[..., DATA_COLUMN["a"]] > 0, 1.0, W0)
    torch.testing.assert_close(result, error.sum(dim=-1).mean())


@pytest.mark.parametrize("scaler_on", ["inner", "wrapper"])
def test_composes_with_node_and_variable_scalers(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    scaler_on: str,
) -> None:
    node_weights = torch.rand(GRID) + 0.5
    variable_weights = torch.tensor([2.0, 1000.0, 3.0])  # model output order: a, tc, b
    scalers = {
        "node_weights": (TensorDim.GRID.value, node_weights),
        "general_variable": (TensorDim.VARIABLE.value, variable_weights),
    }
    names = ["node_weights", "general_variable"]
    config = _config(scalers=names) if scaler_on == "inner" else _config(wrapper_scalers=names)
    loss = get_loss_function(config, scalers=scalers, data_indices=data_indices)

    result = loss(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS)

    error = _reference_error(pred, target, OUTPUT_NAMES, {"tc": W0})
    expected = (error * node_weights[:, None] * variable_weights).sum(dim=-2).mean(dim=(0, 1, 2))
    torch.testing.assert_close(result, expected)


def test_composes_with_scalers_on_variable_subset(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    variable_weights = torch.tensor([2.0, 1000.0, 3.0])
    loss = get_loss_function(
        _config(
            predicted_variables=["b", "tc"],
            target_variables=["b", "tc"],
            scalers=["general_variable", "grid"],
        ),
        scalers={
            "general_variable": (TensorDim.VARIABLE.value, variable_weights),
            "grid": (TensorDim.GRID.value, torch.ones(GRID)),
        },
        data_indices=data_indices,
    )

    result = loss(_pred_view(pred), _target_view(target), squash=False, **LAYOUTS)

    error = _reference_error(pred, target, ["b", "tc"], {"tc": W0}) * torch.tensor([3.0, 1000.0])
    torch.testing.assert_close(
        result[[OUTPUT_COLUMN["b"], OUTPUT_COLUMN["tc"]]],
        error.sum(dim=-2).mean(dim=(0, 1, 2)),
    )


def test_gradients_are_weighted(pred: torch.Tensor, target: torch.Tensor, data_indices: IndexCollection) -> None:
    pred = pred.clone().requires_grad_(True)
    loss = get_loss_function(_config(), data_indices=data_indices)

    loss(_pred_view(pred), _target_view(target), **LAYOUTS).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    tc_target = target[..., DATA_COLUMN["tc"]]
    tc_grad = pred.grad[..., OUTPUT_COLUMN["tc"]]
    plain_grad = 2 * (pred[..., OUTPUT_COLUMN["tc"]] - tc_target).detach() / (3 * 2)  # mean over 3 variables, 2 batch
    torch.testing.assert_close(tc_grad, plain_grad * torch.where(tc_target > 0, 1.0, W0))


def test_works_inside_combined_loss(pred: torch.Tensor, target: torch.Tensor, data_indices: IndexCollection) -> None:
    config = DictConfig(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [
                _config(predicted_variables=["tc"], target_variables=["tc"]),
                {"_target_": "anemoi.training.losses.MSELoss", "predicted_variables": ["a"], "scalers": []},
            ],
            "loss_weights": [1.0, 0.5],
        },
    )
    loss = get_loss_function(config, scalers={}, data_indices=data_indices)

    result = loss(_pred_view(pred), _target_view(target), **LAYOUTS)

    tc_error = _reference_error(pred, target, ["tc"], {"tc": W0}).sum(dim=-2).mean()
    a_error = _reference_error(pred, target, ["a"], {}).sum(dim=-2).mean()
    torch.testing.assert_close(result, tc_error + 0.5 * a_error)


def test_two_wrappers_in_one_combined_loss_use_their_own_columns(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    """Variable names come from the source each mapper selects, so differently subset children do not interfere."""
    config = DictConfig(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [
                _config({"b": 0.1}, predicted_variables=["b", "tc"], target_variables=["b", "tc"]),
                _config({"tc": 0.5}, "MAELoss", predicted_variables=["tc", "a"], target_variables=["tc", "a"]),
            ],
        },
    )
    loss = get_loss_function(config, scalers={}, data_indices=data_indices)

    result = loss(_pred_view(pred), _target_view(target), **LAYOUTS)

    first = _reference_error(pred, target, ["b", "tc"], {"b": 0.1}).sum(dim=-2).mean()
    second = _reference_error(pred, target, ["tc", "a"], {"tc": 0.5}, "MAELoss").sum(dim=-2).mean()
    torch.testing.assert_close(result, first + second)


def test_combined_loss_without_a_mapper_uses_the_source_variables(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Without a LossVariableMapper the names are those of the sources themselves."""
    pred, target = pred[..., :2], target[..., [DATA_COLUMN["a"], DATA_COLUMN["tc"]]]
    wrapper = TargetWeightedLoss(loss_fn=MSELoss(), zero_target_weights={"tc": W0})
    combined = CombinedLoss(wrapper, MSELoss(), loss_weights=(1.0, 0.5))
    pred_view, target_view = _view(pred, ["a", "tc"]), _view(target, ["a", "tc"])

    result = combined(pred_view, target_view, squash=False)

    tc_error = (pred[..., 1] - target[..., 1]) ** 2 * torch.where(target[..., 1] > 0, 1.0, W0)
    a_error = (pred[..., 0] - target[..., 0]) ** 2
    plain = (pred - target) ** 2
    expected_first = torch.stack([a_error.sum(dim=-1).mean(), tc_error.sum(dim=-1).mean()])
    expected = expected_first + 0.5 * plain.sum(dim=-2).mean(dim=(0, 1, 2))
    torch.testing.assert_close(result, expected)


def test_caller_weights_are_multiplied_with_the_target_weights(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    """A ``weights`` argument (e.g. the noise-level weights of a diffusion loss) is kept."""
    sample_weights = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1, 1)
    loss = get_loss_function(_config(), data_indices=data_indices)

    result = loss(_pred_view(pred), _target_view(target), squash=False, weights=sample_weights, **LAYOUTS)

    error = _reference_error(pred, target, OUTPUT_NAMES, {"tc": W0}) * sample_weights
    torch.testing.assert_close(result, error.sum(dim=-2).mean(dim=(0, 1, 2)))


def test_weights_are_applied_in_the_dtype_of_the_inner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_config(), data_indices=data_indices)

    expected = loss(_pred_view(pred), _target_view(target), **LAYOUTS)
    reduced = loss(_pred_view(pred.bfloat16()), _target_view(target.bfloat16()), **LAYOUTS)

    assert reduced.dtype == torch.float32
    torch.testing.assert_close(reduced, expected, rtol=5e-2, atol=5e-2)


def test_tabular_sources() -> None:
    names = ["a", "tc"]
    pred = [torch.tensor([[1.0, 2.0], [2.0, 3.0]]), torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 0.5]])]
    target = [torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 2.0]])]
    loss = TargetWeightedLoss(loss_fn=MSELoss(), zero_target_weights={"tc": W0})

    result = loss(_tabular_view(pred, names), _tabular_view(target, names), squash=False)

    expected = torch.zeros(2)
    for p, t in zip(pred, target, strict=True):
        error = (p - t) ** 2 * torch.stack([torch.ones(len(p)), torch.where(t[:, 1] > 0, 1.0, W0)], dim=-1)
        expected += error.mean(dim=0)
    torch.testing.assert_close(result, expected / 2)


def test_tabular_sources_with_ignore_nans() -> None:
    names = ["a", "tc"]
    pred = [torch.tensor([[1.0, 2.0], [2.0, 3.0], [1.0, 1.0]])]
    target = [torch.tensor([[0.0, 1.0], [1.0, float("nan")], [1.0, 0.0]])]
    loss = TargetWeightedLoss(loss_fn=MSELoss(ignore_nans=True), zero_target_weights={"tc": W0}, ignore_nans=True)

    result = loss(_tabular_view(pred, names), _tabular_view(target, names), squash=False)

    # tc is a mean over the two valid observations: (2 - 1)^2 and W0 * (1 - 0)^2
    torch.testing.assert_close(result, torch.tensor([(1.0 + 1.0 + 0.0) / 3, (1.0 + W0) / 2]))


def test_unknown_variable_raises_when_called(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_config({"missing": W0}), data_indices=data_indices)

    with pytest.raises(ValueError, match=r"zero_target_weights variables \['missing'\]") as excinfo:
        loss(_pred_view(pred), _target_view(target), **LAYOUTS)
    assert "tc" in str(excinfo.value)


def test_variable_not_in_subset_raises_when_called(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(
        _config({"tc": W0}, predicted_variables=["a", "b"], target_variables=["a", "b"]),
        data_indices=data_indices,
    )

    with pytest.raises(ValueError, match=r"zero_target_weights variables \['tc'\]"):
        loss(_pred_view(pred), _target_view(target), **LAYOUTS)


def test_forcing_variable_is_not_a_loss_column(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_config({"f": W0}), data_indices=data_indices)

    with pytest.raises(ValueError, match=r"zero_target_weights variables \['f'\]"):
        loss(_pred_view(pred), _target_view(target), **LAYOUTS)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_weight_values_raise(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        TargetWeightedLoss(loss_fn=MSELoss(), zero_target_weights={"tc": weight})


@pytest.mark.parametrize("weight", ["0.1", None, True, [0.1]])
def test_invalid_weight_types_raise(weight: object) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        TargetWeightedLoss(loss_fn=MSELoss(), zero_target_weights={"tc": weight})  # type: ignore[dict-item]


def test_invalid_weight_raises_through_factory(data_indices: IndexCollection) -> None:
    with pytest.raises(InstantiationException, match="finite and strictly positive"):
        get_loss_function(_config({"tc": -0.1}), data_indices=data_indices)


@pytest.mark.parametrize(
    "inner_config",
    [
        {"_target_": "anemoi.training.losses.RMSELoss"},
        {"_target_": "anemoi.training.losses.EnergyScoreLoss"},
        {"_target_": "anemoi.training.losses.spectral.LogFFT2Distance", "x_dim": 4, "y_dim": 4},
    ],
    ids=["rmse", "energy_score", "spectral"],
)
def test_inner_loss_without_pointwise_error_is_rejected(
    data_indices: IndexCollection,
    inner_config: dict,
) -> None:
    config = DictConfig(
        {
            "_target_": TARGET,
            "zero_target_weights": {"tc": W0},
            "loss_fn": {**inner_config, "scalers": []},
        },
    )

    # hydra wraps the error raised by the constructor
    with pytest.raises(InstantiationException, match="needs a point-wise inner loss"):
        get_loss_function(config, data_indices=data_indices)


def test_nested_wrapper_is_rejected() -> None:
    inner = TargetWeightedLoss(loss_fn=MSELoss())

    with pytest.raises(TypeError, match="needs a point-wise inner loss"):
        TargetWeightedLoss(loss_fn=inner, zero_target_weights={"tc": W0})


def test_non_loss_inner_is_rejected() -> None:
    with pytest.raises(TypeError, match="Expected BaseLoss"):
        TargetWeightedLoss(loss_fn=object())  # type: ignore[arg-type]


def test_is_a_loss_wrapper() -> None:
    loss = TargetWeightedLoss(loss_fn=MSELoss(), zero_target_weights={"tc": W0})

    assert isinstance(loss, BaseLoss)
    assert list(loss.iter_leaf_losses()) == [loss.loss]
    # Loss-weight consumers (e.g. EDM diffusion) look at the leaves
    assert all(leaf.applies_weights for leaf in loss.iter_leaf_losses())


def test_schema_accepts_valid_and_rejects_invalid_config() -> None:
    adapter = TypeAdapter(LossSchemas)
    valid = {
        "_target_": TARGET,
        "scalers": ["variable"],
        "zero_target_weights": {"tc": 1e-3},
        "loss_fn": {"_target_": "anemoi.training.losses.MSELoss", "ignore_nans": True},
    }
    parsed = adapter.validate_python(valid)
    assert parsed.zero_target_weights == {"tc": 1e-3}

    for bad_weight in (0.0, -1.0, float("nan")):
        with pytest.raises(ValidationError):
            adapter.validate_python({**valid, "zero_target_weights": {"tc": bad_weight}})
    with pytest.raises(ValidationError):
        adapter.validate_python({**valid, "loss_fn": {"_target_": "anemoi.training.losses.Unknown"}})


# Ensemble CRPS as inner loss. The weight comes from the target, which has no ensemble members, and multiplies
# the point-wise CRPS of the whole ensemble.

ENS = 4


@pytest.fixture
def ens_pred() -> torch.Tensor:
    torch.manual_seed(2)
    return 2 * torch.rand(2, 1, ENS, GRID, 3)


def _crps_reference(
    ens_pred: torch.Tensor,
    target: torch.Tensor,
    variables: list[str],
    weights: dict[str, float],
    **crps_kwargs: object,
) -> torch.Tensor:
    """Point-wise CRPS of ``variables`` (columns in this order), down-weighted at zero targets."""
    loss = CRPS(**crps_kwargs)
    columns = []
    for name in variables:
        # ensemble last, as the kernel expects: (batch, time, grid, ensemble)
        crps = loss._kernel_crps(
            ens_pred[..., OUTPUT_COLUMN[name]].movedim(2, -1),
            target[..., DATA_COLUMN[name]].movedim(2, -1),
        )
        if name in weights:
            crps = crps * torch.where(target[:, :, 0, :, DATA_COLUMN[name]] > 0, 1.0, weights[name])
        columns.append(crps)
    return torch.stack(columns, dim=-1)


def test_crps_without_weights_is_unchanged(ens_pred: torch.Tensor, target: torch.Tensor) -> None:
    target = target[:, :, :1][..., [DATA_COLUMN[name] for name in OUTPUT_NAMES]]
    loss = CRPS(alpha=0.7)
    loss.add_scaler(TensorDim.GRID.value, torch.rand(GRID) + 0.5, name="grid")

    plain = loss(_pred_view(ens_pred), _pred_view(target), squash=False)

    assert CRPS.applies_weights
    # weights=None and all-ones weights leave the loss as it was
    torch.testing.assert_close(
        loss(_pred_view(ens_pred), _pred_view(target), squash=False, weights=None),
        plain,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        loss(_pred_view(ens_pred), _pred_view(target), squash=False, weights=torch.ones_like(target)),
        plain,
        rtol=0,
        atol=0,
    )


def test_crps_applies_weights_pointwise_before_the_reduction(ens_pred: torch.Tensor, target: torch.Tensor) -> None:
    target = target[:, :, :1][..., [DATA_COLUMN[name] for name in OUTPUT_NAMES]]
    weights = torch.rand(2, 1, 1, GRID, 3)
    loss = CRPS(alpha=0.7)

    weighted = loss(_pred_view(ens_pred), _pred_view(target), squash=False, weights=weights)

    crps = loss._kernel_crps(ens_pred.movedim(2, -1), target.movedim(2, -1))  # (batch, time, grid, variables)
    expected = (crps * weights[:, :, 0]).sum(dim=-2).mean(dim=(0, 1))
    torch.testing.assert_close(weighted, expected)


@pytest.mark.parametrize("backend", ["stable", "naive"])
def test_target_weighted_crps_by_hand(backend: str) -> None:
    """Two members 1 and 2: the fair CRPS is 0.5 (|1 - y| + |2 - y|) - 0.5 = 1 for the targets 0 and 3."""
    pred = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1).repeat(1, 1, 1, 3, 1)
    target = torch.tensor([0.0, 3.0, 1.5]).reshape(1, 1, 1, 3, 1)
    loss = TargetWeightedLoss(loss_fn=CRPS(alpha=1.0, backend=backend), zero_target_weights={"tc": W0})

    result = loss(_view(pred, ["tc"]), _view(target, ["tc"]), squash=False)

    # targets 0 (weight W0), 3 (weight 1) and 1.5 (CRPS 0, weight 1)
    torch.testing.assert_close(result, torch.tensor([W0 + 1.0]))


def test_target_weighted_crps_excludes_nan_targets() -> None:
    pred = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1).repeat(1, 1, 1, 3, 1)
    target = torch.tensor([0.0, float("nan"), 3.0]).reshape(1, 1, 1, 3, 1)
    loss = TargetWeightedLoss(
        loss_fn=CRPS(alpha=1.0, ignore_nans=True),
        zero_target_weights={"tc": W0},
        ignore_nans=True,
    )

    result = loss(_view(pred, ["tc"]), _view(target, ["tc"]), squash=False)

    torch.testing.assert_close(result, torch.tensor([W0 + 1.0]))
    assert torch.isnan(
        TargetWeightedLoss(loss_fn=CRPS(alpha=1.0), zero_target_weights={"tc": W0})(
            _view(pred, ["tc"]),
            _view(target, ["tc"]),
        ),
    )


@pytest.mark.parametrize("backend", ["stable", "naive"])
def test_crps_point_wise_values_are_weighted(
    ens_pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    backend: str,
) -> None:
    config = _config(inner="CRPS", alpha=0.7, backend=backend)
    loss = get_loss_function(config, data_indices=data_indices)
    plain = get_loss_function(_inner_config("CRPS", alpha=0.7, backend=backend), data_indices=data_indices)

    target = target[:, :, :1]  # singleton ensemble dimension of the target
    per_variable = loss(_pred_view(ens_pred), _target_view(target), squash=False, **LAYOUTS)
    plain_per_variable = plain(_pred_view(ens_pred), _target_view(target), squash=False, **LAYOUTS)

    torch.testing.assert_close(per_variable[OUTPUT_COLUMN["a"]], plain_per_variable[OUTPUT_COLUMN["a"]])
    torch.testing.assert_close(per_variable[OUTPUT_COLUMN["b"]], plain_per_variable[OUTPUT_COLUMN["b"]])
    assert per_variable[OUTPUT_COLUMN["tc"]] < plain_per_variable[OUTPUT_COLUMN["tc"]]

    reference = _crps_reference(ens_pred, target, OUTPUT_NAMES, {"tc": W0}, alpha=0.7, backend=backend)
    torch.testing.assert_close(per_variable, reference.sum(dim=-2).mean(dim=(0, 1)))
    torch.testing.assert_close(loss(_pred_view(ens_pred), _target_view(target), **LAYOUTS), per_variable.mean())


@pytest.mark.parametrize("squash", [True, False])
def test_crps_wrapper_equals_crps_when_unconfigured(
    ens_pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    squash: bool,
) -> None:
    loss = get_loss_function(_config({}, "CRPS"), data_indices=data_indices)
    plain = get_loss_function(_inner_config("CRPS"), data_indices=data_indices)
    target = target[:, :, :1]

    torch.testing.assert_close(
        loss(_pred_view(ens_pred), _target_view(target), squash=squash, **LAYOUTS),
        plain(_pred_view(ens_pred), _target_view(target), squash=squash, **LAYOUTS),
    )


def test_crps_wrapper_with_nans_scalers_and_subset(
    ens_pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    target = target[:, :, :1].clone()
    target[0, 0, 0, 1, DATA_COLUMN["tc"]] = float("nan")
    node_weights = torch.rand(GRID) + 0.5
    loss = get_loss_function(
        _config(
            inner="CRPS",
            ignore_nans=True,
            predicted_variables=["tc", "b"],
            target_variables=["tc", "b"],
            scalers=["node_weights"],
        ),
        scalers={"node_weights": (TensorDim.GRID.value, node_weights)},
        data_indices=data_indices,
    )

    result = loss(_pred_view(ens_pred), _target_view(target), squash=False, **LAYOUTS)

    assert torch.isfinite(result).all()
    pred_ref, target_ref = ens_pred.clone(), target.clone()
    pred_ref[0, :, :, 1, OUTPUT_COLUMN["tc"]] = 0.0
    target_ref[0, 0, 0, 1, DATA_COLUMN["tc"]] = 0.0
    reference = _crps_reference(pred_ref, target_ref, ["tc", "b"], {"tc": W0}) * node_weights[:, None]
    expected = reference.sum(dim=-2).mean(dim=(0, 1))
    torch.testing.assert_close(result[[OUTPUT_COLUMN["tc"], OUTPUT_COLUMN["b"]]], expected)


def test_crps_wrapper_works_inside_combined_loss(
    ens_pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
) -> None:
    config = DictConfig(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [
                _config(inner="CRPS", predicted_variables=["tc"], target_variables=["tc"]),
                {"_target_": "anemoi.training.losses.CRPS", "predicted_variables": ["a"], "scalers": []},
            ],
            "loss_weights": [1.0, 0.5],
        },
    )
    loss = get_loss_function(config, scalers={}, data_indices=data_indices)
    target = target[:, :, :1]

    result = loss(_pred_view(ens_pred), _target_view(target), **LAYOUTS)

    tc = _crps_reference(ens_pred, target, ["tc"], {"tc": W0}).sum(dim=-2).mean()
    a = _crps_reference(ens_pred, target, ["a"], {}).sum(dim=-2).mean()
    torch.testing.assert_close(result, tc + 0.5 * a)


def test_crps_wrapper_gradients(ens_pred: torch.Tensor, target: torch.Tensor, data_indices: IndexCollection) -> None:
    ens_pred = ens_pred.clone().requires_grad_(True)
    loss = get_loss_function(_config(inner="CRPS"), data_indices=data_indices)

    loss(_pred_view(ens_pred), _target_view(target[:, :, :1]), **LAYOUTS).backward()

    assert ens_pred.grad is not None
    assert torch.isfinite(ens_pred.grad).all()
    assert ens_pred.grad[..., OUTPUT_COLUMN["tc"]].abs().sum() > 0


# Sharding and compilation


def _sharded_loss(
    data_indices: IndexCollection,
    grid_scaler: torch.Tensor,
    inner: str = "MSELoss",
) -> LossVariableMapper:
    return get_loss_function(
        _config(inner=inner, scalers=["node_weights", "general_variable"]),
        scalers={
            "node_weights": (TensorDim.GRID.value, grid_scaler),
            "general_variable": (TensorDim.VARIABLE.value, torch.tensor([2.0, 1000.0, 3.0])),
        },
        data_indices=data_indices,
    )


@pytest.mark.parametrize("inner", ["MSELoss", "CRPS"])
@pytest.mark.parametrize("shard_bounds", [[0, 3, 6], [0, 1, 4, 6], [0, 2, 2, 6]])
@pytest.mark.parametrize("squash", [True, False])
def test_sharded_losses_sum_to_the_full_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    mocker: MockerFixture,
    shard_bounds: list[int],
    squash: bool,
    inner: str,
) -> None:
    """Each rank scores its slice of the grid (the group reduction sums them): the shards add up to the full loss."""
    # the group reduction of the loss is a sum over ranks; here the ranks are summed by hand
    mocker.patch("anemoi.training.losses.base.reduce_tensor", side_effect=lambda value, _group: value)
    grid_scaler = torch.rand(GRID) + 0.5
    loss = _sharded_loss(data_indices, grid_scaler, inner)
    if inner == "CRPS":  # ensemble prediction, target without members
        pred = pred.expand(-1, -1, ENS, -1, -1) + torch.rand(2, 1, ENS, GRID, 3)
        target = target[:, :, :1]
    full = loss(_pred_view(pred), _target_view(target), squash=squash, **LAYOUTS)

    shards = []
    for start, stop in itertools.pairwise(shard_bounds):
        shards.append(
            loss(
                _pred_view(pred[..., start:stop, :]),
                _target_view(target[..., start:stop, :]),
                squash=squash,
                grid_shard_slice=slice(start, stop),
                group=object(),
                **LAYOUTS,
            ),
        )

    torch.testing.assert_close(torch.stack(shards).sum(dim=0), full)


def test_supports_sharding_matches_the_inner_loss(data_indices: IndexCollection) -> None:
    loss = get_loss_function(_config(), data_indices=data_indices)

    assert loss.supports_sharding is True
    assert loss.loss.supports_sharding is True
    assert TargetWeightedLoss(loss_fn=MSELoss()).supports_sharding is True


def test_supports_sharding_false_is_passed_through() -> None:
    inner = MSELoss()
    inner.supports_sharding = False

    wrapper = TargetWeightedLoss(loss_fn=inner, zero_target_weights={"tc": W0})

    assert wrapper.supports_sharding is False
    mapper = LossVariableMapper(loss=wrapper, predicted_variables=["tc"], target_variables=["tc"])
    assert mapper.supports_sharding is False


def _dynamo_supported() -> bool:
    try:
        import torch._dynamo

        return bool(torch._dynamo.is_dynamo_supported())
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(not _dynamo_supported(), reason="torch.compile is not supported here")
@pytest.mark.parametrize("squash", [True, False])
def test_compiles_without_graph_breaks(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_indices: IndexCollection,
    squash: bool,
) -> None:
    import torch._dynamo

    torch._dynamo.reset()
    loss = _sharded_loss(data_indices, torch.rand(GRID) + 0.5)
    # Compile the wrapper (a mapper selects columns with data dependent Python lists)
    wrapper = loss.loss
    target = target[..., [DATA_COLUMN["a"], DATA_COLUMN["tc"], DATA_COLUMN["b"]]]
    eager_pred = pred.clone().requires_grad_(True)
    compiled_pred = pred.clone().requires_grad_(True)

    compiled = torch.compile(wrapper, backend="eager", fullgraph=True)
    try:
        expected = wrapper(_pred_view(eager_pred), _pred_view(target), squash=squash)
        result = compiled(_pred_view(compiled_pred), _pred_view(target), squash=squash)
    finally:
        torch._dynamo.reset()

    torch.testing.assert_close(result, expected)
    if squash:
        result.backward()
        expected.backward()
        torch.testing.assert_close(compiled_pred.grad, eager_pred.grad)
