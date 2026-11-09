# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import combinations
from typing import get_args

import pytest
import torch
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pytest_mock import MockerFixture
from torch.autograd import gradcheck

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CRPS, EnergyScoreLoss, get_loss_function
from anemoi.training.losses.aggregate import TimeAggregateLossWrapper
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.schemas.training import CombinedLossSchema, LossSchemas
from anemoi.training.utils.enums import EnergyScoreDimension, TensorDim
from anemoi.training.utils.index_space import IndexSpace

Norm = tuple[EnergyScoreDimension, ...]

TENSOR_DIMENSIONS: dict[EnergyScoreDimension, int] = {
    "time": TensorDim.TIME,
    "grid": TensorDim.GRID,
    "variables": TensorDim.VARIABLE,
}
NEGATIVE_DIMENSIONS: dict[EnergyScoreDimension, int] = {"time": -4, "grid": -2, "variables": -1}
ALL_NORMS: list[Norm] = [
    norm
    for size in range(1, len(get_args(EnergyScoreDimension)) + 1)
    for norm in combinations(get_args(EnergyScoreDimension), size)
]
TIME_NORMS = [norm for norm in ALL_NORMS if "time" in norm]
VARIABLE_JOINT_NORMS = [norm for norm in ALL_NORMS if "variables" in norm]
NORM_IDS = ["+".join(norm) for norm in ALL_NORMS]


@pytest.fixture
def score_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor(
        [
            [
                [
                    [[0.0, 1.0], [1.0, 2.0], [2.0, -1.0]],
                    [[1.0, 0.0], [2.0, 3.0], [3.0, 1.0]],
                    [[2.0, 2.0], [1.0, 1.0], [4.0, 0.0]],
                ],
                [
                    [[1.0, 0.5], [0.0, 2.0], [3.0, -2.0]],
                    [[2.0, -0.5], [1.0, 4.0], [2.0, 0.0]],
                    [[0.0, 1.5], [2.0, 1.0], [5.0, 1.0]],
                ],
            ],
            [
                [
                    [[-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [3.0, 2.0], [1.0, -1.0]],
                    [[2.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
                ],
                [
                    [[0.0, -1.0], [2.0, 1.0], [1.0, 3.0]],
                    [[1.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
                    [[3.0, 1.0], [1.0, 0.0], [2.0, 4.0]],
                ],
            ],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [
                [[[1.0, 0.5], [0.0, 2.0], [2.0, 0.0]]],
                [[[0.5, 1.0], [1.0, 2.5], [3.0, -0.5]]],
            ],
            [
                [[[0.0, 1.0], [2.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 0.0], [2.5, 1.5], [1.0, 2.5]]],
            ],
        ],
        dtype=torch.float64,
    )
    return pred, target


def _energy_score_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    fair: bool,
    norm_over: Norm,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score with size one along the ensemble and every norm dimension."""
    if weights is None:
        weights = torch.ones_like(target)
    weighted_pred = pred * torch.sqrt(weights)
    weighted_target = target * torch.sqrt(weights)
    norm_dimensions = tuple(TENSOR_DIMENSIONS[dimension] for dimension in norm_over)

    def norm(values: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(values, dim=norm_dimensions, keepdim=True)

    observation_term = norm(weighted_pred - weighted_target).mean(dim=2, keepdim=True)
    ensemble_size = pred.shape[2]
    pair_distance_sum = torch.zeros_like(observation_term)
    for member in range(ensemble_size - 1):
        pair_distance_sum += norm(
            weighted_pred[:, :, member].unsqueeze(2) - weighted_pred[:, :, member + 1 :],
        ).sum(dim=2, keepdim=True)

    pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if fair else 1.0 / (ensemble_size**2)
    return observation_term - pair_coefficient * pair_distance_sum


def _reduce_per_variable(score: torch.Tensor, num_variables: int) -> torch.Tensor:
    """Repeat a joint score for every variable, sum over steps and nodes, average over the batch."""
    score = score.expand(*score.shape[:-1], num_variables)
    return score.sum(dim=(1, 3)).mean(dim=(0, 1))


def _two_step_inputs(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a second output step that differs from the first."""
    return torch.cat((pred, 0.5 - 0.75 * pred), dim=1), torch.cat((target, 0.25 + 0.5 * target), dim=1)


@pytest.mark.parametrize("fair", [True, False])
@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_matches_reference(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    fair: bool,
    norm_over: Norm,
) -> None:
    pred, target = score_inputs
    loss = EnergyScoreLoss(fair=fair, norm_over=norm_over)

    actual = loss(pred, target, squash=False)
    expected = _reduce_per_variable(
        _energy_score_reference(pred, target, fair=fair, norm_over=norm_over),
        pred.shape[-1],
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_applies_weights_inside_and_outside_the_norm(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: Norm,
) -> None:
    pred, target = score_inputs
    weights = {
        "time": torch.tensor([0.4, 0.6], dtype=pred.dtype).reshape(1, 2, 1, 1, 1),
        "grid": torch.tensor([0.2, 0.3, 0.5], dtype=pred.dtype).reshape(1, 1, 1, 3, 1),
        "variables": torch.tensor([0.25, 0.75], dtype=pred.dtype).reshape(1, 1, 1, 1, 2),
    }
    loss = EnergyScoreLoss(norm_over=norm_over)
    for dimension, weight in weights.items():
        loss.add_scaler(TENSOR_DIMENSIONS[dimension], weight.flatten(), name=dimension)

    norm_weights = torch.ones_like(target)
    outer_weights = torch.ones((), dtype=target.dtype)
    for dimension, weight in weights.items():
        if dimension in norm_over:
            norm_weights = norm_weights * weight
        else:
            outer_weights = outer_weights * weight

    actual = loss(pred, target, squash=False)
    score = _energy_score_reference(pred, target, fair=True, norm_over=norm_over, weights=norm_weights)
    expected = _reduce_per_variable(score * outer_weights, pred.shape[-1])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("norm_over", TIME_NORMS, ids=["+".join(norm) for norm in TIME_NORMS])
def test_time_energy_score_needs_two_output_steps(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: Norm,
) -> None:
    pred, target = score_inputs
    loss = EnergyScoreLoss(norm_over=norm_over)

    with pytest.raises(ValueError, match="at least two output steps"):
        loss(pred[:, :1], target[:, :1])


def test_time_aggregate_wrapper_rejects_time_energy_score() -> None:
    with pytest.raises(ValueError, match="spans several output steps"):
        TimeAggregateLossWrapper(["mean"], EnergyScoreLoss(norm_over=["time"]))


@pytest.mark.parametrize("num_variables", [1, 2])
def test_grid_energy_score_follows_standard_reduction(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    num_variables: int,
) -> None:
    pred, target = score_inputs
    pred = pred[..., :num_variables]
    target = target[..., :num_variables]
    loss = EnergyScoreLoss()

    scalar = loss(pred, target)
    per_variable = loss(pred, target, squash=False)
    summed = loss(pred, target, squash_mode="sum")

    assert scalar.shape == ()
    assert per_variable.shape == (num_variables,)
    torch.testing.assert_close(scalar, per_variable.mean())
    torch.testing.assert_close(summed, per_variable.sum())


@pytest.mark.parametrize("norm_over", VARIABLE_JOINT_NORMS, ids=["+".join(norm) for norm in VARIABLE_JOINT_NORMS])
def test_variable_joint_energy_score_repeats_diagnostic_value(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: Norm,
) -> None:
    pred, target = score_inputs
    loss = EnergyScoreLoss(norm_over=norm_over)

    scalar = loss(pred, target)
    per_variable = loss(pred, target, squash=False)

    assert scalar.shape == ()
    assert per_variable.shape == (pred.shape[-1],)
    torch.testing.assert_close(per_variable, scalar.expand_as(per_variable))

    with pytest.raises(ValueError, match="not defined"):
        loss(pred, target, squash_mode="sum")


NORM_DIMENSION_CASES = [(norm, dimension) for norm in ALL_NORMS for dimension in norm]
NORM_DIMENSION_IDS = [f"{'+'.join(norm)}-{dimension}" for norm, dimension in NORM_DIMENSION_CASES]
DIMENSION_WEIGHTS: dict[EnergyScoreDimension, list[float]] = {
    "time": [0.4, 0.6],
    "grid": [0.2, 0.3, 0.5],
    "variables": [0.25, 0.75],
}


@pytest.mark.parametrize(("norm_over", "dimension"), NORM_DIMENSION_CASES, ids=NORM_DIMENSION_IDS)
def test_energy_score_resolves_negative_norm_dimensions(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: Norm,
    dimension: EnergyScoreDimension,
) -> None:
    pred, target = score_inputs
    weights = torch.tensor(DIMENSION_WEIGHTS[dimension], dtype=torch.float64)
    positive_loss = EnergyScoreLoss(norm_over=norm_over)
    negative_loss = EnergyScoreLoss(norm_over=norm_over)
    positive_loss.add_scaler(TENSOR_DIMENSIONS[dimension], weights)
    negative_loss.add_scaler(NEGATIVE_DIMENSIONS[dimension], weights)

    expected = positive_loss(pred, target, squash=False)
    actual = negative_loss(pred, target, squash=False)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("norm_over", "dimension"), NORM_DIMENSION_CASES, ids=NORM_DIMENSION_IDS)
def test_energy_score_validates_negative_norm_dimensions(
    norm_over: Norm,
    dimension: EnergyScoreDimension,
) -> None:
    weights = torch.tensor(DIMENSION_WEIGHTS[dimension])
    weights[0] = -weights[0]
    loss = EnergyScoreLoss(norm_over=norm_over)

    with pytest.raises(ValueError, match="non-negative"):
        loss.add_scaler(NEGATIVE_DIMENSIONS[dimension], weights)

    loss.add_scaler(NEGATIVE_DIMENSIONS[dimension], torch.ones_like(weights), name="weights")
    with pytest.raises(ValueError, match="non-negative"):
        loss.update_scaler("weights", weights)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0, torch.nan, 1.0]),
        torch.tensor([1.0, torch.inf, 1.0]),
        torch.tensor([1.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j]),
    ],
)
def test_energy_score_requires_finite_real_norm_weights(weights: torch.Tensor) -> None:
    loss = EnergyScoreLoss()

    with pytest.raises(ValueError, match="finite real"):
        loss.add_scaler(TensorDim.GRID, weights)


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_has_finite_gradients_at_zero(norm_over: Norm) -> None:
    pred = torch.zeros(1, 2, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 2, 1, 3, 2, dtype=torch.float64)
    loss = EnergyScoreLoss(norm_over=norm_over)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_has_finite_gradients_for_large_values(norm_over: Norm) -> None:
    pred, target = _two_step_inputs(
        torch.tensor(
            [[[[[1.0e20, -1.0e20], [-1.0e20, 0.5e20]], [[1.5e20, 0.25e20], [-0.5e20, 1.0e20]]]]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[[[[0.25e20, -0.25e20], [0.0, 0.75e20]]]]],
            dtype=torch.float32,
        ),
    )
    pred.requires_grad_()
    loss = EnergyScoreLoss(norm_over=norm_over)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_ignores_missing_features_with_finite_gradients(norm_over: Norm) -> None:
    pred, target = _two_step_inputs(
        torch.tensor(
            [[[[[0.0, 1.0], [1.0, 2.0]], [[2.0, 0.0], [3.0, 1.0]], [[1.0, 3.0], [2.0, 4.0]]]]],
            dtype=torch.float64,
        ),
        torch.tensor([[[[[1.0, 0.5], [2.0, 2.5]]]]], dtype=torch.float64),
    )
    pred[0, 0, 0, 0, 1] = torch.nan
    pred.requires_grad_()
    loss = EnergyScoreLoss(norm_over=norm_over, ignore_nans=True)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_gradcheck(norm_over: Norm) -> None:
    pred, target = _two_step_inputs(
        torch.tensor(
            [[[[[0.1, 1.3], [0.5, 2.1]], [[2.2, -0.7], [1.4, 0.2]], [[1.1, 0.4], [-0.3, 1.7]]]]],
            dtype=torch.float64,
        ),
        torch.tensor([[[[[0.7, -0.2], [1.8, 0.9]]]]], dtype=torch.float64),
    )
    pred.requires_grad_()
    loss = EnergyScoreLoss(norm_over=norm_over)

    assert gradcheck(lambda value: loss(value, target), (pred,), eps=1e-6, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("norm_over", VARIABLE_JOINT_NORMS, ids=["+".join(norm) for norm in VARIABLE_JOINT_NORMS])
def test_filtered_variable_joint_energy_score_maps_repeated_value(norm_over: Norm) -> None:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"a": 0, "b": 1, "c": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.EnergyScoreLoss",
                "scalers": [],
                "norm_over": list(norm_over),
                "predicted_variables": ["a", "c"],
                "target_variables": ["a", "c"],
            },
        ),
        data_indices=data_indices,
    )
    pred, target = _two_step_inputs(
        torch.tensor(
            [[[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], [[2.0, 0.0, 4.0], [3.0, 1.0, 2.0]]]]],
            dtype=torch.float64,
        ),
        torch.tensor([[[[[1.0, 0.5, 2.5], [2.0, 2.5, 1.0]]]]], dtype=torch.float64),
    )
    loss_kwargs = {
        "pred_layout": IndexSpace.MODEL_OUTPUT,
        "target_layout": IndexSpace.DATA_OUTPUT,
    }

    assert isinstance(loss, LossVariableMapper)
    assert loss.scores_across_time is ("time" in norm_over)
    scalar = loss(pred, target, **loss_kwargs)
    per_variable = loss(pred, target, squash=False, **loss_kwargs)

    torch.testing.assert_close(per_variable, torch.stack((scalar, scalar.new_zeros(()), scalar)))


def test_energy_score_schemas_accept_direct_nested_and_combined_configs() -> None:
    direct = TypeAdapter(LossSchemas).validate_python(
        {
            "_target_": "anemoi.training.losses.EnergyScoreLoss",
            "scalers": [],
            "norm_over": ["time", "variables"],
        },
    )
    nested = TypeAdapter(LossSchemas).validate_python(
        {
            "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
            "weights": [1.0],
            "per_scale_loss": {
                "_target_": "anemoi.training.losses.EnergyScoreLoss",
                "scalers": [],
            },
        },
    )
    combined = CombinedLossSchema.model_validate(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "scalers": [],
            "losses": [
                {
                    "_target_": "anemoi.training.losses.EnergyScoreLoss",
                    "scalers": [],
                },
            ],
        },
    )

    assert direct.norm_over == ["time", "variables"]
    assert nested.per_scale_loss.norm_over == ["grid"]
    assert combined.losses[0].target_ == "anemoi.training.losses.EnergyScoreLoss"


@pytest.mark.parametrize("norm_over", ALL_NORMS, ids=NORM_IDS)
def test_energy_score_uses_dimension_specific_sharding_contract(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: Norm,
    mocker: MockerFixture,
) -> None:
    pred, target = score_inputs
    group = object()
    all_to_all = mocker.patch(
        "anemoi.training.losses.energy_score.all_to_all_transpose",
        side_effect=lambda value, *_args: value,
    )
    gather = mocker.patch(
        "anemoi.training.losses.energy_score.gather_tensor",
        side_effect=lambda value, *_args: value,
    )
    mocker.patch(
        "anemoi.training.losses.energy_score.get_shard_sizes",
        return_value=[pred.shape[-1]],
    )
    norm_reduce = mocker.patch(
        "anemoi.training.losses.energy_score.reduce_tensor",
        side_effect=lambda value, _group: value,
    )
    final_reduce = mocker.patch(
        "anemoi.training.losses.base.reduce_tensor",
        side_effect=lambda value, _group: value,
    )
    maximum = mocker.patch.object(EnergyScoreLoss, "_maximum_across_group", side_effect=lambda value, _group: value)
    loss = EnergyScoreLoss(norm_over=norm_over)

    score = loss(
        pred,
        target,
        grid_shard_slice=slice(0, pred.shape[-2]),
        grid_shard_sizes=[pred.shape[-2]],
        grid_dim=TensorDim.GRID,
        group=group,
    )

    transposes = "grid" in norm_over and "variables" not in norm_over
    reduces_in_norm = "grid" in norm_over and "variables" in norm_over
    assert score.shape == ()
    assert all_to_all.call_count == (3 if transposes else 0)
    assert gather.call_count == (1 if transposes else 0)
    assert norm_reduce.call_count == (3 if reduces_in_norm else 0)
    assert final_reduce.call_count == (0 if "grid" in norm_over else 1)
    assert maximum.call_count == (3 if reduces_in_norm else 0)
    assert loss.needs_shard_layout_info is transposes


def test_energy_score_name_follows_tensor_order() -> None:
    assert EnergyScoreLoss(norm_over=["variables", "time"]).name == "fenergy_score_time_variables"
    assert EnergyScoreLoss(fair=False, norm_over=["grid"]).name == "energy_score_grid"


@pytest.mark.parametrize("norm_over", [["unknown"], [], ["grid", "grid"]])
def test_energy_score_rejects_invalid_norm(norm_over: list[str]) -> None:
    with pytest.raises(ValueError, match="Invalid energy score norm"):
        EnergyScoreLoss(norm_over=norm_over)  # type: ignore[arg-type]


@pytest.mark.parametrize("fair", [True, False])
def test_energy_score_with_one_dimensional_outcome_is_crps(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    fair: bool,
) -> None:
    # check that ES reduces to CRPS for one-dimensional outcomes (tested here with 1 variable)
    pred, target = score_inputs
    pred, target = pred[..., :1], target[..., :1]
    loss = EnergyScoreLoss(fair=fair, norm_over=["variables"])
    crps = CRPS(alpha=1.0 if fair else 0.0)

    torch.testing.assert_close(loss(pred, target, squash=False), crps(pred, target, squash=False))
