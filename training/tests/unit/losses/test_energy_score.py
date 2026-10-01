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
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pytest_mock import MockerFixture
from torch.autograd import gradcheck

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import EnergyScoreLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.energy_score import EnergyScoreNorm
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace


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
    norm_over: EnergyScoreNorm,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        weights = torch.ones_like(target)
    weighted_pred = pred * torch.sqrt(weights)
    weighted_target = target * torch.sqrt(weights)
    norm_dimensions = {
        "spatial": (-2,),
        "variables": (-1,),
        "spatial_and_variables": (-2, -1),
    }[norm_over]

    observation_term = torch.linalg.vector_norm(
        weighted_pred - weighted_target,
        dim=norm_dimensions,
    ).mean(dim=2)
    ensemble_size = pred.shape[2]
    pair_distance_sum = torch.zeros_like(observation_term)
    for member in range(ensemble_size - 1):
        pair_distance_sum += torch.linalg.vector_norm(
            weighted_pred[:, :, member].unsqueeze(2) - weighted_pred[:, :, member + 1 :],
            dim=norm_dimensions,
        ).sum(dim=2)

    pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if fair else 1.0 / (ensemble_size**2)
    return observation_term - pair_coefficient * pair_distance_sum


@pytest.mark.parametrize("fair", [True, False])
@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_matches_reference(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    fair: bool,
    norm_over: EnergyScoreNorm,
) -> None:
    pred, target = score_inputs
    loss = EnergyScoreLoss(fair=fair, norm_over=norm_over)

    actual = loss(pred, target, squash=False)
    expected = _energy_score_reference(
        pred,
        target,
        fair=fair,
        norm_over=norm_over,
    )
    if norm_over == "spatial":
        expected = expected.sum(dim=1).mean(dim=0)
    elif norm_over == "variables":
        expected = expected.sum(dim=(1, 2)).mean(dim=0)
    else:
        expected = expected.sum(dim=1).mean(dim=0)
    if norm_over != "spatial":
        expected = expected.expand(pred.shape[-1])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_variables", [1, 2])
def test_spatial_energy_score_follows_standard_reduction(
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


@pytest.mark.parametrize("norm_over", ["variables", "spatial_and_variables"])
def test_variable_joint_energy_score_repeats_diagnostic_value(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: EnergyScoreNorm,
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


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_applies_weights_in_the_norm(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: EnergyScoreNorm,
) -> None:
    pred, target = score_inputs
    grid_weights = torch.tensor([0.2, 0.3, 0.5], dtype=pred.dtype)
    variable_weights = torch.tensor([0.25, 0.75], dtype=pred.dtype)
    loss = EnergyScoreLoss(norm_over=norm_over)
    loss.add_scaler(TensorDim.GRID, grid_weights, name="grid")
    loss.add_scaler(TensorDim.VARIABLE, variable_weights, name="variable")

    actual = loss(pred, target, squash=False)
    if norm_over == "spatial_and_variables":
        weights = grid_weights[:, None] * variable_weights[None, :]
        expected = (
            _energy_score_reference(
                pred,
                target,
                fair=True,
                norm_over=norm_over,
                weights=weights.reshape(1, 1, 1, 3, 2),
            )
            .sum(dim=1)
            .mean(dim=0)
        )
        expected = expected.expand(pred.shape[-1])
    elif norm_over == "spatial":
        weights = grid_weights.reshape(1, 1, 1, 3, 1)
        expected = (
            _energy_score_reference(
                pred,
                target,
                fair=True,
                norm_over=norm_over,
                weights=weights,
            )
            .sum(dim=1)
            .mean(dim=0)
        )
        expected = variable_weights * expected
    else:
        weights = variable_weights.reshape(1, 1, 1, 1, 2)
        expected = _energy_score_reference(
            pred,
            target,
            fair=True,
            norm_over=norm_over,
            weights=weights,
        )
        expected = (expected * grid_weights.reshape(1, 1, 3)).sum(dim=(1, 2)).mean(dim=0)
        expected = expected.expand(pred.shape[-1])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("norm_over", "positive_dimension", "negative_dimension", "weights"),
    [
        ("spatial", TensorDim.GRID, -2, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)),
        ("variables", TensorDim.VARIABLE, -1, torch.tensor([0.25, 0.75], dtype=torch.float64)),
        ("spatial_and_variables", TensorDim.GRID, -2, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)),
        ("spatial_and_variables", TensorDim.VARIABLE, -1, torch.tensor([0.25, 0.75], dtype=torch.float64)),
    ],
)
def test_energy_score_resolves_negative_norm_dimensions(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: EnergyScoreNorm,
    positive_dimension: int,
    negative_dimension: int,
    weights: torch.Tensor,
) -> None:
    pred, target = score_inputs
    positive_loss = EnergyScoreLoss(norm_over=norm_over)
    negative_loss = EnergyScoreLoss(norm_over=norm_over)
    positive_loss.add_scaler(positive_dimension, weights)
    negative_loss.add_scaler(negative_dimension, weights)

    expected = positive_loss(pred, target, squash=False)
    actual = negative_loss(pred, target, squash=False)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("norm_over", "dimension", "weights"),
    [
        ("spatial", -2, torch.tensor([-0.2, 0.3, 0.5])),
        ("variables", -1, torch.tensor([-0.25, 0.75])),
        ("spatial_and_variables", -2, torch.tensor([-0.2, 0.3, 0.5])),
        ("spatial_and_variables", -1, torch.tensor([-0.25, 0.75])),
    ],
)
def test_energy_score_validates_negative_norm_dimensions(
    norm_over: EnergyScoreNorm,
    dimension: int,
    weights: torch.Tensor,
) -> None:
    loss = EnergyScoreLoss(norm_over=norm_over)

    with pytest.raises(ValueError, match="non-negative"):
        loss.add_scaler(dimension, weights)

    loss.add_scaler(dimension, torch.ones_like(weights), name="weights")
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


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_has_finite_gradients_at_zero(norm_over: EnergyScoreNorm) -> None:
    pred = torch.zeros(1, 1, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 1, 1, 3, 2, dtype=torch.float64)
    loss = EnergyScoreLoss(norm_over=norm_over)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_has_finite_gradients_for_large_values(norm_over: EnergyScoreNorm) -> None:
    pred = torch.tensor(
        [[[[[1.0e20, -1.0e20], [-1.0e20, 0.5e20]], [[1.5e20, 0.25e20], [-0.5e20, 1.0e20]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[[0.25e20, -0.25e20], [0.0, 0.75e20]]]]],
        dtype=torch.float32,
    )
    loss = EnergyScoreLoss(norm_over=norm_over)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_ignores_missing_features_with_finite_gradients(norm_over: EnergyScoreNorm) -> None:
    pred = torch.tensor(
        [[[[[0.0, 1.0], [1.0, 2.0]], [[2.0, 0.0], [3.0, 1.0]], [[1.0, 3.0], [2.0, 4.0]]]]],
        dtype=torch.float64,
    )
    pred[0, 0, 0, 0, 1] = torch.nan
    pred.requires_grad_()
    target = torch.tensor([[[[[1.0, 0.5], [2.0, 2.5]]]]], dtype=torch.float64)
    loss = EnergyScoreLoss(norm_over=norm_over, ignore_nans=True)

    score = loss(pred, target)
    score.backward()

    assert torch.isfinite(score)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_gradcheck(norm_over: EnergyScoreNorm) -> None:
    pred = torch.tensor(
        [[[[[0.1, 1.3], [0.5, 2.1]], [[2.2, -0.7], [1.4, 0.2]], [[1.1, 0.4], [-0.3, 1.7]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([[[[[0.7, -0.2], [1.8, 0.9]]]]], dtype=torch.float64)
    loss = EnergyScoreLoss(norm_over=norm_over)

    assert gradcheck(lambda value: loss(value, target), (pred,), eps=1e-6, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("norm_over", ["variables", "spatial_and_variables"])
def test_filtered_variable_joint_energy_score_maps_repeated_value(norm_over: EnergyScoreNorm) -> None:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"a": 0, "b": 1, "c": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.EnergyScoreLoss",
                "scalers": [],
                "norm_over": norm_over,
                "predicted_variables": ["a", "c"],
                "target_variables": ["a", "c"],
            },
        ),
        data_indices=data_indices,
    )
    pred = torch.tensor(
        [[[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], [[2.0, 0.0, 4.0], [3.0, 1.0, 2.0]]]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[[[[1.0, 0.5, 2.5], [2.0, 2.5, 1.0]]]]], dtype=torch.float64)
    loss_kwargs = {
        "pred_layout": IndexSpace.MODEL_OUTPUT,
        "target_layout": IndexSpace.DATA_OUTPUT,
    }

    assert isinstance(loss, LossVariableMapper)
    scalar = loss(pred, target, **loss_kwargs)
    per_variable = loss(pred, target, squash=False, **loss_kwargs)

    torch.testing.assert_close(per_variable, torch.stack((scalar, scalar.new_zeros(()), scalar)))


def test_energy_score_schemas_accept_direct_nested_and_combined_configs() -> None:
    direct = TypeAdapter(LossSchemas).validate_python(
        {
            "_target_": "anemoi.training.losses.EnergyScoreLoss",
            "scalers": [],
            "norm_over": "variables",
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

    assert direct.norm_over == "variables"
    assert nested.per_scale_loss.target_ == "anemoi.training.losses.EnergyScoreLoss"
    assert combined.losses[0].target_ == "anemoi.training.losses.EnergyScoreLoss"


@pytest.mark.parametrize("norm_over", ["spatial", "variables", "spatial_and_variables"])
def test_energy_score_uses_mode_specific_sharding_contract(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    norm_over: EnergyScoreNorm,
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

    assert score.shape == ()
    assert all_to_all.call_count == (3 if norm_over == "spatial" else 0)
    assert gather.call_count == (1 if norm_over == "spatial" else 0)
    assert norm_reduce.call_count == (3 if norm_over == "spatial_and_variables" else 0)
    assert final_reduce.call_count == (1 if norm_over == "variables" else 0)
    assert maximum.call_count == (3 if norm_over == "spatial_and_variables" else 0)
    assert loss.needs_shard_layout_info is (norm_over == "spatial")


def test_energy_score_rejects_unknown_norm() -> None:
    with pytest.raises(ValueError, match="Unknown energy score norm"):
        EnergyScoreLoss(norm_over="unknown")  # type: ignore[arg-type]
