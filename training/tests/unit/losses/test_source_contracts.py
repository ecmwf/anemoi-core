# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data import TensorLayout
from anemoi.models.data.views import GriddedSourceView
from anemoi.models.data.views import TabularSourceView
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CRPS
from anemoi.training.losses import EnergyScoreLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses.graph_energy_score import GraphEnergyScoreLoss
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.edm_diffusion import EDMDiffusionTransportObjective
from anemoi.training.utils.index_space import IndexSpace


def _grid(data: torch.Tensor, layout: TensorLayout | None = None) -> GriddedSourceView:
    return GriddedSourceView(
        name="grid",
        data=data,
        variables=["a", "b"],
        statistics={},
        coordinates=torch.zeros(3, 2),
        layout=layout or TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
        coordinates_are_static=True,
    )


@pytest.mark.parametrize("side", ["prediction", "target"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_training_checks_float32_before_casting(side: str, dtype: torch.dtype) -> None:
    pred = _grid(torch.ones(1, 1, 1, 3, 2, dtype=dtype if side == "prediction" else torch.float32))
    target = _grid(torch.zeros(1, 1, 1, 3, 2, dtype=dtype if side == "target" else torch.float32))
    with pytest.raises(AssertionError, match="must be float32"):
        BaseTrainingModule._evaluate_loss(MSELoss(), pred, target)


def test_training_loss_disables_outer_autocast() -> None:
    pred = _grid(torch.ones(1, 1, 1, 3, 2, requires_grad=True))
    target = _grid(torch.zeros_like(pred.data))

    def loss(p: GriddedSourceView, t: GriddedSourceView) -> torch.Tensor:
        assert not torch.is_autocast_enabled("cpu")
        return MSELoss()(p, t)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = BaseTrainingModule._evaluate_loss(loss, pred, target)
    torch.testing.assert_close(result, torch.tensor(3.0))
    result.backward()
    torch.testing.assert_close(pred.data.grad, torch.ones_like(pred.data))


def test_filtered_loss_preserves_promoted_arithmetic_and_gradients() -> None:
    indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), {"a": 0, "b": 1})
    loss = LossVariableMapper(MSELoss(), predicted_variables=["a"], data_indices=indices)
    pred = _grid(torch.full((1, 1, 1, 3, 2), 300.0, dtype=torch.bfloat16, requires_grad=True))
    target = _grid(torch.zeros_like(pred.data, dtype=torch.float32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = loss(
            pred,
            target,
            squash=False,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            target_layout=IndexSpace.MODEL_OUTPUT,
        )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([270000.0, 0.0]))
    result.sum().backward()
    torch.testing.assert_close(pred.data.grad[..., 0], torch.full_like(pred.data.grad[..., 0], 600.0))
    assert torch.count_nonzero(pred.data.grad[..., 1]) == 0


@pytest.mark.parametrize("loss_type", [EnergyScoreLoss, GraphEnergyScoreLoss])
def test_scores_accept_equivalent_negative_axes(loss_type: type[EnergyScoreLoss] | type[GraphEnergyScoreLoss]) -> None:
    # Every ensemble member is one unit from the target at each of three grid points.
    pred = torch.ones(1, 1, 2, 3, 2, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 1, 1, 3, 2, dtype=torch.float64)
    negative = TensorLayout(batch=-5, time=-4, ensemble=-3, grid=-2, variables=-1)
    loss = loss_type()
    result = loss(_grid(pred, negative), _grid(target, negative))
    reference = loss(_grid(pred), _grid(target))
    torch.testing.assert_close(result, reference)
    expected = 3.0**0.5 if loss_type is EnergyScoreLoss else 3.0
    torch.testing.assert_close(result, result.new_tensor(expected))
    actual_grad = torch.autograd.grad(result, pred, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, pred)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


def _observations() -> TabularSourceView:
    return TabularSourceView(
        name="obs",
        data=[torch.ones(2, 2), torch.ones(3, 2)],
        variables=["a", "b"],
        statistics={},
        coordinates=[torch.zeros(2, 2), torch.zeros(3, 2)],
        layout=TensorLayout(grid=0, variables=1, time_in_grid=True),
    )


def test_sparse_loss_distinguishes_shared_and_per_sample_arguments() -> None:
    view = _observations()
    matrices = [torch.eye(2), torch.ones(2, 2)]
    weights = [torch.tensor(2.0), torch.tensor(3.0)]

    def loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        matrices: list[torch.Tensor],
        weight: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        assert isinstance(matrices, list)
        return ((pred @ matrices[0]) + (target @ matrices[1])).mean() * weight

    result = view.apply_loss(view, loss, matrices=matrices, per_sample_kwargs={"weight": weights})
    torch.testing.assert_close(result, torch.tensor(7.5))


def test_edm_loss_passes_sparse_weights_and_preserves_gradients() -> None:
    pred = _observations().clone(
        data=[torch.ones(2, 2, requires_grad=True), torch.full((3, 2), 2.0, requires_grad=True)],
    )
    target = pred.apply_func(lambda data, **_: torch.zeros_like(data))
    module = SimpleNamespace(
        loss={"obs": WeightedMSELoss()},
        model_comm_group=None,
        _evaluate_loss=BaseTrainingModule._evaluate_loss,
    )
    weights = {"obs": [torch.full((2, 1), 2.0), torch.full((3, 1), 3.0)]}
    result = EDMDiffusionTransportObjective(module).compute_loss(pred, target, dataset_name="obs", weights=weights)
    torch.testing.assert_close(result, torch.tensor(7.0))
    result.backward()
    torch.testing.assert_close(pred.data[0].grad, torch.full((2, 2), 0.5))
    torch.testing.assert_close(pred.data[1].grad, torch.ones(3, 2))


@pytest.mark.parametrize("case", ["length", "duplicate"])
def test_sparse_loss_validates_explicit_sample_arguments(case: str) -> None:
    view = _observations()
    per_sample = {"weight": [torch.tensor(1.0)] * (1 if case == "length" else 2)}
    shared = {} if case == "length" else {"weight": torch.tensor(1.0)}
    with pytest.raises(ValueError, match=r"one value per sample|both shared and per-sample"):
        view.apply_loss(
            view,
            lambda *_args, **_kwargs: pytest.fail("Invalid arguments reached the loss"),
            per_sample_kwargs=per_sample,
            **shared,
        )


@pytest.mark.parametrize("backend", ["naive", "stable"])
def test_sparse_crps_ensemble_axis_and_nan_gradients(backend: str) -> None:
    data = torch.tensor([[[-1.0], [-1.0]], [[1.0], [1.0]]], requires_grad=True)
    pred = TabularSourceView(
        name="obs",
        data=[data],
        variables=["a"],
        statistics={},
        coordinates=[torch.zeros(2, 2)],
        layout=TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True),
    )
    target = pred.clone(data=[torch.tensor([[[0.0], [float("nan")]]])])
    result = CRPS(alpha=0.0, backend=backend, ignore_nans=True)(pred, target)
    # Standard CRPS is 0.5 at the valid node; the masked node contributes zero.
    torch.testing.assert_close(result, torch.tensor(0.25))
    result.backward()
    assert torch.isfinite(data.grad).all()
    assert torch.count_nonzero(data.grad[:, 1]) == 0
    assert data.grad[:, 0].abs().sum() > 0
