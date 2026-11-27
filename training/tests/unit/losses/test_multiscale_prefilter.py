# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MultiscaleLossWrapper selects the variables of an inner LossVariableMapper *before* smoothing.

The smoothing is applied independently to every variable, so selecting first must give exactly the same loss and
gradients as smoothing everything and selecting afterwards, while the smoothing only ever sees the selected variables.
"""

import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses import multiscale as multiscale_module
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.utils.index_space import IndexSpace

NUM_GRID = 4
NUM_TIME = 3
NUM_ENS = 2
MODEL_VARIABLES = 3  # a, b, c
DATA_VARIABLES = 4  # a, b, c + the target-only variable "imerg"
LAYOUT_KWARGS = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_FULL}


@pytest.fixture
def data_indices() -> IndexCollection:
    return IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": ["imerg"]}),
        {"a": 0, "b": 1, "c": 2, "imerg": 3},
    )


@pytest.fixture
def smoothing_provider() -> ProjectionGraphProvider:
    graph = HeteroData()
    graph["src"].num_nodes = NUM_GRID
    graph["dst"].num_nodes = NUM_GRID
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 0]])
    graph[("src", "to", "dst")].edge_weight = torch.ones(8) / 2
    return ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )


def _leaf(target: str, predicted: list[str] | None, target_variables: list[str] | None) -> dict:
    leaf = {"_target_": target, "scalers": []}
    if predicted is not None:
        leaf["predicted_variables"] = predicted
        leaf["target_variables"] = target_variables
    return leaf


def _build(
    kind: str,
    predicted: list[str] | None,
    target_variables: list[str] | None,
    data_indices: IndexCollection,
    provider: ProjectionGraphProvider,
    mocker: MockerFixture,
) -> MultiscaleLossWrapper:
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[None, provider],
    )
    crps = _leaf("anemoi.training.losses.CRPS", predicted, target_variables)
    if kind == "per_step":
        per_scale = crps
    else:
        per_scale = {
            "_target_": "anemoi.training.losses.aggregate.TimeAggregateLossWrapper",
            "scalers": [],
            "time_aggregation_types": ["mean", "max"],
            "loss_fn": crps,
        }
    return get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.MultiscaleLossWrapper", "weights": [1.0, 2.0], "per_scale_loss": per_scale}),
        scalers={},
        data_indices=data_indices,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    pred = torch.rand((1, NUM_TIME, NUM_ENS, NUM_GRID, MODEL_VARIABLES), generator=generator)
    target = torch.rand((1, NUM_TIME, 1, NUM_GRID, DATA_VARIABLES), generator=generator)
    return pred, target


def _loss_and_grad(loss_fn: MultiscaleLossWrapper, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    pred, target = _inputs()
    pred.requires_grad_(True)
    out = loss_fn(pred, target, **LAYOUT_KWARGS, **kwargs)
    out.sum().backward()
    return out.detach(), pred.grad.detach()


@pytest.mark.parametrize("kind", ["per_step", "time_aggregate"])
@pytest.mark.parametrize(("predicted", "target_variables"), [(["b"], ["b"]), (["b"], ["imerg"]), (["c", "a"], ["c", "a"])])
def test_prefilter_matches_stock_loss_and_gradient(
    kind: str,
    predicted: list[str],
    target_variables: list[str],
    data_indices: IndexCollection,
    smoothing_provider: ProjectionGraphProvider,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss_fn = _build(kind, predicted, target_variables, data_indices, smoothing_provider, mocker)

    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", False)
    stock_loss, stock_grad = _loss_and_grad(loss_fn)
    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", True)
    patched_loss, patched_grad = _loss_and_grad(loss_fn)

    torch.testing.assert_close(patched_loss, stock_loss)
    torch.testing.assert_close(patched_grad, stock_grad)
    assert stock_grad.abs().sum() > 0, "the test must exercise a non-zero gradient"


@pytest.mark.parametrize("kind", ["per_step", "time_aggregate"])
def test_smoothing_only_sees_selected_variables(
    kind: str,
    data_indices: IndexCollection,
    smoothing_provider: ProjectionGraphProvider,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss_fn = _build(kind, ["b"], ["imerg"], data_indices, smoothing_provider, mocker)
    spy = mocker.spy(MultiscaleLossWrapper, "_smooth_for_loss")

    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", False)
    _loss_and_grad(loss_fn)
    assert {(call.args[1].shape[-1], call.args[2].shape[-1]) for call in spy.call_args_list} == {(MODEL_VARIABLES, DATA_VARIABLES)}

    spy.reset_mock()
    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", True)
    _loss_and_grad(loss_fn)
    assert {(call.args[1].shape[-1], call.args[2].shape[-1]) for call in spy.call_args_list} == {(1, 1)}


def test_no_variable_selection_is_left_untouched(
    data_indices: IndexCollection,
    smoothing_provider: ProjectionGraphProvider,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without predicted_variables the mapper keeps all model outputs: nothing may be selected before smoothing."""
    loss_fn = _build("per_step", None, None, data_indices, smoothing_provider, mocker)
    spy = mocker.spy(MultiscaleLossWrapper, "_smooth_for_loss")
    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", True)
    patched_loss, patched_grad = _loss_and_grad(loss_fn)
    assert {call.args[1].shape[-1] for call in spy.call_args_list} == {MODEL_VARIABLES}

    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", False)
    stock_loss, stock_grad = _loss_and_grad(loss_fn)
    torch.testing.assert_close(patched_loss, stock_loss)
    torch.testing.assert_close(patched_grad, stock_grad)


def test_per_variable_loss_is_not_prefiltered(
    data_indices: IndexCollection,
    smoothing_provider: ProjectionGraphProvider,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """squash=False needs the full variable axis, so the stock path is kept."""
    loss_fn = _build("per_step", ["b"], ["imerg"], data_indices, smoothing_provider, mocker)
    pred, target = _inputs()

    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", False)
    stock = loss_fn(pred, target, squash=False, **LAYOUT_KWARGS)
    monkeypatch.setattr(multiscale_module, "PREFILTER_VARIABLES", True)
    patched = loss_fn(pred, target, squash=False, **LAYOUT_KWARGS)

    assert patched.shape == (MODEL_VARIABLES,)
    torch.testing.assert_close(patched, stock)
