# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Multiscale loss with one shared neighbour list and a Gaussian width per scale."""

import gc
import logging
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from torch_geometric.data import HeteroData

from anemoi.graphs.builders import build_node_to_node_projection_subgraph
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import _NEIGHBOUR_LISTS
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.losses.multiscale import NeighbourSmoothingScale
from anemoi.training.schemas.training import MultiscaleConfigNeighbourListSchema
from tests.unit.grids import octahedral_graph


class RecordingLoss(BaseLoss):
    """Returns the mean squared difference and keeps the fields it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash: bool = True, **kwargs) -> torch.Tensor:
        del squash, kwargs
        self.inputs.append((pred.detach().clone(), target.detach().clone()))
        return ((pred - target) ** 2).mean()


SIGMAS = (0.08, 0.04)  # widest first
NEIGHBOUR_LIST_CONFIG = {"num_nearest_neighbours": 9, "sigmas": list(SIGMAS)}


def _wrapper(
    graph: HeteroData,
    multiscale_config: dict,
    checkpoint_scales: bool = False,
    sparse_projector_backend: str = "torch",
) -> MultiscaleLossWrapper:
    return MultiscaleLossWrapper(
        per_scale_loss=RecordingLoss(),
        weights=[1.0, 2.0, 3.0],
        multiscale_config=multiscale_config,
        graph_data=graph,
        data_node_name="data",
        checkpoint_scales=checkpoint_scales,
        sparse_projector_backend=sparse_projector_backend,
    )


def _save_gaussian_matrix(graph: HeteroData, num_neighbours: int, sigma: float, path: Path) -> str:
    """Save a KNN Gaussian smoothing matrix as .npz."""
    subgraph = build_node_to_node_projection_subgraph(
        graph,
        "data",
        "data",
        {"num_nearest_neighbours": num_neighbours, "sigma": sigma},
    )
    provider = ProjectionGraphProvider(
        graph=subgraph,
        edges_name=("data", "to", "data"),
        edge_weight_attribute=DEFAULT_EDGE_WEIGHT_ATTRIBUTE,
        backend="torch",
    )
    save_npz(path, csr_matrix(provider.get_edges().to_dense().numpy()))
    return path.name


def _file_config(graph: HeteroData, num_neighbours: int, path: Path) -> dict:
    """File-mode config with one matrix per width in ``SIGMAS``."""
    files = [
        _save_gaussian_matrix(graph, num_neighbours, sigma, path / f"smooth_{i}.npz") for i, sigma in enumerate(SIGMAS)
    ]
    return {"loss_matrices_path": str(path), "loss_matrices": [*files, None]}


def test_neighbour_list_gives_the_scales_of_the_smoothing_matrices(tmp_path: Path) -> None:
    graph = octahedral_graph(6)
    neighbour_list = _wrapper(graph, {"num_nearest_neighbours": 16, "sigmas": list(SIGMAS)})
    # Same neighbours and widths, as matrices from files.
    matrices = _wrapper(graph, _file_config(graph, 16, tmp_path))
    generator = torch.Generator().manual_seed(0)
    num_points = graph["data"].num_nodes
    pred = torch.randn(1, 1, 2, num_points, 3, generator=generator)
    target = torch.randn(1, 1, 1, num_points, 3, generator=generator)

    loss_neighbours = neighbour_list(pred, target)
    loss_matrices = matrices(pred, target)

    torch.testing.assert_close(loss_neighbours, loss_matrices, rtol=1e-5, atol=1e-6)
    for (pred_a, target_a), (pred_b, target_b) in zip(neighbour_list.loss.inputs, matrices.loss.inputs, strict=True):
        torch.testing.assert_close(pred_a, pred_b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(target_a, target_b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_file_mode_providers_keep_the_form_their_backend_reads(tmp_path: Path, backend: str) -> None:
    graph = octahedral_graph(4)
    wrapper = _wrapper(graph, _file_config(graph, 9, tmp_path), sparse_projector_backend=backend)

    for provider in wrapper.smoothers[:-1]:
        assert provider.backend == backend
        assert (provider.projection_table is not None) == (backend == "triton")
        assert (provider.projection_matrix is not None) == (backend == "torch")


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_neighbour_list_scales_use_the_backend_of_the_loss(backend: str) -> None:
    wrapper = _wrapper(octahedral_graph(4), NEIGHBOUR_LIST_CONFIG, sparse_projector_backend=backend)

    assert [scale.backend for scale in wrapper.smoothers[:-1]] == [backend] * len(SIGMAS)


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_compile_for_training_compiles_only_the_torch_smoothing(backend: str) -> None:
    wrapper = _wrapper(octahedral_graph(4), NEIGHBOUR_LIST_CONFIG, sparse_projector_backend=backend)

    wrapper.compile_for_training()

    compiled = [scale.compiled for scale in wrapper.smoothers[:-1]]
    if backend == "torch":
        assert all(passes is not None for passes in compiled)
        assert len({id(passes) for passes in compiled}) == 1
    else:
        assert all(passes is None for passes in compiled)


def test_scales_share_one_neighbour_list_per_graph() -> None:
    graph = octahedral_graph(4)
    first, second = _wrapper(graph, NEIGHBOUR_LIST_CONFIG), _wrapper(graph, NEIGHBOUR_LIST_CONFIG)

    scales = [s for wrapper in (first, second) for s in wrapper.smoothers if s is not None]
    assert all(isinstance(scale, NeighbourSmoothingScale) for scale in scales)
    assert len({id(scale.neighbour_list) for scale in scales}) == 1
    assert [scale.sigma for scale in first.smoothers[:-1]] == list(SIGMAS)
    assert first.smoothers[-1] is None

    other_graph = octahedral_graph(4)
    third = _wrapper(other_graph, NEIGHBOUR_LIST_CONFIG)
    assert third.smoothers[0].neighbour_list is not first.smoothers[0].neighbour_list


def test_neighbour_lists_are_kept_per_node_set_and_number_of_neighbours() -> None:
    graph = octahedral_graph(4)
    graph["other"].x = graph["data"].x
    base = _wrapper(graph, NEIGHBOUR_LIST_CONFIG).smoothers[0].neighbour_list
    more_neighbours = _wrapper(graph, {**NEIGHBOUR_LIST_CONFIG, "num_nearest_neighbours": 12}).smoothers[0]
    other_nodes = MultiscaleLossWrapper(
        per_scale_loss=RecordingLoss(),
        weights=[1.0, 2.0, 3.0],
        multiscale_config=NEIGHBOUR_LIST_CONFIG,
        graph_data=graph,
        data_node_name="other",
    ).smoothers[0]

    assert more_neighbours.neighbour_list is not base
    assert more_neighbours.neighbour_list.num_neighbours == 12
    assert other_nodes.neighbour_list is not base
    assert other_nodes.neighbour_list.nodes_name == "other"


def test_neighbour_list_is_dropped_with_its_graph() -> None:
    graph = octahedral_graph(4)
    _wrapper(graph, NEIGHBOUR_LIST_CONFIG)
    key = (id(graph), "data", NEIGHBOUR_LIST_CONFIG["num_nearest_neighbours"])
    assert key in _NEIGHBOUR_LISTS

    del graph
    gc.collect()

    assert key not in _NEIGHBOUR_LISTS


def test_gradients_reach_the_prediction() -> None:
    graph = octahedral_graph(4)
    wrapper = _wrapper(graph, NEIGHBOUR_LIST_CONFIG)
    pred = torch.randn(1, 1, 2, graph["data"].num_nodes, 2, requires_grad=True)
    target = torch.randn(1, 1, 1, graph["data"].num_nodes, 2)

    wrapper(pred, target).backward()

    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["neighbour_list", "file"])
def test_checkpointed_scales_give_the_same_loss_and_gradients(tmp_path: Path, mode: str) -> None:
    graph = octahedral_graph(4)
    config = NEIGHBOUR_LIST_CONFIG if mode == "neighbour_list" else _file_config(graph, 9, tmp_path)
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 1, 2, graph["data"].num_nodes, 2, generator=generator)
    target = torch.randn(1, 1, 1, graph["data"].num_nodes, 2, generator=generator)

    losses, gradients = [], []
    for checkpoint_scales in (False, True):
        wrapper = _wrapper(graph, config, checkpoint_scales=checkpoint_scales)
        pred_leaf = pred.clone().requires_grad_()
        loss = wrapper(pred_leaf, target)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(pred_leaf.grad)

    # Checkpointing does not change the forward computation.
    torch.testing.assert_close(losses[1], losses[0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-5, atol=1e-6)


def test_checkpointed_scales_without_gradients_give_the_same_loss() -> None:
    graph = octahedral_graph(4)
    pred = torch.randn(1, 1, 2, graph["data"].num_nodes, 2)
    target = torch.randn(1, 1, 1, graph["data"].num_nodes, 2)

    with torch.no_grad():
        plain = _wrapper(graph, NEIGHBOUR_LIST_CONFIG)(pred, target)
        checkpointed = _wrapper(graph, NEIGHBOUR_LIST_CONFIG, checkpoint_scales=True)(pred, target)

    torch.testing.assert_close(checkpointed, plain, rtol=0, atol=0)


def test_neighbours_that_do_not_reach_far_enough_for_the_widest_sigma_give_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = octahedral_graph(4)
    with caplog.at_level(logging.WARNING, logger="anemoi.training.losses.multiscale"):
        _wrapper(graph, NEIGHBOUR_LIST_CONFIG)
        assert "cut off" not in caplog.text
        # The 9 nearest neighbours on O4 reach about 0.4 rad.
        _wrapper(graph, {**NEIGHBOUR_LIST_CONFIG, "sigmas": [0.4, 0.08]})
        assert "cut off" in caplog.text


def test_sigmas_must_go_from_widest_to_narrowest() -> None:
    graph = octahedral_graph(4)
    with pytest.raises(ValueError, match="widest"):
        _wrapper(graph, {"num_nearest_neighbours": 9, "sigmas": [0.04, 0.08]})
    with pytest.raises(ValidationError, match="widest"):
        MultiscaleConfigNeighbourListSchema(num_nearest_neighbours=9, sigmas=[0.04, 0.08])


def test_neighbour_list_keys_do_not_mix_with_other_modes() -> None:
    graph = octahedral_graph(4)
    with pytest.raises(ValueError, match="loss_matrices"):
        _wrapper(graph, {**NEIGHBOUR_LIST_CONFIG, "loss_matrices": [None]})
