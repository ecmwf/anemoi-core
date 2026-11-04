# (C) Copyright 2024-2026 Anemoi contributors.
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
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import GraphLaplacianSmoothnessLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import NaNAwareMSELoss
from anemoi.training.losses.scalers.variable import GeneralVariableLossScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

# ── NaNAwareMSELoss ───────────────────────────────────────────────────────


def test_nan_aware_mse_no_nans_matches_mse() -> None:
    loss = NaNAwareMSELoss()
    pred = torch.randn(2, 1, 1, 8, 3)
    target = torch.randn(2, 1, 1, 8, 3)
    assert torch.allclose(loss(pred, target), MSELoss()(pred, target))


def test_nan_aware_mse_density_weighting() -> None:
    """A 50%-NaN variable with the same per-point error contributes equally to a dense one."""
    loss = NaNAwareMSELoss()
    grid = 8
    pred = torch.zeros(1, 1, 1, grid, 2)
    target = torch.ones(1, 1, 1, grid, 2)  # error of 1 everywhere valid
    target[..., grid // 2 :, 1] = torch.nan  # second variable: half the grid is NaN

    out = loss(pred, target, squash=False)
    # dense variable: sum of 8 unit errors = 8
    # sparse variable: 4 unit errors, density weight 8/4 = 2 -> also 8
    assert torch.allclose(out[0], out[1])


def test_nan_aware_mse_fully_nan_variable() -> None:
    loss = NaNAwareMSELoss()
    pred = torch.randn(1, 1, 1, 4, 1)
    target = torch.full((1, 1, 1, 4, 1), torch.nan)
    assert torch.isfinite(loss(pred, target))


def test_nan_aware_mse_does_not_support_sharding() -> None:
    assert NaNAwareMSELoss().supports_sharding is False


# ── GraphLaplacianSmoothnessLoss ──────────────────────────────────────────


@pytest.fixture
def ring_edge_index() -> torch.Tensor:
    """Bidirectional ring over 6 nodes."""
    num_nodes = 6
    src = list(range(num_nodes)) * 2
    dst = [(i + 1) % num_nodes for i in range(num_nodes)] + [(i - 1) % num_nodes for i in range(num_nodes)]
    return torch.tensor([src, dst], dtype=torch.long)


def test_graph_laplacian_zero_for_constant_field(ring_edge_index: torch.Tensor) -> None:
    loss = GraphLaplacianSmoothnessLoss(edge_index=ring_edge_index)
    pred = torch.ones(2, 1, 1, 6, 3)
    target = torch.zeros_like(pred)
    assert torch.allclose(loss(pred, target), torch.tensor(0.0))


def test_graph_laplacian_positive_for_rough_field(ring_edge_index: torch.Tensor) -> None:
    loss = GraphLaplacianSmoothnessLoss(edge_index=ring_edge_index, penalty_weight=2.0)
    pred = torch.zeros(1, 1, 1, 6, 1)
    pred[..., ::2, :] = 1.0  # alternating field: maximally rough on a ring
    out = loss(pred, torch.zeros_like(pred))
    assert out > 0


def test_graph_laplacian_from_graph_data() -> None:
    graph = HeteroData()
    graph["obs", "to", "obs"].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    loss = GraphLaplacianSmoothnessLoss(graph_data=graph, data_node_name="obs")
    assert loss.edge_index.shape == (2, 2)


def test_graph_laplacian_no_edges_returns_zero() -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 4
    loss = GraphLaplacianSmoothnessLoss(graph_data=graph)
    pred = torch.randn(1, 1, 1, 4, 2)
    assert loss(pred, torch.zeros_like(pred)) == 0.0


def test_graph_laplacian_flags() -> None:
    assert GraphLaplacianSmoothnessLoss.needs_graph_data is True
    loss = GraphLaplacianSmoothnessLoss(edge_index=torch.empty(2, 0, dtype=torch.long))
    assert loss.supports_sharding is False


# ── GeneralVariableLossScaler per-level fallback ──────────────────────────


@pytest.fixture
def scaler_setup() -> tuple[IndexCollection, ExtractVariableGroupAndLevel]:
    config = DictConfig({"data": {"forcing": ["x"], "diagnostic": []}})
    name_to_index = {"x": 0, "z_500": 1, "z_850": 2, "tp": 3}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    metadata_extractor = ExtractVariableGroupAndLevel(variable_groups={"default": "sfc", "pl": ["z"]})
    return data_indices, metadata_extractor


def test_general_variable_scaler_per_level_fallback(
    scaler_setup: tuple[IndexCollection, ExtractVariableGroupAndLevel],
) -> None:
    """Lookup chain: full name (z_500) -> base variable (z) -> default."""
    data_indices, metadata_extractor = scaler_setup
    scaler = GeneralVariableLossScaler(
        data_indices=data_indices,
        weights=DictConfig({"default": 1.0, "z": 700, "z_500": 800}),
        metadata_extractor=metadata_extractor,
    )
    values = scaler.get_scaling_values()
    output_names = [name for name, _ in sorted(data_indices.data.output.name_to_index.items(), key=lambda kv: kv[1])]
    kept_names = [n for n in output_names if n in {"z_500", "z_850", "tp"}]
    by_name = dict(zip(kept_names, values.tolist(), strict=True))
    assert by_name["z_500"] == 800  # full-name override wins
    assert by_name["z_850"] == 700  # base-variable fallback
    assert by_name["tp"] == 1.0  # default fallback
