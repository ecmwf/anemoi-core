# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import get_graph_node_names
from anemoi.graphs.projection_helpers import multiscale_loss_matrices_graph
from anemoi.graphs.projection_helpers import projection_edge_name
from anemoi.graphs.projection_helpers import projection_node_name
from anemoi.graphs.projection_helpers import residual_projection_edge_names
from anemoi.graphs.projection_helpers import residual_projection_truncation_node_name
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph


@pytest.fixture
def fused_graph() -> HeteroData:
    graph = HeteroData()
    for node_name in [
        "era5",
        "cerra",
        "hidden",
        "era5_truncation",
        "era5_smooth_8x",
        "era5_smooth_4x",
    ]:
        graph[node_name].num_nodes = 1

    graph[("era5_smooth_4x", "to", "era5_smooth_4x")].edge_index = [[0], [0]]
    return graph


def test_get_graph_node_names_and_fused_detection_work_for_graphs_and_configs(fused_graph: HeteroData) -> None:
    graph_config = {
        "nodes": {
            "era5": {},
            "cerra": {},
            "hidden": {},
        },
    }
    single_named_dataset_graph = {
        "nodes": {
            "era5": {},
            "hidden": {},
        },
    }
    single_generic_dataset_graph = {
        "nodes": {
            "data": {},
            "hidden": {},
        },
    }

    assert get_graph_node_names(fused_graph) >= {"era5", "cerra", "hidden"}
    assert get_graph_node_names(graph_config) == {"era5", "cerra", "hidden"}
    assert uses_fused_dataset_graph(fused_graph, ["era5", "cerra"]) is True
    assert uses_fused_dataset_graph(graph_config, ["era5", "cerra"]) is True
    assert uses_fused_dataset_graph(single_named_dataset_graph, ["era5"]) is True
    assert uses_fused_dataset_graph(single_generic_dataset_graph, ["era5"]) is False


def test_projection_node_and_edge_names_expand_for_combined_multi_dataset_graphs() -> None:
    graph_config = {
        "nodes": {
            "era5": {},
            "cerra": {},
            "hidden": {},
        },
    }

    assert (
        projection_node_name("data", dataset_name="era5", graph_or_config=graph_config, dataset_names=["era5", "cerra"])
        == "era5"
    )
    assert (
        projection_node_name(
            "smooth_4x",
            dataset_name="era5",
            graph_or_config=graph_config,
            dataset_names=["era5", "cerra"],
        )
        == "era5_smooth_4x"
    )
    assert projection_edge_name(
        "data",
        "smooth_4x",
        dataset_name="era5",
        graph_or_config=graph_config,
        dataset_names=["era5", "cerra"],
    ) == ("era5", "to", "era5_smooth_4x")


def test_residual_projection_helpers_resolve_custom_and_default_truncation_names(
    fused_graph: HeteroData,
) -> None:
    projection_config = {"truncation": {"node_name": "truncation", "relation_name": "projects_to"}}
    truncation_config = projection_config["truncation"]

    assert residual_projection_truncation_node_name(None) == "truncation"
    assert residual_projection_truncation_node_name({"truncation_nodes": "legacy"}) == "legacy"
    assert residual_projection_truncation_node_name(projection_config) == "truncation"
    assert residual_projection_truncation_node_name(truncation_config) == "truncation"

    down_edges, up_edges = residual_projection_edge_names(
        dataset_name="era5",
        graph_or_config=fused_graph,
        dataset_names=["era5", "cerra"],
        truncation_projection_config=truncation_config,
    )

    assert down_edges == ("era5", "projects_to", "era5_truncation")
    assert up_edges == ("era5_truncation", "projects_to", "era5")


def test_multiscale_loss_matrices_graph_builds_graph_entries_from_smoothers() -> None:
    graph_config = {
        "nodes": {
            "era5": {},
            "cerra": {},
            "hidden": {},
        },
    }
    projection_config = OmegaConf.create(
        {
            "smoothers": {
                "smooth_8x": {
                    "edge_weight_attribute": "weight_8x",
                },
                "smooth_4x": {
                    "relation_name": "maps_to",
                    "row_normalize": True,
                    "src_node_weight_attribute": "cell_area",
                },
            },
        },
    )

    matrices = multiscale_loss_matrices_graph(
        projection_config,
        dataset_name="era5",
        graph_or_config=graph_config,
        dataset_names=["era5", "cerra"],
    )

    assert matrices == [
        {
            "edges_name": ["era5_smooth_4x", "maps_to", "era5_smooth_4x"],
            "edge_weight_attribute": "gauss_weight",
            "src_node_weight_attribute": "cell_area",
            "row_normalize": True,
        },
        {
            "edges_name": ["era5_smooth_8x", "to", "era5_smooth_8x"],
            "edge_weight_attribute": "weight_8x",
        },
        None,
    ]


def test_projection_names_can_prefer_existing_graph_names(fused_graph: HeteroData) -> None:
    assert (
        projection_node_name(
            "smooth_4x",
            dataset_name="era5",
            graph_or_config=fused_graph,
            dataset_names=["era5", "cerra"],
            prefer_existing=True,
        )
        == "era5_smooth_4x"
    )
    assert (
        projection_node_name(
            "era5_smooth_4x",
            dataset_name="era5",
            graph_or_config=fused_graph,
            dataset_names=["era5", "cerra"],
            prefer_existing=True,
        )
        == "era5_smooth_4x"
    )
    assert projection_edge_name(
        "smooth_4x",
        "smooth_4x",
        dataset_name="era5",
        graph_or_config=fused_graph,
        dataset_names=["era5", "cerra"],
        prefer_existing=True,
    ) == ("era5_smooth_4x", "to", "era5_smooth_4x")
