# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

lats = [-0.15, 0, 0.15]
lons = [0, 0.25, 0.5, 0.75]


class MockZarrDataset:
    """Mock Zarr dataset with latitudes and longitudes attributes."""

    def __init__(self, latitudes, longitudes, grids=None):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.num_nodes = len(latitudes)
        if grids is not None:
            self.grids = grids


@pytest.fixture
def mock_zarr_dataset() -> MockZarrDataset:
    """Mock zarr dataset with nodes."""
    coords = 2 * torch.pi * np.array([[lat, lon] for lat in lats for lon in lons])
    return MockZarrDataset(latitudes=coords[:, 0], longitudes=coords[:, 1])


@pytest.fixture
def mock_zarr_dataset_cutout() -> MockZarrDataset:
    """Mock zarr dataset with nodes."""
    coords = 2 * torch.pi * np.array([[lat, lon] for lat in lats for lon in lons])
    grids = int(0.3 * len(coords)), int(0.7 * len(coords))
    return MockZarrDataset(latitudes=coords[:, 0], longitudes=coords[:, 1], grids=grids)


@pytest.fixture
def mock_grids_path(tmp_path) -> tuple[str, int]:
    """Mock grid_definition_path with files for 3 resolutions."""
    num_nodes = len(lats) * len(lons)
    for resolution in ["o16", "o48", "5km5"]:
        file_path = tmp_path / f"grid-{resolution}.npz"
        np.savez(file_path, latitudes=np.random.rand(num_nodes), longitudes=np.random.rand(num_nodes))
    return str(tmp_path), num_nodes


@pytest.fixture
def graph_with_nodes() -> HeteroData:
    """Graph with 12 nodes over the globe, stored in \"test_nodes\"."""
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    graph["test_nodes"].mask = torch.tensor([True] * len(coords))
    return graph


@pytest.fixture
def graph_with_isolated_nodes() -> HeteroData:
    graph = HeteroData()
    graph["test_nodes"].x = torch.tensor([[1], [2], [3], [4], [5], [6]])
    graph["test_nodes"]["mask_attr"] = torch.tensor([[1], [1], [1], [0], [0], [0]], dtype=torch.bool)
    graph["test_nodes", "to", "test_nodes"].edge_index = torch.tensor([[2, 3, 4], [1, 2, 3]])
    return graph


@pytest.fixture
def graph_nodes_and_edges() -> HeteroData:
    """Graph with 1 set of nodes and edges."""
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    graph["test_nodes"].mask = torch.tensor([True] * len(coords))
    graph[("test_nodes", "to", "test_nodes")].edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    return graph


@pytest.fixture
def config_file(tmp_path) -> tuple[str, str]:
    """Mock grid_definition_path with files for 3 resolutions."""
    cfg = {
        "nodes": {
            "test_nodes": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.NPZFileNodes",
                    "grid_definition_path": str(tmp_path),
                    "resolution": "o16",
                },
            },
        },
        "edges": [
            {
                "source_name": "test_nodes",
                "target_name": "test_nodes",
                "edge_builders": [
                    {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3},
                ],
                "attributes": {
                    "dist_norm": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength"},
                    "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection"},
                },
            },
        ],
    }
    file_name = "config.yaml"

    with (tmp_path / file_name).open("w") as file:
        yaml.dump(cfg, file)

    return tmp_path, file_name
