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
import xarray as xr
import yaml
from torch_geometric.data import HeteroData

lats = [-0.15, 0, 0.15]
lons = [0, 0.25, 0.5, 0.75]


class MockAnemoiDataset:
    """Mock Zarr dataset with latitudes and longitudes attributes."""

    def __init__(self, latitudes, longitudes, grids=None):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.num_nodes = len(latitudes)
        self.grids = grids if grids is not None else (self.num_nodes,)


@pytest.fixture
def mock_anemoi_dataset() -> MockAnemoiDataset:
    """Mock zarr dataset with nodes."""
    coords = 2 * torch.pi * np.array([[lat, lon] for lat in lats for lon in lons])
    return MockAnemoiDataset(latitudes=coords[:, 0], longitudes=coords[:, 1])


@pytest.fixture
def mock_zarr_dataset_file(tmpdir) -> str:
    lat_vals = np.linspace(-90, 90, 5)
    lon_vals = np.linspace(0, 360, 5, endpoint=False)
    lat, lon = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    data = np.random.randn(5, 5)

    ds = xr.Dataset(
        {
            "variable": (
                ["lat", "lon"],
                data,
            ),
        },
        coords={
            "lat": (["x", "y"], lat),
            "lon": (["x", "y"], lon),
        },
    )

    fn = tmpdir / "tmp.zarr"
    ds.to_zarr(fn, mode="w", consolidated=True)

    return fn


@pytest.fixture
def mock_anemoi_dataset_cutout() -> MockAnemoiDataset:
    """Mock zarr dataset with nodes."""
    coords = 2 * torch.pi * np.array([[lat, lon] for lat in lats for lon in lons])
    grids = int(0.3 * len(coords)), len(coords) - int(0.3 * len(coords))
    return MockAnemoiDataset(latitudes=coords[:, 0], longitudes=coords[:, 1], grids=grids)


@pytest.fixture
def mock_grids_path(tmp_path) -> tuple[str, int]:
    """Mock grid_definition_path with files for 3 resolutions."""
    num_nodes = len(lats) * len(lons)
    for resolution in ["o16", "o48", "5km5"]:
        file_path = tmp_path / f"grid-{resolution}.npz"
        np.savez(
            file_path,
            latitudes=np.random.rand(num_nodes),
            longitudes=np.random.rand(num_nodes),
        )
    return str(tmp_path), num_nodes


@pytest.fixture
def graph_with_nodes() -> HeteroData:
    """Graph with 12 nodes over the globe, stored in \"test_nodes\"."""
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    graph["test_nodes"].mask = torch.tensor([True] * len(coords)).unsqueeze(-1)
    graph["test_nodes"].mask2 = torch.tensor([True] * (len(coords) - 2) + [False] * 2).unsqueeze(-1)
    graph["test_nodes"].interior_mask = torch.tensor(
        [
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    ).unsqueeze(-1)
    graph["test_nodes"]["_grid_reference_distance"] = 0.75
    return graph


@pytest.fixture
def graph_with_rectilinear_nodes() -> HeteroData:
    graph = HeteroData()
    num_lons, num_lats = 10, 10
    lat_grid, lon_grid = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, num_lats), np.linspace(0, 2 * np.pi, num_lons))
    coords = torch.tensor(np.array([lat_grid.ravel(), lon_grid.ravel()]).T)
    graph["test_nodes"].x = coords
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
    graph["test_nodes"].mask = torch.tensor([True] * len(coords)).unsqueeze(-1)
    graph[("test_nodes", "to", "test_nodes")].edge_index = torch.tensor([[3, 1, 2, 0], [2, 0, 1, 3]])
    graph[("test_nodes", "to", "test_nodes")].edge_attr = (
        10 * graph[("test_nodes", "to", "test_nodes")].edge_index[0][:, None]
    )
    return graph


@pytest.fixture
def graph_long_and_short_edges() -> HeteroData:
    """Graph with a pair of short (800km) and a pair of long (20000km) edges."""
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor([[-0.01, 0], [0.01, 0], [-0.01, 0.5], [0.01, 0.5]])
    graph["test_nodes"]["southern_hemisphere_mask"] = torch.tensor([[1], [0], [1], [0]], dtype=torch.bool)
    graph["test_nodes", "to", "test_nodes"].edge_index = torch.tensor([[0, 0, 1, 3], [1, 3, 2, 2]])
    return graph


@pytest.fixture
def config_file(tmp_path) -> tuple[str, str]:
    """Mock grid_definition_path with files for 3 resolutions."""
    cfg = {
        "nodes": {
            "test_nodes": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.NPZFileNodes",
                    "npz_file": str(tmp_path) + "/grid-o16.npz",
                },
            },
        },
        "edges": [
            {
                "source_name": "test_nodes",
                "target_name": "test_nodes",
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.KNNEdges",
                        "num_nearest_neighbours": 3,
                    },
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
