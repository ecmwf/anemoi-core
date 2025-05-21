import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.builders.from_file import XArrayNodes


def test_can_load(mock_zarr_dataset_file):
    graph = HeteroData()

    xarray_nodes = XArrayNodes(
        dataset=mock_zarr_dataset_file,
        name="xarray_nodes",
    )

    graph = xarray_nodes.update_graph(graph)

    assert "xarray_nodes" in graph.node_types
    assert graph["xarray_nodes"].num_nodes == 25


def test_throws_error_with_invalid_lat(mock_zarr_dataset_file):
    graph = HeteroData()

    xarray_nodes = XArrayNodes(
        dataset=mock_zarr_dataset_file,
        name="xarray_nodes",
        lat_name="invalid_latitude",
    )

    with pytest.raises(AssertionError):
        graph = xarray_nodes.update_graph(graph)
