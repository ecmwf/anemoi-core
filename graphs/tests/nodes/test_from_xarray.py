import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.builders.from_file import XArrayNodes


def test_update_graph_creates_expected_nodes(mock_zarr_dataset_file):
    graph = HeteroData()

    xarray_nodes = XArrayNodes(
        dataset=mock_zarr_dataset_file,
        name="xarray_nodes",
    )

    graph = xarray_nodes.update_graph(graph)

    assert "xarray_nodes" in graph.node_types
    assert graph["xarray_nodes"].num_nodes == 25


@pytest.mark.parametrize(
    "lat_key, lon_key",
    [
        ("invalid_latitude", "longitude"),
        ("latitude", "invalid_longitude"),
    ],
)
def test_throws_error_with_invalid_lat(mock_zarr_dataset_file, lat_key, lon_key):
    graph = HeteroData()

    xarray_nodes = XArrayNodes(
        dataset=mock_zarr_dataset_file,
        name="xarray_nodes",
        lat_key=lat_key,
        lon_key=lon_key,
    )

    with pytest.raises(AssertionError):
        graph = xarray_nodes.update_graph(graph)
