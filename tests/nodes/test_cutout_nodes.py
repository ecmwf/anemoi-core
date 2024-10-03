import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders import from_file


def test_init(mocker, mock_zarr_dataset_cutout):
    """Test CutOutZarrDatasetNodes initialization."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.CutOutZarrDatasetNodes(
        forcing_dataset="global.zarr", lam_dataset="lam.zarr", name="test_nodes"
    )

    assert isinstance(node_builder, from_file.BaseNodeBuilder)
    assert isinstance(node_builder, from_file.CutOutZarrDatasetNodes)


def test_fail_init():
    """Test CutOutZarrDatasetNodes initialization with invalid resolution."""
    with pytest.raises(TypeError):
        from_file.CutOutZarrDatasetNodes("global_dataset.zarr", name="test_nodes")


def test_register_nodes(mocker, mock_zarr_dataset_cutout):
    """Test CutOutZarrDatasetNodes register correctly the nodes."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.CutOutZarrDatasetNodes(
        forcing_dataset="global.zarr", lam_dataset="lam.zarr", name="test_nodes"
    )
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (node_builder.dataset.num_nodes, 2)
    assert graph["test_nodes"].node_type == "CutOutZarrDatasetNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights])
def test_register_attributes(mocker, mock_zarr_dataset_cutout, graph_with_nodes: HeteroData, attr_class):
    """Test CutOutZarrDatasetNodes register correctly the weights."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.CutOutZarrDatasetNodes(
        forcing_dataset="global.zarr", lam_dataset="lam.zarr", name="test_nodes"
    )
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
