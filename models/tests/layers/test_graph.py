# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import io

import einops
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import NodeTrainableParameters
from anemoi.models.layers.graph import StaticGraphGeometry
from anemoi.models.layers.graph import TrainableTensor


class TestTrainableTensor:
    @pytest.fixture
    def init(self):
        return 10, 5

    @pytest.fixture
    def trainable_tensor(self, init):
        return TrainableTensor(*init)

    def test_init(self, trainable_tensor):
        assert isinstance(trainable_tensor, TrainableTensor)
        assert isinstance(trainable_tensor.trainable, nn.Parameter)

    def test_forward_backward(self, init, trainable_tensor):
        batch_size = 5
        output = trainable_tensor(batch_size)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size * init[0], init[1])

        # Dummy loss
        target = torch.rand(output.shape)
        loss_fn = nn.MSELoss()

        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        for param in trainable_tensor.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_forward_no_trainable(self, init):
        trainable_tensor = TrainableTensor(init[0], 0)
        assert trainable_tensor.trainable is None
        assert trainable_tensor(batch_size=5) is None


class TestNodeTrainableParameters:
    """Test suite for the NodeTrainableParameters class."""

    nodes_names: list[str] = ["nodes1", "nodes2"]
    num_trainable_params: dict[str, int] = {"nodes1": 3, "nodes2": 5, "nodes1tonodes2": 4}

    @pytest.fixture
    def graph_data(self):
        graph = HeteroData()
        for i, nodes_name in enumerate(TestNodeTrainableParameters.nodes_names):
            graph[nodes_name].x = torch.rand(10 + 5 ** (i + 1), 2)
        return graph

    @pytest.fixture
    def node_parameters(self, graph_data: HeteroData) -> NodeTrainableParameters:
        return NodeTrainableParameters(TestNodeTrainableParameters.num_trainable_params, graph_data)

    def test_init(self, node_parameters):
        assert isinstance(node_parameters, NodeTrainableParameters)

        for nodes_name in self.nodes_names:
            assert node_parameters.num_trainable_parameters[nodes_name] == self.num_trainable_params[nodes_name]
            assert isinstance(node_parameters.trainable_tensors[nodes_name], TrainableTensor)

        # Only graph node types get trainable tensors
        assert "nodes1tonodes2" not in node_parameters.trainable_tensors
        # Missing node types default to 0 trainable parameters
        assert node_parameters.num_trainable_parameters["unknown"] == 0

    def test_contains(self, node_parameters, graph_data):
        for nodes_name in self.nodes_names:
            assert nodes_name in node_parameters
        assert "unknown" not in node_parameters

        no_trainable = NodeTrainableParameters({}, graph_data)
        for nodes_name in self.nodes_names:
            assert nodes_name not in no_trainable

    def test_forward(self, node_parameters, graph_data):
        batch_size = 3
        for nodes_name in self.nodes_names:
            output = node_parameters(nodes_name, batch_size)

            expected_shape = (
                batch_size * graph_data[nodes_name].num_nodes,
                self.num_trainable_params[nodes_name],
            )
            assert output.shape == expected_shape
            assert output.requires_grad

            trainable = node_parameters.trainable_tensors[nodes_name].trainable
            assert torch.equal(output, einops.repeat(trainable, "n f -> (b n) f", b=batch_size))

    def test_forward_unknown_nodes(self, node_parameters):
        assert node_parameters("unknown", batch_size=2) is None

    def test_forward_no_trainable(self, graph_data):
        no_trainable = NodeTrainableParameters({}, graph_data)
        for nodes_name in self.nodes_names:
            assert no_trainable(nodes_name, batch_size=2) is None


def _geometry_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.rand(5, 2)
    graph["hidden"].x = torch.rand(3, 2)
    graph["data", "to", "hidden"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])
    graph["hidden", "to", "data"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    return graph


def test_static_graph_geometry_keeps_counts_and_the_named_coordinates() -> None:
    graph = _geometry_graph()
    geometry = StaticGraphGeometry(graph, coordinate_node_names=["hidden"])

    assert geometry.num_nodes == {"data": 5, "hidden": 3}
    assert geometry.num_edges == {("data", "to", "hidden"): 4, ("hidden", "to", "data"): 3}
    torch.testing.assert_close(geometry.coordinates("hidden"), graph["hidden"].x)
    with pytest.raises(KeyError, match="No coordinates kept for node set 'data'"):
        geometry.coordinates("data")


def test_static_graph_geometry_is_pickled_with_the_model_but_not_in_its_state_dict() -> None:
    """Carried by a pickled model (inference checkpoints), absent from its state_dict (training checkpoints)."""
    graph = _geometry_graph()
    model = nn.Linear(2, 2)
    model.geometry = StaticGraphGeometry(graph, coordinate_node_names=["hidden"])

    assert set(model.state_dict()) == {"weight", "bias"}
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)

    assert restored.geometry.num_edges == model.geometry.num_edges
    torch.testing.assert_close(restored.geometry.coordinates("hidden"), graph["hidden"].x)
    # A copy, not a view of the graph: changing the graph does not change the geometry.
    graph["hidden"].x.zero_()
    assert not torch.equal(model.geometry.coordinates("hidden"), graph["hidden"].x)


def test_static_graph_geometry_keeps_its_dtype_when_the_model_is_cast() -> None:
    """Geometry, like the batch coordinates it is combined with: a model cast does not change it."""
    model = nn.Linear(2, 2)
    model.geometry = StaticGraphGeometry(_geometry_graph(), coordinate_node_names=["hidden"])

    model.to(torch.float64)

    assert model.geometry.coordinates("hidden").dtype == torch.float32
