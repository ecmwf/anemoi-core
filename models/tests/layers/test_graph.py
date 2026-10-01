# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import einops
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import NodeTrainableParameters
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
