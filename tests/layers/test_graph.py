# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import einops
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph import TrainableTensor


class TestTrainableTensor:
    @pytest.fixture
    def init(self):
        return 10, 5

    @pytest.fixture
    def trainable_tensor(self, init):
        return TrainableTensor(*init)

    @pytest.fixture
    def x(self, init):
        size = init[0]
        return torch.rand(size, size)

    def test_init(self, trainable_tensor):
        assert isinstance(trainable_tensor, TrainableTensor)
        assert isinstance(trainable_tensor.trainable, nn.Parameter)

    def test_forward_backward(self, init, trainable_tensor, x):
        batch_size = 5
        output = trainable_tensor(x, batch_size)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size * x.shape[0], sum(init))

        # Dummy loss
        target = torch.rand(output.shape)
        loss_fn = nn.MSELoss()

        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        for param in trainable_tensor.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_forward_no_trainable(self, init, x):
        tensor_size, trainable_size = init[0], 0

        trainable_tensor = TrainableTensor(tensor_size, trainable_size)
        assert trainable_tensor.trainable is None

        batch_size = 5
        output = trainable_tensor(x, batch_size)
        assert output.shape == (batch_size * x.shape[0], tensor_size + trainable_size)


class TestNamedNodesAttributes:
    """Test suite for the NamedNodesAttributes class.

    This class contains test cases to verify the functionality of the NamedNodesAttributes class,
    including initialization, attribute registration, and forward pass operations.
    """

    nodes_names: list[str] = ["nodes1", "nodes2"]
    ndim: int = 2
    num_trainable_params: int = 8

    @pytest.fixture
    def graph_data(self):
        graph = HeteroData()
        for i, nodes_name in enumerate(TestNamedNodesAttributes.nodes_names):
            graph[nodes_name].x = TestNamedNodesAttributes.get_n_random_coords(10 + 5 ** (i + 1))
        return graph

    @staticmethod
    def get_n_random_coords(n: int) -> torch.Tensor:
        coords = torch.rand(n, TestNamedNodesAttributes.ndim)
        coords[:, 0] = np.pi * (coords[:, 0] - 1 / 2)
        coords[:, 1] = 2 * np.pi * coords[:, 1]
        return coords

    @pytest.fixture
    def nodes_attributes(self, graph_data: HeteroData) -> NamedNodesAttributes:
        return NamedNodesAttributes(TestNamedNodesAttributes.num_trainable_params, graph_data)

    def test_init(self, nodes_attributes):
        assert isinstance(nodes_attributes, NamedNodesAttributes)

        for nodes_name in self.nodes_names:
            assert isinstance(nodes_attributes.num_nodes[nodes_name], int)
            assert (
                nodes_attributes.attr_ndims[nodes_name] - 2 * TestNamedNodesAttributes.ndim
                == TestNamedNodesAttributes.num_trainable_params
            )
            assert isinstance(nodes_attributes.trainable_tensors[nodes_name], TrainableTensor)

    def test_forward(self, nodes_attributes, graph_data):
        batch_size = 3
        for nodes_name in self.nodes_names:
            output = nodes_attributes(nodes_name, batch_size)

            expected_shape = (
                batch_size * graph_data[nodes_name].num_nodes,
                2 * TestNamedNodesAttributes.ndim + TestNamedNodesAttributes.num_trainable_params,
            )
            assert output.shape == expected_shape

            # Check if the first part of the output matches the sin-cos transformed coordinates
            latlons = getattr(nodes_attributes, f"latlons_{nodes_name}")
            repeated_latlons = einops.repeat(latlons, "n f -> (b n) f", b=batch_size)
            assert torch.allclose(output[:, : 2 * TestNamedNodesAttributes.ndim], repeated_latlons)

            # Check if the last part of the output is trainable (requires grad)
            assert output[:, 2 * TestNamedNodesAttributes.ndim :].requires_grad

    def test_forward_no_trainable(self, graph_data):
        no_trainable_attributes = NamedNodesAttributes(0, graph_data)
        batch_size = 2

        for nodes_name in self.nodes_names:
            output = no_trainable_attributes(nodes_name, batch_size)

            expected_shape = batch_size * graph_data[nodes_name].num_nodes, 2 * TestNamedNodesAttributes.ndim
            assert output.shape == expected_shape

            # Check if the output exactly matches the sin-cos transformed coordinates
            latlons = getattr(no_trainable_attributes, f"latlons_{nodes_name}")
            repeated_latlons = einops.repeat(latlons, "n f -> (b n) f", b=batch_size)
            assert torch.allclose(output, repeated_latlons)
