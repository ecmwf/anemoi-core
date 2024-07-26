# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.processor import GraphTransformerProcessor


class TestGraphTransformerProcessor:
    """Test the GraphTransformerProcessor class."""

    NUM_NODES: int = 100
    NUM_EDGES: int = 200

    @pytest.fixture
    def fake_graph(self) -> tuple[HeteroData, int]:
        graph = HeteroData()
        graph["nodes"].x = torch.rand((self.NUM_NODES, 2))
        graph[("nodes", "to", "nodes")].edge_index = torch.randint(0, self.NUM_NODES, (2, self.NUM_EDGES))
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 3))
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 4))
        return graph

    @pytest.fixture
    def graphtransformer_init(self, fake_graph: HeteroData):
        num_layers = 2
        num_channels = 128
        num_chunks = 2
        num_heads = 16
        mlp_hidden_ratio = 4
        activation = "GELU"
        cpu_offload = False
        sub_graph = fake_graph[("nodes", "to", "nodes")]
        edge_attributes = ["edge_attr1", "edge_attr2"]
        src_grid_size = 0
        dst_grid_size = 0
        trainable_size = 6
        return (
            num_layers,
            num_channels,
            num_chunks,
            num_heads,
            mlp_hidden_ratio,
            activation,
            cpu_offload,
            sub_graph,
            edge_attributes,
            src_grid_size,
            dst_grid_size,
            trainable_size,
        )

    @pytest.fixture
    def graphtransformer_processor(self, graphtransformer_init):
        (
            num_layers,
            num_channels,
            num_chunks,
            num_heads,
            mlp_hidden_ratio,
            activation,
            cpu_offload,
            sub_graph,
            edge_attributes,
            src_grid_size,
            dst_grid_size,
            trainable_size,
        ) = graphtransformer_init
        return GraphTransformerProcessor(
            num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            cpu_offload=cpu_offload,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            trainable_size=trainable_size,
        )

    def test_graphtransformer_processor_init(self, graphtransformer_processor, graphtransformer_init):
        (
            num_layers,
            num_channels,
            num_chunks,
            _num_heads,
            _mlp_hidden_ratio,
            _activation,
            _cpu_offload,
            _sub_graph,
            _edge_attributes,
            _src_grid_size,
            _dst_grid_size,
            _trainable_size,
        ) = graphtransformer_init
        assert graphtransformer_processor.num_chunks == num_chunks
        assert graphtransformer_processor.num_channels == num_channels
        assert graphtransformer_processor.chunk_size == num_layers // num_chunks
        assert isinstance(graphtransformer_processor.trainable, TrainableTensor)

    def test_forward(self, graphtransformer_processor, graphtransformer_init):
        batch_size = 1
        (
            _num_layers,
            num_channels,
            _num_chunks,
            _num_heads,
            _mlp_hidden_ratio,
            _activation,
            _cpu_offload,
            _sub_graph,
            _edge_attributes,
            _src_grid_size,
            _dst_grid_size,
            trainable_size,
        ) = graphtransformer_init
        x = torch.rand((self.NUM_EDGES, num_channels))
        shard_shapes = [list(x.shape)]

        # Run forward pass of processor
        output = graphtransformer_processor.forward(x, batch_size, shard_shapes)
        assert output.shape == (self.NUM_EDGES, num_channels)

        # Generate dummy target and loss function
        loss_fn = torch.nn.MSELoss()
        target = torch.rand((self.NUM_EDGES, num_channels))
        loss = loss_fn(output, target)

        # Check loss
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients of trainable tensor
        assert graphtransformer_processor.trainable.trainable.grad.shape == (
            self.NUM_EDGES,
            trainable_size,
        )

        # Check gradients of processor
        for param in graphtransformer_processor.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
