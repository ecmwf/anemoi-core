# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class GNNProcessorInit:
    num_channels: int = 128
    num_layers: int = 2
    num_chunks: int = 2
    mlp_extra_layers: int = 0
    trainable_size: int = 8
    src_grid_size: int = 0
    dst_grid_size: int = 0
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class TestGNNProcessor:
    """Test the GNNProcessor class."""

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
    def graphconv_init(self):
        return GNNProcessorInit()

    @pytest.fixture
    def graphconv_processor(self, graphconv_init, fake_graph):
        return GNNProcessor(
            **asdict(graphconv_init),
            sub_graph=fake_graph[("nodes", "to", "nodes")],
            sub_graph_edge_attributes=["edge_attr1", "edge_attr2"],
        )

    def test_graphconv_processor_init(self, graphconv_processor, graphconv_init):
        assert graphconv_processor.num_chunks == graphconv_init.num_chunks
        assert graphconv_processor.num_channels == graphconv_init.num_channels
        assert graphconv_processor.chunk_size == graphconv_init.num_layers // graphconv_init.num_chunks
        assert isinstance(graphconv_processor.trainable, TrainableTensor)

    def test_forward(self, graphconv_processor, graphconv_init):
        batch_size = 1
        x = torch.rand((self.NUM_EDGES, graphconv_init.num_channels))
        shard_shapes = [list(x.shape)]

        # Run forward pass of processor
        output = graphconv_processor.forward(x, batch_size, shard_shapes)
        assert output.shape == (self.NUM_EDGES, graphconv_init.num_channels)

        # Generate dummy target and loss function
        loss_fn = torch.nn.MSELoss()
        target = torch.rand((self.NUM_EDGES, graphconv_init.num_channels))
        loss = loss_fn(output, target)

        # Check loss
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients of trainable tensor
        assert graphconv_processor.trainable.trainable.grad.shape == (self.NUM_EDGES, graphconv_init.trainable_size)

        # Check gradients of processor
        for param in graphconv_processor.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
