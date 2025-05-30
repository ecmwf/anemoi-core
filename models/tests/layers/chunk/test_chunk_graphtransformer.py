# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk
from anemoi.models.layers.utils import load_layer_kernels


class TestGraphTransformerProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels: int = 10
        num_layers: int = 3
        num_heads: int = 16
        mlp_hidden_ratio: int = 4
        qk_norm = True
        edge_dim: int = 32
        layer_kernels = load_layer_kernels()
        return (
            num_channels,
            num_layers,
            layer_kernels,
            num_heads,
            mlp_hidden_ratio,
            qk_norm,
            edge_dim,
        )

    @pytest.fixture
    def processor_chunk(self, init):
        num_channels, num_layers, layer_kernels, num_heads, mlp_hidden_ratio, qk_norm, edge_dim = init
        return GraphTransformerProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            layer_kernels=layer_kernels,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            qk_norm=qk_norm,
            edge_dim=edge_dim,
        )

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, GraphTransformerProcessorBlock) for block in processor_chunk.blocks)
