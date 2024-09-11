# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk


class TestGraphTransformerProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels = 10
        num_layers = 3
        num_heads: int = 16
        mlp_hidden_ratio: int = 4
        activation: str = "GELU"
        edge_dim: int = 32
        return (
            num_channels,
            num_layers,
            num_heads,
            mlp_hidden_ratio,
            activation,
            edge_dim,
        )

    @pytest.fixture
    def processor_chunk(self, init):
        num_channels, num_layers, num_heads, mlp_hidden_ratio, activation, edge_dim = init
        return GraphTransformerProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            edge_dim=edge_dim,
        )

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, GraphTransformerProcessorBlock) for block in processor_chunk.blocks)
