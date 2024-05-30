# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.chunk import TransformerProcessorChunk


class TestGraphTransformerProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels = 512
        num_layers = 3
        num_heads: int = 16
        mlp_hidden_ratio: int = 4
        activation: str = "GELU"
        window_size: int = 13

        # num_heads must be evenly divisible by num_channels for MHSA
        return (
            num_channels,
            num_layers,
            num_heads,
            mlp_hidden_ratio,
            activation,
            window_size,
        )

    @pytest.fixture
    def processor_chunk(self, init):
        (
            num_channels,
            num_layers,
            num_heads,
            mlp_hidden_ratio,
            activation,
            window_size,
        ) = init
        return TransformerProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            window_size=window_size,
        )

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, TransformerProcessorBlock) for block in processor_chunk.blocks)
