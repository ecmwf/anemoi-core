# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.chunk import TransformerProcessorChunk
from anemoi.models.layers.utils import load_layer_kernels


class TestTransformerProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels = 512
        num_layers = 3
        num_heads: int = 16
        mlp_hidden_ratio: int = 4
        window_size: int = 13
        dropout_p: float = 0.1
        layer_kernels = load_layer_kernels()
        attention_implementation = "scaled_dot_product_attention"
        qk_norm = True

        # num_heads must be evenly divisible by num_channels for MHSA
        return (
            num_channels,
            num_layers,
            layer_kernels,
            num_heads,
            mlp_hidden_ratio,
            window_size,
            dropout_p,
            attention_implementation,
            qk_norm,
        )

    @pytest.fixture
    def processor_chunk(self, init):
        (
            num_channels,
            num_layers,
            layer_kernels,
            num_heads,
            mlp_hidden_ratio,
            window_size,
            dropout_p,
            attention_implementation,
            qk_norm,
        ) = init
        return TransformerProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            layer_kernels=layer_kernels,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            qk_norm=qk_norm,
            window_size=window_size,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
        )

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, TransformerProcessorBlock) for block in processor_chunk.blocks)
