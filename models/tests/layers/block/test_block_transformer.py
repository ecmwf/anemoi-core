# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from torch import nn

from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.block import MLP
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.conv import GraphConv

LOGGER = logging.getLogger(__name__)


class TestTransformerProcessorBlock:
    @given(
        factor_attention_heads=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1, max_value=100),
        num_heads=st.integers(min_value=1, max_value=10),
        activation=st.sampled_from(["ReLU", "GELU", "Tanh"]),
        window_size=st.integers(min_value=1, max_value=512),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
        softcap=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_init(self, factor_attention_heads, hidden_dim, num_heads, activation, window_size, dropout_p, softcap):
        num_channels = num_heads * factor_attention_heads
        block = TransformerProcessorBlock(
            num_channels,
            hidden_dim,
            num_heads,
            activation,
            window_size,
            dropout_p=dropout_p,
            attention_implementation="scaled_dot_product_attention",
            softcap=softcap,
        )
        assert isinstance(block, TransformerProcessorBlock)

        assert isinstance(block.layer_norm1, nn.LayerNorm)
        assert isinstance(block.layer_norm2, nn.LayerNorm)
        assert isinstance(block.mlp, nn.Sequential)
        assert isinstance(block.attention, MultiHeadSelfAttention)

    @given(
        factor_attention_heads=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1, max_value=100),
        num_heads=st.integers(min_value=1, max_value=10),
        activation=st.sampled_from(["ReLU", "GELU", "Tanh"]),
        window_size=st.integers(min_value=1, max_value=512),
        shapes=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
        batch_size=st.integers(min_value=1, max_value=40),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
        softcap=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_forward_output(
        self,
        factor_attention_heads,
        hidden_dim,
        num_heads,
        activation,
        window_size,
        shapes,
        batch_size,
        dropout_p,
        softcap,
    ):
        num_channels = num_heads * factor_attention_heads
        block = TransformerProcessorBlock(
            num_channels,
            hidden_dim,
            num_heads,
            activation,
            window_size,
            dropout_p=dropout_p,
            attention_implementation="scaled_dot_product_attention",
            softcap=softcap,
        )

        x = torch.randn((batch_size, num_channels))  # .to(torch.float16, non_blocking=True)
        output = block.forward(x, shapes, batch_size)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, num_channels)


class TestGraphConvProcessorBlock:
    @given(
        in_channels=st.integers(min_value=1, max_value=100),
        out_channels=st.integers(min_value=1, max_value=100),
        mlp_extra_layers=st.integers(min_value=1, max_value=5),
        activation=st.sampled_from(["ReLU", "GELU", "Tanh"]),
        update_src_nodes=st.booleans(),
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_init(
        self,
        in_channels,
        out_channels,
        mlp_extra_layers,
        activation,
        update_src_nodes,
        num_chunks,
    ):
        block = GraphConvProcessorBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
        )

        assert isinstance(block, GraphConvProcessorBlock)
        assert isinstance(block.node_mlp, MLP)
        assert isinstance(block.conv, GraphConv)

        assert block.update_src_nodes == update_src_nodes
        assert block.num_chunks == num_chunks
