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
from anemoi.models.layers.utils import load_layer_kernels

LOGGER = logging.getLogger(__name__)


class TestTransformerProcessorBlock:
    @given(
        factor_attention_heads=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1, max_value=100),
        num_heads=st.integers(min_value=1, max_value=10),
        activation=st.sampled_from(
            [
                "torch.nn.ReLU",
                "torch.nn.GELU",
            ]
        ),
        window_size=st.integers(min_value=1, max_value=512),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
        softcap=st.floats(min_value=0.0, max_value=1.0),
        qk_norm=st.booleans(),
    )
    @settings(max_examples=10)
    def test_init(
        self, factor_attention_heads, hidden_dim, num_heads, activation, window_size, dropout_p, softcap, qk_norm
    ):
        num_channels = num_heads * factor_attention_heads

        layer_kernels = load_layer_kernels({"Activation": {"_target_": activation}})

        block = TransformerProcessorBlock(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
            attention_implementation="scaled_dot_product_attention",
            softcap=softcap,
            qk_norm=qk_norm,
        )
        assert isinstance(block, TransformerProcessorBlock)

        assert isinstance(block.layer_norm_attention, nn.LayerNorm)
        assert isinstance(block.layer_norm_mlp, nn.LayerNorm)
        assert isinstance(block.mlp, nn.Sequential)
        assert isinstance(block.attention, MultiHeadSelfAttention)
        assert block.attention.qk_norm == qk_norm

    @given(
        factor_attention_heads=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1, max_value=100),
        num_heads=st.integers(min_value=1, max_value=10),
        activation=st.sampled_from(
            [
                "torch.nn.ReLU",
                "torch.nn.GELU",
                "anemoi.models.layers.activations.GLU",
                "anemoi.models.layers.activations.SwiGLU",
            ]
        ),
        window_size=st.integers(min_value=1, max_value=512),
        shapes=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
        batch_size=st.integers(min_value=1, max_value=40),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
        softcap=st.floats(min_value=0.0, max_value=1.0),
        qk_norm=st.booleans(),
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
        qk_norm,
    ):
        num_channels = num_heads * factor_attention_heads

        kwargs = dict()
        if "GLU" in activation:
            kwargs["dim"] = hidden_dim
        layer_kernels = load_layer_kernels({"Activation": {"_target_": activation, **kwargs}})

        block = TransformerProcessorBlock(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
            attention_implementation="scaled_dot_product_attention",
            softcap=softcap,
            qk_norm=qk_norm,
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
        activation=st.sampled_from(
            [
                "torch.nn.ReLU",
                "torch.nn.GELU",
                "anemoi.models.layers.activations.GLU",
                "anemoi.models.layers.activations.SwiGLU",
            ]
        ),
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
        kwargs = dict()
        if "GLU" in activation:
            kwargs["dim"] = in_channels
        layer_kernels = load_layer_kernels({"Activation": {"_target_": activation, **kwargs}})

        block = GraphConvProcessorBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_kernels=layer_kernels,
            mlp_extra_layers=mlp_extra_layers,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
        )

        assert isinstance(block, GraphConvProcessorBlock)
        assert isinstance(block.node_mlp, MLP)
        assert isinstance(block.conv, GraphConv)

        assert block.update_src_nodes == update_src_nodes
        assert block.num_chunks == num_chunks
