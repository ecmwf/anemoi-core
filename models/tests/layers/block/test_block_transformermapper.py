# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.block import TransformerMapperBlock
from anemoi.models.layers.utils import load_layer_kernels


@pytest.fixture
def init():
    num_channels: int = 8
    hidden_dim: int = 256
    num_heads: int = 4
    window_size: int = None
    dropout_p: float = (0.0,)
    qk_norm: bool = (False,)
    attention_implementation: str = "scaled_dot_product_attention"
    layer_kernels = load_layer_kernels()
    return (
        num_channels,
        hidden_dim,
        num_heads,
        window_size,
        layer_kernels,
        dropout_p,
        qk_norm,
        attention_implementation,
    )


@pytest.fixture
def mapper_block(init):
    (
        num_channels,
        hidden_dim,
        num_heads,
        window_size,
        layer_kernels,
        dropout_p,
        qk_norm,
        attention_implementation,
    ) = init

    return TransformerMapperBlock(
        num_channels=num_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        window_size=window_size,
        layer_kernels=layer_kernels,
        dropout_p=dropout_p,
        qk_norm=qk_norm,
        attention_implementation=attention_implementation,
    )


def test_TransformerMapperBlock_init(mapper_block):
    block = mapper_block
    assert isinstance(block, TransformerMapperBlock), "block is not an instance of GraphTransformerMapperBlock"
    assert isinstance(block.layer_norm_attention_src, nn.LayerNorm)
    assert isinstance(block.layer_norm_attention_dst, nn.LayerNorm)
    assert isinstance(block.layer_norm_mlp, nn.LayerNorm)
    assert isinstance(block.mlp, nn.Sequential)
    assert isinstance(block.attention, MultiHeadCrossAttention)


@given(
    factor_attention_heads=st.integers(min_value=1, max_value=10),
    hidden_dim=st.integers(min_value=1, max_value=100),
    num_heads=st.integers(min_value=1, max_value=10),
    window_size=st.integers(min_value=1, max_value=512),
    shapes=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
    batch_size=st.integers(min_value=1, max_value=40),
    dropout_p=st.floats(min_value=0.01, max_value=1.0),
    softcap=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=10)
def test_forward_output(
    factor_attention_heads,
    hidden_dim,
    num_heads,
    window_size,
    shapes,
    batch_size,
    dropout_p,
    softcap,
):
    num_channels = num_heads * factor_attention_heads
    layer_kernels = instantiate(load_layer_kernels(kernel_config={}))
    block = TransformerMapperBlock(
        num_channels=num_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        window_size=window_size,
        dropout_p=dropout_p,
        layer_kernels=layer_kernels,
        attention_implementation="scaled_dot_product_attention",
        softcap=softcap,
    )

    x = torch.randn((batch_size, num_channels))  # .to(torch.float16, non_blocking=True)
    output, _ = block.forward((x, x), shapes, batch_size)
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)
    assert output[1].shape == (batch_size, num_channels)
