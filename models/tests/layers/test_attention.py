# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import settings

from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.utils import load_layer_kernels


@pytest.fixture(scope="session")
def layer_kernels():
    return load_layer_kernels()


@given(
    num_heads=st.sampled_from([1, 2, 4, 8, 16]),
    embed_dim_multiplier=st.sampled_from([16, 32, 64]),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
    softcap=st.floats(min_value=0.0, max_value=1.0),
    attention_module=st.sampled_from([MultiHeadSelfAttention, MultiHeadCrossAttention]),
    attention_implementation=st.sampled_from(["scaled_dot_product_attention"]),
)
def test_multi_head_self_attention_init(
    num_heads, embed_dim_multiplier, dropout_p, softcap, attention_module, attention_implementation, layer_kernels
):
    embed_dim = (
        num_heads * embed_dim_multiplier
    )  # TODO: Make assert in MHSA to check if embed_dim is divisible by num_heads

    mhsa = attention_module(
        num_heads,
        embed_dim,
        layer_kernels,
        qk_norm=True,
        dropout_p=dropout_p,
        attention_implementation=attention_implementation,
        softcap=softcap,
    )

    assert isinstance(mhsa, nn.Module)
    assert mhsa.num_heads == num_heads
    assert mhsa.embed_dim == embed_dim
    assert mhsa.head_dim == embed_dim // num_heads
    assert dropout_p == mhsa.dropout_p
    assert mhsa.q_norm.bias is None
    assert mhsa.k_norm.bias is None


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None)
def test_multi_head_self_attention_forward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p, layer_kernels):
    embed_dim = num_heads * embed_dim_multiplier

    mhsa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )

    x = torch.randn(batch_size * 2, embed_dim)
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    assert output.shape == x.shape


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None)
def test_multi_head_self_attention_backward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p, layer_kernels):
    embed_dim = num_heads * embed_dim_multiplier

    mhsa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    # Dummy loss
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None)
def test_multi_head_cross_attention_forward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p):
    embed_dim = num_heads * embed_dim_multiplier

    layer_kernels = load_layer_kernels(kernel_config={})
    mhsa = MultiHeadCrossAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )

    x = torch.randn(batch_size * 2, embed_dim)
    shapes = [list(x.shape)]
    output = mhsa.forward((x, x), shapes, batch_size)

    assert output.shape == x.shape


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None)
def test_multi_head_cross_attention_backward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p):
    embed_dim = num_heads * embed_dim_multiplier

    layer_kernels = load_layer_kernels(kernel_config={})
    mhsa = MultiHeadCrossAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shapes = [list(x.shape)]
    output = mhsa.forward((x, x), shapes, batch_size)

    # Dummy loss
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
