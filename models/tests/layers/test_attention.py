# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import math

import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import settings

from anemoi.models.layers.attention import MultiHeadSelfAttention


@given(
    num_heads=st.sampled_from([1, 2, 4, 8, 16]),
    embed_dim_multiplier=st.sampled_from([16, 32, 64]),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
    softcap=st.floats(min_value=0.0, max_value=1.0),
    attention_implementation=st.sampled_from(["scaled_dot_product_attention", "flex_attention"]),
)
def test_multi_head_self_attention_init(num_heads, embed_dim_multiplier, dropout_p, softcap, attention_implementation):
    embed_dim = (
        num_heads * embed_dim_multiplier
    )  # TODO: Make assert in MHSA to check if embed_dim is divisible by num_heads
    mhsa = MultiHeadSelfAttention(
        num_heads, embed_dim, dropout_p=dropout_p, attention_implementation=attention_implementation, softcap=softcap
    )

    assert isinstance(mhsa, nn.Module)
    assert mhsa.num_heads == num_heads
    assert mhsa.embed_dim == embed_dim
    assert mhsa.head_dim == embed_dim // num_heads
    assert dropout_p == mhsa.dropout_p


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
    dropout_p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None)
def test_multi_head_self_attention_forward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(
        num_heads, embed_dim, dropout_p=dropout_p, attention_implementation="scaled_dot_product_attention"
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
def test_multi_head_self_attention_backward_sdpa(batch_size, num_heads, embed_dim_multiplier, dropout_p):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(
        num_heads, embed_dim, dropout_p=dropout_p, attention_implementation="scaled_dot_product_attention"
    )

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    # Dummy loss
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
@pytest.mark.gpu
@given(
    batch_size=st.sampled_from([2]),
    num_heads=st.sampled_from([1, 2, 4, 8, 16]),
    embed_dim_multiplier=st.sampled_from([16, 32, 64]),
    window_size=st.sampled_from([None, 0, 4, 16]),
)
@settings(max_examples=1, deadline=None)
def test_multi_head_self_attention_forward_flex(batch_size, num_heads, embed_dim_multiplier, window_size):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(
        num_heads, embed_dim, window_size=window_size, dropout_p=0.0, attention_implementation="flex_attention"
    ).cuda()

    x = torch.randn(batch_size * 2, embed_dim).cuda()
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
@pytest.mark.gpu
@given(
    batch_size=st.sampled_from([2]),
    num_heads=st.sampled_from([1, 2, 4, 8, 16]),
    embed_dim_multiplier=st.sampled_from([16, 32, 64]),
    window_size=st.sampled_from([None, 0, 4, 16]),
)
@settings(deadline=None)
def test_multi_head_self_attention_backward_flex(batch_size, num_heads, embed_dim_multiplier, window_size):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(
        num_heads, embed_dim, window_size=window_size, attention_implementation="flex_attention"
    ).cuda()

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True).cuda()
    x.retain_grad()
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    # Dummy loss
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
@pytest.mark.gpu
@given(
    num_heads=st.integers(min_value=1, max_value=1),
    embed_dim=st.one_of(
        st.integers(max_value=15),  # Invalid: less than 16
        st.integers(min_value=16).filter(lambda x: not math.log2(x).is_integer()),  # Invalid: not a power of 2
    ),
    window_size=st.integers(min_value=2, max_value=16),
)
@settings(deadline=None)
def test_invalid_embed_dim_raises_assertion_flex(num_heads, embed_dim, window_size):
    with pytest.raises(AssertionError, match="Embedding dimension"):
        MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dropout_p=0.0,
            attention_implementation="flex_attention",
        ).cuda()
