# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import settings

from anemoi.models.layers.attention import MultiHeadSelfAttention


@given(
    num_heads=st.integers(min_value=1, max_value=50),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
)
def test_multi_head_self_attention_init(num_heads, embed_dim_multiplier):
    embed_dim = (
        num_heads * embed_dim_multiplier
    )  # TODO: Make assert in MHSA to check if embed_dim is divisible by num_heads
    mhsa = MultiHeadSelfAttention(num_heads, embed_dim)

    assert isinstance(mhsa, nn.Module)
    assert mhsa.num_heads == num_heads
    assert mhsa.embed_dim == embed_dim
    assert mhsa.head_dim == embed_dim // num_heads


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_multi_head_self_attention_forward(batch_size, num_heads, embed_dim_multiplier):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(num_heads, embed_dim)

    x = torch.randn(batch_size * 2, embed_dim)
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    assert output.shape == x.shape


@pytest.mark.gpu
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_heads=st.integers(min_value=1, max_value=20),
    embed_dim_multiplier=st.integers(min_value=1, max_value=10),
)
def test_multi_head_self_attention_backward(batch_size, num_heads, embed_dim_multiplier):
    embed_dim = num_heads * embed_dim_multiplier
    mhsa = MultiHeadSelfAttention(num_heads, embed_dim)

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shapes = [list(x.shape)]
    output = mhsa.forward(x, shapes, batch_size)

    # Dummy loss
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
