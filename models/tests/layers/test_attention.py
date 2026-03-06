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


def test_multi_head_self_attention_accepts_triton_attention_alias(layer_kernels, monkeypatch):
    class DummyTritonWrapper(nn.Module):
        def __init__(self):
            super().__init__()

    monkeypatch.setattr("anemoi.models.layers.attention.TritonAttentionWrapper", DummyTritonWrapper)

    mhsa = MultiHeadSelfAttention(
        4,
        64,
        layer_kernels,
        attention_implementation="triton_attention",
    )

    assert isinstance(mhsa.attention, DummyTritonWrapper)
    assert mhsa.attention_implementation == "triton"


def test_multi_head_self_attention_reads_backend_override_at_runtime(layer_kernels, monkeypatch):
    class DummyTritonWrapper(nn.Module):
        def __init__(self):
            super().__init__()

    monkeypatch.setattr("anemoi.models.layers.attention.TritonAttentionWrapper", DummyTritonWrapper)
    monkeypatch.delenv("ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND", raising=False)

    mhsa = MultiHeadSelfAttention(
        4,
        64,
        layer_kernels,
        attention_implementation="scaled_dot_product_attention",
    )
    assert mhsa.attention_implementation == "scaled_dot_product_attention"

    monkeypatch.setenv("ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND", "triton_attention")
    mhsa.set_attention_function()

    assert isinstance(mhsa.attention, DummyTritonWrapper)
    assert mhsa.attention_implementation == "triton"


def test_multi_head_self_attention_builds_backend_when_env_matches_config(layer_kernels, monkeypatch):
    monkeypatch.setenv("ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND", "scaled_dot_product_attention")

    mhsa = MultiHeadSelfAttention(
        4,
        64,
        layer_kernels,
        attention_implementation="scaled_dot_product_attention",
    )

    assert isinstance(mhsa.attention, nn.Module)
    assert mhsa.attention_implementation == "scaled_dot_product_attention"


def test_triton_attention_requires_matching_qkv_shapes():
    from anemoi.models.triton.attention import TritonAttention

    class DummyCtx:
        def save_for_backward(self, *args):
            self.saved_tensors = args

    q = torch.randn(1, 2, 3, 16)
    k = torch.randn(1, 2, 5, 16)
    v = torch.randn(1, 2, 5, 16)

    with pytest.raises(AssertionError, match="share batch, head, and sequence dimensions"):
        TritonAttention.forward(DummyCtx(), q, k, v, False, -1, 0.25)


def test_triton_attention_preserves_zero_window(monkeypatch):
    import anemoi.models.triton.attention as triton_attention_module

    captured = {}

    class DummyKernel:
        def __getitem__(self, _grid):
            def launcher(*args, **kwargs):
                captured.update(kwargs)

            return launcher

    class DummyCtx:
        def save_for_backward(self, *args):
            self.saved_tensors = args

    monkeypatch.setattr(
        triton_attention_module,
        "_system_specific_settings",
        lambda q, k, v, o, _: (None, None, None, None, {}),
    )
    monkeypatch.setattr(triton_attention_module, "_attn_fwd", DummyKernel())
    monkeypatch.setattr(triton_attention_module, "torch_dtype_to_triton", lambda dtype: dtype)

    q = torch.randn(1, 2, 3, 16)
    out = triton_attention_module.TritonAttention.forward(DummyCtx(), q, q, q, False, 0, 0.25)

    assert out.shape == q.shape
    assert captured["WINDOW"] == 0


def test_triton_attention_backward_returns_one_gradient_per_input(monkeypatch):
    import anemoi.models.triton.attention as triton_attention_module

    class DummyKernel:
        def __getitem__(self, _grid):
            def launcher(*args, **kwargs):
                return None

            return launcher

    class DummyCtx:
        def __init__(self, q, k, v, o, m):
            self.saved_tensors = (q, k, v, o, m)
            self.sm_scale = 0.25
            self.causal = False
            self.window = -1

    monkeypatch.setattr(
        triton_attention_module,
        "_system_specific_settings",
        lambda q, k, v, o, _: (None, None, None, None, {}),
    )
    monkeypatch.setattr(triton_attention_module, "_attn_bwd_preprocess", DummyKernel())
    monkeypatch.setattr(triton_attention_module, "_attn_bwd_dkdv", DummyKernel())
    monkeypatch.setattr(triton_attention_module, "_attn_bwd_dq", DummyKernel())
    monkeypatch.setattr(triton_attention_module, "supports_host_descriptor", lambda: True)
    monkeypatch.setattr(triton_attention_module, "torch_dtype_to_triton", lambda dtype: dtype)

    q = torch.randn(1, 2, 3, 16)
    k = torch.randn(1, 2, 3, 16)
    v = torch.randn(1, 2, 3, 16)
    o = torch.randn(1, 2, 3, 16)
    m = torch.randn(1, 2, 128)
    grads = triton_attention_module.TritonAttention.backward(DummyCtx(q, k, v, o, m), torch.randn_like(o))

    assert len(grads) == 6


def test_triton_test_mode_env_controls_autotune_configs(monkeypatch):
    import anemoi.models.triton.attention as triton_attention_module

    monkeypatch.delenv("PYTEST_VERSION", raising=False)

    monkeypatch.setenv("ANEMOI_TRITON_TEST_MODE", "0")
    assert len(triton_attention_module._generate_configs()) > 1
    assert len(triton_attention_module._generate_varlen_configs()) > 1

    monkeypatch.setenv("ANEMOI_TRITON_TEST_MODE", "1")
    # Fix: pytest now uses an explicit ANEMOI_TRITON_TEST_MODE flag, so these helpers must
    # collapse to the single deterministic config that keeps correctness tests fast and stable.
    assert len(triton_attention_module._generate_configs()) == 1
    assert len(triton_attention_module._generate_varlen_configs()) == 1
