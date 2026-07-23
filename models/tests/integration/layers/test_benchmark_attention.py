# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark flex attention against SDPA and flash attention."""

from __future__ import annotations

import contextlib
import time

import pytest
import torch

from anemoi.models.layers.attention import FlashAttentionWrapper
from anemoi.models.layers.attention import FlexAttentionWrapper
from anemoi.models.layers.attention import SDPAAttentionWrapper

Z = 1
H = 16
N_CTX = 40320
HEAD_DIM = 32


def _make_inputs(device, dtype):
    torch.manual_seed(0)
    query = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    return query, key, value


def _run_backend(backend, query, key, value, window_size):
    return backend(
        query,
        key,
        value,
        batch_size=1,
        causal=False,
        window_size=window_size,
        dropout_p=0.0,
        softcap=0.0,
        alibi_slopes=None,
    )


def _time_backend(backend, query, key, value, window_size, mode="fwd"):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Use grad-enabled inputs when benchmarking backward.
    q, k, v = query, key, value
    if mode == "fwd_plus_bwd":
        q = query.detach().clone().requires_grad_(True)
        k = key.detach().clone().requires_grad_(True)
        v = value.detach().clone().requires_grad_(True)

    # Warmup
    out = _run_backend(backend, q, k, v, window_size)
    if mode == "fwd_plus_bwd":
        out.sum().backward()
        q.grad = k.grad = v.grad = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    out = _run_backend(backend, q, k, v, window_size)
    if mode == "fwd_plus_bwd":
        out.sum().backward()
        out = out.detach()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter() - start, out


@pytest.mark.gpu
@pytest.mark.parametrize("window_size", [None, 1120], ids=["global", "sliding_window"])
@pytest.mark.parametrize("mode", ["fwd", "fwd_plus_bwd"], ids=["forward", "forward & backward"])
def test_attention_backend_benchmark(window_size, mode, capsys):
    """Benchmark flex attention against SDPA and flash attention on a large input.

    The global case compares flex, SDPA, and flash. The sliding-window case focuses on flex and
    flash, because a dense SDPA sliding-window mask is not practical at this sequence length.
    """

    pytest.importorskip("torch.nn.attention.flex_attention")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this benchmark")

    device = torch.device("cuda")
    dtype = torch.float16
    query, key, value = _make_inputs(device, dtype)

    backends = {"flex": FlexAttentionWrapper()}
    backends["sdpa"] = SDPAAttentionWrapper()

    try:
        backends["flash"] = FlashAttentionWrapper(head_dim=HEAD_DIM)
    except ImportError:
        pass

    timings = {}
    outputs = {}
    context = torch.inference_mode() if mode == "fwd" else contextlib.nullcontext()
    with context:
        for name, backend in backends.items():
            elapsed, output = _time_backend(backend, query, key, value, window_size, mode=mode)
            timings[name] = elapsed
            outputs[name] = output

    reference_name = "sdpa" if "sdpa" in outputs else "flex"
    reference = outputs[reference_name]

    for name, output in outputs.items():
        if name == reference_name:
            continue
        torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)

    with capsys.disabled():
        print(
            f"Attention benchmark: mode={mode}, window_size={window_size}, dtype={dtype}, shape={(Z, H, N_CTX, HEAD_DIM)}"
        )
        for name, elapsed in timings.items():
            print(f"  {name}: {elapsed * 1e3:.2f} ms")
