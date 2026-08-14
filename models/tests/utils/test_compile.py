# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib.util
import io
import logging
import os

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.normalization import ConditionalLayerNorm
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.utils.compile import _get_compile_entry
from anemoi.models.utils.compile import _meets_library_versions_for_compile
from anemoi.models.utils.compile import mark_for_compilation

HAS_ANEMOI_TRAINING = False
if importlib.util.find_spec("anemoi.training") is not None:
    HAS_ANEMOI_TRAINING = True

LOGGER = logging.getLogger(__name__)


def graphtransformer_compile_config() -> None:
    return OmegaConf.create(
        {
            "compile": [
                {
                    "module": "anemoi.models.layers.conv.GraphTransformerConv",
                },
            ],
        }
    )


def layer_kernel_compile_config() -> None:
    return OmegaConf.create(
        {
            "compile": [
                {
                    "module": "torch.nn.Linear",
                },
            ],
        }
    )


def graphtransformer_ens_compile_config() -> None:
    modules_to_compile = [
        {
            "module": "anemoi.models.layers.conv.GraphTransformerConv",
        },
        {
            "module": "anemoi.models.layers.normalization.ConditionalLayerNorm",
            "options": {
                "dynamic": False,
            },
        },
    ]
    if HAS_ANEMOI_TRAINING:
        modules_to_compile.append(
            {
                "module": "anemoi.training.losses.CRPS",
                "options": {
                    "dynamic": False,
                    "fullgraph": True,
                    "mode": "max-autotune",
                },
            }
        )
    return OmegaConf.create({"compile": modules_to_compile})


def test_compile_config_no_match() -> None:
    """Tests that _get_compile_entry() returns None when no match is found."""
    cfg = graphtransformer_compile_config()

    num_channels = 64
    cond_shape = 16
    model = ConditionalLayerNorm(num_channels, condition_shape=cond_shape)
    result = _get_compile_entry(model, cfg.compile)

    assert result is None


def test_compile_config_match() -> None:
    """Tests that _get_compile_entry() returns a dict when a match is found."""
    cfg = graphtransformer_ens_compile_config()

    num_channels = 64
    cond_shape = 16
    model = ConditionalLayerNorm(num_channels, condition_shape=cond_shape)
    result = _get_compile_entry(model, cfg.compile)

    assert type(result) is DictConfig


def test_compile() -> None:

    # Skip this test if library versions aren't met
    if not _meets_library_versions_for_compile():
        LOGGER.warning("triton not installed. skipping 'test_compile.py::test_compile'")
        return

    num_channels = 64
    cond_shape = 16
    ln = ConditionalLayerNorm(num_channels, condition_shape=cond_shape)
    x_in = torch.randn(num_channels)
    cond = torch.randn(cond_shape)
    result = ln.forward(x_in, cond)

    cfg = graphtransformer_ens_compile_config()
    ln_compiled = mark_for_compilation(ln, cfg.compile)

    result_compiled = ln_compiled.forward(x_in, cond)

    # check the result of the compiled function matches the uncompiled result
    assert torch.allclose(result, result_compiled)


def test_compile_layer_kernel() -> None:

    # Skip this test if library versions aren't met
    if not _meets_library_versions_for_compile():
        LOGGER.warning("triton not installed. skipping 'test_compile.py::test_compile_layer_kernel'")
        return

    cfg = layer_kernel_compile_config()
    layer_kernels = load_layer_kernels(kernel_config={})

    num_heads = 1
    embed_dim = 64
    dropout_p = 0.0
    batch_size = 1
    mhsa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )
    mhsa_compiled = mark_for_compilation(mhsa, cfg.compile)

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shard_info = GraphShardInfo(nodes=[2])

    result = mhsa.forward(x, shard_info, batch_size)
    result_compiled = mhsa_compiled.forward(x, shard_info, batch_size)

    # check the result of the compiled function matches the uncompiled result
    assert torch.allclose(result, result_compiled)


def _run_with_thread_count_change(compile_entry: dict) -> int:
    """Compile a ConditionalLayerNorm via mark_for_compilation, run it under two different
    intra-op thread counts, and return the number of unique compiled graphs."""
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()

    num_channels = 64
    cond_shape = 16
    ln = ConditionalLayerNorm(num_channels, condition_shape=cond_shape)
    ln_compiled = mark_for_compilation(ln, OmegaConf.create({"compile": [compile_entry]}).compile)

    x_in = torch.randn(num_channels)
    cond = torch.randn(cond_shape)
    prev_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(max(prev_num_threads, 2))
        ln_compiled(x_in, cond)
        torch.set_num_threads(1)
        ln_compiled(x_in, cond)
    finally:
        torch.set_num_threads(prev_num_threads)

    return counters["stats"].get("unique_graphs", 0)


def test_guard_filter_suppresses_global_state_recompile() -> None:
    if not _meets_library_versions_for_compile() or torch.__version__ < "2.8":
        LOGGER.warning("skipping 'test_guard_filter_suppresses_global_state_recompile'")
        return

    entry = {"module": "anemoi.models.layers.normalization.ConditionalLayerNorm"}
    assert _run_with_thread_count_change(entry) == 1


def test_guard_filter_opt_out() -> None:
    if not _meets_library_versions_for_compile() or torch.__version__ < "2.8":
        LOGGER.warning("skipping 'test_guard_filter_opt_out'")
        return

    entry = {
        "module": "anemoi.models.layers.normalization.ConditionalLayerNorm",
        "filter_global_state_guards": False,
    }
    assert _run_with_thread_count_change(entry) == 2


def test_guard_filter_mode_translation() -> None:
    """A compile entry with `mode` must not raise the torch.compile mode/options conflict."""
    if not _meets_library_versions_for_compile() or torch.__version__ < "2.8":
        LOGGER.warning("skipping 'test_guard_filter_mode_translation'")
        return

    entry = {
        "module": "anemoi.models.layers.normalization.ConditionalLayerNorm",
        "options": {"mode": "default"},
    }
    assert _run_with_thread_count_change(entry) == 1


def test_compile_save_checkpoint() -> None:
    """Tests that a compiled module can be pickled and saved as a checkpoint"""
    # Skip this test if library versions aren't met
    if not _meets_library_versions_for_compile():
        LOGGER.warning("triton not installed. skipping 'test_compile.py::test_compile_save_checkpoint'")
        return

    num_channels = 64
    cond_shape = 16
    ln = ConditionalLayerNorm(num_channels, condition_shape=cond_shape)
    x_in = torch.randn(num_channels)
    cond = torch.randn(cond_shape)

    cfg = graphtransformer_ens_compile_config()
    ln_compiled = mark_for_compilation(ln, cfg.compile)

    # we dont care about the result, but we need to compute it to trigger compilation
    _ = ln_compiled.forward(x_in, cond)

    # try save checkpoint
    buffer = io.BytesIO()
    torch.save(ln_compiled, buffer)


def test_compile_load_checkpoint() -> None:
    """Tests that a compiled checkpoint can be loaded by a non-compiled model"""
    # Skip this test if library versions aren't met
    if not _meets_library_versions_for_compile():
        LOGGER.warning("triton not installed. skipping 'test_compile.py::test_compile_load_checkpoint'")
        return

    cfg = layer_kernel_compile_config()
    layer_kernels = load_layer_kernels(kernel_config={})

    num_heads = 1
    embed_dim = 64
    dropout_p = 0.0
    batch_size = 1
    mhsa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )
    mhsa_compiled = mark_for_compilation(mhsa, cfg.compile)

    x = torch.randn(batch_size * 2, embed_dim, requires_grad=True)
    shard_info = GraphShardInfo(nodes=[2])

    result_compiled = mhsa_compiled.forward(x, shard_info, batch_size)

    torch.save(mhsa_compiled, "compiled.pt")

    checkpoint = torch.load("compiled.pt", weights_only=False)
    os.remove("compiled.pt")

    new_mhsa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
    )
    new_mhsa.load_state_dict(checkpoint.state_dict(), assign=False)

    result = new_mhsa.forward(x, shard_info, batch_size)
    assert torch.allclose(result, result_compiled)
