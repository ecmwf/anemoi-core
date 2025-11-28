# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import logging
import os

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.normalization import ConditionalLayerNorm
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.utils.compile import _get_compile_entry
from anemoi.models.utils.compile import _meets_library_versions_for_compile
from anemoi.models.utils.compile import mark_for_compilation

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
    return OmegaConf.create(
        {
            "compile": [
                {
                    "module": "anemoi.models.layers.conv.GraphTransformerConv",
                },
                {
                    "module": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                    "options": {
                        "dynamic": False,
                    },
                },
            ],
        }
    )


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
    shapes = [list(x.shape)]

    result = mhsa.forward(x, shapes, batch_size)
    result_compiled = mhsa_compiled.forward(x, shapes, batch_size)

    # check the result of the compiled function matches the uncompiled result
    assert torch.allclose(result, result_compiled)


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
    shapes = [list(x.shape)]

    result_compiled = mhsa_compiled.forward(x, shapes, batch_size)

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

    result = new_mhsa.forward(x, shapes, batch_size)
    assert torch.allclose(result, result_compiled)
