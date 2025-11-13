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
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.layers.normalization import ConditionalLayerNorm
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

    # check the function was compiled
    assert hasattr(ln_compiled, "_compile_kwargs")

    # check the result of the compiled function matches the uncompiled result
    assert torch.allclose(result, result_compiled)
