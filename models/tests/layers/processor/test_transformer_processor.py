# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import pytest
import torch

from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels


@dataclass
class TransformerProcessorConfig:
    num_layers: int = 2
    num_channels: int = 128
    num_chunks: int = 2
    num_heads: int = 16
    mlp_hidden_ratio: int = 4
    dropout_p: float = 0.1
    attention_implementation: str = "scaled_dot_product_attention"
    softcap: float = 0
    use_alibi_slopes: bool = False
    window_size: int = 10
    qk_norm: bool = True
    cpu_offload: bool = False
    layer_kernels = load_layer_kernels()


@pytest.fixture
def transformer_processor_init():
    return TransformerProcessorConfig()


@pytest.fixture
def transformer_processor(transformer_processor_init):
    return TransformerProcessor(
        num_layers=transformer_processor_init.num_layers,
        num_channels=transformer_processor_init.num_channels,
        num_chunks=transformer_processor_init.num_chunks,
        num_heads=transformer_processor_init.num_heads,
        mlp_hidden_ratio=transformer_processor_init.mlp_hidden_ratio,
        dropout_p=transformer_processor_init.dropout_p,
        attention_implementation=transformer_processor_init.attention_implementation,
        softcap=transformer_processor_init.softcap,
        use_alibi_slopes=transformer_processor_init.use_alibi_slopes,
        window_size=transformer_processor_init.window_size,
        qk_norm=transformer_processor_init.qk_norm,
        cpu_offload=transformer_processor_init.cpu_offload,
        layer_kernels=transformer_processor_init.layer_kernels,
    )


def test_transformer_processor_init(transformer_processor, transformer_processor_init):
    assert isinstance(transformer_processor, TransformerProcessor)
    assert transformer_processor.num_chunks == transformer_processor_init.num_chunks
    assert transformer_processor.num_channels == transformer_processor_init.num_channels
    assert (
        transformer_processor.chunk_size
        == transformer_processor_init.num_layers // transformer_processor_init.num_chunks
    )


def test_transformer_processor_forward(transformer_processor, transformer_processor_init):
    gridsize = 100
    batch_size = 1
    x = torch.rand(gridsize, transformer_processor_init.num_channels)
    shard_shapes = [list(x.shape)]

    output = transformer_processor.forward(x, batch_size, shard_shapes)
    assert output.shape == x.shape

    # Generate dummy target and loss function
    target = torch.randn(gridsize, transformer_processor_init.num_channels)
    loss_fn = torch.nn.MSELoss()

    # Compute loss
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    for param in transformer_processor.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert (
            param.grad.shape == param.shape
        ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
