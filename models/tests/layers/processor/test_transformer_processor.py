# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.layers.processor import TransformerProcessor


@pytest.fixture
def transformer_processor_init():
    num_layers = 2
    window_size = 10
    num_channels = 128
    num_chunks = 2
    activation = "GELU"
    cpu_offload = False
    num_heads = 16
    mlp_hidden_ratio = 4
    dropout_p = 0.1
    softcap = 0.5
    attention_implementation = "scaled_dot_product_attention"
    return (
        num_layers,
        window_size,
        num_channels,
        num_chunks,
        activation,
        cpu_offload,
        num_heads,
        mlp_hidden_ratio,
        dropout_p,
        softcap,
        attention_implementation,
    )


@pytest.fixture
def transformer_processor(transformer_processor_init):
    (
        num_layers,
        window_size,
        num_channels,
        num_chunks,
        activation,
        cpu_offload,
        num_heads,
        mlp_hidden_ratio,
        dropout_p,
        softcap,
        attention_implementation,
    ) = transformer_processor_init
    return TransformerProcessor(
        num_layers=num_layers,
        window_size=window_size,
        num_channels=num_channels,
        num_chunks=num_chunks,
        activation=activation,
        cpu_offload=cpu_offload,
        num_heads=num_heads,
        mlp_hidden_ratio=mlp_hidden_ratio,
        dropout_p=dropout_p,
        attention_implementation=attention_implementation,
        softcap=softcap,
    )


def test_transformer_processor_init(transformer_processor, transformer_processor_init):
    (
        num_layers,
        _window_size,
        num_channels,
        num_chunks,
        _activation,
        _cpu_offload,
        _num_heads,
        _mlp_hidden_ratio,
        _dropout_p,
        _attention_implementation,
        _softcap,
    ) = transformer_processor_init
    assert isinstance(transformer_processor, TransformerProcessor)
    assert transformer_processor.num_chunks == num_chunks
    assert transformer_processor.num_channels == num_channels
    assert transformer_processor.chunk_size == num_layers // num_chunks


def test_transformer_processor_forward(transformer_processor, transformer_processor_init):
    (
        _num_layers,
        _window_size,
        num_channels,
        _num_chunks,
        _activation,
        _cpu_offload,
        _num_heads,
        _mlp_hidden_ratio,
        _dropout_p,
        _attention_implementation,
        _softcap,
    ) = transformer_processor_init
    gridsize = 100
    batch_size = 1
    x = torch.rand(gridsize, num_channels)
    shard_shapes = [list(x.shape)]

    output = transformer_processor.forward(x, batch_size, shard_shapes)
    assert output.shape == x.shape

    # Generate dummy target and loss function
    target = torch.randn(gridsize, num_channels)
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
