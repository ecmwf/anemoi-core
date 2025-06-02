# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch

from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


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
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


@pytest.fixture
def transformer_processor_init():
    return TransformerProcessorConfig()


@pytest.fixture
def transformer_processor(transformer_processor_init):
    return TransformerProcessor(
        **asdict(transformer_processor_init),
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
