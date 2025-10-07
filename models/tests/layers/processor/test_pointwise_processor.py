# (C) Copyright 2025 Anemoi contributors.
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
from unittest.mock import MagicMock

import pytest
import torch

from anemoi.models.layers.processor import PointWiseMLPProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class PointWiseMLPProcessorConfig:
    num_layers: int = 2
    num_channels: int = 128
    num_chunks: int = 2
    mlp_hidden_ratio: int = 4
    dropout_p: float = 0.1
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


@pytest.fixture
def pointwisemlp_processor_init():
    return PointWiseMLPProcessorConfig()


@pytest.fixture
def pointwisemlp_processor(pointwisemlp_processor_init):
    return PointWiseMLPProcessor(**asdict(pointwisemlp_processor_init))


def test_pointwisemlp_processor_init(pointwisemlp_processor, pointwisemlp_processor_init):
    assert isinstance(pointwisemlp_processor, PointWiseMLPProcessor)
    assert pointwisemlp_processor.num_chunks == pointwisemlp_processor_init.num_chunks
    assert pointwisemlp_processor.num_channels == pointwisemlp_processor_init.num_channels
    assert (
        pointwisemlp_processor.chunk_size
        == pointwisemlp_processor_init.num_layers // pointwisemlp_processor_init.num_chunks
    )


@pytest.fixture(params=[0.1, None])
def test_pointwisemlp_processor_with_sharding_dropout_forward(pointwisemlp_processor, pointwisemlp_processor_init):
    gridsize = 100
    batch_size = 1
    x = torch.rand(gridsize, pointwisemlp_processor_init.num_channels)
    shard_shapes = [list(x.shape)]

    # Mock distributed group
    fake_model_comm_group = MagicMock()
    fake_model_comm_group.size.return_value = 2

    with pytest.raises(ValueError, match="Dropout is not supported when model is sharded"):
        pointwisemlp_processor.forward(
            x,
            batch_size,
            shard_shapes,
            model_comm_group=fake_model_comm_group,
        )


def test_pointwisemlp_processor_forward(pointwisemlp_processor, pointwisemlp_processor_init):
    gridsize = 100
    batch_size = 1
    x = torch.rand(gridsize, pointwisemlp_processor_init.num_channels)
    shard_shapes = [list(x.shape)]

    output = pointwisemlp_processor.forward(x, batch_size, shard_shapes)
    assert output.shape == x.shape

    # Generate dummy target and loss function
    target = torch.randn(gridsize, pointwisemlp_processor_init.num_channels)
    loss_fn = torch.nn.MSELoss()

    # Compute loss
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    for param in pointwisemlp_processor.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert (
            param.grad.shape == param.shape
        ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
