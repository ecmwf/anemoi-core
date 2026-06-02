# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from unittest.mock import MagicMock

import pytest
import torch

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import PointWiseMLPProcessorBlock
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
def pointwisemlp_processor(pointwisemlp_processor_init, device):
    return PointWiseMLPProcessor(**asdict(pointwisemlp_processor_init)).to(device)


def test_pointwisemlp_processor_init(pointwisemlp_processor, pointwisemlp_processor_init):
    assert isinstance(pointwisemlp_processor, PointWiseMLPProcessor)
    assert pointwisemlp_processor.num_chunks == pointwisemlp_processor_init.num_chunks
    assert pointwisemlp_processor.num_channels == pointwisemlp_processor_init.num_channels
    assert (
        pointwisemlp_processor.chunk_size
        == pointwisemlp_processor_init.num_layers // pointwisemlp_processor_init.num_chunks
    )


def test_pointwisemlp_processor_uses_only_pointwise_blocks(pointwisemlp_processor):
    assert all(isinstance(block, PointWiseMLPProcessorBlock) for block in pointwisemlp_processor.proc)


def test_pointwisemlp_processor_with_sharding_dropout_forward(pointwisemlp_processor, pointwisemlp_processor_init):
    gridsize = 100
    batch_size = 1
    x = torch.rand(
        gridsize, pointwisemlp_processor_init.num_channels, device=next(pointwisemlp_processor.parameters()).device
    )
    shard_info = GraphShardInfo(nodes=[gridsize])

    # Mock distributed group
    fake_model_comm_group = MagicMock()
    fake_model_comm_group.size.return_value = 2

    output = pointwisemlp_processor.forward(
        x,
        batch_size,
        shard_info,
        model_comm_group=fake_model_comm_group,
    )

    assert output.shape == x.shape


def test_pointwisemlp_processor_checkpointed_dropout_preserves_rng(pointwisemlp_processor_init):
    gridsize = 12
    batch_size = 1
    shard_info = GraphShardInfo(nodes=[gridsize])
    fake_model_comm_group = MagicMock()
    fake_model_comm_group.size.return_value = 2

    config = asdict(pointwisemlp_processor_init)
    config["num_channels"] = 8
    config["mlp_hidden_ratio"] = 2
    config["dropout_p"] = 0.5

    checkpointed_config = deepcopy(config)
    checkpointed_config["gradient_checkpointing"] = True
    eager_config = deepcopy(config)
    eager_config["gradient_checkpointing"] = False

    torch.manual_seed(123)
    checkpointed = PointWiseMLPProcessor(**checkpointed_config)
    eager = PointWiseMLPProcessor(**eager_config)
    eager.load_state_dict(checkpointed.state_dict())

    checkpointed.train()
    eager.train()

    x = torch.randn(gridsize, config["num_channels"])
    x_checkpointed = x.clone().requires_grad_()
    x_eager = x.clone().requires_grad_()

    torch.manual_seed(456)
    checkpointed_out = checkpointed(
        x_checkpointed,
        batch_size,
        shard_info,
        model_comm_group=fake_model_comm_group,
    )
    checkpointed_loss = checkpointed_out.square().sum()
    checkpointed_loss.backward()

    torch.manual_seed(456)
    eager_out = eager(
        x_eager,
        batch_size,
        shard_info,
        model_comm_group=fake_model_comm_group,
    )
    eager_loss = eager_out.square().sum()
    eager_loss.backward()

    assert torch.allclose(checkpointed_out, eager_out)
    assert torch.allclose(x_checkpointed.grad, x_eager.grad)
    for checkpointed_param, eager_param in zip(checkpointed.parameters(), eager.parameters(), strict=True):
        assert torch.allclose(checkpointed_param.grad, eager_param.grad)


def test_pointwisemlp_processor_forward(pointwisemlp_processor, pointwisemlp_processor_init):
    gridsize = 100
    batch_size = 1
    x = torch.rand(
        gridsize, pointwisemlp_processor_init.num_channels, device=next(pointwisemlp_processor.parameters()).device
    )
    shard_info = GraphShardInfo(nodes=[gridsize])

    output = pointwisemlp_processor.forward(x, batch_size, shard_info)
    assert output.shape == x.shape

    # Generate dummy target and loss function
    target = torch.randn(gridsize, pointwisemlp_processor_init.num_channels, device=output.device)
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
