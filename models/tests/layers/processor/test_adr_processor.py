# (C) Copyright 2026 Anemoi contributors.
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
from typing import Optional

import pytest
import torch

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.processor import ADRProcessor
from anemoi.models.layers.semi_lagrangian import latlon_cell_centres
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class ADRProcessorConfig:
    """Processor settings; nlat and nlon describe the hidden grid whose node coordinates are passed in."""

    num_layers: int = 4
    num_channels: int = 32
    num_chunks: int = 2
    nlat: int = 8
    nlon: int = 16
    timestep: str = "6h"
    advection_channels: int = 16
    num_heads: Optional[int] = None
    velocity_hidden_dim: int = 12
    reaction_hidden_dim: int = 24
    reaction_num_layers: int = 4
    kernel_size: int = 5
    interpolation: str = "bicubic"
    bias_rank: int = 4
    bias_base_maps: int = 2
    cartesian_displacement: bool = False
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


def _build(cfg: ADRProcessorConfig) -> ADRProcessor:
    kwargs = asdict(cfg)
    lat, lon = latlon_cell_centres(kwargs.pop("nlat"), kwargs.pop("nlon"))
    return ADRProcessor(**kwargs, node_coordinates=torch.stack([lat, lon], dim=-1).float())


@pytest.fixture
def adr_processor_init():
    return ADRProcessorConfig()


@pytest.fixture
def adr_processor(adr_processor_init):
    return _build(adr_processor_init)


def test_adr_processor_init(adr_processor, adr_processor_init):
    assert isinstance(adr_processor, ADRProcessor)
    assert adr_processor.num_chunks == adr_processor_init.num_chunks
    assert adr_processor.num_channels == adr_processor_init.num_channels
    assert adr_processor.chunk_size == adr_processor_init.num_layers // adr_processor_init.num_chunks
    assert len(adr_processor.proc) == adr_processor_init.num_layers
    for block in adr_processor.proc:
        assert isinstance(block, ADRProcessorBlock)
        # The grid size comes from the node coordinates.
        assert (block.nlat, block.nlon) == (adr_processor_init.nlat, adr_processor_init.nlon)
        # By default every moved channel has its own velocity field.
        assert block.advection.num_heads == adr_processor_init.advection_channels
        # The layers split one 6 hour step, with velocities in units of 1 / Earth's rotation rate.
        assert block.advection.time_step == pytest.approx(6 * 3600 * 7.29212e-5 / adr_processor_init.num_layers)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("interpolation", "kernel_size", "cartesian_displacement"),
    [("bicubic", 5, False), ("bilinear", 3, False), ("bicubic", 5, True)],
)
def test_adr_processor_forward_backward(
    adr_processor_init, batch_size, interpolation, kernel_size, cartesian_displacement
):
    adr_processor_init.interpolation = interpolation
    adr_processor_init.kernel_size = kernel_size
    adr_processor_init.cartesian_displacement = cartesian_displacement
    processor = _build(adr_processor_init)

    grid = adr_processor_init.nlat * adr_processor_init.nlon
    x = torch.rand(batch_size * grid, adr_processor_init.num_channels, requires_grad=True)
    shard_info = GraphShardInfo(nodes=[batch_size * grid])

    output = processor.forward(x, batch_size=batch_size, shard_info=shard_info)
    assert output.shape == x.shape

    target = torch.randn_like(output)
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()

    assert x.grad is not None
    for param in processor.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert param.grad.shape == param.shape


def test_adr_processor_batch_members_are_independent(adr_processor_init):
    torch.manual_seed(0)
    processor = _build(adr_processor_init)

    grid = adr_processor_init.nlat * adr_processor_init.nlon
    x = torch.rand(2 * grid, adr_processor_init.num_channels)
    both = processor(x, batch_size=2, shard_info=GraphShardInfo(nodes=[2 * grid]))
    second = processor(x[grid:], batch_size=1, shard_info=GraphShardInfo(nodes=[grid]))
    torch.testing.assert_close(both[grid:], second, atol=1e-4, rtol=1e-4)
