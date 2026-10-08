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
from anemoi.models.layers.block import FlowersProcessorBlock
from anemoi.models.layers.processor import FlowersProcessor
from anemoi.models.layers.semi_lagrangian import latlon_cell_centres
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class FlowersProcessorConfig:
    """Processor settings; nlat and nlon describe the hidden grid whose node coordinates are passed in."""

    num_layers: int = 4
    num_channels: int = 32
    num_chunks: int = 2
    nlat: int = 8
    nlon: int = 16
    num_heads: Optional[int] = None
    mlp_hidden_ratio: float = 2.0
    block_style: str = "pre_norm"
    interpolation: str = "bilinear"
    cartesian_displacement: bool = False
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


def _build(cfg: FlowersProcessorConfig) -> FlowersProcessor:
    kwargs = asdict(cfg)
    lat, lon = latlon_cell_centres(kwargs.pop("nlat"), kwargs.pop("nlon"))
    return FlowersProcessor(**kwargs, node_coordinates=torch.stack([lat, lon], dim=-1).float())


@pytest.fixture
def flowers_processor_init():
    return FlowersProcessorConfig()


def test_flowers_processor_init(flowers_processor_init):
    processor = _build(flowers_processor_init)
    assert len(processor.proc) == flowers_processor_init.num_layers
    for block in processor.proc:
        assert isinstance(block, FlowersProcessorBlock)
        assert (block.nlat, block.nlon) == (flowers_processor_init.nlat, flowers_processor_init.nlon)
        # By default each head moves 4 channels, as in FLOWERS.
        assert block.warp.num_heads == flowers_processor_init.num_channels // 4


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("block_style", "interpolation", "cartesian_displacement"),
    [("pre_norm", "bilinear", False), ("flowers", "bicubic", False), ("pre_norm", "bilinear", True)],
)
def test_flowers_processor_forward_backward(
    flowers_processor_init, batch_size, block_style, interpolation, cartesian_displacement
):
    flowers_processor_init.block_style = block_style
    flowers_processor_init.interpolation = interpolation
    flowers_processor_init.cartesian_displacement = cartesian_displacement
    processor = _build(flowers_processor_init)

    grid = flowers_processor_init.nlat * flowers_processor_init.nlon
    x = torch.rand(batch_size * grid, flowers_processor_init.num_channels, requires_grad=True)
    output = processor(x, batch_size=batch_size, shard_info=GraphShardInfo(nodes=[batch_size * grid]))
    assert output.shape == x.shape

    torch.nn.MSELoss()(output, torch.randn_like(output)).backward()
    assert x.grad is not None
    for param in processor.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert param.grad.shape == param.shape


def test_flowers_block_rejects_unknown_style(flowers_processor_init):
    flowers_processor_init.block_style = "unet"
    with pytest.raises(ValueError, match="block_style"):
        _build(flowers_processor_init)
