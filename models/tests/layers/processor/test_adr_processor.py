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
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.adr import Advection
from anemoi.models.layers.adr import Diffusion
from anemoi.models.layers.adr import Reaction
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.models.layers.processor import ADRProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


def _geometry(**kwargs) -> dict:
    return {
        "_target_": "anemoi.models.layers.geometry.RegularLatLonGeometry",
        "layout_node_attribute_name": "layout_index",
        "bias_rank": 4,
        "bias_base_maps": 2,
        **kwargs,
    }


def _advection(**kwargs) -> dict:
    return {
        "_target_": "anemoi.models.layers.adr.Advection",
        "advection_channels": 16,
        "velocity_hidden_dim": 12,
        **kwargs,
    }


def _diffusion(**kwargs) -> dict:
    return {"_target_": "anemoi.models.layers.adr.Diffusion", **kwargs}


def _reaction(**kwargs) -> dict:
    return {"_target_": "anemoi.models.layers.adr.Reaction", "hidden_dim": 24, "num_layers": 4, **kwargs}


@dataclass
class ADRProcessorConfig:
    """Processor settings; nlat and nlon describe the hidden grid whose nodes are passed in."""

    num_layers: int = 4
    num_channels: int = 32
    num_chunks: int = 2
    nlat: int = 8
    nlon: int = 16
    geometry: dict = field(default_factory=_geometry)
    timestep: str = "6h"
    advection: Optional[dict] = field(default_factory=_advection)
    diffusion: Optional[dict] = field(default_factory=_diffusion)
    reaction: Optional[dict] = field(default_factory=_reaction)
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


def _build(cfg: ADRProcessorConfig) -> ADRProcessor:
    kwargs = asdict(cfg)
    nlat, nlon = kwargs.pop("nlat"), kwargs.pop("nlon")
    lat, lon = latlon_cell_centres(nlat, nlon)
    graph = HeteroData()
    graph["hidden"].x = torch.stack([lat, lon], dim=-1).float()
    graph["hidden"]["layout_index"] = torch.stack(
        torch.meshgrid(torch.arange(nlat), torch.arange(nlon), indexing="ij"), dim=-1
    ).view(-1, 2)
    graph["hidden"]["_layout_shape"] = [nlat, nlon]
    return ADRProcessor(**kwargs, hidden_nodes=graph["hidden"])


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
    # The grid size comes from the hidden nodes' layout, read once by the shared geometry.
    assert (adr_processor.geometry.nlat, adr_processor.geometry.nlon) == (
        adr_processor_init.nlat,
        adr_processor_init.nlon,
    )
    assert all(block.advection.transport.down.stencil.conv is not None for block in adr_processor.proc)
    for block in adr_processor.proc:
        assert isinstance(block, ADRProcessorBlock)
        # The grid size comes from the hidden nodes' layout.
        assert isinstance(block.advection, Advection)
        assert isinstance(block.diffusion, Diffusion)
        assert isinstance(block.reaction, Reaction)
        # By default every moved channel has its own velocity field.
        assert block.advection.transport.num_velocity_fields == adr_processor_init.advection["advection_channels"]
        # The layers split one 6 hour step, with velocities in units of 1 / Earth's rotation rate.
        assert block.advection.transport.time_step == pytest.approx(
            6 * 3600 * 7.29212e-5 / adr_processor_init.num_layers
        )


def test_the_geometry_is_held_once_and_stays_out_of_the_state_dict(adr_processor):
    """The grid is held once by the shared geometry, not once per layer.

    Before the geometry was shared, every layer kept its own copy of the arrival coordinates and
    of the grid index: on a 361 x 720 grid with 16 layers that is tens of megabytes of buffers.
    """
    assert adr_processor.geometry.position.shape == (adr_processor.geometry.nlat, adr_processor.geometry.nlon, 3)
    assert not [name for block in adr_processor.proc for name, _ in block.named_buffers()]
    assert not [name for name in adr_processor.state_dict() if name.startswith("geometry")]
    assert not list(adr_processor.geometry.parameters())


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("interpolation", "kernel_size", "cartesian_displacement"),
    [("bicubic", 5, False), ("bilinear", 3, False), ("bicubic", 5, True)],
)
def test_adr_processor_forward_backward(
    adr_processor_init, batch_size, interpolation, kernel_size, cartesian_displacement
):
    adr_processor_init.geometry = _geometry(interpolation=interpolation, kernel_size=kernel_size)
    adr_processor_init.advection = _advection(cartesian_displacement=cartesian_displacement)
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


@pytest.mark.parametrize(
    "dropped", [("advection",), ("diffusion",), ("reaction",), ("advection", "diffusion", "reaction")]
)
def test_adr_processor_steps_can_be_left_out(adr_processor_init, dropped):
    for name in dropped:
        setattr(adr_processor_init, name, None)
    processor = _build(adr_processor_init)

    for block in processor.proc:
        for name in ("advection", "diffusion", "reaction"):
            assert (getattr(block, name) is None) == (name in dropped)

    grid = adr_processor_init.nlat * adr_processor_init.nlon
    x = torch.rand(grid, adr_processor_init.num_channels)
    output = processor(x, batch_size=1, shard_info=GraphShardInfo(nodes=[grid]))
    assert output.shape == x.shape
    if len(dropped) == 3:
        assert not list(processor.parameters())
        torch.testing.assert_close(output, x)


@pytest.mark.parametrize("step", ["diffusion", "reaction"])
def test_adr_step_parts_can_be_switched_off(adr_processor_init, step):
    make = _diffusion if step == "diffusion" else _reaction
    full = getattr(_build(adr_processor_init).proc[0], step)
    setattr(adr_processor_init, step, make(layer_norm=False, low_rank_bias=False))
    bare = getattr(_build(adr_processor_init).proc[0], step)

    assert full.layer_norm is not None and full.bias is not None
    assert bare.layer_norm is None and bare.bias is None

    def count(module):
        return sum(p.numel() for p in module.parameters())

    assert count(bare) == count(full) - count(full.layer_norm) - count(full.bias)
