# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ADR processor on a MeshGeometry: the same steps as on a grid, over the hidden graph's edges."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import pytest
import torch

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.adr import Advection
from anemoi.models.layers.adr import Diffusion
from anemoi.models.layers.adr import Reaction
from anemoi.models.layers.adr.layers import GraphSpatialMixer
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.edge_stencil import EdgeDepthwiseConv
from anemoi.models.layers.processor import ADRProcessor
from anemoi.models.layers.semi_lagrangian import KNNSampler
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict
from tests.layers.sphere import expand_edges
from tests.layers.sphere import fibonacci_nodes
from tests.layers.sphere import hidden_nodes
from tests.layers.sphere import knn_edges

NUM_NODES = 200
NUM_NEIGHBOURS = 6
EDGE_DIM = 4
MIXER_BLOCKS = ["edge_conv", "transformer", "gnn"]


def _geometry(mixer_block: str = "edge_conv", **kwargs) -> dict:
    mixer_kwargs = {"num_heads": 2, "graph_attention_backend": "pyg"} if mixer_block == "transformer" else {}
    return {
        "_target_": "anemoi.models.layers.geometry.MeshGeometry",
        "mixer_block": mixer_block,
        "mixer_kwargs": mixer_kwargs,
        "bias_rank": 4,
        "bias_base_maps": 2,
        "bias_lat_degree": 4,
        "bias_lon_wavenumber": 3,
        **kwargs,
    }


def _advection(**kwargs) -> dict:
    return {
        "_target_": "anemoi.models.layers.adr.Advection",
        "advection_channels": 8,
        "num_velocity_fields": 4,
        "velocity_hidden_dim": 8,
        **kwargs,
    }


def _diffusion(**kwargs) -> dict:
    return {"_target_": "anemoi.models.layers.adr.Diffusion", **kwargs}


def _reaction(**kwargs) -> dict:
    return {"_target_": "anemoi.models.layers.adr.Reaction", "hidden_dim": 24, "num_layers": 3, **kwargs}


@dataclass
class MeshADRProcessorConfig:
    num_layers: int = 2
    num_channels: int = 16
    num_chunks: int = 1
    geometry: dict = field(default_factory=_geometry)
    timestep: str = "6h"
    advection: Optional[dict] = field(default_factory=_advection)
    diffusion: Optional[dict] = field(default_factory=_diffusion)
    reaction: Optional[dict] = field(default_factory=_reaction)
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


def _build(cfg: MeshADRProcessorConfig, order: Optional[torch.Tensor] = None) -> ADRProcessor:
    lat, lon = fibonacci_nodes(NUM_NODES)
    if order is not None:
        lat, lon = lat[order], lon[order]
    return ADRProcessor(**asdict(cfg), hidden_nodes=hidden_nodes(lat, lon), edge_dim=EDGE_DIM)


def _run(processor: ADRProcessor, x: torch.Tensor, batch_size: int, order: Optional[torch.Tensor] = None):
    lat, lon = fibonacci_nodes(NUM_NODES)
    if order is not None:
        lat, lon = lat[order], lon[order]
    edge_attr, edge_index = expand_edges(*knn_edges(lat, lon, NUM_NEIGHBOURS), NUM_NODES, batch_size)
    return processor(
        x,
        batch_size=batch_size,
        shard_info=GraphShardInfo(nodes=[batch_size * NUM_NODES]),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )


@pytest.fixture
def mesh_adr_processor_init():
    return MeshADRProcessorConfig()


@pytest.mark.parametrize("mixer_block", MIXER_BLOCKS)
def test_mesh_adr_processor_init(mesh_adr_processor_init, mixer_block):
    mesh_adr_processor_init.geometry = _geometry(mixer_block)
    processor = _build(mesh_adr_processor_init)
    block_class = {
        "edge_conv": EdgeDepthwiseConv,
        "transformer": GraphTransformerProcessorBlock,
        "gnn": GraphConvProcessorBlock,
    }[mixer_block]

    assert len(processor.proc) == mesh_adr_processor_init.num_layers
    assert processor.geometry.num_nodes == NUM_NODES
    assert processor.geometry.position.shape == (NUM_NODES, 3)
    for layer in processor.proc:
        assert isinstance(layer, ADRProcessorBlock)
        assert isinstance(layer.advection, Advection)
        assert isinstance(layer.diffusion, Diffusion)
        assert isinstance(layer.reaction, Reaction)
        transport = layer.advection.transport
        assert isinstance(transport.sampler, KNNSampler)
        for mixer in (transport.down, transport.velocity_out, layer.diffusion.mixer):
            assert isinstance(mixer, GraphSpatialMixer)
            assert isinstance(mixer.block, block_class)
        # The layers split one 6 hour step, with velocities in units of 1 / Earth's rotation rate.
        assert transport.time_step == pytest.approx(6 * 3600 * 7.29212e-5 / mesh_adr_processor_init.num_layers)


def test_the_geometry_is_held_once_and_stays_out_of_the_state_dict(mesh_adr_processor_init):
    """The node coordinates live on the one shared geometry, not once per layer."""
    processor = _build(mesh_adr_processor_init)

    assert processor.geometry.position.shape == (NUM_NODES, 3)
    # Nothing geometric is copied into the blocks, and the geometry adds nothing to a checkpoint.
    assert not [name for block in processor.proc for name, _ in block.named_buffers()]
    assert not [name for name in processor.state_dict() if name.startswith("geometry")]
    assert not list(processor.geometry.parameters())


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mixer_block", MIXER_BLOCKS)
@pytest.mark.parametrize("cartesian_displacement", [False, True])
def test_mesh_adr_processor_forward_backward(mesh_adr_processor_init, batch_size, mixer_block, cartesian_displacement):
    mesh_adr_processor_init.geometry = _geometry(mixer_block)
    mesh_adr_processor_init.advection = _advection(cartesian_displacement=cartesian_displacement)
    processor = _build(mesh_adr_processor_init)

    x = torch.rand(batch_size * NUM_NODES, mesh_adr_processor_init.num_channels, requires_grad=True)
    output = _run(processor, x, batch_size)
    assert output.shape == x.shape
    assert output.isfinite().all()

    torch.nn.MSELoss()(output, torch.randn_like(output)).backward()
    assert x.grad is not None
    for name, param in processor.named_parameters():
        assert param.grad is not None, f"param.grad is None for {name}"
        assert param.grad.shape == param.shape


def test_mesh_adr_processor_batch_members_are_independent(mesh_adr_processor_init):
    torch.manual_seed(0)
    processor = _build(mesh_adr_processor_init)
    x = torch.rand(2 * NUM_NODES, mesh_adr_processor_init.num_channels)
    both = _run(processor, x, 2)
    second = _run(processor, x[NUM_NODES:], 1)
    torch.testing.assert_close(both[NUM_NODES:], second, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mixer_block", MIXER_BLOCKS)
def test_mesh_adr_processor_does_not_depend_on_node_order(mesh_adr_processor_init, mixer_block):
    """The same mesh with its nodes stored in another order gives the same values, node by node."""
    mesh_adr_processor_init.geometry = _geometry(mixer_block)
    order = torch.randperm(NUM_NODES, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    stored_in_order = _build(mesh_adr_processor_init)
    shuffled = _build(mesh_adr_processor_init, order)
    with torch.no_grad():
        for parameter in stored_in_order.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    shuffled.load_state_dict(stored_in_order.state_dict())

    x = torch.randn(NUM_NODES, mesh_adr_processor_init.num_channels)
    expected = _run(stored_in_order, x, 1)
    out = _run(shuffled, x[order], 1, order)
    torch.testing.assert_close(out, expected[order], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "dropped", [("advection",), ("diffusion",), ("reaction",), ("advection", "diffusion", "reaction")]
)
def test_mesh_adr_processor_steps_can_be_left_out(mesh_adr_processor_init, dropped):
    for name in dropped:
        setattr(mesh_adr_processor_init, name, None)
    processor = _build(mesh_adr_processor_init)

    x = torch.rand(NUM_NODES, mesh_adr_processor_init.num_channels)
    output = _run(processor, x, 1)
    assert output.shape == x.shape
    if len(dropped) == 3:
        assert not list(processor.parameters())
        torch.testing.assert_close(output, x)


@pytest.mark.parametrize("step", ["diffusion", "reaction"])
def test_mesh_adr_step_parts_can_be_switched_off(mesh_adr_processor_init, step):
    make = _diffusion if step == "diffusion" else _reaction
    full = getattr(_build(mesh_adr_processor_init).proc[0], step)
    setattr(mesh_adr_processor_init, step, make(layer_norm=False, low_rank_bias=False))
    bare = getattr(_build(mesh_adr_processor_init).proc[0], step)

    assert full.layer_norm is not None and full.bias is not None
    assert bare.layer_norm is None and bare.bias is None


def test_mesh_geometry_needs_the_hidden_edges(mesh_adr_processor_init):
    processor = _build(mesh_adr_processor_init)
    x = torch.rand(NUM_NODES, mesh_adr_processor_init.num_channels)
    with pytest.raises(ValueError, match="hidden-to-hidden edges"):
        processor(x, batch_size=1, shard_info=GraphShardInfo(nodes=[NUM_NODES]))
