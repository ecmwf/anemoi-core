# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Semi-Lagrangian blocks and processors split over several GPUs must match the same on one GPU.

The blocks run as the processors run them: the geometry binds the forward pass, and the nodes
are put into the block's layout with ``Layout.to_state`` and taken back with
``Layout.from_state``, so the split into bands of latitude rows is tested too.

Parameter gradients are compared after summing them over the GPUs, which is what training does
with model-parallel parameters.
"""

from typing import Optional

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.block import FlowersProcessorBlock
from anemoi.models.layers.geometry import RegularLatLonGeometry
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.models.layers.processor import ADRProcessor
from anemoi.models.layers.utils import load_layer_kernels
from tests.layers.sphere import fibonacci_nodes
from tests.layers.sphere import hidden_nodes
from tests.layers.sphere import knn_edges

from ._distributed_runner import _run_distributed_test
from .distributed_test_utils import torch_version_less_than


def _latlon_geometry(nlat: int, nlon: int, order: Optional[torch.Tensor] = None) -> RegularLatLonGeometry:
    """A grid geometry whose nodes are stored in the given order, row by row by default."""
    lat, lon = latlon_cell_centres(nlat, nlon)
    order = torch.arange(nlat * nlon) if order is None else order
    nodes = {
        "x": torch.stack([lat, lon], dim=-1).float()[order],
        "layout_index": torch.stack([order // nlon, order % nlon], dim=-1),
    }
    return RegularLatLonGeometry(
        hidden_nodes=nodes, layout_node_attribute_name="layout_index", bias_rank=4, bias_base_maps=2
    )


def _build_block(kind: str, geometry: RegularLatLonGeometry, num_heads: int, cartesian: bool) -> torch.nn.Module:
    torch.manual_seed(0)
    if kind == "flowers":
        return FlowersProcessorBlock(
            num_channels=12,
            hidden_dim=24,
            num_heads=num_heads,
            geometry=geometry,
            layer_kernels=load_layer_kernels(),
            cartesian_displacement=cartesian,
        )
    block = ADRProcessorBlock(
        num_channels=16,
        geometry=geometry,
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
        advection={
            "_target_": "anemoi.models.layers.adr.Advection",
            "advection_channels": 12,
            "num_velocity_fields": num_heads,
            "velocity_hidden_dim": 10,
            "cartesian_displacement": cartesian,
        },
        diffusion={"_target_": "anemoi.models.layers.adr.Diffusion"},
        reaction={"_target_": "anemoi.models.layers.adr.Reaction", "hidden_dim": 24, "num_layers": 3},
    )
    # The bias fields start close to zero; give them values so that every path is tested.
    for bias in (block.advection.transport.velocity_bias, block.diffusion.bias, block.reaction.bias):
        for p in (bias.coefficients, bias.lat_profiles, bias.lon_profiles):
            torch.nn.init.normal_(p)
    return block


def _block_rank(rank, world_size, device, group, kind, cartesian, dtype, nlat, nlon, num_heads, shuffled):
    grid = nlat * nlon
    order = torch.randperm(grid, generator=torch.Generator().manual_seed(2)) if shuffled else None
    # The processor owns the geometry, so it is cast with the blocks; do the same here.
    geometry = _latlon_geometry(nlat, nlon, order).to(device, dtype)
    block = _build_block(kind, geometry, num_heads, cartesian).to(device, dtype)

    def run(x, shard_sizes, model_comm_group=None):
        shard_info = GraphShardInfo(nodes=shard_sizes)
        layout = geometry.bind(x, 1, shard_info, model_comm_group)
        out = block(layout.to_state(x), shard_info, 1, model_comm_group=model_comm_group, layout=layout)[0]
        return layout.from_state(out)

    channels = 16 if kind == "adr" else 12
    # float64 shows that splitting changes nothing beyond rounding; float32 is what training uses.
    atol, rtol = (1e-11, 1e-9) if dtype == torch.float64 else (1e-5, 1e-5)
    grad_atol, grad_rtol = (1e-10, 1e-9) if dtype == torch.float64 else (1e-4, 1e-4)

    torch.manual_seed(1)
    x = torch.randn(grid, channels, device=device, dtype=dtype)
    grad_out = torch.randn(grid, channels, device=device, dtype=dtype)

    x_full = x.clone().requires_grad_()
    out_full = run(x_full, [grid])
    out_full.backward(grad_out)
    grads_full = {name: p.grad.clone() for name, p in block.named_parameters()}
    block.zero_grad()

    shard_sizes = get_balanced_partition_sizes(grid, world_size)
    start, end = get_partition_range(shard_sizes, rank)
    x_local = x[start:end].clone().requires_grad_()
    out_local = run(x_local, shard_sizes, group)
    out_local.backward(grad_out[start:end])

    torch.testing.assert_close(out_local, out_full[start:end].detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_local.grad, x_full.grad[start:end], atol=atol, rtol=rtol)
    for name, p in block.named_parameters():
        grad = p.grad.clone()
        dist.all_reduce(grad, group=group)
        torch.testing.assert_close(grad, grads_full[name], atol=grad_atol, rtol=grad_rtol, msg=f"gradient of {name}")


@pytest.mark.distributed
@pytest.mark.parametrize("kind", ["adr", "flowers"])
@pytest.mark.parametrize("cartesian", [False, True], ids=["local-displacement", "cartesian-displacement"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("shuffled", [False, True], ids=["nodes-row-by-row", "nodes-shuffled"])
@pytest.mark.parametrize(
    ("nlat", "nlon", "num_heads"),
    [
        pytest.param(6, 12, 12, id="one-velocity-per-channel"),
        pytest.param(7, 10, 3, id="grouped-velocities-uneven-split"),
    ],
)
def test_block_matches_single_gpu(
    kind, cartesian, dtype, shuffled, nlat, nlon, num_heads, distributed_backend, distributed_world_size
):
    if dtype == torch.float64 and distributed_backend == "nccl":
        pytest.skip("The float64 comparison runs on CPU.")
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall_transpose requires torch >= 2.6.")
    if num_heads < distributed_world_size:
        pytest.skip("Needs at least as many velocity fields as GPUs.")
    _run_distributed_test(
        _block_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        kind=kind,
        cartesian=cartesian,
        dtype=dtype,
        nlat=nlat,
        nlon=nlon,
        num_heads=num_heads,
        shuffled=shuffled,
    )


def _mesh_processor_rank(rank, world_size, device, group, block, cartesian, dtype):
    num_nodes = 150
    lat, lon = fibonacci_nodes(num_nodes)
    edge_attr, edge_index = knn_edges(lat, lon, 6)
    edge_attr, edge_index = edge_attr.to(device, dtype), edge_index.to(device)
    geometry = {
        "_target_": "anemoi.models.layers.geometry.MeshGeometry",
        "mixer_block": block,
        "mixer_kwargs": {"num_heads": 2, "graph_attention_backend": "pyg"} if block == "transformer" else {},
        "bias_rank": 4,
        "bias_base_maps": 2,
        "bias_lat_degree": 4,
        "bias_lon_wavenumber": 3,
    }

    torch.manual_seed(0)
    processor = ADRProcessor(
        num_layers=2,
        num_channels=16,
        num_chunks=1,
        geometry=geometry,
        hidden_nodes=hidden_nodes(lat, lon),
        edge_dim=edge_attr.shape[1],
        timestep="6h",
        advection={
            "_target_": "anemoi.models.layers.adr.Advection",
            "advection_channels": 12,
            "num_velocity_fields": 6,
            "velocity_hidden_dim": 10,
            "cartesian_displacement": cartesian,
        },
        diffusion={"_target_": "anemoi.models.layers.adr.Diffusion"},
        reaction={"_target_": "anemoi.models.layers.adr.Reaction", "hidden_dim": 24, "num_layers": 3},
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device, dtype)
    # The bias fields start close to zero; give them values so that every path is tested.
    with torch.no_grad():
        for name, p in processor.named_parameters():
            if "_profiles" in name or "coefficients" in name:
                p.normal_()
    # The GNN block and the edge stencil all-gather their input, and the backward of that
    # all-reduces the gradient in float32 (anemoi.models.distributed.primitives._reduce), so their
    # gradients agree only to float32 precision. The transformer exchanges a halo instead.
    exact = dtype == torch.float64 and block == "transformer"
    atol, rtol = (1e-11, 1e-9) if exact else (1e-5, 1e-5)
    grad_atol, grad_rtol = (1e-10, 1e-9) if exact else (1e-4, 1e-4)

    def run(x, shard_sizes, model_comm_group=None):
        return processor(
            x,
            batch_size=1,
            shard_info=GraphShardInfo(nodes=shard_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )

    torch.manual_seed(1)
    x = torch.randn(num_nodes, 16, device=device, dtype=dtype)
    grad_out = torch.randn(num_nodes, 16, device=device, dtype=dtype)

    x_full = x.clone().requires_grad_()
    out_full = run(x_full, [num_nodes])
    out_full.backward(grad_out)
    grads_full = {name: p.grad.clone() for name, p in processor.named_parameters()}
    processor.zero_grad()

    shard_sizes = get_balanced_partition_sizes(num_nodes, world_size)
    start, end = get_partition_range(shard_sizes, rank)
    x_local = x[start:end].clone().requires_grad_()
    out_local = run(x_local, shard_sizes, group)
    out_local.backward(grad_out[start:end])

    torch.testing.assert_close(out_local, out_full[start:end].detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_local.grad, x_full.grad[start:end], atol=atol, rtol=rtol)
    for name, p in processor.named_parameters():
        grad = p.grad.clone()
        dist.all_reduce(grad, group=group)
        torch.testing.assert_close(grad, grads_full[name], atol=grad_atol, rtol=grad_rtol, msg=f"gradient of {name}")


@pytest.mark.distributed
@pytest.mark.parametrize("block", ["edge_conv", "transformer", "gnn"])
@pytest.mark.parametrize("cartesian", [False, True], ids=["local-displacement", "cartesian-displacement"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_mesh_processor_matches_single_gpu(block, cartesian, dtype, distributed_backend, distributed_world_size):
    if dtype == torch.float64 and distributed_backend == "nccl":
        pytest.skip("The float64 comparison runs on CPU.")
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall_transpose requires torch >= 2.6.")
    _run_distributed_test(
        _mesh_processor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        block=block,
        cartesian=cartesian,
        dtype=dtype,
    )
