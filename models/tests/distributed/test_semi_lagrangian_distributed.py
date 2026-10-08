# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Semi-Lagrangian blocks split over several GPUs must match the same blocks on one GPU.

Parameter gradients are compared after summing them over the GPUs, which is what training does
with model-parallel parameters.
"""

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.block import FlowersProcessorBlock
from anemoi.models.layers.utils import load_layer_kernels

from ._distributed_runner import _run_distributed_test
from .distributed_test_utils import torch_version_less_than


def _build_block(
    kind: str, nlat: int, nlon: int, num_heads: int, cartesian: bool, device: torch.device
) -> torch.nn.Module:
    torch.manual_seed(0)
    if kind == "flowers":
        return FlowersProcessorBlock(
            num_channels=12,
            hidden_dim=24,
            num_heads=num_heads,
            nlat=nlat,
            nlon=nlon,
            layer_kernels=load_layer_kernels(),
            cartesian_displacement=cartesian,
        ).to(device)
    block = ADRProcessorBlock(
        num_channels=16,
        advection_channels=12,
        num_heads=num_heads,
        velocity_hidden_dim=10,
        reaction_hidden_dim=24,
        reaction_num_layers=3,
        nlat=nlat,
        nlon=nlon,
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
        bias_rank=4,
        bias_base_maps=2,
        cartesian_displacement=cartesian,
    ).to(device)
    # The bias fields start close to zero; give them values so that every path is tested.
    for bias in (block.advection.velocity_bias, block.diffusion_bias, block.reaction_bias):
        for p in (bias.coefficients, bias.lat_profiles, bias.lon_profiles):
            torch.nn.init.normal_(p)
    return block


def _block_rank(rank, world_size, device, group, kind, cartesian, dtype, nlat, nlon, num_heads):
    grid = nlat * nlon
    block = _build_block(kind, nlat, nlon, num_heads, cartesian, device).to(dtype)
    channels = 16 if kind == "adr" else 12
    # float64 shows that splitting changes nothing beyond rounding; float32 is what training uses.
    atol, rtol = (1e-11, 1e-9) if dtype == torch.float64 else (1e-5, 1e-5)
    grad_atol, grad_rtol = (1e-10, 1e-9) if dtype == torch.float64 else (1e-4, 1e-4)

    torch.manual_seed(1)
    x = torch.randn(grid, channels, device=device, dtype=dtype)
    grad_out = torch.randn(grid, channels, device=device, dtype=dtype)

    x_full = x.clone().requires_grad_()
    out_full = block(x_full, GraphShardInfo(nodes=[grid]), 1)[0]
    out_full.backward(grad_out)
    grads_full = {name: p.grad.clone() for name, p in block.named_parameters()}
    block.zero_grad()

    shard_sizes = get_balanced_partition_sizes(grid, world_size)
    start, end = get_partition_range(shard_sizes, rank)
    x_local = x[start:end].clone().requires_grad_()
    out_local = block(x_local, GraphShardInfo(nodes=shard_sizes), 1, model_comm_group=group)[0]
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
@pytest.mark.parametrize(
    ("nlat", "nlon", "num_heads"),
    [
        pytest.param(6, 12, 12, id="one-velocity-per-channel"),
        pytest.param(7, 10, 3, id="grouped-velocities-uneven-split"),
    ],
)
def test_block_matches_single_gpu(
    kind, cartesian, dtype, nlat, nlon, num_heads, distributed_backend, distributed_world_size
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
    )
