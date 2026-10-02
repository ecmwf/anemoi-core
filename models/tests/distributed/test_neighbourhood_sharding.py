# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention with the points split across GPUs gives the same results as on one GPU.

Every rank builds the same layer and inputs, runs it once on its own over the whole grids (the
reference) and once with the points split across the ranks, and compares outputs, input
gradients and the parameter gradients summed over the ranks. On CPU (gloo) the layers run in
float64 with the dense-mask backend; on GPU (nccl) in float32 with the Triton kernels.

Run with ``pytest --distributed [--distributed-world-size N] [--distributed-backend nccl]``.
"""

import math

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.utils import load_layer_kernels
from tests.distributed._distributed_runner import _run_distributed_test

NUM_CHANNELS = 32
COND_CHANNELS = 3

CASES = {
    # name: (kind, grid family, data grid, hidden grid, kernel size)
    "processor octahedral": ("processor", "octahedral", None, ReducedGrid.octahedral(16), (7, 13)),
    "processor healpix": ("processor", "healpix", None, ReducedGrid.healpix(8), (5, 5)),
    "encoder octahedral": ("encoder", "octahedral", ReducedGrid.octahedral(16), ReducedGrid.octahedral(8), (5, 5)),
    "encoder healpix": ("encoder", "healpix", ReducedGrid.healpix(8), ReducedGrid.healpix(4), (5, 5)),
    "decoder octahedral": ("decoder", "octahedral", ReducedGrid.octahedral(16), ReducedGrid.octahedral(8), (3, 5)),
    "decoder healpix": ("decoder", "healpix", ReducedGrid.healpix(8), ReducedGrid.healpix(4), (3, 5)),
}


def _grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def _build(case: str, num_bands: int, backend: str, conditional: bool):
    kind, family, data, hidden, kernel_size = CASES[case]
    layer_kernels = load_layer_kernels(instance=False)
    if conditional:
        layer_kernels = load_layer_kernels(
            kernel_config={
                "LayerNorm": {
                    "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                    "condition_shape": COND_CHANNELS,
                    "zero_init": False,
                }
            },
            instance=False,
        )
    common = dict(
        num_channels=NUM_CHANNELS,
        num_heads=2,
        mlp_hidden_ratio=2,
        qk_norm=True,
        attention_implementation="neighbourhood",
        neighbourhood={"grid": family, "kernel_size": list(kernel_size), "backend": backend, "num_bands": num_bands},
        rotary_embeddings={"max_frequency": 20, "backend": "torch" if backend == "sdpa" else "triton"},
        layer_kernels=layer_kernels,
    )
    if kind == "processor":
        layer = TransformerProcessor(num_layers=2, num_chunks=2, node_coords=_grid_coords(hidden), **common)
    elif kind == "encoder":
        layer = TransformerForwardMapper(
            in_channels_src=5,
            in_channels_dst=6,
            num_chunks=1,
            src_node_coords=_grid_coords(data),
            dst_node_coords=_grid_coords(hidden),
            **common,
        )
    else:
        layer = TransformerBackwardMapper(
            in_channels_src=NUM_CHANNELS,
            in_channels_dst=6,
            out_channels_dst=4,
            num_chunks=1,
            src_node_coords=_grid_coords(hidden),
            dst_node_coords=_grid_coords(data),
            **common,
        )
    layer.gradient_checkpointing = True
    return layer


def _point_counts(case: str) -> tuple[int, int, int]:
    """Number of source points, destination points and output channels of the layer."""
    kind, _, data, hidden, _ = CASES[case]
    if kind == "processor":
        return hidden.num_points, hidden.num_points, NUM_CHANNELS
    if kind == "encoder":
        return data.num_points, hidden.num_points, NUM_CHANNELS
    return hidden.num_points, data.num_points, 4


def _run(case, layer, inputs, src_sizes, dst_sizes, group, keep_x_dst_sharded):
    """Forward pass; the sizes are the points each rank holds (one entry when not split)."""
    kind = CASES[case][0]
    kwargs = {"cond": inputs["cond"]} if "cond" in inputs else {}
    if kind == "processor":
        return layer(inputs["x"], 1, GraphShardInfo(nodes=dst_sizes), model_comm_group=group, **kwargs)
    shard_info = BipartiteGraphShardInfo(src_nodes=src_sizes, dst_nodes=dst_sizes)
    out = layer(
        (inputs["src"], inputs["dst"]),
        1,
        shard_info,
        model_comm_group=group,
        keep_x_dst_sharded=keep_x_dst_sharded,
        **kwargs,
    )
    return out[1] if kind == "encoder" else out


def _split_matches_one_gpu_rank(rank, world_size, device, group, case, num_bands, conditional, keep_x_dst_sharded):
    if device.type == "cpu":
        dtype, backend, tolerance = torch.float64, "sdpa", dict(rtol=1e-10, atol=1e-10)
    else:
        dtype, backend, tolerance = torch.float32, "triton", dict(rtol=1e-4, atol=1e-5)
    kind = CASES[case][0]
    num_src, num_dst, out_channels = _point_counts(case)

    torch.manual_seed(0)
    one_gpu = _build(case, 1, backend, conditional)
    split = _build(case, num_bands, backend, conditional)
    split.load_state_dict(one_gpu.state_dict())
    one_gpu, split = one_gpu.to(device, dtype), split.to(device, dtype)

    generator = torch.Generator().manual_seed(1)
    if kind == "processor":
        shapes = {"x": (num_dst, NUM_CHANNELS)}
        if conditional:
            shapes["cond"] = (num_dst, COND_CHANNELS)
    else:
        src_channels = 5 if kind == "encoder" else NUM_CHANNELS
        shapes = {"src": (num_src, src_channels), "dst": (num_dst, 6)}
        if conditional:
            shapes["cond"] = ((num_src, COND_CHANNELS), (num_dst, COND_CHANNELS))
    full = {}
    for name, shape in shapes.items():
        if name == "cond" and kind != "processor":
            full[name] = tuple(
                torch.randn(s, generator=generator, dtype=torch.float64).to(device, dtype) for s in shape
            )
        else:
            full[name] = torch.randn(shape, generator=generator, dtype=torch.float64).to(device, dtype)
    upstream = torch.randn(num_dst, out_channels, generator=generator, dtype=torch.float64).to(device, dtype)

    # The reference: this rank alone, over the whole grids.
    ref_inputs = {
        n: tuple(t.clone().requires_grad_() for t in v) if isinstance(v, tuple) else v.clone().requires_grad_()
        for n, v in full.items()
    }
    ref_out = _run(case, one_gpu, ref_inputs, [num_src], [num_dst], None, False)
    (ref_out * upstream).sum().backward()

    # The points split across the ranks; each rank passes in its own points.
    src_sizes = get_balanced_partition_sizes(num_src, world_size)
    dst_sizes = get_balanced_partition_sizes(num_dst, world_size)
    src_own = slice(sum(src_sizes[:rank]), sum(src_sizes[: rank + 1]))
    dst_own = slice(sum(dst_sizes[:rank]), sum(dst_sizes[: rank + 1]))
    own_of = {"x": dst_own, "src": src_own, "dst": dst_own}
    split_inputs = {}
    for name, value in full.items():
        if name == "cond" and kind != "processor":
            split_inputs[name] = (
                value[0][src_own].clone().requires_grad_(),
                value[1][dst_own].clone().requires_grad_(),
            )
        elif name == "cond":
            split_inputs[name] = value[dst_own].clone().requires_grad_()
        else:
            split_inputs[name] = value[own_of[name]].clone().requires_grad_()
    out = _run(case, split, split_inputs, src_sizes, dst_sizes, group, keep_x_dst_sharded)

    output_is_split = kind == "processor" or keep_x_dst_sharded
    if output_is_split:
        torch.testing.assert_close(out, ref_out[dst_own].detach(), **tolerance)
        (out * upstream[dst_own]).sum().backward()
    else:
        torch.testing.assert_close(out, ref_out.detach(), **tolerance)
        (out * upstream).sum().backward()

    for name, value in split_inputs.items():
        if name == "cond" and kind != "processor":
            torch.testing.assert_close(value[0].grad, ref_inputs[name][0].grad[src_own], **tolerance)
            torch.testing.assert_close(value[1].grad, ref_inputs[name][1].grad[dst_own], **tolerance)
        elif name == "cond":
            torch.testing.assert_close(value.grad, ref_inputs[name].grad[dst_own], **tolerance)
        else:
            torch.testing.assert_close(
                value.grad, ref_inputs[name].grad[own_of[name]], msg=lambda m, name=name: f"{name}: {m}", **tolerance
            )

    # Each rank holds the parameter gradient of its own points; together they make the whole. A
    # parameter gradient is a sum over all points, added up in another order when split, so its
    # rounding error scales with its largest entries rather than with each entry.
    ref_params = dict(one_gpu.named_parameters())
    for name, param in split.named_parameters():
        if param.grad is None:
            assert ref_params[name].grad is None, name
            continue
        grad = param.grad.clone()
        dist.all_reduce(grad, group=group)
        expected = ref_params[name].grad
        torch.testing.assert_close(
            grad,
            expected,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"] * max(1.0, expected.abs().max().item()),
            msg=lambda m, name=name: f"{name}: {m}",
        )


@pytest.mark.distributed
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("num_bands", [1, 3])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("keep_x_dst_sharded", [True, False])
def test_split_across_gpus_matches_one_gpu(
    case: str,
    num_bands: int,
    conditional: bool,
    keep_x_dst_sharded: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    if CASES[case][0] == "processor" and not keep_x_dst_sharded:
        pytest.skip("The processor output always stays split.")
    _run_distributed_test(
        _split_matches_one_gpu_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        case=case,
        num_bands=num_bands,
        conditional=conditional,
        keep_x_dst_sharded=keep_x_dst_sharded,
    )
