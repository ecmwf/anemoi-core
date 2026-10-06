# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention layers with the points split across GPUs give the same results as on one GPU.

Every rank builds the same layer and inputs, runs it once on its own over the whole grids (the
reference) and once with the points split across the ranks, and compares outputs, input
gradients and the parameter gradients summed over the ranks. On CPU (gloo) the layers run in
float64 with the dense-mask backend; on GPU (nccl) in float32 with the Triton kernels.

The grids are large enough that every rank holds several rows, so its bands lie both inside its
own points and at the edges of its shard. The points are split three ways: in the balanced runs
the model uses (shard edges inside rows), at row boundaries (a rank may then need nothing from its
neighbours for one of the inputs), and unevenly with one tiny shard.

Run with ``pytest --distributed [--distributed-world-size N] [--distributed-backend nccl]``.
"""

import math

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.utils import load_layer_kernels

NUM_CHANNELS = 32
COND_CHANNELS = 3

CASES = {
    # name: (kind, grid family, data grid, hidden grid, kernel size)
    "processor octahedral": ("processor", "octahedral", None, ReducedGrid.octahedral(32), (7, 13)),
    "processor healpix": ("processor", "healpix", None, ReducedGrid.healpix(16), (5, 5)),
    "encoder octahedral": ("encoder", "octahedral", ReducedGrid.octahedral(32), ReducedGrid.octahedral(16), (5, 5)),
    "encoder healpix": ("encoder", "healpix", ReducedGrid.healpix(16), ReducedGrid.healpix(8), (5, 5)),
    "decoder octahedral": ("decoder", "octahedral", ReducedGrid.octahedral(32), ReducedGrid.octahedral(16), (3, 5)),
    "decoder healpix": ("decoder", "healpix", ReducedGrid.healpix(16), ReducedGrid.healpix(8), (3, 5)),
}

# Combinations of settings, so that each one is tried with the others varied.
VARIANTS = {
    "1 band, balanced": dict(num_bands=1, conditional=False, split="balanced", keep_x_dst_sharded=True),
    "3 bands, balanced, conditional, gathered": dict(
        num_bands=3, conditional=True, split="balanced", keep_x_dst_sharded=False
    ),
    "8 bands, balanced": dict(num_bands=8, conditional=False, split="balanced", keep_x_dst_sharded=True),
    "8 bands, at row boundaries, conditional": dict(
        num_bands=8, conditional=True, split="rows", keep_x_dst_sharded=True
    ),
    "1 band, at row boundaries, conditional, gathered": dict(
        num_bands=1, conditional=True, split="rows", keep_x_dst_sharded=False
    ),
    "3 bands, uneven, gathered": dict(num_bands=3, conditional=False, split="uneven", keep_x_dst_sharded=False),
}


def _grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def split_points(grid: ReducedGrid, world_size: int, how: str) -> list[int]:
    """Number of points per rank: ``balanced`` as the model splits, at ``rows`` boundaries, or ``uneven``."""
    num_points = grid.num_points
    if how == "balanced":
        return get_balanced_partition_sizes(num_points, world_size)
    if how == "rows":
        row_starts = grid.row_starts
        targets = torch.tensor([k * num_points / world_size for k in range(1, world_size)], dtype=torch.float64)
        cuts = [int(row_starts[int((row_starts.double() - t).abs().argmin())]) for t in targets]
    elif how == "uneven":
        # The first rank gets a sliver of the first row; the others grow towards the south.
        cuts = [3] + [round(num_points * (k / world_size) ** 1.5) for k in range(2, world_size)]
    else:
        raise ValueError(how)
    edges = [0, *cuts, num_points]
    sizes = [b - a for a, b in zip(edges[:-1], edges[1:])]
    assert all(s > 0 for s in sizes), (how, sizes)
    return sizes


def _layer_kernels(conditional: bool):
    if not conditional:
        return load_layer_kernels(instance=False)
    return load_layer_kernels(
        kernel_config={
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "condition_shape": COND_CHANNELS,
                "zero_init": False,
            }
        },
        instance=False,
    )


def _build(case: str, num_bands: int, backend: str, conditional: bool):
    kind, family, data, hidden, kernel_size = CASES[case]
    common = dict(
        num_channels=NUM_CHANNELS,
        num_heads=2,
        mlp_hidden_ratio=2,
        qk_norm=True,
        attention_implementation="neighbourhood",
        neighbourhood={"grid": family, "kernel_size": list(kernel_size), "backend": backend, "num_bands": num_bands},
        rotary_embeddings={"max_frequency": 20, "backend": "torch" if backend == "sdpa" else "triton"},
        layer_kernels=_layer_kernels(conditional),
    )
    if kind == "processor":
        layer = TransformerProcessor(num_layers=2, num_chunks=2, node_coords=_grid_coords(hidden), **common)
        layer.gradient_checkpointing = True
        return layer
    common["hidden_dim"] = common.pop("num_channels")
    if kind == "encoder":
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


def _grids(case: str) -> tuple[ReducedGrid, ReducedGrid]:
    """Source and destination grid of the layer."""
    kind, _, data, hidden, _ = CASES[case]
    if kind == "processor":
        return hidden, hidden
    return (data, hidden) if kind == "encoder" else (hidden, data)


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


def _assert_close(actual, expected, tolerance, name, scale_to_largest=False):
    atol = tolerance["atol"] * (max(1.0, expected.abs().max().item()) if scale_to_largest else 1.0)
    torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=atol, msg=lambda m: f"{name}: {m}")


def _split_matches_one_gpu_rank(
    rank, world_size, device, group, case, num_bands, conditional, split, keep_x_dst_sharded
):
    if device.type == "cpu":
        dtype, backend, tolerance = torch.float64, "sdpa", dict(rtol=1e-10, atol=1e-10)
    else:
        dtype, backend, tolerance = torch.float32, "triton", dict(rtol=1e-4, atol=1e-5)
    kind = CASES[case][0]
    src_grid, dst_grid = _grids(case)
    num_src, num_dst = src_grid.num_points, dst_grid.num_points
    out_channels = 4 if kind == "decoder" else NUM_CHANNELS

    torch.manual_seed(0)
    one_gpu = _build(case, 1, backend, conditional)
    split_layer = _build(case, num_bands, backend, conditional)
    split_layer.load_state_dict(one_gpu.state_dict())
    one_gpu, split_layer = one_gpu.to(device, dtype), split_layer.to(device, dtype)

    generator = torch.Generator().manual_seed(1)

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(device, dtype)

    if kind == "processor":
        full = {"x": rand(num_dst, NUM_CHANNELS)}
        if conditional:
            full["cond"] = rand(num_dst, COND_CHANNELS)
    else:
        full = {"src": rand(num_src, 5 if kind == "encoder" else NUM_CHANNELS), "dst": rand(num_dst, 6)}
        if conditional:
            full["cond"] = (rand(num_src, COND_CHANNELS), rand(num_dst, COND_CHANNELS))
    upstream = rand(num_dst, out_channels)

    def leaf(t):
        return tuple(x.clone().requires_grad_() for x in t) if isinstance(t, tuple) else t.clone().requires_grad_()

    # The reference: this rank alone, over the whole grids.
    ref_inputs = {name: leaf(value) for name, value in full.items()}
    ref_out = _run(case, one_gpu, ref_inputs, [num_src], [num_dst], None, False)
    (ref_out * upstream).sum().backward()

    # The points split across the ranks; each rank passes in its own points.
    src_sizes = split_points(src_grid, world_size, split)
    dst_sizes = split_points(dst_grid, world_size, split)
    src_own = slice(sum(src_sizes[:rank]), sum(src_sizes[: rank + 1]))
    dst_own = slice(sum(dst_sizes[:rank]), sum(dst_sizes[: rank + 1]))
    own_of = {"x": dst_own, "src": src_own, "dst": dst_own}
    split_inputs = {}
    for name, value in full.items():
        if name == "cond" and kind != "processor":
            split_inputs[name] = leaf((value[0][src_own], value[1][dst_own]))
        elif name == "cond":
            split_inputs[name] = leaf(value[dst_own])
        else:
            split_inputs[name] = leaf(value[own_of[name]])
    out = _run(case, split_layer, split_inputs, src_sizes, dst_sizes, group, keep_x_dst_sharded)

    if kind == "processor" or keep_x_dst_sharded:
        _assert_close(out, ref_out[dst_own].detach(), tolerance, "output")
        (out * upstream[dst_own]).sum().backward()
    else:
        _assert_close(out, ref_out.detach(), tolerance, "output")
        (out * upstream).sum().backward()

    for name, value in split_inputs.items():
        if name == "cond" and kind != "processor":
            _assert_close(value[0].grad, ref_inputs[name][0].grad[src_own], tolerance, "cond src grad")
            _assert_close(value[1].grad, ref_inputs[name][1].grad[dst_own], tolerance, "cond dst grad")
        elif name == "cond":
            _assert_close(value.grad, ref_inputs[name].grad[dst_own], tolerance, "cond grad")
        else:
            _assert_close(value.grad, ref_inputs[name].grad[own_of[name]], tolerance, f"{name} grad")

    # Each rank holds the parameter gradient of its own points; together they make the whole. A
    # parameter gradient is a sum over all points, added up in another order when split, so its
    # rounding error scales with its largest entries rather than with each entry.
    ref_params = dict(one_gpu.named_parameters())
    without_grad = set()
    for name, param in split_layer.named_parameters():
        assert (param.grad is None) == (ref_params[name].grad is None), name
        if param.grad is None:
            without_grad.add(name)
            continue
        grad = param.grad.clone()
        dist.all_reduce(grad, group=group)
        _assert_close(grad, ref_params[name].grad, tolerance, name, scale_to_largest=True)
    # A mapper block keeps the two layer norms of the self-attention block it builds on and never uses them.
    assert all(n.split(".")[1] in ("layer_norm_attention", "layer_norm_mlp") for n in without_grad), without_grad
    assert kind != "processor" or not without_grad, without_grad


@pytest.mark.distributed
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_split_across_gpus_matches_one_gpu(
    case: str, variant: str, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _split_matches_one_gpu_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        case=case,
        **VARIANTS[variant],
    )


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("how", ["balanced", "rows", "uneven"])
@pytest.mark.parametrize("case", CASES)
def test_the_splits_cover_every_point_once(case: str, how: str, world_size: int) -> None:
    for grid in _grids(case):
        sizes = split_points(grid, world_size, how)
        assert len(sizes) == world_size and sum(sizes) == grid.num_points and min(sizes) > 0
        if how == "rows":
            starts = set(grid.row_starts.tolist())
            assert all(sum(sizes[:k]) in starts for k in range(world_size + 1))
