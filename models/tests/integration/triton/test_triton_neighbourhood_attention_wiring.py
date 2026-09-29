# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The Triton kernels used through the model layers, with nodes stored in any order."""

import math

import pytest
import torch

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttentionWrapper
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.utils import load_layer_kernels

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def shuffled_coords(grid: ReducedGrid, seed: int) -> torch.Tensor:
    perm = torch.randperm(grid.num_points, generator=torch.Generator().manual_seed(seed))
    return grid_coords(grid)[perm]


def _run(module, *inputs):
    inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = module(*inputs)
    out = out[1] if isinstance(out, tuple) else out
    out.backward(torch.ones_like(out) / out.numel() ** 0.5)
    return [out.detach()] + [t.grad for t in inputs]


@pytest.mark.parametrize(
    ("family", "key_grid", "query_grid"),
    [
        ("octahedral", ReducedGrid.octahedral(16), None),
        ("healpix", ReducedGrid.healpix(8), None),
        ("octahedral", ReducedGrid.octahedral(16), ReducedGrid.octahedral(8)),
        ("healpix", ReducedGrid.healpix(4), ReducedGrid.healpix(8)),
    ],
)
def test_wrapper_on_shuffled_nodes_matches_dense_mask(family, key_grid, query_grid):
    key_coords = shuffled_coords(key_grid, seed=0)
    query_coords = None if query_grid is None else shuffled_coords(query_grid, seed=1)
    num_queries = (query_grid or key_grid).num_points

    results = {}
    for backend in ("triton", "sdpa"):
        config = {"grid": family, "kernel_size": [3, 7], "backend": backend}
        wrapper = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(config, key_coords, query_coords)).cuda()
        generator = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(2, 3, num_queries, 32, generator=generator, device="cuda")
        k, v = (torch.randn(2, 3, key_grid.num_points, 32, generator=generator, device="cuda") for _ in range(2))
        results[backend] = _run(lambda q, k, v: wrapper(q, k, v, 2), q, k, v)

    for got, expected in zip(results["triton"], results["sdpa"]):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def _processor(backend: str, coords: torch.Tensor) -> TransformerProcessor:
    torch.manual_seed(0)
    return TransformerProcessor(
        num_layers=2,
        num_channels=64,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        attention_implementation="neighbourhood",
        neighbourhood={"grid": "healpix", "kernel_size": [5, 9], "backend": backend},
        node_coords=coords,
        layer_kernels=load_layer_kernels(instance=False),
    ).cuda()


def test_processor_with_triton_matches_dense_mask():
    grid = ReducedGrid.healpix(8)
    coords = shuffled_coords(grid, seed=2)
    x = torch.randn(2 * grid.num_points, 64, device="cuda")
    shard_info = GraphShardInfo(nodes=[2 * grid.num_points])

    triton = _run(lambda x: _processor("triton", coords)(x, 2, shard_info), x)
    dense = _run(lambda x: _processor("sdpa", coords)(x, 2, shard_info), x)
    for got, expected in zip(triton, dense):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def _encoder(backend: str, data_coords, hidden_coords) -> TransformerForwardMapper:
    torch.manual_seed(0)
    return TransformerForwardMapper(
        in_channels_src=5,
        in_channels_dst=6,
        num_channels=64,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        attention_implementation="neighbourhood",
        neighbourhood={"grid": "octahedral", "kernel_size": [5, 5], "backend": backend},
        src_node_coords=data_coords,
        dst_node_coords=hidden_coords,
        layer_kernels=load_layer_kernels(instance=False),
    ).cuda()


def test_encoder_with_triton_matches_dense_mask():
    data, hidden = ReducedGrid.octahedral(24), ReducedGrid.octahedral(12)
    data_coords, hidden_coords = shuffled_coords(data, seed=3), grid_coords(hidden)
    x_src = torch.randn(data.num_points, 5, device="cuda")
    x_dst = torch.randn(hidden.num_points, 6, device="cuda")
    shard_info = BipartiteGraphShardInfo(src_nodes=[data.num_points], dst_nodes=[hidden.num_points])

    triton = _run(lambda s, d: _encoder("triton", data_coords, hidden_coords)((s, d), 1, shard_info), x_src, x_dst)
    dense = _run(lambda s, d: _encoder("sdpa", data_coords, hidden_coords)((s, d), 1, shard_info), x_src, x_dst)
    for got, expected in zip(triton, dense):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
