# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention worked through bands of query rows gives the same results as on the whole grid."""

import itertools
import math
from dataclasses import replace

import pytest
import torch

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.point_ranges import PointRun
from anemoi.models.distributed.point_ranges import assemble_point_run
from anemoi.models.distributed.point_ranges import build_point_range_exchange
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodBands
from anemoi.models.layers.neighbourhood_attention import split_into_bands
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.reduced_grid import ReducedGridCrossNeighbourhoodMask
from anemoi.models.layers.reduced_grid import ReducedGridNeighbourhoodMask
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.schemas.common_components import NeighbourhoodSchema

NUM_CHANNELS = 32
COND_CHANNELS = 3


def grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def dense(rule) -> torch.Tensor:
    """The whole query x key mask of a neighbourhood rule."""
    q_idx = torch.arange(len(rule.q_rows))[:, None]
    k_idx = torch.arange(len(rule.k_rows))[None, :]
    return rule(None, None, q_idx, k_idx)


def neighbourhood(family: str, query_grid: ReducedGrid, key_grid: ReducedGrid | None, kernel_size) -> GridNeighbourhood:
    config = {"grid": family, "kernel_size": list(kernel_size), "backend": "sdpa"}
    if key_grid is None:
        return GridNeighbourhood.from_config(config, grid_coords(query_grid))
    return GridNeighbourhood.from_config(config, grid_coords(key_grid), grid_coords(query_grid))


CASES = {
    "octahedral self": ("octahedral", ReducedGrid.octahedral(16), None, (7, 13)),
    "octahedral self small kernel": ("octahedral", ReducedGrid.octahedral(12), None, (3, 5)),
    "octahedral coarse to fine": ("octahedral", ReducedGrid.octahedral(24), ReducedGrid.octahedral(12), (5, 5)),
    "octahedral fine to coarse": ("octahedral", ReducedGrid.octahedral(8), ReducedGrid.octahedral(24), (5, 7)),
    "healpix self": ("healpix", ReducedGrid.healpix(8), None, (5, 5)),
    "healpix fine to coarse": ("healpix", ReducedGrid.healpix(4), ReducedGrid.healpix(8), (5, 5)),
    "healpix coarse to fine": ("healpix", ReducedGrid.healpix(8), ReducedGrid.healpix(4), (3, 5)),
    "octahedral self O32": ("octahedral", ReducedGrid.octahedral(32), None, (7, 13)),
    "octahedral O32 to O16": ("octahedral", ReducedGrid.octahedral(16), ReducedGrid.octahedral(32), (5, 5)),
    "healpix self H16": ("healpix", ReducedGrid.healpix(16), None, (5, 5)),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("num_bands", [2, 3, 7, 1000])
def test_every_query_sees_the_same_keys_in_its_band(case: str, num_bands: int) -> None:
    family, query_grid, key_grid, kernel_size = CASES[case]
    nb = neighbourhood(family, query_grid, key_grid, kernel_size)
    if nb.is_self_attention:
        full = dense(ReducedGridNeighbourhoodMask(nb.query_grid, nb.kernel_size))
    else:
        full = dense(ReducedGridCrossNeighbourhoodMask(nb.query_grid, nb.key_grid, nb.kernel_size))

    bands = split_into_bands(nb, num_bands)

    assert len(bands) == min(num_bands, nb.query_grid.num_rows)
    covered = torch.zeros(nb.query_grid.num_points, dtype=torch.long)
    for band in bands:
        covered[band.query_points] += 1
        band_nb = band.attention.neighbourhood
        band_mask = dense(ReducedGridCrossNeighbourhoodMask(band_nb.query_grid, band_nb.key_grid, band_nb.kernel_size))
        assert torch.equal(band_mask, full[band.query_points, band.key_points])
        outside = full[band.query_points].clone()
        outside[:, band.key_points] = False
        assert not outside.any(), "a query attends to a key outside its band's key rows"
    assert torch.equal(covered, torch.ones_like(covered))


def test_bands_hold_about_the_same_number_of_points() -> None:
    nb = neighbourhood("octahedral", ReducedGrid.octahedral(96), None, (7, 13))
    bands = split_into_bands(nb, 16)
    sizes = [band.query_points.stop - band.query_points.start for band in bands]
    assert len(bands) == 16
    assert max(sizes) - min(sizes) <= 2 * max(nb.query_grid.row_lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU as the default device")
@pytest.mark.parametrize("case", ["octahedral self", "octahedral coarse to fine"])
def test_bands_are_planned_with_the_gpu_as_default_device(case: str) -> None:
    # anemoi-inference runs the model with the GPU as the default device, while the grids the
    # bands are cut from stay on the CPU.
    family, query_grid, key_grid, kernel_size = CASES[case]
    nb = replace(neighbourhood(family, query_grid, key_grid, kernel_size), num_bands=3)
    sizes = [7, nb.query_grid.num_points - 7]
    expected = [(plan.query_points, plan.key_points) for plan in NeighbourhoodBands(nb).plans(sizes)]
    with torch.device("cuda"):
        plans = NeighbourhoodBands(nb).plans(sizes)
    assert [(plan.query_points, plan.key_points) for plan in plans] == expected
    assert all(len(plan.bands) > 1 for plan in plans[1:])


def test_bands_need_nodes_in_grid_order() -> None:
    grid = ReducedGrid.octahedral(8)
    coords = grid_coords(grid)[torch.randperm(grid.num_points, generator=torch.Generator().manual_seed(0))]
    config = {"grid": "octahedral", "kernel_size": [3, 5], "backend": "sdpa", "num_bands": 2}
    with pytest.raises(ValueError, match="grid order"):
        GridNeighbourhood.from_config(config, coords)


def test_schema_takes_the_number_of_bands() -> None:
    assert NeighbourhoodSchema(grid="octahedral", kernel_size=(3, 5)).num_bands == 1
    assert NeighbourhoodSchema(grid="octahedral", kernel_size=(3, 5), num_bands=8).num_bands == 8


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


# Bands against the whole grid: on CPU in float64 with the dense-mask backend, which must agree to
# rounding; on GPU in float32 with the Triton kernels, where the bands use the cross-attention
# kernel and the whole grid the self-attention one.
SETTINGS = {
    "sdpa": dict(device="cpu", dtype=torch.float64, rotary="torch", rtol=1e-10, atol=1e-10),
    "triton": dict(device="cuda", dtype=torch.float32, rotary="triton", rtol=1e-4, atol=1e-5),
}
# A mapper block keeps the two layer norms of the self-attention block it builds on and never uses them.
UNUSED_PARAMETERS = {
    f"proc.{norm}.{p}" for norm in ("layer_norm_attention", "layer_norm_mlp") for p in ("weight", "bias")
}
UNUSED_PARAMETERS |= {
    f"proc.{norm}.{lin}.{p}"
    for norm in ("layer_norm_attention", "layer_norm_mlp")
    for lin in ("scale", "bias")
    for p in ("weight", "bias")
}
BACKEND_PARAMS = [
    "sdpa",
    pytest.param("triton", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")),
]


def _neighbourhood_config(family: str, kernel_size, num_bands: int, backend: str = "sdpa") -> dict:
    return {"grid": family, "kernel_size": list(kernel_size), "backend": backend, "num_bands": num_bands}


def _same_results(make, run, num_bands: int, gradient_checkpointing: bool, backend: str) -> None:
    """Build the layer without and with bands, with the same weights, and compare outputs and all gradients."""
    settings = SETTINGS[backend]
    torch.manual_seed(0)
    whole = make(1).to(settings["device"], settings["dtype"])
    banded = make(num_bands).to(settings["device"], settings["dtype"])
    banded.load_state_dict(whole.state_dict())
    assert banded.bands is not None and len(banded.bands.plans()[0].bands) >= min(num_bands, 4)
    for layer in (whole, banded):
        layer.gradient_checkpointing = gradient_checkpointing

    inputs = run.inputs(settings)
    out_whole, grads_whole = run.forward_backward(whole, inputs)
    out_banded, grads_banded = run.forward_backward(banded, inputs)

    rtol, atol = settings["rtol"], settings["atol"]
    torch.testing.assert_close(out_banded, out_whole, rtol=rtol, atol=atol)
    assert grads_banded.keys() == grads_whole.keys()
    without_grad = {name for name, _ in whole.named_parameters()} - grads_whole.keys()
    assert without_grad <= UNUSED_PARAMETERS, f"parameters without gradient: {sorted(without_grad)}"
    for name in grads_whole:
        # A parameter gradient sums over all points, so its rounding error scales with its largest entries.
        scale = max(1.0, grads_whole[name].abs().max().item()) if not name.startswith("input") else 1.0
        torch.testing.assert_close(
            grads_banded[name], grads_whole[name], rtol=rtol, atol=atol * scale, msg=lambda m, n=name: f"{n}: {m}"
        )


class _Run:
    """Random inputs for a layer, and a forward and backward pass that collects every gradient."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], call):
        self.shapes = shapes
        self.call = call

    def inputs(self, settings: dict) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(1)
        return {
            name: torch.randn(shape, generator=generator, dtype=torch.float64).to(settings["device"], settings["dtype"])
            for name, shape in self.shapes.items()
        }

    def forward_backward(self, layer, inputs):
        layer.zero_grad()
        inputs = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
        out = self.call(layer, inputs)
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        (out * weights.to(out.device, out.dtype)).sum().backward()
        grads = {f"input {name}": t.grad for name, t in inputs.items()}
        grads |= {name: p.grad for name, p in layer.named_parameters() if p.grad is not None}
        return out.detach(), grads


MAPPER_CASES = {
    "octahedral": ("octahedral", ReducedGrid.octahedral(32), ReducedGrid.octahedral(16), (5, 7), (3, 5)),
    "healpix": ("healpix", ReducedGrid.healpix(16), ReducedGrid.healpix(8), (5, 5), (3, 5)),
}


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("case", MAPPER_CASES)
@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_encoder_and_decoder_in_bands_match_the_whole_grid(
    case: str, gradient_checkpointing: bool, conditional: bool, backend: str
):
    family, data, hidden, encoder_kernel, decoder_kernel = MAPPER_CASES[case]
    batch = 2

    def mapper_kwargs(num_bands, kernel_size):
        return dict(
            in_channels_src=5,
            in_channels_dst=6,
            num_channels=NUM_CHANNELS,
            num_chunks=1,
            num_heads=2,
            mlp_hidden_ratio=2,
            qk_norm=True,
            attention_implementation="neighbourhood",
            neighbourhood=_neighbourhood_config(family, kernel_size, num_bands, backend),
            rotary_embeddings={"max_frequency": 20, "backend": SETTINGS[backend]["rotary"]},
            layer_kernels=_layer_kernels(conditional),
        )

    def make_encoder(num_bands):
        return TransformerForwardMapper(
            **mapper_kwargs(num_bands, encoder_kernel),
            src_node_coords=grid_coords(data),
            dst_node_coords=grid_coords(hidden),
        )

    def make_decoder(num_bands):
        return TransformerBackwardMapper(
            **mapper_kwargs(num_bands, decoder_kernel),
            out_channels_dst=4,
            src_node_coords=grid_coords(hidden),
            dst_node_coords=grid_coords(data),
        )

    def call(src_points, dst_points, returns_pair):
        def run(layer, t):
            kwargs = {"cond": (t["cond_src"], t["cond_dst"])} if conditional else {}
            shard_info = BipartiteGraphShardInfo(src_nodes=[batch * src_points], dst_nodes=[batch * dst_points])
            out = layer((t["src"], t["dst"]), batch, shard_info, **kwargs)
            return out[1] if returns_pair else out

        return run

    def shapes(src_channels, src_points, dst_points):
        s = {"src": (batch * src_points, src_channels), "dst": (batch * dst_points, 6)}
        if conditional:
            s |= {"cond_src": (batch * src_points, COND_CHANNELS), "cond_dst": (batch * dst_points, COND_CHANNELS)}
        return s

    encoder_run = _Run(shapes(5, data.num_points, hidden.num_points), call(data.num_points, hidden.num_points, True))
    _same_results(make_encoder, encoder_run, 8, gradient_checkpointing, backend)

    decoder_run = _Run(
        shapes(NUM_CHANNELS, hidden.num_points, data.num_points), call(hidden.num_points, data.num_points, False)
    )
    _same_results(make_decoder, decoder_run, 5, gradient_checkpointing, backend)


PROCESSOR_CASES = {
    "octahedral": ("octahedral", ReducedGrid.octahedral(32), (7, 13)),
    "healpix": ("healpix", ReducedGrid.healpix(16), (5, 5)),
}


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("case", PROCESSOR_CASES)
@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_processor_in_bands_matches_the_whole_grid(
    case: str, gradient_checkpointing: bool, conditional: bool, backend: str
):
    family, grid, kernel_size = PROCESSOR_CASES[case]
    batch = 2

    def make(num_bands):
        return TransformerProcessor(
            num_layers=2,
            num_channels=NUM_CHANNELS,
            num_chunks=2,
            num_heads=2,
            mlp_hidden_ratio=2,
            qk_norm=True,
            attention_implementation="neighbourhood",
            neighbourhood=_neighbourhood_config(family, kernel_size, num_bands, backend),
            rotary_embeddings={"max_frequency": 20, "backend": SETTINGS[backend]["rotary"]},
            node_coords=grid_coords(grid),
            layer_kernels=_layer_kernels(conditional),
        )

    def run(layer, t):
        kwargs = {"cond": t["cond"]} if conditional else {}
        return layer(t["x"], batch, GraphShardInfo(nodes=[batch * grid.num_points]), **kwargs)

    shapes = {"x": (batch * grid.num_points, NUM_CHANNELS)}
    if conditional:
        shapes["cond"] = (batch * grid.num_points, COND_CHANNELS)
    _same_results(make, _Run(shapes, run), 8, gradient_checkpointing, backend)


def _splits(num_points: int) -> list[list[int]]:
    """Ways of splitting the points across GPUs: balanced over 2 to 7 GPUs, and some uneven ones."""
    splits = [get_balanced_partition_sizes(num_points, n) for n in range(2, 8)]
    splits.append([1, num_points - 2, 1])
    splits.append([num_points // 3, 5, num_points - num_points // 3 - 5])
    return splits


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("num_bands", [1, 3])
def test_every_gpu_sees_the_keys_of_its_own_queries(case: str, num_bands: int) -> None:
    family, query_grid, key_grid, kernel_size = CASES[case]
    nb = replace(neighbourhood(family, query_grid, key_grid, kernel_size), num_bands=num_bands)
    if nb.is_self_attention:
        full = dense(ReducedGridNeighbourhoodMask(nb.query_grid, nb.kernel_size))
    else:
        full = dense(ReducedGridCrossNeighbourhoodMask(nb.query_grid, nb.key_grid, nb.kernel_size))

    for sizes in _splits(nb.query_grid.num_points):
        plans = NeighbourhoodBands(nb).plans(sizes)
        assert [p.own_points.stop - p.own_points.start for p in plans] == sizes
        for plan in plans:
            own, rows, keys = plan.own_points, plan.query_points, plan.key_points
            assert rows.start <= own.start and own.stop <= rows.stop
            outside = full[own].clone()
            outside[:, keys] = False
            assert not outside.any(), "a query attends to a key the GPU does not fetch"
            covered = torch.zeros(nb.query_grid.num_points, dtype=torch.long)
            for band in plan.bands:
                covered[band.query_points] += 1
                assert keys.start <= band.key_points.start and band.key_points.stop <= keys.stop
                band_nb = band.attention.neighbourhood
                band_mask = dense(
                    ReducedGridCrossNeighbourhoodMask(band_nb.query_grid, band_nb.key_grid, band_nb.kernel_size)
                )
                assert torch.equal(band_mask, full[band.query_points, band.key_points])
            assert torch.equal(covered[rows], torch.ones(rows.stop - rows.start, dtype=torch.long))
            assert covered.sum() == rows.stop - rows.start


@pytest.mark.parametrize("num_points", [10, 97])
def test_point_range_exchange_gives_every_gpu_the_points_it_wants(num_points: int) -> None:
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(num_points, 3, generator=generator)
    for sizes in _splits(num_points):
        starts = [0, *itertools.accumulate(sizes)]
        shards = [values[a:b] for a, b in zip(starts[:-1], starts[1:])]
        wanted = []
        for a, b in zip(starts[:-1], starts[1:]):
            first = int(torch.randint(0, a + 1, (1,), generator=generator))
            stop = int(torch.randint(b, num_points + 1, (1,), generator=generator))
            wanted.append(slice(first, stop))
        exchanges = [build_point_range_exchange(sizes, wanted, r, torch.device("cpu")) for r in range(len(sizes))]
        for rank, exchange in enumerate(exchanges):
            # What each other GPU sends this one, in rank order, as the exchange delivers it.
            received = [shards[other][exchanges[other].send_indices[rank]] for other in range(len(sizes))]
            assert [len(r) for r in received] == list(exchange.recv_counts)
            run = assemble_point_run(shards[rank][exchange.own_part], torch.cat(received), exchange)
            assert run.own.data_ptr() == shards[rank][exchange.own_part].data_ptr(), "own points are copied"
            assert torch.equal(run.take(wanted[rank]), values[wanted[rank]][None])
            for _ in range(5):
                first = int(torch.randint(wanted[rank].start, wanted[rank].stop, (1,), generator=generator))
                stop = int(torch.randint(first + 1, wanted[rank].stop + 1, (1,), generator=generator))
                assert torch.equal(run.take(slice(first, stop)), values[first:stop][None])


def test_points_taken_from_a_run_always_depend_on_its_own_part() -> None:
    """Every band depends on the exchange that holds the own points, so its backward runs after all bands."""
    before = torch.randn(1, 4, 2, requires_grad=True)
    own = torch.randn(1, 6, 2, requires_grad=True)
    after = torch.randn(1, 3, 2, requires_grad=True)
    run = PointRun(start=10, before=before, own=own, after=after)
    for points in (slice(10, 13), slice(20, 23), slice(12, 16), slice(15, 18)):
        own.grad = None
        run.take(points).sum().backward()
        assert own.grad is not None, f"the points {points} do not depend on the own part"
    assert torch.equal(run.take(slice(11, 21)), torch.cat([before, own, after], dim=1)[:, 1:11])
