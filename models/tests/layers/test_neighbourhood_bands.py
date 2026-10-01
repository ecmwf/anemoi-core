# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention worked through bands of query rows gives the same results as on the whole grid."""

import math

import pytest
import torch

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
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


def _neighbourhood_config(family: str, kernel_size, num_bands: int) -> dict:
    return {"grid": family, "kernel_size": list(kernel_size), "backend": "sdpa", "num_bands": num_bands}


def _same_results(make, run, num_bands: int, gradient_checkpointing: bool) -> None:
    """Build the layer without and with bands, with the same weights, and compare outputs and all gradients."""
    torch.manual_seed(0)
    whole = make(1).double()
    banded = make(num_bands).double()
    banded.load_state_dict(whole.state_dict())
    assert banded.bands is not None and len(banded.bands) > 1
    for layer in (whole, banded):
        layer.gradient_checkpointing = gradient_checkpointing

    inputs = run.inputs()
    out_whole, grads_whole = run.forward_backward(whole, inputs)
    out_banded, grads_banded = run.forward_backward(banded, inputs)

    torch.testing.assert_close(out_banded, out_whole, rtol=1e-10, atol=1e-10)
    assert grads_banded.keys() == grads_whole.keys()
    for name in grads_whole:
        torch.testing.assert_close(grads_banded[name], grads_whole[name], rtol=1e-9, atol=1e-10, msg=name)


class _Run:
    """Random inputs for a layer, and a forward and backward pass that collects every gradient."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], call):
        self.shapes = shapes
        self.call = call

    def inputs(self) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(1)
        return {
            name: torch.randn(shape, generator=generator, dtype=torch.float64) for name, shape in self.shapes.items()
        }

    def forward_backward(self, layer, inputs):
        layer.zero_grad()
        inputs = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
        out = self.call(layer, inputs)
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(2), dtype=out.dtype)
        (out * weights).sum().backward()
        grads = {f"input {name}": t.grad for name, t in inputs.items()}
        grads |= {name: p.grad for name, p in layer.named_parameters() if p.grad is not None}
        return out.detach(), grads


MAPPER_CASES = {
    "octahedral": ("octahedral", ReducedGrid.octahedral(16), ReducedGrid.octahedral(8), (5, 7), (3, 5)),
    "healpix": ("healpix", ReducedGrid.healpix(8), ReducedGrid.healpix(4), (5, 5), (3, 5)),
}


@pytest.mark.parametrize("case", MAPPER_CASES)
@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_encoder_and_decoder_in_bands_match_the_whole_grid(case: str, gradient_checkpointing: bool, conditional: bool):
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
            neighbourhood=_neighbourhood_config(family, kernel_size, num_bands),
            rotary_embeddings={"max_frequency": 20, "backend": "torch"},
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
    _same_results(make_encoder, encoder_run, 3, gradient_checkpointing)

    decoder_run = _Run(
        shapes(NUM_CHANNELS, hidden.num_points, data.num_points), call(hidden.num_points, data.num_points, False)
    )
    _same_results(make_decoder, decoder_run, 5, gradient_checkpointing)


PROCESSOR_CASES = {
    "octahedral": ("octahedral", ReducedGrid.octahedral(12), (5, 7)),
    "healpix": ("healpix", ReducedGrid.healpix(8), (3, 5)),
}


@pytest.mark.parametrize("case", PROCESSOR_CASES)
@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_processor_in_bands_matches_the_whole_grid(case: str, gradient_checkpointing: bool, conditional: bool):
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
            neighbourhood=_neighbourhood_config(family, kernel_size, num_bands),
            rotary_embeddings={"max_frequency": 20, "backend": "torch"},
            node_coords=grid_coords(grid),
            layer_kernels=_layer_kernels(conditional),
        )

    def run(layer, t):
        kwargs = {"cond": t["cond"]} if conditional else {}
        return layer(t["x"], batch, GraphShardInfo(nodes=[batch * grid.num_points]), **kwargs)

    shapes = {"x": (batch * grid.num_points, NUM_CHANNELS)}
    if conditional:
        shapes["cond"] = (batch * grid.num_points, COND_CHANNELS)
    _same_results(make, _Run(shapes, run), 4, gradient_checkpointing)
