# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import re

import pytest
import torch
from pydantic import ValidationError

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers import attention as attention_module
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import GRID_KERNELS
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttentionWrapper
from anemoi.models.layers.neighbourhood_attention import build_grid_neighbourhood
from anemoi.models.layers.neighbourhood_attention import check_every_key_attended
from anemoi.models.layers.neighbourhood_attention import grid_from_coords
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.schemas.decoder import TransformerDecoderSchema
from anemoi.models.schemas.processor import TransformerProcessorSchema

NEIGHBOURHOOD = {"grid": "octahedral", "kernel_size": [3, 5], "backend": "sdpa"}


def grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def shuffled(coords: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The coordinates in a random node order, and that order."""
    perm = torch.randperm(coords.shape[0], generator=torch.Generator().manual_seed(seed))
    return coords[perm], perm


@pytest.mark.parametrize(
    ("family", "grid"),
    [
        ("octahedral", ReducedGrid.octahedral(8)),
        ("octahedral", ReducedGrid.octahedral(24)),
        ("healpix", ReducedGrid.healpix(4)),
    ],
)
def test_registered_grids_are_recognised_from_coords(family, grid):
    recognised, order = grid_from_coords(family, grid_coords(grid))
    assert recognised == grid
    assert order is None


def test_every_registered_family_names_importable_kernels():
    pytest.importorskip("triton")
    for kernels in GRID_KERNELS.values():
        assert callable(kernels.load_self_attention())
        assert callable(kernels.load_cross_attention())


def test_grid_of_another_family_is_rejected():
    with pytest.raises(ValueError, match="do not form one of the HEALPix"):
        grid_from_coords("healpix", grid_coords(ReducedGrid.octahedral(8)))
    with pytest.raises(ValueError, match="do not form one of the octahedral"):
        grid_from_coords("octahedral", grid_coords(ReducedGrid.healpix(4)))
    # Rows of a classic reduced Gaussian grid are equally spaced but not octahedral.
    with pytest.raises(ValueError, match="do not form one of the octahedral"):
        grid_from_coords("octahedral", grid_coords(ReducedGrid((18, 25, 25, 18), (60.0, 20.0, -20.0, -60.0))))


def test_unknown_grid_family_is_rejected():
    with pytest.raises(ValueError, match="No neighbourhood attention kernels for grid 'lambert'"):
        grid_from_coords("lambert", grid_coords(ReducedGrid.octahedral(8)))


def test_nodes_in_any_order_are_sorted_into_the_grid():
    grid = ReducedGrid.octahedral(8)
    coords, perm = shuffled(grid_coords(grid))
    recognised, order = grid_from_coords("octahedral", coords)
    assert recognised == grid
    torch.testing.assert_close(coords[order], grid_coords(grid))
    assert torch.equal(perm[order], torch.arange(grid.num_points))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (None, "needs a 'neighbourhood' configuration"),
        ({"grid": "octahedral", "kernel_size": [4, 5]}, "two positive odd numbers"),
        ({"grid": "octahedral", "kernel_size": [3, 5, 7]}, "two positive odd numbers"),
        ({"grid": "octahedral", "kernel_size": [3, 5], "backend": "natten"}, "backend must be one of"),
        ({"grid": "octahedral", "kernel_size": [3, 5], "radius": 2}, "Unknown neighbourhood settings"),
    ],
)
def test_invalid_settings_are_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        GridNeighbourhood.from_config(config, grid_coords(ReducedGrid.octahedral(8)))


def test_cross_attention_that_leaves_keys_unread_is_rejected():
    query_coords, key_coords = grid_coords(ReducedGrid.octahedral(4)), grid_coords(ReducedGrid.octahedral(24))
    config = {"grid": "octahedral", "kernel_size": [1, 3]}
    with pytest.raises(ValueError, match=r"are attended to by no query") as error:
        GridNeighbourhood.from_config(config, key_coords, query_coords=query_coords)
    # The kernel named in the message reads every key point.
    suggested = [int(k) for k in re.search(r"kernel_size \[(\d+), (\d+)\] reads every key", str(error.value)).groups()]
    neighbourhood = GridNeighbourhood.from_config({**config, "kernel_size": suggested}, key_coords, query_coords)
    assert neighbourhood.kernel_size == tuple(suggested)
    assert suggested[0] >= 1 and suggested[1] >= 3


def test_cross_attention_logs_how_queries_and_keys_are_connected(caplog):
    query_grid, key_grid = ReducedGrid.octahedral(4), ReducedGrid.octahedral(8)
    with caplog.at_level(logging.INFO, logger="anemoi.models.layers.neighbourhood_attention"):
        check_every_key_attended(query_grid, key_grid, (5, 5))
    assert f"from {key_grid.num_points} key points ({key_grid.num_rows} rows)" in caplog.text
    assert f"to {query_grid.num_points} queries ({query_grid.num_rows} rows)" in caplog.text
    assert "each query reads 25 to 25 keys, 25.0 on average" in caplog.text
    assert "each key point is read by" in caplog.text


def test_neighbourhood_is_only_built_for_neighbourhood_attention():
    coords = grid_coords(ReducedGrid.octahedral(8))
    assert build_grid_neighbourhood("scaled_dot_product_attention", None, coords) is None
    with pytest.raises(ValueError, match="only used with attention_implementation 'neighbourhood'"):
        build_grid_neighbourhood("flash_attention", NEIGHBOURHOOD, coords)
    neighbourhood = build_grid_neighbourhood("neighbourhood", NEIGHBOURHOOD, coords)
    assert neighbourhood.is_self_attention
    assert neighbourhood.kernel_size == (3, 5)


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_self_attention_follows_the_node_order(backend):
    """Nodes stored in another order give the same result for each node."""
    grid = ReducedGrid.octahedral(8)
    config = {**NEIGHBOURHOOD, "backend": backend}
    in_order = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(config, grid_coords(grid)))
    coords, perm = shuffled(grid_coords(grid))
    out_of_order = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(config, coords))

    q, k, v = (torch.randn(2, 3, grid.num_points, 8, generator=torch.Generator().manual_seed(i)) for i in range(3))
    expected = in_order(q, k, v, 2)[:, :, perm]
    got = out_of_order(q[:, :, perm], k[:, :, perm], v[:, :, perm], 2)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_cross_attention_follows_the_node_order():
    key_grid, query_grid = ReducedGrid.octahedral(8), ReducedGrid.octahedral(4)
    in_order = NeighbourhoodAttentionWrapper(
        GridNeighbourhood.from_config(NEIGHBOURHOOD, grid_coords(key_grid), grid_coords(query_grid))
    )
    key_coords, key_perm = shuffled(grid_coords(key_grid), seed=1)
    query_coords, query_perm = shuffled(grid_coords(query_grid), seed=2)
    out_of_order = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(NEIGHBOURHOOD, key_coords, query_coords))
    assert not out_of_order.neighbourhood.is_self_attention

    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, query_grid.num_points, 8, generator=generator)
    k, v = (torch.randn(1, 2, key_grid.num_points, 8, generator=generator) for _ in range(2))
    expected = in_order(q, k, v, 1)[:, :, query_perm]
    got = out_of_order(q[:, :, query_perm], k[:, :, key_perm], v[:, :, key_perm], 1)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_node_orders_are_not_saved_with_the_weights():
    coords, _ = shuffled(grid_coords(ReducedGrid.octahedral(8)))
    wrapper = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(NEIGHBOURHOOD, coords))
    assert wrapper.query_order is not None
    assert wrapper.state_dict() == {}


def test_wrapper_rejects_other_masks_and_sizes():
    grid = ReducedGrid.octahedral(8)
    wrapper = NeighbourhoodAttentionWrapper(GridNeighbourhood.from_config(NEIGHBOURHOOD, grid_coords(grid)))
    q = torch.randn(1, 2, grid.num_points, 8)
    with pytest.raises(ValueError, match="causal and window_size"):
        wrapper(q, q, q, 1, window_size=16)
    with pytest.raises(NotImplementedError, match="Softcap"):
        wrapper(q, q, q, 1, softcap=1.0)
    with pytest.raises(ValueError, match="set up for 544 queries"):
        wrapper(q[:, :, :-1], q, q, 1)


def _self_attention_layer(neighbourhood):
    return MultiHeadSelfAttention(
        num_heads=2,
        embed_dim=16,
        layer_kernels=load_layer_kernels(),
        attention_implementation="neighbourhood",
        neighbourhood=neighbourhood,
    )


def test_self_attention_layer_ignores_the_backend_environment_variable(monkeypatch):
    grid = ReducedGrid.octahedral(8)
    monkeypatch.setattr(attention_module, "ATTENTION_BACKEND", "scaled_dot_product_attention")
    layer = _self_attention_layer(GridNeighbourhood.from_config(NEIGHBOURHOOD, grid_coords(grid)))
    x = torch.randn(grid.num_points, 16)
    out = layer(x, GraphShardInfo(nodes=[grid.num_points]), batch_size=1)
    assert out.shape == x.shape
    assert isinstance(layer.attention, NeighbourhoodAttentionWrapper)


def test_self_attention_layer_needs_a_neighbourhood():
    with pytest.raises(ValueError, match="needs a GridNeighbourhood"):
        _self_attention_layer(None)


def test_processor_only_mixes_neighbouring_nodes():
    """With one layer, each output depends only on the inputs inside its neighbourhood."""
    grid = ReducedGrid.octahedral(8)
    coords, _ = shuffled(grid_coords(grid))
    processor = TransformerProcessor(
        num_layers=1,
        num_channels=16,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="neighbourhood",
        neighbourhood=NEIGHBOURHOOD,
        node_coords=coords,
        layer_kernels=load_layer_kernels(instance=False),
    )
    wrapper = processor.proc[0].attention.attention
    assert isinstance(wrapper, NeighbourhoodAttentionWrapper)

    x = torch.randn(2 * grid.num_points, 16, requires_grad=True)
    out = processor(x, 2, GraphShardInfo(nodes=[2 * grid.num_points]))
    assert out.shape == x.shape

    # Node 7 of the second sample, and the nodes it may attend to, in node order.
    node = 7
    order = wrapper.query_order
    grid_index = int(torch.nonzero(order == node))
    allowed_in_grid = wrapper._mask_on(torch.device("cpu"))[grid_index]
    allowed = torch.zeros(grid.num_points, dtype=torch.bool)
    allowed[order[allowed_in_grid]] = True

    out[grid.num_points + node].sum().backward()
    touched = x.grad.abs().sum(dim=1) > 0
    assert not touched[: grid.num_points].any()
    assert torch.equal(touched[grid.num_points :], allowed)


def _mapper_kwargs(**kwargs):
    return dict(
        in_channels_src=5,
        in_channels_dst=6,
        num_channels=16,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="neighbourhood",
        neighbourhood=NEIGHBOURHOOD,
        layer_kernels=load_layer_kernels(instance=False),
        **kwargs,
    )


def test_encoder_and_decoder_attend_between_grids():
    data, hidden = ReducedGrid.octahedral(8), ReducedGrid.octahedral(4)
    data_coords, hidden_coords = grid_coords(data), grid_coords(hidden)

    encoder = TransformerForwardMapper(**_mapper_kwargs(src_node_coords=data_coords, dst_node_coords=hidden_coords))
    neighbourhood = encoder.proc.attention.attention.neighbourhood
    assert (neighbourhood.query_grid, neighbourhood.key_grid) == (hidden, data)
    x_src, x_dst = torch.randn(data.num_points, 5), torch.randn(hidden.num_points, 6)
    shard_info = BipartiteGraphShardInfo(src_nodes=[data.num_points], dst_nodes=[hidden.num_points])
    _, latent = encoder((x_src, x_dst), 1, shard_info)
    assert latent.shape == (hidden.num_points, 16)

    decoder = TransformerBackwardMapper(
        **_mapper_kwargs(out_channels_dst=4, src_node_coords=hidden_coords, dst_node_coords=data_coords)
    )
    shard_info = BipartiteGraphShardInfo(src_nodes=[hidden.num_points], dst_nodes=[data.num_points])
    out = decoder((latent, torch.randn(data.num_points, 6)), 1, shard_info)
    assert out.shape == (data.num_points, 4)


def _processor_schema(**kwargs):
    return dict(
        _target_="anemoi.models.layers.processor.TransformerProcessor",
        cpu_offload=False,
        num_chunks=1,
        mlp_hidden_ratio=4,
        num_heads=16,
        num_channels=512,
        num_layers=16,
        window_size=None,
        dropout_p=0.0,
        attention_implementation="neighbourhood",
        qk_norm=False,
        softcap=0.0,
        **kwargs,
    )


def test_schema_accepts_neighbourhood_settings():
    schema = TransformerProcessorSchema(**_processor_schema(neighbourhood={"grid": "healpix", "kernel_size": [7, 13]}))
    assert schema.neighbourhood.backend == "triton"
    assert schema.model_dump(by_alias=True)["neighbourhood"] == {
        "grid": "healpix",
        "kernel_size": (7, 13),
        "backend": "triton",
        "num_bands": 1,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "needs a 'neighbourhood' section"),
        ({"neighbourhood": {"grid": "lambert", "kernel_size": [7, 13]}}, "No neighbourhood attention kernels"),
        ({"neighbourhood": {"grid": "octahedral", "kernel_size": [6, 13]}}, "must be odd"),
        ({"neighbourhood": NEIGHBOURHOOD, "window_size": 512}, "window_size must be null"),
        (
            {"neighbourhood": NEIGHBOURHOOD, "attention_implementation": "flash_attention"},
            "only used with attention_implementation 'neighbourhood'",
        ),
    ],
)
def test_schema_rejects_inconsistent_settings(kwargs, message):
    settings = _processor_schema()
    settings.update(kwargs)
    with pytest.raises(ValidationError, match=message):
        TransformerProcessorSchema(**settings)


def test_decoder_schema_accepts_neighbourhood_settings():
    schema = TransformerDecoderSchema(
        _target_="anemoi.models.layers.mapper.TransformerBackwardMapper",
        cpu_offload=False,
        num_chunks=1,
        mlp_hidden_ratio=4,
        num_heads=16,
        num_channels=512,
        window_size=None,
        dropout_p=0.0,
        attention_implementation="neighbourhood",
        softcap=0.0,
        neighbourhood={"grid": "octahedral", "kernel_size": [3, 5]},
    )
    assert schema.neighbourhood.kernel_size == (3, 5)
