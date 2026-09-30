# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch
from pydantic import ValidationError

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.spherical_rotary import SphericalRotaryEmbedding
from anemoi.models.layers.spherical_rotary import apply_rotary
from anemoi.models.layers.spherical_rotary import build_spherical_rotary
from anemoi.models.layers.spherical_rotary import rotary_angles
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.schemas.decoder import TransformerDecoderSchema
from anemoi.models.schemas.encoder import TransformerEncoderSchema
from anemoi.models.schemas.processor import TransformerProcessorSchema

ROTARY = {"max_frequency": 30.0, "backend": "torch"}


def coords_of(grid: ReducedGrid) -> torch.Tensor:
    return grid.coords.float()


def unit_vectors(coords: torch.Tensor) -> torch.Tensor:
    lat, lon = coords[:, 0].double(), coords[:, 1].double()
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=1)


def shuffled(coords: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(coords.shape[0], generator=torch.Generator().manual_seed(seed))
    return coords[perm], perm


def turn(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    return apply_rotary(x, torch.cos(angles), torch.sin(angles))


# --------------------------------------------------------------------------------- angles and the turn


def test_angles_turn_by_every_frequency_along_each_axis():
    coords = coords_of(ReducedGrid.octahedral(4))
    angles = rotary_angles(coords, head_dim=16, max_frequency=30.0)
    xyz = unit_vectors(coords)
    frequencies = torch.tensor([1.0, 30.0], dtype=torch.float64)
    assert angles.shape == (coords.shape[0], 6) and angles.dtype == torch.float64
    for axis in range(3):
        torch.testing.assert_close(angles[:, 2 * axis : 2 * axis + 2], xyz[:, axis : axis + 1] * frequencies)


def test_frequencies_are_spread_evenly_on_a_log_scale():
    coords = torch.tensor([[0.0, 0.0]])  # x = 1: the x angles are the frequencies themselves
    angles = rotary_angles(coords, head_dim=24, max_frequency=1000.0)
    torch.testing.assert_close(angles[0, :4], torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=torch.float64))


@pytest.mark.parametrize(("head_dim", "max_frequency", "message"), [(4, 10.0, "at least 6"), (16, 0.5, "at least 1")])
def test_angles_need_six_channels_and_a_frequency_of_at_least_one(head_dim, max_frequency, message):
    with pytest.raises(ValueError, match=message):
        rotary_angles(coords_of(ReducedGrid.octahedral(4)), head_dim=head_dim, max_frequency=max_frequency)


def test_rotated_scores_depend_on_the_straight_line_from_query_to_key():
    """Turning the query by a and the key by b scores like turning the query alone by a - b."""
    query_coords, key_coords = coords_of(ReducedGrid.octahedral(4)), coords_of(ReducedGrid.octahedral(8))
    q_angles = rotary_angles(query_coords, 16, 30.0)[:1]
    k_angles = rotary_angles(key_coords, 16, 30.0)[:5]
    # The difference of the angles is the frequency times the straight line between the points.
    offset = unit_vectors(query_coords)[:1] - unit_vectors(key_coords)[:5]
    frequencies = torch.tensor([1.0, 30.0], dtype=torch.float64)
    torch.testing.assert_close(q_angles - k_angles, (offset[:, :, None] * frequencies).flatten(1))

    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, 16, generator=generator, dtype=torch.float64)
    k = torch.randn(5, 16, generator=generator, dtype=torch.float64)
    by_offset = (turn(q.expand(5, -1), q_angles - k_angles) * k).sum(-1)
    torch.testing.assert_close((turn(q, q_angles) * turn(k, k_angles)).sum(-1), by_offset)


@pytest.mark.parametrize("head_dim", [16, 17])
def test_turn_leaves_the_other_channels_alone_and_keeps_lengths(head_dim):
    x = torch.randn(3, head_dim, dtype=torch.float64)
    angles = torch.rand(3, 6, dtype=torch.float64) * 10
    turned = turn(x, angles)
    half = head_dim // 2
    assert torch.equal(turned[:, 6:half], x[:, 6:half]) and torch.equal(turned[:, half + 6 :], x[:, half + 6 :])
    torch.testing.assert_close(turned.norm(dim=-1), x.norm(dim=-1))


def test_turn_works_in_float32_for_low_precision_and_keeps_float64():
    angles = torch.rand(4, 3, dtype=torch.float64) * 100
    assert turn(torch.randn(4, 12, dtype=torch.bfloat16), angles).dtype == torch.float32
    x = torch.randn(4, 12, dtype=torch.float64)
    exact = x.clone()
    for i in range(3):
        c, s = torch.cos(angles[:, i]), torch.sin(angles[:, i])
        exact[:, i], exact[:, 6 + i] = x[:, i] * c - x[:, 6 + i] * s, x[:, i] * s + x[:, 6 + i] * c
    torch.testing.assert_close(turn(x, angles), exact, rtol=0, atol=1e-14)


def test_turn_in_bfloat16_is_as_close_as_rounding_the_input():
    """Angles of up to 100 radians become cosines and sines before anything is rounded to bfloat16."""
    grid = ReducedGrid.octahedral(16)
    module = SphericalRotaryEmbedding(coords_of(grid), None, 32, 100.0, backend="torch")
    x = torch.randn(2, 4, grid.num_points, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    low, _ = module(x.bfloat16(), x.bfloat16())
    exact = turn(x, rotary_angles(coords_of(grid), 32, 100.0))
    rounding = (x.bfloat16().double() - x).abs().max()
    assert low.dtype == torch.bfloat16
    assert (low.double() - exact).abs().max() <= 3 * rounding


# --------------------------------------------------------------------------------------------- module


def test_self_attention_shares_one_table_and_saves_none():
    module = SphericalRotaryEmbedding(coords_of(ReducedGrid.octahedral(4)), None, 16, 30.0, backend="torch")
    assert module.shared and not hasattr(module, "key_cos")
    assert module.query_cos.dtype == torch.float32 and module.query_cos.is_contiguous()
    assert module.state_dict() == {}


def test_cross_attention_turns_queries_and_keys_by_their_own_points():
    query_grid, key_grid = ReducedGrid.octahedral(4), ReducedGrid.octahedral(8)
    module = SphericalRotaryEmbedding(coords_of(query_grid), coords_of(key_grid), 16, 30.0, backend="torch")
    q = torch.randn(2, 3, query_grid.num_points, 16)
    k = torch.randn(2, 3, key_grid.num_points, 16)
    turned_q, turned_k = module(q, k)
    torch.testing.assert_close(turned_q, turn(q, rotary_angles(coords_of(query_grid), 16, 30.0).float()))
    torch.testing.assert_close(turned_k, turn(k, rotary_angles(coords_of(key_grid), 16, 30.0).float()))


def test_module_follows_the_node_order():
    grid = ReducedGrid.octahedral(8)
    coords, perm = shuffled(coords_of(grid))
    in_order = SphericalRotaryEmbedding(coords_of(grid), None, 16, 30.0, backend="torch")
    out_of_order = SphericalRotaryEmbedding(coords, None, 16, 30.0, backend="torch")
    x = torch.randn(2, 3, grid.num_points, 16)
    expected, _ = in_order(x, x)
    got, _ = out_of_order(x[:, :, perm], x[:, :, perm])
    torch.testing.assert_close(got, expected[:, :, perm])


def test_module_keeps_the_dtype_and_checks_shapes():
    module = SphericalRotaryEmbedding(coords_of(ReducedGrid.octahedral(4)), None, 16, 30.0, backend="torch")
    n = ReducedGrid.octahedral(4).num_points
    q, k = module(torch.randn(1, 2, n, 16, dtype=torch.bfloat16), torch.randn(1, 2, n, 16, dtype=torch.float16))
    assert (q.dtype, k.dtype) == (torch.bfloat16, torch.float16)
    with pytest.raises(ValueError, match="query points"):
        module(torch.randn(1, 2, n + 1, 16), torch.randn(1, 2, n, 16))
    with pytest.raises(ValueError, match="16 channels"):
        module(torch.randn(1, 2, n, 32), torch.randn(1, 2, n, 32))


def test_triton_backend_needs_a_gpu_tensor():
    pytest.importorskip("triton")
    module = SphericalRotaryEmbedding(coords_of(ReducedGrid.octahedral(4)), None, 16, 30.0, backend="triton")
    x = torch.randn(1, 2, ReducedGrid.octahedral(4).num_points, 16)
    with pytest.raises(ValueError, match="GPU"):
        module(x, x)


def test_module_can_be_listed_for_compilation():
    from anemoi.models.utils.compile import _get_compile_entry

    module = SphericalRotaryEmbedding(coords_of(ReducedGrid.octahedral(4)), None, 16, 30.0, backend="torch")
    entry = {"module": "anemoi.models.layers.spherical_rotary.SphericalRotaryEmbedding"}
    assert _get_compile_entry(module, [entry]) is entry


@pytest.mark.parametrize(
    ("config", "coords", "message"),
    [
        ({"max_frequency": 30.0, "radius": 2}, True, "Unknown rotary_embeddings settings"),
        ({"backend": "torch"}, True, "needs a max_frequency"),
        ({"max_frequency": 30.0}, False, "coordinates of the graph nodes"),
        ({"max_frequency": 30.0, "backend": "natten"}, True, "backend must be one of"),
        ({"max_frequency": 0.5}, True, "at least 1"),
    ],
)
def test_invalid_settings_are_rejected(config, coords, message):
    with pytest.raises(ValueError, match=message):
        build_spherical_rotary(config, 16, coords_of(ReducedGrid.octahedral(4)) if coords else None)


def test_settings_build_the_module():
    coords = coords_of(ReducedGrid.octahedral(4))
    assert build_spherical_rotary(None, 16, coords) is None
    module = build_spherical_rotary({"max_frequency": 50}, 16, coords)
    assert (module.backend, module.max_frequency, module.head_dim, module.shared) == ("triton", 50.0, 16, True)


# ------------------------------------------------------------------------------------ attention layers


def _capture_attention_inputs(layer):
    """Keep the queries and keys the layer's attention function receives."""
    seen = {}

    def keep(module, args):
        seen["query"], seen["key"] = args[0].detach(), args[1].detach()

    layer.attention.register_forward_pre_hook(keep)
    return seen


@pytest.mark.parametrize("attention_implementation", ["scaled_dot_product_attention", "neighbourhood"])
def test_attention_layer_turns_queries_and_keys_after_their_norms(attention_implementation):
    grid = ReducedGrid.octahedral(4)
    coords, _ = shuffled(coords_of(grid))
    settings = dict(num_heads=2, embed_dim=16, layer_kernels=load_layer_kernels(), qk_norm=True)
    settings["attention_implementation"] = attention_implementation
    if attention_implementation == "neighbourhood":
        settings["neighbourhood"] = GridNeighbourhood.from_config(
            {"grid": "octahedral", "kernel_size": [3, 5], "backend": "sdpa"}, coords
        )
    torch.manual_seed(0)
    plain = MultiHeadSelfAttention(**settings)
    rotary = MultiHeadSelfAttention(**settings, rotary=build_spherical_rotary(ROTARY, 8, coords))
    rotary.load_state_dict(plain.state_dict())
    plain_seen, rotary_seen = _capture_attention_inputs(plain), _capture_attention_inputs(rotary)

    x = torch.randn(2 * grid.num_points, 16)
    shard = GraphShardInfo(nodes=[2 * grid.num_points])
    out_plain, out_rotary = plain(x, shard, 2), rotary(x, shard, 2)
    angles = rotary_angles(coords, 8, 30.0).float()
    torch.testing.assert_close(rotary_seen["query"], turn(plain_seen["query"], angles))
    torch.testing.assert_close(rotary_seen["key"], turn(plain_seen["key"], angles))
    assert not torch.allclose(out_plain, out_rotary)


def test_cross_attention_layer_turns_queries_and_keys_by_their_own_nodes():
    query_grid, key_grid = ReducedGrid.octahedral(4), ReducedGrid.octahedral(8)
    settings = dict(
        num_heads=2,
        embed_dim=16,
        layer_kernels=load_layer_kernels(),
        attention_implementation="scaled_dot_product_attention",
    )
    torch.manual_seed(0)
    plain = MultiHeadCrossAttention(**settings)
    rotary = MultiHeadCrossAttention(
        **settings, rotary=build_spherical_rotary(ROTARY, 8, coords_of(query_grid), coords_of(key_grid))
    )
    rotary.load_state_dict(plain.state_dict())
    plain_seen, rotary_seen = _capture_attention_inputs(plain), _capture_attention_inputs(rotary)

    x = (torch.randn(key_grid.num_points, 16), torch.randn(query_grid.num_points, 16))
    shard = BipartiteGraphShardInfo(src_nodes=[key_grid.num_points], dst_nodes=[query_grid.num_points])
    plain(x, shard, 1)
    rotary(x, shard, 1)
    q_angles = rotary_angles(coords_of(query_grid), 8, 30.0).float()
    k_angles = rotary_angles(coords_of(key_grid), 8, 30.0).float()
    torch.testing.assert_close(rotary_seen["query"], turn(plain_seen["query"], q_angles))
    torch.testing.assert_close(rotary_seen["key"], turn(plain_seen["key"], k_angles))


# -------------------------------------------------------------------------------- processor and mapper


def test_processor_layers_share_one_module():
    grid = ReducedGrid.octahedral(4)
    processor = TransformerProcessor(
        num_layers=2,
        num_channels=16,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="scaled_dot_product_attention",
        rotary_embeddings=ROTARY,
        node_coords=coords_of(grid),
        layer_kernels=load_layer_kernels(instance=False),
    )
    modules = [layer.attention.rotary for layer in processor.proc]
    assert isinstance(modules[0], SphericalRotaryEmbedding) and modules[0] is modules[1]
    assert modules[0].head_dim == 8 and modules[0].shared
    out = processor(torch.randn(grid.num_points, 16), 1, GraphShardInfo(nodes=[grid.num_points]))
    assert out.shape == (grid.num_points, 16)


def test_mapper_turns_destination_queries_and_source_keys():
    data, hidden = ReducedGrid.octahedral(8), ReducedGrid.octahedral(4)
    encoder = TransformerForwardMapper(
        in_channels_src=5,
        in_channels_dst=6,
        num_channels=16,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="scaled_dot_product_attention",
        rotary_embeddings=ROTARY,
        src_node_coords=coords_of(data),
        dst_node_coords=coords_of(hidden),
        layer_kernels=load_layer_kernels(instance=False),
    )
    module = encoder.proc.attention.rotary
    assert module.query_cos.shape[0] == hidden.num_points and module.key_cos.shape[0] == data.num_points
    shard = BipartiteGraphShardInfo(src_nodes=[data.num_points], dst_nodes=[hidden.num_points])
    _, latent = encoder((torch.randn(data.num_points, 5), torch.randn(hidden.num_points, 6)), 1, shard)
    assert latent.shape == (hidden.num_points, 16)


def test_mapper_without_coordinates_is_rejected():
    with pytest.raises(ValueError, match="coordinates of the graph nodes"):
        TransformerForwardMapper(
            in_channels_src=5,
            in_channels_dst=6,
            num_channels=16,
            num_chunks=1,
            num_heads=2,
            mlp_hidden_ratio=2,
            attention_implementation="scaled_dot_product_attention",
            rotary_embeddings=ROTARY,
            layer_kernels=load_layer_kernels(instance=False),
        )


# -------------------------------------------------------------------------------------------- schemas


COMMON = dict(
    cpu_offload=False,
    num_chunks=1,
    mlp_hidden_ratio=4,
    num_heads=16,
    num_channels=512,
    window_size=None,
    dropout_p=0.0,
    attention_implementation="scaled_dot_product_attention",
    softcap=0.0,
)


@pytest.mark.parametrize(
    ("schema", "target", "extra"),
    [
        (
            TransformerProcessorSchema,
            "anemoi.models.layers.processor.TransformerProcessor",
            {"num_layers": 4, "qk_norm": False},
        ),
        (TransformerEncoderSchema, "anemoi.models.layers.mapper.TransformerForwardMapper", {}),
        (TransformerDecoderSchema, "anemoi.models.layers.mapper.TransformerBackwardMapper", {}),
    ],
    ids=["processor", "encoder", "decoder"],
)
def test_schemas_accept_rotary_settings(schema, target, extra):
    settings = {**COMMON, **extra, "_target_": target}
    assert schema(**settings).rotary_embeddings is None
    component = schema(**settings, rotary_embeddings={"max_frequency": 100})
    assert (component.rotary_embeddings.max_frequency, component.rotary_embeddings.backend) == (100.0, "triton")
    with pytest.raises(ValidationError):
        schema(**settings, rotary_embeddings={"max_frequency": 0.5})
    with pytest.raises(ValidationError):
        schema(**settings, rotary_embeddings={"max_frequency": 100, "backend": "natten"})


def test_forty_thousand_km_over_the_frequency_is_one_turn():
    # The config documentation's rule of thumb: max_frequency 100 repeats every ~400 km.
    earth_radius_km = 6371.0
    assert math.isclose(2 * math.pi * earth_radius_km / 100, 400, rel_tol=0.01)
