# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention through the registry and hooks of a grid family, with the test family of
:mod:`tests.layers.window_neighbourhood`.
"""

import pickle

import pytest
import torch
from pydantic import ValidationError

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers import attention as attention_module
from anemoi.models.layers import neighbourhood_attention
from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.neighbourhood_attention import build_grid_neighbourhood
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.schemas.decoder import TransformerDecoderSchema
from anemoi.models.schemas.encoder import TransformerEncoderSchema
from anemoi.models.schemas.processor import TransformerProcessorSchema
from tests.layers import window_neighbourhood
from tests.layers.window_neighbourhood import WindowAttention
from tests.layers.window_neighbourhood import WindowNeighbourhood
from tests.layers.window_neighbourhood import register_window_family

NUM_CHANNELS = 16
COND_CHANNELS = 3


@pytest.fixture(autouse=True)
def window_family(monkeypatch):
    register_window_family(monkeypatch)


def line_coords(num_points: int) -> torch.Tensor:
    """Coordinates for the window family; only their number matters to it."""
    return torch.stack([torch.zeros(num_points), torch.linspace(0, 6, num_points)], dim=1)


def window_config(radius: int = 2, num_bands: int = 1, backend: str = "sdpa") -> dict:
    return {"grid": "window", "radius": radius, "num_bands": num_bands, "backend": backend}


def masked_attention(query, key, value, num_queries, num_keys, radius):
    """Attention to the keys within ``radius`` of each query's position, written out by hand."""
    centre = torch.arange(num_queries) * num_keys // num_queries
    mask = (centre[:, None] - torch.arange(num_keys)[None, :]).abs() <= radius
    scores = query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ value


# ------------------------------------------------------------------------------- registry and settings


def test_neighbourhood_is_only_built_for_neighbourhood_attention():
    coords = line_coords(20)
    assert build_grid_neighbourhood("scaled_dot_product_attention", None, coords) is None
    with pytest.raises(ValueError, match="only used with attention_implementation 'neighbourhood'"):
        build_grid_neighbourhood("triton_attention", window_config(), coords)
    with pytest.raises(ValueError, match="needs a 'neighbourhood' configuration"):
        build_grid_neighbourhood("neighbourhood", None, coords)
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(radius=3), coords)
    assert isinstance(neighbourhood, WindowNeighbourhood)
    assert (neighbourhood.radius, neighbourhood.is_self_attention) == (3, True)


@pytest.mark.parametrize(
    ("config", "query_coords", "message"),
    [
        ({"grid": "hexagons", "radius": 2}, None, "No neighbourhood attention kernels for grid 'hexagons'"),
        ({"grid": "window", "radius": 2, "kernel_size": [3, 3]}, None, "Unknown neighbourhood settings"),
        ({"grid": "window", "radius": 2, "backend": "natten"}, None, "backend must be one of"),
    ],
)
def test_invalid_settings_are_rejected(config, query_coords, message):
    with pytest.raises(ValueError, match=message):
        build_grid_neighbourhood("neighbourhood", config, line_coords(20), query_coords)


def test_neighbourhood_needs_node_coordinates():
    with pytest.raises(ValueError, match="coordinates of the graph nodes"):
        build_grid_neighbourhood("neighbourhood", window_config(), None)


def test_cross_attention_is_refused_for_a_family_with_self_attention_only(monkeypatch):
    family = neighbourhood_attention.GRID_FAMILIES["window"]
    monkeypatch.setitem(
        neighbourhood_attention.GRID_FAMILIES,
        "window",
        neighbourhood_attention.GridFamily(
            description=family.description, neighbourhood=family.neighbourhood, self_attention=family.self_attention
        ),
    )
    with pytest.raises(ValueError, match="only does self attention"):
        build_grid_neighbourhood("neighbourhood", window_config(), line_coords(20), line_coords(10))


# ------------------------------------------------------------------------------------ attention layers


def test_self_attention_layer_uses_the_family_module():
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(radius=2), line_coords(20))
    layer = MultiHeadSelfAttention(
        num_heads=2,
        embed_dim=NUM_CHANNELS,
        layer_kernels=load_layer_kernels(),
        attention_implementation="neighbourhood",
        neighbourhood=neighbourhood,
    )
    assert isinstance(layer.attention, WindowAttention)

    q, k, v = (torch.randn(2, 2, 20, 8, generator=torch.Generator().manual_seed(i)) for i in range(3))
    torch.testing.assert_close(layer.attention(q, k, v, 2), masked_attention(q, k, v, 20, 20, 2))


def test_cross_attention_module_attends_around_each_query():
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(radius=3), line_coords(30), line_coords(12))
    assert not neighbourhood.is_self_attention
    module = neighbourhood.attention_module()
    q = torch.randn(1, 2, 12, 8)
    k, v = torch.randn(1, 2, 30, 8), torch.randn(1, 2, 30, 8)
    torch.testing.assert_close(module(q, k, v, 1), masked_attention(q, k, v, 12, 30, 3))
    with pytest.raises(ValueError, match="set up for 12 queries and 30 keys"):
        module(q, k[:, :, :20], v[:, :, :20], 1)


def test_neighbourhood_implementation_needs_a_neighbourhood():
    with pytest.raises(ValueError, match="needs a neighbourhood"):
        MultiHeadSelfAttention(
            num_heads=2,
            embed_dim=NUM_CHANNELS,
            layer_kernels=load_layer_kernels(),
            attention_implementation="neighbourhood",
        )


@pytest.mark.parametrize("option", [{"window_size": 4}, {"dropout_p": 0.1}])
def test_options_a_backend_cannot_handle_are_rejected_when_built(option, monkeypatch):
    monkeypatch.setattr(neighbourhood_attention, "is_triton_available", lambda: True)
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(backend="triton"), line_coords(20))
    with pytest.raises(NotImplementedError, match="does not support"):
        MultiHeadSelfAttention(
            num_heads=2,
            embed_dim=NUM_CHANNELS,
            layer_kernels=load_layer_kernels(),
            attention_implementation="neighbourhood",
            neighbourhood=neighbourhood,
            **option,
        )


def test_triton_backend_runs_the_registered_kernel(monkeypatch):
    monkeypatch.setattr(neighbourhood_attention, "is_triton_available", lambda: True)
    monkeypatch.setattr(window_neighbourhood, "KERNEL_CALLS", [])
    module = build_grid_neighbourhood(
        "neighbourhood", window_config(backend="triton"), line_coords(30), line_coords(12)
    ).attention_module()
    q = torch.randn(1, 2, 12, 16)
    k, v = torch.randn(1, 2, 30, 16), torch.randn(1, 2, 30, 16)
    torch.testing.assert_close(module(q, k, v, 1), masked_attention(q, k, v, 12, 30, 2))
    assert window_neighbourhood.KERNEL_CALLS == [(12, 30)]


def test_triton_backend_needs_triton(monkeypatch):
    monkeypatch.setattr(neighbourhood_attention, "is_triton_available", lambda: False)
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(backend="triton"), line_coords(20))
    with pytest.raises(ImportError, match="use the 'sdpa' backend"):
        neighbourhood.attention_module()


@pytest.mark.parametrize("head_dim", [8, 24])
def test_triton_backend_rejects_head_sizes_that_are_not_powers_of_two_from_16(head_dim, monkeypatch):
    monkeypatch.setattr(neighbourhood_attention, "is_triton_available", lambda: True)
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(backend="triton"), line_coords(20))
    with pytest.raises(ValueError, match="power of two of at least 16"):
        MultiHeadSelfAttention(
            num_heads=2,
            embed_dim=2 * head_dim,
            layer_kernels=load_layer_kernels(),
            attention_implementation="neighbourhood",
            neighbourhood=neighbourhood,
        )


def test_backend_override_from_the_environment_leaves_neighbourhood_attention_alone(monkeypatch):
    monkeypatch.setattr(attention_module, "ATTENTION_BACKEND", "scaled_dot_product_attention")
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(), line_coords(20))
    layer = MultiHeadSelfAttention(
        num_heads=2,
        embed_dim=NUM_CHANNELS,
        layer_kernels=load_layer_kernels(),
        attention_implementation="neighbourhood",
        neighbourhood=neighbourhood,
    )
    layer(torch.randn(20, NUM_CHANNELS), GraphShardInfo(nodes=[20]), batch_size=1)
    assert layer.attention_implementation == "neighbourhood"
    assert isinstance(layer.attention, WindowAttention)


def test_masks_are_not_saved_with_the_model():
    neighbourhood = build_grid_neighbourhood("neighbourhood", window_config(), line_coords(20))
    layer = MultiHeadCrossAttention(
        num_heads=2,
        embed_dim=NUM_CHANNELS,
        layer_kernels=load_layer_kernels(),
        attention_implementation="neighbourhood",
        neighbourhood=neighbourhood,
    )
    x = torch.randn(20, NUM_CHANNELS)
    layer((x, x), BipartiteGraphShardInfo(src_nodes=[20], dst_nodes=[20]), batch_size=1)
    assert layer.attention._masks
    assert pickle.loads(pickle.dumps(layer.attention))._masks == {}


# ---------------------------------------------------------------------------- whole grid against bands


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


def _forward_backward(layer, inputs: dict, call):
    layer.zero_grad()
    inputs = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
    out = call(layer, inputs)
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(2), dtype=out.dtype)
    (out * weights).sum().backward()
    grads = {f"input {name}": t.grad for name, t in inputs.items()}
    grads |= {name: p.grad for name, p in layer.named_parameters() if p.grad is not None}
    return out.detach(), grads


def _same_results(make, shapes: dict, call, num_bands: int, gradient_checkpointing: bool) -> None:
    """Build the layer in one go and in bands, with the same weights; compare outputs and gradients."""
    torch.manual_seed(0)
    whole, banded = make(1).double(), make(num_bands).double()
    banded.load_state_dict(whole.state_dict())
    assert not whole.split.used(sharded=False) and banded.split.used(sharded=False)
    assert len(banded.split.work(1, torch.device("cpu")).bands) == num_bands
    for layer in (whole, banded):
        layer.gradient_checkpointing = gradient_checkpointing

    generator = torch.Generator().manual_seed(1)
    inputs = {name: torch.randn(shape, generator=generator, dtype=torch.float64) for name, shape in shapes.items()}
    out_whole, grads_whole = _forward_backward(whole, inputs, call)
    out_banded, grads_banded = _forward_backward(banded, inputs, call)

    torch.testing.assert_close(out_banded, out_whole, rtol=1e-10, atol=1e-10)
    assert grads_banded.keys() == grads_whole.keys()
    for name in grads_whole:
        torch.testing.assert_close(grads_banded[name], grads_whole[name], rtol=1e-10, atol=1e-10, msg=name)


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_processor_in_bands_matches_the_whole_grid(gradient_checkpointing, conditional):
    num_points, batch = 40, 2

    def make(num_bands):
        return TransformerProcessor(
            num_layers=2,
            num_channels=NUM_CHANNELS,
            num_chunks=2,
            num_heads=2,
            mlp_hidden_ratio=2,
            qk_norm=True,
            attention_implementation="neighbourhood",
            neighbourhood=window_config(radius=3, num_bands=num_bands),
            rotary_embeddings={"max_frequency": 20, "backend": "torch"},
            node_coords=line_coords(num_points),
            layer_kernels=_layer_kernels(conditional),
        )

    def call(layer, t):
        kwargs = {"cond": t["cond"]} if conditional else {}
        return layer(t["x"], batch, GraphShardInfo(nodes=[batch * num_points]), **kwargs)

    shapes = {"x": (batch * num_points, NUM_CHANNELS)}
    if conditional:
        shapes["cond"] = (batch * num_points, COND_CHANNELS)
    _same_results(make, shapes, call, 4, gradient_checkpointing)


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("conditional", [False, True])
def test_encoder_and_decoder_in_bands_match_the_whole_grid(gradient_checkpointing, conditional):
    data, hidden, batch = 48, 16, 2

    def mapper_kwargs(num_bands):
        return dict(
            in_channels_src=5,
            in_channels_dst=6,
            num_channels=NUM_CHANNELS,
            num_chunks=1,
            num_heads=2,
            mlp_hidden_ratio=2,
            qk_norm=True,
            attention_implementation="neighbourhood",
            neighbourhood=window_config(radius=4, num_bands=num_bands),
            rotary_embeddings={"max_frequency": 20, "backend": "torch"},
            layer_kernels=_layer_kernels(conditional),
        )

    def make_encoder(num_bands):
        return TransformerForwardMapper(
            **mapper_kwargs(num_bands), src_node_coords=line_coords(data), dst_node_coords=line_coords(hidden)
        )

    def make_decoder(num_bands):
        return TransformerBackwardMapper(
            **mapper_kwargs(num_bands),
            out_channels_dst=4,
            src_node_coords=line_coords(hidden),
            dst_node_coords=line_coords(data),
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

    _same_results(make_encoder, shapes(5, data, hidden), call(data, hidden, True), 3, gradient_checkpointing)
    _same_results(
        make_decoder, shapes(NUM_CHANNELS, hidden, data), call(hidden, data, False), 5, gradient_checkpointing
    )


def test_cpu_offload_runs_all_points_at_once():
    processor = TransformerProcessor(
        num_layers=1,
        num_channels=NUM_CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="neighbourhood",
        neighbourhood=window_config(num_bands=3),
        node_coords=line_coords(20),
        cpu_offload=True,
        layer_kernels=load_layer_kernels(instance=False),
    )
    assert processor.split is None


# ------------------------------------------------------------------------------------------- schemas

COMMON = dict(
    cpu_offload=False,
    num_chunks=1,
    mlp_hidden_ratio=4,
    num_heads=16,
    num_channels=512,
    window_size=None,
    dropout_p=0.0,
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
def test_schemas_check_neighbourhood_settings_against_the_implementation(schema, target, extra):
    settings = {**COMMON, **extra, "_target_": target}
    component = schema(**settings, attention_implementation="neighbourhood", neighbourhood=window_config())
    assert component.neighbourhood == window_config()
    with pytest.raises(ValidationError, match="needs a 'neighbourhood' section"):
        schema(**settings, attention_implementation="neighbourhood")
    with pytest.raises(ValidationError, match="only used with attention_implementation 'neighbourhood'"):
        schema(**settings, attention_implementation="scaled_dot_product_attention", neighbourhood=window_config())
    with pytest.raises(ValidationError, match="window_size must be null"):
        schema(
            **{**settings, "window_size": 4}, attention_implementation="neighbourhood", neighbourhood=window_config()
        )
