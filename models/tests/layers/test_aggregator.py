# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.aggregator import ConcatAggregator
from anemoi.models.layers.aggregator import GatedFusionAggregator
from anemoi.models.layers.aggregator import GatedLatentFusion
from anemoi.models.layers.aggregator import MeanAggregator
from anemoi.models.layers.aggregator import PointwiseCrossAttentionAggregator
from anemoi.models.layers.aggregator import SumAggregator


def test_sum_and_mean_aggregators_accept_active_source_subsets() -> None:
    hidden = torch.randn(6, 3)
    a = torch.randn(6, 4)
    b = torch.randn(6, 4)
    source_channels = {"a": 4, "b": 4}

    summed = SumAggregator(input_channels=3, source_channels=source_channels)(hidden, {"b": b, "a": a})
    mean = MeanAggregator(input_channels=3, source_channels=source_channels)(hidden, {"b": b})

    torch.testing.assert_close(summed, a + b)
    torch.testing.assert_close(mean, b)


def test_concat_aggregator_uses_configured_source_order_and_widths() -> None:
    aggregator = ConcatAggregator(input_channels=3, source_channels={"global": 2, "regional": 3})
    hidden = torch.randn(6, 3)
    global_latent = torch.randn(6, 2)
    regional_latent = torch.randn(6, 3)

    output = aggregator(hidden, {"regional": regional_latent, "global": global_latent})

    torch.testing.assert_close(output, torch.cat((global_latent, regional_latent), dim=-1))
    assert aggregator.hidden_dim == 5


def test_concat_aggregator_requires_every_configured_source() -> None:
    aggregator = ConcatAggregator(input_channels=3, source_channels={"global": 2, "regional": 3})

    with pytest.raises(ValueError, match="missing.*regional"):
        aggregator(torch.randn(6, 3), {"global": torch.randn(6, 2)})


@pytest.mark.parametrize(
    ("latents", "error"),
    [
        ({"unknown": torch.randn(6, 4)}, "Unknown latent sources"),
        ({"a": torch.randn(6, 5)}, "must have 4 channels"),
        ({"a": torch.randn(7, 4)}, "matching leading dimensions"),
    ],
)
def test_aggregator_validates_named_source_shapes(latents: dict[str, torch.Tensor], error: str) -> None:
    aggregator = SumAggregator(input_channels=3, source_channels={"a": 4})

    with pytest.raises(ValueError, match=error):
        aggregator(torch.randn(6, 3), latents)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_pointwise_cross_attention_aggregator_supports_different_source_widths_and_gradients(
    gradient_checkpointing: bool, qk_norm: bool
) -> None:
    aggregator = PointwiseCrossAttentionAggregator(
        input_channels=3,
        source_channels={"global": 4, "regional": 6},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
        attn_channels=6,
        qk_norm=qk_norm,
        gradient_checkpointing=gradient_checkpointing,
    )
    hidden = torch.randn(7, 3, requires_grad=True)
    latents = {
        "global": torch.randn(7, 4, requires_grad=True),
        "regional": torch.randn(7, 6, requires_grad=True),
    }

    output = aggregator(hidden, latents)
    output.square().mean().backward()

    assert output.shape == (7, 8)
    assert aggregator.hidden_dim == 8
    assert hidden.grad is not None
    assert all(latent.grad is not None for latent in latents.values())
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


def test_pointwise_cross_attention_aggregator_accepts_an_active_source_subset() -> None:
    aggregator = PointwiseCrossAttentionAggregator(
        input_channels=3,
        source_channels={"a": 4, "b": 6},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
    ).eval()

    output = aggregator(torch.randn(7, 3), {"b": torch.randn(7, 6)})

    assert output.shape == (7, 8)


def test_pointwise_cross_attention_aggregator_is_shard_local() -> None:
    aggregator = PointwiseCrossAttentionAggregator(
        input_channels=3,
        source_channels={"a": 4, "b": 4},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
    ).eval()
    hidden = torch.randn(9, 3)
    latents = {"a": torch.randn(9, 4), "b": torch.randn(9, 4)}

    full_output = aggregator(hidden, latents)
    sharded_output = torch.cat(
        [
            aggregator(hidden[:4], {name: latent[:4] for name, latent in latents.items()}),
            aggregator(hidden[4:], {name: latent[4:] for name, latent in latents.items()}),
        ],
    )

    torch.testing.assert_close(full_output, sharded_output)


def test_pointwise_cross_attention_aggregator_validates_attention_width() -> None:
    with pytest.raises(ValueError, match="must be divisible by number of heads"):
        PointwiseCrossAttentionAggregator(
            input_channels=3,
            source_channels={"a": 4},
            num_channels=8,
            num_heads=3,
            attn_channels=8,
            layer_kernels={},
        )


def _gated_aggregator(**kwargs) -> GatedFusionAggregator:
    torch.manual_seed(0)
    return GatedFusionAggregator(input_channels=3, source_channels={"data": 4, "waves": 4, "ocean": 4}, **kwargs)


def test_gated_latent_fusion_is_a_no_op_when_gate_is_closed() -> None:
    block = GatedLatentFusion(hidden_dim=4)
    latent, encoder_output = torch.randn(6, 4), torch.randn(6, 4)

    output = block(latent, encoder_output)
    assert output.shape == latent.shape
    assert not torch.allclose(output, latent)

    # A strongly negative gate bias closes the gate: the block becomes the identity.
    with torch.no_grad():
        block.to_gate[0].weight.zero_()
        block.to_gate[0].bias.fill_(-100.0)
    torch.testing.assert_close(block(latent, encoder_output), latent)


def test_gated_fusion_aggregator_builds_one_block_per_auxiliary_source() -> None:
    aggregator = _gated_aggregator(principal_source="data")

    assert aggregator.hidden_dim == 4
    assert set(aggregator.fusion) == {"waves", "ocean"}
    # Defaults to the first configured source.
    assert _gated_aggregator().principal_source == "data"


def test_gated_fusion_aggregator_matches_sequential_fusion_blocks_in_eval() -> None:
    aggregator = _gated_aggregator(principal_source="data").eval()
    hidden = torch.randn(6, 3)
    data, waves, ocean = torch.randn(6, 4), torch.randn(6, 4), torch.randn(6, 4)

    output = aggregator(hidden, {"ocean": ocean, "data": data, "waves": waves})

    # In eval mode sources are folded in configured order: waves, then ocean.
    expected = aggregator.fusion["ocean"](aggregator.fusion["waves"](data, waves), ocean)
    torch.testing.assert_close(output, expected)


def test_gated_fusion_aggregator_accepts_an_active_source_subset() -> None:
    aggregator = _gated_aggregator(principal_source="data").eval()
    hidden = torch.randn(6, 3)
    data, waves = torch.randn(6, 4), torch.randn(6, 4)

    torch.testing.assert_close(aggregator(hidden, {"data": data}), data)
    torch.testing.assert_close(
        aggregator(hidden, {"data": data, "waves": waves}),
        aggregator.fusion["waves"](data, waves),
    )


def test_gated_fusion_aggregator_propagates_gradients_to_all_active_sources() -> None:
    aggregator = _gated_aggregator(principal_source="data").train()
    latents = {name: torch.randn(6, 4, requires_grad=True) for name in ("data", "waves", "ocean")}

    aggregator(torch.randn(6, 3), latents).sum().backward()

    assert all(latent.grad is not None and torch.count_nonzero(latent.grad) > 0 for latent in latents.values())
    assert all(p.grad is not None for p in aggregator.parameters())


def test_gated_fusion_aggregator_supports_gradient_checkpointing() -> None:
    plain = _gated_aggregator(principal_source="data").eval()
    checkpointed = _gated_aggregator(principal_source="data", gradient_checkpointing=True).eval()
    checkpointed.load_state_dict(plain.state_dict())
    hidden = torch.randn(6, 3)
    latents = {"data": torch.randn(6, 4), "waves": torch.randn(6, 4), "ocean": torch.randn(6, 4)}

    torch.testing.assert_close(checkpointed(hidden, latents), plain(hidden, latents))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"source_channels": {"data": 4, "waves": 5}}, "same channel dimension"),
        ({"source_channels": {"data": 4}, "principal_source": "waves"}, "principal_source 'waves'"),
    ],
)
def test_gated_fusion_aggregator_validates_construction(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        GatedFusionAggregator(input_channels=3, **kwargs)


def test_gated_fusion_aggregator_requires_principal_source_at_call_time() -> None:
    aggregator = _gated_aggregator(principal_source="data")

    with pytest.raises(ValueError, match="requires the principal source 'data'"):
        aggregator(torch.randn(6, 3), {"waves": torch.randn(6, 4)})
