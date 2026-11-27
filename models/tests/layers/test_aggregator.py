# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import random

import pytest
import torch

from anemoi.models.layers.aggregator import ConcatAggregator
from anemoi.models.layers.aggregator import GatedFusionAggregator
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
    return GatedFusionAggregator(input_channels=3, source_channels={"principal": 4, "aux_a": 4, "aux_b": 4}, **kwargs)


def test_gated_fusion_aggregator_folds_active_sources_in_configured_order() -> None:
    aggregator = _gated_aggregator().eval()
    hidden = torch.randn(6, 3)
    principal, aux_a, aux_b = torch.randn(6, 4), torch.randn(6, 4), torch.randn(6, 4)

    # principal_source defaults to the first configured source; one block per auxiliary source.
    assert aggregator.principal_source == "principal"
    assert set(aggregator.fusion) == {"aux_a", "aux_b"}
    assert aggregator.hidden_dim == 4

    # In eval mode sources are folded in configured order (aux_a, then aux_b), whatever the mapping order.
    expected = aggregator.fusion["aux_b"](aggregator.fusion["aux_a"](principal, aux_a), aux_b)
    torch.testing.assert_close(aggregator(hidden, {"aux_b": aux_b, "principal": principal, "aux_a": aux_a}), expected)

    # Absent auxiliary sources are skipped.
    torch.testing.assert_close(
        aggregator(hidden, {"principal": principal, "aux_a": aux_a}), aggregator.fusion["aux_a"](principal, aux_a)
    )
    torch.testing.assert_close(aggregator(hidden, {"principal": principal}), principal)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_gated_fusion_aggregator_supports_different_source_widths_and_gradients(gradient_checkpointing: bool) -> None:
    source_channels = {"principal": 4, "aux_a": 6, "aux_b": 2}
    aggregator = GatedFusionAggregator(
        input_channels=3,
        source_channels=source_channels,
        principal_source="aux_a",
        gradient_checkpointing=gradient_checkpointing,
    ).train()
    latents = {name: torch.randn(6, width, requires_grad=True) for name, width in source_channels.items()}

    output = aggregator(torch.randn(6, 3), latents)
    output.sum().backward()

    # The output width is the principal source's width.
    assert aggregator.hidden_dim == 6
    assert output.shape == (6, 6)
    assert all(latent.grad is not None and torch.count_nonzero(latent.grad) > 0 for latent in latents.values())
    assert all(p.grad is not None for p in aggregator.parameters())


@pytest.mark.parametrize("seed", range(5))
def test_gated_fusion_aggregator_checkpointing_keeps_shuffled_order_in_backward(seed: int) -> None:
    # The recompute in backward must use the same (shuffled) merge order as the forward pass.
    source_channels = {"principal": 4, "aux_a": 4, "aux_b": 4, "aux_c": 4}
    plain = GatedFusionAggregator(input_channels=3, source_channels=source_channels).train()
    checkpointed = GatedFusionAggregator(
        input_channels=3,
        source_channels=source_channels,
        gradient_checkpointing=True,
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    hidden = torch.randn(6, 3)
    inputs = {name: torch.randn(6, 4) for name in source_channels}

    grads = []
    for aggregator in (plain, checkpointed):
        random.seed(seed)
        torch.manual_seed(seed)
        latents = {name: tensor.clone().requires_grad_() for name, tensor in inputs.items()}
        aggregator(hidden, latents).square().sum().backward()
        grads.append([latents[name].grad for name in source_channels] + [p.grad for p in aggregator.parameters()])

    for plain_grad, checkpointed_grad in zip(*grads, strict=True):
        torch.testing.assert_close(checkpointed_grad, plain_grad)


def test_gated_fusion_aggregator_zero_value_init_scale_is_an_identity_that_still_learns() -> None:
    aggregator = _gated_aggregator(value_init_scale=0.0)
    principal = torch.randn(6, 4)
    latents = {"principal": principal, "aux_a": torch.randn(6, 4), "aux_b": torch.randn(6, 4)}

    output = aggregator(torch.randn(6, 3), latents)
    torch.testing.assert_close(output, principal)

    # The gates stay open, so each zeroed value projection receives gradient and can move away from zero.
    output.sum().backward()
    for block in aggregator.fusion.values():
        assert torch.count_nonzero(block.to_value[-1].weight.grad) > 0
        assert torch.count_nonzero(block.to_value[-1].bias.grad) > 0


def test_gated_fusion_aggregator_value_init_scale_gives_encoder_gradient_from_the_start() -> None:
    torch.manual_seed(0)
    reference = GatedFusionAggregator(input_channels=3, source_channels={"principal": 4, "aux_a": 4})
    torch.manual_seed(0)
    scaled = GatedFusionAggregator(
        input_channels=3, source_channels={"principal": 4, "aux_a": 4}, value_init_scale=1e-3
    )
    torch.testing.assert_close(
        scaled.fusion["aux_a"].to_value[-1].weight, 1e-3 * reference.fusion["aux_a"].to_value[-1].weight
    )

    zero = _gated_aggregator(value_init_scale=0.0)
    principal = torch.randn(6, 4)
    encoder_grads = {}
    for name, aggregator in (("scaled", scaled), ("zero", zero)):
        aux_a = torch.randn(6, 4, requires_grad=True)
        output = aggregator(torch.randn(6, 3), {"principal": principal, "aux_a": aux_a})
        # Near-identity: the update is small compared with the latent.
        assert (output - principal).abs().max() < 1e-2 * principal.abs().max()
        output.sum().backward()
        encoder_grads[name] = aux_a.grad

    # Unlike zero init, the encoder feeding the block gets gradient from the first step.
    assert torch.count_nonzero(encoder_grads["zero"]) == 0
    assert torch.count_nonzero(encoder_grads["scaled"]) > 0


def test_gated_fusion_aggregator_validates_sources() -> None:
    with pytest.raises(ValueError, match="principal_source 'aux_a'"):
        GatedFusionAggregator(input_channels=3, source_channels={"principal": 4}, principal_source="aux_a")
    with pytest.raises(ValueError, match="requires the principal source 'principal'"):
        _gated_aggregator()(torch.randn(6, 3), {"aux_a": torch.randn(6, 4)})
    with pytest.raises(ValueError, match="must be non-negative"):
        _gated_aggregator(value_init_scale=-1.0)
