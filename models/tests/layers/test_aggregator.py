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
from anemoi.models.layers.aggregator import CrossAttentionAggregator
from anemoi.models.layers.aggregator import MeanAggregator
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


def test_cross_attention_aggregator_supports_different_source_widths_and_gradients() -> None:
    aggregator = CrossAttentionAggregator(
        input_channels=3,
        source_channels={"global": 4, "regional": 6},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
        gradient_checkpointing=False,
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


def test_cross_attention_aggregator_is_independent_of_mapping_order() -> None:
    aggregator = CrossAttentionAggregator(
        input_channels=3,
        source_channels={"a": 4, "b": 4},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
    ).eval()
    hidden = torch.randn(7, 3)
    a = torch.randn(7, 4)
    b = torch.randn(7, 4)

    output = aggregator(hidden, {"a": a, "b": b})
    reversed_output = aggregator(hidden, {"b": b, "a": a})

    torch.testing.assert_close(output, reversed_output)


def test_cross_attention_aggregator_accepts_an_active_source_subset() -> None:
    aggregator = CrossAttentionAggregator(
        input_channels=3,
        source_channels={"a": 4, "b": 6},
        num_channels=8,
        num_heads=2,
        layer_kernels={},
    ).eval()

    output = aggregator(torch.randn(7, 3), {"b": torch.randn(7, 6)})

    assert output.shape == (7, 8)


def test_cross_attention_aggregator_is_shard_local() -> None:
    aggregator = CrossAttentionAggregator(
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


def test_cross_attention_aggregator_validates_attention_width() -> None:
    with pytest.raises(ValueError, match="must be divisible by number of heads"):
        CrossAttentionAggregator(
            input_channels=3,
            source_channels={"a": 4},
            num_channels=8,
            num_heads=3,
            attn_channels=8,
            layer_kernels={},
        )
