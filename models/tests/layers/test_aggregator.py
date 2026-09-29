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


def test_sum_mean_concat_aggregators_mask_dropped_sources() -> None:
    hidden = torch.randn(6, 3)
    a = torch.randn(6, 4)
    b = torch.randn(6, 4)
    source_channels = {"a": 4, "b": 4}
    latents = {"a": a, "b": b}

    summed = SumAggregator(input_channels=3, source_channels=source_channels)(hidden, latents, dropped_sources={"b"})
    mean = MeanAggregator(input_channels=3, source_channels=source_channels)(hidden, latents, dropped_sources={"b"})
    concat = ConcatAggregator(input_channels=3, source_channels=source_channels)(hidden, latents, dropped_sources={"b"})

    torch.testing.assert_close(summed, a)
    torch.testing.assert_close(mean, a)
    torch.testing.assert_close(concat, torch.cat((a, torch.zeros_like(b)), dim=-1))


def test_mean_aggregator_rejects_dropping_every_source() -> None:
    aggregator = MeanAggregator(input_channels=3, source_channels={"a": 4})

    with pytest.raises(ValueError, match="every latent source was dropped"):
        aggregator(torch.randn(6, 3), {"a": torch.randn(6, 4)}, dropped_sources={"a"})


def _gated_aggregator(**kwargs) -> GatedFusionAggregator:
    torch.manual_seed(0)
    return GatedFusionAggregator(input_channels=3, source_channels={"data": 4, "waves": 4, "ocean": 4}, **kwargs)


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


def test_gated_fusion_aggregator_dropped_source_is_a_no_op_but_keeps_gradients() -> None:
    aggregator = _gated_aggregator(principal_source="data").eval()
    hidden = torch.randn(6, 3)
    data, waves, ocean = torch.randn(6, 4), torch.randn(6, 4), torch.randn(6, 4)

    with_ocean_dropped = aggregator(hidden, {"data": data, "waves": waves, "ocean": ocean}, dropped_sources={"ocean"})
    without_ocean = aggregator(hidden, {"data": data, "waves": waves})
    torch.testing.assert_close(with_ocean_dropped, without_ocean)

    # The dropped block still participates in the graph (gate * 0), so DDP sees
    # every parameter: its grads exist and are exactly zero.
    with_ocean_dropped.sum().backward()
    ocean_grads = [p.grad for p in aggregator.fusion["ocean"].parameters()]
    assert all(g is not None for g in ocean_grads)
    assert all(torch.count_nonzero(g) == 0 for g in ocean_grads)
    assert any(torch.count_nonzero(p.grad) > 0 for p in aggregator.fusion["waves"].parameters())


def test_gated_fusion_aggregator_supports_gradient_checkpointing() -> None:
    plain = _gated_aggregator(principal_source="data").eval()
    checkpointed = _gated_aggregator(principal_source="data", gradient_checkpointing=True).eval()
    checkpointed.load_state_dict(plain.state_dict())
    hidden = torch.randn(6, 3)
    latents = {"data": torch.randn(6, 4), "waves": torch.randn(6, 4), "ocean": torch.randn(6, 4)}

    torch.testing.assert_close(
        checkpointed(hidden, latents, dropped_sources={"waves"}),
        plain(hidden, latents, dropped_sources={"waves"}),
    )


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
    with pytest.raises(ValueError, match="cannot be dropped"):
        aggregator(torch.randn(6, 3), {"data": torch.randn(6, 4)}, dropped_sources={"data"})
