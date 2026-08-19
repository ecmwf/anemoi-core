# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import patch

import pytest
import torch

from anemoi.models.layers.fusion import CrossAttentionLatentFusion
from anemoi.models.layers.fusion import SumLatentFusion
from anemoi.models.layers.normalization import AutocastLayerNorm

FUSION_CASES = [
    (SumLatentFusion, {"input_channels": 3, "layer_kernels": {}}, False),
    (CrossAttentionLatentFusion, {"input_channels": 3, "num_heads": 2, "layer_kernels": {}}, True),
]


class _CaptureAttention(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        self.query = query
        self.key = key
        self.value = value
        return torch.zeros_like(query)


def test_sum_latent_fusion_matches_elementwise_sum() -> None:
    fusion = SumLatentFusion(input_channels=3, num_channels=4, dataset_names=["a", "b"], layer_kernels={})
    hidden = torch.randn(6, 3)
    a = torch.randn(6, 4)
    b = torch.randn(6, 4)

    output = fusion(hidden, {"b": b, "a": a})

    torch.testing.assert_close(output, a + b)
    assert fusion.gradient_checkpointing is False


@pytest.mark.parametrize(("fusion_class", "kwargs", "uses_hidden"), FUSION_CASES)
def test_latent_fusion_preserves_shape_and_gradients(fusion_class, kwargs, uses_hidden) -> None:
    fusion = fusion_class(num_channels=8, dataset_names=["a", "b"], **kwargs)
    latents = {
        "a": torch.randn(7, 8, requires_grad=True),
        "b": torch.randn(7, 8, requires_grad=True),
    }
    hidden = torch.randn(7, 3, requires_grad=True)

    output = fusion(hidden, latents)
    output.square().mean().backward()

    assert output.shape == (7, 8)
    assert all(latent.grad is not None for latent in latents.values())
    assert all(torch.isfinite(latent.grad).all() for latent in latents.values())
    if uses_hidden:
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()


def test_cross_attention_latent_fusion_is_independent_of_mapping_order() -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
    ).eval()
    hidden = torch.randn(7, 3)
    a = torch.randn(7, 8)
    b = torch.randn(7, 8)

    output = fusion(hidden, {"a": a, "b": b})
    reversed_output = fusion(hidden, {"b": b, "a": a})

    torch.testing.assert_close(output, reversed_output)


def test_cross_attention_latent_fusion_is_shard_local() -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
    ).eval()
    hidden = torch.randn(9, 3)
    latents = {"a": torch.randn(9, 8), "b": torch.randn(9, 8)}

    full_output = fusion(hidden, latents)
    sharded_output = torch.cat(
        [
            fusion(hidden[:4], {name: latent[:4] for name, latent in latents.items()}),
            fusion(hidden[4:], {name: latent[4:] for name, latent in latents.items()}),
        ]
    )

    torch.testing.assert_close(full_output, sharded_output)


def test_cross_attention_latent_fusion_accepts_an_active_dataset_subset() -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
    ).eval()

    output = fusion(torch.randn(7, 3), {"b": torch.randn(7, 8)})

    assert output.shape == (7, 8)


def test_cross_attention_latent_fusion_selects_attention_implementation() -> None:
    with patch(
        "anemoi.models.layers.attention.load_attention_implementation",
        return_value=torch.nn.Identity(),
    ) as load_attention:
        fusion = CrossAttentionLatentFusion(
            input_channels=3,
            num_channels=4,
            dataset_names=["a", "b"],
            layer_kernels={},
            num_heads=2,
            attention_implementation="flash_attention",
        )

    load_attention.assert_called_once_with("flash_attention", head_dim=2)
    assert fusion.attention.attention_implementation == "flash_attention"


def test_cross_attention_latent_fusion_uses_configured_layer_kernels() -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=4,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={
            "Linear": {"_target_": "torch.nn.modules.linear.NonDynamicallyQuantizableLinear"},
            "LayerNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm"},
        },
    )

    projections = (
        fusion.hidden_projection,
        fusion.attention.lin_q,
        fusion.attention.lin_k,
        fusion.attention.lin_v,
        fusion.attention.projection,
    )
    assert all(
        isinstance(projection, torch.nn.modules.linear.NonDynamicallyQuantizableLinear) for projection in projections
    )
    assert isinstance(fusion.hidden_norm, AutocastLayerNorm)
    assert isinstance(fusion.source_norm, AutocastLayerNorm)


def test_cross_attention_latent_fusion_normalizes_each_source_token_and_query() -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=4,
        num_channels=4,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
        gradient_checkpointing=False,
    )
    fusion.hidden_projection = torch.nn.Identity()
    capture_attention = _CaptureAttention()
    fusion.attention = capture_attention
    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.0, 4.0, 8.0],
            [3.0, -1.0, 5.0, -3.0],
        ]
    )
    base_latent = torch.tensor(
        [
            [-4.0, -1.0, 2.0, 7.0],
            [1.0, 3.0, 6.0, 10.0],
            [-5.0, -2.0, 4.0, 8.0],
        ]
    )
    latents = {
        "a": base_latent * 2 + 5,
        "b": base_latent * 20 - 30,
    }

    output = fusion(hidden, latents)

    torch.testing.assert_close(output, hidden)
    for normalized in (capture_attention.query, capture_attention.value):
        torch.testing.assert_close(
            normalized.mean(dim=-1),
            torch.zeros_like(normalized[..., 0]),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            normalized.var(dim=-1, unbiased=False),
            torch.ones_like(normalized[..., 0]),
            atol=1e-4,
            rtol=0,
        )
    torch.testing.assert_close(capture_attention.value[:, 0], capture_attention.value[:, 1])
    dataset_embeddings = torch.stack([fusion.dataset_embeddings[name] for name in latents])
    torch.testing.assert_close(
        capture_attention.key - capture_attention.value,
        dataset_embeddings.unsqueeze(0).expand_as(capture_attention.key),
    )


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
def test_cross_attention_latent_fusion_checkpointing_is_configurable(gradient_checkpointing: bool) -> None:
    fusion = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=4,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
        gradient_checkpointing=gradient_checkpointing,
    )

    with patch("anemoi.models.layers.fusion.maybe_checkpoint") as maybe_checkpoint:
        maybe_checkpoint.side_effect = lambda function, enabled, *args: function(*args)
        fusion(torch.randn(5, 3), {"a": torch.randn(5, 4), "b": torch.randn(5, 4)})

    assert maybe_checkpoint.call_args.args[1] is gradient_checkpointing


def test_dataset_embeddings_are_keyed_by_name_across_constructor_order() -> None:
    source = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
    )
    target = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["b", "a"],
        num_heads=2,
        layer_kernels={},
    )
    with torch.no_grad():
        source.dataset_embeddings["a"].fill_(1.0)
        source.dataset_embeddings["b"].fill_(2.0)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.dataset_embeddings["a"], source.dataset_embeddings["a"])
    torch.testing.assert_close(target.dataset_embeddings["b"], source.dataset_embeddings["b"])
    assert "dataset_embeddings.a" in source.state_dict()
    assert "dataset_embeddings.b" in source.state_dict()


def test_dataset_embeddings_support_adding_and_removing_datasets() -> None:
    source = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["a", "b"],
        num_heads=2,
        layer_kernels={},
    )
    target = CrossAttentionLatentFusion(
        input_channels=3,
        num_channels=8,
        dataset_names=["b", "c"],
        num_heads=2,
        layer_kernels={},
    )
    with torch.no_grad():
        source.dataset_embeddings["b"].fill_(2.0)
    new_dataset_initialization = target.dataset_embeddings["c"].detach().clone()

    incompatible_keys = target.load_state_dict(source.state_dict(), strict=False)

    torch.testing.assert_close(target.dataset_embeddings["b"], source.dataset_embeddings["b"])
    torch.testing.assert_close(target.dataset_embeddings["c"], new_dataset_initialization)
    assert "dataset_embeddings.c" in incompatible_keys.missing_keys
    assert "dataset_embeddings.a" in incompatible_keys.unexpected_keys
