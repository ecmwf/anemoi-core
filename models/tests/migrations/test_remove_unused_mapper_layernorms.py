# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib

import pytest
import torch

from anemoi.models.layers.block import TransformerMapperBlock
from anemoi.models.layers.utils import load_layer_kernels

migrate = importlib.import_module("anemoi.models.migrations.scripts.1791277156_remove_unused_mapper_layernorms").migrate

MAPPER_PREFIX = "model.model.encoder.data.proc."
PROCESSOR_NORM_KEY = "model.model.processor.proc.0.blocks.0.layer_norm_mlp.weight"


def _make_mapper_block(layer_kernels_config: dict | None) -> TransformerMapperBlock:
    return TransformerMapperBlock(
        num_channels=8,
        hidden_dim=16,
        num_heads=2,
        window_size=None,
        layer_kernels=load_layer_kernels(layer_kernels_config),
        attention_implementation="scaled_dot_product_attention",
    )


def _old_layout_state_dict(block: TransformerMapperBlock) -> dict[str, torch.Tensor]:
    """State dict of ``block`` as an older anemoi-models would have saved it inside a full model."""
    state_dict = {}
    for key, value in block.state_dict().items():
        state_dict[MAPPER_PREFIX + key.replace("layer_norm_mlp.", "layer_norm_mpl.")] = value.clone()
        if key.startswith("layer_norm_mlp."):
            # The two unused norms had the same layout as the MLP norm.
            state_dict[MAPPER_PREFIX + key] = torch.full_like(value, 99.0)
            state_dict[MAPPER_PREFIX + key.replace("layer_norm_mlp.", "layer_norm_attention.")] = torch.full_like(
                value, 99.0
            )
    state_dict[PROCESSOR_NORM_KEY] = torch.ones(8)
    return state_dict


@pytest.mark.parametrize(
    "layer_kernels_config",
    [
        None,
        {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "condition_shape": 4,
            }
        },
    ],
    ids=["layer_norm", "conditional_layer_norm"],
)
def test_migrated_mapper_weights_load_into_current_block(layer_kernels_config: dict | None) -> None:
    source = _make_mapper_block(layer_kernels_config)
    checkpoint = {"state_dict": _old_layout_state_dict(source)}

    migrate(checkpoint)

    state_dict = checkpoint["state_dict"]
    assert state_dict.pop(PROCESSOR_NORM_KEY).eq(1.0).all()

    mapper_state = {k.removeprefix(MAPPER_PREFIX): v for k, v in state_dict.items()}
    target = _make_mapper_block(layer_kernels_config)
    target.load_state_dict(mapper_state, strict=True)

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key


def test_migration_leaves_checkpoint_without_mapper_norms_unchanged() -> None:
    state_dict = {
        PROCESSOR_NORM_KEY: torch.ones(8),
        "model.model.processor.proc.0.blocks.0.layer_norm_attention.weight": torch.ones(8),
    }
    checkpoint = {"state_dict": dict(state_dict)}

    migrate(checkpoint)

    assert checkpoint["state_dict"].keys() == state_dict.keys()
