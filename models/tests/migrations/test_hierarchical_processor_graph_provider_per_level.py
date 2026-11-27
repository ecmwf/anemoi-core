# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib

import torch
from omegaconf import OmegaConf

migrate = importlib.import_module(
    "anemoi.models.migrations.scripts.1795779750_hierarchical_processor_graph_provider_per_level"
).migrate

PROVIDER = "model.model.processor_graph_provider"
LEVEL_PROVIDERS = (
    "model.model.down_level_processor_graph_providers.hidden_1",
    "model.model.up_level_processor_graph_providers.hidden_1",
)
OTHER_KEY = "model.model.processor.proc.0.weight"


def _checkpoint(hidden_nodes_name: object, provider_prefix: str = PROVIDER) -> dict:
    state_dict = {
        f"{provider_prefix}.trainable.trainable": torch.ones(3),
        f"{provider_prefix}.trainable_layout_version": torch.tensor(1),
        OTHER_KEY: torch.ones(2),
    }
    for level_provider in LEVEL_PROVIDERS:
        state_dict[f"{level_provider}.trainable.trainable"] = torch.ones(4)
        state_dict[f"{level_provider}.trainable_layout_version"] = torch.tensor(1)
    return {
        "hyper_parameters": {
            "config": OmegaConf.create({"model": {"model": {"hidden_nodes_name": hidden_nodes_name}}}),
        },
        "state_dict": state_dict,
    }


def test_hierarchical_keys_are_nested_under_the_last_level() -> None:
    checkpoint = _checkpoint(["hidden_1", "hidden_2", "hidden_3"])
    trainable = checkpoint["state_dict"][f"{PROVIDER}.trainable.trainable"]

    migrate(checkpoint)

    level_keys = {
        f"{level_provider}.{name}"
        for level_provider in LEVEL_PROVIDERS
        for name in ("trainable.trainable", "trainable_layout_version")
    }
    assert set(checkpoint["state_dict"]) == {
        f"{PROVIDER}.hidden_3.trainable.trainable",
        f"{PROVIDER}.hidden_3.trainable_layout_version",
        OTHER_KEY,
        *level_keys,
    }
    assert checkpoint["state_dict"][f"{PROVIDER}.hidden_3.trainable.trainable"] is trainable


def test_single_hidden_mesh_checkpoint_is_unchanged() -> None:
    checkpoint = _checkpoint("hidden", provider_prefix=f"{PROVIDER}.hidden")
    keys = set(checkpoint["state_dict"])

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == keys


def test_dataset_mapping_checkpoint_is_unchanged() -> None:
    checkpoint = _checkpoint({"data": "hidden_a", "other": "hidden_b"}, provider_prefix=f"{PROVIDER}.hidden_a")
    keys = set(checkpoint["state_dict"])

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == keys


def test_keys_already_under_a_level_name_are_unchanged() -> None:
    """A model built with a list of one level already keeps its provider under that level's name."""
    checkpoint = _checkpoint(["hidden_1"], provider_prefix=f"{PROVIDER}.hidden_1")
    keys = set(checkpoint["state_dict"])

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == keys


def test_only_the_processor_graph_provider_is_renamed() -> None:
    checkpoint = _checkpoint(["hidden_1", "hidden_2"])
    unrelated = "model.model.encoder.processor_graph_provider.weight"
    checkpoint["state_dict"][unrelated] = torch.ones(1)

    migrate(checkpoint)

    assert unrelated in checkpoint["state_dict"]
