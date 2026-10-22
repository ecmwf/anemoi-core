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
    "anemoi.models.migrations.scripts.1792683306_processor_graph_provider_per_hidden_mesh"
).migrate

PROVIDER = "model.model.processor_graph_provider"
OTHER_KEY = "model.model.processor.proc.0.weight"


def _checkpoint(hidden_nodes_name: object, provider_prefix: str = PROVIDER) -> dict:
    return {
        "hyper_parameters": {"config": OmegaConf.create({"model": {"hidden_nodes_name": hidden_nodes_name}})},
        "state_dict": {
            f"{provider_prefix}.trainable.trainable": torch.ones(3),
            f"{provider_prefix}.trainable_layout_version": torch.tensor(1),
            OTHER_KEY: torch.ones(2),
        },
    }


def test_single_hidden_mesh_keys_are_nested_under_hidden_name() -> None:
    checkpoint = _checkpoint("hidden")

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == {
        f"{PROVIDER}.hidden.trainable.trainable",
        f"{PROVIDER}.hidden.trainable_layout_version",
        OTHER_KEY,
    }


def test_dataset_mapping_checkpoint_is_unchanged() -> None:
    checkpoint = _checkpoint({"data": "hidden_a", "other": "hidden_b"}, provider_prefix=f"{PROVIDER}.hidden_a")
    keys = set(checkpoint["state_dict"])

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == keys


def test_hierarchical_checkpoint_is_unchanged() -> None:
    checkpoint = _checkpoint(["hidden_1", "hidden_2"])
    keys = set(checkpoint["state_dict"])

    migrate(checkpoint)

    assert set(checkpoint["state_dict"]) == keys
