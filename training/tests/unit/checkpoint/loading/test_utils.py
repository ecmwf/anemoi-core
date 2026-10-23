# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for state dict utilities."""

import torch

from anemoi.training.checkpoint.loading.utils import filter_state_dict


def test_filter_state_dict_removes_shape_mismatches() -> None:
    """Non-mutating filter: returns new dict, does not modify input."""
    source = {
        "layer1.weight": torch.randn(10, 5),
        "layer2.weight": torch.randn(20, 10),
        "layer3.bias": torch.randn(5),
    }
    target = {
        "layer1.weight": torch.randn(10, 5),  # Same shape
        "layer2.weight": torch.randn(30, 10),  # Different shape
        "layer3.bias": torch.randn(5),  # Same shape
    }

    filtered, skipped = filter_state_dict(source, target)

    assert "layer1.weight" in filtered
    assert "layer2.weight" not in filtered  # Shape mismatch
    assert "layer3.bias" in filtered
    assert "layer2.weight" in skipped
    # Original dict not mutated
    assert "layer2.weight" in source


def test_filter_state_dict_does_not_mutate_input() -> None:
    """``filter_state_dict`` returns new dicts and leaves its input unchanged."""
    source = {"a": torch.randn(3), "b": torch.randn(5)}
    target = {"a": torch.randn(3), "b": torch.randn(10)}
    original_keys = set(source.keys())

    filter_state_dict(source, target)

    assert set(source.keys()) == original_keys


def test_filter_state_dict_maps_each_skipped_key_to_its_reason() -> None:
    source = {
        "shared.weight": torch.full((5, 10), 1.0),  # matches
        "missing_key": torch.full((3,), 2.0),  # absent from target
        "mismatch_key": torch.full((7, 10), 3.0),  # present but wrong shape
    }
    target = {
        "shared.weight": torch.full((5, 10), 0.0),
        "mismatch_key": torch.full((5, 10), 0.0),
    }

    filtered, skipped = filter_state_dict(source, target)

    assert "shared.weight" in filtered
    assert skipped["missing_key"] == "Key not in target"
    assert skipped["mismatch_key"].startswith("Shape mismatch:")


def test_filter_state_dict_keeps_an_adapted_layers_keys_together() -> None:
    # The layer got narrower: base_layer and lora_B change shape, lora_A does not.
    source = {
        "head.base_layer.weight": torch.zeros(4, 8),
        "head.base_layer.bias": torch.zeros(4),
        "head.lora_A.default.weight": torch.zeros(2, 8),
        "head.lora_B.default.weight": torch.zeros(4, 2),
        "body.weight": torch.zeros(8, 8),
    }
    target = {
        "head.base_layer.weight": torch.zeros(5, 8),
        "head.base_layer.bias": torch.zeros(5),
        "head.lora_A.default.weight": torch.zeros(2, 8),
        "head.lora_B.default.weight": torch.zeros(5, 2),
        "body.weight": torch.zeros(8, 8),
    }
    filtered, skipped = filter_state_dict(source, target)
    assert set(filtered) == {"body.weight"}
    assert "partly incompatible" in skipped["head.lora_A.default.weight"]
    assert "Shape mismatch" in skipped["head.lora_B.default.weight"]
