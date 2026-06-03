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
    """Critical: PR #458 uses non-mutating filter. Legacy code mutated. We do NOT mutate."""
    source = {"a": torch.randn(3), "b": torch.randn(5)}
    target = {"a": torch.randn(3), "b": torch.randn(10)}
    original_keys = set(source.keys())

    filter_state_dict(source, target)

    assert set(source.keys()) == original_keys
