"""Gate loading-g1: State dict utilities.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

import torch

from anemoi.training.checkpoint.loading.utils import filter_state_dict
from anemoi.training.checkpoint.loading.utils import match_state_dict_keys


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


def test_match_state_dict_keys() -> None:
    """Return missing, unexpected, and shape-mismatched keys."""
    source_dict = {"a": torch.randn(3), "b": torch.randn(5), "c": torch.randn(2)}
    target_dict = {"a": torch.randn(3), "b": torch.randn(10), "d": torch.randn(4)}

    result = match_state_dict_keys(source_dict, target_dict)

    assert "d" in result.missing_in_source  # In target, not in source
    assert "c" in result.unexpected_in_source  # In source, not in target
    assert "b" in result.shape_mismatches  # Both have it, shapes differ
