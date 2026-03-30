"""Loading-specific test fixtures."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Minimal model for loading strategy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


class ModelWithMetadata(nn.Module):
    """Model with Anemoi-specific metadata attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self._ckpt_model_name_to_index = None


class SourceModel(nn.Module):
    """Source architecture for transfer learning tests."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)
        self.old_head = nn.Linear(5, 3)


class TargetModel(nn.Module):
    """Target architecture for transfer learning tests (different head)."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)
        self.new_head = nn.Linear(5, 7)  # Different output size


@pytest.fixture
def simple_model() -> SimpleModel:
    return SimpleModel()


@pytest.fixture
def model_with_metadata() -> ModelWithMetadata:
    return ModelWithMetadata()


@pytest.fixture
def target_model() -> TargetModel:
    return TargetModel()


@pytest.fixture
def simple_state_dict() -> dict[str, Any]:
    """State dict matching SimpleModel architecture."""
    return {
        "linear.weight": torch.randn(5, 10),
        "linear.bias": torch.randn(5),
    }


@pytest.fixture
def simple_checkpoint_data(simple_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint data dict as it would appear in context.checkpoint_data."""
    return {"state_dict": simple_state_dict}
