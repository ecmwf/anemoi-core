"""Source-specific test fixtures."""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def sample_checkpoint(tmp_path: Path) -> Path:
    """Create a real checkpoint file for source testing."""
    ckpt_path = tmp_path / "test.ckpt"
    state = {"state_dict": {"layer.weight": torch.randn(10, 5)}, "epoch": 3}
    torch.save(state, ckpt_path)
    return ckpt_path
