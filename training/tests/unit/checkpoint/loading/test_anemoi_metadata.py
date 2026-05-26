"""Gate loading-g6: Anemoi Metadata Handling.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class ModelWithMetadata(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self._ckpt_model_name_to_index = None


@pytest.mark.asyncio
async def test_preserves_ckpt_model_name_to_index() -> None:
    """All loaders must preserve _ckpt_model_name_to_index from hyper_parameters."""
    model = ModelWithMetadata()
    name_to_index = {"temperature": 0, "wind_u": 1, "wind_v": 2}
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": type("obj", (), {"name_to_index": name_to_index})()},
    }

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.model._ckpt_model_name_to_index == name_to_index
