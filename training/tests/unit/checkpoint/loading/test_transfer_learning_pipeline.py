# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end tests for transfer learning checkpoint loading.

These exercise the full ``LocalSource -> TransferLearningLoader`` pipeline over a
real on-disk checkpoint, covering behaviours the narrow unit tests in
``test_transfer_learning.py`` do not:

- a *true* shape mismatch (same key, different tensor shape) being skipped when
  ``skip_mismatched=True`` (the existing unit test only hits the "key not in
  target" branch);
- ``skip_mismatched=False`` raising :class:`CheckpointIncompatibleError`;
- the source-to-loader stage hand-off through :class:`CheckpointPipeline`;
- optimiser/scheduler being discarded by the transfer-learning path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.pipeline import CheckpointPipeline
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from pathlib import Path


class SourceArch(nn.Module):
    """Pretrained architecture saved to the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 8)  # matches target exactly
        self.head = nn.Linear(8, 5)  # same key as target, different out_features
        self.aux = nn.Linear(8, 4)  # key absent from target


class TargetArch(nn.Module):
    """Architecture being fine-tuned from the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 8)
        self.head = nn.Linear(8, 3)  # true shape mismatch vs source head


@pytest.fixture
def source_model() -> SourceArch:
    """A pretrained source model with deterministic weights."""
    torch.manual_seed(0)
    return SourceArch()


@pytest.fixture
def saved_checkpoint(source_model: SourceArch, tmp_path: Path) -> Path:
    """Write the source model to a Lightning-style checkpoint on disk."""
    ckpt_path = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": source_model.state_dict(), "epoch": 7}, ckpt_path)
    return ckpt_path


def _pipeline(*, skip_mismatched: bool = True) -> CheckpointPipeline:
    """Build a source-then-transfer-learning pipeline."""
    return CheckpointPipeline([LocalSource(), TransferLearningLoader(skip_mismatched=skip_mismatched)])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_transfers_matching_skips_mismatched(
    source_model: SourceArch,
    saved_checkpoint: Path,
) -> None:
    """Matching layers load; same-key shape mismatch and absent keys are skipped."""
    target = TargetArch()
    context = CheckpointContext(model=target, checkpoint_path=saved_checkpoint)

    result = await _pipeline().execute(context)

    # Matching encoder weights are transferred verbatim from the checkpoint.
    assert torch.equal(result.model.encoder.weight, source_model.encoder.weight)
    transferred = result.metadata["transferred_params"]
    assert "encoder.weight" in transferred
    assert "encoder.bias" in transferred

    # The mismatched head (true shape mismatch) is skipped with a shape reason.
    skipped = result.metadata["skipped_params"]
    assert "Shape mismatch" in skipped["head.weight"]
    # The source-only aux layer is skipped as absent from the target.
    assert skipped["aux.weight"] == "Key not in target"

    assert result.metadata["loading_strategy"] == "transfer_learning"
    assert getattr(result.model, "weights_initialized", False) is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_leaves_target_mismatched_weights_untouched(
    saved_checkpoint: Path,
) -> None:
    """The mismatched target layer keeps its fresh initialisation."""
    target = TargetArch()
    original_head = target.head.weight.detach().clone()

    context = CheckpointContext(model=target, checkpoint_path=saved_checkpoint)
    result = await _pipeline().execute(context)

    assert torch.equal(result.model.head.weight, original_head)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_strict_transfer_raises_on_shape_mismatch(
    source_model: SourceArch,
) -> None:
    """With ``skip_mismatched=False`` a true shape mismatch is fatal."""
    target = TargetArch()
    context = CheckpointContext(
        model=target,
        checkpoint_data={"state_dict": source_model.state_dict()},
    )

    loader = TransferLearningLoader(skip_mismatched=False)
    with pytest.raises(CheckpointIncompatibleError) as exc_info:
        await loader.process(context)

    # The error names the offending key so the failure is actionable.
    assert "head" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_discards_optimizer_and_scheduler(
    saved_checkpoint: Path,
) -> None:
    """Transfer learning starts fresh training state, dropping optimiser/scheduler."""
    target = TargetArch()
    optimizer = torch.optim.Adam(target.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    context = CheckpointContext(
        model=target,
        checkpoint_path=saved_checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    result = await _pipeline().execute(context)

    assert result.optimizer is None
    assert result.scheduler is None
