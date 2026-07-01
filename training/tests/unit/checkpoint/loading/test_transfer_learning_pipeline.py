# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline-level tests for transfer learning checkpoint loading.

These drive the ``LocalSource -> TransferLearningLoader`` plumbing through
:class:`CheckpointPipeline` over a real on-disk checkpoint, covering behaviours
the narrow unit tests in ``test_transfer_learning.py`` do not:

- a *true* shape mismatch (same key, different tensor shape) skipped when
  ``skip_mismatched=True`` (the existing unit test only hits the "key not in
  target" branch);
- a clean full-architecture match transferring every key with nothing skipped;
- a brand-new target-only module (absent from the checkpoint) left at its fresh
  initialisation;
- ``skip_mismatched=False`` raising :class:`CheckpointIncompatibleError`;
- the fine-tuning contract: optimiser/scheduler discarded and saved training
  progress (``epoch``/``global_step``) *not* restored.

Scope: these use toy ``nn.Linear`` models, so the anemoi-specific steps the
loader also runs (format migrations, processor refresh, ``name_to_index``
metadata) are no-ops here; those are covered elsewhere and a realistic exercise
needs a full anemoi model with graph, data-indices and config. The
``skip_mismatched=False`` case is checked at the loader level because it raises
before the source stage is reached.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class TargetArch(nn.Module):
    """Architecture being fine-tuned from the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 8)
        self.head = nn.Linear(8, 3)  # true shape mismatch vs source head
        self.new_head = nn.Linear(8, 2)  # brand-new fine-tuning head, absent from source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


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

    # The mismatched head (true shape mismatch) is skipped with a shape reason
    # for both its weight and bias, and never appears among transferred params.
    skipped = result.metadata["skipped_params"]
    assert "Shape mismatch" in skipped["head.weight"]
    assert "Shape mismatch" in skipped["head.bias"]
    # The source-only aux layer is skipped as absent from the target.
    assert skipped["aux.weight"] == "Key not in target"
    for key in ("head.weight", "head.bias", "aux.weight"):
        assert key not in transferred

    assert result.metadata["loading_strategy"] == "transfer_learning"
    assert getattr(result.model, "weights_initialized", False) is True

    # The fine-tuned model stays trainable and runnable after transfer.
    assert all(p.requires_grad for p in result.model.parameters())
    assert result.model(torch.randn(2, 10)).shape == (2, 3)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_full_match_transfers_every_key(
    source_model: SourceArch,
    saved_checkpoint: Path,
) -> None:
    """A same-architecture target transfers every key with nothing skipped."""
    target = SourceArch()  # identical architecture, different random init

    context = CheckpointContext(model=target, checkpoint_path=saved_checkpoint)
    result = await _pipeline().execute(context)

    # Nothing is skipped and every source key is transferred.
    assert result.metadata["skipped_params"] == {}
    assert set(result.metadata["transferred_params"]) == set(source_model.state_dict())

    # Every parameter now matches the checkpoint, including the previously
    # mismatched/absent layers.
    assert torch.equal(result.model.head.weight, source_model.head.weight)
    assert torch.equal(result.model.aux.weight, source_model.aux.weight)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_leaves_target_mismatched_weights_untouched(
    source_model: SourceArch,
    saved_checkpoint: Path,
) -> None:
    """The mismatched target layer keeps its init while matching layers still load."""
    target = TargetArch()
    original_head = target.head.weight.detach().clone()

    context = CheckpointContext(model=target, checkpoint_path=saved_checkpoint)
    result = await _pipeline().execute(context)

    # Positive control: the encoder really did load from the checkpoint,
    # so a silent no-op load would fail this test...
    assert torch.equal(result.model.encoder.weight, source_model.encoder.weight)
    # ...while the shape-mismatched head kept its fresh initialisation.
    assert torch.equal(result.model.head.weight, original_head)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_leaves_target_only_module_at_init(
    saved_checkpoint: Path,
) -> None:
    """A target-only module absent from the checkpoint is left at fresh init."""
    target = TargetArch()
    original_new_head = target.new_head.weight.detach().clone()

    context = CheckpointContext(model=target, checkpoint_path=saved_checkpoint)
    result = await _pipeline().execute(context)

    assert torch.equal(result.model.new_head.weight, original_new_head)
    # A target-only key is neither transferred nor recorded as skipped:
    # filter_state_dict iterates source keys, so it never sees it.
    assert "new_head.weight" not in result.metadata["transferred_params"]
    assert "new_head.weight" not in result.metadata["skipped_params"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_strict_transfer_raises_on_shape_mismatch(
    source_model: SourceArch,
) -> None:
    """With ``skip_mismatched=False`` a true shape mismatch is fatal.

    Checked at the loader level: the strategy raises before the source stage
    is reached, so the on-disk / pipeline machinery is not involved.
    """
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
async def test_pipeline_discards_optimizer_scheduler_and_progress(
    saved_checkpoint: Path,
) -> None:
    """Transfer learning starts fresh: optimiser/scheduler dropped, progress not restored."""
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

    # The checkpoint carried training progress (epoch=7)...
    assert result.checkpoint_data["epoch"] == 7
    # ...but transfer learning must not restore it into training state
    # (contrast WarmStartLoader, which does).
    assert result.metadata.get("epoch") is None
    assert "global_step" not in result.metadata
