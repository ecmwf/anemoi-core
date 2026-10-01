# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch

from anemoi.training.train.methods.base import BaseTrainingModule


def _draw_after_batch_start(model_comm_group_id: int, epoch: int, batch_idx: int) -> torch.Tensor:
    module = SimpleNamespace(model_comm_group_id=model_comm_group_id, current_epoch=epoch)
    BaseTrainingModule.on_train_batch_start(module, batch={}, batch_idx=batch_idx)
    return torch.rand(4)


def test_batch_random_numbers_do_not_depend_on_earlier_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed run draws the same numbers for a batch as the uninterrupted run."""
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1000")

    uninterrupted = [_draw_after_batch_start(0, epoch=1, batch_idx=batch_idx) for batch_idx in range(3)]
    torch.rand(100)  # random numbers drawn elsewhere, for example before the interruption
    resumed = _draw_after_batch_start(0, epoch=1, batch_idx=2)

    assert torch.equal(resumed, uninterrupted[2])
    assert not torch.equal(uninterrupted[0], uninterrupted[1])


def test_batch_random_numbers_differ_between_model_groups_and_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model groups, such as ensemble members, and epochs draw different numbers for the same batch."""
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1000")

    reference = _draw_after_batch_start(0, epoch=1, batch_idx=2)

    assert not torch.equal(_draw_after_batch_start(1, epoch=1, batch_idx=2), reference)
    assert not torch.equal(_draw_after_batch_start(0, epoch=2, batch_idx=2), reference)


def test_batch_random_numbers_follow_base_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different base seed draws different numbers for the same batch."""
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1000")
    reference = _draw_after_batch_start(0, epoch=0, batch_idx=0)

    monkeypatch.setenv("ANEMOI_BASE_SEED", "1001")

    assert not torch.equal(_draw_after_batch_start(0, epoch=0, batch_idx=0), reference)
