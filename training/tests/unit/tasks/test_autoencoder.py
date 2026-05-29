# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import torch

from anemoi.training.tasks import Autoencoder


def test_autoencoder_input_and_output_offsets_are_both_zero() -> None:
    """Autoencoder operates on a single snapshot at t=0."""
    task = Autoencoder()
    assert task.offsets.input == [datetime.timedelta(0)]
    assert task.offsets.output == [datetime.timedelta(0)]
    assert task.offsets.all == [datetime.timedelta(0)]


def test_autoencoder_has_exactly_one_step_with_no_kwargs() -> None:
    """Autoencoder runs exactly one step and passes no step-specific kwargs."""
    task = Autoencoder()
    assert list(task.steps("training")) == [{}]
    assert list(task.steps("validation")) == [{}]
    assert list(task.steps("testing")) == [{}]


def test_autoencoder_advance_input_returns_input_unchanged() -> None:
    """advance_input for a single-step task is a no-op (returns first positional arg)."""
    task = Autoencoder()
    x = {"data": torch.randn(2, 1, 1, 4, 2)}
    result = task.advance_input(x, {}, {})
    assert result is x
