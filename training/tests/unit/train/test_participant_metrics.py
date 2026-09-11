# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from pytest_mock import MockFixture

from anemoi.training.train.participant_metrics import ParticipantMetrics


def _values(results: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return {p: {m: round(float(v), 6) for m, v in metrics.items()} for p, metrics in results.items()}


def test_weighted_means_per_participant() -> None:
    acc = ParticipantMetrics()
    acc.update("h1", {"mse": torch.tensor(1.0), "loss": torch.tensor(10.0)}, batch_size=2)
    acc.update("h1", {"mse": torch.tensor([4.0]), "loss": torch.tensor(40.0)}, batch_size=1)  # shape [1] as scalar
    acc.update("h2", {"mse": torch.tensor(7.0)}, batch_size=3)

    assert _values(acc.compute()) == {"h1": {"mse": 2.0, "loss": 20.0}, "h2": {"mse": 7.0}}


def test_none_participant_is_ignored_and_reset_clears() -> None:
    acc = ParticipantMetrics()
    acc.update(None, {"mse": torch.tensor(1.0)}, batch_size=2)
    assert acc.compute() == {}

    acc.update("h1", {"mse": torch.tensor(1.0)}, batch_size=2)
    acc.reset()
    assert acc.compute() == {}


def test_compute_agrees_on_keys_and_reduces_across_ranks(mocker: MockFixture) -> None:
    """Keys are the union over ranks (fixed order); missing keys take part in the all-reduce with zeros."""
    acc = ParticipantMetrics()
    acc.update("h1", {"mse": torch.tensor(1.0)}, batch_size=2)

    mocker.patch("torch.distributed.is_available", return_value=True)
    mocker.patch("torch.distributed.is_initialized", return_value=True)
    mocker.patch("torch.distributed.get_world_size", return_value=2)

    def all_gather_object(out: list, obj: list) -> None:  # the other rank only saw h2
        out[0], out[1] = obj, [("h2", "mse")]

    reduced = []

    def all_reduce(tensor: torch.Tensor) -> None:  # the other rank contributes sum 6 over 3 samples to everything
        reduced.append(tensor.clone())
        tensor.add_(torch.tensor([6.0, 3.0], dtype=torch.float64))

    mocker.patch("torch.distributed.all_gather_object", side_effect=all_gather_object)
    mocker.patch("torch.distributed.all_reduce", side_effect=all_reduce)

    results = _values(acc.compute(device="cpu"))

    assert [tuple(t.tolist()) for t in reduced] == [(2.0, 2.0), (0.0, 0.0)]  # h1 local sums, h2 zeros
    assert results == {"h1": {"mse": 1.6}, "h2": {"mse": 2.0}}
