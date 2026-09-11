# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch.utils.data import default_collate

from anemoi.training.data.batch_meta import META_KEY
from anemoi.training.data.batch_meta import PARTICIPANT_FIELD
from anemoi.training.data.batch_meta import meta_participant
from anemoi.training.data.batch_meta import split_meta


def test_split_meta_without_metadata_returns_batch_unchanged() -> None:
    batch = {"data": torch.zeros(1)}
    stripped, meta = split_meta(batch)
    assert stripped is batch
    assert meta is None

    tensor = torch.zeros(1)
    assert split_meta(tensor) == (tensor, None)


def test_split_meta_strips_reserved_key_without_mutating_input() -> None:
    tensor = torch.zeros(1)
    batch = {"data": tensor, META_KEY: {PARTICIPANT_FIELD: ["h1", "h1"]}}
    stripped, meta = split_meta(batch)

    assert stripped == {"data": tensor}
    assert meta == {PARTICIPANT_FIELD: ["h1", "h1"]}
    assert META_KEY in batch  # input untouched


def test_meta_participant() -> None:
    assert meta_participant(None) is None
    assert meta_participant({}) is None
    assert meta_participant({PARTICIPANT_FIELD: "h1"}) == "h1"
    assert meta_participant({PARTICIPANT_FIELD: ["h2", "h2", "h2"]}) == "h2"

    with pytest.raises(ValueError, match=r"mixes participants \['h1', 'h2'\]"):
        meta_participant({PARTICIPANT_FIELD: ["h1", "h2"]})


def test_default_collate_keeps_participant_strings_per_sample() -> None:
    samples = [{"data": torch.full((2,), float(i)), META_KEY: {PARTICIPANT_FIELD: "h1"}} for i in range(3)]
    batch = default_collate(samples)

    assert batch["data"].shape == (3, 2)
    assert batch[META_KEY] == {PARTICIPANT_FIELD: ["h1", "h1", "h1"]}
    stripped, meta = split_meta(batch)
    assert set(stripped) == {"data"}
    assert meta_participant(meta) == "h1"
