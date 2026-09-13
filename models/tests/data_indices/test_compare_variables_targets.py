# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection

# Checkpoint trained without targets; the data now inserts two target variables in the middle,
# shifting the absolute indices of every later variable.
CKPT = {"a": 0, "b": 1, "c": 2, "d": 3}
DATA_WITH_TARGETS = {"a": 0, "b": 1, "obs_1": 2, "obs_2": 3, "c": 4, "d": 5}


def _indices(name_to_index: dict, target: list[str] | None = None) -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": target or []}), name_to_index)


def test_added_target_variables_are_tolerated() -> None:
    _indices(DATA_WITH_TARGETS, target=["obs_1", "obs_2"]).compare_variables(CKPT, DATA_WITH_TARGETS)


def test_removed_checkpoint_target_variables_are_tolerated_when_named() -> None:
    di = _indices(CKPT)
    di.compare_variables(DATA_WITH_TARGETS, CKPT, ignore_variables=["obs_1", "obs_2"])
    with pytest.raises(ValueError):
        di.compare_variables(DATA_WITH_TARGETS, CKPT)


def test_reorder_of_model_variables_still_raises() -> None:
    reordered = {"a": 0, "c": 1, "obs_1": 2, "obs_2": 3, "b": 4, "d": 5}
    with pytest.raises(ValueError):
        _indices(reordered, target=["obs_1", "obs_2"]).compare_variables(CKPT, reordered)


def test_added_model_variable_still_raises() -> None:
    data = {"a": 0, "b": 1, "obs_1": 2, "e": 3, "c": 4, "d": 5}
    with pytest.raises(ValueError):
        _indices(data, target=["obs_1"]).compare_variables(CKPT, data)


def test_identical_and_rename_same_order_unchanged() -> None:
    _indices(CKPT).compare_variables(CKPT, CKPT)
    _indices({"a": 0, "b": 1, "c": 2, "x": 3}).compare_variables(CKPT, {"a": 0, "b": 1, "c": 2, "x": 3})
