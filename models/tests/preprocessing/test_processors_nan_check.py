# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pickle

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors

# Full data tensor columns: prognostic "x" (0), forcing "z" (1), diagnostic "tc" (2) and "other_diag" (3)
NAME_TO_INDEX = {"x": 0, "z": 1, "tc": 2, "other_diag": 3}


def _data_indices(allow_nans: list[str]) -> IndexCollection:
    config = DictConfig({"forcing": ["z"], "diagnostic": ["tc", "other_diag"], "allow_nans": allow_nans})
    return IndexCollection(data_config=config, name_to_index=NAME_TO_INDEX)


@pytest.fixture()
def data_indices() -> IndexCollection:
    return _data_indices(["tc"])


@pytest.fixture()
def batch() -> torch.Tensor:
    # batch, time, ensemble, grid, variables
    return torch.ones(2, 3, 1, 5, len(NAME_TO_INDEX))


def test_nan_in_allowed_variable_passes(data_indices: IndexCollection, batch: torch.Tensor) -> None:
    batch[..., NAME_TO_INDEX["tc"]] = float("nan")
    processors = Processors([], data_indices=data_indices)

    out = processors(batch)

    assert torch.isnan(out[..., NAME_TO_INDEX["tc"]]).all()


@pytest.mark.parametrize("name", ["x", "z", "other_diag"])
def test_nan_in_other_variable_fails(data_indices: IndexCollection, batch: torch.Tensor, name: str) -> None:
    # NaNs in the allowed variable do not hide NaNs in any other variable, including other diagnostics
    batch[..., NAME_TO_INDEX["tc"]] = float("nan")
    batch[0, 0, 0, 0, NAME_TO_INDEX[name]] = float("nan")
    processors = Processors([], data_indices=data_indices)

    with pytest.raises(AssertionError, match=r"NaNs \(1\) found"):
        processors(batch)


def test_nan_in_model_input_tensor_fails(data_indices: IndexCollection, batch: torch.Tensor) -> None:
    # Model-input tensors (no diagnostic variables) are checked in full
    model_input = batch[..., data_indices.data.input.full.long()].clone()
    model_input[0, 0, 0, 0, 0] = float("nan")
    processors = Processors([], data_indices=data_indices)

    with pytest.raises(AssertionError, match=r"NaNs \(1\) found"):
        processors(model_input)


@pytest.mark.parametrize("name", ["tc", "other_diag"])
def test_no_allow_nans_checks_all_variables(batch: torch.Tensor, name: str) -> None:
    batch[..., NAME_TO_INDEX[name]] = float("nan")
    processors = Processors([], data_indices=_data_indices([]))

    with pytest.raises(AssertionError, match="NaNs"):
        processors(batch)


def test_no_data_indices_checks_all_variables(batch: torch.Tensor) -> None:
    batch[..., NAME_TO_INDEX["tc"]] = float("nan")
    processors = Processors([])

    with pytest.raises(AssertionError, match="NaNs"):
        processors(batch)


def test_inverse_processors_do_not_check_nans(data_indices: IndexCollection, batch: torch.Tensor) -> None:
    batch[..., NAME_TO_INDEX["other_diag"]] = float("nan")
    processors = Processors([], inverse=True, data_indices=data_indices)

    assert torch.isnan(processors(batch)).any()


def test_nan_check_only_runs_on_first_call(data_indices: IndexCollection, batch: torch.Tensor) -> None:
    processors = Processors([], data_indices=data_indices)
    processors(batch)

    batch[0, 0, 0, 0, NAME_TO_INDEX["x"]] = float("nan")

    assert torch.isnan(processors(batch)).any()


def _legacy_data_indices() -> IndexCollection:
    """Index collection as unpickled from a checkpoint created before `allow_nans` existed."""
    indices = _data_indices([])
    del indices.__dict__["allow_nans"]
    legacy = pickle.loads(pickle.dumps(indices))
    assert "allow_nans" not in legacy.__dict__
    return legacy


def test_legacy_index_collection_without_nans_passes(batch: torch.Tensor) -> None:
    processors = Processors([], data_indices=_legacy_data_indices())

    assert not torch.isnan(processors(batch)).any()


@pytest.mark.parametrize("name", ["x", "tc", "other_diag"])
def test_legacy_index_collection_checks_all_variables(batch: torch.Tensor, name: str) -> None:
    batch[..., NAME_TO_INDEX[name]] = float("nan")
    processors = Processors([], data_indices=_legacy_data_indices())

    with pytest.raises(AssertionError, match="NaNs"):
        processors(batch)
