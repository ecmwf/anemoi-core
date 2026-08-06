# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection


@pytest.fixture()
def data_indices():
    config = DictConfig(
        {
            "data": {
                "forcing": ["x", "e"],
                "diagnostic": ["z", "q"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    return IndexCollection(config.data, name_to_index=name_to_index)


@pytest.fixture()
def data_indices_with_target():
    # mimicks example where we try to predict precipitation (tp) by comparing predictions with point measurements (tp_point) and radar data (tp_radar) and using topography (dem) as forcing
    config = DictConfig(
        {
            "data": {
                "forcing": ["tp_point", "tp_radar", "dem"],
                "diagnostic": [],
                "target": ["tp_point", "tp_radar"],
            },
        },
    )
    name_to_index = {"tp_point": 0, "tp_radar": 1, "tp": 2, "dem": 3}
    return IndexCollection(config.data, name_to_index=name_to_index)


def test_dataindices_init(data_indices) -> None:
    # if the variables are correctly mapped to their indices we don't have to keep the order
    assert set(data_indices.data.input.includes) == {"x", "e", "y", "d", "other"}
    assert set(data_indices.data.input.excludes) == {"z", "q"}
    assert set(data_indices.data.output.includes) == {"z", "q", "y", "d", "other"}
    assert set(data_indices.data.output.excludes) == {"x", "e"}
    assert set(data_indices.model.input.includes) == {"x", "e", "y", "d", "other"}
    assert set(data_indices.model.input.excludes) == set()
    assert set(data_indices.model.output.includes) == {"z", "q", "y", "d", "other"}
    assert set(data_indices.model.output.excludes) == set()
    assert data_indices.data.input.name_to_index == {
        "x": 0,
        "y": 1,
        "z": 2,
        "q": 3,
        "e": 4,
        "d": 5,
        "other": 6,
    }
    assert data_indices.data.output.name_to_index == {
        "x": 0,
        "y": 1,
        "z": 2,
        "q": 3,
        "e": 4,
        "d": 5,
        "other": 6,
    }
    assert data_indices.model.input.name_to_index == {
        "x": 0,
        "y": 1,
        "e": 2,
        "d": 3,
        "other": 4,
    }
    assert data_indices.model.output.name_to_index == {
        "y": 0,
        "z": 1,
        "q": 2,
        "d": 3,
        "other": 4,
    }


def test_dataindices_max(data_indices) -> None:
    assert max(data_indices.data.input.full) == max(data_indices.data.input.name_to_index.values())
    assert max(data_indices.data.output.full) == max(data_indices.data.output.name_to_index.values())
    assert max(data_indices.model.input.full) == max(data_indices.model.input.name_to_index.values())
    assert max(data_indices.model.output.full) == max(data_indices.model.output.name_to_index.values())


def test_dataindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 4, 5, 6]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([0, 4]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 5, 6]).to(torch.int),
            "name_to_index": dict(x=0, y=1, z=2, q=3, e=4, d=5, other=6),
        },
        "output": {
            "full": torch.Tensor([1, 2, 3, 5, 6]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([0, 4]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 5, 6]).to(torch.int),
            "name_to_index": dict(x=0, y=1, z=2, q=3, e=4, d=5, other=6),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.data.todict()[key].items():
            assert subkey in expected_output[key]
            if isinstance(value, dict):
                assert value == expected_output[key][subkey]
            else:
                assert torch.allclose(value, expected_output[key][subkey])


def test_modelindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([0, 2]).to(torch.int),
            "diagnostic": torch.Tensor([]).to(torch.int),
            "prognostic": torch.Tensor([1, 3, 4]).to(torch.int),
            "name_to_index": dict(x=0, y=1, e=2, d=3, other=4),
        },
        "output": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([]).to(torch.int),
            "diagnostic": torch.Tensor([1, 2]).to(torch.int),
            "prognostic": torch.Tensor([0, 3, 4]).to(torch.int),
            "name_to_index": dict(y=0, z=1, q=2, d=3, other=4),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.model.todict()[key].items():
            assert subkey in expected_output[key]
            if isinstance(value, dict):
                assert value == expected_output[key][subkey]
            else:
                assert torch.allclose(value, expected_output[key][subkey])


def test_data_indices_with_target(data_indices_with_target) -> None:
    assert set(data_indices_with_target.data.input.includes) == {
        "tp",
        "tp_point",
        "tp_radar",
        "dem",
    }
    assert set(data_indices_with_target.data.input.excludes) == set()
    assert set(data_indices_with_target.data.output.includes) == {
        "tp",
        "tp_point",
        "tp_radar",
    }
    assert set(data_indices_with_target.data.output.excludes) == {"dem"}
    assert set(data_indices_with_target.model.input.includes) == {
        "tp",
        "tp_point",
        "tp_radar",
        "dem",
    }
    assert set(data_indices_with_target.model.input.excludes) == set()
    assert set(data_indices_with_target.model.output.includes) == {"tp"}  # the model only predicts tp
    assert set(data_indices_with_target.model.output.excludes) == set()
    assert (
        data_indices_with_target.data.input.name_to_index
        == data_indices_with_target.model.input.name_to_index
        == {"tp_point": 0, "tp_radar": 1, "tp": 2, "dem": 3}
    )
    assert data_indices_with_target.model.output.name_to_index == {"tp": 0}


def test_data_indices_cross_space_positions(data_indices) -> None:
    assert data_indices.data_full_ordered_names == [
        "x",
        "y",
        "z",
        "q",
        "e",
        "d",
        "other",
    ]
    assert data_indices.data_full_name_to_position == {
        "x": 0,
        "y": 1,
        "z": 2,
        "q": 3,
        "e": 4,
        "d": 5,
        "other": 6,
    }
    assert data_indices.data_output_positions_in_data_full == [1, 2, 3, 5, 6]
    assert data_indices.model_output_positions_in_data_full == [1, 2, 3, 5, 6]
    assert data_indices.model_output_positions_in_data_output == [0, 1, 2, 3, 4]
    assert data_indices.model_output_in_data_output_is_identity is True
    assert data_indices.model_output_in_data_output_is_contiguous is True
    assert data_indices.model_output_in_data_output_contiguous_start == 0
    assert data_indices.model_output_in_data_output_contiguous_length == 5


def test_data_indices_cross_space_positions_with_target(
    data_indices_with_target,
) -> None:
    assert data_indices_with_target.data_full_ordered_names == [
        "tp_point",
        "tp_radar",
        "tp",
        "dem",
    ]
    assert data_indices_with_target.data_output_positions_in_data_full == [0, 1, 2]
    assert data_indices_with_target.model_output_positions_in_data_full == [2]
    assert data_indices_with_target.model_output_positions_in_data_output == [2]
    assert data_indices_with_target.model_output_in_data_output_is_identity is False
    assert data_indices_with_target.model_output_in_data_output_is_contiguous is True
    assert data_indices_with_target.model_output_in_data_output_contiguous_start == 2
    assert data_indices_with_target.model_output_in_data_output_contiguous_length == 1


# ── compare_variables: fine-tuning into a model with FEWER variables (issue #838) ──

_CKPT = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def test_compare_variables_subset_raises_without_allow_subset(data_indices) -> None:
    """A strict variable subset raises by default — this is the #838 pre-fix failure."""
    data = {"a": 0, "b": 1, "c": 2}  # dropped d, e
    with pytest.raises(ValueError, match="variable order"):
        data_indices.compare_variables(_CKPT, data)


def test_compare_variables_subset_passes_with_allow_subset(data_indices) -> None:
    """A strict subset with the same relative order is accepted when allow_subset=True (#838 fix)."""
    data = {"a": 0, "b": 1, "c": 2}  # dropped the tail
    data_indices.compare_variables(_CKPT, data, allow_subset=True)  # must not raise


def test_compare_variables_subset_dropping_middle_reindexes_ok(data_indices) -> None:
    """Dropping middle variables re-indexes the survivors; still accepted with allow_subset."""
    data = {"a": 0, "c": 1, "e": 2}  # relative order a<c<e preserved despite reindexing
    data_indices.compare_variables(_CKPT, data, allow_subset=True)  # must not raise


def test_compare_variables_reordered_subset_still_raises(data_indices) -> None:
    """A subset that reorders the shared variables is a genuine mismatch even with allow_subset."""
    data = {"c": 0, "a": 1, "e": 2}  # relative order changed
    with pytest.raises(ValueError):
        data_indices.compare_variables(_CKPT, data, allow_subset=True)


def test_compare_variables_added_variable_still_raises(data_indices) -> None:
    """Introducing a variable absent from the checkpoint is not a subset; allow_subset does not help."""
    data = {"a": 0, "b": 1, "new": 2}  # 'new' not in the checkpoint
    with pytest.raises(ValueError, match="variable order"):
        data_indices.compare_variables(_CKPT, data, allow_subset=True)


def test_compare_variables_exact_match_passes(data_indices) -> None:
    """An exact match never raises, regardless of allow_subset."""
    data_indices.compare_variables(_CKPT, _CKPT, allow_subset=False)
    data_indices.compare_variables(_CKPT, _CKPT, allow_subset=True)
