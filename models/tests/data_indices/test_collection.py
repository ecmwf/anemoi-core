# (C) Copyright 2024 Anemoi contributors.
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
                "remapped": {
                    "e": ["e_1", "e_2"],
                    "d": ["d_1", "d_2"],
                },
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    return IndexCollection(config=config, name_to_index=name_to_index)


def test_dataindices_init(data_indices) -> None:
    assert data_indices.data.input.includes == ["x", "e"]
    assert data_indices.data.input.excludes == ["z", "q"]
    assert data_indices.internal_data.input.includes == ["x", "e_1", "e_2"]
    assert data_indices.internal_data.input.excludes == ["z", "q"]
    assert data_indices.internal_data.output.includes == ["z", "q"]
    assert data_indices.internal_data.output.excludes == ["x", "e_1", "e_2"]
    assert data_indices.data.output.includes == ["z", "q"]
    assert data_indices.data.output.excludes == ["x", "e"]
    assert data_indices.model.input.includes == ["x", "e"]
    assert data_indices.model.input.excludes == []
    assert data_indices.internal_model.input.includes == ["x", "e_1", "e_2"]
    assert data_indices.internal_model.input.excludes == []
    assert data_indices.internal_model.output.includes == ["z", "q"]
    assert data_indices.internal_model.output.excludes == []
    assert data_indices.model.output.includes == ["z", "q"]
    assert data_indices.model.output.excludes == []
    assert data_indices.data.input.name_to_index == {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    assert data_indices.internal_data.input.name_to_index == {
        "x": 0,
        "y": 1,
        "z": 2,
        "q": 3,
        "other": 4,
        "e_1": 5,
        "e_2": 6,
        "d_1": 7,
        "d_2": 8,
    }
    assert data_indices.internal_data.output.name_to_index == {
        "x": 0,
        "y": 1,
        "z": 2,
        "q": 3,
        "other": 4,
        "e_1": 5,
        "e_2": 6,
        "d_1": 7,
        "d_2": 8,
    }
    assert data_indices.data.output.name_to_index == {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    assert data_indices.model.input.name_to_index == {"x": 0, "y": 1, "e": 2, "d": 3, "other": 4}
    assert data_indices.internal_model.input.name_to_index == {
        "x": 0,
        "y": 1,
        "other": 2,
        "e_1": 3,
        "e_2": 4,
        "d_1": 5,
        "d_2": 6,
    }
    assert data_indices.internal_model.output.name_to_index == {"y": 0, "z": 1, "q": 2, "other": 3, "d_1": 4, "d_2": 5}
    assert data_indices.model.output.name_to_index == {"y": 0, "z": 1, "q": 2, "d": 3, "other": 4}


def test_dataindices_max(data_indices) -> None:
    assert max(data_indices.data.input.full) == max(data_indices.data.input.name_to_index.values())
    assert max(data_indices.internal_data.input.full) == max(data_indices.internal_data.input.name_to_index.values())
    assert max(data_indices.internal_data.output.full) == max(data_indices.internal_data.output.name_to_index.values())
    assert max(data_indices.data.output.full) == max(data_indices.data.output.name_to_index.values())
    assert max(data_indices.model.input.full) == max(data_indices.model.input.name_to_index.values())
    assert max(data_indices.internal_model.input.full) == max(data_indices.internal_model.input.name_to_index.values())
    assert max(data_indices.internal_model.output.full) == max(
        data_indices.internal_model.output.name_to_index.values()
    )
    assert max(data_indices.model.output.full) == max(data_indices.model.output.name_to_index.values())


def test_dataindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 4, 5, 6]).to(torch.int),
            "forcing": torch.Tensor([0, 4]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 5, 6]).to(torch.int),
        },
        "output": {
            "full": torch.Tensor([1, 2, 3, 5, 6]).to(torch.int),
            "forcing": torch.Tensor([0, 4]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 5, 6]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.data.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])


def test_internaldataindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 4, 5, 6, 7, 8]).to(torch.int),
            "forcing": torch.Tensor([0, 5, 6]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 4, 7, 8]).to(torch.int),
        },
        "output": {
            "full": torch.Tensor([1, 2, 3, 4, 7, 8]).to(torch.int),
            "forcing": torch.Tensor([0, 5, 6]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 4, 7, 8]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.internal_data.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])


def test_modelindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "forcing": torch.Tensor([0, 2]).to(torch.int),
            "diagnostic": torch.Tensor([]).to(torch.int),
            "prognostic": torch.Tensor([1, 3, 4]).to(torch.int),
        },
        "output": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "forcing": torch.Tensor([]).to(torch.int),
            "diagnostic": torch.Tensor([1, 2]).to(torch.int),
            "prognostic": torch.Tensor([0, 3, 4]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.model.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])


def test_internalmodelindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 2, 3, 4, 5, 6]).to(torch.int),
            "forcing": torch.Tensor([0, 3, 4]).to(torch.int),
            "diagnostic": torch.Tensor([]).to(torch.int),
            "prognostic": torch.Tensor([1, 2, 5, 6]).to(torch.int),
        },
        "output": {
            "full": torch.Tensor([0, 1, 2, 3, 4, 5]).to(torch.int),
            "forcing": torch.Tensor([]).to(torch.int),
            "diagnostic": torch.Tensor([1, 2]).to(torch.int),
            "prognostic": torch.Tensor([0, 3, 4, 5]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.internal_model.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])
