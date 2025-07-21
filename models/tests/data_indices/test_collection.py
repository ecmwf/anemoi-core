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
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    return IndexCollection(config=config, name_to_index=name_to_index)


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
    return IndexCollection(config=config, name_to_index=name_to_index)


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
    assert data_indices.data.input.name_to_index == {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    assert data_indices.data.output.name_to_index == {"x": 0, "y": 1, "z": 2, "q": 3, "e": 4, "d": 5, "other": 6}
    assert data_indices.model.input.name_to_index == {"x": 0, "y": 1, "e": 2, "d": 3, "other": 4}
    assert data_indices.model.output.name_to_index == {"y": 0, "z": 1, "q": 2, "d": 3, "other": 4}


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
        },
        "output": {
            "full": torch.Tensor([1, 2, 3, 5, 6]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([0, 4]).to(torch.int),
            "diagnostic": torch.Tensor([2, 3]).to(torch.int),
            "prognostic": torch.Tensor([1, 5, 6]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.data.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])


def test_modelindices_todict(data_indices) -> None:
    expected_output = {
        "input": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([0, 2]).to(torch.int),
            "diagnostic": torch.Tensor([]).to(torch.int),
            "prognostic": torch.Tensor([1, 3, 4]).to(torch.int),
        },
        "output": {
            "full": torch.Tensor([0, 1, 2, 3, 4]).to(torch.int),
            "target": torch.Tensor([]).to(torch.int),
            "forcing": torch.Tensor([]).to(torch.int),
            "diagnostic": torch.Tensor([1, 2]).to(torch.int),
            "prognostic": torch.Tensor([0, 3, 4]).to(torch.int),
        },
    }

    for key in ["output", "input"]:
        for subkey, value in data_indices.model.todict()[key].items():
            assert subkey in expected_output[key]
            assert torch.allclose(value, expected_output[key][subkey])


def test_data_indices_with_target(data_indices_with_target) -> None:
    assert set(data_indices_with_target.data.input.includes) == {"tp", "tp_point", "tp_radar", "dem"}
    assert set(data_indices_with_target.data.input.excludes) == set()
    assert set(data_indices_with_target.data.output.includes) == {"tp", "tp_point", "tp_radar"}
    assert set(data_indices_with_target.data.output.excludes) == {"dem"}
    assert set(data_indices_with_target.model.input.includes) == {"tp", "tp_point", "tp_radar", "dem"}
    assert set(data_indices_with_target.model.input.excludes) == set()
    assert set(data_indices_with_target.model.output.includes) == {"tp"}  # the model only predicts tp
    assert set(data_indices_with_target.model.output.excludes) == set()
    assert (
        data_indices_with_target.data.input.name_to_index
        == data_indices_with_target.model.input.name_to_index
        == {"tp_point": 0, "tp_radar": 1, "tp": 2, "dem": 3}
    )
    assert data_indices_with_target.model.output.name_to_index == {"tp": 0}
