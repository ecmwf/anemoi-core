# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.imputer import InputImputer
from anemoi.models.preprocessing.remapper import Remapper


@pytest.fixture()
def input_remapper():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {
                    "cos_sin": {
                        "d": ["cos_d", "sin_d"],
                    }
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": {
                    "d": ["cos_d", "sin_d"],
                },
            },
        },
    )
    statistics = {}
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "d": 4, "other": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return Remapper(config=config.data.remapper, data_indices=data_indices, statistics=statistics)


@pytest.fixture()
def input_remapper_1d():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {
                    "log1p": "d",
                    "sqrt": "q",
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    statistics = {}
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "d": 4, "other": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return Remapper(config=config.data.remapper, data_indices=data_indices, statistics=statistics)


@pytest.fixture()
def input_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {
                    "cos_sin": {
                        "d": ["cos_d", "sin_d"],
                    }
                },
                "imputer": {"default": "none", "mean": ["y", "d"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": {
                    "d": ["cos_d", "sin_d"],
                },
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "d": 4, "other": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return InputImputer(config=config.data.imputer, data_indices=data_indices, statistics=statistics)


def test_remap_not_inplace(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    input_remapper(x, in_place=False)
    assert torch.allclose(
        x,
        torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]]),
    )


def test_remap(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    expected_output = torch.Tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, -0.8660254, 0.5],
            [6.0, 7.0, 8.0, 9.0, 10.0, -0.93358043, -0.35836795],
        ]
    )
    assert torch.allclose(input_remapper.transform(x), expected_output)


def test_inverse_transform(input_remapper) -> None:
    x = torch.Tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, -0.8660254, 0.5],
            [6.0, 7.0, 8.0, 9.0, 10.0, -0.93358043, -0.35836795],
        ]
    )
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    assert torch.allclose(input_remapper.inverse_transform(x), expected_output)


def test_remap_inverse_transform(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    assert torch.allclose(
        input_remapper.inverse_transform(input_remapper.transform(x, in_place=False), in_place=False),
        x,
    )


def test_transform_loss_mask(input_imputer, input_remapper) -> None:
    x = torch.Tensor([[1.0, np.nan, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, np.nan, 10.0]])
    expected_output = torch.Tensor([[1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
    input_imputer.transform(x)
    input_remapper.transform(x)
    loss_mask_training = input_imputer.loss_mask_training
    loss_mask_training = input_remapper.transform_loss_mask(loss_mask_training)
    assert torch.allclose(loss_mask_training, expected_output)


def test_monomap_transform(input_remapper_1d) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    expected_output = torch.Tensor(
        [
            [1.0, 2.0, 3.0, np.sqrt(4.0), np.log1p(150.0), 5.0],
            [6.0, 7.0, 8.0, np.sqrt(9.0), np.log1p(201.0), 10.0],
        ]
    )
    assert torch.allclose(input_remapper_1d.transform(x, in_place=False), expected_output)
    # inference mode (without prognostic variables)
    assert torch.allclose(
        input_remapper_1d.transform(
            x[..., input_remapper_1d.data_indices.data.todict()["input"]["full"]], in_place=False
        ),
        expected_output[..., input_remapper_1d.data_indices.data.todict()["input"]["full"]],
    )
    # this one actually changes the values in x so need to be last
    assert torch.allclose(input_remapper_1d.transform(x), expected_output)


def test_monomap_inverse_transform(input_remapper_1d) -> None:
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    y = torch.Tensor(
        [
            [1.0, 2.0, 3.0, np.sqrt(4.0), np.log1p(150.0), 5.0],
            [6.0, 7.0, 8.0, np.sqrt(9.0), np.log1p(201.0), 10.0],
        ]
    )
    assert torch.allclose(input_remapper_1d.inverse_transform(y, in_place=False), expected_output)
    # inference mode (without prognostic variables)
    assert torch.allclose(
        input_remapper_1d.inverse_transform(
            y[..., input_remapper_1d.data_indices.data.todict()["output"]["full"]], in_place=False
        ),
        expected_output[..., input_remapper_1d.data_indices.data.todict()["output"]["full"]],
    )


def test_unsupported_remapper():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {"log1p": "q", "cos_sin": "d"},
                "forcing": [],
                "diagnostic": [],
            },
        }
    )
    statistics = {}
    name_to_index = {"x": 0, "y": 1, "q": 2, "d": 3}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)

    with pytest.raises(NotImplementedError):
        Remapper(
            config=config.data.remapper,
            data_indices=data_indices,
            statistics=statistics,
        )
