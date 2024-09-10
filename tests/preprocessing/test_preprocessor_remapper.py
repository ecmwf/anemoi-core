# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
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


def test_remap_not_inplace(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    input_remapper(x, in_place=False)
    assert torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]]))


def test_remap(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    expected_output = torch.Tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, -0.8660254, 0.5], [6.0, 7.0, 8.0, 9.0, 10.0, -0.93358043, -0.35836795]]
    )
    assert torch.allclose(input_remapper.transform(x), expected_output)


def test_inverse_transform(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, -0.8660254, 0.5], [6.0, 7.0, 8.0, 9.0, 10.0, -0.93358043, -0.35836795]])
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    assert torch.allclose(input_remapper.inverse_transform(x), expected_output)


def test_remap_inverse_transform(input_remapper) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    assert torch.allclose(
        input_remapper.inverse_transform(input_remapper.transform(x, in_place=False), in_place=False), x
    )
