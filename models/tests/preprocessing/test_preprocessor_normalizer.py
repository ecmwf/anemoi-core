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
from anemoi.models.preprocessing.normalizer import InputNormalizer


@pytest.fixture()
def input_normalizer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normalizer": {"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"], "mean-std": ["q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=statistics)


@pytest.fixture()
def remap_normalizer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normalizer": {
                    "default": "mean-std",
                    "remap": {"x": "z", "y": "x"},
                    "min-max": ["x"],
                    "max": ["y"],
                    "none": ["z"],
                    "mean-std": ["q"],
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=statistics)


def test_normalizer_not_inplace(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    input_normalizer(x, in_place=False)
    assert torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))


def test_normalizer_inplace(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    out = input_normalizer(x, in_place=True)
    assert not torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))
    assert torch.allclose(x, out)


def test_normalize(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.0, 0.2, 3.0, -0.5, 1 / 7], [0.5, 0.7, 8.0, 4.5, 0.5]])
    assert torch.allclose(input_normalizer.transform(x), expected_output)


def test_normalize_small(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.0, 0.2, 3.0, -0.5], [0.5, 0.7, 8.0, 4.5]])
    assert torch.allclose(
        input_normalizer.transform(x[..., [0, 1, 2, 3]], data_index=[0, 1, 2, 3], in_place=False),
        expected_output,
    )
    assert torch.allclose(input_normalizer.transform(x[..., [0, 1, 2, 3]]), expected_output)


def test_inverse_transform_small(input_normalizer) -> None:
    expected_output = torch.Tensor([[1.0, 2.0, 5.0], [6.0, 7.0, 10.0]])
    x = torch.Tensor([[0.0, 0.2, 1 / 7], [0.5, 0.7, 0.5]])
    assert torch.allclose(input_normalizer.inverse_transform(x, data_index=[0, 1, 4], in_place=False), expected_output)
    assert torch.allclose(input_normalizer.inverse_transform(x), expected_output)


def test_inverse_transform(input_normalizer) -> None:
    x = torch.Tensor([[0.0, 0.2, 3.0, -0.5, 1 / 7], [0.5, 0.7, 8.0, 4.5, 0.5]])
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.allclose(input_normalizer.inverse_transform(x), expected_output)


def test_normalize_inverse_transform(input_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.allclose(
        input_normalizer.inverse_transform(input_normalizer.transform(x, in_place=False), in_place=False), x
    )


def test_normalizer_not_inplace_remap(remap_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    remap_normalizer(x, in_place=False)
    assert torch.allclose(x, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]))


def test_normalize_remap(remap_normalizer) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    expected_output = torch.Tensor([[0.0, 2 / 11, 3.0, -0.5, 1 / 7], [5 / 9, 7 / 11, 8.0, 4.5, 0.5]])
    assert torch.allclose(remap_normalizer.transform(x), expected_output)
