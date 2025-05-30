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
from anemoi.models.preprocessing.postprocessor import ConditionalZeroPostprocessor
from anemoi.models.preprocessing.postprocessor import CustomReluPostprocessor
from anemoi.models.preprocessing.postprocessor import Postprocessor


@pytest.fixture()
def postprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "postprocessor": {"default": "none", "relu": ["q"], "hardtanh": ["x"]},
                "forcing": ["z"],
                "diagnostic": ["other"],
                "remapped": {},
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return Postprocessor(config=config.data.postprocessor, data_indices=data_indices)


@pytest.fixture()
def output_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [-1.0, 1, 8.0, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def inference_output_data():
    base = torch.Tensor([[1.0, 2.0, -1, 5.0], [-2, 1, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 0.0, 5.0], [-1, 1, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def customrelupostprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "customrelupostprocessor": {"default": "none", 0: ["q"], -1.5: ["x"]},
                "forcing": ["z"],
                "diagnostic": ["other"],
                "remapped": {},
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return CustomReluPostprocessor(config=config.data.customrelupostprocessor, data_indices=data_indices)


@pytest.fixture()
def customrelu_output_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [-1.5, 1, 8.0, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def customrelu_inference_output_data():
    base = torch.Tensor([[1.0, 2.0, -1, 5.0], [-2, 1, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 0.0, 5.0], [-1, 1, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def conditionalzeropostprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "conditionalzeropostprocessor": {"default": "none", 0: ["q"], -1.5: ["x"], "remap": "y"},
                "forcing": ["z"],
                "diagnostic": ["other"],
                "remapped": {},
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConditionalZeroPostprocessor(config=config.data.conditionalzeropostprocessor, data_indices=data_indices)


@pytest.fixture()
def conditionalzero_output_data():
    base = torch.Tensor([[[1.0, 0.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]]])
    expected = torch.Tensor([[[-1.5, 0.0, 3.0, 0.0, 5.0], [-2, 1, 8.0, 9.0, 10.0]]])
    return base, expected


@pytest.fixture()
def conditionalzero_inference_output_data():
    base = torch.Tensor([[[1.0, 0.0, -1, 5.0], [-2, 1, 9.0, 10.0]]])
    expected = torch.Tensor([[[-1.5, 0.0, 0.0, 5.0], [-2, 1, 9.0, 10.0]]])
    return base, expected


fixture_combinations = (
    ("postprocessor", "output_data"),
    ("postprocessor", "inference_output_data"),
    ("customrelupostprocessor", "customrelu_output_data"),
    ("customrelupostprocessor", "customrelu_inference_output_data"),
    ("conditionalzeropostprocessor", "conditionalzero_output_data"),
    ("conditionalzeropostprocessor", "conditionalzero_inference_output_data"),
)


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_postprocessor_not_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _ = request.getfixturevalue(data_fixture)
    postprocessor = request.getfixturevalue(postprocessor_fixture)
    x_old = x.clone()
    postprocessor.inverse_transform(x, in_place=False)
    assert torch.allclose(x, x_old, equal_nan=True), "Postprocessor does not handle in_place=False correctly."


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_postprocessor_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _ = request.getfixturevalue(data_fixture)
    postprocessor = request.getfixturevalue(postprocessor_fixture)
    x_old = x.clone()
    out = postprocessor.inverse_transform(x, in_place=True)
    assert not torch.allclose(x, x_old, equal_nan=True)
    assert torch.allclose(x, out, equal_nan=True)
