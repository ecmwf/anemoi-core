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
from anemoi.models.preprocessing.postprocessor import ConditionalNaNPostprocessor
from anemoi.models.preprocessing.postprocessor import ConditionalZeroPostprocessor
from anemoi.models.preprocessing.postprocessor import NormalizedReluPostprocessor
from anemoi.models.preprocessing.postprocessor import Postprocessor


@pytest.fixture()
def postprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "postprocessor": {"default": "none", "relu": ["q"], "hardtanh": ["x"], "hardtanh_0_1": ["y"]},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return Postprocessor(config=config.data.postprocessor, data_indices=data_indices)


@pytest.fixture()
def output_data():
    base = torch.Tensor([[1.0, -2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 0.0, 3.0, 0.0, 5.0], [-1.0, 1, 8.0, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def inference_output_data():
    base = torch.Tensor([[1.0, -2.0, -1, 5.0], [-2, 1, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 0.0, 0.0, 5.0], [-1, 1, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def normmrelupostprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normmrelupostprocessor": {"default": "none", 0: ["q"], -1.5: ["x"], "normalizer": "none"},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)

    return NormalizedReluPostprocessor(
        config=config.data.normmrelupostprocessor,
        data_indices=data_indices,
        statistics=statistics,
    )


@pytest.fixture()
def normmrelupostprocessor_output_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [-1.5, 1, 8.0, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def normmrelupostprocessor_inference_output_data():
    base = torch.Tensor([[1.0, 2.0, -1, 5.0], [-2, 1, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 0.0, 5.0], [-1.5, 1, 9.0, 10.0]])
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


@pytest.fixture()
def conditionalnanpostprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "conditionalnanpostprocessor": {"default": "none", "nan": ["other", "y"], "remap": "x"},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConditionalNaNPostprocessor(config=config.data.conditionalnanpostprocessor, data_indices=data_indices)


@pytest.fixture()
def conditionalnan_output_data():
    base = torch.Tensor([[[1.0, 0.0, 3.0, -1, 5.0], [torch.nan, 1, 8.0, 9.0, 10.0]]])
    expected = torch.Tensor([[[1.0, 0.0, 3.0, -1, 5.0], [torch.nan, torch.nan, 8.0, 9.0, torch.nan]]])
    return base, expected


@pytest.fixture()
def conditionalnan_inference_output_data():
    base = torch.Tensor([[[1.0, 0.0, -1, 5.0], [torch.nan, 1, 9.0, 10.0]]])
    expected = torch.Tensor([[[1.0, 0.0, -1, 5.0], [torch.nan, torch.nan, 9.0, torch.nan]]])
    return base, expected


fixture_combinations = (
    ("postprocessor", "output_data"),
    ("postprocessor", "inference_output_data"),
    ("normmrelupostprocessor", "normmrelupostprocessor_output_data"),
    ("normmrelupostprocessor", "normmrelupostprocessor_inference_output_data"),
    ("conditionalzeropostprocessor", "conditionalzero_output_data"),
    ("conditionalzeropostprocessor", "conditionalzero_inference_output_data"),
    ("conditionalnanpostprocessor", "conditionalnan_output_data"),
    ("conditionalnanpostprocessor", "conditionalnan_inference_output_data"),
)


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_postprocessor_not_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the postprocessor does not modify the input tensor when in_place=False."""
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
    """Check that the postprocessor does not modify the input tensor when in_place=False and whether output is correct."""
    x, x_processed = request.getfixturevalue(data_fixture)
    postprocessor = request.getfixturevalue(postprocessor_fixture)
    x_old = x.clone()
    out = postprocessor.inverse_transform(x, in_place=True)
    assert not torch.allclose(x, x_old, equal_nan=True), "Postprocessor does not handle in_place=True correctly."
    assert torch.allclose(x, out, equal_nan=True), "Postprocessor does not handle in_place=True correctly."
    assert torch.allclose(x_processed, out, equal_nan=True), "Postprocessor produces wrong outputs."


@pytest.fixture()
def input_normalizer_postprocessor():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normalizer": {
                    "default": "mean-std",
                    "min-max": ["x"],
                    "max": ["y"],
                    "none": ["z", "other"],
                    "mean-std": ["q"],
                },
                "normmrelupostprocessor_ms": {"default": "none", 0: ["q"], "normalizer": "mean-std"},
                "normmrelupostprocessor_mm": {"default": "none", -1.5: ["x"], "normalizer": "min-max"},
                "forcing": ["z"],
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
    return (
        InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=statistics),
        NormalizedReluPostprocessor(
            config=config.data.normmrelupostprocessor_ms,
            data_indices=data_indices,
            statistics=statistics,
        ),
        NormalizedReluPostprocessor(
            config=config.data.normmrelupostprocessor_mm,
            data_indices=data_indices,
            statistics=statistics,
        ),
    )


@pytest.fixture()
def chained_processors_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    base_normalized = torch.Tensor([[0.0, 0.2, 3.0, -5.5, 5.0], [-0.3, 0.1, 8.0, 4.5, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [-1.5, 1, 8.0, 9.0, 10.0]])
    return base, base_normalized, expected


@pytest.fixture()
def chained_processors_inference_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    base_normalized = torch.Tensor([[0.0, 0.2, -5.5, 5.0], [-0.3, 0.1, 4.5, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 0.0, 5.0], [-1.5, 1, 9.0, 10.0]])
    return base, base_normalized, expected


fixture_combinations = (
    ("input_normalizer_postprocessor", "chained_processors_input_data"),
    ("input_normalizer_postprocessor", "chained_processors_inference_input_data"),
)


@pytest.mark.parametrize(
    ("postprocessor_fixture", "data_fixture"),
    fixture_combinations,
)
def test_chained_postprocessor_inplace(postprocessor_fixture, data_fixture, request) -> None:
    """Check that the postprocessor does not modify the input tensor when in_place=False."""
    x, x_norm, out = request.getfixturevalue(data_fixture)
    postprocessors = request.getfixturevalue(postprocessor_fixture)
    for postprocessor in postprocessors:
        x = postprocessor.transform(x, in_place=False)
    # replace with normalized tensor in correct size
    x = x_norm.clone()
    for postprocessor in postprocessors[::-1]:
        x = postprocessor.inverse_transform(x, in_place=False)
    assert torch.allclose(x, out, equal_nan=True)
