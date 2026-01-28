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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
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


# ============================================================================
# Tests for target-only variables (e.g., satellite observations like 'imerg')
# ============================================================================

@pytest.fixture()
def normalizer_with_target_only():
    """Create normalizer with target-only variable 'imerg'.
    
    Setup:
    - 4 regular variables (x, y, z, q) in both model.output and data.output
    - 1 target-only variable (imerg) only in data.output
    - data.output has 5 variables, model.output has 4
    """
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "normalizer": {"default": "mean-std"},
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],  # target-only variable
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "minimum": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "maximum": np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
    }
    # 5 variables: x, y, z, q are regular; imerg is target-only
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "imerg": 4}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=statistics)


def test_model_output_idx_buffer_created(normalizer_with_target_only) -> None:
    """Test that _model_output_idx buffer is created and has correct size."""
    assert hasattr(normalizer_with_target_only, "_model_output_idx")
    # model.output has 4 variables (excludes imerg)
    assert len(normalizer_with_target_only._model_output_idx) == 4
    # data.output has 5 variables (includes imerg)
    assert len(normalizer_with_target_only._output_idx) == 5


def test_model_output_idx_excludes_target_only(normalizer_with_target_only) -> None:
    """Test that _model_output_idx excludes target-only variables."""
    model_idx = normalizer_with_target_only._model_output_idx
    output_idx = normalizer_with_target_only._output_idx
    
    # imerg is at index 4, should not be in model_output_idx
    assert 4 not in model_idx.tolist()
    # But should be in output_idx
    assert 4 in output_idx.tolist()


def test_inverse_transform_model_output_size(normalizer_with_target_only) -> None:
    """Test inverse_transform with model output size (excludes target-only vars).
    
    This tests the scenario where the model predicts 4 variables but
    data.output has 5 variables (including target-only 'imerg').
    """
    # Normalized tensor with 4 variables (model output, no imerg)
    # Normalized values: (x - mean) / stdev
    normalized = torch.Tensor([
        [0.0, 0.0, 0.0, 0.0],  # All at mean
        [2.0, 2.0, 2.0, 2.0],  # All at mean + stdev
    ])
    
    # After inverse transform: x * stdev + mean = x * 0.5 + [1,2,3,4]
    expected = torch.Tensor([
        [1.0, 2.0, 3.0, 4.0],  # means
        [2.0, 3.0, 4.0, 5.0],  # mean + stdev
    ])
    
    result = normalizer_with_target_only.inverse_transform(normalized, in_place=False)
    assert torch.allclose(result, expected)


def test_inverse_transform_data_output_size(normalizer_with_target_only) -> None:
    """Test inverse_transform with data output size (includes target-only vars).
    
    This tests the scenario where we have the full data.output with 5 variables
    including the target-only 'imerg'.
    """
    # Normalized tensor with 5 variables (data output, includes imerg)
    normalized = torch.Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ])
    
    # After inverse transform
    expected = torch.Tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],  # means
        [2.0, 3.0, 4.0, 5.0, 6.0],  # mean + stdev
    ])
    
    result = normalizer_with_target_only.inverse_transform(normalized, in_place=False)
    assert torch.allclose(result, expected)


def test_inverse_transform_different_tensor_sizes(normalizer_with_target_only) -> None:
    """Test that inverse_transform handles both model and data output sizes correctly."""
    # Model output size (4 variables)
    model_output = torch.zeros(2, 4)
    result_model = normalizer_with_target_only.inverse_transform(model_output, in_place=False)
    assert result_model.shape[-1] == 4
    
    # Data output size (5 variables)
    data_output = torch.zeros(2, 5)
    result_data = normalizer_with_target_only.inverse_transform(data_output, in_place=False)
    assert result_data.shape[-1] == 5
    
    # Values should be correct for each
    # Model output gets indices [0,1,2,3], data output gets indices [0,1,2,3,4]
    assert torch.allclose(result_model, torch.Tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]))
    assert torch.allclose(result_data, torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]))

