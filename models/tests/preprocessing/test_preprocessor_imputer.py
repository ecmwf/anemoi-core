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
from anemoi.models.preprocessing.imputer import ConstantImputer
from anemoi.models.preprocessing.imputer import InputImputer


@pytest.fixture()
def non_default_input_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": "none", "mean": ["y"], "maximum": ["x"], "none": ["z"], "minimum": ["q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": {},
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
    return InputImputer(config=config.data.imputer, data_indices=data_indices, statistics=statistics)


@pytest.fixture()
def default_input_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": "minimum"},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": [],
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
    return InputImputer(config=config.data.imputer, statistics=statistics, data_indices=data_indices)


@pytest.fixture()
def non_default_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, np.nan, 5.0], [6.0, np.nan, 8.0, 9.0, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 1.0, 5.0], [6.0, 2.0, 8.0, 9.0, 10.0]])
    return base, expected


@pytest.fixture()
def default_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, np.nan, 5.0], [6.0, np.nan, 8.0, 9.0, 0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 1.0, 5.0], [6.0, 1.0, 8.0, 9.0, 0]])
    return base, expected


@pytest.fixture()
def default_constant_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": "none", 0: ["x"], 3.0: ["y"], 22.7: ["z"], 10: ["q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": [],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConstantImputer(config=config.data.imputer, statistics=None, data_indices=data_indices)


@pytest.fixture()
def non_default_constant_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": 22.7},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "remapped": [],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConstantImputer(config=config.data.imputer, statistics=None, data_indices=data_indices)


@pytest.fixture()
def non_default_constant_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, np.nan, 5.0], [6.0, np.nan, 8.0, 9.0, 0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 22.7, 5.0], [6.0, 22.7, 8.0, 9.0, 0]])
    return base, expected


@pytest.fixture()
def default_constant_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, np.nan, 5.0], [6.0, np.nan, 8.0, 9.0, 0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 10, 5.0], [6.0, 3.0, 8.0, 9.0, 0]])
    return base, expected


fixture_combinations = (
    ("default_constant_imputer", "default_constant_data"),
    ("non_default_constant_imputer", "non_default_constant_data"),
    ("default_input_imputer", "default_input_data"),
    ("non_default_input_imputer", "non_default_input_data"),
)


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_imputer_not_inplace(imputer_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    x_old = x.clone()
    imputer(x, in_place=False)
    assert torch.allclose(x, x_old, equal_nan=True), "Imputer does not handle in_place=False correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_imputer_inplace(imputer_fixture, data_fixture, request) -> None:
    """Check that the imputer modifies the input tensor when in_place=True."""
    x, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    x_old = x.clone()
    out = imputer(x, in_place=True)
    assert not torch.allclose(x, x_old, equal_nan=True)
    assert torch.allclose(x, out, equal_nan=True)


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_transform_with_nan(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_transform_with_nan_small(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    x_small = x[..., [0, 1, 2, 3]]
    expected_small = expected[..., [0, 1, 2, 3]]
    transformed_small = imputer.transform(x_small, in_place=False)
    assert torch.allclose(
        transformed_small,
        expected_small,
        equal_nan=True,
    ), "Transform (in inference) does not handle NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_transform_with_nan_inference(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs in inference."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    # Split data to "inference size" removing "diagnostics"
    x_small_in = x[..., [0, 1, 2, 3]]
    x_small_out = x[..., [0, 1, 4]]
    expected_small_in = expected[..., [0, 1, 2, 3]]
    expected_small_out = expected[..., [0, 1, 4]]
    transformed_small = imputer.transform(x_small_in, in_place=False)
    assert torch.allclose(
        transformed_small,
        expected_small_in,
        equal_nan=True,
    ), "Transform (in inference) does not handle NaNs correctly."
    # Check that the inverse also performs correctly
    restored = imputer.inverse_transform(expected_small_out, in_place=False)
    assert torch.allclose(
        restored, x_small_out, equal_nan=True
    ), "Inverse transform does not restore NaNs correctly in inference."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_transform_noop(imputer_fixture, data_fixture, request):
    """Check that the imputer does not modify a tensor without NaNs."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    _ = imputer.transform(x)
    transformed = imputer.transform(expected)
    assert torch.allclose(transformed, expected), "Transform does not handle NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_inverse_transform(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly inverts the transformation."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    restored = imputer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(restored, x, equal_nan=True), "Inverse transform does not restore NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_mask_saving(imputer_fixture, data_fixture, request):
    """Check that the imputer saves the NaN mask correctly."""
    x, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    expected_mask = torch.isnan(x)
    imputer.transform(x)
    assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after first run."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_loss_nan_mask(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs."""
    x, _ = request.getfixturevalue(data_fixture)
    expected = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])  # only prognostic and diagnostic variables
    imputer = request.getfixturevalue(imputer_fixture)
    imputer.transform(x)
    assert torch.allclose(
        imputer.loss_mask_training, expected
    ), "Transform does not calculate NaN-mask for loss function scaling correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    [
        ("default_constant_imputer", "default_constant_data"),
        ("non_default_constant_imputer", "non_default_constant_data"),
        ("default_input_imputer", "default_input_data"),
        ("non_default_input_imputer", "non_default_input_data"),
    ],
)
def test_reuse_imputer(imputer_fixture, data_fixture, request):
    """Check that the imputer reuses the mask correctly on subsequent runs."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    x2 = x**2.0
    _ = imputer.transform(x2, in_place=False)
    transformed2 = imputer.transform(x, in_place=False)
    assert torch.allclose(
        transformed2, expected, equal_nan=True
    ), "Imputer does not reuse mask correctly on subsequent runs."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_inference_imputer(imputer_fixture, data_fixture, request):
    """Check that the imputer resets its mask during inference."""
    x, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)

    # Check training flag
    assert imputer.training, "Imputer is not set to training mode."

    expected_mask = torch.isnan(x)
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    restored = imputer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(restored, x, equal_nan=True), "Inverse transform does not restore NaNs correctly."
    assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after first run."

    imputer.eval()
    with torch.no_grad():
        x2 = x.roll(-1, dims=0)
        expected2 = expected.roll(-1, dims=0)
        expected_mask2 = torch.isnan(x2)

        assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after first run."

        # Check training flag
        assert not imputer.training, "Imputer is not set to evaluation mode."

        assert not torch.allclose(x, x2, equal_nan=True), "Failed to modify the input data."
        assert not torch.allclose(expected, expected2, equal_nan=True), "Failed to modify the expected data."
        assert not torch.allclose(expected_mask, expected_mask2, equal_nan=True), "Failed to modify the nan mask."

        transformed = imputer.transform(x2, in_place=False)
        assert torch.allclose(transformed, expected2, equal_nan=True), "Transform does not handle NaNs correctly."
        restored = imputer.inverse_transform(transformed, in_place=False)
        assert torch.allclose(restored, x2, equal_nan=True), "Inverse transform does not restore NaNs correctly."

        assert torch.equal(imputer.nan_locations, expected_mask2), "Mask not saved correctly after evaluation run."
