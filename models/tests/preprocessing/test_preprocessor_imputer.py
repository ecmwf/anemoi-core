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
from anemoi.models.preprocessing.imputer import CopyImputer
from anemoi.models.preprocessing.imputer import InputImputer


@pytest.fixture()
def non_default_input_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {
                    "default": "none",
                    "mean": ["y", "other"],
                    "maximum": ["x"],
                    "none": ["z"],
                    "minimum": ["q"],
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14, 1.0]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
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
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14, 1.0]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return InputImputer(config=config.data.imputer, statistics=statistics, data_indices=data_indices)


@pytest.fixture()
def non_default_input_data():
    # one sample, two time steps, two grid points, 6 variables
    base = torch.Tensor(
        [
            [
                [[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]],
                [[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, 1.0, 8.0, 9.0, np.nan, 1.0]],
            ]
        ]
    )
    expected = torch.Tensor(
        [
            [
                [[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 2.0, 8.0, 9.0, 3.0, 1.0]],
                [[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 1.0, 8.0, 9.0, 3.0, 1.0]],
            ]
        ]
    )
    restored = torch.Tensor(
        [
            [
                [[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 3.0, 1.0]],
                [[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 3.0, 1.0]],
            ]
        ]
    )
    return base, expected, restored


@pytest.fixture()
def default_input_data():
    # one sample, one time step, two grid points, 6 variables
    base = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]]]])
    expected = torch.Tensor([[[[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 1.0, 8.0, 9.0, 1.0, 1.0]]]])
    restored = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 1.0, 1.0]]]])
    return base, expected, restored


@pytest.fixture()
def non_default_constant_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": "none", 0: ["x"], 3.0: ["y", "other"], 22.7: ["z"], 10: ["q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConstantImputer(config=config.data.imputer, statistics=None, data_indices=data_indices)


@pytest.fixture()
def default_constant_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"default": 22.7},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ConstantImputer(config=config.data.imputer, statistics=None, data_indices=data_indices)


@pytest.fixture()
def default_constant_data():
    # one sample, one time step, two grid points, 6 variables
    base = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]]]])
    expected = torch.Tensor([[[[1.0, 2.0, 3.0, 22.7, 5.0, 1.0], [6.0, 22.7, 8.0, 9.0, 22.7, 1.0]]]])
    restored = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 22.7, 1.0]]]])
    return base, expected, restored


@pytest.fixture()
def non_default_constant_data():
    # one sample, one time step, two grid points, 6 variables
    base = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]]]])
    expected = torch.Tensor([[[[1.0, 2.0, 3.0, 10.0, 5.0, 1.0], [6.0, 3.0, 8.0, 9.0, 3.0, 1.0]]]])
    restored = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 3.0, 1.0]]]])
    return base, expected, restored


@pytest.fixture()
def copy_imputer():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {"x": ["y", "other", "q"]},
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return CopyImputer(config=config.data.imputer, statistics=None, data_indices=data_indices)


@pytest.fixture()
def copy_data():
    # one sample, one time step, two grid points, 6 variables
    base = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]]]])
    expected = torch.Tensor([[[[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 6.0, 8.0, 9.0, 6.0, 1.0]]]])
    restored = torch.Tensor([[[[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, 6.0, 1.0]]]])
    return base, expected, restored


fixture_combinations = (
    ("default_constant_imputer", "default_constant_data"),
    ("non_default_constant_imputer", "non_default_constant_data"),
    ("default_input_imputer", "default_input_data"),
    ("non_default_input_imputer", "non_default_input_data"),
    ("copy_imputer", "copy_data"),
)


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_imputer_not_inplace(imputer_fixture, data_fixture, request) -> None:
    """Check that the imputer does not modify the input tensor when in_place=False."""
    x, _, _ = request.getfixturevalue(data_fixture)
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
    x, _, _ = request.getfixturevalue(data_fixture)
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
    x, expected, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_transform_with_nan_inference(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs in inference."""
    x, expected, expected_restored = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    # transform on training data to set nan mask
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    # Split data to "inference size" removing "diagnostics"
    x_small_in = x[..., imputer.data_indices.data.input.full]
    x_small_out = expected_restored[..., imputer.data_indices.data.output.full]
    expected_small_in = expected[..., imputer.data_indices.data.input.full]
    expected_small_out = expected[..., imputer.data_indices.data.output.full]
    # transform on inference data
    transformed_small = imputer.transform(x_small_in, in_place=False)
    assert torch.allclose(
        transformed_small,
        expected_small_in,
        equal_nan=True,
    ), "Transform (in inference) does not handle NaNs correctly."
    # inverse transform on inference data
    imputer.transform(x_small_in, in_place=False)
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
    x, expected, _ = request.getfixturevalue(data_fixture)
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
    x, expected, expected_restored = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    restored = imputer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(
        restored, expected_restored, equal_nan=True
    ), "Inverse transform does not restore NaNs correctly."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_mask_saving(imputer_fixture, data_fixture, request):
    """Check that the imputer saves the NaN mask correctly."""
    x, _, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    # reduce time dimension
    expected_mask = torch.isnan(x)[:, 0][..., imputer.data_indices.data.input.full]
    imputer.transform(x)
    assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after first run."


@pytest.mark.parametrize(
    ("imputer_fixture", "data_fixture"),
    fixture_combinations,
)
def test_loss_nan_mask(imputer_fixture, data_fixture, request):
    """Check that the imputer correctly transforms a tensor with NaNs."""
    x, _, _ = request.getfixturevalue(data_fixture)
    expected = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]])  # only prognostic and diagnostic variables
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
    x, expected, _ = request.getfixturevalue(data_fixture)
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
def test_changing_nan_locations(imputer_fixture, data_fixture, request):
    """Check that the imputer resets its mask during inference."""
    x, expected, expected_restored = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)

    # reduce time dimension
    expected_mask = torch.isnan(x)[:, 0][..., imputer.data_indices.data.input.full]
    transformed = imputer.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True), "Transform does not handle NaNs correctly."
    restored = imputer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(
        restored, expected_restored, equal_nan=True
    ), "Inverse transform does not restore NaNs correctly."
    assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after first run."

    # change nan locations by rolling the tensor
    x = x.roll(1, dims=0)
    expected = expected.roll(1, dims=0)
    expected_restored = expected_restored.roll(1, dims=0)
    # reduce time dimension
    expected_mask = torch.isnan(x)[:, 0][..., imputer.data_indices.data.input.full]
    imputer.transform(x, in_place=False)
    assert torch.allclose(
        imputer.transform(x, in_place=False), expected, equal_nan=True
    ), "Transform does not handle changed NaNs correctly."
    restored = imputer.inverse_transform(imputer.transform(x, in_place=False), in_place=False)
    assert torch.allclose(
        restored, expected_restored, equal_nan=True
    ), "Inverse transform does not restore changed NaNs correctly."
    assert torch.equal(imputer.nan_locations, expected_mask), "Mask not saved correctly after changing nan locations."
