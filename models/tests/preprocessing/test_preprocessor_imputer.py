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

from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.data.views import create_source_view
from anemoi.models.preprocessing.imputer import ConstantImputer
from anemoi.models.preprocessing.imputer import CopyImputer
from anemoi.models.preprocessing.imputer import InputImputer

VARIABLES = ["x", "y", "z", "q", "other", "prog"]

STATISTICS = {
    "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
    "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0, 1.0]),
    "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
}


def make_gridded_view(payload: torch.Tensor, variables=VARIABLES, statistics=STATISTICS):
    """Wrap a (points, variables) payload in a GriddedSourceView."""
    points, num_vars = payload.shape
    data = payload.reshape(1, 1, points, num_vars).clone()
    layout = TensorLayout(batch=0, time=1, grid=2, variables=3)
    return create_source_view(
        name="gridded",
        data=data,
        variables=list(variables),
        statistics=statistics,
        coordinates=None,
        layout=layout,
        is_static=True,
    )


def make_tabular_view(payload: torch.Tensor, variables=VARIABLES, statistics=STATISTICS):
    """Wrap a (points, variables) payload in a TabularSourceView (single tensor)."""
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    return create_source_view(
        name="tabular",
        data=[payload.clone()],
        variables=list(variables),
        statistics=statistics,
        coordinates=None,
        layout=layout,
        is_static=False,
        boundaries=None,
    )


def view_data_2d(view) -> torch.Tensor:
    """Extract the (points, variables) payload back out of either view kind."""
    if isinstance(view.data, list):
        return view.data[0]
    points = view.data.shape[view.layout.grid]
    num_vars = view.data.shape[view.layout.variables]
    return view.data.reshape(points, num_vars)


@pytest.fixture(params=["gridded", "tabular"])
def make_view(request):
    return make_gridded_view if request.param == "gridded" else make_tabular_view


@pytest.fixture()
def non_default_input_imputer():
    config = DictConfig(
        {
            "default": "none",
            "mean": ["y", "other"],
            "maximum": ["x"],
            "none": ["z"],
            "minimum": ["q"],
        },
    )
    return InputImputer(config=config)


@pytest.fixture()
def default_input_imputer():
    config = DictConfig({"default": "minimum"})
    return InputImputer(config=config)


@pytest.fixture()
def non_default_constant_imputer():
    config = DictConfig({"default": "none", 0: ["x"], 3.0: ["y", "other"], 22.7: ["z"], 10: ["q"]})
    return ConstantImputer(config=config)


@pytest.fixture()
def default_constant_imputer():
    config = DictConfig({"default": 22.7})
    return ConstantImputer(config=config)


@pytest.fixture()
def copy_imputer():
    config = DictConfig({"x": ["y", "other", "q"]})
    return CopyImputer(config=config)


@pytest.fixture()
def non_default_input_data():
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 2.0, 8.0, 9.0, 3.0, 1.0]])
    return base, expected


@pytest.fixture()
def default_input_data():
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 1.0, 8.0, 9.0, 1.0, 1.0]])
    return base, expected


@pytest.fixture()
def non_default_constant_data():
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 10.0, 5.0, 1.0], [6.0, 3.0, 8.0, 9.0, 3.0, 1.0]])
    return base, expected


@pytest.fixture()
def default_constant_data():
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 22.7, 5.0, 1.0], [6.0, 22.7, 8.0, 9.0, 22.7, 1.0]])
    return base, expected


@pytest.fixture()
def copy_data():
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 5.0, 1.0], [6.0, 6.0, 8.0, 9.0, 6.0, 1.0]])
    return base, expected


fixture_combinations = (
    ("default_constant_imputer", "default_constant_data"),
    ("non_default_constant_imputer", "non_default_constant_data"),
    ("default_input_imputer", "default_input_data"),
    ("non_default_input_imputer", "non_default_input_data"),
    ("copy_imputer", "copy_data"),
)


@pytest.mark.parametrize(("imputer_fixture", "data_fixture"), fixture_combinations)
def test_imputer_not_inplace(imputer_fixture, data_fixture, make_view, request) -> None:
    """The imputer must not modify the input view when in_place=False."""
    base, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    view = make_view(base)
    original = view_data_2d(view).clone()
    imputer(view, in_place=False)
    assert torch.allclose(view_data_2d(view), original, equal_nan=True)


@pytest.mark.parametrize(("imputer_fixture", "data_fixture"), fixture_combinations)
def test_imputer_inplace(imputer_fixture, data_fixture, make_view, request) -> None:
    """The imputer must modify the input view when in_place=True."""
    base, _ = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    view = make_view(base)
    original = view_data_2d(view).clone()
    out = imputer(view, in_place=True)
    assert not torch.allclose(view_data_2d(view), original, equal_nan=True)
    assert torch.allclose(view_data_2d(view), view_data_2d(out), equal_nan=True)


@pytest.mark.parametrize(("imputer_fixture", "data_fixture"), fixture_combinations)
def test_transform_with_nan(imputer_fixture, data_fixture, make_view, request) -> None:
    """The imputer replaces NaNs with the configured values."""
    base, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    out = imputer.transform(make_view(base))
    assert torch.allclose(view_data_2d(out), expected, equal_nan=True)


@pytest.mark.parametrize(("imputer_fixture", "data_fixture"), fixture_combinations)
def test_transform_noop_without_nan(imputer_fixture, data_fixture, make_view, request) -> None:
    """Transforming NaN-free data leaves it unchanged."""
    _, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    out = imputer.transform(make_view(expected))
    assert torch.allclose(view_data_2d(out), expected, equal_nan=True)


@pytest.mark.parametrize(("imputer_fixture", "data_fixture"), fixture_combinations)
def test_inverse_transform_is_noop(imputer_fixture, data_fixture, make_view, request) -> None:
    """inverse_transform is a no-op: filled values are returned unchanged (NaNs not restored)."""
    base, expected = request.getfixturevalue(data_fixture)
    imputer = request.getfixturevalue(imputer_fixture)
    transformed = imputer.transform(make_view(base), in_place=False)
    restored = imputer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(view_data_2d(restored), expected, equal_nan=True)
    assert not torch.isnan(view_data_2d(restored)).any()


def test_copy_imputer_raises_on_nan_source(copy_imputer, make_view) -> None:
    """CopyImputer must fail if the source variable is NaN where the target needs imputation."""
    base = torch.tensor([[np.nan, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, 7.0, 8.0, 9.0, 10.0, 1.0]])
    with pytest.raises(AssertionError):
        copy_imputer.transform(make_view(base))


def test_input_imputer_uses_view_statistics(non_default_input_imputer, make_view) -> None:
    """Replacement values are read from the SourceView statistics, not the constructor."""
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0]])
    custom_statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0, 1.0]),
        "minimum": np.array([1.0, 1.0, 1.0, 42.0, 1.0, 0.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
    }
    out = non_default_input_imputer.transform(make_view(base, statistics=custom_statistics))
    # q uses the "minimum" statistic -> 42.0
    assert view_data_2d(out)[0, 3] == pytest.approx(42.0)


def test_tabular_multiple_tensors(default_constant_imputer) -> None:
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    base = torch.tensor([[1.0, 2.0, 3.0, np.nan, 5.0, 1.0], [6.0, np.nan, 8.0, 9.0, np.nan, 1.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 22.7, 5.0, 1.0], [6.0, 22.7, 8.0, 9.0, 22.7, 1.0]])
    view = create_source_view(
        name="tabular",
        data=[base.clone(), base.clone()],
        variables=list(VARIABLES),
        statistics=STATISTICS,
        coordinates=None,
        layout=layout,
        is_static=False,
        boundaries=None,
    )
    out = default_constant_imputer.transform(view, in_place=False)
    assert len(out.data) == 2
    for tensor in out.data:
        assert torch.allclose(tensor, expected, equal_nan=True)
