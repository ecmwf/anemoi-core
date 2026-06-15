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
from anemoi.models.preprocessing.normalizer import InputNormalizer

VARIABLES = ["x", "y", "z", "q", "other"]

STATISTICS = {
    "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
    "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0]),
    "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
}


def make_gridded_view(payload: torch.Tensor, variables=VARIABLES, statistics=STATISTICS):
    """Wrap a (points, variables) payload in a GriddedSourceView.

    The data tensor is shaped ``(batch, time, grid, variables)`` so that the
    variable axis stays last and normalization parameters broadcast correctly.
    """
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
def input_normalizer():
    config = DictConfig(
        {
            "default": "mean-std",
            "min-max": ["x"],
            "max": ["y"],
            "none": ["z"],
            "mean-std": ["q"],
        },
    )
    return InputNormalizer(config=config)


@pytest.fixture()
def base_payload():
    return torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])


@pytest.fixture()
def normalized_payload():
    return torch.Tensor([[0.0, 0.2, 3.0, -0.5, 1 / 7], [0.5, 0.7, 8.0, 4.5, 0.5]])


def test_validate_unknown_method_raises() -> None:
    config = DictConfig({"default": "mean-std", "not-a-method": ["x"]})
    with pytest.raises(AssertionError):
        InputNormalizer(config=config)


def test_norm_parameters_cache_is_device_aware(input_normalizer) -> None:
    """Cached normalization parameters must be keyed by device, not only by variables.

    Reusing the same variable set across devices (e.g. metrics on GPU, plotting on
    CPU) previously returned parameters built for the first device, causing
    'Expected all tensors to be on the same device' at denormalisation time.
    """
    name_to_index = {name: idx for idx, name in enumerate(VARIABLES)}

    mul_cpu, add_cpu = input_normalizer.get_norm_parameters(STATISTICS, name_to_index, torch.device("cpu"))
    assert mul_cpu.device.type == "cpu"
    assert add_cpu.device.type == "cpu"

    # The cache key must encode the device so a different device cannot reuse these tensors.
    assert (tuple(name_to_index.keys()), "cpu") in input_normalizer._param_cache

    if torch.cuda.is_available():
        mul_gpu, add_gpu = input_normalizer.get_norm_parameters(STATISTICS, name_to_index, torch.device("cuda:0"))
        assert mul_gpu.device.type == "cuda"
        assert add_gpu.device.type == "cuda"
        assert (tuple(name_to_index.keys()), "cuda:0") in input_normalizer._param_cache
        # The original CPU entry is preserved, not clobbered by the GPU call.
        mul_cpu_again, _ = input_normalizer.get_norm_parameters(STATISTICS, name_to_index, torch.device("cpu"))
        assert mul_cpu_again.device.type == "cpu"


def test_normalizer_not_inplace(input_normalizer, make_view, base_payload) -> None:
    view = make_view(base_payload)
    original = view_data_2d(view).clone()
    input_normalizer(view, in_place=False)
    assert torch.allclose(view_data_2d(view), original)


def test_normalizer_inplace(input_normalizer, make_view, base_payload) -> None:
    view = make_view(base_payload)
    original = view_data_2d(view).clone()
    out = input_normalizer(view, in_place=True)
    assert not torch.allclose(view_data_2d(view), original)
    assert torch.allclose(view_data_2d(view), view_data_2d(out))


def test_normalize(input_normalizer, make_view, base_payload, normalized_payload) -> None:
    view = make_view(base_payload)
    out = input_normalizer.transform(view)
    assert torch.allclose(view_data_2d(out), normalized_payload)


def test_inverse_transform(input_normalizer, make_view, base_payload, normalized_payload) -> None:
    view = make_view(normalized_payload)
    out = input_normalizer.inverse_transform(view)
    assert torch.allclose(view_data_2d(out), base_payload)


def test_normalize_inverse_roundtrip(input_normalizer, make_view, base_payload) -> None:
    view = make_view(base_payload)
    transformed = input_normalizer.transform(view, in_place=False)
    restored = input_normalizer.inverse_transform(transformed, in_place=False)
    assert torch.allclose(view_data_2d(restored), base_payload)


def test_std_and_default_methods(make_view) -> None:
    config = DictConfig({"default": "none", "std": ["q"], "mean-std": ["other"]})
    normalizer = InputNormalizer(config=config)
    payload = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    view = make_view(payload)
    out = view_data_2d(normalizer.transform(view))
    # std: q -> data / stdev (stdev=1.0), unchanged
    assert torch.allclose(out[..., 3], payload[..., 3])
    # mean-std: other -> (data - mean) / stdev = (data - 3) / 14
    assert torch.allclose(out[..., 4], (payload[..., 4] - 3.0) / 14.0)
    # none/default: x, y, z unchanged
    assert torch.allclose(out[..., [0, 1, 2]], payload[..., [0, 1, 2]])


def test_near_zero_variance_warns_and_skips(make_view) -> None:
    config = DictConfig({"default": "mean-std"})
    normalizer = InputNormalizer(config=config)
    statistics = {
        "mean": np.array([5.0]),
        "stdev": np.array([0.0]),
        "minimum": np.array([5.0]),
        "maximum": np.array([5.0]),
    }
    payload = torch.Tensor([[5.0], [5.0]])
    view = make_view(payload, variables=["x"], statistics=statistics)
    with pytest.warns(UserWarning, match="near-zero variance"):
        out = normalizer.transform(view)
    assert torch.allclose(view_data_2d(out), payload)


def test_near_zero_range_warns_and_shifts(make_view) -> None:
    config = DictConfig({"default": "min-max"})
    normalizer = InputNormalizer(config=config)
    statistics = {
        "mean": np.array([5.0]),
        "stdev": np.array([1.0]),
        "minimum": np.array([5.0]),
        "maximum": np.array([5.0]),
    }
    payload = torch.Tensor([[5.0], [7.0]])
    view = make_view(payload, variables=["x"], statistics=statistics)
    with pytest.warns(UserWarning, match="near-zero range"):
        out = normalizer.transform(view)
    # norm_mul stays 1, norm_add = -minimum -> data - 5
    assert torch.allclose(view_data_2d(out), payload - 5.0)


def test_data_index_deprecation_warning(input_normalizer, make_view, base_payload) -> None:
    view = make_view(base_payload)
    with pytest.warns(DeprecationWarning, match="data_index"):
        input_normalizer.transform(view, in_place=False, data_index=[0, 1, 2])


def test_parameter_caching(input_normalizer, make_view, base_payload) -> None:
    assert len(input_normalizer._param_cache) == 0
    input_normalizer.transform(make_view(base_payload), in_place=False)
    assert len(input_normalizer._param_cache) == 1
    # second call with the same variable set hits the cache, no new entry
    input_normalizer.transform(make_view(base_payload), in_place=False)
    assert len(input_normalizer._param_cache) == 1
    input_normalizer.reset_cache()
    assert len(input_normalizer._param_cache) == 0


def test_tabular_multiple_tensors(input_normalizer, normalized_payload) -> None:
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    payload_a = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    payload_b = payload_a.clone()
    view = create_source_view(
        name="tabular",
        data=[payload_a, payload_b],
        variables=list(VARIABLES),
        statistics=STATISTICS,
        coordinates=None,
        layout=layout,
        is_static=False,
        boundaries=None,
    )
    out = input_normalizer.transform(view, in_place=False)
    assert len(out.data) == 2
    for tensor in out.data:
        assert torch.allclose(tensor, normalized_payload)
