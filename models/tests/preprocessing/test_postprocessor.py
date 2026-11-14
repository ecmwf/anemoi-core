# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import io

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.activations import CustomRelu
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.normalization import NORMALIZATION_STATISTICS
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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)

    # normalizer "none": the thresholds are used as given, so no statistics are needed.
    return NormalizedReluPostprocessor(config=config.data.normmrelupostprocessor, data_indices=data_indices)


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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
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
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
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


CHAINED_VARIABLES = ["x", "y", "z", "q", "other"]

CHAINED_STATISTICS = {
    "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
    "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
    "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
}


def make_gridded_view(payload: torch.Tensor, indices: list[int]) -> GriddedSource:
    """Wrap a (points, variables) payload holding the ``indices`` subset of CHAINED_VARIABLES in a GriddedSource."""
    points, num_vars = payload.shape
    return GriddedSource(
        name="gridded",
        data=payload.reshape(1, 1, points, num_vars).clone(),
        variables=[CHAINED_VARIABLES[i] for i in indices],
        statistics={key: value[indices] for key, value in CHAINED_STATISTICS.items()},
        coordinates=torch.zeros(points, 2),
        layout=TensorLayout(batch=0, time=1, grid=2, variables=3),
    )


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
    name_to_index = {name: idx for idx, name in enumerate(CHAINED_VARIABLES)}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return (
        InputNormalizer(config=config.data.normalizer),
        NormalizedReluPostprocessor(
            config=config.data.normmrelupostprocessor_ms,
            data_indices=data_indices,
        ),
        NormalizedReluPostprocessor(
            config=config.data.normmrelupostprocessor_mm,
            data_indices=data_indices,
        ),
    )


@pytest.fixture()
def chained_processors_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    base_normalized = torch.Tensor([[0.0, 0.2, 3.0, -5.5, 5.0], [-0.3, 0.1, 8.0, 4.5, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 3.0, 0.0, 5.0], [-1.5, 1, 8.0, 9.0, 10.0]])
    return base, base_normalized, expected, [0, 1, 2, 3, 4]


@pytest.fixture()
def chained_processors_inference_input_data():
    base = torch.Tensor([[1.0, 2.0, 3.0, -1, 5.0], [-2, 1, 8.0, 9.0, 10.0]])
    base_normalized = torch.Tensor([[0.0, 0.2, -5.5, 5.0], [-0.3, 0.1, 4.5, 10.0]])
    expected = torch.Tensor([[1.0, 2.0, 0.0, 5.0], [-1.5, 1, 9.0, 10.0]])
    # inference output excludes the forcing "z"
    return base, base_normalized, expected, [0, 1, 3, 4]


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
    x, x_norm, out, output_indices = request.getfixturevalue(data_fixture)
    postprocessors = request.getfixturevalue(postprocessor_fixture)
    view = make_gridded_view(x, list(range(len(CHAINED_VARIABLES))))
    for postprocessor in postprocessors:
        view = postprocessor(view, in_place=False)
    # replace with normalized tensor in correct size
    view = make_gridded_view(x_norm, output_indices)
    for postprocessor in postprocessors[::-1]:
        view = postprocessor(view, in_place=False, inverse=True)
    assert torch.allclose(view.data.reshape(out.shape), out, equal_nan=True)


def _mean_std_relu_on_q() -> NormalizedReluPostprocessor:
    """A NormalizedReluPostprocessor clipping q at 0 (physical units) under mean-std normalization."""
    config = DictConfig(
        {
            "data": {
                "relu": {"default": "none", 0: ["q"], "normalizer": "mean-std"},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {name: idx for idx, name in enumerate(CHAINED_VARIABLES)}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return NormalizedReluPostprocessor(config=config.data.relu, data_indices=data_indices)


def test_normalized_relu_thresholds_follow_view_statistics() -> None:
    """The threshold is normalized with the statistics of the view, not fixed at construction."""
    postprocessor = _mean_std_relu_on_q()
    q = CHAINED_VARIABLES.index("q")
    payload = torch.zeros(2, len(CHAINED_VARIABLES))
    payload[:, q] = torch.tensor([-5.0, -1.0])

    # CHAINED_STATISTICS: q has mean 4.5, stdev 1 -> threshold (0 - 4.5) / 1 = -4.5.
    out = postprocessor(make_gridded_view(payload, list(range(len(CHAINED_VARIABLES)))), inverse=True, in_place=False)
    assert torch.equal(out.data[0, 0, :, q], torch.tensor([-4.5, -1.0]))

    # Same processor, a view whose q has mean 0, stdev 2 -> threshold 0.
    view = make_gridded_view(payload, list(range(len(CHAINED_VARIABLES))))
    statistics = {key: value.copy() for key, value in view.statistics.items()}
    statistics["mean"][q], statistics["stdev"][q] = 0.0, 2.0
    out = postprocessor(view.clone(statistics=statistics), inverse=True, in_place=False)
    assert torch.equal(out.data[0, 0, :, q], torch.tensor([0.0, 0.0]))


def test_normalized_relu_requires_matching_statistics() -> None:
    postprocessor = _mean_std_relu_on_q()
    x = torch.zeros(2, len(CHAINED_VARIABLES))
    with pytest.raises(ValueError, match="needs the view's statistics"):
        postprocessor.inverse_transform(x)
    with pytest.raises(ValueError, match="needs the statistics"):
        postprocessor.inverse_transform(x, statistics={"mean": CHAINED_STATISTICS["mean"]})
    with pytest.raises(ValueError, match="has 4 values, but the tensor has 5 variables"):
        postprocessor.inverse_transform(x, statistics={key: value[:4] for key, value in CHAINED_STATISTICS.items()})


def test_normalized_relu_loads_pickle_with_frozen_thresholds() -> None:
    """Checkpoints pickled before the per-call normalization keep working.

    Those instances carry ``statistics`` and already-normalized ``CustomRelu`` objects in
    ``postprocessorfunctions``; the thresholds must come from the config instead.
    """
    postprocessor = _mean_std_relu_on_q()
    # Recreate the state such an instance had: frozen data-space statistics, normalized ReLUs.
    postprocessor.statistics = CHAINED_STATISTICS
    postprocessor.postprocessorfunctions = [CustomRelu(-4.5)]

    buffer = io.BytesIO()
    torch.save(postprocessor, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    q = CHAINED_VARIABLES.index("q")
    payload = torch.zeros(2, len(CHAINED_VARIABLES))
    payload[:, q] = torch.tensor([-5.0, -1.0])
    out = loaded(make_gridded_view(payload, list(range(len(CHAINED_VARIABLES)))), inverse=True, in_place=False)
    assert torch.equal(out.data[0, 0, :, q], torch.tensor([-4.5, -1.0]))


class _LegacyProcessor(BasePreprocessor):
    """A third-party processor written against the old constructor signature."""

    def __init__(self, config=None, data_indices=None, statistics=None) -> None:
        super().__init__(config, data_indices, statistics)

    def transform(self, x, **kwargs):
        return x

    def inverse_transform(self, x, **kwargs):
        return x


def test_constructor_statistics_are_deprecated() -> None:
    config = DictConfig({"default": "none"})
    with pytest.warns(FutureWarning, match="'statistics' constructor argument is deprecated"):
        _LegacyProcessor(config, None, CHAINED_STATISTICS)


@pytest.mark.parametrize("method", list(NORMALIZATION_STATISTICS))
def test_normalized_relu_thresholds_match_the_normalizer(method) -> None:
    """The threshold lands where the normaliser puts the same physical value, for every method."""
    config = DictConfig(
        {
            "data": {
                "relu": {"default": "none", 2.5: ["q"], "normalizer": method},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {name: idx for idx, name in enumerate(CHAINED_VARIABLES)}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    postprocessor = NormalizedReluPostprocessor(config=config.data.relu, data_indices=data_indices)
    q = CHAINED_VARIABLES.index("q")

    view = make_gridded_view(torch.full((1, len(CHAINED_VARIABLES)), -1.0e6), list(range(len(CHAINED_VARIABLES))))
    out = postprocessor(view, inverse=True, in_place=False)

    normalizer = InputNormalizer(config={"default": method})
    norm_mul, norm_add = normalizer.get_norm_parameters(view.statistics, view.name_to_index, torch.device("cpu"))
    torch.testing.assert_close(out.data[0, 0, 0, q], 2.5 * norm_mul[q] + norm_add[q])
