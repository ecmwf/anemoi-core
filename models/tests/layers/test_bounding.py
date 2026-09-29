# (C) Copyright 2024-2026 Anemoi contributors.
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
from hydra.utils import instantiate

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.layers.bounding import FractionBounding
from anemoi.models.layers.bounding import HardtanhBounding
from anemoi.models.layers.bounding import LeakyFractionBounding
from anemoi.models.layers.bounding import LeakyHardtanhBounding
from anemoi.models.layers.bounding import LeakyReluBounding
from anemoi.models.layers.bounding import NormalizedLeakyReluBounding
from anemoi.models.layers.bounding import NormalizedReluBounding
from anemoi.models.layers.bounding import ReluBounding
from anemoi.utils.config import DotDict


@pytest.fixture
def config():
    return DotDict({"variables": ["var1", "var2"], "total_var": "total_var"})


VARIABLES = ["var1", "var2", "total_var"]


@pytest.fixture
def input_tensor():
    return torch.tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [0.5, 0.5, 0.5]])


@pytest.fixture
def statistics():
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5]),
        "min": np.array([1.0, 1.0, 1.0]),
        "max": np.array([11.0, 10.0, 10.0]),
    }
    return statistics


def _make_gridded(payload: torch.Tensor, variables: list[str], statistics: dict) -> GriddedSource:
    """Wrap a (points, variables) payload in a GriddedSource of shape (batch, time, grid, variables)."""
    points, num_vars = payload.shape
    return GriddedSource(
        name="gridded",
        data=payload.reshape(1, 1, points, num_vars).clone(),
        variables=list(variables),
        statistics=statistics,
        coordinates=torch.zeros(points, 2),
        layout=TensorLayout(batch=0, time=1, grid=2, variables=3),
    )


def _make_tabular(payload: torch.Tensor, variables: list[str], statistics: dict) -> TabularSource:
    """Wrap a (points, variables) payload in a single-sample TabularSource."""
    points = payload.shape[0]
    return TabularSource(
        name="tabular",
        data=[payload.clone()],
        variables=list(variables),
        statistics=statistics,
        coordinates=[torch.zeros(points, 2)],
        layout=TensorLayout(grid=0, variables=1),
        timedeltas=[torch.zeros(points)],
        boundaries=[(slice(0, points),)],
    )


def _to_2d(source) -> torch.Tensor:
    """Return the (points, variables) payload of a source built by the helpers above."""
    if isinstance(source, TabularSource):
        return source.data[0]
    return source.data.reshape(-1, source.data.shape[-1])


@pytest.fixture(params=["gridded", "tabular"])
def source(request, input_tensor, statistics):
    builder = _make_gridded if request.param == "gridded" else _make_tabular
    return builder(input_tensor, VARIABLES, statistics)


def test_relu_bounding(config, source):
    bounding = ReluBounding(variables=config.variables)
    output = _to_2d(bounding(source))
    expected_output = torch.tensor([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_normalized_relu_bounding(config, source):
    min_val = [2.0, 2.0]
    normalizer = ["mean-std", "min-max"]
    bounding = NormalizedReluBounding(
        variables=config.variables,
        min_val=min_val,
        normalizer=normalizer,
    )
    output = _to_2d(bounding(source))
    expected_output = torch.tensor([[2.0, 2.0, 3.0], [4.0, 0.1111, 6.0], [2.0, 0.5, 0.5]])
    assert torch.allclose(output, expected_output, atol=1e-4)

    # test with order of variables in configuration different to input tensor
    bounding = NormalizedReluBounding(
        variables=config.variables[::-1],  # reverse order
        min_val=min_val[::-1],  # reverse order
        normalizer=normalizer[::-1],  # reverse order
    )
    output = _to_2d(bounding(source))
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_hardtanh_bounding(config, source):
    minimum, maximum = -1.0, 1.0
    bounding = HardtanhBounding(variables=config.variables, min_val=minimum, max_val=maximum)
    output = _to_2d(bounding(source))
    expected_output = torch.tensor([[minimum, maximum, 3.0], [maximum, minimum, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_fraction_bounding(config, source):
    bounding = FractionBounding(variables=config.variables, min_val=0.0, max_val=1.0, total_var=config.total_var)
    output = _to_2d(bounding(source))
    expected_output = torch.tensor([[0.0, 3.0, 3.0], [6.0, 0.0, 6.0], [0.25, 0.25, 0.5]])

    assert torch.equal(output, expected_output)


def test_multi_chained_bounding(config, source):
    # Apply Relu first on the first variable only
    bounding1 = ReluBounding(variables=config.variables[:-1])
    expected_output = torch.tensor([[0.0, 2.0, 3.0], [4.0, -5.0, 6.0], [0.5, 0.5, 0.5]])
    # Check intemediate result
    assert torch.equal(_to_2d(bounding1(source)), expected_output)
    minimum, maximum = 0.5, 1.75
    bounding2 = HardtanhBounding(variables=config.variables, min_val=minimum, max_val=maximum)
    # Use full chaining on the input tensor
    output = _to_2d(bounding2(bounding1(source)))
    # Data with Relu applied first and then Hardtanh
    expected_output = torch.tensor([[minimum, maximum, 3.0], [maximum, minimum, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_hydra_instantiate_bounding(config, source):
    layer_definitions = [
        {
            "_target_": "anemoi.models.layers.bounding.ReluBounding",
            "variables": config.variables,
        },
        {
            "_target_": "anemoi.models.layers.bounding.LeakyReluBounding",
            "variables": config.variables,
        },
        {
            "_target_": "anemoi.models.layers.bounding.HardtanhBounding",
            "variables": config.variables,
            "min_val": 0.0,
            "max_val": 1.0,
        },
        {
            "_target_": "anemoi.models.layers.bounding.LeakyHardtanhBounding",
            "variables": config.variables,
            "min_val": 0.0,
            "max_val": 1.0,
        },
        {
            "_target_": "anemoi.models.layers.bounding.FractionBounding",
            "variables": config.variables,
            "min_val": 0.0,
            "max_val": 1.0,
            "total_var": config.total_var,
        },
        {
            "_target_": "anemoi.models.layers.bounding.LeakyFractionBounding",
            "variables": config.variables,
            "min_val": 0.0,
            "max_val": 1.0,
            "total_var": config.total_var,
        },
        {
            "_target_": "anemoi.models.layers.bounding.NormalizedLeakyReluBounding",
            "variables": config.variables,
            "min_val": [2.0, 2.0],
            "normalizer": ["min-max", "mean-std"],
        },
    ]
    for layer_definition in layer_definitions:
        bounding = instantiate(layer_definition)
        bounding(source)


def test_leaky_relu_bounding(config, source):
    bounding = LeakyReluBounding(variables=config.variables)
    output = _to_2d(bounding(source))
    # LeakyReLU should keep negative values but scale them by 0.01 (default negative_slope)
    expected_output = torch.tensor([[-0.01, 2.0, 3.0], [4.0, -0.05, 6.0], [0.5, 0.5, 0.5]])
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_leaky_hardtanh_bounding(config, source):
    minimum, maximum = -1.0, 1.0
    bounding = LeakyHardtanhBounding(variables=config.variables, min_val=minimum, max_val=maximum)
    output = _to_2d(bounding(source))
    # Values below min_val should be min_val + 0.01 * (input - min_val)
    # Values above max_val should be max_val + 0.01 * (input - max_val)
    expected_output = torch.tensor(
        [
            [minimum + 0.01 * (-1.0 - minimum), maximum + 0.01 * (2.0 - maximum), 3.0],
            [maximum + 0.01 * (4.0 - maximum), minimum + 0.01 * (-5.0 - minimum), 6.0],
            [0.5, 0.5, 0.5],
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_leaky_fraction_bounding(config, source):
    bounding = LeakyFractionBounding(variables=config.variables, min_val=0.0, max_val=1.0, total_var=config.total_var)
    output = _to_2d(bounding(source))
    # First apply leaky hardtanh, then multiply by total_var
    expected_output = torch.tensor(
        [
            [-0.03, 3.03, 3.0],  # [-1, 2, 3] -> [leaky(0), leaky(1), 3] -> [leaky(0)*3, leaky(1)*3, 3]
            [6.18, -0.3, 6.0],  # [4, -5, 6] -> [leaky(1), leaky(0), 6] -> [leaky(1)*6, leaky(0)*6, 6]
            [0.25, 0.25, 0.5],  # [0.5, 0.5, 0.5] -> [0.5, 0.5, 0.5] -> [0.5*0.5, 0.5*0.5, 0.5]
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_multi_chained_bounding_with_leaky(config, source):
    # Apply LeakyReLU first on the first variable only
    bounding1 = LeakyReluBounding(variables=config.variables[:-1])
    expected_output = torch.tensor([[-0.01, 2.0, 3.0], [4.0, -5.0, 6.0], [0.5, 0.5, 0.5]])
    # Check intermediate result
    assert torch.allclose(_to_2d(bounding1(source)), expected_output, atol=1e-4)

    minimum, maximum = 0.5, 1.75
    bounding2 = LeakyHardtanhBounding(variables=config.variables, min_val=minimum, max_val=maximum)
    # Use full chaining on the input tensor
    output = _to_2d(bounding2(bounding1(source)))
    # Data with LeakyReLU applied first and then LeakyHardtanh
    expected_output = torch.tensor(
        [
            [minimum + 0.01 * (-0.01 - minimum), maximum + 0.01 * (2.0 - maximum), 3.0],
            [maximum + 0.01 * (4.0 - maximum), minimum + 0.01 * (-5.0 - minimum), 6.0],
            [0.5, 0.5, 0.5],
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_normalized_leaky_relu_bounding(config, source):
    bounding = NormalizedLeakyReluBounding(
        variables=config.variables,
        min_val=[2.0, 2.0],
        normalizer=["mean-std", "min-max"],
    )
    output = _to_2d(bounding(source))

    # For mean-std normalization:
    # normalized = (input - mean) / stdev
    # For min-max normalization:
    # normalized = (input - min) / (max - min)

    # First variable (mean-std):
    # [-1, 4, 0.5] -> [(-1-1)/0.5, (4-1)/0.5, (0.5-1)/0.5] = [-4, 6, -1]
    # Then leaky_relu: [-4, 6, -1] -> [-4*0.01, 6, -1*0.01] = [-0.04, 6, -0.01]
    # Then add min_val: [-0.04+2, 6+2, -0.01+2] = [1.96, 8, 1.99]

    # Second variable (min-max):
    # [2, -5, 0.5] -> [(2-1)/(10-1), (-5-1)/(10-1), (0.5-1)/(10-1)] = [0.111, -0.667, -0.056]
    # Then leaky_relu: [0.111, -0.667, -0.056] -> [0.111, -0.667*0.01, -0.056*0.01] = [0.111, -0.00667, -0.00056]
    # Then add min_val: [0.111+2, -0.00667+2, -0.00056+2] = [2.111, 1.993, 1.999]

    expected_output = torch.tensor(
        [
            [1.97, 2.0, 3.0],  # [-1, 2, 3] -> [1.97, 2.0, 3.0]
            [4.0, 0.06, 6.0],  # [4, -5, 6] -> [4.0, 0.06, 6.0]
            [1.985, 0.5, 0.5],  # [0.5, 0.5, 0.5] -> [1.985, 0.5, 0.5]
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


@pytest.mark.parametrize(
    "cls, extra_kwargs",
    [
        (ReluBounding, {}),
        (LeakyReluBounding, {}),
        (HardtanhBounding, {"min_val": -1.0, "max_val": 1.0}),
        (LeakyHardtanhBounding, {"min_val": -1.0, "max_val": 1.0}),
        (FractionBounding, {"min_val": 0.0, "max_val": 1.0, "total_var": "total_var"}),
        (LeakyFractionBounding, {"min_val": 0.0, "max_val": 1.0, "total_var": "total_var"}),
    ],
)
def test_skip_missing_variables(cls, extra_kwargs, source):
    """Variables absent from the source are skipped; the others are bounded as usual."""
    reference = cls(variables=["var1"], **extra_kwargs)(source)
    output = cls(variables=["var1", "missing_var"], **extra_kwargs)(source)
    assert torch.equal(_to_2d(output), _to_2d(reference))


@pytest.mark.parametrize("cls", [NormalizedReluBounding, NormalizedLeakyReluBounding])
def test_normalized_fail_with_missing_variables(cls, source):
    """Normalized boundings need statistics for every variable, so a missing one raises."""
    bounding = cls(variables=["var1", "missing_var"], min_val=[2.0, 99.0], normalizer=["mean-std", "mean-std"])
    with pytest.raises(KeyError):
        bounding(source)


@pytest.mark.parametrize("cls", [FractionBounding, LeakyFractionBounding])
def test_fraction_fail_with_missing_total_var(cls, source):
    bounding = cls(variables=["var1"], min_val=0.0, max_val=1.0, total_var="missing_total")
    with pytest.raises(KeyError):
        bounding(source)
