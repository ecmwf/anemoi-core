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
from hydra.utils import instantiate

from anemoi.models.layers.bounding import FractionBounding
from anemoi.models.layers.bounding import HardtanhBounding
from anemoi.models.layers.bounding import LeakyFractionBounding
from anemoi.models.layers.bounding import LeakyHardtanhBounding
from anemoi.models.layers.bounding import LeakyNormalizedReluBounding
from anemoi.models.layers.bounding import LeakyReluBounding
from anemoi.models.layers.bounding import NormalizedReluBounding
from anemoi.models.layers.bounding import ReluBounding
from anemoi.utils.config import DotDict


@pytest.fixture
def config():
    return DotDict({"variables": ["var1", "var2"], "total_var": "total_var"})


@pytest.fixture
def name_to_index():
    return {"var1": 0, "var2": 1, "total_var": 2}


@pytest.fixture
def name_to_index_stats():
    return {"var1": 0, "var2": 1, "total_var": 2}


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


def test_relu_bounding(config, name_to_index, input_tensor):
    bounding = ReluBounding(variables=config.variables, name_to_index=name_to_index)
    output = bounding(input_tensor.clone())
    expected_output = torch.tensor([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_normalized_relu_bounding(config, name_to_index, name_to_index_stats, input_tensor, statistics):
    bounding = NormalizedReluBounding(
        variables=config.variables,
        name_to_index=name_to_index,
        min_val=[2.0, 2.0],
        normalizer=["mean-std", "min-max"],
        statistics=statistics,
        name_to_index_stats=name_to_index_stats,
    )
    output = bounding(input_tensor.clone())
    expected_output = torch.tensor([[2.0, 2.0, 3.0], [4.0, 0.1111, 6.0], [2.0, 0.5, 0.5]])
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_hardtanh_bounding(config, name_to_index, input_tensor):
    minimum, maximum = -1.0, 1.0
    bounding = HardtanhBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=minimum, max_val=maximum
    )
    output = bounding(input_tensor.clone())
    expected_output = torch.tensor([[minimum, maximum, 3.0], [maximum, minimum, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_fraction_bounding(config, name_to_index, input_tensor):
    bounding = FractionBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=0.0, max_val=1.0, total_var=config.total_var
    )
    output = bounding(input_tensor.clone())
    expected_output = torch.tensor([[0.0, 3.0, 3.0], [6.0, 0.0, 6.0], [0.25, 0.25, 0.5]])

    assert torch.equal(output, expected_output)


def test_multi_chained_bounding(config, name_to_index, input_tensor):
    # Apply Relu first on the first variable only
    bounding1 = ReluBounding(variables=config.variables[:-1], name_to_index=name_to_index)
    expected_output = torch.tensor([[0.0, 2.0, 3.0], [4.0, -5.0, 6.0], [0.5, 0.5, 0.5]])
    # Check intemediate result
    assert torch.equal(bounding1(input_tensor.clone()), expected_output)
    minimum, maximum = 0.5, 1.75
    bounding2 = HardtanhBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=minimum, max_val=maximum
    )
    # Use full chaining on the input tensor
    output = bounding2(bounding1(input_tensor.clone()))
    # Data with Relu applied first and then Hardtanh
    expected_output = torch.tensor([[minimum, maximum, 3.0], [maximum, minimum, 6.0], [0.5, 0.5, 0.5]])
    assert torch.equal(output, expected_output)


def test_hydra_instantiate_bounding(config, name_to_index, name_to_index_stats, input_tensor, statistics):
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
            "_target_": "anemoi.models.layers.bounding.LeakyNormalizedReluBounding",
            "variables": config.variables,
            "min_val": [2.0, 2.0],
            "normalizer": ["mean-std", "min-max"],
            "statistics": statistics,
            "name_to_index_stats": name_to_index_stats,
        },
    ]
    for layer_definition in layer_definitions:
        bounding = instantiate(layer_definition, name_to_index=name_to_index)
        bounding(input_tensor.clone())


def test_leaky_relu_bounding(config, name_to_index, input_tensor):
    bounding = LeakyReluBounding(variables=config.variables, name_to_index=name_to_index)
    output = bounding(input_tensor.clone())
    # LeakyReLU should keep negative values but scale them by 0.01 (default negative_slope)
    expected_output = torch.tensor([[-0.01, 2.0, 3.0], [4.0, -0.05, 6.0], [0.5, 0.5, 0.5]])
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_leaky_hardtanh_bounding(config, name_to_index, input_tensor):
    minimum, maximum = -1.0, 1.0
    bounding = LeakyHardtanhBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=minimum, max_val=maximum
    )
    output = bounding(input_tensor.clone())
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


def test_leaky_fraction_bounding(config, name_to_index, input_tensor):
    bounding = LeakyFractionBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=0.0, max_val=1.0, total_var=config.total_var
    )
    output = bounding(input_tensor.clone())
    # First apply leaky hardtanh, then multiply by total_var
    expected_output = torch.tensor(
        [
            [-0.03, 3.03, 3.0],  # [-1, 2, 3] -> [leaky(0), leaky(1), 3] -> [leaky(0)*3, leaky(1)*3, 3]
            [6.18, -0.3, 6.0],  # [4, -5, 6] -> [leaky(1), leaky(0), 6] -> [leaky(1)*6, leaky(0)*6, 6]
            [0.25, 0.25, 0.5],  # [0.5, 0.5, 0.5] -> [0.5, 0.5, 0.5] -> [0.5*0.5, 0.5*0.5, 0.5]
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_multi_chained_bounding_with_leaky(config, name_to_index, input_tensor):
    # Apply LeakyReLU first on the first variable only
    bounding1 = LeakyReluBounding(variables=config.variables[:-1], name_to_index=name_to_index)
    expected_output = torch.tensor([[-0.01, 2.0, 3.0], [4.0, -5.0, 6.0], [0.5, 0.5, 0.5]])
    # Check intermediate result
    assert torch.allclose(bounding1(input_tensor.clone()), expected_output, atol=1e-4)

    minimum, maximum = 0.5, 1.75
    bounding2 = LeakyHardtanhBounding(
        variables=config.variables, name_to_index=name_to_index, min_val=minimum, max_val=maximum
    )
    # Use full chaining on the input tensor
    output = bounding2(bounding1(input_tensor.clone()))
    # Data with LeakyReLU applied first and then LeakyHardtanh
    expected_output = torch.tensor(
        [
            [minimum + 0.01 * (-0.01 - minimum), maximum + 0.01 * (2.0 - maximum), 3.0],
            [maximum + 0.01 * (4.0 - maximum), minimum + 0.01 * (-5.0 - minimum), 6.0],
            [0.5, 0.5, 0.5],
        ]
    )
    assert torch.allclose(output, expected_output, atol=1e-4)


def test_leaky_normalized_relu_bounding(config, name_to_index, name_to_index_stats, input_tensor, statistics):
    bounding = LeakyNormalizedReluBounding(
        variables=config.variables,
        name_to_index=name_to_index,
        min_val=[2.0, 2.0],
        normalizer=["mean-std", "min-max"],
        statistics=statistics,
        name_to_index_stats=name_to_index_stats,
    )
    output = bounding(input_tensor.clone())

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
