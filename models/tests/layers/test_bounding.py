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
    # breakpoint()
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


def test_hydra_instantiate_bounding(config, name_to_index, input_tensor):
    layer_definitions = [
        {
            "_target_": "anemoi.models.layers.bounding.ReluBounding",
            "variables": config.variables,
        },
        {
            "_target_": "anemoi.models.layers.bounding.HardtanhBounding",
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
    ]
    for layer_definition in layer_definitions:
        bounding = instantiate(layer_definition, name_to_index=name_to_index)
        bounding(input_tensor.clone())
