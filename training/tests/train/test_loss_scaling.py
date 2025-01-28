# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.train.scaling import GeneralVariableLossScaler


@pytest.fixture
def fake_data(request: SubRequest) -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
                "remapped": {
                    "d": ["cos_d", "sin_d"],
                },
            },
            "training": {
                "variable_loss_scaling": {
                    "variable_groups": {
                        "default": "sfc",
                        "pl": ["y"],
                    },
                    "default": 1,
                    "z": 0.1,
                    "other": 100,
                    "y": 0.5,
                },
                "metrics": ["other", "y_850"],
                "additional_scalers": request.param,
            },
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    statistics = {"stdev": [0.0, 10.0, 10, 10, 7.0, 3.0, 1.0, 2.0, 3.5]}
    statistics_tendencies = {"stdev": [0.0, 5, 5, 5, 4.0, 7.5, 8.6, 1, 10]}
    return config, data_indices, statistics, statistics_tendencies


linear_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.LinearVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.0,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
relu_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.ReluVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.2,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
constant_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.NoVariableLevelScaler",
        "group": "pl",
        "y_intercept": 1.0,
        "slope": 0.0,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]
polynomial_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.PolynomialVariableLevelScaler",
        "group": "pl",
        "y_intercept": 0.2,
        "slope": 0.001,
        "scale_dim": -1,
        "name": "variable_pressure_level",
    },
]

std_dev_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.StdevTendencyScaler",
        "name": "tendency",
        "scale_dim": -1,
    },
]

var_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.VarTendencyScaler",
        "name": "tendency",
        "scale_dim": -1,
    },
]

no_tend_scaler = [
    {
        "_target_": "anemoi.training.train.scaling.NoTendencyScaler",
        "name": "tendency",
        "scale_dim": -1,
    },
]

expected_linear_scaling = torch.Tensor(
    [
        50 / 1000 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_relu_scaling = torch.Tensor(
    [
        0.2 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_constant_scaling = torch.Tensor(
    [
        1 * 0.5,  # y_50
        1 * 0.5,  # y_500
        1 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)
expected_polynomial_scaling = torch.Tensor(
    [
        ((50 / 1000) ** 2 + 0.2) * 0.5,  # y_50
        ((500 / 1000) ** 2 + 0.2) * 0.5,  # y_500
        ((850 / 1000) ** 2 + 0.2) * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # cos_d
        1,  # sin_d
    ],
)

expected_no_tendency_scaling = torch.Tensor(
    [
        1 * 0.5,  # y_50
        1 * 0.5,  # y_500
        1 * 0.5,  # y_850
        1 * 1,  # q
        1 * 0.1,  # z
        1 * 100,  # other
        1 * 1,  # cos_d
        1 * 1,  # sin_d
    ],
)

expected_stdev_tendency_scaling = torch.Tensor(
    [
        (10.0 / 5.0) * 0.5,  # y_50
        (10.0 / 5.0) * 0.5,  # y_500
        (10.0 / 5.0) * 0.5,  # y_850
        1 * 1,  # q (diagnostic)
        1 * 0.1,  # z (diagnostic)
        (1 / 8.6) * 100,  # other
        1 * 1,  # cos_d (remapped)
        1 * 1,  # sin_d (remapped)
    ],
)

expected_var_tendency_scaling = torch.Tensor(
    [
        (10.0**2) / (5.0**2) * 0.5,  # y_50
        (10.0**2) / (5.0**2) * 0.5,  # y_500
        (10.0**2) / (5.0**2) * 0.5,  # y_850
        1,  # q (diagnostic)
        0.1,  # z (diagnostic)
        (1**2) / (8.6**2) * 100,  # other
        1 * 1,  # cos_d (remapped)
        1 * 1,  # sin_d (remapped)
    ],
)


@pytest.mark.parametrize(
    ("fake_data", "expected_scaling"),
    [
        (linear_scaler, expected_linear_scaling),
        (relu_scaler, expected_relu_scaling),
        (constant_scaler, expected_constant_scaling),
        (polynomial_scaler, expected_polynomial_scaling),
        (no_tend_scaler, expected_no_tendency_scaling),
        (std_dev_scaler, expected_stdev_tendency_scaling),
        (var_scaler, expected_var_tendency_scaling),
    ],
    indirect=["fake_data"],
)
def test_variable_loss_scaling_vals(
    fake_data: tuple[DictConfig, IndexCollection, torch.Tensor, torch.Tensor],
    expected_scaling: torch.Tensor,
) -> None:
    config, data_indices, statistics, statistics_tendencies = fake_data
    variable_scaling = GeneralVariableLossScaler(
        config.training.variable_loss_scaling,
        data_indices,
    ).get_scaling()

    scaler = [
        (
            instantiate(
                scaler_config,
                scaling_config=config.training.variable_loss_scaling,
                data_indices=data_indices,
                statistics=statistics,
                statistics_tendencies=statistics_tendencies,
                metadata_variables=None,
            )
        )
        for scaler_config in config.training.additional_scalers
    ]

    scalers = {
        "variable": (-1, variable_scaling),
    }
    # add addtional user-defined scalers

    [scalers.update({scale.name: (scale.scale_dim, scale.get_scaling())}) for scale in scaler]
    keys_list = list(scalers.keys())
    assert torch.allclose(torch.tensor(scalers[keys_list[0]][1] * scalers[keys_list[1]][1]), expected_scaling)


@pytest.mark.parametrize("fake_data", [linear_scaler], indirect=["fake_data"])
def test_metric_range(fake_data: tuple[DictConfig, IndexCollection]) -> None:
    config, data_indices, _, _ = fake_data

    metric_range, metric_ranges_validation = GraphForecaster.get_val_metric_ranges(config, data_indices)

    del metric_range["all"]
    del metric_ranges_validation["all"]

    expected_metric_range_validation = {
        "pl_y": [
            data_indices.model.output.name_to_index["y_50"],
            data_indices.model.output.name_to_index["y_500"],
            data_indices.model.output.name_to_index["y_850"],
        ],
        "sfc_other": [data_indices.model.output.name_to_index["other"]],
        "sfc_q": [data_indices.model.output.name_to_index["q"]],
        "sfc_z": [data_indices.model.output.name_to_index["z"]],
        "other": [data_indices.model.output.name_to_index["other"]],
        "sfc_d": [data_indices.model.output.name_to_index["d"]],
        "y_850": [data_indices.model.output.name_to_index["y_850"]],
    }

    expected_metric_range = expected_metric_range_validation.copy()
    del expected_metric_range["sfc_d"]
    expected_metric_range["sfc_cos_d"] = [data_indices.internal_model.output.name_to_index["cos_d"]]
    expected_metric_range["sfc_sin_d"] = [data_indices.internal_model.output.name_to_index["sin_d"]]

    assert metric_ranges_validation == expected_metric_range_validation
    assert metric_range == expected_metric_range
