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
from anemoi.models.data_indices.collection import IndexCollection
from omegaconf import DictConfig

from anemoi.training.train.forecaster import GraphForecaster


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
                "loss_scaling": {
                    "default": 1,
                    "sfc": {
                        "z": 0.1,
                        "other": 100,
                    },
                    "pl": {"y": 0.5},
                },
                "metrics": ["other", "y_850"],
                "pressure_level_scaler": request.param,
            },
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return config, data_indices


linear_scaler = {
    "_target_": "anemoi.training.data.scaling.LinearPressureLevelScaler",
    "minimum": 0.0,
    "slope": 0.001,
}
relu_scaler = {
    "_target_": "anemoi.training.data.scaling.ReluPressureLevelScaler",
    "minimum": 0.2,
    "slope": 0.001,
}
constant_scaler = {
    "_target_": "anemoi.training.data.scaling.NoPressureLevelScaler",
    "minimum": 1.0,
    "slope": 0.0,
}
polynomial_scaler = {
    "_target_": "anemoi.training.data.scaling.PolynomialPressureLevelScaler",
    "minimum": 0.2,
    "slope": 0.001,
}

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


@pytest.mark.parametrize(
    ("fake_data", "expected_scaling"),
    [
        (linear_scaler, expected_linear_scaling),
        (relu_scaler, expected_relu_scaling),
        (constant_scaler, expected_constant_scaling),
        (polynomial_scaler, expected_polynomial_scaling),
    ],
    indirect=["fake_data"],
)
def test_loss_scaling_vals(fake_data: tuple[DictConfig, IndexCollection], expected_scaling: torch.Tensor) -> None:
    config, data_indices = fake_data

    _, _, loss_scaling = GraphForecaster.metrics_loss_scaling(config, data_indices)

    assert torch.allclose(loss_scaling, expected_scaling)


@pytest.mark.parametrize("fake_data", [linear_scaler], indirect=["fake_data"])
def test_metric_range(fake_data: tuple[DictConfig, IndexCollection]) -> None:
    config, data_indices = fake_data

    metric_range, metric_ranges_validation, _ = GraphForecaster.metrics_loss_scaling(config, data_indices)

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

    assert dict(metric_ranges_validation) == expected_metric_range_validation
    assert dict(metric_range) == expected_metric_range
