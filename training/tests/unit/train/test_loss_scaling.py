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
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


@pytest.fixture
def fake_data(request: SubRequest) -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "training": {
                "training_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": ["general_variable", "additional_scaler"],
                },
                "variable_groups": {
                    "default": "sfc",
                    "pl": ["y"],
                },
                "scalers": {
                    "builders": {
                        "additional_scaler": request.param,
                        "general_variable": {
                            "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                            "weights": {
                                "default": 1,
                                "z": 0.1,
                                "other": 100,
                                "y": 0.5,
                            },
                        },
                    },
                },
                "metrics": ["other", "y_850"],
            },
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    statistics = {"stdev": [0.0, 10.0, 10, 10, 7.0, 3.0, 1.0, 2.0, 3.5]}
    statistics_tendencies = {"stdev": [0.0, 5, 5, 5, 4.0, 7.5, 8.6, 1, 10]}
    return config, data_indices, statistics, statistics_tendencies


@pytest.fixture
def fake_data_no_param() -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "training": {
                "training_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": ["variable_masking"],
                },
                "variable_groups": {
                    "default": "sfc",
                    "pl": ["y"],
                },
                "scalers": {
                    "builders": {
                        "variable_masking": {
                            "_target_": "anemoi.training.losses.scalers.VariableMaskingLossScaler",
                            "variables": ["z", "other", "q"],
                        },
                    },
                },
            },
            "metrics": [],
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    statistics = {"stdev": [0.0, 10.0, 10, 10, 7.0, 3.0, 1.0, 2.0, 3.5]}
    statistics_tendencies = {"stdev": [0.0, 5, 5, 5, 4.0, 7.5, 8.6, 1, 10]}
    return config, data_indices, statistics, statistics_tendencies


linear_scaler = {
    "_target_": "anemoi.training.losses.scalers.LinearVariableLevelScaler",
    "group": "pl",
    "y_intercept": 0.0,
    "slope": 0.001,
}

relu_scaler = {
    "_target_": "anemoi.training.losses.scalers.ReluVariableLevelScaler",
    "group": "pl",
    "y_intercept": 0.2,
    "slope": 0.001,
}

constant_scaler = {
    "_target_": "anemoi.training.losses.scalers.NoVariableLevelScaler",
    "group": "pl",
}
polynomial_scaler = {
    "_target_": "anemoi.training.losses.scalers.PolynomialVariableLevelScaler",
    "group": "pl",
    "y_intercept": 0.2,
    "slope": 0.001,
}


std_dev_scaler = {"_target_": "anemoi.training.losses.scalers.StdevTendencyScaler"}

var_scaler = {"_target_": "anemoi.training.losses.scalers.VarTendencyScaler"}

no_tend_scaler = {"_target_": "anemoi.training.losses.scalers.NoTendencyScaler"}

graph_node_scaler = {
    "_target_": "anemoi.training.losses.scalers.GraphNodeAttributeScaler",
    "nodes_name": "test_nodes",
    "nodes_attribute_name": "test_attr",
    "norm": "unit-sum",
}

reweighted_graph_node_scaler = {
    "_target_": "anemoi.training.losses.scalers.ReweightedGraphNodeAttributeScaler",
    "nodes_name": "test_nodes",
    "nodes_attribute_name": "test_attr",
    "scaling_mask_attribute_name": "mask",
    "weight_frac_of_total": 0.4,
    "norm": "unit-sum",
}

expected_linear_scaling = torch.Tensor(
    [
        50 / 1000 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
        1,  # d
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
        1,  # d
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
        1,  # d
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
        1,  # d
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
        1 * 1,  # d
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
        1 * 2.0,  # d
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
        (2**2) / (1**2),  # d
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
    graph_with_nodes: HeteroData,
) -> None:
    config, data_indices, statistics, statistics_tendencies = fake_data

    metadata_extractor = ExtractVariableGroupAndLevel(
        config.training.variable_groups,
    )

    scalers, _ = create_scalers(
        config.training.scalers.builders,
        data_indices=data_indices,
        graph_data=graph_with_nodes,
        statistics=statistics,
        statistics_tendencies=statistics_tendencies,
        metadata_extractor=metadata_extractor,
        output_mask=NoOutputMask(),
    )

    loss = get_loss_function(config.training.training_loss, scalers=scalers)

    final_variable_scaling = loss.scaler.subset_by_dim(TensorDim.VARIABLE.value).get_scaler(len(TensorDim))

    assert torch.allclose(torch.tensor(final_variable_scaling), expected_scaling)


@pytest.mark.parametrize("fake_data", [linear_scaler], indirect=["fake_data"])
def test_metric_range(fake_data: tuple[DictConfig, IndexCollection]) -> None:
    config, data_indices, _, _ = fake_data

    metadata_extractor = ExtractVariableGroupAndLevel(config.training.variable_groups)
    metric_range = get_metric_ranges(config, data_indices, metadata_extractor=metadata_extractor)

    del metric_range["all"]

    expected_metric_range = {
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

    assert metric_range == expected_metric_range


def test_variable_masking(
    fake_data_no_param: tuple[DictConfig, IndexCollection, torch.Tensor, torch.Tensor],
    graph_with_nodes: HeteroData,
) -> None:
    config, data_indices, statistics, statistics_tendencies = fake_data_no_param

    metadata_extractor = ExtractVariableGroupAndLevel(
        config.training.variable_groups,
    )

    scalers, _ = create_scalers(
        config.training.scalers.builders,
        data_indices=data_indices,
        graph_data=graph_with_nodes,
        statistics=statistics,
        statistics_tendencies=statistics_tendencies,
        metadata_extractor=metadata_extractor,
        output_mask=NoOutputMask(),
    )
    vars_to_mask = ["z", "other", "q"]
    indices_to_mask = [data_indices.model.output.name_to_index[v] for v in vars_to_mask]
    assert scalers["variable_masking"][0][0] == len(vars_to_mask)
    assert not scalers["variable_masking"][1][indices_to_mask].any(), "Expected scalers for masked variables to be zero"

    config.training.scalers.builders["variable_masking"].update(inverse=True)
    scalers, _ = create_scalers(
        config.training.scalers.builders,
        data_indices=data_indices,
        graph_data=graph_with_nodes,
        statistics=statistics,
        statistics_tendencies=statistics_tendencies,
        metadata_extractor=metadata_extractor,
        output_mask=NoOutputMask(),
    )
    assert scalers["variable_masking"][0][0] == len(vars_to_mask)
    assert scalers["variable_masking"][1][indices_to_mask].all(), "Expected scalers for unmasked variables to be one"
