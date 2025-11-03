# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# (C) Copyright 2025 Anemoi contributors.
# Apache 2.0 license…


import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


@pytest.fixture
def fake_data() -> tuple[DictConfig, IndexCollection]:
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


@pytest.fixture
def fake_data_single_variable() -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": [],
            },
            "training": {
                "training_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": ["general_variable"],
                },
                "variable_groups": {
                    "default": "sfc",
                    "pl": ["t"],
                },
                "scalers": {
                    "builders": {
                        "general_variable": {
                            "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                            "weights": {"default": 1},
                        },
                    },
                },
            },
            "metrics": [],
        },
    )
    name_to_index = {"x": 0, "t_50": 1}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    statistics = {"stdev": [0.0, 10.0, 10, 10, 7.0, 3.0, 1.0, 2.0, 3.5]}
    statistics_tendencies = {"stdev": [0.0, 5, 5, 5, 4.0, 7.5, 8.6, 1, 10]}
    return config, data_indices, statistics, statistics_tendencies


def test_variable_scaling_multi_variable(
    fake_data: tuple[DictConfig, IndexCollection, torch.Tensor, torch.Tensor],
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

    print_variable_scaling(loss, data_indices)


def test_variable_scaling_single_variable(
    fake_data_single_variable: tuple[DictConfig, IndexCollection, torch.Tensor, torch.Tensor],
    graph_with_nodes: HeteroData,
) -> None:
    config, data_indices, statistics, statistics_tendencies = fake_data_single_variable

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

    print_variable_scaling(loss, data_indices)
