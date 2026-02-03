# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import DictConfig

from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.filtering import FilteringLossWrapper
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


def test_instantiation_with_filtering() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    """Test that loss function can be instantiated."""
    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {"tp": 0, "other_var": 1}
    data_indices = IndexCollection(DictConfig(data_config).data, name_to_index)
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.LogFFT2Distance",
                "predicted_variables": ["tp"],
                "target_variables": ["tp"],
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
        data_indices=data_indices,
    )
    assert isinstance(loss, FilteringLossWrapper)
    assert isinstance(loss.loss, BaseLoss)
    assert hasattr(loss.loss, "y_dim")
    assert hasattr(loss.loss, "x_dim")

    assert hasattr(loss, "predicted_indices")

    assert loss.predicted_variables == ["tp"]
    # tensors are of size (batch, output_steps, ens, latlon, vars)
    right_shaped_pred_output_pair = (
        torch.ones((6, 1, 710 * 640, 2)),
        torch.zeros((6, 1, 710 * 640, 2)),
    )
    loss_value = loss(*right_shaped_pred_output_pair, squash=False)
    assert loss_value.shape[0] == len(
        name_to_index.keys(),
    ), "Loss output with squash=False should match length of all variables"
    assert (
        torch.nonzero(loss_value)[0].tolist() == loss.predicted_indices
    ), "Filtered out variables should have zero loss"
    loss_total = loss(*right_shaped_pred_output_pair, squash=True)
    assert (
        loss_total == loss_value[0]
    ), "Loss output with squash=True should be the value of loss for predicted variables"


def test_print_variable_scaling() -> None:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.scalers.scalers import create_scalers
    from anemoi.training.losses.utils import print_variable_scaling
    from anemoi.utils.config import DotDict

    data_config = {"data": {"forcing": ["f1"], "target": [], "prognostic": ["f2"], "diagnostic": ["tp", "imerg"]}}
    name_to_index = {"tp": 0, "imerg": 1, "f1": 2, "f2": 3}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    metadata_extractor = ExtractVariableGroupAndLevel(
        DotDict(
            {
                "default": "sfc",
            },
        ),
    )
    scalers, _ = create_scalers(
        DotDict(
            {
                "general_variable": {
                    "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                    "weights": {
                        "default": 1,
                        "tp": 0.1,
                        "imerg": 100,
                        "f2": 0.5,
                    },
                },
            },
        ),
        data_indices=data_indices,
        metadata_extractor=metadata_extractor,
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": ["general_variable"],
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MAELoss",
                        "scalers": ["general_variable"],
                        "predicted_variables": ["tp"],
                        "target_variables": ["imerg"],
                    },
                ],
            },
        ),
        data_indices=data_indices,
        scalers=scalers,
    )
    scaling_dict = print_variable_scaling(loss, data_indices)
    assert "FilteringLossWrapper" in scaling_dict  # loss is filtered
    assert "tp" in scaling_dict["FilteringLossWrapper"]
    assert [var not in scaling_dict["FilteringLossWrapper"] for var in data_indices.name_to_index if var != "tp"]
