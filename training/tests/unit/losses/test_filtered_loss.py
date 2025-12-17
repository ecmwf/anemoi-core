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
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.filtering import FilteringLossWrapper


def test_filtered_loss() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    """Test that loss function can be instantiated."""
    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {"tp": 0, "other_var": 1}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.filtering.FilteringLossWrapper",
                "predicted_variables": ["tp"],
                "target_variables": ["tp"],
                "loss": {
                    "_target_": "anemoi.training.losses.spatial.LogFFT2Distance",
                    "x_dim": 710,
                    "y_dim": 640,
                    "scalers": [],
                },
            },
        ),
        data_indices=data_indices,
    )
    assert isinstance(loss, FilteringLossWrapper)
    assert isinstance(loss.loss, FunctionalLoss)
    assert hasattr(loss.loss, "y_dim")
    assert hasattr(loss.loss, "x_dim")

    loss.set_data_indices(data_indices)
    assert hasattr(loss, "predicted_indices")

    assert loss.predicted_variables == ["tp"]

    right_shaped_pred_output_pair = (torch.ones((6, 1, 710 * 640, 2)), torch.zeros((6, 1, 710 * 640, 2)))
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
    # test instantiation with a str loss
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.filtering.FilteringLossWrapper",
                "predicted_variables": ["tp"],
                "target_variables": ["tp"],
                "loss": "anemoi.training.losses.MSELoss",
            },
        ),
        data_indices=data_indices,
    )
    loss.set_data_indices(data_indices)

    assert isinstance(loss, FilteringLossWrapper)
    assert isinstance(loss.loss, FunctionalLoss)
