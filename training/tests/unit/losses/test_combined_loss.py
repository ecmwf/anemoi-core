# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.filtering import FilteringLossWrapper


def test_combined_loss() -> None:
    """Test the combined loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "scalers": ["test"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" in loss.losses[1].scaler


def test_combined_loss_invalid_loss_weights() -> None:
    """Test the combined loss function with invalid loss weights."""
    with pytest.raises(InstantiationException):
        get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.combined.CombinedLoss",
                    "losses": [
                        {"_target_": "anemoi.training.losses.MSELoss"},
                        {"_target_": "anemoi.training.losses.MAELoss"},
                    ],
                    "scalers": ["test"],
                    "loss_weights": [1.0, 0.5, 1],
                },
            ),
            scalers={"test": (-1, torch.ones(2))},
        )


def test_combined_loss_equal_weighting() -> None:
    """Test equal weighting when not given."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
            },
        ),
        scalers={},
    )
    assert all(weight == 1.0 for weight in loss.loss_weights)


def test_combined_loss_seperate_scalers() -> None:
    """Test that scalers are passed to the correct loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["test"]},
                    {"_target_": "anemoi.training.losses.MAELoss", "scalers": ["test2"]},
                ],
                "scalers": ["test", "test2"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2)), "test2": (-1, torch.ones(2))},
    )
    assert isinstance(loss, CombinedLoss)

    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler
    assert "test2" not in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" not in loss.losses[1].scaler
    assert "test2" in loss.losses[1].scaler


def test_combined_loss_with_data_indices_but_no_filtering() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {"tp": 0, "other_var": 1}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    # first no filter, but we add data_indices which should trigger the wrapper
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "loss_weights": [1.0, 0.5],
            },
        ),
        data_indices=data_indices,
    )
    assert isinstance(loss, CombinedLoss)
    for i in range(len(loss.losses)):
        # get_loss_function with data_indices triggers FilteringLossWrapper
        assert isinstance(loss.losses[i], FilteringLossWrapper)
        assert loss.losses[i].predicted_variables == ["tp", "other_var"]
        assert loss.losses[i].target_variables == ["tp", "other_var"]
    tensordim = (2, 1, 4, 2)
    loss_value = loss(
        torch.ones(tensordim),
        torch.zeros(tensordim),
        squash_mode="sum",
    )
    assert loss_value == torch.tensor(12.0)  # MSE=1 + 0.5*MAE=1 (*4 grid points * 2 vars) => 8+4=12


def test_combined_loss_with_data_indices_and_filtering() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {"tp": 0, "other_var": 1}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    tensordim = (2, 1, 4, 2)
    # now filtering inside the CombinedLoss
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MSELoss",
                        "predicted_variables": ["tp"],
                        "target_variables": ["tp"],
                    },
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "loss_weights": [1.0, 0.5],
            },
        ),
        data_indices=data_indices,
    )
    assert isinstance(loss, CombinedLoss)
    for i in range(len(loss.losses)):
        # get_loss_function with data_indices triggers FilteringLossWrapper
        assert isinstance(loss.losses[i], FilteringLossWrapper)
    assert loss.losses[0].predicted_variables == ["tp"]
    assert loss.losses[0].target_variables == ["tp"]
    assert loss.losses[1].predicted_variables == ["tp", "other_var"]
    assert loss.losses[1].target_variables == ["tp", "other_var"]
    loss_value = loss(
        torch.ones(tensordim),
        torch.zeros(tensordim),
        squash_mode="sum",
    )
    assert loss_value == torch.tensor(8.0)  # MSE=1 for tp only (*4 grid points),
    # 0.5*MAE=1 (*4 grid points *2 vars) => 4+4=8
