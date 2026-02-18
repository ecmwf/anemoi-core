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
from anemoi.training.losses.index_space import IndexSpace
from anemoi.training.losses.variable_mapper import LossVariableMapper


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


def test_combined_loss_top_level_scaler_exclusion_propagates_to_sublosses() -> None:
    """Top-level exclusions like ['*', '!name'] must be respected by sub-losses."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "scalers": ["*", "!s2"],
                "loss_weights": [1.0, 1.0],
            },
        ),
        scalers={"s1": (-1, torch.ones(2)), "s2": (-1, torch.ones(2))},
    )

    assert isinstance(loss, CombinedLoss)
    assert "s1" in loss.losses[0].scaler
    assert "s1" in loss.losses[1].scaler
    assert "s2" not in loss.losses[0].scaler
    assert "s2" not in loss.losses[1].scaler


def test_combined_loss_without_top_level_scalers_does_not_auto_apply_scalers() -> None:
    """Without top-level `scalers`, CombinedLoss should not auto-inject scaler universe."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "loss_weights": [1.0, 1.0],
            },
        ),
        scalers={"s1": (-1, torch.ones(2)), "s2": (-1, torch.ones(2))},
    )

    assert isinstance(loss, CombinedLoss)
    assert "s1" not in loss.losses[0].scaler
    assert "s1" not in loss.losses[1].scaler
    assert "s2" not in loss.losses[0].scaler
    assert "s2" not in loss.losses[1].scaler


def test_combined_loss_with_data_indices_and_filtering() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {"tp": 0, "other_var": 1}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    tensordim = (2, 1, 1, 4, 2)
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
    assert isinstance(loss.losses[0], LossVariableMapper)
    assert loss.losses[0].predicted_variables == ["tp"]
    assert loss.losses[0].target_variables == ["tp"]
    loss_value = loss(
        torch.ones(tensordim),
        torch.zeros(tensordim),
        squash_mode="sum",
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    assert loss_value == torch.tensor(8.0)  # MSE=1 for tp only (*4 grid points),
    # 0.5*MAE=1 (*4 grid points *2 vars) => 4+4=8


def test_combined_loss_filtered_and_unfiltered_with_scalers() -> None:
    """Test CombinedLoss with one filtered loss (tp only) and one unfiltered loss (all vars) with scalers.

    This matches the real use case:
    - First loss: filters to "tp" variable only with scalers
    - Second loss: computes on all variables with scalers
    """
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.utils import print_variable_scaling

    # Setup: Lots of variables total, but we'll use 3 for simplicity
    n_vars = 3
    data_config = {"data": {"forcing": [], "diagnostic": []}}
    name_to_index = {
        "var1": 0,
        "var2": 1,
        "tp": 2,
    }
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)

    # Create scalers with variable dimension
    scaler_pressure = (4, torch.ones(n_vars) * 2.0)  # shape [3], dim 4 is VARIABLE
    scaler_general = (4, torch.ones(n_vars) * 0.5)
    scaler_weights = (3, torch.ones(4))

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MSELoss",
                        "predicted_variables": ["tp"],
                        "target_variables": ["tp"],
                        "scalers": ["pressure_level", "general_variable", "uniform_weight"],
                    },
                    {
                        "_target_": "anemoi.training.losses.MSELoss",
                        # No variable filtering - uses all variables
                        "scalers": ["pressure_level", "general_variable", "uniform_weight"],
                    },
                ],
                "loss_weights": [1.0, 1.0],
                "scalers": ["*"],
            },
        ),
        scalers={
            "pressure_level": scaler_pressure,
            "general_variable": scaler_general,
            "uniform_weight": scaler_weights,
        },
        data_indices=data_indices,
    )

    assert isinstance(loss, CombinedLoss)
    # First loss should be wrapped (has predicted_variables)
    assert isinstance(loss.losses[0], LossVariableMapper)
    assert loss.losses[0].predicted_variables == ["tp"]

    # Test forward pass with compatible tensor shapes
    batch_size, ensemble, grid_points = 1, 1, 4
    pred = torch.ones(batch_size, 1, ensemble, grid_points, n_vars)
    target = torch.zeros(batch_size, 1, ensemble, grid_points, n_vars)

    # Should not raise shape mismatch errors
    loss_value = loss(
        pred,
        target,
        squash_mode="sum",
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    assert loss_value.item() > 0  # Non-zero loss since pred != target

    # Should not raise IndexError - print_variable_scaling should work with filtered losses
    scaling_values = print_variable_scaling(loss, data_indices)
    assert "tp" in scaling_values["LossVariableMapper"]  # The filtered variable should be in the output
    assert "LossVariableMapper_2" in scaling_values  # duplicate wrapper names must be disambiguated


def test_combined_loss_update_scaler_preserves_variable_mapper_remapping() -> None:
    """CombinedLoss must update sub-loss scalers through wrappers, not through raw scaler objects."""
    from anemoi.models.data_indices.collection import IndexCollection

    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"tp": 0, "u": 1})

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MSELoss",
                        "predicted_variables": ["tp"],
                        "target_variables": ["tp"],
                        "scalers": ["grid_uniform", "dynamic"],
                    },
                ],
                "loss_weights": [1.0],
                "scalers": ["*"],
            },
        ),
        scalers={
            "grid_uniform": (3, torch.ones(4)),
            "dynamic": (4, torch.ones(2)),
        },
        data_indices=data_indices,
    )

    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.losses[0], LossVariableMapper)
    # Initial add_scaler path should filter VARIABLE axis from 2 vars -> 1 var ("tp")
    torch.testing.assert_close(loss.losses[0].loss.scaler.tensors["dynamic"][1], torch.tensor([1.0]))

    # Update with full-width variable scaler should still be filtered to selected variables.
    loss.update_scaler("dynamic", torch.tensor([3.0, 5.0]), override=True)
    updated = loss.losses[0].loss.scaler.tensors["dynamic"][1]
    assert updated.shape == (1,)
    torch.testing.assert_close(updated, torch.tensor([3.0]))

    pred = torch.ones(1, 1, 1, 4, 2)
    target = torch.zeros(1, 1, 1, 4, 2)
    out = loss(
        pred,
        target,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    assert torch.isfinite(out)
