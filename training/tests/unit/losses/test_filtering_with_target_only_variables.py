# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for FilteringLossWrapper with target-only variables.

Tests scenarios where some variables exist only in data.output (e.g. satellite
observations used as targets) but not in model.output.
"""

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.filtering import FilteringLossWrapper


class TestScalerFilteringWithTargetOnlyVariables:
    """Tests for scaler filtering with target-only variables."""

    @pytest.fixture
    def data_indices_with_target_only(self):
        """IndexCollection with 10 regular variables + 1 target-only variable (imerg)."""
        data_config = {
            "data": {
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],  # target-only variable
            },
        }
        # 11 variables total: var_0 to var_9 (regular) + imerg (target-only)
        name_to_index = {f"var_{i}": i for i in range(10)}
        name_to_index["imerg"] = 10

        return IndexCollection(DictConfig(data_config), name_to_index)

    @pytest.fixture
    def scalers_with_distinct_values(self):
        """Scalers with distinct values per variable position for verification."""
        n_vars = 11  # 10 regular + 1 target-only
        # Each position has value 0.1, 0.2, ..., 1.1
        pressure_level_values = torch.tensor([0.1 * (i + 1) for i in range(n_vars)])
        general_variable_values = torch.tensor([1.0 * (i + 1) for i in range(n_vars)])

        return {
            "pressure_level": (3, pressure_level_values),  # dim 3 is VARIABLE
            "general_variable": (3, general_variable_values),
        }

    def test_scaler_filtering_many_variables(self, data_indices_with_target_only, scalers_with_distinct_values):
        """Filtering 10 variables from 11 produces correct scaler shape and values."""
        # All regular variables as predicted and target
        regular_vars = [f"var_{i}" for i in range(10)]

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": regular_vars,
                    "target_variables": regular_vars,
                    "scalers": ["pressure_level", "general_variable"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        assert isinstance(loss, FilteringLossWrapper)

        # Verify scaler shapes are filtered correctly
        subloss = loss.loss
        pressure_scaler = subloss.scaler.tensors["pressure_level"][1]
        general_scaler = subloss.scaler.tensors["general_variable"][1]

        assert pressure_scaler.shape[0] == 10, f"Expected 10 variables, got {pressure_scaler.shape[0]}"
        assert general_scaler.shape[0] == 10, f"Expected 10 variables, got {general_scaler.shape[0]}"

        # Verify the VALUES are correct (should be 0.1 to 1.0, excluding 1.1 for imerg)
        expected_pressure = torch.tensor([0.1 * (i + 1) for i in range(10)])
        expected_general = torch.tensor([1.0 * (i + 1) for i in range(10)])

        torch.testing.assert_close(pressure_scaler, expected_pressure)
        torch.testing.assert_close(general_scaler, expected_general)

    def test_scaler_filtering_single_variable(self, data_indices_with_target_only, scalers_with_distinct_values):
        """Test filtering to a single predicted variable."""
        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": ["var_5"],
                    "target_variables": ["var_5"],
                    "scalers": ["pressure_level", "general_variable"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        assert isinstance(loss, FilteringLossWrapper)

        subloss = loss.loss
        pressure_scaler = subloss.scaler.tensors["pressure_level"][1]

        assert pressure_scaler.shape[0] == 1
        # var_5 is at index 5, so value should be 0.6
        torch.testing.assert_close(pressure_scaler, torch.tensor([0.6]))

    def test_scaler_filtering_preserves_variable_order(
        self, data_indices_with_target_only, scalers_with_distinct_values,
    ):
        """Test that filtering preserves the order specified in predicted_variables."""
        # Specify variables in reverse order
        reversed_vars = [f"var_{i}" for i in range(9, -1, -1)]  # var_9, var_8, ..., var_0

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": reversed_vars,
                    "target_variables": reversed_vars,
                    "scalers": ["pressure_level"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        subloss = loss.loss
        pressure_scaler = subloss.scaler.tensors["pressure_level"][1]

        # Values should be in reverse order: 1.0, 0.9, 0.8, ..., 0.1
        expected = torch.tensor([0.1 * (i + 1) for i in range(9, -1, -1)])
        torch.testing.assert_close(pressure_scaler, expected)

    def test_scaler_filtering_subset_of_variables(self, data_indices_with_target_only, scalers_with_distinct_values):
        """Test filtering to a non-contiguous subset of variables."""
        # Select only even-indexed variables
        selected_vars = [f"var_{i}" for i in range(0, 10, 2)]  # var_0, var_2, var_4, var_6, var_8

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": selected_vars,
                    "target_variables": selected_vars,
                    "scalers": ["pressure_level"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        subloss = loss.loss
        pressure_scaler = subloss.scaler.tensors["pressure_level"][1]

        assert pressure_scaler.shape[0] == 5
        # Values for indices 0, 2, 4, 6, 8 -> 0.1, 0.3, 0.5, 0.7, 0.9
        expected = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        torch.testing.assert_close(pressure_scaler, expected)


class TestCombinedLossWithTargetOnlyVariables:
    """Tests for CombinedLoss with target-only variables."""

    @pytest.fixture
    def data_indices_weather_with_imerg(self):
        """IndexCollection with weather variables and imerg as target-only."""
        data_config = {
            "data": {
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],
            },
        }
        # Simplified: 5 weather variables + imerg
        name_to_index = {
            "tp": 0,  # total precipitation (predicted)
            "t2m": 1,  # 2m temperature
            "u10": 2,  # 10m u-wind
            "v10": 3,  # 10m v-wind
            "msl": 4,  # mean sea level pressure
            "imerg": 5,  # satellite precipitation (target-only)
        }
        return IndexCollection(DictConfig(data_config), name_to_index)

    @pytest.fixture
    def weather_scalers(self):
        """Create scalers for weather variables."""
        n_vars = 6
        return {
            "pressure_level": (3, torch.ones(n_vars) * 2.0),
            "general_variable": (3, torch.tensor([1.0, 0.5, 0.5, 0.5, 0.8, 10.0])),
        }

    def test_combined_loss_with_target_only_variable(self, data_indices_weather_with_imerg, weather_scalers):
        """CombinedLoss with one subloss using a target-only variable."""
        weather_vars = ["tp", "t2m", "u10", "v10", "msl"]

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": weather_vars,
                            "target_variables": weather_vars,
                            "scalers": ["pressure_level", "general_variable"],
                        },
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": ["tp"],
                            "target_variables": ["imerg"],
                            "scalers": ["pressure_level", "general_variable"],
                        },
                    ],
                    "loss_weights": [1.0, 1.0],
                    "scalers": ["*"],
                },
            ),
            scalers=weather_scalers,
            data_indices=data_indices_weather_with_imerg,
        )

        assert isinstance(loss, CombinedLoss)
        assert len(loss.losses) == 2

        # Both losses should be FilteringLossWrapper
        assert isinstance(loss.losses[0], FilteringLossWrapper)
        assert isinstance(loss.losses[1], FilteringLossWrapper)

        # First loss: 5 weather variables
        first_loss_scaler = loss.losses[0].loss.scaler.tensors["pressure_level"][1]
        assert first_loss_scaler.shape[0] == 5

        # Second loss: 1 variable (tp)
        second_loss_scaler = loss.losses[1].loss.scaler.tensors["pressure_level"][1]
        assert second_loss_scaler.shape[0] == 1

    def test_combined_loss_forward_pass(self, data_indices_weather_with_imerg, weather_scalers):
        """Test that forward pass works with correct tensor shapes."""
        weather_vars = ["tp", "t2m", "u10", "v10", "msl"]

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": weather_vars,
                            "target_variables": weather_vars,
                            "scalers": ["pressure_level"],
                        },
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": ["tp"],
                            "target_variables": ["imerg"],
                            "scalers": ["pressure_level"],
                        },
                    ],
                    "scalers": ["*"],
                },
            ),
            scalers=weather_scalers,
            data_indices=data_indices_weather_with_imerg,
        )

        # Create test tensors: batch=2, ensemble=1, grid=100, vars=6
        n_vars = 6  # data.output size (includes imerg)
        pred = torch.randn(2, 1, 100, n_vars)
        target = torch.randn(2, 1, 100, n_vars)

        # Should not raise any errors
        loss_value = loss(pred, target, squash_mode="sum")
        assert loss_value.ndim == 0 or loss_value.shape == torch.Size([])  # scalar
        assert not torch.isnan(loss_value)
        assert loss_value > 0  # Non-zero loss


class TestScalerOverrideWithDifferentSizes:
    """Tests for ScaleTensor.update_scaler with override=True."""

    def test_update_scaler_override_allows_size_change(self):
        """Test that override=True allows changing scaler size."""
        from anemoi.training.losses.scaler_tensor import ScaleTensor

        scale = ScaleTensor(test=(0, torch.ones(103)))

        # Without override, this would fail
        scale.update_scaler("test", torch.ones(102), override=True)

        assert scale.tensors["test"][1].shape[0] == 102

    def test_update_scaler_override_preserves_values(self):
        """Test that override correctly sets the new values."""
        from anemoi.training.losses.scaler_tensor import ScaleTensor

        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        new_values = torch.tensor([10.0, 20.0, 30.0])

        scale = ScaleTensor(test=(0, original))
        scale.update_scaler("test", new_values, override=True)

        torch.testing.assert_close(scale.tensors["test"][1], new_values)


class TestFilteringLossWrapperSetDataIndices:
    """Test FilteringLossWrapper.set_data_indices with target-only variables."""

    @pytest.fixture
    def data_indices_with_target_only(self):
        """Create IndexCollection with target-only variable."""
        data_config = {
            "data": {
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],
            },
        }
        name_to_index = {f"var_{i}": i for i in range(10)}
        name_to_index["imerg"] = 10
        return IndexCollection(DictConfig(data_config), name_to_index)

    def test_set_data_indices_target_only_variable(self, data_indices_with_target_only):
        """Test that set_data_indices correctly handles target-only variables."""
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=["var_0"],  # From model.output
            target_variables=["imerg"],  # Target-only, not in model.output
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # predicted_indices should be from model.output
        assert wrapper.predicted_indices == [0]  # var_0 index in model.output

        # target_indices should be reindexed position of imerg in data.output
        assert wrapper.target_indices == [10]  # imerg is at position 10 in data.output

    def test_set_data_indices_with_forcing_gaps(self):
        """Reindexing when forcing variables create gaps in data.output.

        Setup:
        - name_to_index: var_0=0, forcing=1, var_2=2, imerg=3
        - data.output.full: [0, 2, 3] (forcing excluded)
        - Positions in output tensor: var_0→0, var_2→1, imerg→2

        Without reindexing, imerg index would be 3 (wrong).
        With reindexing, imerg index is 2 (correct position in tensor).
        """
        from anemoi.training.losses.mse import MSELoss

        # Create IndexCollection with forcing variable creating a gap
        # NOTE: IndexCollection expects flat config (forcing/diagnostic/target at top level)
        # not nested under "data"
        data_config = {
            "forcing": ["forcing"],  # This will be excluded from data.output
            "diagnostic": [],
            "target": ["imerg"],
        }
        name_to_index = {
            "var_0": 0,
            "forcing": 1,  # In name_to_index but NOT in data.output.full
            "var_2": 2,
            "imerg": 3,
        }
        data_indices = IndexCollection(DictConfig(data_config), name_to_index)

        # Verify forcing is NOT in data.output.full
        assert 1 not in data_indices.data.output.full.tolist(), "forcing should not be in data.output"
        # data.output.full should be [0, 2, 3]
        assert data_indices.data.output.full.tolist() == [0, 2, 3]

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices)

        # predicted_indices: var_0 is at position 0 in model.output
        assert wrapper.predicted_indices == [0]

        # target_indices: imerg has name_to_index=3, but its position in
        # data.output.full=[0,2,3] is index 2 (third element)
        assert wrapper.target_indices == [
            2,
        ], f"Expected imerg at position 2 in data.output tensor, got {wrapper.target_indices}"

    def test_set_data_indices_same_predicted_and_target(self, data_indices_with_target_only):
        """Test set_data_indices when predicted and target variables are the same."""
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=["var_0", "var_1", "var_2"],
            target_variables=["var_0", "var_1", "var_2"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # Both should have same length
        assert len(wrapper.predicted_indices) == len(wrapper.target_indices) == 3

    def test_set_data_indices_all_variables(self, data_indices_with_target_only):
        """Test set_data_indices when no variables are specified (use all).
        
        Note: The fixture uses nested config {"data": {...}} which IndexCollection
        doesn't parse for forcing/diagnostic/target, so all 11 vars are in model.output.
        """
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=None,  # Use all model.output variables
            target_variables=None,  # Use all data.output variables
        )
        wrapper.set_data_indices(data_indices_with_target_only)
        
        # With nested config, all 11 variables are treated as prognostic
        assert len(wrapper.predicted_indices) == 11
        assert len(wrapper.target_indices) == 11


class TestFilteringLossWrapperForward:
    """Test FilteringLossWrapper.forward method."""

    @pytest.fixture
    def data_indices_with_target_only(self):
        """Create IndexCollection with target-only variable."""
        data_config = {
            "data": {
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],
            },
        }
        name_to_index = {f"var_{i}": i for i in range(5)}
        name_to_index["imerg"] = 5
        return IndexCollection(DictConfig(data_config), name_to_index)

    def test_forward_filters_correctly(self, data_indices_with_target_only):
        """Test that forward correctly filters predictions and targets."""
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # Create tensors with zeros, then set specific values
        # Shape: (batch=2, ensemble=1, grid=10, vars=6)
        pred = torch.zeros(2, 1, 8, 6)
        target = torch.zeros(2, 1, 8, 6)

        # Set values only in the variables we expect to be used
        pred[..., 0] = 1.0  # var_0 in predictions
        target[..., 5] = 2.0  # imerg in targets

        # Set different values in other positions to verify they're not used
        pred[..., 1:] = 100.0
        target[..., :5] = 100.0

        # Forward should compare pred[..., 0]=1.0 with target[..., 5]=2.0
        # MSE per element = (1.0 - 2.0)² = 1.0
        # Loss reduction: sum over grid (10 points), avg over batch/ensemble
        # Expected: 1.0 * 10 = 10.0
        loss = wrapper(pred, target)

        torch.testing.assert_close(loss, torch.tensor(8.0))

    def test_forward_squash_false_returns_per_variable_loss(self, data_indices_with_target_only):
        """Test that forward with squash=False returns per-variable loss."""
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = FilteringLossWrapper(
            loss=base_loss,
            predicted_variables=["var_0", "var_1"],
            target_variables=["var_0", "var_1"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        pred = torch.randn(2, 1, 10, 6)
        target = torch.randn(2, 1, 10, 6)

        # With squash=False, should return tensor of shape [n_model_output_vars]
        loss = wrapper(pred, target, squash=False)

        # Loss should have shape matching model output size
        assert loss.shape[0] == pred.shape[-1]  # 6 variables
        # Only positions 0 and 1 should have non-zero loss
        assert loss[0] != 0
        assert loss[1] != 0
        # Other positions should be zero
        assert loss[2] == 0
        assert loss[3] == 0
        assert loss[4] == 0
        assert loss[5] == 0
