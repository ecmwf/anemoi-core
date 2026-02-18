# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for LossVariableMapper with target-only variables.

Tests scenarios where some variables exist only in data.output (e.g. satellite
observations used as targets) but not in model.output.
"""

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.index_space import IndexSpace
from anemoi.training.losses.variable_mapper import LossVariableMapper


class TestScalerFilteringWithTargetOnlyVariables:
    """Tests for scaler filtering with target-only variables."""

    @pytest.fixture
    def data_indices_with_target_only(self) -> IndexCollection:
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
    def scalers_with_distinct_values(self) -> dict:
        """Scalers with distinct values per variable position for verification."""
        n_variables = 11  # 10 regular + 1 target-only
        # Each position has value 0.1, 0.2, ..., 1.1
        pressure_level_values = torch.tensor([0.1 * (i + 1) for i in range(n_variables)])
        general_variable_values = torch.tensor([1.0 * (i + 1) for i in range(n_variables)])

        return {
            "pressure_level": (4, pressure_level_values),  # dim 4 is VARIABLE
            "general_variable": (4, general_variable_values),
        }

    def test_scaler_filtering_many_variables(
        self,
        data_indices_with_target_only: IndexCollection,
        scalers_with_distinct_values: dict,
    ) -> None:
        """Filtering 10 variables from 11 produces correct scaler shape and values."""
        # All regular variables as predicted and target
        regular_variables = [f"var_{i}" for i in range(10)]

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": regular_variables,
                    "target_variables": regular_variables,
                    "scalers": ["pressure_level", "general_variable"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        assert isinstance(loss, LossVariableMapper)

        # Verify scaler shapes are filtered correctly
        # Scalers are forwarded to the inner loss, so access them there
        pressure_scaler = loss.loss.scaler.tensors["pressure_level"][1]
        general_scaler = loss.loss.scaler.tensors["general_variable"][1]

        assert pressure_scaler.shape[0] == 10, f"Expected 10 variables, got {pressure_scaler.shape[0]}"
        assert general_scaler.shape[0] == 10, f"Expected 10 variables, got {general_scaler.shape[0]}"

        # Verify the VALUES are correct (should be 0.1 to 1.0, excluding 1.1 for imerg)
        expected_pressure = torch.tensor([0.1 * (i + 1) for i in range(10)])
        expected_general = torch.tensor([1.0 * (i + 1) for i in range(10)])

        torch.testing.assert_close(pressure_scaler, expected_pressure)
        torch.testing.assert_close(general_scaler, expected_general)

    def test_scaler_filtering_single_variable(
        self,
        data_indices_with_target_only: IndexCollection,
        scalers_with_distinct_values: dict,
    ) -> None:
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

        assert isinstance(loss, LossVariableMapper)

        # Scalers are forwarded to the inner loss, so access them there
        pressure_scaler = loss.loss.scaler.tensors["pressure_level"][1]

        assert pressure_scaler.shape[0] == 1
        # var_5 is at index 5, so value should be 0.6
        torch.testing.assert_close(pressure_scaler, torch.tensor([0.6]))

    def test_scaler_filtering_preserves_variable_order(
        self,
        data_indices_with_target_only: IndexCollection,
        scalers_with_distinct_values: dict,
    ) -> None:
        """Test that filtering preserves the order specified in predicted_variables."""
        # Specify variables in reverse order
        reversed_variables = [f"var_{i}" for i in range(9, -1, -1)]  # var_9, var_8, ..., var_0

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": reversed_variables,
                    "target_variables": reversed_variables,
                    "scalers": ["pressure_level"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        # Scalers are forwarded to the inner loss, so access them there
        pressure_scaler = loss.loss.scaler.tensors["pressure_level"][1]

        # Values should be in reverse order: 1.0, 0.9, 0.8, ..., 0.1
        expected = torch.tensor([0.1 * (i + 1) for i in range(9, -1, -1)])
        torch.testing.assert_close(pressure_scaler, expected)

    def test_scaler_filtering_subset_of_variables(
        self,
        data_indices_with_target_only: IndexCollection,
        scalers_with_distinct_values: dict,
    ) -> None:
        """Test filtering to a non-contiguous subset of variables."""
        # Select only even-indexed variables
        selected_variables = [f"var_{i}" for i in range(0, 10, 2)]  # var_0, var_2, var_4, var_6, var_8

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": selected_variables,
                    "target_variables": selected_variables,
                    "scalers": ["pressure_level"],
                },
            ),
            scalers=scalers_with_distinct_values,
            data_indices=data_indices_with_target_only,
        )

        # Scalers are forwarded to the inner loss, so access them there
        pressure_scaler = loss.loss.scaler.tensors["pressure_level"][1]

        assert pressure_scaler.shape[0] == 5
        # Values for indices 0, 2, 4, 6, 8 -> 0.1, 0.3, 0.5, 0.7, 0.9
        expected = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        torch.testing.assert_close(pressure_scaler, expected)


class TestCombinedLossWithTargetOnlyVariables:
    """Tests for CombinedLoss with target-only variables."""

    @pytest.fixture
    def data_indices_with_imerg(self) -> IndexCollection:
        """IndexCollection with multiple variables and imerg as target-only."""
        data_config = {
            "data": {
                "forcing": [],
                "diagnostic": [],
                "target": ["imerg"],
            },
        }
        # Simplified: 5 variables + imerg
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
    def scalers_custom(self) -> dict:
        """Create scalers with custom values per variable."""
        n_variables = 6
        return {
            "pressure_level": (4, torch.ones(n_variables) * 2.0),
            "general_variable": (4, torch.tensor([1.0, 0.5, 0.5, 0.5, 0.8, 10.0])),
            "uniform_weight": (3, torch.ones(100)),
        }

    def test_combined_loss_with_target_only_variable(
        self,
        data_indices_with_imerg: IndexCollection,
        scalers_custom: dict,
    ) -> None:
        """CombinedLoss with one subloss using a target-only variable."""
        variables = ["tp", "t2m", "u10", "v10", "msl"]

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": variables,
                            "target_variables": variables,
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
            scalers=scalers_custom,
            data_indices=data_indices_with_imerg,
        )

        assert isinstance(loss, CombinedLoss)
        assert len(loss.losses) == 2

        # Both losses should be LossVariableMapper
        assert isinstance(loss.losses[0], LossVariableMapper)
        assert isinstance(loss.losses[1], LossVariableMapper)

        # Scalers are forwarded to the inner loss, so access them there
        # First loss: 5 variables
        first_loss_scaler = loss.losses[0].loss.scaler.tensors["pressure_level"][1]
        assert first_loss_scaler.shape[0] == 5

        # Second loss: 1 variable (tp)
        second_loss_scaler = loss.losses[1].loss.scaler.tensors["pressure_level"][1]
        assert second_loss_scaler.shape[0] == 1

    def test_combined_loss_forward_pass(self, data_indices_with_imerg: IndexCollection, scalers_custom: dict) -> None:
        """Test that forward pass works with correct tensor shapes."""
        variables = ["tp", "t2m", "u10", "v10", "msl"]
        scalers_with_grid = dict(scalers_custom)
        scalers_with_grid["uniform_grid"] = (3, torch.ones(100))

        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": variables,
                            "target_variables": variables,
                            "scalers": ["pressure_level", "uniform_weight"],
                        },
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": ["tp"],
                            "target_variables": ["imerg"],
                            "scalers": ["pressure_level", "uniform_weight"],
                        },
                    ],
                    "scalers": ["*"],
                },
            ),
            scalers=scalers_with_grid,
            data_indices=data_indices_with_imerg,
        )

        # Create test tensors: batch=2, ensemble=1, grid=100, variables=6
        n_variables = 6  # data.output size (includes imerg)
        pred = torch.randn(2, 1, 1, 100, n_variables)
        target = torch.randn(2, 1, 1, 100, n_variables)

        # Should not raise any errors
        loss_value = loss(
            pred,
            target,
            squash_mode="sum",
            pred_layout=IndexSpace.DATA_FULL,
            target_layout=IndexSpace.DATA_FULL,
        )
        assert loss_value.ndim == 0 or loss_value.shape == torch.Size([])  # scalar
        assert not torch.isnan(loss_value)
        assert loss_value > 0  # Non-zero loss


class TestScalerOverrideWithDifferentSizes:
    """Tests for ScaleTensor.update_scaler with override=True."""

    def test_update_scaler_override_allows_size_change(self) -> None:
        """Test that override=True allows changing scaler size."""
        from anemoi.training.losses.scaler_tensor import ScaleTensor

        scale = ScaleTensor(test=(0, torch.ones(103)))

        # Without override, this would fail
        scale.update_scaler("test", torch.ones(102), override=True)

        assert scale.tensors["test"][1].shape[0] == 102

    def test_update_scaler_override_preserves_values(self) -> None:
        """Test that override correctly sets the new values."""
        from anemoi.training.losses.scaler_tensor import ScaleTensor

        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        new_values = torch.tensor([10.0, 20.0, 30.0])

        scale = ScaleTensor(test=(0, original))
        scale.update_scaler("test", new_values, override=True)

        torch.testing.assert_close(scale.tensors["test"][1], new_values)


class TestLossVariableMapperSetDataIndices:
    """Test LossVariableMapper.set_data_indices with target-only variables."""

    @pytest.fixture
    def data_indices_with_target_only(self) -> IndexCollection:
        """Create IndexCollection with target-only variable and forcing gaps.

        Layout:
        - var_0, var_1: prognostic (indices 0, 1)
        - f_0, f_1: forcing (indices 2, 3) - excluded from data.output
        - var_2, var_3: prognostic (indices 4, 5)
        - f_2, f_3: forcing (indices 6, 7) - excluded from data.output
        - var_4: prognostic (index 8)
        - imerg: target-only (index 9)

        data.output.full = [0, 1, 4, 5, 8, 9] (forcings excluded)
        """
        data_config = {
            "forcing": ["f_0", "f_1", "f_2", "f_3"],
            "diagnostic": [],
            "target": ["imerg"],
        }
        name_to_index = {
            "var_0": 0,
            "var_1": 1,
            "f_0": 2,
            "f_1": 3,
            "var_2": 4,
            "var_3": 5,
            "f_2": 6,
            "f_3": 7,
            "var_4": 8,
            "imerg": 9,
        }
        return IndexCollection(DictConfig(data_config), name_to_index)

    def test_set_data_indices_target_only_variable(self, data_indices_with_target_only: IndexCollection) -> None:
        """Test that set_data_indices correctly handles target-only variables.

        Fixture layout:
        - data.output.full = [0, 1, 4, 5, 8, 9] (forcings at 2,3,6,7 excluded)
        - Reindexed positions: var_0→0, var_1→1, var_2→2, var_3→3, var_4→4, imerg→5
        """
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = LossVariableMapper(
            loss=base_loss,
            predicted_variables=["var_0"],  # From model.output
            target_variables=["imerg"],  # Target-only, not in model.output
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # predicted indices should be from model.output layout
        assert wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0]  # var_0 index in model.output

        # target_indices: imerg has name_to_index=9
        assert wrapper.target_indices_by_layout[IndexSpace.DATA_FULL] == [9]

    def test_set_data_indices_with_forcing_gaps(self) -> None:
        """Reindexing when forcing variables create gaps in data.output.

        Setup:
        - name_to_index: var_0=0, forcing=1, var_2=2, imerg=3
        - data.output.full: [0, 2, 3] (forcing excluded)
        - Positions in output tensor: var_0→0, var_2→1, imerg→2
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
        wrapper = LossVariableMapper(
            loss=base_loss,
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices)

        # predicted indices: var_0 is at position 0 in model.output
        assert wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0]

        # target_indices: imerg has name_to_index=3
        assert wrapper.target_indices_by_layout[IndexSpace.DATA_FULL] == [
            3,
        ], (
            "Expected imerg at position 2 in data.output tensor, "
            f"got {wrapper.target_indices_by_layout[IndexSpace.DATA_FULL]}"
        )

    def test_set_data_indices_same_predicted_and_target(self, data_indices_with_target_only: IndexCollection) -> None:
        """Test set_data_indices when predicted and target variables are the same.

        Fixture layout:
        - model.output.name_to_index: var_0=0, var_1=1, var_2=2, var_3=3, var_4=4
        - data.output.full = [0, 1, 4, 5, 8, 9]
        """
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = LossVariableMapper(
            loss=base_loss,
            predicted_variables=["var_0", "var_1", "var_2"],
            target_variables=["var_0", "var_1", "var_2"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # predicted indices come from model.output.name_to_index
        assert wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0, 1, 2]
        # var_0, var_1 are at positions 0, 1 in data.output.full
        # var_2 has name_to_index=4
        assert wrapper.target_indices_by_layout[IndexSpace.DATA_FULL] == [0, 1, 4]

    def test_set_data_indices_all_variables(self, data_indices_with_target_only: IndexCollection) -> None:
        """Test set_data_indices when no variables are specified (use all).

        When both are None, set_data_indices defaults to model.output.full for both
        (to ensure same length for loss computation).
        """
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = LossVariableMapper(
            loss=base_loss,
            predicted_variables=None,
            target_variables=None,
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # Both default to model.output.full (5 prognostic variables)
        # imerg is ignored though it is present as a target
        # They must have the same length for loss computation
        assert len(wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT]) == 5
        assert len(wrapper.target_indices_by_layout[IndexSpace.DATA_FULL]) == 5
        assert wrapper.predicted_variables == wrapper.target_variables

    def test_set_data_indices_defaults_target_to_predicted_order(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """Default target mapping keeps predicted variable order."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_3", "var_0"],
            target_variables=None,
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        assert wrapper.target_variables == ["var_3", "var_0"]
        assert wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [3, 0]
        assert wrapper.target_indices_by_layout[IndexSpace.DATA_FULL] == [5, 0]

    def test_set_data_indices_uses_model_output_index_order_for_defaults(self) -> None:
        """Default predicted variable order must follow model-output tensor order."""
        from anemoi.training.losses.mse import MSELoss

        data_indices = IndexCollection(
            DictConfig({"forcing": [], "diagnostic": [], "target": []}),
            {"z_var": 0, "a_var": 1, "m_var": 2},
        )
        wrapper = LossVariableMapper(loss=MSELoss(), predicted_variables=None, target_variables=None)
        wrapper.set_data_indices(data_indices)

        assert wrapper.predicted_variables == ["z_var", "a_var", "m_var"]
        assert wrapper.target_variables == ["z_var", "a_var", "m_var"]
        assert wrapper.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0, 1, 2]


class TestLossVariableMapperForward:
    """Test LossVariableMapper.forward method."""

    @pytest.fixture
    def data_indices_with_target_only(self) -> IndexCollection:
        """Create IndexCollection with target-only variable and forcing gaps.

        Layout:
        - var_0, var_1: prognostic (indices 0, 1)
        - f_0, f_1: forcing (indices 2, 3) - excluded from data.output
        - var_2, var_3: prognostic (indices 4, 5)
        - f_2, f_3: forcing (indices 6, 7) - excluded from data.output
        - var_4: prognostic (index 8)
        - imerg: target-only (index 9)

        data.output.full = [0, 1, 4, 5, 8, 9] (forcings excluded)
        Total tensor size: 6 variables in data.output
        """
        data_config = {
            "forcing": ["f_0", "f_1", "f_2", "f_3"],
            "diagnostic": [],
            "target": ["imerg"],
        }
        name_to_index = {
            "var_0": 0,
            "var_1": 1,
            "f_0": 2,
            "f_1": 3,
            "var_2": 4,
            "var_3": 5,
            "f_2": 6,
            "f_3": 7,
            "var_4": 8,
            "imerg": 9,
        }
        return IndexCollection(DictConfig(data_config), name_to_index)

    def test_forward_filters_correctly(self, data_indices_with_target_only: IndexCollection) -> None:
        """Test that forward correctly filters predictions and targets.

        Fixture layout:
        - data.output.full = [0, 1, 4, 5, 8, 9] (tensor size 6)
        """
        from anemoi.training.losses.mse import MSELoss

        base_loss = MSELoss()
        wrapper = LossVariableMapper(
            loss=base_loss,
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # Create tensors with zeros, then set specific values
        # Shape: (batch=2, ensemble=1, grid=8, variables=6) - 6 variables in data.output
        pred = torch.zeros(2, 1, 1, 8, 6)
        target = torch.zeros(2, 1, 1, 8, len(data_indices_with_target_only.name_to_index))

        # Set values only in the variables we expect to be used
        pred[..., 0] = 1.0  # var_0 in predictions (position 0)
        target[..., 9] = 2.0  # imerg in targets

        # Set different values in other positions to verify they're not used
        pred[..., 1:] = 100.0
        target[..., :9] = 100.0

        # Forward should compare pred[..., 0]=1.0 with target[..., 9]=2.0
        # MSE per element = (1.0 - 2.0)² = 1.0
        # Loss reduction: sum over grid (8 points), avg over batch/ensemble
        loss = wrapper(
            pred,
            target,
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )

        torch.testing.assert_close(loss, torch.tensor(8.0))

    def test_forward_requires_explicit_layout_kwargs(self, data_indices_with_target_only: IndexCollection) -> None:
        """Forward should fail fast when layout metadata is missing."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        pred = torch.zeros(1, 1, 1, 2, 6)
        target = torch.zeros(1, 1, 1, 2, len(data_indices_with_target_only.name_to_index))

        with pytest.raises(ValueError, match="requires both 'pred_layout' and 'target_layout'"):
            wrapper(pred, target)

    def test_forward_remaps_scaler_indices_from_layout_space_to_filtered_space(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """Variable scaler_indices should be remapped from layout indices to filtered tensor indices."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0", "var_3"],
            target_variables=["var_0", "var_3"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        # Add required GRID scaler and a VARIABLE scaler to exercise scaler_indices in BaseLoss.scale.
        wrapper.add_scaler(dimension=3, scaler=torch.ones(8), name="grid_uniform")
        wrapper.add_scaler(dimension=4, scaler=torch.tensor([2.0, 3.0]), name="variable_weight")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0  # var_0 in data.output space
        pred[..., 3] = 1.0  # var_3 in data.output space

        # Request only global index 3 (var_3). Wrapper should remap to local filtered index 1.
        loss = wrapper(
            pred,
            target,
            scaler_indices=(..., [3]),
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )

        # MSE=(1-0)^2=1, weighted by variable_weight[1]=3, summed over 8 grid points.
        torch.testing.assert_close(loss, torch.tensor(24.0))

    def test_forward_with_empty_remapped_scaler_indices_returns_zero(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """If scaler_indices selects no filtered variables, the wrapper should return zero instead of crashing."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["var_0"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)
        wrapper.add_scaler(dimension=3, scaler=torch.ones(8), name="grid_uniform")
        wrapper.add_scaler(dimension=4, scaler=torch.tensor([2.0]), name="variable_weight")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0

        # Global index 3 is outside the filtered selection ["var_0"] for DATA_OUTPUT layout.
        loss = wrapper(
            pred,
            target,
            scaler_indices=(..., [3]),
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_add_scaler_filters_variable_axis_for_mixed_dim_scalers(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """Mixed-dimension scalers containing VARIABLE must be subset to predicted vars."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        n_model_vars = len(data_indices_with_target_only.model.output.full)
        mixed_scaler = torch.ones(2, 8, n_model_vars)  # (BATCH, GRID, VARIABLE)
        wrapper.add_scaler(dimension=(0, 3, 4), scaler=mixed_scaler, name="batch_grid_variable")

        dims, scaler = wrapper.loss.scaler.tensors["batch_grid_variable"]
        assert dims == (0, 3, 4)
        assert scaler.shape == (2, 8, 1)

    def test_add_scaler_keeps_broadcast_variable_axis_for_mixed_dim_scalers(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """Broadcast VARIABLE axis (size 1) must not be index-selected."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        mixed_scaler = torch.ones(2, 8, 1)  # (BATCH, GRID, VARIABLE[broadcast])
        wrapper.add_scaler(dimension=(0, 3, 4), scaler=mixed_scaler, name="broadcast_variable")

        dims, scaler = wrapper.loss.scaler.tensors["broadcast_variable"]
        assert dims == (0, 3, 4)
        assert scaler.shape == (2, 8, 1)

    def test_add_scaler_filters_data_full_variable_axis_with_forcing_gaps(self) -> None:
        """VARIABLE scalers in data-full space should use DATA_FULL indices when sizes match."""
        from anemoi.training.losses.mse import MSELoss

        data_indices = IndexCollection(
            DictConfig({"data": {"forcing": ["f0"], "diagnostic": [], "target": []}}),
            {"f0": 0, "var_0": 1, "var_1": 2},
        )
        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_1"],
            target_variables=["var_1"],
        )
        wrapper.set_data_indices(data_indices)

        data_full_scaler = torch.tensor([10.0, 20.0, 30.0])  # data-full order
        wrapper.add_scaler(dimension=4, scaler=data_full_scaler, name="data_full_variable")

        dims, scaler = wrapper.loss.scaler.tensors["data_full_variable"]
        assert dims == (4,)
        torch.testing.assert_close(scaler, torch.tensor([30.0]))

    def test_update_scaler_preserves_variable_axis_filtering(
        self,
        data_indices_with_target_only: IndexCollection,
    ) -> None:
        """Updating variable scalers should keep the filtered variable subset."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_2"],
            target_variables=["var_2"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        full_var_scaler = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        wrapper.add_scaler(dimension=4, scaler=full_var_scaler, name="variable_weights")
        torch.testing.assert_close(wrapper.loss.scaler.tensors["variable_weights"][1], torch.tensor([3.0]))

        updated_full_var_scaler = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        wrapper.update_scaler("variable_weights", updated_full_var_scaler, override=True)

        dims, scaler = wrapper.loss.scaler.tensors["variable_weights"]
        assert dims == (4,)
        assert scaler.shape == (1,)
        torch.testing.assert_close(scaler, torch.tensor([30.0]))

    def test_forward_rejects_unavailable_target_layout(self, data_indices_with_target_only: IndexCollection) -> None:
        """Target-only mappings are not available in model-output layout."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        pred = torch.zeros(1, 1, 1, 2, 6)
        target = torch.zeros(1, 1, 1, 2, len(data_indices_with_target_only.name_to_index))

        with pytest.raises(ValueError, match="target_layout 'model_output' is not available"):
            wrapper(
                pred,
                target,
                pred_layout=IndexSpace.DATA_OUTPUT,
                target_layout=IndexSpace.MODEL_OUTPUT,
            )

    def test_forward_rejects_invalid_layout_value(self, data_indices_with_target_only: IndexCollection) -> None:
        """Invalid layout names should fail with a clear message."""
        from anemoi.training.losses.mse import MSELoss

        wrapper = LossVariableMapper(
            loss=MSELoss(),
            predicted_variables=["var_0"],
            target_variables=["imerg"],
        )
        wrapper.set_data_indices(data_indices_with_target_only)

        pred = torch.zeros(1, 1, 1, 2, 6)
        target = torch.zeros(1, 1, 1, 2, len(data_indices_with_target_only.name_to_index))

        with pytest.raises(ValueError, match="Invalid pred_layout"):
            wrapper(
                pred,
                target,
                pred_layout="not_a_layout",
                target_layout=IndexSpace.DATA_FULL,
            )

    def test_forward_multiple_variables_and_target_only(self, data_indices_with_target_only: IndexCollection) -> None:
        """Test CombinedLoss with prognostic variables in first loss and target-only in second.

        Fixture layout:
        - data.output.full = [0, 1, 4, 5, 8, 9] (tensor size 6)
        - model.output prognostic variables: var_0=0, var_1=1, var_2=2, var_3=3, var_4=4

        This test also verifies that scalers are actually applied during loss computation
        by comparing loss values with and without scalers.
        """
        # First loss: MSE on prognostic variables
        # Second loss: MSE on var_0 predicted vs imerg target
        prognostic_variables = ["var_0", "var_1", "var_2", "var_3", "var_4"]

        # First, compute loss WITHOUT scalers
        loss_no_scalers = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": prognostic_variables,
                            "target_variables": prognostic_variables,
                        },
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": ["var_0"],
                            "target_variables": ["imerg"],
                        },
                    ],
                    "loss_weights": [1.0, 1.0],
                },
            ),
            data_indices=data_indices_with_target_only,
        )

        # Now compute loss WITH scalers (2x weight for all variables)
        n_vars = len(data_indices_with_target_only.name_to_index)
        scalers = {"double_weight": (4, torch.ones(n_vars) * 2.0), "uniform_weight": (3, torch.ones(4))}

        loss_with_scalers = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.CombinedLoss",
                    "losses": [
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": prognostic_variables,
                            "target_variables": prognostic_variables,
                            "scalers": ["double_weight", "uniform_weight"],
                        },
                        {
                            "_target_": "anemoi.training.losses.MSELoss",
                            "predicted_variables": ["var_0"],
                            "target_variables": ["imerg"],
                            "scalers": ["double_weight", "uniform_weight"],
                        },
                    ],
                    "loss_weights": [1.0, 1.0],
                    "scalers": ["*"],
                },
            ),
            scalers=scalers,
            data_indices=data_indices_with_target_only,
        )

        assert isinstance(loss_no_scalers, CombinedLoss)
        assert isinstance(loss_with_scalers, CombinedLoss)
        assert len(loss_no_scalers.losses) == 2
        assert len(loss_with_scalers.losses) == 2
        assert isinstance(loss_no_scalers.losses[0], LossVariableMapper)
        assert isinstance(loss_with_scalers.losses[0], LossVariableMapper)

        # Verify first loss uses 5 prognostic variables
        assert loss_no_scalers.losses[0].predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0, 1, 2, 3, 4]
        assert loss_no_scalers.losses[0].target_indices_by_layout[IndexSpace.DATA_FULL] == [0, 1, 4, 5, 8]
        assert len(loss_no_scalers.losses[0].predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT]) == len(
            prognostic_variables,
        )

        # Verify indices for second loss
        assert loss_no_scalers.losses[1].predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0]
        assert loss_no_scalers.losses[1].target_indices_by_layout[IndexSpace.DATA_FULL] == [9]

        # Create tensors with controlled values
        # Shape: (batch=2, ensemble=1, grid=4, variables=6) - 6 variables in data.output
        pred = torch.zeros(2, 1, 1, 4, 6)
        target = torch.zeros(2, 1, 1, 4, len(data_indices_with_target_only.name_to_index))

        # Set prediction values for prognostic variables
        for j, i in enumerate(data_indices_with_target_only.model.output.full):
            pred[..., i] = j + 1.0  # var_0

        # Set target values for first loss
        for i in data_indices_with_target_only.data.output.full:
            target[..., i] = 2.0

        # Set imerg target for second loss
        target[..., 9] = 5.0  # imerg target

        # Compute combined loss without scalers
        loss_value_no_scalers = loss_no_scalers(
            pred,
            target,
            squash_mode="sum",
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )

        # First loss: MSE on prognostic variables
        # (1-2)² + (2-2)² + (3-2)² + (4-2)² + (5-2)² = 1 + 0 + 1 + 4 + 9 = 15 per grid point
        # summed over grid = 15 * 4 = 60
        # Second loss: MSE on var_0 vs imerg
        # (1-5)² = 16 per grid point, summed = 16 * 4 = 64
        # Combined (weight 1.0 each): 60 + 64 = 124
        torch.testing.assert_close(loss_value_no_scalers, torch.tensor(124.0))

        # Compute combined loss WITH scalers
        loss_value_with_scalers = loss_with_scalers(
            pred,
            target,
            squash_mode="sum",
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )

        # With 2x scaler applied, loss should be 2x: 124 * 2 = 248
        torch.testing.assert_close(loss_value_with_scalers, torch.tensor(248.0))
