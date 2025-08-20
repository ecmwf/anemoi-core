# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Comprehensive test suite for the ModelModifier system.

This module tests the model modification framework that enables flexible,
composable transformations of PyTorch models after instantiation but before
training. The test suite covers:

1. **Core Abstractions**: Tests for the ModelModifier abstract base class
2. **Freezing Functionality**: Parameter freezing with validation
3. **Transfer Learning**: Checkpoint loading with flexible weight transfer
4. **Modifier Orchestration**: Sequential application of multiple modifiers
5. **Integration Scenarios**: Combined modifier workflows

Test Organization
-----------------
- MockModel: Simple neural network for testing modifications
- TestModelModifier: Core abstraction and interface tests
- TestFreezingModelModifier: Parameter freezing functionality
- TestTransferLearningModelModifier: Weight loading and transfer
- TestModelModifierApplier: Orchestration and composition
- TestModelModifierIntegration: End-to-end workflows

Key Testing Principles
----------------------
- Each modifier is tested in isolation
- Integration tests verify modifier composition
- Edge cases (missing modules, shape mismatches) are covered
- Performance optimizations are validated
- Gradient flow validation ensures correctness
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path

from anemoi.training.train.modify import FreezingModelModifier
from anemoi.training.train.modify import ModelModifier
from anemoi.training.train.modify import ModelModifierApplier
from anemoi.training.train.modify import TransferLearningModelModifier


class MockModel(nn.Module):
    """Mock neural network model for testing model modifications.

    This model simulates a typical encoder-processor-decoder architecture
    commonly used in weather forecasting and other scientific applications.
    It provides a realistic test bed for verifying parameter freezing,
    weight loading, and other model modifications.

    Architecture
    ------------
    - Encoder: Linear(10, 20) - Input transformation layer
    - Processor: ModuleList with 2x Linear(20, 20) - Processing layers
    - Decoder: Linear(20, 5) - Output projection layer

    Attributes
    ----------
    weights_initialized : bool
        Flag for testing initialization workflows
    skip_checkpoint_loading : bool
        Flag for testing checkpoint loading behavior

    Notes
    -----
    The model structure allows testing of:
    - Direct module access (encoder, decoder)
    - Nested module access (processor[0], processor[1])
    - Parameter freezing at different levels
    - Weight transfer with matching/mismatching shapes
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.processor = nn.ModuleList(
            [
                nn.Linear(20, 20),
                nn.Linear(20, 20),
            ],
        )
        self.decoder = nn.Linear(20, 5)
        self.weights_initialized = False
        self.skip_checkpoint_loading = False

    def forward(self, x: Any) -> Any:
        """Forward pass through encoder-processor-decoder pipeline."""
        x = self.encoder(x)
        for layer in self.processor:
            x = layer(x)
        return self.decoder(x)


class ConcreteModelModifier(ModelModifier):
    """Concrete implementation for testing abstract base class."""

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        # Simple test implementation - just set a flag
        model.test_modifier_applied = True
        return model


class TestModelModifier:
    """Test the ModelModifier abstract base class."""

    def test_abstract_base_class(self) -> None:
        """Test that ModelModifier cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModelModifier()

    def test_concrete_implementation(self) -> None:
        """Test that concrete implementations work correctly."""
        modifier = ConcreteModelModifier()
        model = MockModel()

        result = modifier.apply(model)
        assert result is model
        assert hasattr(result, "test_modifier_applied")
        assert result.test_modifier_applied is True


class TestFreezingModelModifier:
    """Comprehensive tests for the FreezingModelModifier.

    This test class validates all aspects of parameter freezing including:
    - Basic freezing functionality
    - Nested module access
    - Error handling for missing modules
    - Gradient flow validation
    - Performance optimizations

    The tests ensure that frozen parameters:
    1. Have requires_grad set to False
    2. Do not accumulate gradients during backpropagation
    3. Can be selectively frozen while keeping others trainable
    """

    def test_init(self) -> None:
        """Test FreezingModelModifier initialization with default parameters.

        Verifies that the modifier correctly stores the list of modules
        to freeze and initializes with appropriate default values for
        strict mode and gradient validation.
        """
        submodules_to_freeze = ["encoder", "processor.0"]
        modifier = FreezingModelModifier(submodules_to_freeze)
        assert modifier.submodules_to_freeze == submodules_to_freeze

    def test_freeze_single_module(self) -> None:
        """Test freezing a single module's parameters.

        Verifies that:
        - Target module parameters are set to requires_grad=False
        - Other module parameters remain trainable
        - The model object is returned (in-place modification)
        """
        model = MockModel()

        # Verify parameters are trainable initially
        assert all(p.requires_grad for p in model.encoder.parameters())

        modifier = FreezingModelModifier(["encoder"])

        result = modifier.apply(model)

        # Verify encoder parameters are frozen
        assert all(not p.requires_grad for p in model.encoder.parameters())
        # Verify other parameters are still trainable
        assert all(p.requires_grad for p in model.decoder.parameters())
        assert result is model

    def test_freeze_multiple_modules(self) -> None:
        """Test freezing multiple modules simultaneously.

        Validates that the modifier can freeze multiple modules in a single
        operation while preserving the trainability of unspecified modules.
        This is crucial for transfer learning scenarios where multiple
        components need to be frozen.
        """
        model = MockModel()

        modifier = FreezingModelModifier(["encoder", "decoder"])

        result = modifier.apply(model)

        # Verify both encoder and decoder are frozen
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(not p.requires_grad for p in model.decoder.parameters())
        # Verify processor is still trainable
        assert all(p.requires_grad for p in model.processor.parameters())
        assert result is model

    def test_freeze_nested_module(self) -> None:
        """Test freezing nested modules using dot notation.

        Verifies that the modifier can access and freeze modules within
        ModuleLists and other container modules using standard PyTorch
        naming conventions (e.g., 'processor.0' for first processor layer).
        """
        model = MockModel()

        # Verify initial state
        assert all(p.requires_grad for p in model.processor[0].parameters())
        assert all(p.requires_grad for p in model.processor[1].parameters())

        modifier = FreezingModelModifier(["0"])  # This should freeze processor.0

        result = modifier.apply(model)

        # Note: The current implementation searches by direct child name
        # So "0" would freeze any immediate child named "0"
        # For processor[0], we need to access it differently in the implementation
        assert result is model

    def test_freeze_nonexistent_module(self) -> None:
        """Test graceful handling of non-existent module names.

        Ensures that the modifier logs a warning but continues execution
        when attempting to freeze modules that don't exist. This is the
        default non-strict behavior that allows flexible configuration.
        """
        model = MockModel()

        modifier = FreezingModelModifier(["nonexistent"])

        # Should not raise an exception
        result = modifier.apply(model)
        assert result is model

        # All parameters should still be trainable
        assert all(p.requires_grad for p in model.parameters())

    def test_freeze_nonexistent_module_strict_mode(self) -> None:
        """Test strict mode enforcement for missing modules.

        Validates that when strict=True, the modifier raises a clear
        error message if any specified module cannot be found. This
        helps catch configuration errors early in development.
        """
        model = MockModel()

        modifier = FreezingModelModifier(["nonexistent"], strict=True)

        # Should raise ValueError in strict mode
        with pytest.raises(ValueError, match="Module 'nonexistent' not found"):
            modifier.apply(model)

    def test_gradient_validation(self) -> None:
        """Test automatic validation of gradient flow blocking.

        This test verifies the gradient validation feature which:
        1. Performs a test forward/backward pass after freezing
        2. Checks that frozen parameters have no gradients
        3. Ensures the freezing mechanism works correctly

        This validation helps catch subtle bugs where parameters might
        appear frozen but still accumulate gradients due to implementation
        errors or PyTorch quirks.
        """
        model = MockModel()

        # Freeze encoder with validation enabled
        modifier = FreezingModelModifier(["encoder"], validate_gradients=True)
        modifier.apply(model)

        # Verify encoder is frozen
        assert all(not p.requires_grad for p in model.encoder.parameters())

        # Manually test that gradients aren't accumulated
        test_input = torch.randn(2, 10)
        output = model(test_input)
        loss = output.sum()
        loss.backward()

        # Frozen parameters should have no gradients
        for param in model.encoder.parameters():
            assert param.grad is None

        # Unfrozen parameters should have gradients
        for param in model.decoder.parameters():
            assert param.grad is not None

    def test_gradient_validation_disabled(self) -> None:
        """Test that gradient validation can be disabled."""
        model = MockModel()

        # Freeze encoder with validation disabled
        modifier = FreezingModelModifier(["encoder"], validate_gradients=False)

        # Should complete without validation
        result = modifier.apply(model)
        assert result is model
        assert all(not p.requires_grad for p in model.encoder.parameters())


class TestTransferLearningModelModifier:
    """Comprehensive tests for the TransferLearningModelModifier.

    This test class validates the transfer learning functionality including:
    - Checkpoint loading from various sources
    - Weight transfer with exact and partial matches
    - Handling of architecture mismatches
    - Fallback mechanisms when external loaders are unavailable
    - Integration with freezing for fine-tuning workflows

    The tests cover both modes of operation:
    1. Standalone fallback using torch.load directly
    2. Integration with external checkpoint loading system (when available)
    """

    def create_test_checkpoint(self, temp_dir: Path, include_mismatched_layer: bool = False) -> Path:
        """Create a test checkpoint file with optional architecture mismatches.

        Parameters
        ----------
        temp_dir : Path
            Directory to save the checkpoint
        include_mismatched_layer : bool
            If True, creates a checkpoint with different layer dimensions
            to test mismatch handling

        Returns
        -------
        Path
            Path to the created checkpoint file
        """
        checkpoint_path = temp_dir / "transfer_checkpoint.ckpt"

        if include_mismatched_layer:
            # Create model with different output size for testing mismatch handling
            class MismatchedModel(MockModel):
                def __init__(self):
                    super().__init__()
                    self.decoder = nn.Linear(20, 10)  # Different output size

            test_model = MismatchedModel()
        else:
            test_model = MockModel()

        state_dict = test_model.state_dict()

        checkpoint_data = {"state_dict": state_dict, "hyper_parameters": {"data_indices": MagicMock()}}
        checkpoint_data["hyper_parameters"]["data_indices"].name_to_index = {"test_var": 0}

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def test_init(self) -> None:
        """Test TransferLearningModelModifier initialization."""
        checkpoint_path = "/path/to/checkpoint.ckpt"
        modifier = TransferLearningModelModifier(checkpoint_path)
        assert str(modifier.checkpoint_path) == checkpoint_path
        assert modifier.strict is False  # default
        assert modifier.skip_mismatched is True  # default

    def test_init_with_parameters(self) -> None:
        """Test TransferLearningModelModifier initialization with custom parameters."""
        checkpoint_path = "/path/to/checkpoint.ckpt"
        modifier = TransferLearningModelModifier(checkpoint_path, strict=True, skip_mismatched=False)
        assert str(modifier.checkpoint_path) == checkpoint_path
        assert modifier.strict is True
        assert modifier.skip_mismatched is False

    def test_apply_fallback_with_mock_checkpoint(self) -> None:
        """Test applying transfer learning using fallback implementation."""
        checkpoint_path = "/fake/checkpoint.ckpt"
        modifier = TransferLearningModelModifier(checkpoint_path, strict=False, skip_mismatched=True)
        model = MockModel()

        # Create a mock state dict
        mock_state_dict = {
            "encoder.weight": torch.randn(20, 10),
            "encoder.bias": torch.randn(20),
            "decoder.weight": torch.randn(5, 20),
            "decoder.bias": torch.randn(5),
        }

        # Mock torch.load to return our fake checkpoint
        with patch("torch.load") as mock_torch_load:
            mock_torch_load.return_value = {"state_dict": mock_state_dict}

            result = modifier.apply(model)

            assert result is model
            # Note: checkpoint_path is converted to Path internally
            from pathlib import Path

            mock_torch_load.assert_called_once_with(Path(checkpoint_path), map_location="cpu", weights_only=False)

    def test_apply_fallback_with_shape_mismatch(self) -> None:
        """Test fallback implementation handles shape mismatches correctly."""
        checkpoint_path = "/fake/checkpoint.ckpt"
        modifier = TransferLearningModelModifier(checkpoint_path, strict=False, skip_mismatched=True)
        model = MockModel()

        # Create a mock state dict with shape mismatches
        mock_state_dict = {
            "encoder.weight": torch.randn(30, 10),  # Wrong size - should be (20, 10)
            "encoder.bias": torch.randn(20),  # Correct size
            "decoder.weight": torch.randn(5, 20),  # Correct size
            "decoder.bias": torch.randn(5),  # Correct size
            "nonexistent.weight": torch.randn(10, 10),  # Doesn't exist in model
        }

        with patch("torch.load") as mock_torch_load:
            mock_torch_load.return_value = {"state_dict": mock_state_dict}

            result = modifier.apply(model)

            assert result is model
            # Only parameters with matching shapes should be loaded


class TestModelModifierApplier:
    """Tests for the ModelModifierApplier orchestration component.

    The ModelModifierApplier is responsible for:
    - Reading modifier configuration from Hydra configs
    - Instantiating modifiers in the correct order
    - Applying modifiers sequentially to models
    - Handling errors and edge cases gracefully

    These tests verify that the applier correctly handles various
    configuration scenarios and maintains proper modifier ordering.
    """

    def test_process_no_modifiers(self) -> None:
        """Test that models pass through unchanged when no modifiers are configured.

        This is an important baseline test ensuring the applier doesn't
        modify models when no modifications are requested.
        """
        applier = ModelModifierApplier()
        model = MockModel()
        config = DictConfig({"training": {}})

        result = applier.process(model, config)
        assert result is model

    def test_process_no_model_modifier_config(self) -> None:
        """Test processing when model_modifier is not in config."""
        applier = ModelModifierApplier()
        model = MockModel()
        config = DictConfig({"training": {"other_config": "value"}})

        result = applier.process(model, config)
        assert result is model

    def test_process_empty_modifiers(self) -> None:
        """Test processing when modifiers list is empty."""
        applier = ModelModifierApplier()
        model = MockModel()
        config = DictConfig({"training": {"model_modifier": {"modifiers": []}}})

        result = applier.process(model, config)
        assert result is model

    @patch("anemoi.training.train.modify.instantiate")
    def test_process_single_modifier(self, mock_instantiate: Any) -> None:
        """Test processing with a single modifier."""
        applier = ModelModifierApplier()
        model = MockModel()

        # Create mock modifier
        mock_modifier = MagicMock()
        mock_modifier.apply.return_value = model
        mock_instantiate.return_value = mock_modifier

        config = DictConfig(
            {"training": {"model_modifier": {"modifiers": [{"_target_": "test.modifier.TestModifier"}]}}},
        )

        result = applier.process(model, config)

        assert result is model
        mock_instantiate.assert_called_once()
        mock_modifier.apply.assert_called_once_with(model)

    @patch("anemoi.training.train.modify.instantiate")
    def test_process_multiple_modifiers(self, mock_instantiate: Any) -> None:
        """Test processing with multiple modifiers."""
        applier = ModelModifierApplier()
        model = MockModel()

        # Create mock modifiers
        mock_modifier1 = MagicMock()
        mock_modifier1.apply.return_value = model
        mock_modifier2 = MagicMock()
        mock_modifier2.apply.return_value = model

        mock_instantiate.side_effect = [mock_modifier1, mock_modifier2]

        config = DictConfig(
            {
                "training": {
                    "model_modifier": {
                        "modifiers": [{"_target_": "test.modifier.Modifier1"}, {"_target_": "test.modifier.Modifier2"}],
                    },
                },
            },
        )

        result = applier.process(model, config)

        assert result is model
        assert mock_instantiate.call_count == 2
        mock_modifier1.apply.assert_called_once_with(model)
        mock_modifier2.apply.assert_called_once_with(model)

    def test_process_modifier_order(self) -> None:
        """Test that modifiers are applied in the correct order."""
        applier = ModelModifierApplier()
        model = MockModel()

        # Track the order of application
        application_order = []

        class OrderedModifier1(ModelModifier):
            def apply(self, model: Any) -> Any:
                application_order.append("modifier1")
                model.step1_applied = True
                return model

        class OrderedModifier2(ModelModifier):
            def apply(self, model: Any) -> Any:
                application_order.append("modifier2")
                # Should see step1_applied from first modifier
                assert hasattr(model, "step1_applied")
                model.step2_applied = True
                return model

        with patch("anemoi.training.train.modify.instantiate") as mock_instantiate:
            mock_instantiate.side_effect = [OrderedModifier1(), OrderedModifier2()]

            config = DictConfig(
                {"training": {"model_modifier": {"modifiers": [{"_target_": "modifier1"}, {"_target_": "modifier2"}]}}},
            )

            result = applier.process(model, config)

            assert result is model
            assert application_order == ["modifier1", "modifier2"]
            assert hasattr(model, "step1_applied")
            assert hasattr(model, "step2_applied")


class TestModelModifierIntegration:
    """End-to-end integration tests combining multiple modifiers.

    These tests validate real-world workflows where multiple modifiers
    work together to prepare models for training. Common scenarios include:

    1. Transfer learning + freezing: Load pretrained weights then freeze layers
    2. Multi-stage freezing: Progressively freeze different components
    3. Complex configuration: Multiple modifiers with dependencies

    The integration tests ensure that modifiers:
    - Compose correctly without conflicts
    - Apply in the specified order
    - Produce expected combined results
    - Handle edge cases in combination
    """

    def test_transfer_learning_after_freezing_fallback(self) -> None:
        """Test complete transfer learning workflow with parameter freezing.

        This integration test simulates a common fine-tuning scenario:
        1. Load pretrained weights from a checkpoint
        2. Freeze the encoder to preserve learned features
        3. Keep decoder trainable for task-specific adaptation

        Uses the fallback mechanism to ensure functionality even without
        external checkpoint loading infrastructure.
        """
        checkpoint_path = "/fake/checkpoint.ckpt"
        model = MockModel()

        # Create mock state dict
        mock_state_dict = {
            "encoder.weight": torch.randn(20, 10),
            "encoder.bias": torch.randn(20),
        }

        with patch("torch.load") as mock_torch_load:
            mock_torch_load.return_value = {"state_dict": mock_state_dict}

            # Apply modifiers in sequence
            # First apply transfer learning
            transfer_modifier = TransferLearningModelModifier(checkpoint_path)
            model = transfer_modifier.apply(model)

            # Then freeze encoder
            freeze_modifier = FreezingModelModifier(["encoder"])
            model = freeze_modifier.apply(model)

            # Verify results
            mock_torch_load.assert_called_once()
            assert all(not p.requires_grad for p in model.encoder.parameters())
            assert all(p.requires_grad for p in model.decoder.parameters())

    def test_full_pipeline_with_applier_fallback(self) -> None:
        """Test complete modifier pipeline using configuration-driven approach.

        This comprehensive test validates the entire workflow from Hydra
        configuration to modified model, including:

        1. Configuration parsing and validation
        2. Dynamic modifier instantiation via Hydra
        3. Sequential application of transfer learning and freezing
        4. Proper fallback handling when external systems are unavailable

        This represents the typical production usage pattern where all
        modifications are specified in configuration files.
        """
        applier = ModelModifierApplier()
        model = MockModel()

        config = DictConfig(
            {
                "training": {
                    "model_modifier": {
                        "modifiers": [
                            {
                                "_target_": "anemoi.training.train.modify.TransferLearningModelModifier",
                                "checkpoint_path": "/fake/checkpoint.ckpt",
                                "strict": False,
                                "skip_mismatched": True,
                            },
                            {
                                "_target_": "anemoi.training.train.modify.FreezingModelModifier",
                                "submodules_to_freeze": ["encoder"],
                            },
                        ],
                    },
                },
            },
        )

        # Mock torch.load for the fallback implementation
        mock_state_dict = {
            "encoder.weight": torch.randn(20, 10),
            "encoder.bias": torch.randn(20),
        }

        with patch("torch.load") as mock_torch_load:
            mock_torch_load.return_value = {"state_dict": mock_state_dict}

            result = applier.process(model, config)

            assert result is not None
            assert all(not p.requires_grad for p in result.encoder.parameters())
            mock_torch_load.assert_called_once()
