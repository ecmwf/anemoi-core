# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from anemoi.training.utils.model_loading import ModelLoaderRegistry
from anemoi.training.utils.model_loading import StandardModelLoader
from anemoi.training.utils.model_loading import TransferLearningModelLoader
from anemoi.training.utils.model_loading import load_model_from_checkpoint


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class TestStandardModelLoader:
    """Test StandardModelLoader functionality."""

    def test_load_standard_checkpoint(self) -> None:
        """Test loading a standard Lightning checkpoint."""
        loader = StandardModelLoader()
        model = DummyModel()

        # Create checkpoint with state_dict
        state_dict = model.state_dict()
        checkpoint = {
            "state_dict": state_dict,
            "hyper_parameters": {"data_indices": Mock(name_to_index={"var1": 0, "var2": 1})},
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            tmp_path = Path(tmp_file.name)

        try:
            with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
                mock_load.return_value = checkpoint
                loaded_model = loader.load_model_weights(model, tmp_path)

                assert hasattr(loaded_model, "_ckpt_model_name_to_index")
                assert loaded_model._ckpt_model_name_to_index == {"var1": 0, "var2": 1}
        finally:
            tmp_path.unlink()

    def test_load_checkpoint_missing_state_dict(self) -> None:
        """Test error handling when checkpoint has no state_dict."""
        loader = StandardModelLoader()
        model = DummyModel()

        with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
            mock_load.return_value = {"some_other_key": "value"}

            with pytest.raises(ValueError, match="No 'state_dict' found"):
                loader.load_model_weights(model, "/fake/path.pt")


class TestTransferLearningModelLoader:
    """Test TransferLearningModelLoader functionality."""

    def test_load_with_size_mismatch(self) -> None:
        """Test loading with size mismatches (should skip mismatched layers)."""
        loader = TransferLearningModelLoader()
        model = DummyModel()

        # Create checkpoint with mismatched layer size
        mismatched_state_dict = {
            "layer1.weight": torch.randn(10, 5),  # Correct size
            "layer1.bias": torch.randn(5),  # Correct size
            "layer2.weight": torch.randn(1, 20),  # Wrong size (should be 1, 5)
            "layer2.bias": torch.randn(1),  # Correct size
        }

        checkpoint = {
            "state_dict": mismatched_state_dict,
            "hyper_parameters": {"data_indices": Mock(name_to_index={"var1": 0, "var2": 1})},
        }

        with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
            mock_load.return_value = checkpoint
            loaded_model = loader.load_model_weights(model, "/fake/path.pt", skip_mismatched=True)

            # Check that model has weights_initialized flag
            assert hasattr(loaded_model, "weights_initialized")
            assert loaded_model.weights_initialized is True

    def test_load_without_skip_mismatched(self) -> None:
        """Test loading without skipping mismatched layers (should fail)."""
        loader = TransferLearningModelLoader()
        model = DummyModel()

        # Create checkpoint with mismatched layer size
        mismatched_state_dict = {
            "layer1.weight": torch.randn(10, 5),  # Correct size
            "layer2.weight": torch.randn(1, 20),  # Wrong size
        }

        checkpoint = {"state_dict": mismatched_state_dict}

        with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
            mock_load.return_value = checkpoint

            with pytest.raises(RuntimeError):  # PyTorch raises RuntimeError for size mismatch
                loader.load_model_weights(model, "/fake/path.pt", skip_mismatched=False, strict=True)


class TestModelLoaderRegistry:
    """Test ModelLoaderRegistry functionality."""

    def test_registry_has_default_loaders(self) -> None:
        """Test that registry comes with default loaders."""
        registry = ModelLoaderRegistry()

        assert "standard" in registry._loaders
        assert "transfer_learning" in registry._loaders
        assert "weights_only" in registry._loaders

    def test_get_loader_by_name(self) -> None:
        """Test retrieving loaders by name."""
        registry = ModelLoaderRegistry()

        standard_loader = registry.get_loader("standard")
        assert isinstance(standard_loader, StandardModelLoader)

        transfer_loader = registry.get_loader("transfer_learning")
        assert isinstance(transfer_loader, TransferLearningModelLoader)

    def test_unknown_loader_error(self) -> None:
        """Test error handling for unknown loader names."""
        registry = ModelLoaderRegistry()

        with pytest.raises(ValueError, match="Unknown loader"):
            registry.get_loader("nonexistent_loader")

    def test_register_custom_loader(self) -> None:
        """Test registering a custom loader."""
        registry = ModelLoaderRegistry()
        custom_loader = Mock()

        registry.register("custom", custom_loader)
        assert registry.get_loader("custom") is custom_loader


class TestLoadModelFromCheckpoint:
    """Test the convenience function."""

    def test_load_model_from_checkpoint_function(self) -> None:
        """Test the main convenience function."""
        model = DummyModel()

        # Create a simple checkpoint
        state_dict = model.state_dict()
        checkpoint = {"state_dict": state_dict}

        with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
            mock_load.return_value = checkpoint

            loaded_model = load_model_from_checkpoint(
                model=model,
                checkpoint_source="/fake/path.pt",
                loader_type="standard",
            )

            assert loaded_model is model  # Should return the same model instance

    def test_load_with_transfer_learning(self) -> None:
        """Test loading with transfer learning type."""
        model = DummyModel()

        checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": {"data_indices": Mock(name_to_index={})}}

        with patch("anemoi.training.utils.model_loading.load_checkpoint_from_source") as mock_load:
            mock_load.return_value = checkpoint

            loaded_model = load_model_from_checkpoint(
                model=model,
                checkpoint_source="/fake/path.pt",
                loader_type="transfer_learning",
            )

            assert hasattr(loaded_model, "weights_initialized")
            assert loaded_model.weights_initialized is True
