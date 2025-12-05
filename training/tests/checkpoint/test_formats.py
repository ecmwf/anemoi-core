# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint format detection and conversion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.formats import convert_lightning_to_pytorch
from anemoi.training.checkpoint.formats import detect_checkpoint_format
from anemoi.training.checkpoint.formats import extract_state_dict
from anemoi.training.checkpoint.formats import is_format_available
from anemoi.training.checkpoint.formats import load_checkpoint
from anemoi.training.checkpoint.formats import save_checkpoint

if TYPE_CHECKING:
    from conftest import SimpleModel


class TestFormatDetection:
    """Test checkpoint format detection functionality."""

    @pytest.mark.unit
    def test_detect_lightning_format(self, checkpoint_files: dict[str, Path]) -> None:
        """Test detection of Lightning checkpoint format."""
        lightning_path = checkpoint_files["lightning"]

        assert detect_checkpoint_format(lightning_path) == "lightning"

    @pytest.mark.unit
    def test_detect_pytorch_format(self, checkpoint_files: dict[str, Path]) -> None:
        """Test detection of PyTorch checkpoint format."""
        pytorch_path = checkpoint_files["pytorch"]

        assert detect_checkpoint_format(pytorch_path) == "pytorch"

    @pytest.mark.unit
    def test_detect_state_dict_format(self, checkpoint_files: dict[str, Path]) -> None:
        """Test detection of raw state dict format."""
        state_dict_path = checkpoint_files["state_dict"]

        assert detect_checkpoint_format(state_dict_path) == "state_dict"

    @pytest.mark.unit
    def test_detect_format_with_path_string(self, checkpoint_files: dict[str, Path]) -> None:
        """Test format detection with string path."""
        lightning_path_str = str(checkpoint_files["lightning"])

        assert detect_checkpoint_format(lightning_path_str) == "lightning"

    @pytest.mark.unit
    def test_detect_format_unknown_extension(self, temp_checkpoint_dir: Path, lightning_checkpoint: dict) -> None:
        """Test format detection with unknown file extension."""
        unknown_path = temp_checkpoint_dir / "model.unknown"
        torch.save(lightning_checkpoint, unknown_path)

        # Should default to lightning for unknown extensions after inspection
        assert detect_checkpoint_format(unknown_path) == "lightning"

    @pytest.mark.unit
    def test_detect_format_corrupted_file(self, temp_checkpoint_dir: Path) -> None:
        """Test format detection with corrupted checkpoint file."""
        corrupted_path = temp_checkpoint_dir / "corrupted.ckpt"

        # Write corrupted data
        with corrupted_path.open("wb") as f:
            f.write(b"corrupted data that is not a valid checkpoint")

        # Should default to lightning for corrupted files
        assert detect_checkpoint_format(corrupted_path) == "lightning"

    @pytest.mark.unit
    def test_detect_format_non_dict_checkpoint(self, temp_checkpoint_dir: Path, simple_model: SimpleModel) -> None:
        """Test format detection with non-dict checkpoint (raw model)."""
        model_path = temp_checkpoint_dir / "raw_model.pt"
        torch.save(simple_model, model_path)

        assert detect_checkpoint_format(model_path) == "pytorch"

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".ckpt", ".pt", ".pth", ".bin"])
    def test_detect_format_supported_extensions(
        self,
        temp_checkpoint_dir: Path,
        lightning_checkpoint: dict,
        extension: str,
    ) -> None:
        """Test format detection with all supported extensions."""
        checkpoint_path = temp_checkpoint_dir / f"model{extension}"
        torch.save(lightning_checkpoint, checkpoint_path)

        # Should detect as lightning based on content
        assert detect_checkpoint_format(checkpoint_path) == "lightning"


class TestFormatLoading:
    """Test checkpoint loading functionality."""

    @pytest.mark.unit
    def test_load_checkpoint_auto_detect(self, checkpoint_files: dict[str, Path]) -> None:
        """Test loading checkpoint with automatic format detection."""
        lightning_data = load_checkpoint(checkpoint_files["lightning"])

        assert isinstance(lightning_data, dict)
        assert "state_dict" in lightning_data
        assert "pytorch-lightning_version" in lightning_data

    @pytest.mark.unit
    def test_load_checkpoint_explicit_format(self, checkpoint_files: dict[str, Path]) -> None:
        """Test loading checkpoint with explicit format specification."""
        pytorch_data = load_checkpoint(checkpoint_files["pytorch"], checkpoint_format="pytorch")

        assert isinstance(pytorch_data, dict)
        assert "model_state_dict" in pytorch_data
        assert "optimizer_state_dict" in pytorch_data

    @pytest.mark.unit
    def test_load_checkpoint_state_dict_format(self, checkpoint_files: dict[str, Path]) -> None:
        """Test loading raw state dict checkpoint."""
        state_dict_data = load_checkpoint(checkpoint_files["state_dict"], checkpoint_format="state_dict")

        assert isinstance(state_dict_data, dict)
        # Should be a dict of tensors
        assert all(isinstance(v, torch.Tensor) for v in state_dict_data.values())

    @pytest.mark.unit
    def test_load_checkpoint_nonexistent_file(self) -> None:
        """Test loading checkpoint from nonexistent file."""
        nonexistent_path = Path("/nonexistent/checkpoint.ckpt")

        with pytest.raises(CheckpointNotFoundError):
            load_checkpoint(nonexistent_path)


class TestStateDictExtraction:
    """Test state dict extraction from different checkpoint formats."""

    @pytest.mark.unit
    def test_extract_state_dict_lightning(self, lightning_checkpoint: dict) -> None:
        """Test extracting state dict from Lightning checkpoint."""
        state_dict = extract_state_dict(lightning_checkpoint)

        assert isinstance(state_dict, dict)
        assert state_dict is lightning_checkpoint["state_dict"]

    @pytest.mark.unit
    def test_extract_state_dict_pytorch(self, pytorch_checkpoint: dict) -> None:
        """Test extracting state dict from PyTorch checkpoint."""
        state_dict = extract_state_dict(pytorch_checkpoint)

        assert isinstance(state_dict, dict)
        assert state_dict is pytorch_checkpoint["model_state_dict"]

    @pytest.mark.unit
    def test_extract_state_dict_model_key(self, sample_state_dict: dict) -> None:
        """Test extracting state dict using 'model' key."""
        checkpoint = {"model": sample_state_dict, "epoch": 5}

        state_dict = extract_state_dict(checkpoint)

        assert isinstance(state_dict, dict)
        assert state_dict is sample_state_dict

    @pytest.mark.unit
    def test_extract_state_dict_raw(self, sample_state_dict: dict) -> None:
        """Test extracting state dict from raw state dict (no wrapper)."""
        state_dict = extract_state_dict(sample_state_dict)

        assert isinstance(state_dict, dict)
        assert state_dict is sample_state_dict

    @pytest.mark.unit
    def test_extract_state_dict_empty(self) -> None:
        """Test extracting state dict from empty checkpoint."""
        empty_checkpoint = {}

        with pytest.raises(CheckpointValidationError) as exc_info:
            extract_state_dict(empty_checkpoint)

        assert "Cannot find model state" in str(exc_info.value)

    @pytest.mark.unit
    def test_extract_state_dict_precedence(self) -> None:
        """Test state dict extraction precedence (state_dict > model_state_dict > model)."""
        checkpoint = {
            "state_dict": {"a": torch.tensor(1.0)},
            "model_state_dict": {"b": torch.tensor(2.0)},
            "model": {"c": torch.tensor(3.0)},
        }

        state_dict = extract_state_dict(checkpoint)

        # Should extract 'state_dict' as highest priority
        assert "a" in state_dict
        assert "b" not in state_dict
        assert "c" not in state_dict


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    @pytest.mark.unit
    def test_save_checkpoint_pytorch_format(self, temp_checkpoint_dir: Path, pytorch_checkpoint: dict) -> None:
        """Test saving checkpoint in PyTorch format."""
        save_path = temp_checkpoint_dir / "saved_pytorch.pt"

        save_checkpoint(pytorch_checkpoint, save_path, checkpoint_format="pytorch")

        # Verify file was created and can be loaded
        assert save_path.exists()
        loaded_data = torch.load(save_path, map_location="cpu", weights_only=False)
        assert loaded_data["epoch"] == pytorch_checkpoint["epoch"]

    @pytest.mark.unit
    def test_save_checkpoint_lightning_format(self, temp_checkpoint_dir: Path, lightning_checkpoint: dict) -> None:
        """Test saving checkpoint in Lightning format."""
        save_path = temp_checkpoint_dir / "saved_lightning.ckpt"

        save_checkpoint(lightning_checkpoint, save_path, checkpoint_format="lightning")

        # Verify file was created and can be loaded
        assert save_path.exists()
        loaded_data = torch.load(save_path, map_location="cpu", weights_only=False)
        assert "pytorch-lightning_version" in loaded_data

    @pytest.mark.unit
    def test_save_checkpoint_state_dict_format(self, temp_checkpoint_dir: Path, sample_state_dict: dict) -> None:
        """Test saving raw state dict."""
        save_path = temp_checkpoint_dir / "saved_state_dict.pth"

        save_checkpoint(sample_state_dict, save_path, checkpoint_format="state_dict")

        # Verify file was created and can be loaded
        assert save_path.exists()
        loaded_data = torch.load(save_path, map_location="cpu", weights_only=False)
        assert all(isinstance(v, torch.Tensor) for v in loaded_data.values())

    @pytest.mark.unit
    def test_save_checkpoint_string_path(self, temp_checkpoint_dir: Path, pytorch_checkpoint: dict) -> None:
        """Test saving checkpoint with string path."""
        save_path_str = str(temp_checkpoint_dir / "string_path.pt")

        save_checkpoint(pytorch_checkpoint, save_path_str, checkpoint_format="pytorch")

        assert Path(save_path_str).exists()


class TestLightningToPyTorchConversion:
    """Test conversion from Lightning to PyTorch format."""

    @pytest.mark.unit
    def test_convert_lightning_to_pytorch_model_only(self, lightning_checkpoint: dict) -> None:
        """Test converting Lightning checkpoint to PyTorch format (model only)."""
        pytorch_checkpoint = convert_lightning_to_pytorch(lightning_checkpoint, extract_model_only=True)

        assert isinstance(pytorch_checkpoint, dict)
        assert "model_state_dict" in pytorch_checkpoint
        assert pytorch_checkpoint["model_state_dict"] is lightning_checkpoint["state_dict"]

        # Should not contain optimizer/scheduler info for model-only extraction
        assert "optimizer_state_dict" not in pytorch_checkpoint
        assert "scheduler_state_dict" not in pytorch_checkpoint
        assert "epoch" not in pytorch_checkpoint
        assert "global_step" not in pytorch_checkpoint

    @pytest.mark.unit
    def test_convert_lightning_to_pytorch_full(self, lightning_checkpoint: dict) -> None:
        """Test converting Lightning checkpoint to PyTorch format (full)."""
        pytorch_checkpoint = convert_lightning_to_pytorch(lightning_checkpoint, extract_model_only=False)

        assert isinstance(pytorch_checkpoint, dict)
        assert "model_state_dict" in pytorch_checkpoint
        assert "optimizer_state_dict" in pytorch_checkpoint
        assert "scheduler_state_dict" in pytorch_checkpoint
        assert "epoch" in pytorch_checkpoint
        assert "global_step" in pytorch_checkpoint

        # Verify values are transferred correctly
        assert pytorch_checkpoint["epoch"] == lightning_checkpoint["epoch"]
        assert pytorch_checkpoint["global_step"] == lightning_checkpoint["global_step"]

    @pytest.mark.unit
    def test_convert_lightning_minimal_checkpoint(self, sample_state_dict: dict) -> None:
        """Test converting minimal Lightning checkpoint."""
        minimal_lightning = {
            "state_dict": sample_state_dict,
            "pytorch-lightning_version": "2.0.0",
        }

        pytorch_checkpoint = convert_lightning_to_pytorch(minimal_lightning, extract_model_only=False)

        assert "model_state_dict" in pytorch_checkpoint
        # Should not fail on missing optional keys
        assert "optimizer_state_dict" not in pytorch_checkpoint

    @pytest.mark.unit
    def test_convert_lightning_missing_state_dict(self) -> None:
        """Test converting Lightning checkpoint without state dict."""
        lightning_without_state = {
            "pytorch-lightning_version": "2.0.0",
            "epoch": 5,
        }

        with pytest.raises(CheckpointValidationError) as exc_info:
            convert_lightning_to_pytorch(lightning_without_state)

        assert "no recognizable Lightning components" in str(exc_info.value)


class TestFormatAvailability:
    """Test format availability checking."""

    @pytest.mark.unit
    def test_pytorch_format_always_available(self) -> None:
        """Test that PyTorch format is always available."""
        assert is_format_available("pytorch") is True

    @pytest.mark.unit
    def test_lightning_format_always_available(self) -> None:
        """Test that Lightning format is always available."""
        assert is_format_available("lightning") is True

    @pytest.mark.unit
    def test_state_dict_format_always_available(self) -> None:
        """Test that state dict format is always available."""
        assert is_format_available("state_dict") is True


class TestFormatIntegration:
    """Integration tests for format operations."""

    @pytest.mark.integration
    def test_save_load_cycle_pytorch(self, temp_checkpoint_dir: Path, pytorch_checkpoint: dict) -> None:
        """Test complete save/load cycle for PyTorch format."""
        save_path = temp_checkpoint_dir / "cycle_test.pt"

        # Save checkpoint
        save_checkpoint(pytorch_checkpoint, save_path, checkpoint_format="pytorch")

        # Load and verify
        loaded_checkpoint = load_checkpoint(save_path)

        assert loaded_checkpoint["epoch"] == pytorch_checkpoint["epoch"]
        assert loaded_checkpoint["loss"] == pytorch_checkpoint["loss"]

    @pytest.mark.integration
    def test_save_load_cycle_lightning(self, temp_checkpoint_dir: Path, lightning_checkpoint: dict) -> None:
        """Test complete save/load cycle for Lightning format."""
        save_path = temp_checkpoint_dir / "cycle_test.ckpt"

        # Save checkpoint
        save_checkpoint(lightning_checkpoint, save_path, checkpoint_format="lightning")

        # Load and verify
        loaded_checkpoint = load_checkpoint(save_path)

        assert loaded_checkpoint["epoch"] == lightning_checkpoint["epoch"]
        assert "pytorch-lightning_version" in loaded_checkpoint

    @pytest.mark.integration
    def test_format_conversion_chain(self, temp_checkpoint_dir: Path, lightning_checkpoint: dict) -> None:
        """Test conversion chain: Lightning -> PyTorch -> save/load."""
        # Convert Lightning to PyTorch
        pytorch_checkpoint = convert_lightning_to_pytorch(lightning_checkpoint, extract_model_only=False)

        # Save as PyTorch format
        save_path = temp_checkpoint_dir / "converted.pt"
        save_checkpoint(pytorch_checkpoint, save_path, checkpoint_format="pytorch")

        # Load and verify
        loaded_checkpoint = load_checkpoint(save_path)

        assert "model_state_dict" in loaded_checkpoint
        assert loaded_checkpoint["epoch"] == lightning_checkpoint["epoch"]

    @pytest.mark.integration
    def test_cross_format_state_dict_extraction(self, checkpoint_files: dict[str, Path]) -> None:
        """Test state dict extraction across different saved formats."""
        lightning_data = load_checkpoint(checkpoint_files["lightning"])
        pytorch_data = load_checkpoint(checkpoint_files["pytorch"])
        state_dict_data = load_checkpoint(checkpoint_files["state_dict"])

        # Extract state dicts from all formats
        lightning_state = extract_state_dict(lightning_data)
        pytorch_state = extract_state_dict(pytorch_data)
        raw_state = extract_state_dict(state_dict_data)

        # All should be dictionaries
        assert isinstance(lightning_state, dict)
        assert isinstance(pytorch_state, dict)
        assert isinstance(raw_state, dict)

        # All should contain tensors
        assert all(isinstance(v, torch.Tensor) for v in lightning_state.values())
        assert all(isinstance(v, torch.Tensor) for v in pytorch_state.values())
        assert all(isinstance(v, torch.Tensor) for v in raw_state.values())

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_checkpoint_format_operations(self, temp_checkpoint_dir: Path, large_checkpoint_data: dict) -> None:
        """Test format operations with large checkpoint data."""
        save_path = temp_checkpoint_dir / "large_checkpoint.ckpt"

        # Test saving large checkpoint
        save_checkpoint(large_checkpoint_data, save_path, checkpoint_format="lightning")

        # Test loading large checkpoint
        loaded_data = load_checkpoint(save_path)

        # Verify structure preserved
        assert "state_dict" in loaded_data
        assert "optimizer_state_dict" in loaded_data
        assert loaded_data["epoch"] == large_checkpoint_data["epoch"]

    @pytest.mark.integration
    def test_format_detection_with_real_saves(self, temp_checkpoint_dir: Path, sample_state_dict: dict) -> None:
        """Test format detection with actually saved files."""
        # Save in different formats
        lightning_path = temp_checkpoint_dir / "test.ckpt"
        pytorch_path = temp_checkpoint_dir / "test.pt"
        state_dict_path = temp_checkpoint_dir / "test.pth"

        lightning_data = {"state_dict": sample_state_dict, "epoch": 1, "pytorch-lightning_version": "2.0.0"}
        pytorch_data = {"model_state_dict": sample_state_dict, "epoch": 1}

        save_checkpoint(lightning_data, lightning_path, "lightning")
        save_checkpoint(pytorch_data, pytorch_path, "pytorch")
        save_checkpoint(sample_state_dict, state_dict_path, "state_dict")

        # Test detection
        assert detect_checkpoint_format(lightning_path) == "lightning"
        assert detect_checkpoint_format(pytorch_path) == "pytorch"
        assert detect_checkpoint_format(state_dict_path) == "state_dict"


class TestFormatEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_detect_format_empty_file(self, temp_checkpoint_dir: Path) -> None:
        """Test format detection with empty file."""
        empty_path = temp_checkpoint_dir / "empty.ckpt"
        empty_path.touch()

        # Should default to lightning for empty/unreadable files
        assert detect_checkpoint_format(empty_path) == "lightning"

    @pytest.mark.unit
    def test_load_checkpoint_corrupted_pickle(self, temp_checkpoint_dir: Path) -> None:
        """Test loading checkpoint with corrupted pickle data."""
        corrupted_path = temp_checkpoint_dir / "corrupted.pt"

        # Write invalid pickle data
        with corrupted_path.open("wb") as f:
            f.write(b"invalid pickle data")

        with pytest.raises(CheckpointLoadError):
            load_checkpoint(corrupted_path)

    @pytest.mark.unit
    def test_extract_state_dict_nested_structure(self) -> None:
        """Test state dict extraction with nested checkpoint structure."""
        nested_checkpoint = {
            "checkpoint": {
                "state_dict": {"layer.weight": torch.tensor([1.0, 2.0])},
                "epoch": 5,
            },
            "metadata": {"version": "1.0"},
        }

        # Should raise validation error since no direct state_dict key found
        with pytest.raises(CheckpointValidationError) as exc_info:
            extract_state_dict(nested_checkpoint)

        assert "Cannot find model state" in str(exc_info.value)

    @pytest.mark.unit
    def test_convert_lightning_empty_checkpoint(self) -> None:
        """Test converting empty Lightning checkpoint."""
        empty_lightning = {}

        with pytest.raises(CheckpointValidationError) as exc_info:
            convert_lightning_to_pytorch(empty_lightning)

        assert "no recognizable Lightning components" in str(exc_info.value)

    @pytest.mark.unit
    def test_save_checkpoint_creates_directory(self, pytorch_checkpoint: dict, tmp_path: Path) -> None:
        """Test saving checkpoint creates missing parent directories."""
        nested_path = tmp_path / "new_directory" / "nested" / "checkpoint.pt"

        # Should create parent directories
        save_checkpoint(pytorch_checkpoint, nested_path, checkpoint_format="pytorch")

        # Verify parent directory was created
        assert nested_path.parent.exists()
        assert nested_path.exists()
