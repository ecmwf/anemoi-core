# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint utility functions."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
import torch


from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.utils import calculate_checksum
from anemoi.training.checkpoint.utils import compare_state_dicts
from anemoi.training.checkpoint.utils import estimate_checkpoint_memory
from anemoi.training.checkpoint.utils import format_size
from anemoi.training.checkpoint.utils import get_checkpoint_metadata
from anemoi.training.checkpoint.utils import validate_checkpoint


class TestValidateCheckpoint:
    """Test checkpoint validation functionality."""

    @pytest.mark.unit
    def test_validate_checkpoint_valid_lightning(self, lightning_checkpoint: dict) -> None:
        """Test validation of valid Lightning checkpoint."""
        assert validate_checkpoint(lightning_checkpoint) is True

    @pytest.mark.unit
    def test_validate_checkpoint_valid_pytorch(self, pytorch_checkpoint: dict) -> None:
        """Test validation of valid PyTorch checkpoint."""
        assert validate_checkpoint(pytorch_checkpoint) is True

    @pytest.mark.unit
    def test_validate_checkpoint_valid_state_dict(self, sample_state_dict: dict) -> None:
        """Test validation of valid state dict."""
        # Wrap state dict to include a recognizable key
        checkpoint = {"state_dict": sample_state_dict}
        assert validate_checkpoint(checkpoint) is True

    @pytest.mark.unit
    def test_validate_checkpoint_empty(self) -> None:
        """Test validation of empty checkpoint."""
        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint({})

        assert "Checkpoint is empty" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_checkpoint_missing_model_keys(self) -> None:
        """Test validation failure when no model keys present."""
        invalid_checkpoint = {
            "epoch": 5,
            "optimizer_state_dict": {},
        }

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(invalid_checkpoint)

        assert "No model state found" in str(exc_info.value)
        assert "state_dict" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_checkpoint_nan_tensors(self, sample_state_dict: dict) -> None:
        """Test validation failure with NaN tensors."""
        # Create checkpoint with NaN values
        corrupted_state_dict = sample_state_dict.copy()
        corrupted_state_dict["corrupted_tensor"] = torch.tensor([1.0, float("nan"), 3.0])

        checkpoint = {"state_dict": corrupted_state_dict}

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint, check_tensors=True)

        assert "NaN values" in str(exc_info.value)
        assert "corrupted_tensor" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_checkpoint_inf_tensors(self, sample_state_dict: dict) -> None:
        """Test validation failure with infinite tensors."""
        # Create checkpoint with Inf values
        corrupted_state_dict = sample_state_dict.copy()
        corrupted_state_dict["inf_tensor"] = torch.tensor([1.0, float("inf"), 3.0])

        checkpoint = {"state_dict": corrupted_state_dict}

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint, check_tensors=True)

        assert "infinite values" in str(exc_info.value)
        assert "inf_tensor" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_checkpoint_nested_tensors(self) -> None:
        """Test validation of nested tensor structures."""
        checkpoint = {
            "model_state_dict": {
                "layer1": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.1])},
                "layer2": {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            },
        }

        assert validate_checkpoint(checkpoint, check_tensors=True) is True

    @pytest.mark.unit
    def test_validate_checkpoint_nested_nan_tensors(self) -> None:
        """Test validation failure with NaN in nested structures."""
        checkpoint = {
            "model_state_dict": {
                "layer1": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([float("nan")])},
            },
        }

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint, check_tensors=True)

        assert "model_state_dict.layer1.bias" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_checkpoint_multiple_errors(self) -> None:
        """Test validation with multiple errors."""
        checkpoint = {
            "state_dict": {
                "nan_tensor": torch.tensor([float("nan")]),
                "inf_tensor": torch.tensor([float("inf")]),
            },
        }

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint, check_tensors=True)

        # Should capture both errors
        error_str = str(exc_info.value)
        assert "nan_tensor" in error_str
        assert "inf_tensor" in error_str

    @pytest.mark.unit
    def test_validate_checkpoint_mixed_tensor_non_tensor(self, sample_state_dict: dict) -> None:
        """Test validation with mixed tensor and non-tensor data."""
        checkpoint = {
            "state_dict": sample_state_dict,
            "epoch": 5,
            "metadata": {"version": "1.0"},
            "optimizer_state_dict": {
                "param_groups": [{"lr": 1e-3}],
                "state": {},
            },
        }

        assert validate_checkpoint(checkpoint) is True


class TestGetCheckpointMetadata:
    """Test checkpoint metadata extraction."""

    @pytest.mark.unit
    def test_get_checkpoint_metadata_lightning(self, checkpoint_files: dict[str, Path]) -> None:
        """Test metadata extraction from Lightning checkpoint."""
        metadata = get_checkpoint_metadata(checkpoint_files["lightning"])

        assert isinstance(metadata, dict)
        assert "epoch" in metadata
        assert "global_step" in metadata
        assert "file_size_mb" in metadata
        assert "file_path" in metadata
        assert "num_parameters" in metadata

        assert metadata["epoch"] == 10  # From fixture
        assert metadata["global_step"] == 1000  # From fixture
        assert metadata["file_size_mb"] > 0

    @pytest.mark.unit
    def test_get_checkpoint_metadata_pytorch(self, checkpoint_files: dict[str, Path]) -> None:
        """Test metadata extraction from PyTorch checkpoint."""
        metadata = get_checkpoint_metadata(checkpoint_files["pytorch"])

        assert isinstance(metadata, dict)
        assert "epoch" in metadata
        assert "global_step" in metadata
        assert metadata["epoch"] == 5  # From fixture
        assert metadata["global_step"] == 500  # From fixture

    @pytest.mark.unit
    def test_get_checkpoint_metadata_state_dict(self, checkpoint_files: dict[str, Path]) -> None:
        """Test metadata extraction from raw state dict."""
        metadata = get_checkpoint_metadata(checkpoint_files["state_dict"])

        assert isinstance(metadata, dict)
        assert "file_size_mb" in metadata
        assert "file_path" in metadata
        # Raw state dict shouldn't have epoch/step info
        assert "epoch" not in metadata
        assert "global_step" not in metadata

    @pytest.mark.unit
    def test_get_checkpoint_metadata_nonexistent_file(self) -> None:
        """Test metadata extraction from nonexistent file."""
        nonexistent_path = Path("/nonexistent/checkpoint.ckpt")

        with pytest.raises(CheckpointLoadError) as exc_info:
            get_checkpoint_metadata(nonexistent_path)

        assert "File not found" in str(exc_info.value.original_error)

    @pytest.mark.unit
    def test_get_checkpoint_metadata_corrupted_file(self, temp_checkpoint_dir: Path) -> None:
        """Test metadata extraction from corrupted file."""
        corrupted_path = temp_checkpoint_dir / "corrupted.ckpt"

        # Write invalid data
        with corrupted_path.open("wb") as f:
            f.write(b"not a valid checkpoint")

        with pytest.raises(CheckpointLoadError):
            get_checkpoint_metadata(corrupted_path)

    @pytest.mark.unit
    def test_get_checkpoint_metadata_with_nested_metadata(
        self,
        temp_checkpoint_dir: Path,
        sample_state_dict: dict,
    ) -> None:
        """Test metadata extraction with nested metadata field."""
        checkpoint_with_meta = {
            "state_dict": sample_state_dict,
            "epoch": 42,
            "metadata": {
                "model_name": "test_model",
                "training_config": {"lr": 1e-4},
            },
        }

        checkpoint_path = temp_checkpoint_dir / "with_meta.ckpt"
        torch.save(checkpoint_with_meta, checkpoint_path)

        metadata = get_checkpoint_metadata(checkpoint_path)

        assert metadata["epoch"] == 42
        assert "model_name" in metadata
        assert "training_config" in metadata

    @pytest.mark.unit
    @pytest.mark.slow
    def test_get_checkpoint_metadata_large_file(self, temp_checkpoint_dir: Path, large_checkpoint_data: dict) -> None:
        """Test metadata extraction from large checkpoint file."""
        large_path = temp_checkpoint_dir / "large.ckpt"
        torch.save(large_checkpoint_data, large_path)

        start_time = time.time()
        metadata = get_checkpoint_metadata(large_path)
        extraction_time = time.time() - start_time

        assert isinstance(metadata, dict)
        assert "epoch" in metadata
        assert metadata["epoch"] == 50  # From fixture

        # Should be fast since we don't load full model weights
        assert extraction_time < 10.0  # Should be much faster than full loading


class TestCalculateChecksum:
    """Test file checksum calculation."""

    @pytest.mark.unit
    def test_calculate_checksum_sha256(self, checkpoint_files: dict[str, Path]) -> None:
        """Test SHA256 checksum calculation."""
        checksum = calculate_checksum(checkpoint_files["lightning"], algorithm="sha256")

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in checksum)

    @pytest.mark.unit
    def test_calculate_checksum_md5(self, checkpoint_files: dict[str, Path]) -> None:
        """Test MD5 checksum calculation."""
        checksum = calculate_checksum(checkpoint_files["lightning"], algorithm="md5")

        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hex digest length

    @pytest.mark.unit
    def test_calculate_checksum_consistency(self, temp_checkpoint_dir: Path) -> None:
        """Test checksum consistency for same file."""
        test_path = temp_checkpoint_dir / "checksum_test.bin"
        test_data = b"test data for checksum calculation"

        test_path.write_bytes(test_data)

        checksum1 = calculate_checksum(test_path)
        checksum2 = calculate_checksum(test_path)

        assert checksum1 == checksum2

    @pytest.mark.unit
    def test_calculate_checksum_different_algorithms(self, temp_checkpoint_dir: Path) -> None:
        """Test different checksum algorithms on same file."""
        test_path = temp_checkpoint_dir / "multi_algo_test.bin"
        test_data = b"test data"

        test_path.write_bytes(test_data)

        sha256_sum = calculate_checksum(test_path, algorithm="sha256")
        md5_sum = calculate_checksum(test_path, algorithm="md5")
        sha1_sum = calculate_checksum(test_path, algorithm="sha1")

        assert len(sha256_sum) == 64
        assert len(md5_sum) == 32
        assert len(sha1_sum) == 40

        # Different algorithms should produce different results
        assert sha256_sum != md5_sum
        assert sha256_sum != sha1_sum
        assert md5_sum != sha1_sum

    @pytest.mark.unit
    def test_calculate_checksum_large_file(self, temp_checkpoint_dir: Path) -> None:
        """Test checksum calculation on large file."""
        large_path = temp_checkpoint_dir / "large_checksum.bin"

        # Create file larger than chunk size
        chunk_size = 8192
        large_data = b"x" * (chunk_size * 10)  # 80KB file
        large_path.write_bytes(large_data)

        checksum = calculate_checksum(large_path)

        # Verify by calculating expected checksum
        expected = hashlib.sha256(large_data).hexdigest()
        assert checksum == expected

    @pytest.mark.unit
    def test_calculate_checksum_empty_file(self, temp_checkpoint_dir: Path) -> None:
        """Test checksum calculation on empty file."""
        empty_path = temp_checkpoint_dir / "empty.bin"
        empty_path.touch()

        checksum = calculate_checksum(empty_path)

        # SHA256 of empty data
        expected = hashlib.sha256(b"").hexdigest()
        assert checksum == expected


class TestCompareStateDicts:
    """Test state dictionary comparison utilities."""

    @pytest.mark.unit
    def test_compare_state_dicts_identical(self, sample_state_dict: dict) -> None:
        """Test comparison of identical state dicts."""
        missing, unexpected, mismatches = compare_state_dicts(sample_state_dict, sample_state_dict)

        assert len(missing) == 0
        assert len(unexpected) == 0
        assert len(mismatches) == 0

    @pytest.mark.unit
    def test_compare_state_dicts_missing_keys(self, sample_state_dict: dict) -> None:
        """Test comparison with missing keys."""
        source_dict = {k: v for k, v in sample_state_dict.items() if "linear1" not in k}

        missing, unexpected, _mismatches = compare_state_dicts(source_dict, sample_state_dict)

        assert len(missing) > 0
        assert len(unexpected) == 0
        assert any("linear1" in key for key in missing)

    @pytest.mark.unit
    def test_compare_state_dicts_unexpected_keys(self, sample_state_dict: dict) -> None:
        """Test comparison with unexpected keys."""
        source_dict = sample_state_dict.copy()
        source_dict["extra_layer.weight"] = torch.tensor([1.0, 2.0, 3.0])

        missing, unexpected, _mismatches = compare_state_dicts(source_dict, sample_state_dict)

        assert len(missing) == 0
        assert len(unexpected) == 1
        assert "extra_layer.weight" in unexpected

    @pytest.mark.unit
    def test_compare_state_dicts_shape_mismatches(self, sample_state_dict: dict) -> None:
        """Test comparison with shape mismatches."""
        source_dict = sample_state_dict.copy()
        # Change shape of an existing tensor
        for key, tensor in source_dict.items():
            if tensor.dim() > 1:
                # Change the shape by truncating one dimension
                source_dict[key] = tensor[: tensor.shape[0] // 2]
                break

        missing, unexpected, mismatches = compare_state_dicts(source_dict, sample_state_dict)

        assert len(missing) == 0
        assert len(unexpected) == 0
        assert len(mismatches) > 0

    @pytest.mark.unit
    def test_compare_state_dicts_complex_differences(self, sample_state_dict: dict, complex_state_dict: dict) -> None:
        """Test comparison with multiple types of differences."""
        missing, unexpected, mismatches = compare_state_dicts(sample_state_dict, complex_state_dict)

        # Should have differences since models are different architectures
        total_differences = len(missing) + len(unexpected) + len(mismatches)
        assert total_differences > 0

    @pytest.mark.unit
    def test_compare_state_dicts_empty_dicts(self) -> None:
        """Test comparison of empty state dicts."""
        empty_dict = {}

        missing, unexpected, mismatches = compare_state_dicts(empty_dict, empty_dict)

        assert len(missing) == 0
        assert len(unexpected) == 0
        assert len(mismatches) == 0

    @pytest.mark.unit
    def test_compare_state_dicts_one_empty(self, sample_state_dict: dict) -> None:
        """Test comparison with one empty state dict."""
        empty_dict = {}

        missing, unexpected, mismatches = compare_state_dicts(empty_dict, sample_state_dict)

        assert len(missing) == len(sample_state_dict)
        assert len(unexpected) == 0
        assert len(mismatches) == 0


class TestFormatSize:
    """Test human-readable size formatting."""

    @pytest.mark.unit
    def test_format_size_bytes(self) -> None:
        """Test formatting bytes."""
        assert format_size(512) == "512.00 B"
        assert format_size(1000) == "1000.00 B"

    @pytest.mark.unit
    def test_format_size_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert format_size(1536) == "1.50 KB"  # 1.5 KB
        assert format_size(2048) == "2.00 KB"  # 2 KB

    @pytest.mark.unit
    def test_format_size_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert format_size(1572864) == "1.50 MB"  # 1.5 MB
        assert format_size(2097152) == "2.00 MB"  # 2 MB

    @pytest.mark.unit
    def test_format_size_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert format_size(1610612736) == "1.50 GB"  # 1.5 GB
        assert format_size(2147483648) == "2.00 GB"  # 2 GB

    @pytest.mark.unit
    def test_format_size_terabytes(self) -> None:
        """Test formatting terabytes."""
        assert format_size(1649267441664) == "1.50 TB"  # 1.5 TB

    @pytest.mark.unit
    def test_format_size_petabytes(self) -> None:
        """Test formatting petabytes."""
        extremely_large = 1024**5 * 2  # 2 PB
        result = format_size(extremely_large)
        assert "PB" in result

    @pytest.mark.unit
    def test_format_size_zero(self) -> None:
        """Test formatting zero size."""
        assert format_size(0) == "0.00 B"

    @pytest.mark.unit
    def test_format_size_precision(self) -> None:
        """Test formatting precision."""
        # Test that it shows 2 decimal places
        assert format_size(1536) == "1.50 KB"
        assert format_size(1000000) == "976.56 KB"  # 1MB in binary


class TestEstimateCheckpointMemory:
    """Test checkpoint memory estimation."""

    @pytest.mark.unit
    def test_estimate_checkpoint_memory_simple(self, sample_state_dict: dict) -> None:
        """Test memory estimation for simple state dict."""
        checkpoint = {"state_dict": sample_state_dict}

        memory_bytes = estimate_checkpoint_memory(checkpoint)

        assert isinstance(memory_bytes, int)
        assert memory_bytes > 0

    @pytest.mark.unit
    def test_estimate_checkpoint_memory_complex(self, lightning_checkpoint: dict) -> None:
        """Test memory estimation for complex checkpoint."""
        memory_bytes = estimate_checkpoint_memory(lightning_checkpoint)

        assert isinstance(memory_bytes, int)
        assert memory_bytes > 0

    @pytest.mark.unit
    def test_estimate_checkpoint_memory_nested(self) -> None:
        """Test memory estimation with nested structures."""
        checkpoint = {
            "model_state_dict": {
                "layer1": {"weight": torch.randn(100, 50), "bias": torch.randn(100)},
                "layer2": {"weight": torch.randn(10, 100)},
            },
            "optimizer_state_dict": {
                "state": {
                    0: {"momentum_buffer": torch.randn(100, 50)},
                    1: {"momentum_buffer": torch.randn(100)},
                },
                "param_groups": [{"lr": 1e-3}],  # Non-tensor, should be ignored
            },
        }

        memory_bytes = estimate_checkpoint_memory(checkpoint)

        # Should account for all tensors in nested structure
        expected_elements = 100 * 50 + 100 + 10 * 100 + 100 * 50 + 100  # All tensor elements
        expected_bytes = expected_elements * 4  # float32 = 4 bytes per element

        assert memory_bytes == expected_bytes

    @pytest.mark.unit
    def test_estimate_checkpoint_memory_empty(self) -> None:
        """Test memory estimation for empty checkpoint."""
        empty_checkpoint = {}

        memory_bytes = estimate_checkpoint_memory(empty_checkpoint)

        assert memory_bytes == 0

    @pytest.mark.unit
    def test_estimate_checkpoint_memory_non_tensors(self) -> None:
        """Test memory estimation ignoring non-tensor data."""
        checkpoint = {
            "epoch": 5,
            "metadata": {"version": "1.0"},
            "config": {"lr": 1e-3, "batch_size": 32},
        }

        memory_bytes = estimate_checkpoint_memory(checkpoint)

        assert memory_bytes == 0  # No tensors, so 0 bytes

    @pytest.mark.unit
    @pytest.mark.slow
    def test_estimate_checkpoint_memory_large(self, large_checkpoint_data: dict) -> None:
        """Test memory estimation for large checkpoint."""
        memory_bytes = estimate_checkpoint_memory(large_checkpoint_data)

        assert isinstance(memory_bytes, int)
        assert memory_bytes > 1000000  # Should be > 1MB for large checkpoint

        # Verify the estimation is reasonable
        formatted_size = format_size(memory_bytes)
        assert "MB" in formatted_size or "GB" in formatted_size


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    @pytest.mark.integration
    def test_validate_and_extract_metadata_cycle(self, checkpoint_files: dict[str, Path]) -> None:
        """Test validation followed by metadata extraction."""
        # Only test proper checkpoint formats (not raw state_dict which lacks wrapper keys)
        for name, checkpoint_path in checkpoint_files.items():
            if name == "state_dict":
                # Raw state dict doesn't have model key wrapper, skip validation
                continue

            # Load and validate checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if checkpoint_data:  # Skip if somehow empty
                assert validate_checkpoint(checkpoint_data) is True

                # Extract metadata
                metadata = get_checkpoint_metadata(checkpoint_path)
                assert isinstance(metadata, dict)
                assert "file_size_mb" in metadata

    @pytest.mark.integration
    def test_checksum_and_metadata_correlation(self, checkpoint_files: dict[str, Path]) -> None:
        """Test that checksum changes when file content changes."""
        original_path = checkpoint_files["pytorch"]

        # Calculate original checksum
        original_checksum = calculate_checksum(original_path)
        original_metadata = get_checkpoint_metadata(original_path)

        # Modify the checkpoint
        checkpoint_data = torch.load(original_path, map_location="cpu", weights_only=False)
        checkpoint_data["modified"] = True

        modified_path = original_path.parent / "modified_pytorch.pt"
        torch.save(checkpoint_data, modified_path)

        # Calculate new checksum and metadata
        modified_checksum = calculate_checksum(modified_path)
        modified_metadata = get_checkpoint_metadata(modified_path)

        # Checksum should be different
        assert original_checksum != modified_checksum

        # File size should be different
        assert original_metadata["file_size_mb"] != modified_metadata["file_size_mb"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_estimation_vs_actual_loading(self, temp_checkpoint_dir: Path, large_checkpoint_data: dict) -> None:
        """Test that memory estimation correlates with actual memory usage."""
        checkpoint_path = temp_checkpoint_dir / "memory_test.ckpt"
        torch.save(large_checkpoint_data, checkpoint_path)

        # Estimate memory
        estimated_bytes = estimate_checkpoint_memory(large_checkpoint_data)

        # Get actual file size
        actual_file_size = checkpoint_path.stat().st_size

        # Memory estimate should be in reasonable relation to file size
        # (Typically smaller due to compression and metadata overhead)
        assert 0.1 * actual_file_size <= estimated_bytes <= actual_file_size
