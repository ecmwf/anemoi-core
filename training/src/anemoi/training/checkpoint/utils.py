# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utility functions for checkpoint operations."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path  # noqa: TC003 - Used for actual Path operations, not just type hints
from typing import Any

import aiohttp
import torch

from .exceptions import CheckpointLoadError
from .exceptions import CheckpointSourceError
from .exceptions import CheckpointTimeoutError
from .exceptions import CheckpointValidationError

LOGGER = logging.getLogger(__name__)


async def download_with_retry(
    url: str,
    dest: Path,
    max_retries: int = 3,
    timeout: int = 300,
    chunk_size: int = 8192,
) -> Path:
    """Download file with exponential backoff retry.

    Downloads a file from a URL to a destination path with automatic
    retry on failure using exponential backoff.

    Parameters
    ----------
    url : str
        URL to download from
    dest : Path
        Destination path for downloaded file
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)
    timeout : int, optional
        Timeout in seconds for each attempt (default: 300)
    chunk_size : int, optional
        Size of chunks to download (default: 8192)

    Returns
    -------
    Path
        Path to downloaded file

    Raises
    ------
    CheckpointSourceError
        If download fails after all retries
    CheckpointTimeoutError
        If download times out

    Examples
    --------
    >>> import asyncio
    >>> async def download():
    ...     path = await download_with_retry(
    ...         "https://example.com/model.ckpt",
    ...         Path("/tmp/model.ckpt")
    ...     )
    ...     return path
    >>> asyncio.run(download())
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            result = await _attempt_download(url, dest, timeout, chunk_size, attempt + 1, max_retries)
            if result:
                return result
        except asyncio.TimeoutError as e:
            _handle_timeout_error(e, attempt, max_retries, url, timeout)
        except aiohttp.ClientError as e:
            _handle_client_error(e, attempt, max_retries, url)
        except Exception as e:
            LOGGER.exception("Unexpected error during download")
            msg = "http"
            raise CheckpointSourceError(msg, url, e, {"attempts": attempt + 1}) from e

        # Exponential backoff
        if attempt < max_retries - 1:
            wait_time = 2**attempt
            LOGGER.info("Waiting %ds before retry", wait_time)
            await asyncio.sleep(wait_time)

    # Should not reach here, but just in case
    msg = "http"
    raise CheckpointSourceError(msg, url, None, {"attempts": max_retries, "reason": "All retries exhausted"})


async def _attempt_download(
    url: str,
    dest: Path,
    timeout: int,
    chunk_size: int,
    attempt: int,
    max_retries: int,
) -> Path | None:
    """Attempt a single download."""
    LOGGER.info("Download attempt %d/%d for %s", attempt, max_retries, url)

    timeout_config = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_config) as session, session.get(url) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with dest.open("wb") as f:
            async for chunk in response.content.iter_chunked(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                        LOGGER.debug("Download progress: %.1f%%", progress)

    LOGGER.info("Successfully downloaded %s to %s", url, dest)
    return dest


def _handle_timeout_error(e: asyncio.TimeoutError, attempt: int, max_retries: int, url: str, timeout: int) -> None:
    """Handle timeout errors during download."""
    LOGGER.warning("Download timeout on attempt %d/%d", attempt + 1, max_retries)
    if attempt == max_retries - 1:
        msg = f"Download of {url}"
        raise CheckpointTimeoutError(
            msg,
            timeout,
            {"url": url, "attempts": max_retries},
        ) from e


def _handle_client_error(e: aiohttp.ClientError, attempt: int, max_retries: int, url: str) -> None:
    """Handle client errors during download."""
    LOGGER.warning("Download failed on attempt %d/%d: %s", attempt + 1, max_retries, e)
    if attempt == max_retries - 1:
        msg = "http"
        raise CheckpointSourceError(msg, url, e, {"attempts": max_retries}) from e


def validate_checkpoint(checkpoint_data: dict[str, Any]) -> bool:
    """Validate checkpoint structure and contents.

    Performs validation checks on a loaded checkpoint to ensure
    it contains expected keys and valid data.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data dictionary to validate

    Returns
    -------
    bool
        True if checkpoint is valid

    Raises
    ------
    CheckpointValidationError
        If validation fails

    Examples
    --------
    >>> checkpoint = torch.load('model.ckpt')
    >>> is_valid = validate_checkpoint(checkpoint)
    """
    validation_errors = []

    # Check for empty checkpoint
    if not checkpoint_data:
        validation_errors.append("Checkpoint is empty")

    # Check if at least one model key exists
    _validate_model_keys(checkpoint_data, validation_errors)

    # Check for corrupt tensors
    _validate_tensors(checkpoint_data, validation_errors)

    if validation_errors:
        msg = "Checkpoint validation failed"
        raise CheckpointValidationError(
            msg,
            validation_errors,
            {"num_keys": len(checkpoint_data)},
        )

    LOGGER.debug("Checkpoint validation passed with %d keys", len(checkpoint_data))
    return True


def _validate_model_keys(checkpoint_data: dict[str, Any], validation_errors: list[str]) -> None:
    """Validate that checkpoint contains model state."""
    model_keys = ["state_dict", "model_state_dict", "model"]
    if not any(key in checkpoint_data for key in model_keys):
        validation_errors.append(f"No model state found. Expected one of: {model_keys}")


def _validate_tensors(checkpoint_data: dict[str, Any], validation_errors: list[str]) -> None:
    """Validate tensor values in checkpoint."""
    for key, value in checkpoint_data.items():
        if isinstance(value, torch.Tensor):
            _check_tensor_validity(key, value, validation_errors)
        elif isinstance(value, dict):
            _validate_nested_tensors(key, value, validation_errors)


def _check_tensor_validity(name: str, tensor: torch.Tensor, validation_errors: list[str]) -> None:
    """Check a single tensor for NaN/Inf values."""
    if torch.isnan(tensor).any():
        validation_errors.append(f"Tensor '{name}' contains NaN values")
    if torch.isinf(tensor).any():
        validation_errors.append(f"Tensor '{name}' contains Inf values")


def _validate_nested_tensors(parent_key: str, nested_dict: dict, validation_errors: list[str]) -> None:
    """Validate tensors in nested dictionaries."""
    for sub_key, sub_value in nested_dict.items():
        if isinstance(sub_value, torch.Tensor):
            full_key = f"{parent_key}.{sub_key}"
            _check_tensor_validity(full_key, sub_value, validation_errors)


def get_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Extract metadata without loading full checkpoint.

    Loads only the metadata from a checkpoint file without loading
    the full model weights, which can save memory for large models.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    dict
        Checkpoint metadata dictionary

    Raises
    ------
    CheckpointLoadError
        If checkpoint cannot be loaded

    Examples
    --------
    >>> metadata = get_checkpoint_metadata(Path('model.ckpt'))
    >>> print(f"Epoch: {metadata.get('epoch', 'unknown')}")
    """
    if not checkpoint_path.exists():
        raise CheckpointLoadError(
            checkpoint_path,
            FileNotFoundError(f"File not found: {checkpoint_path}"),
            {"exists": False},
        )

    try:
        # Load checkpoint with weights on CPU to save GPU memory
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,  # Need to load optimizer states etc.
        )

        # Extract metadata (non-tensor data)
        metadata = {}
        for key, value in checkpoint.items():
            if not isinstance(value, (torch.Tensor, dict)) or key in [
                "epoch",
                "global_step",
                "iteration",
                "best_score",
            ]:
                metadata[key] = value
            elif key == "metadata" and isinstance(value, dict):
                metadata.update(value)

        # Add file information
        metadata["file_size_mb"] = checkpoint_path.stat().st_size / (1024 * 1024)
        metadata["file_path"] = str(checkpoint_path)

        # Count parameters if state dict exists
        if "state_dict" in checkpoint:
            metadata["num_parameters"] = len(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            metadata["num_parameters"] = len(checkpoint["model_state_dict"])

        return metadata  # noqa: TRY300

    except (OSError, RuntimeError, KeyError, TypeError) as e:
        raise CheckpointLoadError(checkpoint_path, e, {"operation": "extract_metadata"}) from e


def calculate_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to file
    algorithm : str, optional
        Hash algorithm to use (default: 'sha256')

    Returns
    -------
    str
        Hexadecimal checksum string

    Examples
    --------
    >>> checksum = calculate_checksum(Path('model.ckpt'))
    >>> print(f"SHA256: {checksum}")
    """
    hash_func = hashlib.new(algorithm)

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def compare_state_dicts(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[set, set, dict[str, tuple[torch.Size, torch.Size]]]:
    """Compare two state dictionaries.

    Compares keys and shapes between source and target state dictionaries
    to identify missing keys, unexpected keys, and shape mismatches.

    Parameters
    ----------
    source : dict
        Source state dictionary
    target : dict
        Target state dictionary

    Returns
    -------
    tuple
        (missing_keys, unexpected_keys, shape_mismatches)
        where shape_mismatches is {key: (source_shape, target_shape)}

    Examples
    --------
    >>> missing, unexpected, mismatches = compare_state_dicts(
    ...     checkpoint['state_dict'],
    ...     model.state_dict()
    ... )
    """
    source_keys = set(source.keys())
    target_keys = set(target.keys())

    missing_keys = target_keys - source_keys
    unexpected_keys = source_keys - target_keys

    shape_mismatches = {}
    for key in source_keys.intersection(target_keys):
        source_shape = source[key].shape
        target_shape = target[key].shape

        if source_shape != target_shape:
            shape_mismatches[key] = (source_shape, target_shape)

    return missing_keys, unexpected_keys, shape_mismatches


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string (e.g., '1.5 GB')

    Examples
    --------
    >>> format_size(1536000000)
    '1.43 GB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_checkpoint_memory(checkpoint_data: dict[str, Any]) -> int:
    """Estimate memory usage of checkpoint data.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data dictionary

    Returns
    -------
    int
        Estimated memory usage in bytes

    Examples
    --------
    >>> mem_bytes = estimate_checkpoint_memory(checkpoint)
    >>> print(f"Estimated memory: {format_size(mem_bytes)}")
    """
    total_bytes = 0

    def estimate_tensor_size(tensor: torch.Tensor) -> int:
        """Estimate memory size of a tensor."""
        return tensor.numel() * tensor.element_size()

    def estimate_dict_size(d: dict) -> int:
        """Recursively estimate dictionary size."""
        size = 0
        for value in d.values():
            if isinstance(value, torch.Tensor):
                size += estimate_tensor_size(value)
            elif isinstance(value, dict):
                size += estimate_dict_size(value)
        return size

    for value in checkpoint_data.values():
        if isinstance(value, torch.Tensor):
            total_bytes += estimate_tensor_size(value)
        elif isinstance(value, dict):
            total_bytes += estimate_dict_size(value)

    return total_bytes
