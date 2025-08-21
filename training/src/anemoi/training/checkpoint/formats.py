# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint format detection and conversion utilities.

This module provides utilities for detecting checkpoint formats and converting
between different checkpoint types (Lightning, PyTorch, safetensors, state_dict).
"""

from __future__ import annotations

import pickle  # noqa: S403 - Required for PyTorch checkpoint loading
from pathlib import Path
from typing import Any
from typing import Literal

import torch

# Optional import for safetensors
try:
    import safetensors.torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def detect_checkpoint_format(
    checkpoint_path: Path | str,
) -> Literal["lightning", "pytorch", "safetensors", "state_dict"]:
    """Detect the format of a checkpoint file.

    Uses file extension and structure inspection to determine format.

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to the checkpoint file

    Returns
    -------
    str
        Format of the checkpoint: "lightning", "pytorch", "safetensors", or "state_dict"
    """
    path = Path(checkpoint_path)

    # Check file extension first
    extension = path.suffix.lower()

    if extension == ".safetensors":
        return "safetensors"

    # For other extensions, load and inspect structure
    if extension in [".ckpt", ".pt", ".pth", ".bin"]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                # Non-dict checkpoint, likely a raw model
                return "pytorch"

            # Check for Lightning-specific keys
            lightning_keys = {
                "pytorch-lightning_version",
                "callbacks",
                "optimizer_states",
                "lr_schedulers",
                "epoch",
                "global_step",
                "loops",
            }

            if any(key in checkpoint for key in lightning_keys):
                return "lightning"

            # If it's just a dict of tensors, it's a state dict
            if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return "state_dict"
            # Default to pytorch for structured checkpoints
            return "pytorch"  # noqa: TRY300

        except (OSError, RuntimeError, pickle.UnpicklingError):
            # If we can't load it (file corruption, etc.), default to lightning
            return "lightning"

    # Default to lightning for unknown extensions
    return "lightning"


def load_checkpoint(
    checkpoint_path: Path | str,
    checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint file in any supported format.

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to the checkpoint file
    checkpoint_format : str, optional
        Format of the checkpoint. If None, will auto-detect.

    Returns
    -------
    dict
        Loaded checkpoint data
    """
    path = Path(checkpoint_path)

    if checkpoint_format is None:
        checkpoint_format = detect_checkpoint_format(path)

    if checkpoint_format == "safetensors":
        if not HAS_SAFETENSORS:
            msg = "safetensors is required to load safetensors checkpoints. Install with: pip install safetensors"
            raise ImportError(msg)
        return safetensors.torch.load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Extract the state dict from a checkpoint.

    Handles different checkpoint structures.

    Parameters
    ----------
    checkpoint_data : dict
        Loaded checkpoint data

    Returns
    -------
    dict
        Extracted state dictionary
    """
    if "state_dict" in checkpoint_data:
        return checkpoint_data["state_dict"]
    if "model_state_dict" in checkpoint_data:
        return checkpoint_data["model_state_dict"]
    if "model" in checkpoint_data:
        return checkpoint_data["model"]
    # Assume the checkpoint itself is the state dict
    return checkpoint_data


def save_checkpoint(
    checkpoint_data: dict[str, Any],
    checkpoint_path: Path | str,
    checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"] = "pytorch",
) -> None:
    """Save a checkpoint in the specified format.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data to save
    checkpoint_path : Path or str
        Path where to save the checkpoint
    checkpoint_format : str
        Format to save in: "lightning", "pytorch", "safetensors", or "state_dict"
    """
    path = Path(checkpoint_path)

    if checkpoint_format == "safetensors":
        if not HAS_SAFETENSORS:
            msg = "safetensors is required to save safetensors checkpoints. Install with: pip install safetensors"
            raise ImportError(msg)
        # Extract just the state dict for safetensors
        state_dict = extract_state_dict(checkpoint_data)
        safetensors.torch.save_file(state_dict, str(path))
    else:
        torch.save(checkpoint_data, path)


def convert_lightning_to_pytorch(
    lightning_checkpoint: dict[str, Any],
    extract_model_only: bool = True,
) -> dict[str, Any]:
    """Convert a Lightning checkpoint to PyTorch format.

    Parameters
    ----------
    lightning_checkpoint : dict
        Lightning checkpoint data
    extract_model_only : bool
        If True, extract only model weights. If False, keep optimizer/scheduler state.

    Returns
    -------
    dict
        PyTorch format checkpoint
    """
    pytorch_checkpoint = {}

    # Extract model state
    if "state_dict" in lightning_checkpoint:
        pytorch_checkpoint["model_state_dict"] = lightning_checkpoint["state_dict"]

    if not extract_model_only:
        # Keep training state if requested
        if "optimizer_states" in lightning_checkpoint:
            pytorch_checkpoint["optimizer_state_dict"] = lightning_checkpoint["optimizer_states"]
        if "lr_schedulers" in lightning_checkpoint:
            pytorch_checkpoint["scheduler_state_dict"] = lightning_checkpoint["lr_schedulers"]
        if "epoch" in lightning_checkpoint:
            pytorch_checkpoint["epoch"] = lightning_checkpoint["epoch"]
        if "global_step" in lightning_checkpoint:
            pytorch_checkpoint["global_step"] = lightning_checkpoint["global_step"]

    return pytorch_checkpoint


def is_format_available(checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"]) -> bool:
    """Check if a checkpoint format is available for use.

    Parameters
    ----------
    checkpoint_format : str
        Format to check: "lightning", "pytorch", "safetensors", or "state_dict"

    Returns
    -------
    bool
        True if the format is available
    """
    if checkpoint_format == "safetensors":
        return HAS_SAFETENSORS
    return True  # All other formats are always available
