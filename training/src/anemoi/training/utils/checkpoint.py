# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import pickle  # noqa: S403  # pickle required for PyTorch checkpoint loading
from pathlib import Path

import torch
import torch.nn as nn

# Import pipeline exceptions for consistent error handling
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.utils.checkpoints import save_metadata

LOGGER = logging.getLogger(__name__)


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    module = GraphForecaster.load_from_checkpoint(lightning_checkpoint_path)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath


def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    """Load checkpoint with transfer learning capabilities (DEPRECATED).

    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        Use the checkpoint pipeline system instead:

        .. code-block:: python

            from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext
            from omegaconf import OmegaConf

            config = OmegaConf.create({
                "stages": [
                    {
                        "_target_": "anemoi.training.checkpoint.sources.LocalSource",
                        "path": ckpt_path
                    },
                    {
                        "_target_": "anemoi.training.checkpoint.loaders.TransferLearningLoader",
                        "skip_mismatched": True,
                        "strict": False
                    }
                ]
            })

            pipeline = CheckpointPipeline.from_config(config)
            context = CheckpointContext(model=model)
            result = await pipeline.execute(context)
            model = result.model
    """
    import warnings

    warnings.warn(
        "transfer_learning_loading() is deprecated and will be removed in a future version. "
        "Use the checkpoint pipeline system with TransferLearningLoader instead. "
        "See documentation for migration guide.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Load the checkpoint with proper error handling
    # Get device from model parameters, defaulting to CPU
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else "cpu"
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError as e:
        raise CheckpointNotFoundError(ckpt_path, {"original_error": str(e)}) from e
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError, ValueError, TypeError) as e:
        raise CheckpointLoadError(ckpt_path, e) from e

    # Validate checkpoint structure
    if "state_dict" not in checkpoint:
        msg = "Invalid checkpoint format: missing 'state_dict' key"
        raise CheckpointIncompatibleError(
            msg,
            details={"available_keys": list(checkpoint.keys()), "checkpoint_path": str(ckpt_path)},
        )

    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"]
    model_state_dict = model.state_dict()
    shape_mismatches = {}

    for key in state_dict.copy():
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            shape_mismatches[key] = (model_state_dict[key].shape, state_dict[key].shape)
            del state_dict[key]  # Remove the mismatched key

    # Load the filtered state_dict into the model
    try:
        incompatible_keys = model.load_state_dict(state_dict, strict=False)

        # Log loading results for transparency
        if incompatible_keys.missing_keys:
            LOGGER.info("Missing keys in checkpoint: %s", incompatible_keys.missing_keys)
        if incompatible_keys.unexpected_keys:
            LOGGER.info("Unexpected keys in checkpoint: %s", incompatible_keys.unexpected_keys)
        if shape_mismatches:
            LOGGER.info("Shape mismatches handled: %s", list(shape_mismatches.keys()))

    except Exception as e:
        msg = f"Failed to load state dict into model: {e}"
        raise CheckpointIncompatibleError(
            msg,
            missing_keys=getattr(incompatible_keys, "missing_keys", None),
            unexpected_keys=getattr(incompatible_keys, "unexpected_keys", None),
            shape_mismatches=shape_mismatches,
            details={"checkpoint_path": str(ckpt_path)},
        ) from e
    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str, _is_recursive_call: bool = False) -> bool:
    """Recursively freezes the parameters of a submodule with the specified name (DEPRECATED).

    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        Use the checkpoint pipeline system instead:

        .. code-block:: python

            from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext
            from omegaconf import OmegaConf

            config = OmegaConf.create({
                "stages": [
                    {
                        "_target_": "anemoi.training.checkpoint.modifiers.FreezingModifier",
                        "layers": [target_name]
                    }
                ]
            })

            pipeline = CheckpointPipeline.from_config(config)
            context = CheckpointContext(model=module)
            result = await pipeline.execute(context)

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.

    Returns
    -------
    bool
        True if target module was found and frozen, False otherwise
    """
    import warnings

    warnings.warn(
        "freeze_submodule_by_name() is deprecated and will be removed in a future version. "
        "Use the checkpoint pipeline system with FreezingModifier instead. "
        "See documentation for migration guide.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Validate inputs
    if not target_name or not isinstance(target_name, str):
        msg = f"Invalid target_name: must be a non-empty string, got {target_name!r}"
        raise CheckpointConfigError(
            msg,
            details={"target_name": target_name, "type": type(target_name).__name__},
        )

    # Track if we found the target module
    found_target = False

    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            param_count = 0
            for param in child.parameters():
                param.requires_grad = False
                param_count += 1

            LOGGER.info("Frozen submodule '%s' with %d parameters", target_name, param_count)
            found_target = True
        else:
            # Recursively search within children
            if freeze_submodule_by_name(child, target_name, _is_recursive_call=True):
                found_target = True

    # Warn if target was not found (but don't raise error for backward compatibility)
    if not found_target and not _is_recursive_call:  # Only warn at top level
        available_modules = [name for name, _ in module.named_children()]
        LOGGER.warning("Target submodule '%s' not found. Available modules: %s", target_name, available_modules)

    return found_target
