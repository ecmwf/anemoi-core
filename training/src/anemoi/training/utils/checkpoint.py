# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from anemoi.models.migrations import Migrator
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint
from anemoi.utils.checkpoints import save_metadata

chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
trainable_edge_perm_fix_migration = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate

LOGGER = logging.getLogger(__name__)


def _filter_state_dict_size_mismatches(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    for key in list(state_dict):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]


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
    module = BaseTrainingModule.load_from_checkpoint(lightning_checkpoint_path, weights_only=False)
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

    metadata = _add_reconstruction_metadata(dict(metadata), model)

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath


def _add_reconstruction_metadata(metadata: dict, model: torch.nn.Module) -> dict:
    """Inject the pickle-free reconstruction bundle into the checkpoint metadata.

    The bundle (serialised data indices, timestep counts, graph structure) lets a model be
    rebuilt from config + state_dict without unpickling. Best-effort: if ``model`` is not a
    full ``AnemoiModelInterface`` the metadata is returned unchanged. See no-pickle docs.
    """
    from anemoi.models.checkpoint import RECONSTRUCTION_KEY
    from anemoi.models.checkpoint import build_reconstruction_metadata_from_interface

    try:
        metadata[RECONSTRUCTION_KEY] = build_reconstruction_metadata_from_interface(model)
    except Exception as err:  # noqa: BLE001 - never block checkpoint saving on this
        LOGGER.warning("Could not build reconstruction metadata: %s", err)
    return metadata


def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    # Load the checkpoint
    # Load to CPU explictly, to avoid loading entire model on GPU initially
    # Modifications to the model occur on cpu,
    # The model will be sent to GPU when trainer.fit() is called
    LOGGER.debug("Loading checkpoint to device: cpu")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # Refresh processor stats from the current dataset if configured.
    model._update_checkpoint_state_dict_for_load(checkpoint)

    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"]
    _filter_state_dict_size_mismatches(state_dict, model.state_dict())

    # Runtime migration: the graph-provider permutation depends on instantiated provider state.
    checkpoint = trainable_edge_perm_fix_migration(checkpoint, model)
    state_dict = checkpoint["state_dict"]

    # Load the filtered state_dict into the model
    model.load_state_dict(state_dict, strict=False)

    ## Needed for data indices check
    data_indices = checkpoint["hyper_parameters"]["data_indices"]

    if isinstance(data_indices, dict):
        # New format: data_indices is always a dict in new code (even for single-dataset)
        LOGGER.info("Loading checkpoint with datasets: %s", list(data_indices.keys()))
        model._ckpt_model_name_to_index = {
            dataset_name: indices.name_to_index for dataset_name, indices in data_indices.items()
        }
    else:
        # Old format: data_indices is a single IndexCollection object (not dict)
        msg = (
            f"Checkpoint at '{ckpt_path}' was created with an older version of anemoi-core "
            "that does not support multi-dataset training. This checkpoint is incompatible "
            "with transfer learning in the current version."
        )
        raise TypeError(msg)

    # Extract variables_metadata for unit compatibility check
    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
        checkpoint,
        model._ckpt_model_name_to_index,
    )

    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str, base_target: str = "") -> bool:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze. Examples: "encoder", "encoder.lam".
    base_target : str
        Used for logging to show the full path of the current module being checked. Should not be set by the user.

    Returns
    -------
    bool
        True if the target submodule was found and frozen, False otherwise.
    """
    are_submodules_found = False
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                LOGGER.info("Freezing parameter %s: %s", base_target + name, param.shape)
                param.requires_grad = False
            are_submodules_found = True
        elif target_name.startswith(name + "."):
            new_target = target_name.replace(name + ".", "", 1)
            is_found = freeze_submodule_by_name(child, new_target, base_target=base_target + name + ".")
            are_submodules_found = are_submodules_found or is_found
        else:
            LOGGER.debug("Skipping submodule (looking for %s): %s", base_target + target_name, name)
    return are_submodules_found


class LoggingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> str:
        if "anemoi.training" in module:
            msg = (
                f"anemoi-training Pydantic schemas found in model's metadata: "
                f"({module}, {name}) Please review Pydantic schemas to avoid this."
            )
            raise ValueError(msg)
        return super().find_class(module, name)


def check_classes(model: torch.nn.Module) -> None:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    _ = LoggingUnpickler(buffer).load()


class RegisterMigrations(Callback):
    """Callback that register all existing migrations to a checkpoint before storing it."""

    def __init__(self):
        self.migrator = Migrator()

    def on_save_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        self.migrator.register_migrations(checkpoint)
