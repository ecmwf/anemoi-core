# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
from anemoi.training.train.tasks.base import BaseGraphModule
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
    module = BaseGraphModule.load_from_checkpoint(lightning_checkpoint_path)
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
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"]

    model_state_dict = model.state_dict()

    for key in state_dict.copy():
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]  # Remove the mismatched key

    # Load the filtered st-ate_dict into the model
    model.load_state_dict(state_dict, strict=False)
    # Needed for data indices check
    model._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index
    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str) -> None:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.
    """
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively search within children
            freeze_submodule_by_name(child, target_name)


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
