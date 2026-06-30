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
import copy
from collections import OrderedDict

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

    # Handle processor chunk changes if necessary
    old_num_chunks = checkpoint["hyper_parameters"]["config"].model.processor.num_chunks
    new_num_chunks = model.model.model.processor.num_chunks
    old_num_layers = checkpoint["hyper_parameters"]["config"].model.processor.num_layers
    num_layers = new_num_chunks * model.model.model.processor.chunk_size
    assert old_num_layers == num_layers, "Changing number of layers is not supported"
    if old_num_chunks != new_num_chunks:
        LOGGER.info("Changing number of processor chunks from %d to %d", old_num_chunks, new_num_chunks)
        state_dict = change_num_processor_chunks(state_dict, old_num_chunks, new_num_chunks, num_layers)

    # Load the filtered state_dict into the model
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

def change_num_processor_chunks(state_dict: OrderedDict, old_num_chunks: int, new_num_chunks: int, num_layers: int) -> OrderedDict:
    """Change the number of processor chunks in the state_dict.

    This function modifies the state_dict in place to change the number of processor chunks
    by adjusting the relevant parameters.

    Parameters
    ----------
    state_dict : OrderedDict
        The state dictionary of the model.
    old_num_chunks : int
        The original number of processor chunks.
    new_num_chunks : int
        The desired number of processor chunks.

    Returns
    -------
    OrderedDict
        The modified state dictionary with updated processor chunk parameters.
    """
    _state_dict = copy.deepcopy(state_dict)

    for key in state_dict:
        if "model.model.processor.proc" in key:
            chunk = int(key.split(".proc.")[1].split(".")[0])
            block = int(key.split(".blocks.")[1].split(".")[0])
            new_chunk, new_block = remap_block(chunk, block, old_num_chunks, new_num_chunks, num_layers)
            #print("chunk, block -> new_chunk, new_block", chunk, block, "->", new_chunk, new_block)
            new_key = key.replace(f".proc.{chunk}.", f".proc.{new_chunk}.").replace(f".blocks.{block}.", f".blocks.{new_block}.")
            _state_dict[new_key] = state_dict[key]
    
    return _state_dict

def remap_block(old_chunk, old_block, old_num_chunks, new_num_chunks, total_blocks):
    """
    Map a block's (chunk, block) index from one chunking scheme to another.

    Args:
        old_chunk (int): The chunk index in the old scheme.
        old_block (int): The block index inside that chunk.
        old_num_chunks (int): Number of chunks in the old scheme.
        new_num_chunks (int): Number of chunks in the new scheme.
        total_blocks (int): Total number of blocks (fixed).

    Returns:
        (new_chunk, new_block) : tuple[int, int]
    """

    base_size_old = total_blocks // old_num_chunks
    remainder_old = total_blocks % old_num_chunks

    start_old = old_chunk * base_size_old + min(old_chunk, remainder_old)
    global_idx = start_old + old_block

    base_size_new = total_blocks // new_num_chunks
    remainder_new = total_blocks % new_num_chunks

    for c in range(new_num_chunks):
        start_new = c * base_size_new + min(c, remainder_new)
        end_new = start_new + base_size_new + (1 if c < remainder_new else 0)
        if start_new <= global_idx < end_new:
            new_chunk = c
            new_block = global_idx - start_new
            return new_chunk, new_block

    raise RuntimeError("Mapping failed — check inputs")


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
