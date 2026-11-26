# (C) Copyright 2024-2026 Anemoi contributors.
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
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.graphs.utils import load_graph_from_file
from anemoi.graphs.utils import validate_loaded_graph
from anemoi.models.migrations import Migrator
from anemoi.models.preprocessing.imputer import BaseImputer
from anemoi.utils.checkpoints import save_metadata

LOGGER = logging.getLogger(__name__)


def clear_imputer_runtime_state(model: torch.nn.Module) -> None:
    """Clear imputer state before serialising a model."""
    for module in model.modules():
        if isinstance(module, BaseImputer):
            module.nan_locations = None
            module.loss_mask_training = None


def load_and_prepare_model(
    lightning_checkpoint_path: str,
    graph_path: Path | str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint.
    graph_path : Path | str | None, optional
        path to a graph file. If given, it is used instead of the ``graph_data`` stored in the
        checkpoint (which is then not required). The graph is not checked against the checkpoint
        beyond the presence of the required dataset nodes.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata.

    """
    from hydra.utils import get_class

    # memory-mapped: only the hyperparameters are needed here, not the weights
    hyper_parameters = torch.load(
        lightning_checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["hyper_parameters"]

    overrides = {}
    if graph_path is not None:
        dataset_names = list(hyper_parameters["data_indices"])
        graph = load_graph_from_file(graph_path)
        fused = uses_fused_dataset_graph(graph, dataset_names)
        validate_loaded_graph(graph, dataset_names if fused else [DEFAULT_DATASET_NAME])
        overrides["graph_data"] = graph
    elif "graph_data" not in hyper_parameters:
        msg = (
            f"The checkpoint {lightning_checkpoint_path} does not contain the hyperparameter 'graph_data'. "
            "Supply a graph with --graph."
        )
        raise ValueError(msg)

    training_method_cls = get_class(hyper_parameters["config"].training.method._target_)

    module = training_method_cls.load_from_checkpoint(lightning_checkpoint_path, weights_only=False, **overrides)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(
    model: torch.nn.Module,
    metadata: dict,
    save_path: Path | str,
    supporting_arrays: dict | None = None,
) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model.
    metadata : dict
        Anemoi Metadata to inject into checkpoint.
    save_path : Path | str
        File path of the anemoi inference checkpoint to write.
    supporting_arrays : dict | None, optional
        Supporting arrays to inject into checkpoint.

    Returns
    -------
    Path
        Path to saved checkpoint.
    """
    inference_filepath = Path(save_path)

    clear_imputer_runtime_state(model)
    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata, supporting_arrays=supporting_arrays)
    return inference_filepath


def load_graph_from_checkpoint(ckpt_path: Path | str) -> HeteroData | None:
    """Load the graph stored in the hyper-parameters of a Lightning checkpoint.

    The checkpoint is memory-mapped so that the model weights are not read, and the
    graph is cloned so that it does not keep the checkpoint file mapped.

    Parameters
    ----------
    ckpt_path : Path | str
        Path to the Lightning checkpoint.

    Returns
    -------
    HeteroData | None
        The graph the checkpoint was trained with, or None if it does not hold one.
    """
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu", mmap=True)
    graph = checkpoint.get("hyper_parameters", {}).get("graph_data")
    if not isinstance(graph, HeteroData):
        LOGGER.warning(
            "Checkpoint %s does not store a fused HeteroData graph (found %s).",
            ckpt_path,
            type(graph).__name__,
        )
        return None
    return graph.clone()


def check_node_coordinates_match_checkpoint(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    """Raise if the node coordinates in the checkpoint differ from those of the model's graph.

    The node coordinates (``latlons_*`` buffers) and trainable node tensors are restored
    from the checkpoint, but the edges are not, so a graph whose nodes are ordered
    differently from the checkpoint's would load without error and silently mismatch.

    Parameters
    ----------
    model : torch.nn.Module
        Model built from the current graph, before the checkpoint state is loaded.
    state_dict : dict[str, Any]
        State dict of the checkpoint, with keys as in ``model.state_dict()``.
    """
    mismatched = []
    for key, current in model.state_dict().items():
        if not key.rsplit(".", 1)[-1].startswith("latlons_") or key not in state_dict:
            continue
        stored = state_dict[key]
        if stored.shape != current.shape:
            mismatched.append(f"{key}: shape {tuple(stored.shape)} in checkpoint vs {tuple(current.shape)} in graph")
            continue
        n_rows = int((~torch.isclose(stored.cpu(), current.cpu(), atol=1e-6).all(dim=-1)).sum())
        if n_rows:
            mismatched.append(f"{key}: {n_rows}/{current.shape[0]} nodes differ")

    if mismatched:
        msg = (
            "Node coordinates in the checkpoint do not match the graph used to build the model:\n  "
            + "\n  ".join(mismatched)
            + "\nThe graph was probably rebuilt with a different node ordering. Use the graph the "
            "checkpoint was trained with (unset `graph.overwrite`, or point `system.input.graph` to "
            "the original graph file)."
        )
        raise RuntimeError(msg)


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
