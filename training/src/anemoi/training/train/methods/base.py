# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import importlib
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import replace as dataclass_replace
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from timm.scheduler.scheduler import Scheduler as TimmScheduler

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.models.data import Batch
from anemoi.models.data.views import TabularSourceView
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

_chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
_trainable_edge_perm_fix_migration = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
    from pytorch_lightning.utilities.types import OptimizerLRScheduler
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data.views import SourceView
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.tasks.base import BaseTask
    from anemoi.training.train.step_output import TrainingStepOutput

LOGGER = logging.getLogger(__name__)


class BaseTrainingModule(pl.LightningModule, ABC):
    """Abstract base class for Anemoi GNN forecasters using PyTorch Lightning.

    This class encapsulates the shared functionality for distributed training,
    scaling, and evaluation of graph-based neural network models across multiple GPUs and nodes.
    It provides hooks for defining losses, metrics, optimizers, and distributed sharding strategies.

    Key Features
    ------------
    - Supports model and data parallelism through model and reader process groups.
    - Handles graph data via `torch_geometric.data.HeteroData` format.
    - Supports sharded input batches and reconstruction via `allgather`.
    - Integrates modular loss and metric functions with support for variable scaling.
    - Enables deferred creation of variable scalers post-model instantiation.
    - Fully compatible with PyTorch Lightning training and validation loops.

    Subclass Responsibilities
    -------------------------
    Child classes must implement the `_step` method, which defines the forward and loss computation
    for training and validation steps.

    Parameters
    ----------
    config : BaseSchema
        Configuration object defining all parameters.
    statistics : dict
        Dictionary of training statistics (mean, std, etc.) used for normalization.
    statistics_tendencies : dict
        Statistics related to tendencies (if used).
    data_indices : dict[str, IndexCollection]
        Maps feature names to index ranges used for training and loss functions.
    metadata : dict
        Dictionary with metadata such as dataset provenance and variable descriptions.
    supporting_arrays : dict
        Numpy arrays (e.g., topography, masks) needed during inference and stored in checkpoints.

    Attributes
    ----------
    model : AnemoiModelInterface
        Wrapper for the underlying GNN model and its pre/post-processing logic.
    loss : BaseLoss
        Training loss function, optionally supporting variable scaling and sharding.
    metrics : dict[str, BaseLoss | Callable]
        Dictionary of validation metrics (often loss-style) computed during evaluation.
    scalers : dict
        Variable-wise scaling functions (e.g., standardization).
    val_metric_ranges : dict
        Mapping of variable groups for which to calculate validation metrics.
    output_mask : nn.Module
        Masking module that filters outputs during inference.
    n_step_input : int
        Number of input timesteps provided to the model.
    n_step_output : int
        Number of output timesteps predicted by the model.
    keep_batch_sharded : bool
        Whether to keep input batches split across GPUs instead of gathering them.

    Distributed Training
    --------------------
    The module can be configured to work in multi-node, multi-GPU environments with support for:
    - Custom communication groups for model and reader parallelism
    - Sharded input and output tensors
    - Support for `ZeroRedundancyOptimizer` and learning rate warmup

    Notes
    -----
    - This class should not be used directly. Subclass it and override `_step`.

    See Also
    --------
    - `AnemoiModelInterface`
    - `BaseLoss`
    - `IndexCollection`
    - `CosineLRScheduler`
    - `create_scalers`, `grad_scaler`

    """

    def __init__(
        self,
        *,
        config: BaseSchema,
        task: BaseTask,
        statistics: dict[str, dict],
        statistics_tendencies: dict[str, dict],
        data_indices: dict[str, IndexCollection],
        data_readers: dict,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict[str, IndexCollection]
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()
        self.task = task
        self.config = config

        assert isinstance(data_indices, dict), "data_indices must be a dict keyed by dataset name"

        # Handle dictionary of graph_data
        self.dataset_names = list(data_indices.keys())

        # Create output_mask dictionary for each dataset
        self.output_mask = {
            name: instantiate(
                config.model.output_mask,
                nodes=data_readers[name],  # TODO(Mario): Fix.
            )
            for name in self.dataset_names
        }

        # Handle supporting_arrays merge with all output masks
        combined_supporting_arrays = supporting_arrays.copy()
        for dataset_name, mask in self.output_mask.items():
            combined_supporting_arrays[dataset_name].update(mask.supporting_arrays)

        self.n_step_input = self.task.num_input_timesteps
        self.n_step_output = self.task.num_output_timesteps

        self.model = AnemoiModelInterface(
            config=config,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            data_readers=data_readers,
            metadata=metadata,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            supporting_arrays=combined_supporting_arrays,
        )

        self.data_indices = data_indices

        # `task` and `data_readers` hold runtime objects (e.g. anemoi-datasets
        # readers wrapping `lru.LRU` caches) that cannot be deepcopied by
        # Lightning's `save_hyperparameters` and are not meaningful to persist
        # in the checkpoint hyperparameters anyway.
        self.save_hyperparameters(ignore=["task", "data_readers"])

        self.statistics_tendencies = statistics_tendencies

        # Initialize components for multi-dataset
        self.target_dataset_names = []  # list of dataset names used for loss computation
        self.scalers = {}  # dict of dict of tensors
        self.updating_scalars = {}  # dict of dict of objects
        self.val_metric_ranges = {}  # dict of dict of lists
        self._scaling_values_log = {}  # dict of dict[str, float]
        self.loss = torch.nn.ModuleDict()
        self.metrics = torch.nn.ModuleDict()

        dataset_variable_groups = get_multiple_datasets_config(self.config.training.variable_groups)
        loss_configs = get_multiple_datasets_config(config.training.training_loss)
        scalers_configs = get_multiple_datasets_config(config.training.scalers)
        val_metrics_configs = get_multiple_datasets_config(config.training.validation_metrics)
        metrics_to_log = get_multiple_datasets_config(config.training.metrics)
        for dataset_name in self.dataset_names:
            if dataset_name not in loss_configs or loss_configs[dataset_name] is None:
                LOGGER.warning("Dataset %s is skipped for loss & metric computation.", dataset_name)
                continue

            self.target_dataset_names.append(dataset_name)

            # TODO : How to handle the graph_data objects that are now being used here?
            fused = True  # TODO: ??? uses_fused_dataset_graph(graph_data, self.dataset_names)
            data_node_name = dataset_name if fused else DEFAULT_DATASET_NAME

            # Create dataset-specific metadata extractor
            metadata_extractor = ExtractVariableGroupAndLevel(
                variable_groups=dataset_variable_groups[dataset_name],
                metadata_variables=metadata["dataset"][dataset_name].get("variables_metadata"),
            )

            dataset_scalers, dataset_updating_scalars = create_scalers(
                scalers_configs[dataset_name],
                data_indices=data_indices[dataset_name],
                task=self.task,
                graph_data=self.model.model._graph_data,
                statistics=statistics[dataset_name],
                statistics_tendencies=(
                    statistics_tendencies[dataset_name] if statistics_tendencies is not None else None
                ),
                metadata_extractor=metadata_extractor,
                nodes_name=dataset_name,
                output_mask=self.output_mask[dataset_name],
            )
            self.scalers[dataset_name] = dataset_scalers
            self.updating_scalars[dataset_name] = dataset_updating_scalars

            self.val_metric_ranges[dataset_name] = get_metric_ranges(
                metadata_extractor,
                output_data_indices=data_indices[dataset_name].model.output,
                metrics_to_log=metrics_to_log[dataset_name],
            )

            self.loss[dataset_name] = get_loss_function(
                loss_configs[dataset_name],
                dataset_scalers,
                data_indices[dataset_name],
                # graph_data=graph_data,
                data_node_name=data_node_name,
            )

            self.metrics[dataset_name] = self._build_metrics_for_dataset(
                val_metrics_configs[dataset_name],
                scalers=dataset_scalers,
                data_indices=data_indices[dataset_name],
                # graph_data=graph_data,
                data_node_name=data_node_name,
            )
            self._scaling_values_log[dataset_name] = print_variable_scaling(
                self.loss[dataset_name],
                data_indices[dataset_name],
            )

        if config.training.loss_gradient_scaling:
            # Multi-dataset: register hook for each loss
            for loss_fn in self.loss.values():
                loss_fn.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True

        LOGGER.info("GraphModule with n_step_input=%s and n_step_output=%s", self.n_step_input, self.n_step_output)
        self.effective_lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.optimization.lr
            / config.system.hardware.num_gpus_per_model
        )
        self.model_comm_group = None
        self.reader_groups = None

        self.reader_group_size = self.config.dataloader.read_group_size

        self.grid_dim = -2

        # check sharding support
        self.keep_batch_sharded = self.config.model.keep_batch_sharded
        read_group_supports_sharding = self.reader_group_size == self.config.system.hardware.num_gpus_per_model
        assert read_group_supports_sharding or not self.keep_batch_sharded, (
            f"Reader group size {self.reader_group_size} does not match the number of GPUs per model "
            f"{self.config.system.hardware.num_gpus_per_model}, but `model.keep_batch_sharded=True` was set. ",
            "Please set `model.keep_batch_sharded=False` or set `dataloader.read_group_size` ="
            "`hardware.num_gpus_per_model`.",
        )

        # set flag if loss and metrics support sharding
        self._check_sharding_support()

        LOGGER.debug("n_step_input: %d", self.n_step_input)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_size = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1

    @property
    def plot_adapter(self) -> Any:
        """Single entry point for diagnostics plot callbacks (replaces 5 small methods)."""
        return self.task._plot_adapter

    def _get_loss_name(self) -> str:
        """Get the loss name for multi-dataset cases."""
        # For multi-dataset, use a generic name or combine dataset names
        return "multi_dataset"

    def _check_sharding_support(self) -> None:
        self.loss_supports_sharding = all(
            getattr(leaf, "supports_sharding", False) for loss in self.loss.values() for leaf in loss.iter_leaf_losses()
        )
        self.metrics_support_sharding = all(
            getattr(metric, "supports_sharding", False)
            for dataset_metrics in self.metrics.values()
            for metric in dataset_metrics.values()
        )
        if not self.loss_supports_sharding and self.keep_batch_sharded:
            unsupported_losses = [
                type(leaf).__name__
                for loss in self.loss.values()
                for leaf in loss.iter_leaf_losses()
                if not getattr(leaf, "supports_sharding", False)
            ]
            LOGGER.warning(
                "Some loss functions do not support sharding: %s. "
                "This may lead to increased memory usage and slower training.",
                ", ".join(unsupported_losses),
            )
        if not self.metrics_support_sharding and self.keep_batch_sharded:
            unsupported_metrics = [
                f"{dataset_name}.{metric_name}"
                for dataset_name, dataset_metrics in self.metrics.items()
                for metric_name, metric in dataset_metrics.items()
                if not getattr(metric, "supports_sharding", False)
            ]
            LOGGER.warning(
                "Some validation metrics do not support sharding: %s. "
                "This may lead to increased memory usage and slower training.",
                ", ".join(unsupported_metrics),
            )

    @cached_property
    def logger_enabled(self) -> bool:
        return self.trainer.logger is not None

    def _build_metrics_for_dataset(
        self,
        validation_metrics_configs: dict,
        scalers: dict,
        data_indices: IndexCollection,
        graph_data: object | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                metric_name: get_loss_function(
                    val_metric_config,
                    scalers=scalers,
                    data_indices=data_indices,
                    graph_data=graph_data,
                    data_node_name=data_node_name,
                )
                for metric_name, val_metric_config in validation_metrics_configs.items()
            },
        )

    def forward(self, x: Batch | dict[str, torch.Tensor], **kwargs) -> Batch:
        """Forward method.

        This method calls the model's forward method with the appropriate
        communication group and sharding information.

        Accepts either a :class:`Batch` (preferred) or a legacy
        ``dict[str, Tensor]`` of per-dataset input tensors. The dict path
        is a transitional shim until ``_step`` is migrated to construct a
        :class:`Batch` end-to-end.
        """
        batch = x if isinstance(x, Batch) else Batch(data=x)
        return self.model(
            batch,
            model_comm_group=self.model_comm_group,
            **kwargs,
        )

    def _update_checkpoint_state_dict_for_load(self, checkpoint: dict[str, Any]) -> None:
        update_cfg = self.config.training.update_ds_stats_on_ckpt_load
        update_states = update_cfg.states
        update_tendencies = update_cfg.tendencies
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict) or not (update_states or update_tendencies):
            return

        processor_prefixes: tuple[str, ...] = ()
        if update_states:
            processor_prefixes += ("model.pre_processors.", "model.post_processors.")
        if update_tendencies:
            processor_prefixes += (
                "model.pre_processors_tendencies.",
                "model.post_processors_tendencies.",
            )

        if not processor_prefixes:
            return
        for key in list(state_dict.keys()):
            if key.startswith(processor_prefixes):
                del state_dict[key]

        model_state_dict = self.model.state_dict()
        processor_prefixes += tuple(f"model.{k}" for k in model_state_dict if "model_output_idx" in k)
        for key, value in model_state_dict.items():
            full_key = f"model.{key}"
            if full_key.startswith(processor_prefixes):
                state_dict[full_key] = value

    def on_load_checkpoint(self, checkpoint: torch.nn.Module) -> None:
        # Apply migrations to handle state_dict key changes from older checkpoints.
        # These are idempotent: already-migrated checkpoints are unaffected.
        _trainable_edge_perm_fix_migration(checkpoint, model=self)
        self._update_checkpoint_state_dict_for_load(checkpoint)

        self._ckpt_model_name_to_index = {
            dataset_name: data_indices.name_to_index
            for dataset_name, data_indices in checkpoint["hyper_parameters"]["data_indices"].items()
        }

        # Extract variables_metadata for unit compatibility check
        self._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
            checkpoint,
            self._ckpt_model_name_to_index,
        )

    def _update_scaler_for_dataset(
        self,
        name: str,
        scaler_builder: BaseScaler,
        callback: AvailableCallbacks,
        loss_obj: torch.nn.Module,
        metrics_dict: dict,
        dataset_name: str,
    ) -> None:
        """Update a single scaler for loss and metrics objects."""
        kwargs = {"model": self.model, "dataset_name": dataset_name}

        scaler = scaler_builder.update_scaling_values(callback, **kwargs)
        if scaler is None:  # If scaler is None, no update to be applied
            return

        if self._can_update_scaler(loss_obj, name):
            loss_obj.update_scaler(scaler=scaler[1], name=name)  # Only update the values

        for metric in metrics_dict.values():  # If scalar in metrics, update it
            if self._can_update_scaler(metric, name):
                metric.update_scaler(scaler=scaler[1], name=name)  # Only update the values

    @staticmethod
    def _can_update_scaler(loss_or_metric: torch.nn.Module, scaler_name: str) -> bool:
        """Whether a module can update a scaler with this name.

        Standard losses/metrics expose a ``scaler`` container, while composite losses
        (e.g., ``CombinedLoss``) intentionally remove this attribute and route updates
        through their ``update_scaler`` implementation.
        """
        if not hasattr(loss_or_metric, "update_scaler"):
            return False

        scaler = getattr(loss_or_metric, "scaler", None)
        if scaler is None:
            return True

        return scaler_name in scaler

    def update_scalers(self, callback: AvailableCallbacks) -> None:
        """Update scalers, calling the defined function on them, updating if not None."""
        # Multi-dataset case: {'dataset_a': {'nan_mask_weights': scaler, ...}, 'dataset_b': {...}}
        for dataset_name, dataset_scalers in self.updating_scalars.items():
            for name, scaler_builder in dataset_scalers.items():
                self._update_scaler_for_dataset(
                    name,
                    scaler_builder,
                    callback,
                    self.loss[dataset_name],
                    self.metrics[dataset_name],
                    dataset_name=dataset_name,
                )

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def _grid_shard_sizes(self, source: SourceView) -> Any:
        if isinstance(source, TabularSourceView):
            return None
        return getattr(source, "shard_sizes", None)

    def _grid_shard_slice(self, source: SourceView) -> slice | None:
        """Local grid shard slice for ``source``, derived from its shard sizes.

        Returns ``None`` when the source is replicated (not sharded).
        """
        # no per-grid scalers/masks for TabularSourceView that would need to be sliced
        if isinstance(source, TabularSourceView):
            return None

        shard_sizes = getattr(source, "shard_sizes", None)
        if not shard_sizes:
            return None
        assert isinstance(shard_sizes, list) and all(
            isinstance(s, int) for s in shard_sizes
        ), "shard_sizes must be a list of integers"
        start, end = get_partition_range(shard_sizes, self.model_comm_group_rank)
        return slice(start, end)

    def _prepare_tensors_for_loss(
        self,
        y_pred: SourceView,
        y: SourceView,
        dataset_name: str,
        validation_mode: bool = False,
    ) -> tuple[SourceView, SourceView, slice | None]:
        """Prepare tensors for loss computation, handling sharding if necessary.

        Parameters
        ----------
        y_pred : SourceView
            Predicted values
        y : SourceView
            Target values
        validation_mode : bool
            Whether in validation mode

        Returns
        -------
        tuple[SourceView, SourceView, slice | None]
            Prepared y_pred, y, and grid_shard_slice
        """
        # Sharding metadata now lives on the source views (None when replicated).
        grid_shard_slice = self._grid_shard_slice(y)
        is_sharded = grid_shard_slice is not None

        sharding_supported = (self.loss_supports_sharding) and (  # loss calculated in training and validation mode
            self.metrics_support_sharding or not validation_mode
        )

        if is_sharded and not sharding_supported:  # gather tensors if loss or metrics do not support sharding
            y_pred_full = y_pred.allgather(self.model_comm_group)
            y_full = y.allgather(self.model_comm_group)
            final_grid_shard_slice = None
        else:
            y_pred_full, y_full = y_pred, y
            final_grid_shard_slice = grid_shard_slice

        if final_grid_shard_slice == slice(None):
            final_grid_shard_slice = None

        return y_pred_full, y_full, final_grid_shard_slice

    def _compute_loss(
        self,
        y_pred: SourceView,
        y: SourceView,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the loss function.

        Parameters
        ----------
        y_pred : SourceView
            Predicted values
        y : SourceView
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        dataset_name : str
            Dataset name for multi-dataset scenarios
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        loss = self.loss[dataset_name]
        loss_kwargs = {
            "grid_shard_slice": grid_shard_slice,
            "group": self.model_comm_group,
        }
        if pred_layout is not None:
            loss_kwargs["pred_layout"] = pred_layout
        if target_layout is not None:
            loss_kwargs["target_layout"] = target_layout

        if getattr(loss, "needs_shard_layout_info", False):
            loss_kwargs.update(
                grid_dim=self.grid_dim,
                grid_shard_sizes=self._grid_shard_sizes(y),
            )

        return loss(y_pred, y, **loss_kwargs)

    def _compute_metrics(
        self,
        y_pred: SourceView,
        y: SourceView,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        rollout_step: int | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute validation metrics.

        Parameters
        ----------
        y_pred : SourceView
            Predicted values
        y : SourceView
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        rollout_step : int | None
            Current rollout step index, used to produce per-step metric key suffixes.

        Returns
        -------
        dict[str, torch.Tensor]
            Computed metrics
        """
        return self.calculate_val_metrics(
            y_pred,
            y,
            step=rollout_step,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            pred_layout=pred_layout,
            target_layout=target_layout,
        )

    def compute_dataset_loss_metrics(
        self,
        y_pred: SourceView,
        y: SourceView,
        validation_mode: bool = False,
        dataset_name: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], SourceView]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : SourceView
            Predicted values
        y : SourceView
            Target values
        step : int, optional
            Current step
        validation_mode : bool, optional
            Whether to compute validation metrics
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], SourceView]
            Loss, metrics dictionary (if validation_mode), and full predictions
        """
        # Prepare tensors for loss/metrics computation
        y_pred_full, y_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            validation_mode=validation_mode,
            dataset_name=dataset_name,
        )

        loss = self._compute_loss(
            y_pred=y_pred_full,
            y=y_full,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            **kwargs,
        )

        # Compute metrics if in validation mode
        metrics_next = {}
        if validation_mode:
            metrics_next = self._compute_metrics(
                y_pred_full,
                y_full,
                grid_shard_slice=grid_shard_slice,
                dataset_name=dataset_name,
                **kwargs,
            )

        return loss, metrics_next, y_pred

    def compute_loss_metrics(
        self,
        y_pred: Batch,
        y: Batch,
        validation_mode: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], dict[str, SourceView]]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : Batch
            Predicted values
        y : Batch
            Target values
        step : int, optional
            Current step
        validation_mode : bool, optional
            Whether to compute validation metrics
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], dict[str, SourceView]]
            Loss, metrics dictionary (if validation_mode), and full predictions
        """
        assert isinstance(y_pred, Batch), "y_pred must be a dict keyed by dataset name"
        assert isinstance(y, Batch), "y must be a dict keyed by dataset name"
        # Prepare tensors for loss/metrics computation
        total_loss, metrics_next, y_preds = None, {}, {}
        for dataset_name in self.target_dataset_names:
            if dataset_name not in y_pred:
                LOGGER.warning(
                    "Dataset %s is missing from predictions, skipping loss and metric computation for this dataset.",
                    dataset_name,
                )
                continue

            dataset_loss, dataset_metrics, y_preds[dataset_name] = self.compute_dataset_loss_metrics(
                y_pred[dataset_name],
                y[dataset_name],
                validation_mode=validation_mode,
                dataset_name=dataset_name,
                **kwargs,
            )

            if dataset_loss is not None:
                dataset_loss_sum = dataset_loss.sum()  # collapse potential multi-scale loss
                total_loss = dataset_loss_sum if total_loss is None else total_loss + dataset_loss_sum

                if validation_mode:
                    loss_obj = self.loss[dataset_name]
                    loss_name = getattr(loss_obj, "name", loss_obj.__class__.__name__.lower())
                    metrics_next[f"{dataset_name}_{loss_name}_loss"] = dataset_loss

            # Prefix dataset name to metric keys
            for metric_name, metric_value in dataset_metrics.items():
                metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

        return total_loss, metrics_next, y_preds

    def preprocess_batch(self, batch: Batch) -> Batch:
        """
        Preprocess the batch using the model's pre-processors.
        This includes the imputers, where we have to separate between inputs and outputs.
        Only the former get imputed with default values; the outputs are left untouched.
        """
        new_batch = batch
        input_indices = self.task.get_batch_input_indices()
        for dataset_name, pre_processors in self.model.pre_processors.items():
            view = batch[dataset_name]
            impute_mask = self._build_input_impute_mask(view, input_indices)
            updated_view = pre_processors(view, impute_mask=impute_mask)
            # TODO: remove update_source, maybe batch.clone()??
            new_batch = new_batch.update_source(dataset_name, updated_view)
        return new_batch

    @staticmethod
    def _build_input_impute_mask(view: SourceView, input_indices: list[int]) -> list[torch.Tensor] | None:
        """Build a per-sample boolean grid mask that identifies model-input observations.

        Only sparse datasets are masked: for each sample, points that
        belong to an input time-window have mask==True (imputed), all
        other points (target windows) have mask==False (NaNs preserved for loss masking).

        For gridded datasets (or sparse views without boundaries), we return None.
        """
        if not view.layout.time_in_grid or view.boundaries is None:
            return None

        grid_dim = view.layout.grid
        masks: list[torch.Tensor] = []
        for sample_tensor, sample_bounds in zip(view.data, view.boundaries):
            n_points = sample_tensor.shape[grid_dim]
            mask = torch.zeros(n_points, dtype=torch.bool, device=sample_tensor.device)
            for t in input_indices:
                s = sample_bounds[t]
                mask[s.start : s.stop] = True
            masks.append(mask)
        return masks

    def on_after_batch_transfer(self, batch: Batch, _: int) -> Batch:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Also normalize the batch in-place if needed.

        Parameters
        ----------
        batch : Batch
            Batch to transfer

        Returns
        -------
        Batch
            Batch after transfer
        """
        assert isinstance(batch, Batch), "batch must be a Batch instance"
        # Gathering/sharding of batch
        if not self.keep_batch_sharded:
            batch = self.allgather_batch(batch)

        # Batch normalization (the underlying ``batch.data`` dict should be mutated in-place)
        batch = self.preprocess_batch(batch)

        # Debug-log the batch contents (per-dataset shape + layout) so that
        # layout/shape mismatches can be diagnosed from a real run.
        LOGGER.debug("on_after_batch_transfer batch:\n%r", batch)

        # Prepare scalers, e.g. init delayed scalers and update scalers
        self._prepare_loss_scalers()

        return batch

    def transfer_batch_to_device(
        self,
        batch: Batch,
        device: torch.device,
        _dataloader_idx: int = 0,
    ) -> Batch:
        """Transfer the :class:`Batch` to ``device`` (skipping static coords)."""
        return batch.to(device, non_blocking=True)

    def _prepare_loss_scalers(self) -> None:
        """Prepare scalers for training and validation before every step."""
        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.update_scalers(callback=AvailableCallbacks.ON_TRAINING_START)
            self.is_first_step = False
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)
        return

    @abstractmethod
    def _step(
        self,
        batch: Batch,
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        pass

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        return batch.allgather(self.reader_groups[self.reader_group_id])

    def _align_view_to_layout(
        self,
        view: SourceView,
        layout: IndexSpace | str | None,
        dataset_name: str,
    ) -> SourceView:
        """Realign a view's variable metadata to the variables of a given index space.

        A prediction view produced by the model inherits the full target metadata
        (variables / name_to_index / statistics) even though its tensor only
        holds the model-output (or data-output) subset of variables. Computing
        per-variable normalisation parameters from the full metadata would then
        produce tensors that do not broadcast against the (smaller) data tensor.

        The alignment is driven entirely by the layout and the view's metadata, so it
        works regardless both gridded fields and sparse obs.

        When the metadata already matches the layout variable set
        (e.g. a full data-space target), the view is returned unchanged.

        Parameters
        ----------
        view : SourceView
            View whose metadata may describe more variables than its tensor holds.
        layout : IndexSpace | str | None
            Index space the view's tensor lives in. None is treated as `IndexSpace.DATA_FULL`.
        dataset_name : str
            Dataset the view belongs to, used to look up the data indices.

        Returns
        -------
        SourceView
            View with metadata aligned to the layout variable set.
        """
        layout = IndexSpace.DATA_FULL if layout is None else IndexSpace(layout)
        data_indices = self.data_indices[dataset_name]
        if layout == IndexSpace.MODEL_OUTPUT:
            names = list(data_indices.model.output.ordered_names)
        elif layout == IndexSpace.DATA_OUTPUT:
            names = list(data_indices.data.output.ordered_names)
        else:
            names = list(data_indices.data_full_ordered_names)

        # The view's metadata already matches the layout's variable set.
        if list(view.variables) == names:
            return view

        positions = [view.name_to_index[name] for name in names]
        return dataclass_replace(
            view,
            variables=names,
            statistics={key: value[positions] for key, value in view.statistics.items()},
        )

    def calculate_val_metrics(
        self,
        y_pred: SourceView,
        y: SourceView,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        step: int | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: SourceView
            Predicted ensemble
        y: SourceView
            Ground truth (target).
        step: int, optional
            Step number

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions
        """
        metrics = {}

        # Handle multi-dataset case for post-processors
        post_processor = self.model.post_processors[dataset_name]
        metrics_dict = self.metrics[dataset_name]
        val_metric_ranges = self.val_metric_ranges[dataset_name]

        # y (target) and y_pred (model output) can live in different index
        # spaces and therefore hold different numbers of variables along the
        # variable axis.
        # The metadata of the prediction view must be realigned to
        # the variables actually present in its tensor *before* the normalisation
        y_postprocessed = post_processor(self._align_view_to_layout(y, target_layout, dataset_name), in_place=False)
        y_pred_postprocessed = post_processor(
            self._align_view_to_layout(y_pred, pred_layout, dataset_name),
            in_place=False,
        )

        suffix = "" if step is None else f"/{step + 1}"
        for metric_name, metric in metrics_dict.items():
            # Validation now compares the model output tensor with the full target tensor.
            # Those can contain different variables, so the metric needs to know how to
            # line them up before computing the score. Other metrics do not have this information.
            assert isinstance(
                metric,
                BaseLoss,
            ), f"Validation metric {metric_name!r} must inherit BaseLoss, got {type(metric)}"

            for mkey, indices in val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{dataset_name}/{mkey}{suffix}"
                if metric.has_scaler_for_dim(TensorDim.VARIABLE):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metric_kwargs = {
                    "scaler_indices": (..., indices),
                    "grid_shard_slice": grid_shard_slice,
                    "group": self.model_comm_group,
                }
                if pred_layout is not None:
                    metric_kwargs["pred_layout"] = pred_layout
                if target_layout is not None:
                    metric_kwargs["target_layout"] = target_layout
                if getattr(metric, "needs_shard_layout_info", False):
                    metric_kwargs.update(
                        grid_dim=self.grid_dim,
                        grid_shard_sizes=self._grid_shard_sizes(y),
                    )

                metrics[metric_step_name] = metric(y_pred_postprocessed, y_postprocessed, **metric_kwargs)

        return metrics

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        assert isinstance(batch, Batch), "batch must be a Batch instance"

        step_output = self._step(batch)
        train_loss = step_output.loss.sum()

        self.log(
            "train_" + self._get_loss_name() + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.size,
            sync_dist=True,
        )

        self.task.log_extra(logger=self.log, logger_enabled=self.logger_enabled)

        return train_loss

    def validation_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : Batch
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        TrainingStepOutput
            Output of the validation step.
        """
        del batch_idx
        assert isinstance(batch, Batch), "batch must be a Batch instance"

        with torch.no_grad():
            step_output = self._step(batch, validation_mode=True)
        val_loss_scales = step_output.loss
        metrics = step_output.metrics
        val_loss = val_loss_scales.sum()

        self.log(
            "val_" + self._get_loss_name() + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.size,
            sync_dist=True,
        )

        if val_loss_scales.numel() > 1:
            loss_name = self._get_loss_name()
            if len(self.loss) == 1:
                loss_obj = next(iter(self.loss.values()))
                loss_name = getattr(loss_obj, "name", loss_obj.__class__.__name__.lower())
            for scale in range(val_loss_scales.numel()):
                self.log(
                    "val_" + loss_name + "_loss" + "_scale_" + str(scale),
                    val_loss_scales[scale],
                    on_epoch=True,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch.size,
                    sync_dist=True,
                )

        for mname, mvalue in metrics.items():
            for scale in range(mvalue.numel()):

                log_val = mvalue[scale] if mvalue.numel() > 1 else mvalue

                self.log(
                    "val_" + mname + "_scale_" + str(scale),
                    log_val,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch.size,
                    sync_dist=True,
                )

        return step_output

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : LRSchedulerTypeUnion
            Learning rate scheduler object.
        metric : Any
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        if isinstance(scheduler, TimmScheduler):
            cfg = next(c for c in self.trainer.lr_scheduler_configs if c.scheduler is scheduler)
            if cfg.interval == "step":
                scheduler.step_update(self.trainer.global_step, metric)
            else:
                scheduler.step(self.current_epoch + 1, metric)
            return

        super().lr_scheduler_step(scheduler, metric)

    def on_train_epoch_end(self) -> None:
        self.task.on_train_epoch_end(current_epoch=self.current_epoch)
        self.trainer.datamodule.set_epoch(self.current_epoch + 1)

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        """Create optimizer and LR scheduler based on Hydra config."""
        optimization_config = self.config.training.optimization
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(optimization_config.optimizer, params=params, lr=self.effective_lr)
        self.log_optimizer(optimizer)

        if not getattr(optimization_config, "lr_scheduler", None):
            return optimizer

        scheduler = instantiate(optimization_config.lr_scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, **optimization_config.pl_lr_scheduler}]  # type: ignore[return-value]

    @staticmethod
    def log_optimizer(optimizer: torch.optim.Optimizer) -> None:
        """Log optimizer type and settings."""
        defaults_to_log = {k: v for k, v in optimizer.defaults.items() if k != "params"}
        LOGGER.info("Optimizer initialized: %s", type(optimizer).__name__)
        LOGGER.info("Optimizer settings: %s", defaults_to_log)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called after model is initialized but before training starts."""
        if stage == "fit" and self.trainer.is_global_zero and self.logger is not None:
            hyper_params = OmegaConf.to_container(self.config, resolve=True)
            hyper_params.update({"variable_loss_scaling": self._scaling_values_log})
            self.logger.log_hyperparams(hyper_params)
