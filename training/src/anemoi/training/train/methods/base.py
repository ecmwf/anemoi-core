# (C) Copyright 2024-2026 Anemoi contributors.
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
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import graph_participants
from anemoi.graphs.projection_helpers import participant_node_name
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.batch_meta import meta_participant
from anemoi.training.data.batch_meta import split_meta
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.losses.utils import check_loss_tree_variable_units
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.train.participant_metrics import ParticipantMetrics
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.masks import build_output_masks
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

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.tasks.base import BaseTask
    from anemoi.training.train.step_output import TrainingStepOutput
    from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)

# `torch.nn.ModuleDict` keys must be strings, so single-domain (participant-less) training
# objects are stored under this key.
DEFAULT_PARTICIPANT_KEY = "default"


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
    task : BaseTask
        Training task that defines the prediction workflow.
    graph_data : HeteroData
        Graph-structured input data containing node and edge features, keyed by dataset name.
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
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        task : BaseTask
            Training task.
        graph_data : HeteroData
            Graph objects keyed by dataset name
        statistics : dict
            Statistics of the training data
        statistics_tendencies : dict
            Statistics of data tendencies.
        data_indices : dict[str, IndexCollection]
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()
        self.task = task

        assert isinstance(graph_data, HeteroData), "graph_data must be a HeteroData object"
        assert isinstance(data_indices, dict), "data_indices must be a dict keyed by dataset name"

        graph_data = graph_data.to(self.device)
        self.dataset_names = list(data_indices.keys())

        # Multi-domain training: the graph carries one node group per participant
        # (``<dataset>_<participant>``). Every grid-sized training object is built once per
        # participant and selected by `set_active_participant` before each batch.
        self._participants = graph_participants(graph_data, self.dataset_names[0])
        self._active_participant = self._participants[0] if self._participants else None
        self._fused_dataset_graph = bool(self._participants) or uses_fused_dataset_graph(
            graph_data,
            self.dataset_names,
        )

        # Create output_mask dictionary for each dataset, per participant
        output_mask_configs = get_multiple_datasets_config(config.model.output_mask)
        self._output_mask = {
            participant: build_output_masks(output_mask_configs, graph_data, participant=participant)
            for participant in self._participant_keys
        }
        self.output_mask = self._output_mask[self._active_participant]

        # Handle supporting_arrays merge with all output masks
        combined_supporting_arrays = supporting_arrays.copy()
        for dataset_name, mask in self.output_mask.items():
            combined_supporting_arrays[dataset_name].update(mask.supporting_arrays)

        self.n_step_input = self.task.num_input_timesteps
        self.n_step_output = self.task.num_output_timesteps

        self.model = AnemoiModelInterface(
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            supporting_arrays=combined_supporting_arrays,
            graph_data=graph_data,
            config=config,
        )
        self.config = config

        self.data_indices = data_indices

        self.save_hyperparameters()

        self.statistics_tendencies = statistics_tendencies

        # Initialize components for multi-dataset
        self.target_dataset_names = []  # list of dataset names used for loss computation
        self._scalers = {}  # participant -> dict of dict of tensors
        self._updating_scalars = {}  # participant -> dict of dict of objects
        self.val_metric_ranges = {}  # dict of dict of lists
        self._scaling_values_log = {}  # dict of dict[str, float]
        self._loss = torch.nn.ModuleDict()  # participant key -> ModuleDict of losses
        self._metrics = torch.nn.ModuleDict()  # participant key -> ModuleDict of metric ModuleDicts

        dataset_variable_groups = get_multiple_datasets_config(self.config.training.variable_groups)
        loss_configs = get_multiple_datasets_config(config.training.training_loss)
        self._resolve_subgrid(loss_configs)

        scalers_configs = get_multiple_datasets_config(config.training.scalers)
        val_metrics_configs = get_multiple_datasets_config(config.training.validation_metrics)
        metrics_to_log = get_multiple_datasets_config(config.training.metrics)

        metadata_extractors = {}
        for dataset_name in self.dataset_names:
            if loss_configs.get(dataset_name) is None:
                LOGGER.warning("Dataset %s is skipped for loss & metric computation.", dataset_name)
                continue

            self.target_dataset_names.append(dataset_name)

            # Create dataset-specific metadata extractor
            metadata_extractors[dataset_name] = ExtractVariableGroupAndLevel(
                variable_groups=dataset_variable_groups[dataset_name],
                metadata_variables=metadata["dataset"][dataset_name].get("variables_metadata"),
            )
            self.val_metric_ranges[dataset_name] = get_metric_ranges(
                metadata_extractors[dataset_name],
                output_data_indices=data_indices[dataset_name].model.output,
                metrics_to_log=metrics_to_log[dataset_name],
            )

        self._build_participant_training_objects(
            graph_data=graph_data,
            data_indices=data_indices,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            metadata=metadata,
            metadata_extractors=metadata_extractors,
            scalers_configs=scalers_configs,
            loss_configs=loss_configs,
            val_metrics_configs=val_metrics_configs,
            loss_gradient_scaling=config.training.loss_gradient_scaling,
        )

        self.is_first_step = True
        self._current_meta = None  # metadata of the batch being processed (see batch_meta.split_meta)
        # validation metrics per participant (multi-domain training); reduced at validation epoch end
        self._participant_metrics = ParticipantMetrics()

        LOGGER.info("GraphModule with n_step_input=%s and n_step_output=%s", self.n_step_input, self.n_step_output)
        self.effective_lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.optimization.lr
            / config.system.hardware.num_gpus_per_model
        )
        self.model_comm_group = None
        self.reader_groups = None

        reader_group_size = self.config.dataloader.read_group_size

        self._shard_sizes, self._grid_sizes = {}, {}
        self._build_participant_grid_sizes(graph_data, reader_group_size)

        # `output_mask`, `scalers`, `updating_scalars`, `loss`, `metrics`, `grid_sizes` and
        # `shard_sizes` always hold the active participant's objects.
        self._select_participant_objects()

        self.grid_dim = -2

        # check sharding support
        self.keep_batch_sharded = self.config.model.keep_batch_sharded
        read_group_supports_sharding = reader_group_size == self.config.system.hardware.num_gpus_per_model
        assert read_group_supports_sharding or not self.keep_batch_sharded, (
            f"Reader group size {reader_group_size} does not match the number of GPUs per model "
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

        self.grid_shard_sizes = dict.fromkeys(self.dataset_names, None)
        self.grid_shard_slice = dict.fromkeys(self.dataset_names, None)

    # ------------------------------------------------------------------
    # Multi-domain (participant) support
    # ------------------------------------------------------------------
    # Class-level defaults so participant-aware code also works on instances that
    # were not built through `__init__` (e.g. `__new__`-built test doubles).
    _participants: tuple[str, ...] = ()
    _active_participant: str | None = None

    @property
    def participants(self) -> list[str]:
        """Participants of a multi-domain graph, empty for a single-domain graph."""
        return list(self._participants)

    @property
    def active_participant(self) -> str | None:
        """Participant the grid-sized training objects currently resolve to."""
        return self._active_participant

    @property
    def _participant_keys(self) -> list[str | None]:
        """Keys the per-participant training objects are stored under."""
        return list(self._participants) if self._participants else [None]

    @staticmethod
    def _participant_key(participant: str | None) -> str:
        """Map a participant to a `torch.nn.ModuleDict` key, which must be a string."""
        return DEFAULT_PARTICIPANT_KEY if participant is None else participant

    def _data_node_name(self, dataset_name: str, participant: str | None) -> str:
        """Graph node group holding the grid of `dataset_name` for `participant`."""
        base = dataset_name if self._fused_dataset_graph else DEFAULT_DATASET_NAME
        return participant_node_name(base, participant)

    def _build_participant_training_objects(
        self,
        *,
        graph_data: HeteroData,
        data_indices: dict[str, IndexCollection],
        statistics: dict,
        statistics_tendencies: dict | None,
        metadata: dict,
        metadata_extractors: dict,
        scalers_configs: dict,
        loss_configs: dict,
        val_metrics_configs: dict,
        loss_gradient_scaling: bool,
    ) -> None:
        """Build the scalers, losses and metrics of every participant.

        They are grid-sized, so multi-domain training needs one set per participant;
        `_select_participant_objects` then exposes the active participant's set.
        """
        for participant in self._participant_keys:
            participant_key = self._participant_key(participant)
            self._scalers[participant] = {}
            self._updating_scalars[participant] = {}
            self._loss[participant_key] = torch.nn.ModuleDict()
            self._metrics[participant_key] = torch.nn.ModuleDict()

            for dataset_name in self.target_dataset_names:
                data_node_name = self._data_node_name(dataset_name, participant)

                dataset_scalers, dataset_updating_scalars = create_scalers(
                    scalers_configs[dataset_name],
                    data_indices=data_indices[dataset_name],
                    task=self.task,
                    graph_data=graph_data,
                    statistics=statistics[dataset_name],
                    statistics_tendencies=(
                        statistics_tendencies[dataset_name] if statistics_tendencies is not None else None
                    ),
                    metadata_extractor=metadata_extractors[dataset_name],
                    nodes_name=participant_node_name(dataset_name, participant),
                    output_mask=self._output_mask[participant][dataset_name],
                )
                self._scalers[participant][dataset_name] = dataset_scalers
                self._updating_scalars[participant][dataset_name] = dataset_updating_scalars

                dataset_loss = get_loss_function(
                    loss_configs[dataset_name],
                    dataset_scalers,
                    data_indices[dataset_name],
                    graph_data=graph_data,
                    data_node_name=data_node_name,
                )
                self._loss[participant_key][dataset_name] = dataset_loss

                # Check unit compatibility between predicted and target variables
                ds_variables_metadata = metadata["dataset"][dataset_name].get("variables_metadata")
                check_loss_tree_variable_units(dataset_loss, ds_variables_metadata)

                dataset_metrics = self._build_metrics_for_dataset(
                    val_metrics_configs[dataset_name],
                    scalers=dataset_scalers,
                    data_indices=data_indices[dataset_name],
                    graph_data=graph_data,
                    data_node_name=data_node_name,
                )
                self._metrics[participant_key][dataset_name] = dataset_metrics

                self._initialise_updating_scalers(
                    scalers=dataset_scalers,
                    updating_scalers=dataset_updating_scalars,
                    loss_obj=dataset_loss,
                    metrics_dict=dataset_metrics,
                )
                self._scaling_values_log[dataset_name] = print_variable_scaling(
                    dataset_loss,
                    data_indices[dataset_name],
                )

        if loss_gradient_scaling:
            # Multi-dataset: register hook for each loss of every participant
            for participant_losses in self._loss.values():
                for loss_fn in participant_losses.values():
                    loss_fn.register_full_backward_hook(grad_scaler, prepend=False)

    def _build_participant_grid_sizes(self, graph_data: HeteroData, reader_group_size: int) -> None:
        """Read the grid size of every dataset of every participant off the graph."""
        for participant in self._participant_keys:
            self._grid_sizes[participant] = {}
            self._shard_sizes[participant] = {}
            for dataset_name in self.dataset_names:
                node_name = participant_node_name(dataset_name, participant)
                # TODO(Mario): Replace by dataset.grid_size
                self._grid_sizes[participant][dataset_name] = graph_data[node_name].num_nodes
                self._shard_sizes[participant][dataset_name] = get_balanced_partition_sizes(
                    self._grid_sizes[participant][dataset_name],
                    reader_group_size,
                )

    def _select_participant_objects(self) -> None:
        """Point the grid-sized training objects at the active participant."""
        participant = self._active_participant
        participant_key = self._participant_key(participant)
        self.output_mask = self._output_mask[participant]
        self.scalers = self._scalers[participant]
        self.updating_scalars = self._updating_scalars[participant]
        self.grid_sizes = self._grid_sizes[participant]
        self.shard_sizes = self._shard_sizes[participant]
        self.loss = self._loss[participant_key]
        self.metrics = self._metrics[participant_key]

    def set_active_participant(self, participant: str | None) -> None:
        """Select the participant the model and the grid-sized training objects resolve to.

        A no-op for single-domain graphs, so it can be called unconditionally.
        """
        if not self._participants:
            return

        if participant not in self._participants:
            msg = f"Unknown participant '{participant}', expected one of {self._participants}."
            raise ValueError(msg)

        if participant == self._active_participant:
            return

        self._active_participant = participant
        self.model.model.set_active_participant(participant)
        self._select_participant_objects()

    @property
    def plot_adapter(self) -> Any:
        """Single entry point for diagnostics plot callbacks (replaces 5 small methods)."""
        return self.task._plot_adapter

    def _get_loss_name(self) -> str:
        """Get the loss name for multi-dataset cases."""
        # For multi-dataset, use a generic name or combine dataset names
        return "multi_dataset"

    def _check_sharding_support(self) -> None:
        all_losses = [loss for participant_losses in self._loss.values() for loss in participant_losses.values()]
        all_metrics = [
            (dataset_name, dataset_metrics)
            for participant_metrics in self._metrics.values()
            for dataset_name, dataset_metrics in participant_metrics.items()
        ]
        self.loss_supports_sharding = all(
            getattr(leaf, "supports_sharding", False) for loss in all_losses for leaf in loss.iter_leaf_losses()
        )
        self.metrics_support_sharding = all(
            getattr(metric, "supports_sharding", False)
            for _, dataset_metrics in all_metrics
            for metric in dataset_metrics.values()
        )
        if not self.loss_supports_sharding and self.keep_batch_sharded:
            unsupported_losses = [
                type(leaf).__name__
                for loss in all_losses
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
                for dataset_name, dataset_metrics in all_metrics
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

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward method.

        This method calls the model's forward method with the appropriate
        communication group and sharding information.
        """
        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_sizes=self.grid_shard_sizes,
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

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["task_state"] = self.task.training_runtime_state_dict()

    def on_load_checkpoint(self, checkpoint: torch.nn.Module) -> None:
        # Apply migrations to handle state_dict key changes from older checkpoints.
        # These are idempotent: already-migrated checkpoints are unaffected.
        _trainable_edge_perm_fix_migration(checkpoint, model=self)
        self._update_checkpoint_state_dict_for_load(checkpoint)

        self._ckpt_model_name_to_index = {
            dataset_name: data_indices.name_to_index
            for dataset_name, data_indices in checkpoint["hyper_parameters"]["data_indices"].items()
        }

        if not self.config.training.load_weights_only:
            self.task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))

            # Anemoi constructs the task and datasets from the config before Lightning
            # restores their checkpoint state. Now that the checkpoint rollout is restored,
            # update any constructed datasets so workers load the required input and target
            # time steps. Checkpoint conversion loads the module without creating a Trainer
            # or datamodule, so only synchronize if a datamodule is attached.
            trainer = getattr(self, "_trainer", None)
            if trainer is not None and trainer.datamodule is not None:
                trainer.datamodule.sync_dataset_state()

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

        self._apply_scaler_update(name, scaler[1], loss_obj, metrics_dict)

    @classmethod
    def _initialise_updating_scalers(
        cls,
        scalers: dict[str, TENSOR_SPEC],
        updating_scalers: dict[str, BaseUpdatingScaler],
        loss_obj: torch.nn.Module,
        metrics_dict: dict[str, torch.nn.Module],
    ) -> None:
        """Move updating scalers into runtime storage before training starts."""
        for name in updating_scalers:
            cls._apply_scaler_update(name, scalers[name][1], loss_obj, metrics_dict)

    @classmethod
    def _apply_scaler_update(
        cls,
        name: str,
        scaler: torch.Tensor,
        loss_obj: torch.nn.Module,
        metrics_dict: dict[str, torch.nn.Module],
    ) -> None:
        """Apply one updating scaler to every loss or metric that uses it."""
        if cls._can_update_scaler(loss_obj, name):
            loss_obj.update_scaler(scaler=scaler, name=name)

        for metric in metrics_dict.values():
            if cls._can_update_scaler(metric, name):
                metric.update_scaler(scaler=scaler, name=name)

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

    def _prepare_tensors_for_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, slice | None]:
        """Prepare tensors for loss computation, handling sharding if necessary.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        dataset_name : str
            Dataset being processed.
        validation_mode : bool
            Whether in validation mode

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, slice | None]
            Prepared y_pred, y, and grid_shard_slice
        """
        # Handle multi-dataset case for grid shard slice and sizes
        grid_shard_slice = self.grid_shard_slice[dataset_name]
        grid_shard_sizes = self.grid_shard_sizes[dataset_name]

        is_sharded = grid_shard_slice is not None

        sharding_supported = (self.loss_supports_sharding) and (  # loss calculated in training and validation mode
            self.metrics_support_sharding or not validation_mode
        )

        if is_sharded and not sharding_supported:  # gather tensors if loss or metrics do not support sharding
            y_pred_full = gather_tensor(torch.clone(y_pred), self.grid_dim, grid_shard_sizes, self.model_comm_group)
            y_full = gather_tensor(torch.clone(y), self.grid_dim, grid_shard_sizes, self.model_comm_group)
            final_grid_shard_slice = None
        else:
            y_pred_full, y_full = y_pred, y
            final_grid_shard_slice = grid_shard_slice

        return y_pred_full, y_full, final_grid_shard_slice

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        dataset_name : str
            Dataset name for multi-dataset scenarios
        pred_layout : IndexSpace | str | None
            Variable layout of the predictions.
        target_layout : IndexSpace | str | None
            Variable layout of the targets.
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
            # grid_shard_sizes must stay consistent with grid_shard_slice: if the tensors were
            # gathered to the full grid (grid_shard_slice is None), the loss must be told it is
            # not sharded, otherwise it would re-shard an already-full tensor. See _prepare_tensors_for_loss.
            loss_kwargs.update(
                grid_dim=self.grid_dim,
                grid_shard_sizes=self.grid_shard_sizes[dataset_name] if grid_shard_slice is not None else None,
            )

        return loss(y_pred, y, **loss_kwargs)

    def _compute_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
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
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        dataset_name : str | None
            Dataset name for multi-dataset scenarios.
        pred_layout : IndexSpace | str | None
            Variable layout of the predictions.
        target_layout : IndexSpace | str | None
            Variable layout of the targets.
        rollout_step : int | None
            Current rollout step index, used to produce per-step metric key suffixes.
        **_kwargs
            Additional arguments.

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
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        dataset_name: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        validation_mode : bool, optional
            Whether to compute validation metrics
        dataset_name : str | None, optional
            Dataset being processed.
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]
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
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        validation_mode: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : dict[str, torch.Tensor]
            Predicted values
        y : dict[str, torch.Tensor]
            Target values
        validation_mode : bool, optional
            Whether to compute validation metrics
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], dict[str, torch.Tensor]]
            Loss, metrics dictionary (if validation_mode), and full predictions
        """
        assert isinstance(y_pred, dict), "y_pred must be a dict keyed by dataset name"
        assert isinstance(y, dict), "y must be a dict keyed by dataset name"
        # Prepare tensors for loss/metrics computation
        total_loss, metrics_next, y_preds = None, {}, {}
        for dataset_name in self.target_dataset_names:
            if dataset_name not in y_pred:
                err_msg = (
                    f"Your model is not predicting dataset '{dataset_name}' (not included in any decoder) but "
                    f"you have defined a loss function over it."
                )
                raise ValueError(err_msg)

            dataset_loss, dataset_metrics, y_preds[dataset_name] = self.compute_dataset_loss_metrics(
                y_pred[dataset_name],
                y[dataset_name],
                validation_mode=validation_mode,
                dataset_name=dataset_name,
                **kwargs,
            )

            if dataset_loss is not None:
                total_loss = dataset_loss if total_loss is None else total_loss + dataset_loss

                if validation_mode:
                    loss_obj = self.loss[dataset_name]
                    loss_name = getattr(loss_obj, "name", loss_obj.__class__.__name__.lower())
                    metrics_next[f"{dataset_name}_{loss_name}_loss"] = dataset_loss

            # Prefix dataset name to metric keys
            for metric_name, metric_value in dataset_metrics.items():
                metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

        return total_loss, metrics_next, y_preds

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], _: int) -> dict[str, torch.Tensor]:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Also normalize the batch in-place if needed. Batch metadata (reserved
        key ``META_KEY``, e.g. the participant a multi-domain batch was sampled
        from) is split off first and stashed in ``self._current_meta``, so
        nothing downstream sees anything but tensors keyed by dataset name.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch to transfer

        Returns
        -------
        dict[str, torch.Tensor]
            Batch after transfer
        """
        batch, self._current_meta = split_meta(batch)
        participant = meta_participant(self._current_meta)  # also checks the batch is participant-pure
        if participant is not None:
            LOGGER.debug("Batch sampled from participant '%s'", participant)
            # Must happen before sharding: the shard sizes follow the participant's grid.
            self.set_active_participant(participant)
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        # Gathering/sharding of batch
        batch = self._setup_batch_sharding(batch)

        # Batch normalization
        batch = self._normalize_batch(batch)

        # Prepare scalers, e.g. init delayed scalers and update scalers
        self._prepare_loss_scalers()

        return batch

    def _setup_batch_sharding(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Setup batch sharding before every step.

        If the batch is sharded, it will be setup with the grid shard sizes and slice.
        Otherwise, the batch will be allgathered.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch to setup

        Returns
        -------
        dict[str, torch.Tensor]
            Batch after setup
        """
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        self.grid_shard_sizes = {}
        self.grid_shard_slice = {}

        for dataset_name in self.dataset_names:
            if self.keep_batch_sharded and self.model_comm_group_size > 1:
                self.grid_shard_sizes[dataset_name] = self.shard_sizes[dataset_name]
                start, end = get_partition_range(
                    partition_sizes=self.grid_shard_sizes[dataset_name],
                    partition_id=self.reader_group_rank,
                )
                self.grid_shard_slice[dataset_name] = slice(start, end)
            else:
                self.grid_shard_sizes[dataset_name] = None
                self.grid_shard_slice[dataset_name] = None
                batch[dataset_name] = self.allgather_batch(batch[dataset_name], dataset_name)
        return batch

    def transfer_batch_to_device(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        _dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Transfer batch to device, handling dictionary batches."""
        transferred_batch = {}
        for dataset_name, dataset_batch in batch.items():
            transferred_batch[dataset_name] = (
                dataset_batch.to(device, non_blocking=True)
                if isinstance(dataset_batch, torch.Tensor)
                else dataset_batch
            )
        return transferred_batch

    def _normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize batch for training and validation before every step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch to prepare

        Returns
        -------
        dict[str, torch.Tensor]
            Normalized batch
        """
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        for dataset_name in batch:
            batch[dataset_name] = self.model.pre_processors[dataset_name](batch[dataset_name])  # normalized in-place
        return batch

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
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        pass

    def allgather_batch(self, batch: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank
        dataset_name : str
            Dataset name

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_size = self.grid_sizes[dataset_name]
        grid_shard_sizes = self.shard_sizes[dataset_name]

        if grid_size == batch.shape[self.grid_dim] or self.reader_group_size == 1:
            return batch  # already have the full grid

        return gather_tensor(
            batch,
            self.grid_dim,
            grid_shard_sizes,
            self.reader_groups[self.reader_group_id],
        )

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        step: int | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        without_scalers: list[str] | list[int] | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted ensemble
        y : torch.Tensor
            Ground truth (target).
        grid_shard_slice : slice | None, optional
            Grid shard slice for distributed validation.
        dataset_name : str | None, optional
            Dataset being processed.
        step : int | None, optional
            Step number
        pred_layout : IndexSpace | str | None, optional
            Variable layout of the predictions.
        target_layout : IndexSpace | str | None, optional
            Variable layout of the targets.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to omit from metric calculation.
        **_kwargs
            Additional arguments.

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

        y_postprocessed = post_processor(y, in_place=False)
        y_pred_postprocessed = post_processor(y_pred, in_place=False)

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

                scaler_index = torch.as_tensor(indices, device=y_pred_postprocessed.device, dtype=torch.long)
                metric_kwargs = {
                    "scaler_indices": (..., scaler_index),
                    "grid_shard_slice": grid_shard_slice,
                    "group": self.model_comm_group,
                }
                # tensor 'scaler_indices[1]' size mismatch at index 0. expected 13, actual 1"
                torch._dynamo.mark_dynamic(metric_kwargs["scaler_indices"][-1], 0)
                if pred_layout is not None:
                    metric_kwargs["pred_layout"] = pred_layout
                if target_layout is not None:
                    metric_kwargs["target_layout"] = target_layout
                if without_scalers is not None:
                    metric_kwargs["without_scalers"] = without_scalers
                if getattr(metric, "needs_shard_layout_info", False):
                    # grid_shard_sizes must stay consistent with grid_shard_slice: if the tensors
                    # were gathered to the full grid (grid_shard_slice is None), the metric must be
                    # told it is not sharded, otherwise it would re-shard an already-full tensor.
                    metric_kwargs.update(
                        grid_dim=self.grid_dim,
                        grid_shard_sizes=self.grid_shard_sizes[dataset_name] if grid_shard_slice is not None else None,
                    )

                metric_value = metric(y_pred_postprocessed, y_postprocessed, **metric_kwargs)
                # Detach and clone the metric value to avoid in-place modifications affecting the original tensor
                # This was impacting cuda graphs
                # TODO(cathal): double check now that everything compiles
                metrics[metric_step_name] = metric_value.detach().clone()

        return metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        # Get batch size (handle dict of tensors)
        batch_size = next(iter(batch.values())).shape[0]

        step_output = self._step(batch)
        train_loss = step_output.loss

        self.log(
            "train_" + self._get_loss_name() + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch_size,
            sync_dist=True,
        )

        self.task.log_extra(logger=self.log, logger_enabled=self.logger_enabled)

        return train_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> TrainingStepOutput:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        TrainingStepOutput
            Output of the validation step.
        """
        del batch_idx
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"

        # Get batch size (handle dict of tensors)
        batch_size = next(iter(batch.values())).shape[0]

        with torch.no_grad():
            step_output = self._step(batch, validation_mode=True)
        val_loss = step_output.loss
        metrics = step_output.metrics

        self.log(
            "val_" + self._get_loss_name() + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch_size,
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Per-participant copies of the same metrics (no-op when the batch carries no participant).
        self._participant_metrics.update(
            meta_participant(self._current_meta),
            {self._get_loss_name() + "_loss": val_loss, **metrics},
            batch_size,
        )

        return step_output

    def on_validation_epoch_start(self) -> None:
        self._participant_metrics.reset()
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        # Collective call on every rank (see ParticipantMetrics.compute); logs nothing without participants.
        for participant, metrics in self._participant_metrics.compute(self.device).items():
            for mname, mvalue in metrics.items():
                self.log(f"val_{mname}/{participant}", mvalue, on_epoch=True, logger=self.logger_enabled)
        self._participant_metrics.reset()
        super().on_validation_epoch_end()

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

    def on_train_start(self) -> None:
        """Log the effective task state after checkpoint restoration."""
        super().on_train_start()
        self.task.log_training_state()

    def on_train_epoch_end(self) -> None:
        self.task.on_train_epoch_end(current_epoch=self.current_epoch)
        # Default epoch checkpoints are saved at validation end, before this
        # hook. On resume Lightning finishes the saved epoch here, advancing the
        # dataloader before newly created workers derive the seed for that epoch.
        self.trainer.datamodule.set_epoch(self.current_epoch + 1)
        super().on_train_epoch_end()

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

    def _resolve_subgrid(self, config: dict) -> None:
        def per_dataset_resolve(per_dataset_config: dict, dataset_name: str) -> None:
            for k, v in per_dataset_config.items():
                if isinstance(v, dict):
                    per_dataset_resolve(v, dataset_name)
                elif (k, v) == ("subgrid", "output_mask"):
                    per_dataset_config[k] = self.output_mask[dataset_name].as_tuple()

        for dataset_name, dataset_config in config.items():
            if dataset_config is not None:
                per_dataset_resolve(dataset_config, dataset_name)
