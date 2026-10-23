# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

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

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.models.data import Batch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.loading.base import extract_checkpoint_variables_metadata
from anemoi.training.checkpoint.loading.base import preserve_anemoi_metadata
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.utils import check_loss_tree_variable_units
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.masks import BaseMask
from anemoi.training.utils.masks import create_output_masks
from anemoi.training.utils.resolve_config import is_container
from anemoi.training.utils.resolve_config import is_mapping
from anemoi.training.utils.resolve_config import is_sequence
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import choose_base_seed
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.seeding import seed_random_generators
from anemoi.training.utils.seeding import set_base_seed
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
    from pytorch_lightning.utilities.types import OptimizerLRScheduler
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data.sources.base import Source
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.models.distributed.shapes import DatasetShardSizes
    from anemoi.models.distributed.shapes import ShardSizes
    from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
    from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
    from anemoi.training.losses.scalers.base_scaler import BaseScaler
    from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.tasks.base import BaseTask

LOGGER = logging.getLogger(__name__)


# helpers for building the loss config
def resolve_subgrid_node(node: object, output_mask: object, dataset_name: str) -> None:
    """Recursively replace ``subgrid: output_mask`` placeholders with the mask tuple."""
    if is_mapping(node):
        for k, v in node.items():
            if is_container(v) and not isinstance(v, str):
                resolve_subgrid_node(v, output_mask, dataset_name)
            elif (k, v) == ("subgrid", "output_mask"):
                node[k] = output_mask.as_tuple()
                LOGGER.info("Resolved subgrid for dataset '%s' to output_mask as tuple: %s", dataset_name, node[k])
    elif is_sequence(node) and not isinstance(node, str):
        for item in node:
            resolve_subgrid_node(item, output_mask, dataset_name)


def resolve_subgrid(config: Mapping, output_mask: Mapping) -> None:
    """Resolve ``subgrid: output_mask`` placeholders for every dataset in ``config``."""
    for dataset_name, dataset_config in config.items():
        # Loss configs may name datasets the model does not have (e.g. a default ``data`` entry
        # merged into a multi-dataset config); no loss is built for those, so skip them.
        if dataset_config is not None and dataset_name in output_mask:
            resolve_subgrid_node(dataset_config, output_mask[dataset_name], dataset_name)


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
    statistics : dict
        Dictionary of training statistics (mean, std, etc.) used for normalization.
    statistics_tendencies : dict
        Statistics related to tendencies (if used).
    data_indices : dict[str, IndexCollection]
        Maps feature names to index ranges used for training and loss functions.
    data_types : dict[str, str]
        Kind of data of each dataset (``"gridded"``, ``"tabular"``), passed to the model interface.
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
    n_step_input : dict[str, int]
        Number of input timesteps provided to the model for each dataset and location.
    n_step_output : dict[str, int]
        Number of output timesteps predicted by the model for each dataset and location.
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
    - `create_scalers`

    """

    def __init__(
        self,
        *,
        config: BaseSchema,
        task: BaseTask,
        statistics: dict[str, dict],
        statistics_tendencies: dict[str, dict],
        data_indices: dict[str, IndexCollection],
        data_types: dict[str, str],
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
        statistics : dict
            Statistics of the training data
        statistics_tendencies : dict
            Statistics of data tendencies.
        data_indices : dict[str, IndexCollection]
            Indices of the training data.
        data_types : dict[str, str]
            Kind of data of each dataset (``"gridded"``, ``"tabular"``).
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()
        self.task = task
        self.config = config

        assert isinstance(data_indices, dict), "data_indices must be a dict keyed by dataset name"

        self.dataset_names = list(data_indices.keys())

        # Define number of input/output timesteps per node for each dataset
        self.n_step_input = self.task.num_input_timesteps_by_dataset or dict.fromkeys(
            self.dataset_names,
            self.task.num_input_timesteps,
        )
        self.n_step_output = self.task.num_output_timesteps_by_dataset or dict.fromkeys(
            self.dataset_names,
            self.task.num_output_timesteps,
        )

        self.model = AnemoiModelInterface(
            config=config,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            data_types=data_types,
            metadata=metadata,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            supporting_arrays=supporting_arrays.copy(),
        )

        # Output masks read node attributes (e.g. the LAM `cutout_mask`) from the graph,
        # which the model builds, so they are created once the model exists.
        self.output_mask: dict[str, BaseMask] = create_output_masks(
            get_multiple_datasets_config(config.model.output_mask),
            graph_data=self.model.graph_data,
            dataset_names=self.dataset_names,
        )

        # Handle supporting_arrays merge with all output masks
        for dataset_name, mask in self.output_mask.items():
            self.model.supporting_arrays[dataset_name].update(mask.supporting_arrays)

        self.data_indices = data_indices

        # `task` holds runtime objects that cannot be deepcopied by Lightning's
        # `save_hyperparameters` and are not meaningful to persist in the
        # checkpoint hyperparameters anyway.
        self.save_hyperparameters(
            ignore=["task", *config.diagnostics.checkpoint.exclude_hyperparameters],
        )

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
        loss_configs = self._build_loss_configs()
        zero_target_losses = [
            name
            for name, loss_config in loss_configs.items()
            if loss_config is not None and self.n_step_output[name] == 0
        ]
        if zero_target_losses:
            msg = f"Datasets {zero_target_losses} have a training loss configured but no model targets."
            raise ValueError(msg)

        scalers_configs = get_multiple_datasets_config(config.training.scalers)
        val_metrics_configs = get_multiple_datasets_config(config.training.validation_metrics)
        metrics_to_log = get_multiple_datasets_config(config.training.metrics)
        # Losses that declare needs_graph_data (spectral projections, on-the-fly multiscale
        # smoothers, graph scores) read the model graph.
        graph_data = self.model.graph_data
        for dataset_name in self.dataset_names:
            if dataset_name not in loss_configs or loss_configs[dataset_name] is None:
                LOGGER.warning("Dataset %s is skipped for loss & metric computation.", dataset_name)
                continue

            self.target_dataset_names.append(dataset_name)

            # The model graph has one node group per dataset (see BaseGraphModel).
            data_node_name = dataset_name

            # Create dataset-specific metadata extractor
            metadata_extractor = ExtractVariableGroupAndLevel(
                variable_groups=dataset_variable_groups[dataset_name],
                metadata_variables=metadata["dataset"][dataset_name].get("variables_metadata"),
            )

            dataset_scalers, dataset_updating_scalars = create_scalers(
                scalers_configs[dataset_name],
                dataset_name=dataset_name,
                data_indices=data_indices[dataset_name],
                task=self.task,
                graph_data=graph_data,
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
                graph_data=graph_data,
                data_node_name=data_node_name,
            )

            # Check unit compatibility between predicted and target variables
            ds_variables_metadata = metadata["dataset"][dataset_name].get("variables_metadata")
            check_loss_tree_variable_units(self.loss[dataset_name], ds_variables_metadata)

            self.metrics[dataset_name] = self._build_metrics_for_dataset(
                val_metrics_configs[dataset_name],
                scalers=dataset_scalers,
                data_indices=data_indices[dataset_name],
                graph_data=graph_data,
                data_node_name=data_node_name,
            )
            self._initialise_updating_scalers(
                scalers=dataset_scalers,
                updating_scalers=dataset_updating_scalars,
                loss_obj=self.loss[dataset_name],
                metrics_dict=self.metrics[dataset_name],
            )
            self._scaling_values_log[dataset_name] = print_variable_scaling(
                self.loss[dataset_name],
                data_indices[dataset_name],
            )

        # Loss gradient scaling (arXiv:2306.06079, section 4.3.2) has been removed for now, to be revisited.
        # The config key stays so existing configs validate; asking for it fails loudly.
        if config.training.loss_gradient_scaling:
            msg = "training.loss_gradient_scaling is temporarily unavailable; set it to false."
            raise NotImplementedError(msg)

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

        self._validate_spatial_processor_target_grid(self.model.graph_data)

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

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_size = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1

        # Static (time-invariant) coordinate tensors, kept on the model device across
        # steps so transfer_batch_to_device copies them once per run instead of per
        # batch. Derived state - deliberately not a buffer, nothing to checkpoint.
        self._static_coord_cache: dict[str, torch.Tensor] = {}

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
        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            **kwargs,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["task_state"] = self.task.training_runtime_state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # The task's training runtime state (e.g. rollout step) is resume state the
        # checkpoint pipeline does not apply, so restore it here.
        self.task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))

        # A resume is loaded once, by Trainer.fit(ckpt_path=), which calls this hook
        # on the dict it is about to load: this is the only place the corrections can
        # run for it (the pipeline resolves the file and loads nothing). Lightning
        # holds a reference to this dict, so a replacement (a migration may return a
        # new object) is written back in place rather than rebound.
        trainer = getattr(self, "_trainer", None)
        corrected = apply_checkpoint_corrections(checkpoint, self, self.config)
        if corrected is not checkpoint:
            checkpoint.clear()
            checkpoint.update(corrected)

        # Checkpoint conversion loads the module without creating a Trainer; only a
        # training run that continues from the checkpoint takes over its seed.
        if trainer is not None:
            self._restore_base_seed(checkpoint)

        # Anemoi constructs the task and datasets from the config before Lightning
        # restores their checkpoint state. Now that the checkpoint rollout is restored,
        # update any constructed datasets so workers load the required input and target
        # time steps. Checkpoint conversion creates no datamodule either, so only
        # synchronize if a datamodule is attached.
        if trainer is not None and trainer.datamodule is not None:
            trainer.datamodule.sync_dataset_state()

        preserve_anemoi_metadata(self, checkpoint)
        extract_checkpoint_variables_metadata(self, checkpoint)

    def _restore_base_seed(self, checkpoint: dict) -> None:
        """Continue with the checkpoint's base seed, unless ``training.seed`` sets one.

        The chosen seed replaces the one picked at start-up, before the data order and
        the random numbers of the training steps are derived from it. It also goes into
        the run metadata, together with the trainer seed derived from it, so that the
        checkpoints of this run store the seed it trains with.
        """
        checkpoint_seed = checkpoint["hyper_parameters"]["metadata"].get("base_seed")
        if checkpoint_seed is None:
            # Checkpoints from older versions store no base seed. The run keeps the seed it
            # started with, which all ranks already share.
            LOGGER.warning(
                "The checkpoint stores no base seed; training continues with base seed %d, "
                "and the run does not continue exactly.",
                get_base_seed(),
            )
            return

        base_seed = choose_base_seed(self.config.training.seed, checkpoint_seed)
        set_base_seed(base_seed)
        LOGGER.info("Training continues with base seed %d", base_seed)

        trainer_seed = derive_seed(base_seed, SeedContext.TRAINER)
        metadata = self.hparams["metadata"]
        for entry in (metadata, metadata["metadata_inference"]):
            entry["base_seed"] = base_seed
            entry["seed"] = trainer_seed

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

    def _grid_shard_sizes(self, source: Source | Batch) -> ShardSizes | DatasetShardSizes:
        """Grid shard sizes of the source, or per dataset when given a Batch.

        Returns None when the source is replicated (not sharded) or tabular.
        """
        if isinstance(source, Batch):
            return {name: self._grid_shard_sizes(dataset_source) for name, dataset_source in source.items()}
        return source.grid_shard_sizes

    def _grid_shard_slice(self, source: Source) -> slice | None:
        """Local grid shard slice for ``source``, derived from its shard sizes.

        Returns None when the source is replicated (not sharded).
        """
        # Tabular sources report no grid shard sizes: they have no per-grid scalers/masks to slice.
        shard_sizes = source.grid_shard_sizes
        if not shard_sizes:
            return None
        assert isinstance(shard_sizes, list) and all(
            isinstance(s, int) for s in shard_sizes
        ), "shard_sizes must be a list of integers"
        start, end = get_partition_range(shard_sizes, self.model_comm_group_rank)
        return slice(start, end)

    def _prepare_tensors_for_loss(
        self,
        y_pred: Source,
        y: Source,
        dataset_name: str,
        validation_mode: bool = False,
    ) -> tuple[Source, Source, slice | None]:
        """Prepare tensors for loss computation, handling sharding if necessary.

        Parameters
        ----------
        y_pred : Source
            Predicted values.
        y : Source
            Target values.
        dataset_name : str
            Dataset being processed.
        validation_mode : bool
            Whether in validation mode.

        Returns
        -------
        tuple[Source, Source, slice | None]
            Prepared y_pred, y, and grid_shard_slice.
        """
        # Sharding metadata now lives on the source views (None when replicated).
        # A single gather decision is applied to both views, so they must agree.
        y_shard_sizes = getattr(y, "shard_sizes", None)
        y_pred_shard_sizes = getattr(y_pred, "shard_sizes", None)
        if y_shard_sizes != y_pred_shard_sizes:
            msg = (
                f"Prediction and target sharding disagree for dataset {dataset_name!r}: "
                f"prediction shard_sizes={y_pred_shard_sizes}, target shard_sizes={y_shard_sizes}. "
                "Both views must be sharded identically (or both replicated) before the loss."
            )
            raise ValueError(msg)

        # grid_shard_slice is gridded-only (it slices per-grid scalers/masks), so the gather
        # decision must come from the shard metadata itself - otherwise a sharded tabular
        # view would be left un-gathered and the loss computed on a partial grid.
        grid_shard_slice = self._grid_shard_slice(y)
        is_sharded = y_shard_sizes is not None

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

    @staticmethod
    def _evaluate_loss(
        loss: Callable,
        pred: Source,
        target: Source,
        **kwargs,
    ) -> torch.Tensor:
        """Check training precision before promoting inputs and evaluating the loss."""
        full_precision = (torch.float32, torch.float64)
        assert (
            pred.dtype in full_precision
        ), f"Prediction for {pred.name!r} must be float32 or float64, got {pred.dtype}."
        assert (
            target.dtype in full_precision
        ), f"Target for {target.name!r} must be float32 or float64, got {target.dtype}."
        dtype = torch.promote_types(torch.promote_types(pred.dtype, target.dtype), torch.float32)
        pred = pred.map_data(lambda data: data.to(dtype))
        target = target.map_data(lambda data: data.to(dtype))
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return loss(pred, target, **kwargs)

    @staticmethod
    def _nan_flags_per_variable(source: Source, variables: Mapping[str, Any]) -> tuple[list[int], torch.Tensor] | None:
        """Return the positions in ``source`` of the ``variables`` it holds, and whether each contains a NaN.

        The flags stay on the device, so callers can check several sources with a single sync.
        Returns ``None`` if ``source`` holds none of ``variables``.
        """
        indices = [i for i, name in enumerate(source.variables) if name in variables]
        if not indices:
            return None
        samples = source.data if source.is_tabular else [source.data]
        nan_per_variable = torch.zeros(len(indices), dtype=torch.bool, device=samples[0].device)
        for sample in samples:
            variables_axis = source.layout.axis("variables", ndim=sample.ndim)
            values = sample.index_select(variables_axis, torch.as_tensor(indices, device=sample.device))
            nan_per_variable |= torch.isnan(values.movedim(variables_axis, -1)).reshape(-1, len(indices)).any(dim=0)
        return indices, nan_per_variable

    @staticmethod
    def _variables_with_nans(source: Source, indices: list[int], nan_per_variable: torch.Tensor) -> list[str]:
        """Return the names of the flagged variables."""
        return [source.variables[i] for i, has_nan in zip(indices, nan_per_variable.tolist(), strict=True) if has_nan]

    def _check_nan_inputs(self, batch: Batch) -> None:
        """Fail loudly when model inputs still contain NaNs after preprocessing.

        The model has no NaN handling: one NaN input spreads through the graph and makes the whole
        prediction NaN. Losses with ``ignore_nans=True`` then mask every point and report a loss of 0
        with no gradient, so training silently stops learning. Missing input values must be filled
        by an imputer in the dataset's processors. Only the model's input variables are checked.

        Parameters
        ----------
        batch : Batch
            Preprocessed model inputs.
        """
        flags = {
            name: self._nan_flags_per_variable(source, self.data_indices[name].model.input.name_to_index)
            for name, source in batch.items()
            if name in self.data_indices
        }
        flags = {name: flag for name, flag in flags.items() if flag is not None}
        if not flags or not bool(torch.cat([nan_per_variable for _, nan_per_variable in flags.values()]).any()):
            return

        nan_inputs = {
            name: nan_variables
            for name, flag in flags.items()
            if (nan_variables := self._variables_with_nans(batch[name], *flag))
        }
        msg = (
            f"Model inputs contain NaNs after preprocessing: {nan_inputs}. The model cannot handle NaN inputs: "
            "the whole prediction becomes NaN, and losses with ignore_nans=True then report 0. Add an imputer "
            "(e.g. anemoi.models.preprocessing.imputer.InputImputer) for these variables to the dataset's processors."
        )
        raise ValueError(msg)

    def _check_nan_targets(self, y: Source, dataset_name: str, losses: dict[str, torch.nn.Module]) -> None:
        """Fail loudly when targets contain NaNs that a loss or metric cannot handle.

        Targets are not imputed, so missing values reach the losses as NaNs. A loss or metric
        with ``ignore_nans=False`` would then return NaN. The check only runs when such a loss is
        configured for ``dataset_name``, and only looks at the model's output variables.

        Parameters
        ----------
        y : Source
            Target values.
        dataset_name : str
            Dataset name.
        losses : dict[str, torch.nn.Module]
            Losses or metrics to check, keyed by the name used in the error message.
        """
        # Only the leaf losses compute anything: a container such as CombinedLoss keeps its own
        # (unused) ignore_nans=False even when all of its children ignore NaNs.
        intolerant = [
            name
            for name, loss in losses.items()
            if any(
                getattr(leaf, "ignore_nans", True) is False
                for leaf in (loss.iter_leaf_losses() if isinstance(loss, BaseLoss) else loss.modules())
            )
        ]
        if not intolerant:
            return

        flags = self._nan_flags_per_variable(y, self.data_indices[dataset_name].model.output.name_to_index)
        if flags is None or not bool(flags[1].any()):
            return

        nan_variables = self._variables_with_nans(y, *flags)
        msg = (
            f"Dataset '{dataset_name}': the targets of {nan_variables} contain NaNs, but {intolerant} "
            "use ignore_nans=False, so they would be NaN. Targets are not imputed; set ignore_nans: True "
            "for these losses and metrics."
        )
        raise ValueError(msg)

    def _compute_loss(
        self,
        y_pred: Source,
        y: Source,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the loss function.

        Parameters
        ----------
        y_pred : Source
            Predicted values.
        y : Source
            Target values.
        grid_shard_slice : slice | None
            Grid shard slice for distributed training.
        dataset_name : str
            Dataset name for multi-dataset scenarios.
        pred_layout : IndexSpace | str | None
            Variable layout of the predictions.
        target_layout : IndexSpace | str | None
            Variable layout of the targets.
        **_kwargs
            Additional arguments.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        loss = self.loss[dataset_name]
        self._check_nan_targets(y, dataset_name, {"training_loss": loss})
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
                grid_shard_sizes=self._grid_shard_sizes(y),
            )

        return self._evaluate_loss(loss, y_pred, y, **loss_kwargs)

    def _compute_metrics(
        self,
        y_pred: Source,
        y: Source,
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
        y_pred : Source
            Predicted values.
        y : Source
            Target values.
        grid_shard_slice : slice | None
            Grid shard slice for distributed training.
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
            Computed metrics.
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
        y_pred: Source,
        y: Source,
        validation_mode: bool = False,
        dataset_name: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], Source]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : Source
            Predicted values.
        y : Source
            Target values.
        validation_mode : bool, optional
            Whether to compute validation metrics.
        dataset_name : str | None, optional
            Dataset being processed.
        **kwargs
            Additional arguments to pass to loss computation.

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], Source]
            Loss, metrics dictionary (if validation_mode), and full predictions.
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
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], Batch]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : Batch
            Predicted values.
        y : Batch
            Target values.
        validation_mode : bool, optional
            Whether to compute validation metrics.
        **kwargs
            Additional arguments to pass to loss computation.

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], Batch]
            Loss, metrics dictionary (if validation_mode), and full predictions.
        """
        assert isinstance(y_pred, Batch), "y_pred must be a dict keyed by dataset name"
        assert isinstance(y, Batch), "y must be a dict keyed by dataset name"
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
                    metrics_next[self._dataset_loss_metric_name(dataset_name)] = dataset_loss

            # Prefix dataset name to metric keys
            for metric_name, metric_value in dataset_metrics.items():
                metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

        return total_loss, metrics_next, Batch(y_preds)

    def _map_dataset_processors(self, batch: Batch, processors: Any, **kwargs) -> Batch:
        """Apply per-dataset processors without mutating the selected batch."""
        processed_batch = batch
        for dataset_name, processor in processors.items():
            if dataset_name in batch:
                processed_batch = processed_batch.replace(
                    dataset_name,
                    processor(batch[dataset_name], in_place=False, **kwargs),
                )
        return processed_batch

    def preprocess_inputs(self, batch: Batch) -> Batch:
        """Transform selected model inputs into model-input space, and check that no NaNs are left."""
        processed = self._map_dataset_processors(batch, self.model.pre_processors)
        self._check_nan_inputs(processed)
        return processed

    def preprocess_targets(self, batch: Batch) -> Batch:
        """Normalize selected targets while preserving missing values."""
        return self._map_dataset_processors(batch, self.model.pre_processors, skip_imputation=True)

    def postprocess_targets(
        self,
        batch: Batch,
        layouts: dict[str, IndexSpace | str | None] | IndexSpace | str | None = None,
    ) -> Batch:
        """Transform normalized targets to physical space without mutating them."""
        processed = batch
        for dataset_name in self.model.post_processors:
            if dataset_name not in batch:
                continue
            layout = layouts.get(dataset_name) if isinstance(layouts, dict) else layouts
            processed = processed.replace(
                dataset_name,
                self._postprocess_dataset_view(batch[dataset_name], dataset_name, layout),
            )
        return processed

    def _postprocess_dataset_view(
        self,
        view: Source,
        dataset_name: str,
        layout: IndexSpace | str | None = None,
    ) -> Source:
        """Postprocess one view using the same metadata alignment as batch targets."""
        aligned = self._align_view_to_layout(view, layout, dataset_name)
        return self.model.post_processors[dataset_name](aligned, in_place=False)

    def _dataset_loss_metric_name(self, dataset_name: str) -> str:
        """Metric key under which the validation loss of ``dataset_name`` is logged."""
        loss_obj = self.loss[dataset_name]
        loss_name = getattr(loss_obj, "name", loss_obj.__class__.__name__.lower())
        return f"{dataset_name}_{loss_name}_loss"

    def on_after_batch_transfer(self, batch: Batch, _: int) -> Batch:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Also normalize the batch in-place if needed.

        Parameters
        ----------
        batch : Batch
            Batch to transfer.

        Returns
        -------
        Batch
            Batch after transfer.
        """
        assert isinstance(batch, Batch), "batch must be a Batch instance"
        # Gathering/sharding of batch
        if not self.keep_batch_sharded:
            batch = self.allgather_batch(batch)

        # Spatial preprocessing (e.g. CrossGridProjector for downscaling).
        # Owned by the model; applied before normalization so projectors see raw values.
        batch = self.model.apply_spatial_pre_processors(batch, model_comm_group=self.model_comm_group)

        # Debug-log the batch contents (per-dataset shape + layout) so that
        # layout/shape mismatches can be diagnosed from a real run.
        LOGGER.debug("on_after_batch_transfer batch:\n%r", batch)

        return batch

    def transfer_batch_to_device(
        self,
        batch: Batch,
        device: torch.device,
        _dataloader_idx: int = 0,
    ) -> Batch:
        """Transfer the :class:`Batch` to ``device``, reusing cached static coordinates."""
        return batch.to(device, non_blocking=True, static_coord_cache=self._static_coord_cache)

    @abstractmethod
    def _step(
        self,
        batch: Batch,
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        pass

    def _combine_loss_and_metrics(
        self,
        step_losses: list[torch.Tensor],
        step_metrics: list[dict[str, torch.Tensor]],
        predictions: list[dict[str, torch.Tensor]],
    ) -> TrainingStepOutput:
        """Average all losses over the training rollout; keep the per-step metrics of every unrolled step.

        Validation may unroll more steps than training. The total loss and the per-dataset
        losses only cover the first ``task.num_steps`` steps, so ``val_*_loss`` is directly
        comparable to ``train_*_loss``. Validation metrics are reported for every step.
        """
        num_loss_steps = self.task.num_steps
        loss = sum(step_losses[:num_loss_steps]) / num_loss_steps
        loss_names = {self._dataset_loss_metric_name(name) for name in self.target_dataset_names}
        metrics: dict[str, torch.Tensor] = {}
        for i, metrics_next in enumerate(step_metrics):
            for name, value in metrics_next.items():
                if name not in loss_names:
                    metrics[name] = value
                elif i < num_loss_steps:
                    metrics[name] = metrics.get(name, 0.0) + value / num_loss_steps
        return TrainingStepOutput(loss=loss, metrics=metrics, predictions=predictions)

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the shards of a grid-sharded tensor across the reader group.

        The shard sizes must be supplied by the caller because a tensor may live on the
        reader grid (before spatial preprocessing) or on the projected grid (after it).

        Gathering over the reader group is valid for both: post-projection shards are laid
        out over the model comm group, and ``keep_batch_sharded`` asserts in ``__init__``
        that the reader group and the model comm group coincide. When they do not,
        ``grid_shard_sizes`` is ``None`` post-projection and this returns early.

        Parameters
        ----------
        batch : torch.Tensor
            Grid-shard of the current reader rank.

        Returns
        -------
        torch.Tensor
            Allgathered (full) tensor.
        """
        # Without reader groups (single-device training) there is a single reader, so nothing is
        # sharded; Source.allgather treats a None group as single-rank and validates any shard sizes.
        group = None if self.reader_groups is None else self.reader_groups[self.reader_group_id]
        return batch.allgather(group)

    def _align_view_to_layout(
        self,
        view: Source,
        layout: IndexSpace | str | None,
        dataset_name: str,
    ) -> Source:
        """Select the data and metadata for the requested variable index space.

        Views already in that space are returned unchanged.
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
        return view.select(variables=positions)

    def calculate_val_metrics(
        self,
        y_pred: Source,
        y: Source,
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
        y_pred: Source
            Predicted ensemble.
        y: Source
            Ground truth (target).
        grid_shard_slice : slice | None, optional
            Grid shard slice for distributed validation.
        dataset_name : str | None, optional
            Dataset being processed.
        step : int | None, optional
            Step number.
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
            validation metrics and predictions.
        """
        metrics = {}

        # Handle multi-dataset case for post-processors
        metrics_dict = self.metrics[dataset_name]
        self._check_nan_targets(
            y,
            dataset_name,
            {f"validation_metrics.{name}": metric for name, metric in metrics_dict.items()},
        )
        val_metric_ranges = self.val_metric_ranges[dataset_name]

        # y (target) and y_pred (model output) can live in different index
        # spaces and therefore hold different numbers of variables along the
        # variable axis.
        # The metadata of the prediction view must be realigned to
        # the variables actually present in its tensor *before* the normalisation
        y_postprocessed = self._postprocess_dataset_view(y, dataset_name, target_layout)
        y_pred_postprocessed = self._postprocess_dataset_view(y_pred, dataset_name, pred_layout)

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
                        "Validation metrics cannot be scaled over the variable dimension in the post processed space."
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
                        grid_shard_sizes=self._grid_shard_sizes(y),
                    )

                metric_value = metric(y_pred_postprocessed, y_postprocessed, **metric_kwargs)
                # Detach and clone the metric value to avoid in-place modifications affecting the original tensor
                # This was impacting cuda graphs
                # TODO(cathal): double check now that everything compiles
                metrics[metric_step_name] = metric_value.detach().clone()

        return metrics

    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> None:
        """Seed the random number generators for this training batch.

        The seed depends only on the base seed, the model communication group, the
        epoch and the batch, so a run resumed from a checkpoint draws the same random
        numbers as an uninterrupted run. Ranks of one model group share the seed and
        stay in step; different groups, such as different ensemble members, draw
        different numbers. Seeding takes well under a millisecond and runs on the
        host only, hence does not slow down training.
        """
        del batch
        seed = derive_seed(
            get_base_seed(),
            SeedContext.TRAINING_BATCH,
            self.model_comm_group_id,
            self.current_epoch,
            batch_idx,
        )
        seed_random_generators(seed)
        LOGGER.debug("Epoch %d, batch %d: random number generators seeded with %d", self.current_epoch, batch_idx, seed)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        assert isinstance(batch, Batch), "batch must be a Batch instance"

        step_output = self._step(batch)
        train_loss = step_output.loss

        self.log(
            "train_" + self._get_loss_name() + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.batch_size,
            sync_dist=True,
        )

        self.task.log_extra(logger=self.log, logger_enabled=self.logger_enabled, batch_size=batch.batch_size)

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
        val_loss = step_output.loss
        metrics = step_output.metrics

        self.log(
            "val_" + self._get_loss_name() + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.batch_size,
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
                batch_size=batch.batch_size,
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

    def on_train_start(self) -> None:
        """Log the effective task state after checkpoint restoration."""
        super().on_train_start()
        self.task.log_training_state()

    def _ran_all_batches_of_epoch(self) -> bool:
        """Whether the epoch that is ending trained on all of its batches.

        Training that stops early, for example on max_steps or a time limit, also
        ends the current epoch. The checkpoint saved at the end of training then
        resumes inside that epoch.
        """
        batches_run = self.trainer.fit_loop.epoch_loop.batch_progress.current.ready
        return batches_run == self.trainer.num_training_batches

    def on_train_epoch_end(self) -> None:
        # The rollout and the data shuffle move on only after a full epoch, so that
        # a run resumed inside an epoch keeps that epoch's rollout and shuffle.
        if self._ran_all_batches_of_epoch():
            self.task.on_train_epoch_end(current_epoch=self.current_epoch)
            # Default epoch checkpoints are saved at validation end, before this
            # hook. On resume Lightning finishes the saved epoch here, so the
            # datamodule also reaches the next epoch before its dataloaders are used.
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

    def _validate_spatial_processor_target_grid(self, graph_data: HeteroData) -> None:
        """Check each spatial projector's target grid against its dataset's graph nodes."""
        for dataset_name, projector in self.model.spatial_pre_processors.items():
            if dataset_name not in graph_data.node_types:
                # Datasets on a dynamic graph (e.g. observations) have no static node set.
                continue
            dataset_grid_size = graph_data[dataset_name].num_nodes

            if projector.output_grid_size != dataset_grid_size:
                msg = (
                    f"Spatial processor for dataset {dataset_name!r} produces a target grid of "
                    f"{projector.output_grid_size} points, but its graph node set {dataset_name!r} has "
                    f"{dataset_grid_size}. The encoder runs after projection, so this node set must be "
                    f"built on the projector's output grid. Check that the graph node set is correctly "
                    "constructed and that the projection matrix matches the graph."
                )
                raise ValueError(msg)

    def _validate_spatial_processor_grid_sizes(self) -> None:
        """Check each spatial projector's source grid against the data it will be fed.

        ``shard_sizes`` is derived from ``input_grid_size`` and is what the reader slices
        zarr with, so a disagreement truncates every rank's read instead of raising.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not self.model.spatial_pre_processors:
            return

        data_readers = datamodule.ds_train.data_readers
        for dataset_name, projector in self.model.spatial_pre_processors.items():
            reader = data_readers.get(dataset_name)
            if reader is None:
                msg = (
                    f"Spatial processor configured for dataset {dataset_name!r}, "
                    "but no corresponding training data reader exists. "
                    "Check data.datasets and dataloader.training.datasets."
                )
                raise ValueError(msg)

            if reader.data_type == "gridded" and reader.grid_size != projector.input_grid_size:
                msg = (
                    f"Spatial processor for dataset {dataset_name!r} expects a source grid of "
                    f"{projector.input_grid_size} points, but the dataset provides {reader.grid_size}. "
                    f"Check that the projection matrix matches the dataset resolution."
                )
                raise ValueError(msg)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called after model is initialized but before training starts."""
        if stage == "fit":
            self._validate_spatial_processor_grid_sizes()

        if stage == "fit" and self.trainer.is_global_zero and self.logger is not None:
            hyper_params = OmegaConf.to_container(self.config, resolve=True)
            hyper_params.update({"variable_loss_scaling": self._scaling_values_log})
            self.logger.log_hyperparams(hyper_params)

    def _build_loss_configs(self) -> dict:
        """Fetch the per-dataset training-loss configs and resolve pre-defined entries."""
        loss_configs = get_multiple_datasets_config(self.config.training.training_loss)
        resolve_subgrid(loss_configs, self.output_mask)
        return loss_configs
