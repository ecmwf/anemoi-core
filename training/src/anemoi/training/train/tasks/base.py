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
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class BaseGraphModule(pl.LightningModule, ABC):
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
    graph_data : HeteroData
        Graph-structured input data containing node and edge features.
    truncation_data : dict
        Information for input/output truncation masks.
    statistics : dict
        Dictionary of training statistics (mean, std, etc.) used for normalization.
    statistics_tendencies : dict
        Statistics related to tendencies (if used).
    data_indices : IndexCollection
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
    multi_step : bool
        Flag to enable autoregressive rollouts (used in multi-step forecasting).
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
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        self.output_mask = instantiate(config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data)

        self.model = AnemoiModelInterface(
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            truncation_data=truncation_data,
            config=convert_to_omegaconf(config),
        )
        self.config = config
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.statistics_tendencies = statistics_tendencies

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        metadata_extractor = ExtractVariableGroupAndLevel(
            variable_groups=config.model_dump(by_alias=True).training.variable_groups,
            metadata_variables=metadata["dataset"].get("variables_metadata"),
        )

        # Instantiate all scalers with the training configuration
        self.scalers, self.updating_scalars = create_scalers(
            config.model_dump(by_alias=True).training.scalers,
            data_indices=data_indices,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            metadata_extractor=metadata_extractor,
            output_mask=self.output_mask,
        )

        self.val_metric_ranges = get_metric_ranges(
            config,
            data_indices,
            metadata_extractor=metadata_extractor,
        )

        self.loss = get_loss_function(
            config.model_dump(by_alias=True).training.training_loss,
            scalers=self.scalers,
            data_indices=self.data_indices,
        )
        self._scaling_values_log = print_variable_scaling(
            self.loss,
            data_indices,
        )

        self.metrics = torch.nn.ModuleDict(
            {
                metric_name: get_loss_function(val_metric_config, scalers=self.scalers, data_indices=self.data_indices)
                for metric_name, val_metric_config in config.model_dump(
                    by_alias=True,
                ).training.validation_metrics.items()
            },
        )

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True
        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_warmup = config.training.lr.warmup
        self.lr_min = config.training.lr.min
        self.optimizer_settings = config.training.optimizer

        self.model_comm_group = None
        self.reader_groups = None

        reader_group_size = self.config.dataloader.read_group_size
        self.grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        self.grid_indices.setup(graph_data)
        self.grid_dim = -2

        # check sharding support
        self.keep_batch_sharded = self.config.model.keep_batch_sharded
        read_group_supports_sharding = reader_group_size == self.config.hardware.num_gpus_per_model
        assert read_group_supports_sharding or not self.keep_batch_sharded, (
            f"Reader group size {reader_group_size} does not match the number of GPUs per model "
            f"{self.config.hardware.num_gpus_per_model}, but `model.keep_batch_sharded=True` was set. ",
            "Please set `model.keep_batch_sharded=False` or set `dataloader.read_group_size` ="
            "`hardware.num_gpus_per_model`.",
        )

        # set flag if loss and metrics support sharding
        self.loss_supports_sharding = getattr(self.loss, "supports_sharding", False)
        self.metrics_support_sharding = all(
            getattr(metric, "supports_sharding", False) for metric in self.metrics.values()
        )

        if not self.loss_supports_sharding and self.keep_batch_sharded:
            LOGGER.warning(
                "Loss function %s does not support sharding. "
                "This may lead to increased memory usage and slower training.",
                self.loss.name,
            )
        if not self.metrics_support_sharding and self.keep_batch_sharded:
            LOGGER.warning(
                "Validation metrics %s do not support sharding. "
                "This may lead to increased memory usage and slower training.",
                ", ".join(self.metrics.keys()),
            )

        LOGGER.debug("Multistep: %d", self.multi_step)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_size = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.grid_shard_shapes = None
        self.grid_shard_slice = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )

    def on_load_checkpoint(self, checkpoint: torch.nn.Module) -> None:
        self._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index

    def update_scalers(self, callback: AvailableCallbacks) -> None:
        """Update scalers, calling the defined function on them, updating if not None."""
        for name, scaler_builder in self.updating_scalars.items():
            scaler = scaler_builder.update_scaling_values(callback, model=self.model)
            if scaler is None:  # If scalar is None, no update to be applied
                continue

            if name in self.loss.scaler:  # If scalar in loss, update it
                self.loss.update_scaler(scaler=scaler[1], name=name)  # Only update the values

            for metric in self.metrics.values():  # If scalar in metrics, update it
                if name in metric.scaler:
                    metric.update_scaler(scaler=scaler[1], name=name)  # Only update the values

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
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, slice | None]:
        """Prepare tensors for loss computation, handling sharding if necessary.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        validation_mode : bool
            Whether in validation mode

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, slice | None]
            Prepared y_pred, y, and grid_shard_slice
        """
        is_sharded = self.grid_shard_slice is not None

        sharding_supported = (self.loss_supports_sharding or validation_mode) and (
            self.metrics_support_sharding or not validation_mode
        )

        if is_sharded and not sharding_supported:  # gather tensors if loss or metrics do not support sharding
            shard_shapes = apply_shard_shapes(y_pred, self.grid_dim, self.grid_shard_shapes)
            y_pred_full = gather_tensor(torch.clone(y_pred), self.grid_dim, shard_shapes, self.model_comm_group)
            y_full = gather_tensor(torch.clone(y), self.grid_dim, shard_shapes, self.model_comm_group)
            grid_shard_slice = None
        else:
            y_pred_full, y_full = y_pred, y
            grid_shard_slice = self.grid_shard_slice

        return y_pred_full, y_full, grid_shard_slice

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
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
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        return self.loss(
            y_pred,
            y,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _compute_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int = 0,
        grid_shard_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute validation metrics.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        rollout_step : int
            Current rollout step
        grid_shard_slice : slice | None
            Grid shard slice for distributed training

        Returns
        -------
        dict[str, torch.Tensor]
            Computed metrics
        """
        return self.calculate_val_metrics(
            y_pred,
            y,
            rollout_step,
            grid_shard_slice=grid_shard_slice,
        )

    def compute_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        validation_mode: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        rollout_step : int
            Current rollout step
        validation_mode : bool
            Whether to compute validation metrics
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor]]
            Loss and metrics dictionary (if validation_mode)
        """
        # Prepare tensors for loss/metrics computation
        y_pred_full, y_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            validation_mode,
        )

        loss = self._compute_loss(y_pred=y_pred_full, y=y_full, grid_shard_slice=grid_shard_slice, **kwargs)

        # Compute metrics if in validation mode
        metrics_next = {}
        if validation_mode:
            metrics_next = self._compute_metrics(y_pred_full, y_full, rollout_step, grid_shard_slice)

        return loss, metrics_next

    def on_after_batch_transfer(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Also normalize the batch in-place if needed.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to transfer

        Returns
        -------
        torch.Tensor
            Batch after transfer
        """
        # Gathering/sharding of batch
        batch = self._setup_batch_sharding(batch)

        # Batch normalization
        batch = self._normalize_batch(batch)

        # Prepare scalers, e.g. init delayed scalers and update scalers
        self._prepare_loss_scalers()

        return batch

    def _setup_batch_sharding(self, batch: torch.Tensor) -> torch.Tensor:
        """Setup batch sharding before every step.

        If the batch is sharded, it will be setup with the grid shard shapes and slice.
        Otherwise, the batch will be allgathered.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to setup

        Returns
        -------
        torch.Tensor
            Batch after setup
        """
        if self.keep_batch_sharded and self.model_comm_group_size > 1:
            self.grid_shard_shapes = self.grid_indices.shard_shapes
            self.grid_shard_slice = self.grid_indices.get_shard_slice(self.reader_group_rank)
        else:
            batch = self.allgather_batch(batch)
            self.grid_shard_shapes, self.grid_shard_slice = None, None
        return batch

    def _normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalize batch for training and validation before every step.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to prepare

        Returns
        -------
        torch.Tensor
            Normalized batch
        """
        return self.model.pre_processors(batch)

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
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
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
        grid_shard_shapes = self.grid_indices.shard_shapes
        grid_size = self.grid_indices.grid_size

        if grid_size == batch.shape[self.grid_dim] or self.reader_group_size == 1:
            return batch  # already have the full grid

        shard_shapes = apply_shard_shapes(batch, self.grid_dim, grid_shard_shapes)
        tensor_list = [torch.empty(shard_shape, device=batch.device, dtype=batch.dtype) for shard_shape in shard_shapes]

        torch.distributed.all_gather(
            tensor_list,
            batch,
            group=self.reader_groups[self.reader_group_id],
        )

        return torch.cat(tensor_list, dim=self.grid_dim)

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int = 0,
        grid_shard_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                    scaler_indices=[..., indices],
                    grid_shard_slice=grid_shard_slice,
                    group=self.model_comm_group,
                )

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx

        train_loss, _, _ = self._step(batch)
        self.log(
            "train_" + self.loss.name + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Any
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        """
        del batch_idx

        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, validation_mode=True)

        self.log(
            "val_" + self.loss.name + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
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
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configure the optimizers and learning rate scheduler.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            List of optimizers and list of dictionaries containing the
            learning rate scheduler
        """
        if self.optimizer_settings.zero:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                lr=self.lr,
                optimizer_class=torch.optim.AdamW,
                **self.optimizer_settings.kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.lr,
                **self.optimizer_settings.kwargs,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.lr_warmup,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def setup(self, stage: str) -> None:
        """Lightning hook that is called after model is initialized but before training starts."""
        # The conditions should be separate, but are combined due to pre-commit hook
        if stage == "fit" and self.trainer.is_global_zero and self.logger is not None:
            # Log hyperparameters on rank 0
            hyper_params = OmegaConf.to_container(convert_to_omegaconf(self.config), resolve=True)
            hyper_params.update({"variable_loss_scaling": self._scaling_values_log})
            # Log hyperparameters
            self.logger.log_hyperparams(hyper_params)
