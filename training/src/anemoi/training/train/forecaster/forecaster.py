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
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

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

        # Instantiate all scalers with the training configuration
        self.scalers, self.delayed_scaler_builders = create_scalers(
            config.model_dump(by_alias=True).training.scalers,
            group_config=config.model_dump(by_alias=True).training.variable_groups,
            data_indices=data_indices,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            metadata_variables=metadata["dataset"].get("variables_metadata"),
            output_mask=self.output_mask,
        )

        self.val_metric_ranges = get_metric_ranges(
            config,
            data_indices,
            metadata["dataset"].get("variables_metadata"),
        )

        self.loss = get_loss_function(
            config.model_dump(by_alias=True).training.training_loss,
            scalers=self.scalers,
            data_indices=self.data_indices,
        )
        print_variable_scaling(self.loss, data_indices)

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
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

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
        model_supports_sharding = getattr(self.model.model, "supports_sharded_input", False)
        assert model_supports_sharding or not self.keep_batch_sharded, (
            f"Model {self.model.model} does not support sharded inputs, but `model.keep_batch_sharded=True` was set. ",
            "Please set `model.keep_batch_sharded=False` or use a model that supports sharded inputs.",
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

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
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

    def on_load_checkpoint(self, checkpoint: torch.nn.module) -> None:
        self._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index

    def define_delayed_scalers(self) -> None:
        """Update delayed scalers such as the loss weights mask for imputed variables."""
        for name, scaler_builder in self.delayed_scaler_builders.items():
            self.scalers[name] = scaler_builder.get_delayed_scaling(model=self.model)
            self.loss.update_scaler(scaler=self.scalers[name][1], name=name)

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

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ...,
            self.data_indices.model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(
            x[:, -1],
            batch[:, self.multi_step + rollout_step],
            self.data_indices,
        )

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.data.input.forcing,
        ]
        return x

    def compute_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> torch.Tensor:
        is_sharded = self.grid_shard_slice is not None

        sharding_supported = (self.loss_supports_sharding or not training_mode) and (
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

        loss = (
            self.loss(
                y_pred_full,
                y_full,
                grid_shard_slice=grid_shard_slice,
                group=self.model_comm_group,
            )
            if training_mode
            else None
        )

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(
                y_pred_full,
                y_full,
                rollout_step,
                grid_shard_slice=grid_shard_slice,
            )

        return loss, metrics_next

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        batch = self.model.pre_processors(batch)  # normalized in-place

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                rollout_step,
                training_mode,
                validation_mode,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred

    def on_after_batch_transfer(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to transfer
        dataloader_idx : int
            Dataloader index

        Returns
        -------
        torch.Tensor
            Batch after transfer
        """
        if self.keep_batch_sharded and self.model_comm_group_size > 1:
            self.grid_shard_shapes = self.grid_indices.shard_shapes
            self.grid_shard_slice = self.grid_indices.get_shard_indices(self.reader_group_rank)
        else:
            batch = self.allgather_batch(batch)
            self.grid_shard_shapes, self.grid_shard_slice = None, None

        return batch

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        del batch_idx

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

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
        rollout_step: int,
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
        train_loss, _, _ = self._step(batch, batch_idx)
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
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        """
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

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
