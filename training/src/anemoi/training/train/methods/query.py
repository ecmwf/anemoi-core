# (C) Copyright 2026 Anemoi contributors.

"""Training method for one query-selected scalar field per example."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from timm.scheduler.scheduler import Scheduler as TimmScheduler

from anemoi.models.models.query_forecaster import QueryForecaster

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import LRSchedulerTypeUnion

    from anemoi.training.query.batch import QueryBatch


class QueryTraining(pl.LightningModule):
    """Train normalized direct forecasts with equal per-query spatial reduction."""

    def __init__(
        self,
        *,
        config: Any,
        task: Any,
        graph_data: Any,
        metadata: dict,
        supporting_arrays: dict,
        data_indices: dict,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        if config.system.hardware.num_gpus_per_model != 1:
            msg = "The initial query model requires system.hardware.num_gpus_per_model=1."
            raise ValueError(msg)
        self.config = config
        self.task = task
        self.data_indices = data_indices
        self.model = QueryForecaster(config, graph_data, metadata, supporting_arrays)
        self.effective_lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.optimization.lr
        )
        self.model_comm_group = None
        self.reader_groups = None
        self.dataset_names = list(data_indices)
        self.shard_sizes = {name: [graph_data[name].num_nodes] for name in self.dataset_names}
        self.save_hyperparameters(ignore=["graph_data"])

    @property
    def plot_adapter(self) -> None:
        return None

    def forward(self, batch: QueryBatch) -> torch.Tensor:
        query = {
            "metadata": batch.query_metadata,
            "variable_id": batch.query_variable_id,
            "provenance_id": batch.query_provenance_id,
            "grid": batch.target_dataset,
        }
        return self.model(
            batch.inputs,
            query,
            output_coordinates=batch.output_coordinates,
            model_comm_group=self.model_comm_group,
        )

    def _step(self, batch: QueryBatch, stage: str) -> torch.Tensor:
        prediction = self(batch)
        mask = batch.target_mask
        if mask is None or batch.target is None or batch.loss_weight is None:
            msg = "Training batches require target values, validity masks, and loss weights."
            raise ValueError(msg)
        spatial_weights = mask
        if self.task.spatial_weighting == "cosine_latitude":
            if batch.output_coordinates is None:
                msg = "cosine_latitude weighting requires target coordinates."
                raise ValueError(msg)
            spatial_weights = mask * torch.cos(batch.output_coordinates[:, 0])[None]
        valid = spatial_weights.sum()
        if not valid:
            msg = f"Query {batch.query} has no valid target points in its requested region."
            raise ValueError(msg)
        loss = (((prediction - batch.target) ** 2) * spatial_weights).sum() / valid
        loss = loss * batch.loss_weight
        self.log(
            f"{stage}_query_mse",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def training_step(self, batch: QueryBatch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: QueryBatch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def transfer_batch_to_device(
        self,
        batch: QueryBatch,
        device: torch.device,
        _dataloader_idx: int = 0,
    ) -> QueryBatch:
        return batch.to(device, non_blocking=True)

    def configure_optimizers(self) -> Any:
        optimization = self.config.training.optimization
        optimizer = instantiate(
            optimization.optimizer,
            params=self.parameters(),
            lr=self.effective_lr,
        )
        if not optimization.get("lr_scheduler"):
            return optimizer
        scheduler = instantiate(optimization.lr_scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, **optimization.pl_lr_scheduler}]

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None = None) -> None:
        if isinstance(scheduler, TimmScheduler):
            scheduler_config = next(
                value for value in self.trainer.lr_scheduler_configs if value.scheduler is scheduler
            )
            if scheduler_config.interval == "step":
                scheduler.step_update(self.trainer.global_step, metric)
            else:
                scheduler.step(self.current_epoch + 1, metric)
            return
        super().lr_scheduler_step(scheduler, metric)

    def set_model_comm_group(
        self,
        model_comm_group: Any,
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
        reader_groups: list[Any],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def on_train_epoch_end(self) -> None:
        self.task.on_train_epoch_end(self.current_epoch)
        self.trainer.datamodule.set_epoch(self.current_epoch + 1)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["task_state"] = self.task.training_runtime_state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if not self.config.training.load_weights_only:
            self.task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))
