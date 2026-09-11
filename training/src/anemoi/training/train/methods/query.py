# (C) Copyright 2026 Anemoi contributors.

"""Training method for one query-selected scalar field per example."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from timm.scheduler.scheduler import Scheduler as TimmScheduler

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.models.query_forecaster import QueryForecaster
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.losses import CRPS

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
        self.config = config
        self.task = task
        self.data_indices = data_indices
        self.model = QueryForecaster(config, graph_data, metadata, supporting_arrays)
        model_group_size = config.system.hardware.num_gpus_per_model
        ensemble_group_size = config.system.hardware.num_gpus_per_ensemble
        if ensemble_group_size % model_group_size:
            msg = (
                "system.hardware.num_gpus_per_ensemble must be divisible by "
                "system.hardware.num_gpus_per_model."
            )
            raise ValueError(msg)
        self.effective_lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.optimization.lr
            / ensemble_group_size
        )
        self.model_comm_group = None
        self.reader_groups = None
        self.dataset_names = list(data_indices)
        self.shard_sizes = {
            name: get_balanced_partition_sizes(
                graph_data[name].num_nodes,
                model_group_size,
            )
            for name in self.dataset_names
        }
        loss_configs = get_multiple_datasets_config(config.training.training_loss)
        self.loss = torch.nn.ModuleDict(
            {
                name: instantiate(loss_configs[name])
                for name in self.dataset_names
                if name in loss_configs and loss_configs[name] is not None
            },
        )
        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = (
            self.nens_per_device * ensemble_group_size // model_group_size
        )
        self.ens_comm_group = None
        self.ens_comm_group_id = 0
        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_size = 1
        self.ens_comm_subgroup = None
        self.ens_comm_subgroup_id = 0
        self.ens_comm_subgroup_rank = 0
        self.ens_comm_subgroup_num_groups = 1
        self.ens_comm_subgroup_size = 1
        self._query_diagnostics_enabled = False
        self._query_diagnostics_capture_prediction = False
        self._query_diagnostics_step: dict[str, Any] | None = None
        self.save_hyperparameters(ignore=["graph_data"])

    @property
    def plot_adapter(self) -> None:
        return None

    def forward(self, batch: QueryBatch) -> torch.Tensor:
        query = {
            "metadata": batch.query_metadata,
            "variable_id": batch.query_variable_id,
            "provenance_id": batch.query_provenance_id,
            "unit_id": batch.query_unit_id,
            "grid_id": batch.query_grid_id,
            "grid": batch.target_dataset,
        }
        return self.model(
            batch.inputs,
            query,
            # A full native-grid query can use the cached decoder connectivity.
            # Crops retain their explicit coordinates and dynamic decoder.
            output_coordinates=(
                batch.output_coordinates
                if batch.query.get("bbox") is not None
                else None
            ),
            model_comm_group=self.model_comm_group,
            grid_shard_sizes=self.shard_sizes,
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
        if batch.target_dataset not in self.loss:
            msg = f"No query loss is configured for target dataset {batch.target_dataset!r}."
            raise ValueError(msg)
        loss_module = self.loss[batch.target_dataset]
        if isinstance(loss_module, CRPS):
            if self.nens_per_device != 1:
                msg = "Sharded query ensembles require training.ensemble_size_per_device=1."
                raise ValueError(msg)
            ensemble_prediction = gather_tensor(
                prediction[:, None],
                dim=1,
                sizes=[1] * self.ens_comm_subgroup_size,
                mgroup=self.ens_comm_subgroup,
            )
            if ensemble_prediction.shape[1] < 2:
                msg = (
                    "CRPS requires at least two query ensemble members; use "
                    "DDPEnsGroupStrategy with num_gpus_per_ensemble at least twice "
                    "num_gpus_per_model."
                )
                raise ValueError(msg)
            # CRPS._kernel_crps is the canonical point-wise Anemoi implementation.
            # Keep the point dimension here so query validity and cosine-latitude
            # weights can be applied before the spatial reduction.
            with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
                pointwise = loss_module._kernel_crps(
                    ensemble_prediction.float().permute(0, 2, 1)[:, None, None],
                    batch.target.float()[:, None, None, :, None],
                ).reshape_as(prediction)
            normalized_score = (pointwise * spatial_weights).sum() / valid
            score_name = loss_module.name
            physical_score = normalized_score.detach() * float(
                (batch.diagnostic_context or {}).get("target", {}).get("stdev", 1.0),
            )
        else:
            pointwise = (prediction - batch.target) ** 2
            normalized_score = (pointwise * spatial_weights).sum() / valid
            score_name = "mse"
            physical_score = normalized_score.detach().sqrt() * float(
                (batch.diagnostic_context or {}).get("target", {}).get("stdev", 1.0),
            )
        loss = normalized_score * batch.loss_weight
        if self._query_diagnostics_enabled:
            self._query_diagnostics_step = {
                "stage": stage,
                "score_name": score_name,
                "normalized_score": normalized_score.detach(),
                "physical_score": physical_score,
                "valid_target_count": valid.detach(),
                "loss_weight": batch.loss_weight.detach(),
                "prediction": prediction.detach() if self._query_diagnostics_capture_prediction else None,
            }
            self._query_diagnostics_capture_prediction = False
        self.log(
            f"{stage}_query_{score_name}",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def enable_query_diagnostics(self) -> None:
        """Enable lightweight per-example accounting for an opt-in callback."""
        self._query_diagnostics_enabled = True

    def request_query_diagnostic_prediction(self) -> None:
        """Retain the next already-computed prediction, detached from autograd."""
        self._query_diagnostics_capture_prediction = True

    def pop_query_diagnostics_step(self) -> dict[str, Any] | None:
        """Return and clear the most recent detached diagnostic record."""
        value = self._query_diagnostics_step
        self._query_diagnostics_step = None
        return value

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

    def set_ens_comm_group(
        self,
        ens_comm_group: Any,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup: Any,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

    def on_train_epoch_end(self) -> None:
        self.task.on_train_epoch_end(self.current_epoch)
        self.trainer.datamodule.set_epoch(self.current_epoch + 1)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["task_state"] = self.task.training_runtime_state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if not self.config.training.load_weights_only:
            self.task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))
