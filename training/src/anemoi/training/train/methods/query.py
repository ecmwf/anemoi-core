# (C) Copyright 2026 Anemoi contributors.

"""Training method for one query-selected scalar field per example."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from timm.scheduler.scheduler import Scheduler as TimmScheduler

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.models.query_forecaster import QueryForecaster
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.losses import CRPS
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import LRSchedulerTypeUnion

    from anemoi.training.query.batch import QueryBatch


def precipitation_log_spectral_distance(
    predicted_mm: torch.Tensor,
    target_mm: torch.Tensor,
    valid: torch.Tensor,
    shape: tuple[int, int],
    spacing_km: float,
) -> torch.Tensor:
    """Compare 10--200 km precipitation power on a regular native grid."""
    predicted_mm = F.softplus(predicted_mm / 0.05) * 0.05
    target_mm = target_mm.clamp_min(0.0)
    predicted_field = torch.log1p(predicted_mm / 0.1).reshape(-1, *shape)
    target_field = torch.log1p(target_mm / 0.1).reshape(-1, *shape)
    valid = valid.reshape(-1, *shape)
    valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    target_fill = (target_field * valid).sum(dim=(-2, -1), keepdim=True) / valid_count
    predicted_fill = (predicted_field * valid).sum(dim=(-2, -1), keepdim=True) / valid_count
    target_field = torch.where(valid, target_field, target_fill)
    predicted_field = torch.where(valid, predicted_field, predicted_fill)
    target_field = target_field - target_field.mean(dim=(-2, -1), keepdim=True)
    predicted_field = predicted_field - predicted_field.mean(dim=(-2, -1), keepdim=True)
    window = torch.outer(
        torch.hann_window(shape[0], device=predicted_mm.device),
        torch.hann_window(shape[1], device=predicted_mm.device),
    )
    target_power = torch.fft.rfft2(target_field * window).abs().square()
    predicted_power = torch.fft.rfft2(predicted_field * window).abs().square()
    fy = torch.fft.fftfreq(shape[0], d=spacing_km, device=predicted_mm.device)[:, None]
    fx = torch.fft.rfftfreq(shape[1], d=spacing_km, device=predicted_mm.device)[None, :]
    frequency = torch.sqrt(fy.square() + fx.square())
    selected = (frequency >= 1.0 / 200.0) & (frequency <= 1.0 / 10.0)
    if not torch.any(selected):
        raise ValueError(f"Grid shape={shape} and spacing={spacing_km} km contain no 10--200 km modes.")
    floor = target_power.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0) * 1.0e-8
    db_scale = 10.0 / torch.log(torch.tensor(10.0, device=predicted_mm.device))
    log_difference = db_scale * (
        torch.log(predicted_power + floor) - torch.log(target_power + floor)
    )
    return torch.sqrt(log_difference[:, selected].square().mean() + 1.0e-8)


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
        loss_factory_only_keys = {
            "scalers",
            "predicted_variables",
            "target_variables",
            "check_variables_compatibility",
        }
        self.loss = torch.nn.ModuleDict(
            {
                # Query losses reduce each sampled field directly.  Older
                # Anemoi configs and the validated schema carry loss-factory
                # metadata which is consumed by get_loss_function(), not by
                # the loss constructor. Query training does its own scalar
                # target selection, so none of that metadata applies here.
                name: instantiate(
                    {
                        key: value
                        for key, value in loss_configs[name].items()
                        if key not in loss_factory_only_keys
                    },
                )
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
        output_coordinates = batch.output_coordinates if batch.query.get("bbox") is not None else None
        if self.model.bundle_outputs_by_provenance:
            encoded = self.model.encode_query_inputs(
                batch.inputs,
                model_comm_group=self.model_comm_group,
                grid_shard_sizes=self.shard_sizes,
            )
            correction = self.model.decode_query_bundle(
                encoded,
                query,
                output_coordinates=output_coordinates,
                model_comm_group=self.model_comm_group,
            )
        else:
            correction = self.model(
                batch.inputs,
                query,
                output_coordinates=output_coordinates,
                model_comm_group=self.model_comm_group,
                grid_shard_sizes=self.shard_sizes,
            )
        return self._add_residual_baseline(correction, batch)

    def _add_residual_baseline(self, correction: torch.Tensor, batch: QueryBatch) -> torch.Tensor:
        """Add a nearest-grid exact-semantic source field in target-normalized units."""

        source_name = batch.residual_baseline_source
        columns = batch.residual_baseline_input_column
        indices = batch.residual_baseline_indices
        multiplier = batch.residual_baseline_multiplier
        offset = batch.residual_baseline_offset
        if source_name is None:
            return correction
        if columns is None or indices is None or multiplier is None or offset is None:
            raise ValueError("Residual baseline metadata is incomplete.")
        source = batch.inputs[source_name]
        if isinstance(columns, int):
            columns = correction.new_tensor([columns], dtype=torch.long)
        matched = columns >= 0
        safe_columns = columns.clamp_min(0)
        source_values = source.values[:, :, safe_columns]
        source_mask = source.mask[:, :, safe_columns]
        source_sizes = self.shard_sizes[source_name]
        if source_values.shape[1] != sum(source_sizes):
            source_values = gather_tensor(
                source_values,
                dim=1,
                sizes=source_sizes,
                mgroup=self.model_comm_group,
            )
            source_mask = gather_tensor(
                source_mask.to(torch.uint8),
                dim=1,
                sizes=source_sizes,
                mgroup=self.model_comm_group,
            ).bool()
        baseline = source_values.index_select(1, indices)[0].T
        baseline = baseline * multiplier[:, None] + offset[:, None]
        baseline = torch.where(matched[:, None], baseline, torch.zeros_like(baseline))
        baseline_valid = source_mask.index_select(1, indices)[0].T
        if not torch.all(baseline_valid[matched]):
            raise ValueError("Exact residual baseline contains invalid source values.")
        if correction.shape[-1] != baseline.shape[-1]:
            target_sizes = self.shard_sizes[batch.target_dataset]
            if correction.shape[-1] != sum(target_sizes):
                raise ValueError(
                    f"Residual correction has {correction.shape[-1]} points but its local baseline has "
                    f"{baseline.shape[-1]}."
                )
            baseline = gather_tensor(
                baseline,
                dim=-1,
                sizes=target_sizes,
                mgroup=self.model_comm_group,
            )
        return correction + baseline

    def _step(self, batch: QueryBatch, stage: str) -> torch.Tensor:
        prediction = self(batch)
        mask = batch.target_mask
        if mask is None or batch.target is None or batch.loss_weight is None:
            msg = "Training batches require target values, validity masks, and loss weights."
            raise ValueError(msg)
        scored_prediction = prediction
        if prediction.shape[-1] != batch.target.shape[-1]:
            if batch.query.get("bbox") is not None:
                msg = (
                    f"Cropped prediction contains {prediction.shape[-1]} points but its target contains "
                    f"{batch.target.shape[-1]}."
                )
                raise ValueError(msg)
            shard_sizes = self.shard_sizes[batch.target_dataset]
            shard_start, shard_end = get_partition_range(
                shard_sizes,
                self.model_comm_group_rank,
            )
            if batch.target.shape[-1] != shard_end - shard_start:
                msg = (
                    f"Target contains {batch.target.shape[-1]} points, expected the rank-"
                    f"{self.model_comm_group_rank} shard [{shard_start}, {shard_end}) of "
                    f"{batch.target_dataset}."
                )
                raise ValueError(msg)
            scored_prediction = prediction[..., shard_start:shard_end]
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
            if prediction.shape[0] != 1:
                raise ValueError("Bundled query training currently supports deterministic losses only.")
            if self.nens_per_device != 1:
                msg = "Sharded query ensembles require training.ensemble_size_per_device=1."
                raise ValueError(msg)
            ensemble_prediction = gather_tensor(
                scored_prediction[:, None],
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
                ).reshape_as(scored_prediction)
            normalized_score = (pointwise * spatial_weights).sum() / valid
            score_name = loss_module.name
            physical_score = normalized_score.detach() * float(
                (batch.diagnostic_context or {}).get("target", {}).get("stdev", 1.0),
            )
        else:
            pointwise = (scored_prediction - batch.target) ** 2
            field_valid = spatial_weights.sum(dim=-1).clamp_min(1)
            field_scores = (pointwise * spatial_weights).sum(dim=-1) / field_valid
            field_weights = batch.loss_weight.reshape(-1)
            normalized_score = (field_scores * field_weights).sum() / field_weights.sum().clamp_min(1.0e-12)
            score_name = "mse"
            physical_score = field_scores[0].detach().sqrt() * float(
                (batch.diagnostic_context or {}).get("target", {}).get("stdev", 1.0),
            )
        precipitation_lsd = self._precipitation_lsd(prediction, batch)
        loss = normalized_score + float(self.task.precipitation_lsd_weight) * precipitation_lsd
        if not torch.isfinite(loss):
            msg = (
                f"Non-finite {stage} query loss for dataset={batch.target_dataset}, "
                f"variable={batch.query.get('variable')}, normalized_score={normalized_score.detach()}, "
                f"precipitation_lsd={precipitation_lsd.detach()}."
            )
            raise FloatingPointError(msg)
        # Every distributed rank must enter the same synchronized logging
        # collectives. Different model groups intentionally draw different
        # queries, so making this call conditional on ``variable == "tp"``
        # deadlocks whenever only a subset of groups draws precipitation.
        # Non-precipitation groups contribute zero and an explicit fraction is
        # logged alongside it; their ratio gives the conditional tp LSD.
        bundle_fields = (batch.diagnostic_context or {}).get("bundle_fields", [])
        precipitation_query = prediction.new_tensor(
            float(any(item.get("variable") == "tp" for item in bundle_fields) or batch.query.get("variable") == "tp"),
        )
        self.log(
            f"{stage}_query_precipitation_lsd_contribution",
            precipitation_lsd,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"{stage}_query_precipitation_fraction",
            precipitation_query,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=1,
        )
        if self._query_diagnostics_enabled:
            self._query_diagnostics_step = {
                "stage": stage,
                "score_name": score_name,
                "normalized_score": normalized_score.detach(),
                "physical_score": physical_score,
                "valid_target_count": valid.detach(),
                "loss_weight": batch.loss_weight.detach().mean(),
                "prediction": prediction.detach() if self._query_diagnostics_capture_prediction else None,
            }
            self._query_diagnostics_capture_prediction = False
        # Keep the proper data score and the optimized objective distinct.
        # In particular, query_mse must remain the raw normalized MSE; the
        # precipitation LSD penalty belongs only to query_loss.
        self.log(
            f"{stage}_query_{score_name}",
            normalized_score,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"{stage}_query_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def _precipitation_lsd(self, prediction: torch.Tensor, batch: QueryBatch) -> torch.Tensor:
        """Return a texture-sensitive precipitation loss on the native 2-D grid."""
        zero = prediction.new_zeros(())
        bundle_fields = (batch.diagnostic_context or {}).get("bundle_fields", [])
        precipitation_indices = [
            index for index, item in enumerate(bundle_fields) if item.get("variable") == "tp"
        ]
        if not precipitation_indices and batch.query.get("variable") == "tp":
            precipitation_indices = [0]
        if not precipitation_indices or not self.task.precipitation_lsd_weight:
            return zero
        if len(precipitation_indices) != 1:
            raise ValueError("A domain query bundle must contain exactly one precipitation field.")
        precipitation_index = precipitation_indices[0]
        prediction = prediction[precipitation_index : precipitation_index + 1]
        context = (batch.diagnostic_context or {}).get("target", {})
        shape = tuple(int(value) for value in context.get("field_shape", ()))
        if len(shape) != 2:
            msg = f"Precipitation LSD requires a 2-D field_shape, got {shape!r}."
            raise ValueError(msg)
        point_count = shape[0] * shape[1]
        full_prediction = prediction
        if prediction.shape[-1] != point_count:
            shard_sizes = self.shard_sizes[batch.target_dataset]
            if prediction.shape[-1] not in shard_sizes:
                msg = (
                    f"Prediction contains {prediction.shape[-1]} points, but {batch.target_dataset} "
                    f"has shape {shape} and shard sizes {shard_sizes}."
                )
                raise ValueError(msg)
            full_prediction = gather_tensor(
                prediction,
                dim=-1,
                sizes=shard_sizes,
                mgroup=self.model_comm_group,
            )
        if batch.target is None or batch.target_mask is None:
            raise ValueError("Precipitation LSD requires a target and validity mask.")
        full_target = batch.target[precipitation_index : precipitation_index + 1]
        full_target_mask = batch.target_mask[precipitation_index : precipitation_index + 1]
        if full_target.shape[-1] != point_count:
            shard_sizes = self.shard_sizes[batch.target_dataset]
            if full_target.shape[-1] not in shard_sizes:
                msg = (
                    f"Target contains {full_target.shape[-1]} points, but {batch.target_dataset} "
                    f"has shape {shape} and shard sizes {shard_sizes}."
                )
                raise ValueError(msg)
            full_target = gather_tensor(
                full_target,
                dim=-1,
                sizes=shard_sizes,
                mgroup=self.model_comm_group,
            )
            full_target_mask = gather_tensor(
                full_target_mask.to(torch.uint8),
                dim=-1,
                sizes=shard_sizes,
                mgroup=self.model_comm_group,
            ).bool()

        scale_to_mm = self.task.precipitation_unit_scale_to_mm.get(batch.target_dataset)
        if scale_to_mm is None or scale_to_mm <= 0:
            msg = f"Missing positive precipitation_unit_scale_to_mm for {batch.target_dataset}."
            raise ValueError(msg)
        precipitation_context = bundle_fields[precipitation_index] if bundle_fields else context
        mean = float(precipitation_context["mean"])
        stdev = float(precipitation_context["stdev"])
        with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
            predicted_mm = (full_prediction.float() * stdev + mean) * scale_to_mm
            target_mm = (full_target.float() * stdev + mean) * scale_to_mm
            return precipitation_log_spectral_distance(
                predicted_mm,
                target_mm,
                full_target_mask,
                shape,
                float(context["resolution_km"]),
            )

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
        checkpoint_data_indices = checkpoint.get("hyper_parameters", {}).get("data_indices")
        if isinstance(checkpoint_data_indices, dict):
            model_name_to_index = {
                dataset_name: indices.name_to_index
                for dataset_name, indices in checkpoint_data_indices.items()
                if indices is not None
            }
            self._ckpt_model_name_to_index = model_name_to_index or None
            self._ckpt_variables_metadata = (
                extract_variables_metadata_from_checkpoint(checkpoint, model_name_to_index)
                if model_name_to_index
                else None
            )
        if not self.config.training.load_weights_only:
            self.task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))
