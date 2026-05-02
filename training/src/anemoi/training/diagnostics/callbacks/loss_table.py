# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Export validation losses over all validation samples."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)


class ExportValidationLossTable(pl.Callback):
    """Write configured validation losses by batch/sample/lead/variable.

    This callback is intentionally computed inside the validation loop. That lets
    it use the exact loss object and scalers configured for training, including
    dynamic target-value/tendency scalers.
    """

    def __init__(
        self,
        config: Any,
        output_dir: str | None = None,
        every_n_batches: int = 1,
        variables: list[str] | None = None,
        write_sample_variable_rows: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(config.system.output.plots) / "loss_tables"
        self.every_n_batches = int(every_n_batches)
        self.variables = variables
        self.write_sample_variable_rows = bool(write_sample_variable_rows)
        self._detail_path: Path | None = None
        self._summary_path: Path | None = None
        self._detail_header_written = False
        self._summary_header_written = False
        self._epoch_weighted_loss = 0.0
        self._epoch_weight = 0

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._detail_path = self.output_dir / f"validation_loss_detail_epoch{trainer.current_epoch:03d}.csv"
            self._summary_path = self.output_dir / "validation_loss_epoch_summary.csv"
            if self._detail_path.exists():
                self._detail_path.unlink()
            self._detail_header_written = False
            self._summary_header_written = self._summary_path.exists() and self._summary_path.stat().st_size > 0
        self._epoch_weighted_loss = 0.0
        self._epoch_weight = 0

    @staticmethod
    def _dataset_tensor(batch: dict[str, torch.Tensor], dataset_name: str) -> torch.Tensor:
        if dataset_name in batch:
            return batch[dataset_name]
        if len(batch) == 1:
            return next(iter(batch.values()))
        raise KeyError(f"Dataset {dataset_name!r} not found in validation batch.")

    @staticmethod
    def _as_float(value: torch.Tensor | float) -> float:
        if torch.is_tensor(value):
            return float(value.detach().sum().cpu().item())
        return float(value)

    def _write_detail_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows or self._detail_path is None:
            return
        with self._detail_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not self._detail_header_written:
                writer.writeheader()
                self._detail_header_written = True
            writer.writerows(rows)

    def _write_summary_row(self, row: dict[str, Any]) -> None:
        if self._summary_path is None:
            return
        with self._summary_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._summary_header_written:
                writer.writeheader()
                self._summary_header_written = True
            writer.writerow(row)

    @staticmethod
    def _scaler_names(loss_fn: torch.nn.Module) -> list[str]:
        return list(getattr(loss_fn.scaler, "tensors", {}).keys()) if hasattr(loss_fn, "scaler") else []

    def _loss_values(
        self,
        loss_fn: torch.nn.Module,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_idx: int | None = None,
        lead_idx: int | None = None,
        var_idx: int | None = None,
    ) -> float:
        indexers: list[Any] = [slice(None)] * y_pred.ndim
        if sample_idx is not None:
            indexers[0] = slice(sample_idx, sample_idx + 1)
        if lead_idx is not None:
            indexers[1] = slice(lead_idx, lead_idx + 1)
        if var_idx is not None:
            indexers[-1] = [var_idx]
        scaler_indices = tuple(indexers)
        scaled_loss = loss_fn(y_pred, y_true, scaler_indices=scaler_indices)
        raw_loss = loss_fn(
            y_pred,
            y_true,
            scaler_indices=scaler_indices,
            without_scalers=self._scaler_names(loss_fn),
        )
        return self._as_float(scaled_loss), self._as_float(raw_loss)

    @staticmethod
    def _row_loss_columns(
        scaled_loss: float,
        raw_loss: float | str = "",
        loss_contribution_to_total: float | None = None,
    ) -> dict[str, float | str]:
        contribution = scaled_loss if loss_contribution_to_total is None else loss_contribution_to_total
        return {
            "loss": scaled_loss,
            "scaled_loss": scaled_loss,
            "raw_unscaled_loss": raw_loss,
            "loss_contribution_to_total": contribution,
        }

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: tuple[Any, ...] | list[Any] | None,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not trainer.is_global_zero:
            return
        if outputs is None or len(outputs) < 2:
            return
        if batch_idx % self.every_n_batches != 0:
            return

        val_step_loss = self._as_float(outputs[0])
        batch_size = int(next(iter(batch.values())).shape[0])
        self._epoch_weighted_loss += val_step_loss * batch_size
        self._epoch_weight += batch_size

        y_preds_by_rollout = outputs[1]
        rows: list[dict[str, Any]] = [
            {
                "epoch": trainer.current_epoch,
                "batch_idx": batch_idx,
                "sample_in_batch": "all",
                "sample_global_index": "all",
                "dataset": "all",
                "rollout_step": "all",
                "lead_index": "all",
                "variable": "all",
                "scope": "lightning_val_step_total",
                **self._row_loss_columns(val_step_loss),
                "note": "This is the exact validation_step loss logged by Lightning for this batch.",
            },
        ]

        with torch.no_grad():
            for dataset_name, loss_fn in pl_module.loss.items():
                data_indices = pl_module.data_indices[dataset_name]
                dataset_batch = self._dataset_tensor(batch, dataset_name)
                loss_fn = loss_fn.to(device=pl_module.device)

                name_to_index = data_indices.model.output.name_to_index
                if self.variables:
                    variable_items = [(name, name_to_index[name]) for name in self.variables if name in name_to_index]
                else:
                    variable_items = sorted(name_to_index.items(), key=lambda item: item[1])
                n_output_variables = max(1, len(name_to_index))

                for rollout_step, rollout_pred in enumerate(y_preds_by_rollout):
                    y_pred = rollout_pred[dataset_name] if isinstance(rollout_pred, dict) else rollout_pred
                    y_pred = y_pred.detach()
                    start = pl_module.n_step_input + rollout_step * pl_module.n_step_output
                    y_time = dataset_batch.narrow(1, start, pl_module.n_step_output)
                    output_indices = data_indices.data.output.full.to(device=dataset_batch.device)
                    y_true = y_time.index_select(-1, output_indices).detach()
                    scaled_loss, raw_loss = self._loss_values(loss_fn, y_pred, y_true)

                    rows.append(
                        {
                            "epoch": trainer.current_epoch,
                            "batch_idx": batch_idx,
                            "sample_in_batch": "all",
                            "sample_global_index": "all",
                            "dataset": dataset_name,
                            "rollout_step": rollout_step,
                            "lead_index": "all",
                            "variable": "all",
                            "scope": "configured_loss_rollout_all_samples_all_variables",
                            **self._row_loss_columns(scaled_loss, raw_loss),
                            "note": "Loss for this rollout before the training code averages over rollout count.",
                        },
                    )

                    if self.write_sample_variable_rows:
                        for variable_name, variable_index in variable_items:
                            scaled_loss, raw_loss = self._loss_values(loss_fn, y_pred, y_true, var_idx=variable_index)
                            rows.append(
                                {
                                    "epoch": trainer.current_epoch,
                                    "batch_idx": batch_idx,
                                    "sample_in_batch": "all",
                                    "sample_global_index": "all",
                                    "dataset": dataset_name,
                                    "rollout_step": rollout_step,
                                    "lead_index": "all",
                                    "variable": variable_name,
                                    "scope": "configured_loss_rollout_variable_all_samples",
                                    **self._row_loss_columns(
                                        scaled_loss,
                                        raw_loss,
                                        scaled_loss / n_output_variables,
                                    ),
                                    "note": "Single-variable loss and its estimated contribution after full-variable averaging.",
                                },
                            )

                    for sample_idx in range(y_pred.shape[0]):
                        sample_global_index = batch_idx * batch_size + sample_idx
                        scaled_loss, raw_loss = self._loss_values(loss_fn, y_pred, y_true, sample_idx=sample_idx)
                        rows.append(
                            {
                                "epoch": trainer.current_epoch,
                                "batch_idx": batch_idx,
                                "sample_in_batch": sample_idx,
                                "sample_global_index": sample_global_index,
                                "dataset": dataset_name,
                                "rollout_step": rollout_step,
                                "lead_index": "all",
                                "variable": "all",
                                "scope": "configured_loss_sample_all_variables",
                                **self._row_loss_columns(scaled_loss, raw_loss),
                                "note": "",
                            },
                        )

                        for lead_idx in range(y_pred.shape[1]):
                            scaled_loss, raw_loss = self._loss_values(
                                loss_fn,
                                y_pred,
                                y_true,
                                sample_idx=sample_idx,
                                lead_idx=lead_idx,
                            )
                            rows.append(
                                {
                                    "epoch": trainer.current_epoch,
                                    "batch_idx": batch_idx,
                                    "sample_in_batch": sample_idx,
                                    "sample_global_index": sample_global_index,
                                    "dataset": dataset_name,
                                    "rollout_step": rollout_step,
                                    "lead_index": lead_idx,
                                    "variable": "all",
                                    "scope": "configured_loss_sample_lead_all_variables",
                                    **self._row_loss_columns(scaled_loss, raw_loss),
                                    "note": "",
                                },
                            )

                            if self.write_sample_variable_rows:
                                for variable_name, variable_index in variable_items:
                                    scaled_loss, raw_loss = self._loss_values(
                                        loss_fn,
                                        y_pred,
                                        y_true,
                                        sample_idx=sample_idx,
                                        lead_idx=lead_idx,
                                        var_idx=variable_index,
                                    )
                                    rows.append(
                                        {
                                            "epoch": trainer.current_epoch,
                                            "batch_idx": batch_idx,
                                            "sample_in_batch": sample_idx,
                                            "sample_global_index": sample_global_index,
                                            "dataset": dataset_name,
                                            "rollout_step": rollout_step,
                                            "lead_index": lead_idx,
                                            "variable": variable_name,
                                            "scope": "configured_loss_sample_lead_variable",
                                            **self._row_loss_columns(
                                                scaled_loss,
                                                raw_loss,
                                                scaled_loss / n_output_variables,
                                            ),
                                            "note": "",
                                        },
                                    )

        self._write_detail_rows(rows)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        epoch_loss = self._epoch_weighted_loss / self._epoch_weight if self._epoch_weight else float("nan")
        self._write_summary_row(
            {
                "epoch": trainer.current_epoch,
                "weighted_mean_lightning_val_step_total": epoch_loss,
                "num_weighted_samples": self._epoch_weight,
                "detail_file": str(self._detail_path) if self._detail_path else "",
                "note": "Should match val_*_loss_epoch for single-rank validation with every_n_batches=1.",
            },
        )
        LOGGER.info("Validation loss table written to %s", self.output_dir)
