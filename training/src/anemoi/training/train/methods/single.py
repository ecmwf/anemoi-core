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
import random

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class SingleTraining(BaseTrainingModule):
    """Base class for deterministic prediction tasks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.primary_dataset = getattr(self.model.model, "principal_dataset_name", self.dataset_names[0])
        if self.primary_dataset not in self.dataset_names:
            self.primary_dataset = self.dataset_names[0]

        self.dropout_by_dataset = self._build_dropout_map(
            self.config.training.get("dataset_dropout_p", 0.0),
            label="dataset",
        )
        self.decoder_dropout_by_dataset = self._build_dropout_map(
            self.config.training.get("decoder_dropout_p", 0.0),
            label="decoder",
        )

        # Datasets to treat as optional (their inputs may be entirely NaN
        # on padded / missing dates). Must be listed explicitly under
        # `dataloader.optional_datasets` — nothing is optional by default.
        # The primary dataset is stripped from the list as a safety guard.
        cfg_optional = self.config.dataloader.get("optional_datasets", None)
        if cfg_optional is None:
            self.optional_datasets: set[str] = set()
        else:
            requested = set(cfg_optional)
            unknown = requested - set(self.dataset_names)
            if unknown:
                LOGGER.warning(
                    "dataloader.optional_datasets references unknown datasets %s; ignoring.",
                    sorted(unknown),
                )
            self.optional_datasets = {
                name for name in requested if name in self.dataset_names and name != self.primary_dataset
            }
        if self.optional_datasets:
            LOGGER.info(
                "SingleTraining will auto-drop optional datasets on NaN inputs: %s (principal='%s')",
                sorted(self.optional_datasets),
                self.primary_dataset,
            )

    def _build_dropout_map(self, dropout_cfg, *, label: str) -> dict[str, float]:
        dropout_map = {name: 0.0 for name in self.dataset_names if name != self.primary_dataset}
        if isinstance(dropout_cfg, (int, float)):
            dropout_value = float(dropout_cfg)
            for name in dropout_map:
                dropout_map[name] = dropout_value
                LOGGER.info(f"{label} dropout probability for dataset '{name}': {dropout_value}")
        else:
            for name in dropout_map:
                dropout_map[name] = float(dropout_cfg.get(name, 0.0))
                LOGGER.info(f"{label} dropout probability for dataset '{name}': {dropout_map[name]}")
        return dropout_map

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, list]:
        """Training / validation step."""
        # Debug counters: how many times this rank has entered `_step` for
        # each mode. Split so training and validation don't interleave. Useful
        # to compare against `self.global_step` (optimizer-step counter) to
        # verify whether `accum_grad_batches` is actually accumulating.
        if validation_mode:
            self._val_step_call_count = getattr(self, "_val_step_call_count", 0) + 1
            step_call = self._val_step_call_count
        else:
            self._train_step_call_count = getattr(self, "_train_step_call_count", 0) + 1
            step_call = self._train_step_call_count

        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)

        task_steps = self.task.steps("training" if not validation_mode else "validation")

        # Define dataset dropout once per batch-step and reuse for all rollout
        # iterations so the same datasets are used within this sequence.
        dropped_datasets = None
        decoder_dropped_datasets = None
        if not validation_mode:
            if len(self.dropout_by_dataset) > 0:
                dropped_datasets = [
                    name
                    for name, dropout_p in self.dropout_by_dataset.items()
                    if random.random() < dropout_p
                ]
            if len(self.decoder_dropout_by_dataset) > 0:
                already_dropped = set(dropped_datasets or [])
                decoder_dropped_datasets = [
                    name
                    for name, dropout_p in self.decoder_dropout_by_dataset.items()
                    if name not in already_dropped and random.random() < dropout_p
                ]
        # Auto-drop set for this batch, derived from the pre-imputation NaN
        # snapshot recorded in _normalize_batch. Same for every rollout step
        # of this _step call, so we compute (and log) it once per batch.
        batch_auto_dropped: set[str] = set()
        if self.optional_datasets:
            pre_impute_nan = getattr(self, "_batch_nan_datasets", set())
            batch_auto_dropped = self.optional_datasets & pre_impute_nan

            # One line per batch per rank (so we see every sample across all
            # DDP replicas). "DROP"/"OK" leads the line for easy grep/scan.
            # `call=K` is a per-rank counter of `_step` invocations for this
            # mode; comparing it with `step=N` (global_step / optimizer step)
            # tells you whether `accum_grad_batches` is actually accumulating
            # (expect call to advance by accum_grad_batches per unit of step).
            # tag = "DROP" if batch_auto_dropped else "OK  "
            # mode = "val" if validation_mode else "train"
            # print(
            #     f"{tag} [optional-drop] rank={getattr(self, 'global_rank', '?')} {mode} "
            #     f"epoch={getattr(self, 'current_epoch', '?')} "
            #     f"step={getattr(self, 'global_step', '?')} "
            #     f"call={step_call} "
            #     f"dropped={sorted(batch_auto_dropped) if batch_auto_dropped else '[]'}",
            #     flush=True,
            # )

        for step_idx, task_kwargs in enumerate(task_steps):
            # For decoder-only dropout, the dataset is fed to the encoder on the
            # first step but its decoder output is masked. On subsequent rollout
            # steps there is no valid predicted state to advance from, so the
            # dataset becomes fully dropped (encoder also gated).
            if step_idx == 0:
                current_dropped = dropped_datasets
                current_decoder_dropped = decoder_dropped_datasets
            else:
                current_dropped = list(set(dropped_datasets or []) | set(decoder_dropped_datasets or []))
                current_decoder_dropped = None

            # Fold the batch-level auto-drop into this rollout step's dropped set.
            if batch_auto_dropped:
                current_dropped = list(set(current_dropped or []) | batch_auto_dropped)
                if current_decoder_dropped is not None:
                    current_decoder_dropped = [n for n in current_decoder_dropped if n not in batch_auto_dropped]

            # if x has nan
            for name, tensor in x.items():
                if name in (current_dropped or []):
                    # Dropped datasets were zero-filled above; skip the assertion.
                    continue
                assert not torch.isnan(tensor).any(), f"NaN values found in input for dataset {name}."

            y_pred = self(
                x,
                dropped_dataset_names=current_dropped,
                decoder_dropped_dataset_names=current_decoder_dropped,
            )

            y = self.task.get_targets(batch, **task_kwargs)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                **task_kwargs,
                validation_mode=validation_mode,
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
                use_reentrant=False,
            )

            # Advance input state for each dataset. Datasets whose decoder was
            # dropped this step produced NaN forecasts and must be zero-filled
            # (same handling as fully-dropped datasets).
            advance_dropped = list(set(current_dropped or []) | set(current_decoder_dropped or []))
            x = self.task.advance_input(
                x,
                y_preds_next,
                batch,
                **task_kwargs,
                data_indices=self.data_indices,
                output_mask=self.output_mask,
                grid_shard_slice=self.grid_shard_slice,
                dropped_datasets=advance_dropped,
            )

            loss = loss + loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / len(task_steps)
        return loss, metrics, y_preds
