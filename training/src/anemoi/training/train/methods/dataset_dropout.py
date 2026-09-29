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
import random

import torch

LOGGER = logging.getLogger(__name__)


class DatasetDropoutMixin:
    """Per-batch dataset dropout for multi-encoder training methods.

    Shared by :class:`SingleTraining` and :class:`EnsembleTraining`. Provides

    * encoder dropout (``training.dataset_dropout_p``): the dataset is gated off
      at the encoder and its decoder output is NaN-masked;
    * decoder-only dropout (``training.decoder_dropout_p``): the dataset still
      feeds the encoder on the first rollout step but its decoder output is
      NaN-masked, so it receives no gradient;
    * auto-drop of optional datasets (``dataloader.optional_datasets``) whose
      pre-imputation input for this batch was entirely NaN on some timestep
      (see ``BaseTrainingModule._normalize_batch``).

    The host class must define ``self.config``, ``self.dataset_names`` and
    ``self.model`` before calling :meth:`_setup_dataset_dropout`. The step
    helpers tolerate a host that skipped the setup (e.g. unit-test stubs) and
    then apply no dropout at all.
    """

    def _setup_dataset_dropout(self) -> None:
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
                "%s will auto-drop optional datasets on NaN inputs: %s (principal='%s')",
                type(self).__name__,
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

    def _sample_dataset_dropout(
        self,
        validation_mode: bool,
    ) -> tuple[list[str] | None, list[str] | None, set[str]]:
        """Draw the dataset dropout for one batch.

        Called once per ``_step`` so the same datasets are dropped for every
        rollout iteration of that batch.

        Returns
        -------
        tuple[list[str] | None, list[str] | None, set[str]]
            ``(dropped, decoder_dropped, batch_auto_dropped)``: randomly dropped
            datasets, randomly decoder-only-dropped datasets (both ``None`` in
            validation mode), and optional datasets auto-dropped because their
            pre-imputation input was NaN.
        """
        dropped_datasets = None
        decoder_dropped_datasets = None
        if not validation_mode:
            dropout_by_dataset = getattr(self, "dropout_by_dataset", {})
            if len(dropout_by_dataset) > 0:
                dropped_datasets = [
                    name for name, dropout_p in dropout_by_dataset.items() if random.random() < dropout_p
                ]
            decoder_dropout_by_dataset = getattr(self, "decoder_dropout_by_dataset", {})
            if len(decoder_dropout_by_dataset) > 0:
                already_dropped = set(dropped_datasets or [])
                decoder_dropped_datasets = [
                    name
                    for name, dropout_p in decoder_dropout_by_dataset.items()
                    if name not in already_dropped and random.random() < dropout_p
                ]

        # Auto-drop set for this batch, derived from the pre-imputation NaN
        # snapshot recorded in _normalize_batch.
        batch_auto_dropped: set[str] = set()
        optional_datasets = getattr(self, "optional_datasets", set())
        if optional_datasets:
            pre_impute_nan = getattr(self, "_batch_nan_datasets", set())
            batch_auto_dropped = optional_datasets & pre_impute_nan

        return dropped_datasets, decoder_dropped_datasets, batch_auto_dropped

    @staticmethod
    def _dropout_for_step(
        step_idx: int,
        dropped_datasets: list[str] | None,
        decoder_dropped_datasets: list[str] | None,
        batch_auto_dropped: set[str],
    ) -> tuple[list[str] | None, list[str] | None]:
        """Resolve the encoder- and decoder-dropped datasets for one rollout step.

        For decoder-only dropout, the dataset is fed to the encoder on the
        first step but its decoder output is masked. On subsequent rollout
        steps there is no valid predicted state to advance from, so the
        dataset becomes fully dropped (encoder also gated). The batch-level
        auto-drop is folded into every step.
        """
        if step_idx == 0:
            current_dropped = dropped_datasets
            current_decoder_dropped = decoder_dropped_datasets
        else:
            current_dropped = list(set(dropped_datasets or []) | set(decoder_dropped_datasets or []))
            current_decoder_dropped = None

        if batch_auto_dropped:
            current_dropped = list(set(current_dropped or []) | batch_auto_dropped)
            if current_decoder_dropped is not None:
                current_decoder_dropped = [n for n in current_decoder_dropped if n not in batch_auto_dropped]

        return current_dropped, current_decoder_dropped

    @staticmethod
    def _assert_inputs_finite(x: dict[str, torch.Tensor], dropped_datasets: list[str] | None) -> None:
        """Assert no NaN reaches the model except through dropped (zero-filled) datasets."""
        for name, tensor in x.items():
            if name in (dropped_datasets or []):
                continue
            assert not torch.isnan(tensor).any(), f"NaN values found in input for dataset {name}."

    @staticmethod
    def _advance_dropped(
        current_dropped: list[str] | None,
        current_decoder_dropped: list[str] | None,
    ) -> list[str]:
        """Datasets whose forecast this step is NaN and must be zero-filled before advancing the input."""
        return list(set(current_dropped or []) | set(current_decoder_dropped or []))
