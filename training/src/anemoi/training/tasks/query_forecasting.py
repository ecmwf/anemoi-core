# (C) Copyright 2026 Anemoi contributors.

"""Configuration holder for query-first direct forecasting."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


class QueryForecasting:
    """Define the distributions and context used by query-aware readers."""

    name = "query-forecasting"

    def __init__(
        self,
        lead_times: list[str],
        input_history: str,
        samples_per_epoch: int,
        reference_provenance: str,
        target_variables: list[str] | None = None,
        input_variables: list[str] | None = None,
        source_dropout: float = 0.0,
        field_dropout: float = 0.0,
        history_dropout: float = 0.0,
        max_input_times: int = 4,
        input_context_margin_degrees: float = 0.0,
        global_context_sources: list[str] | None = None,
        target_regions: list[list[float]] | None = None,
        target_regions_by_provenance: dict[str, list[list[float]]] | None = None,
        variable_weights: dict[str, float] | None = None,
        provenance_weights: dict[str, float] | None = None,
        loss_weights: dict[str, float] | None = None,
        spatial_weighting: str = "uniform",
        aliases: dict[str, str] | None = None,
        availability_policy: str = "retrospective",
        availability_lag: dict[str, str] | None = None,
        validation_seed: int = 17,
        validation_samples: int = 16,
        seed: int = 42,
        **_kwargs: Any,
    ) -> None:
        self.lead_times = [frequency_to_timedelta(value) for value in lead_times]
        if any(value <= timedelta(0) for value in self.lead_times):
            msg = "Query lead_times must all be positive."
            raise ValueError(msg)
        self.input_history = frequency_to_timedelta(input_history)
        self.samples_per_epoch = samples_per_epoch
        self.reference_provenance = reference_provenance
        self.target_variables = target_variables
        self.input_variables = input_variables
        self.source_dropout = source_dropout
        self.field_dropout = field_dropout
        self.history_dropout = history_dropout
        self.max_input_times = max_input_times
        self.input_context_margin_degrees = input_context_margin_degrees
        self.global_context_sources = global_context_sources or []
        self.target_regions = target_regions or []
        self.target_regions_by_provenance = target_regions_by_provenance or {}
        self.variable_weights = variable_weights or {}
        self.provenance_weights = provenance_weights or {}
        self.loss_weights = loss_weights or {}
        self.spatial_weighting = spatial_weighting
        self.aliases = aliases or {}
        self.availability_policy = availability_policy
        if availability_policy != "retrospective":
            msg = "Only the explicit 'retrospective' availability policy is implemented."
            raise ValueError(msg)
        self.availability_lag = {
            name: frequency_to_timedelta(value) for name, value in (availability_lag or {}).items()
        }
        self.validation_seed = validation_seed
        self.validation_samples = validation_samples
        self.seed = seed
        self._plot_adapter = None

    @property
    def num_input_timesteps(self) -> int:
        return 1

    @property
    def num_output_timesteps(self) -> int:
        return 1

    def fill_metadata(self, metadata: dict) -> None:
        metadata["task"] = self.name
        metadata["metadata_inference"]["task"] = self.name
        metadata["metadata_inference"]["query"] = {
            "lead_times_seconds": [int(value.total_seconds()) for value in self.lead_times],
            "input_history_seconds": int(self.input_history.total_seconds()),
            "reference_provenance": self.reference_provenance,
            "availability_policy": self.availability_policy,
            "availability_lag_seconds": {
                name: int(value.total_seconds()) for name, value in self.availability_lag.items()
            },
            "bbox_order": ["west", "south", "east", "north"],
            "coordinate_order": ["latitude", "longitude"],
            "coordinate_units": ["degrees_north", "degrees_east"],
            "time_semantics": "lead_time, cadence, and represented interval are distinct",
            "input_offsets": [frequency_to_string(-self.input_history), "0h"],
        }

    def log_extra(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def log_training_state(self) -> None:
        return

    def training_runtime_state_dict(self) -> dict:
        return {}

    def load_training_runtime_state_dict(self, _state: dict) -> None:
        return

    def on_train_epoch_end(self, _current_epoch: int) -> None:
        return
