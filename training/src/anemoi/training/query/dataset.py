# (C) Copyright 2026 Anemoi contributors.

"""Query-first sampling over independent Anemoi datasets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from anemoi.training.query.batch import QueryBatch
from anemoi.training.query.batch import QueryInput
from anemoi.training.query.query import ForecastQuery

if TYPE_CHECKING:
    from anemoi.training.data.data_reader import NativeGridDataset
    from anemoi.training.query.catalogue import CatalogueField
    from anemoi.training.query.catalogue import QueryCatalogue


class QueryDataset(Dataset):
    """Sample a supported request before selecting targets and inputs."""

    def __init__(  # noqa: C901
        self,
        readers: dict[str, NativeGridDataset],
        catalogue: QueryCatalogue,
        task: Any,
        sampling_weights: dict[str, float],
        length: int,
        seed: int,
    ) -> None:
        self.readers = readers
        self.catalogue = catalogue
        self.task = task
        self.sampling_weights = sampling_weights
        self.length = length
        self.seed = seed
        self.epoch = 0
        self.model_group_id = 0

        requested_fields = catalogue.restrict_targets(task.target_variables)
        if not requested_fields:
            msg = "No query targets remain after intersecting target_variables with catalogue availability."
            raise ValueError(msg)

        self.fields_by_dataset = {
            name: [
                field
                for field in catalogue.fields
                if field.dataset == name
                and field.input_supported
                and (not task.input_variables or field.variable in set(task.input_variables))
            ]
            for name in readers
        }
        unavailable_context = [name for name in task.global_context_sources if not self.fields_by_dataset.get(name)]
        if unavailable_context:
            msg = f"Required global context sources have no selectable input fields: {unavailable_context}."
            raise ValueError(msg)
        self.date_values = {
            name: np.asarray(reader.dates).astype("datetime64[ns]").astype(np.int64) for name, reader in readers.items()
        }
        self.missing_positions = {name: reader.missing_positions() for name, reader in readers.items()}
        self.target_positions = {}
        for name, values in self.date_values.items():
            for lead_time in task.lead_times:
                lead_ns = int(lead_time.total_seconds() * 1e9)
                valid_context = np.ones(len(values), dtype=bool)
                origins = values - lead_ns
                if self.missing_positions[name]:
                    valid_context[list(self.missing_positions[name])] = False
                history_ns = int(task.input_history.total_seconds() * 1e9)
                for source_name in task.global_context_sources:
                    if source_name not in self.date_values:
                        valid_context[:] = False
                        continue
                    lag_ns = int(
                        task.availability_lag.get(
                            source_name,
                            timedelta(0),
                        ).total_seconds()
                        * 1e9,
                    )
                    source_dates = np.delete(
                        self.date_values[source_name],
                        list(self.missing_positions[source_name]),
                    )
                    source_positions = np.searchsorted(source_dates, origins - lag_ns, side="right") - 1
                    has_history = source_positions >= 0
                    has_history[has_history] &= (
                        source_dates[source_positions[has_history]] >= origins[has_history] - history_ns
                    )
                    valid_context &= has_history
                any_context = np.zeros(len(values), dtype=bool)
                for source_name, source_values in self.date_values.items():
                    if not self.fields_by_dataset[source_name]:
                        continue
                    lag_ns = int(task.availability_lag.get(source_name, timedelta(0)).total_seconds() * 1e9)
                    source_dates = np.delete(
                        source_values,
                        list(self.missing_positions[source_name]),
                    )
                    source_positions = np.searchsorted(source_dates, origins - lag_ns, side="right") - 1
                    has_history = source_positions >= 0
                    has_history[has_history] &= (
                        source_dates[source_positions[has_history]] >= origins[has_history] - history_ns
                    )
                    any_context |= has_history
                valid_context &= any_context
                self.target_positions[name, lead_ns] = np.flatnonzero(
                    valid_context,
                ).astype(np.int64)
        self.sampling_options = []
        for field in requested_fields:
            configured_regions = task.target_regions_by_provenance.get(
                field.provenance,
                task.target_regions,
            )
            regions = [tuple(area) for area in configured_regions] or [None]
            self.sampling_options.extend(
                (field, lead_time, area)
                for lead_time in task.lead_times
                for area in regions
                if len(
                    self.target_positions[
                        field.dataset,
                        int(lead_time.total_seconds() * 1e9),
                    ],
                )
                and self._bbox_mask(readers[field.dataset], area).any()
            )
        # A provenance weight of zero is the supported way to retain a source
        # as input context without supervising it as a target. Remove those
        # options before the variable-first draw, otherwise a variable found
        # only in a context source creates a zero-sum provenance distribution.
        self.sampling_options = [
            option
            for option in self.sampling_options
            if self.task.variable_weights.get(option[0].variable, 1.0) > 0
            and self.task.provenance_weights.get(option[0].provenance, 1.0)
            * self.sampling_weights[option[0].provenance]
            > 0
        ]
        self.target_fields = list({option[0] for option in self.sampling_options})
        if not self.sampling_options:
            msg = "No supported target field/lead/region combination has a matching origin and required context."
            raise ValueError(
                msg,
            )

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_comm_group_info(
        self,
        _global_rank: int,
        model_group_id: int,
        _model_group_rank: int,
        _num_model_groups: int,
        _reader_group_rank: int,
        _reader_group_size: int,
        _shard_sizes: dict,
    ) -> None:
        self.model_group_id = model_group_id

    def _sample_option(
        self,
        rng: np.random.Generator,
    ) -> tuple[CatalogueField, timedelta, tuple[float, float, float, float] | None]:
        variables = sorted({field.variable for field in self.target_fields})
        variable_weights = np.asarray(
            [self.task.variable_weights.get(name, 1.0) for name in variables],
            dtype=float,
        )
        variable = variables[rng.choice(len(variables), p=variable_weights / variable_weights.sum())]
        candidates = [option for option in self.sampling_options if option[0].variable == variable]
        provenances = sorted({option[0].provenance for option in candidates})
        provenance_weights = np.asarray(
            [
                self.task.provenance_weights.get(provenance, 1.0) * self.sampling_weights[provenance]
                for provenance in provenances
            ],
            dtype=float,
        )
        provenance = provenances[
            rng.choice(
                len(provenances),
                p=provenance_weights / provenance_weights.sum(),
            )
        ]
        candidates = [option for option in candidates if option[0].provenance == provenance]
        return candidates[rng.integers(len(candidates))]

    @staticmethod
    def _bbox_mask(
        reader: NativeGridDataset,
        bbox: tuple[float, float, float, float] | None,
    ) -> np.ndarray:
        if bbox is None:
            return np.ones(reader.grid_size, dtype=bool)
        west, south, east, north = bbox
        latitudes = np.asarray(reader.data.latitudes)
        longitudes = (np.asarray(reader.data.longitudes) + 180) % 360 - 180
        return (longitudes >= west) & (longitudes <= east) & (latitudes >= south) & (latitudes <= north)

    def __getitem__(self, index: int) -> QueryBatch:  # noqa: C901
        rng = np.random.default_rng(
            self.seed + self.epoch * self.length + index + self.model_group_id * 1_000_003,
        )
        field, lead_time, bbox = self._sample_option(rng)
        target_reader = self.readers[field.dataset]
        lead_ns = int(lead_time.total_seconds() * 1e9)
        target_candidates = self.target_positions[field.dataset, lead_ns]
        if not len(target_candidates):
            msg = (
                f"Dataset {field.dataset!r} has no target/origin pairs for lead time {lead_time}. "
                "Change task.lead_times or the dataset date selection."
            )
            raise ValueError(
                msg,
            )
        target_position = int(rng.choice(target_candidates))
        valid_time = np.datetime64(target_reader.dates[target_position], "ns")
        origin = valid_time - np.timedelta64(lead_ns, "ns")
        query = ForecastQuery(
            variable=field.variable,
            lead_time=lead_time,
            provenance=field.provenance,
            unit=field.units or "unknown",
            output_frequency=target_reader.frequency,
            model_type={"surface": "sfc", "pressure": "pl", "model": "ml"}.get(
                field.level_type,
                field.level_type,
            ),
            level=(
                field.pressure_pa / 100
                if field.pressure_pa is not None
                else field.model_level
                if field.model_level is not None
                else field.height_m
            ),
            level_unit=("hPa" if field.pressure_pa is not None else "m" if field.height_m is not None else None),
            aggregation_type=field.aggregation_type,
            temporal_aggregation_window=(
                None
                if field.temporal_aggregation_window_hours is None
                else tuple(timedelta(hours=value) for value in field.temporal_aggregation_window_hours)
            ),
            bbox=bbox,
            grid=field.dataset,
            grid_spacing_km=field.resolution_km,
            spatial_support_km=field.spatial_support_km,
        )

        inputs: dict[str, QueryInput] = {}
        input_context: dict[str, Any] = {}
        all_source_names = list(self.readers)
        source_names = list(all_source_names)
        if self.task.source_dropout and len(source_names) > 1:
            source_names = [
                name
                for name in source_names
                if name in self.task.global_context_sources or rng.random() >= self.task.source_dropout
            ]
            if not source_names:
                source_names = [field.dataset]

        for source_name in source_names:
            reader = self.readers[source_name]
            lag = self.task.availability_lag.get(source_name, timedelta(0))
            latest = origin - np.timedelta64(int(lag.total_seconds() * 1e9), "ns")
            earliest = origin - np.timedelta64(
                int(self.task.input_history.total_seconds() * 1e9),
                "ns",
            )
            source_dates = self.date_values[source_name]
            start = np.searchsorted(
                source_dates,
                earliest.astype(np.int64),
                side="left",
            )
            stop = np.searchsorted(source_dates, latest.astype(np.int64), side="right")
            positions = list(range(start, stop))
            positions = [position for position in positions if position not in self.missing_positions[source_name]]
            if not positions or not self.fields_by_dataset[source_name]:
                continue
            positions = positions[-self.task.max_input_times :]
            available_time_count = len(positions)
            if self.task.history_dropout and len(positions) > 1:
                latest_position = positions[-1]
                positions = [position for position in positions if rng.random() >= self.task.history_dropout]
                if not positions:
                    positions = [latest_position]

            source_fields = list(self.fields_by_dataset[source_name])
            available_field_count = len(source_fields)
            if self.task.field_dropout and len(source_fields) > 1:
                source_fields = [field_ for field_ in source_fields if rng.random() >= self.task.field_dropout]
                if not source_fields:
                    source_fields = [
                        self.fields_by_dataset[source_name][rng.integers(len(self.fields_by_dataset[source_name]))],
                    ]

            context_bbox = None if source_name in self.task.global_context_sources else bbox
            if (
                bbox is not None
                and source_name not in self.task.global_context_sources
                and self.task.input_context_margin_degrees
            ):
                west, south, east, north = bbox
                margin = self.task.input_context_margin_degrees
                context_bbox = (
                    west - margin,
                    south - margin,
                    east + margin,
                    north + margin,
                )
            spatial_mask = self._bbox_mask(reader, context_bbox)
            grid_indices = np.flatnonzero(spatial_mask)
            if not len(grid_indices):
                continue
            loaded = reader.get_sample(
                0,
                np.asarray(positions),
                None if context_bbox is None else grid_indices,
            )[:, 0]
            values = []
            metadata = []
            variable_ids = []
            provenance_ids = []
            unit_ids = []
            resolved_fields = []
            for time_index, position in enumerate(positions):
                offset_hours = float(
                    (np.datetime64(reader.dates[position], "ns") - origin) / np.timedelta64(1, "h"),
                )
                for source_field in source_fields:
                    raw = loaded[
                        time_index,
                        :,
                        reader.name_to_index[source_field.field_name],
                    ].float()
                    values.append((raw - source_field.mean) / source_field.stdev)
                    metadata.append(
                        self.catalogue.encode_metadata(source_field, offset_hours),
                    )
                    variable_ids.append(
                        self.catalogue.variable_to_id[source_field.variable],
                    )
                    provenance_ids.append(
                        self.catalogue.provenance_to_id[source_field.provenance],
                    )
                    unit_ids.append(
                        self.catalogue.unit_to_id[source_field.units or "unknown"],
                    )
                    resolved_fields.append(
                        {
                            **asdict(source_field),
                            "actual_time": str(np.datetime64(reader.dates[position], "ns")),
                            "time_offset_hours": offset_hours,
                        },
                    )

            selected_values = torch.stack(values, dim=-1)
            values_tensor = torch.zeros(
                (reader.grid_size, selected_values.shape[-1]),
                dtype=selected_values.dtype,
            )
            mask = torch.zeros_like(values_tensor, dtype=torch.bool)
            values_tensor[grid_indices] = torch.nan_to_num(selected_values)
            mask[grid_indices] = torch.isfinite(selected_values)
            inputs[source_name] = QueryInput(
                values=values_tensor[None],
                metadata=torch.from_numpy(np.stack(metadata))[None],
                variable_ids=torch.tensor(variable_ids, dtype=torch.long)[None],
                provenance_ids=torch.tensor(provenance_ids, dtype=torch.long)[None],
                unit_ids=torch.tensor(unit_ids, dtype=torch.long)[None],
                mask=mask[None],
            )
            input_context[source_name] = {
                "resolved_fields": resolved_fields,
                "actual_times": [str(np.datetime64(reader.dates[position], "ns")) for position in positions],
                "available_field_count": available_field_count,
                "selected_field_count": len(source_fields),
                "available_time_count": available_time_count,
                "selected_time_count": len(positions),
                "selected_node_count": len(grid_indices),
                "native_node_count": int(reader.grid_size),
                "context_bbox_degrees": context_bbox,
            }

        if not inputs:
            msg = f"No source has data available at forecast origin {origin} for target {valid_time}."
            raise ValueError(msg)

        target_nodes = self._bbox_mask(target_reader, bbox)
        target_indices = np.flatnonzero(target_nodes)
        target = target_reader.get_sample(
            0,
            np.asarray([target_position]),
            target_indices,
        )[
            0,
            0,
            :,
            target_reader.name_to_index[field.field_name],
        ].float()
        target = (target - field.mean) / field.stdev
        target_mask = torch.isfinite(target)
        target_latitudes = np.asarray(target_reader.data.latitudes)[target_indices]
        target_longitudes = (np.asarray(target_reader.data.longitudes)[target_indices] + 180) % 360 - 180
        output_coordinates = torch.deg2rad(
            torch.from_numpy(
                np.stack((target_latitudes, target_longitudes), axis=-1),
            ).float(),
        )
        query_metadata = self.catalogue.encode_metadata(
            query,
            lead_time.total_seconds() / 3600,
        )
        loss_weight = (
            self.task.loss_weights.get(field.variable, 1.0)
            * self.task.loss_weights.get(field.provenance, 1.0)
            * self.task.loss_weights.get(f"{field.variable}@{field.provenance}", 1.0)
        )
        return QueryBatch(
            inputs=inputs,
            query_metadata=torch.from_numpy(query_metadata)[None],
            query_variable_id=torch.tensor(
                [self.catalogue.variable_to_id[field.variable]],
            ),
            query_provenance_id=torch.tensor(
                [self.catalogue.provenance_to_id[field.provenance]],
            ),
            query_unit_id=torch.tensor(
                [self.catalogue.unit_to_id[field.units or "unknown"]],
            ),
            target_dataset=field.dataset,
            query=query.as_serialisable_dict(),
            output_coordinates=output_coordinates,
            target=torch.nan_to_num(target)[None],
            target_mask=target_mask[None],
            loss_weight=torch.tensor(loss_weight, dtype=torch.float32),
            diagnostic_context={
                "sample_index": int(index),
                "forecast_origin": str(origin),
                "valid_time": str(valid_time),
                "requested_query": query.as_serialisable_dict(),
                "resolved_query": {
                    **query.as_serialisable_dict(),
                    "target_dataset": field.dataset,
                    "target_field_name": field.field_name,
                    "target_units": field.units,
                },
                "target": {
                    **asdict(field),
                    "native_index_count": len(target_indices),
                    "native_index_min": int(target_indices.min()),
                    "native_index_max": int(target_indices.max()),
                    "valid_node_count": int(target_mask.sum()),
                    "selected_node_count": len(target_indices),
                },
                "inputs": input_context,
                "configured_sources": all_source_names,
                "dataset_sampling_weights": dict(self.sampling_weights),
                "active_sources": list(inputs),
                "source_dropout_sources": [name for name in all_source_names if name not in source_names],
                "unavailable_sources": [name for name in source_names if name not in inputs],
            },
        )


def collate_query_batch(samples: list[QueryBatch]) -> QueryBatch:
    """Keep variable source sets intact; one task per model group is intentional."""
    if len(samples) != 1:
        msg = "Query-based forecasting currently requires dataloader batch_size=1."
        raise ValueError(msg)
    return samples[0]
