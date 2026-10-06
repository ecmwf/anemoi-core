# (C) Copyright 2026 Anemoi contributors.

"""Query-first sampling over independent Anemoi datasets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from anemoi.models.distributed.balanced_partition import get_partition_range
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
        full_valid_time_pass: bool = False,
    ) -> None:
        self.readers = readers
        self.catalogue = catalogue
        self.task = task
        self.sampling_weights = sampling_weights
        self.length = length
        self.seed = seed
        self.full_valid_time_pass = full_valid_time_pass
        self.epoch = 0
        self.model_group_id = 0
        self.model_group_rank = 0
        self.sample_group_id = 0
        self.sample_group_count = 1
        self.shard_sizes: dict[str, list[int]] | None = None
        self._residual_baseline_tree = None
        self._residual_baseline_index_cache: dict[tuple, np.ndarray] = {}
        self._full_permutation_epoch: int | None = None
        self._full_permutation: np.ndarray | None = None

        requested_fields = catalogue.restrict_targets(task.target_variables)
        requested_fields = [
            field
            for field in requested_fields
            if field.variable
            not in task.excluded_target_variables_by_provenance.get(field.provenance, ())
        ]
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
        if task.residual_baseline_source is not None:
            source_reader = readers[task.residual_baseline_source]
            self._residual_baseline_tree = cKDTree(
                self._spherical_xyz(source_reader.data.latitudes, source_reader.data.longitudes),
            )
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
        # options before the hierarchical draw, otherwise a variable found
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

        self.full_samples: list[
            tuple[CatalogueField, timedelta, tuple[float, float, float, float] | None, int]
        ] = []
        if self.full_valid_time_pass:
            if not self.task.query_all_fields_per_provenance:
                msg = "full_valid_time_pass requires query_all_fields_per_provenance=true."
                raise ValueError(msg)
            # A bundled example contains every configured field for one
            # provenance.  Therefore the natural complete epoch is every
            # eligible (provenance, lead, region, valid time) exactly once,
            # rather than every scalar field/date combination.
            representatives: dict[
                tuple[str, str, int, tuple[float, float, float, float] | None],
                tuple[CatalogueField, timedelta, tuple[float, float, float, float] | None],
            ] = {}
            for option in self.sampling_options:
                field, lead_time, bbox = option
                key = (
                    field.dataset,
                    field.provenance,
                    int(lead_time.total_seconds() * 1e9),
                    bbox,
                )
                representatives.setdefault(key, option)
            for key in sorted(representatives, key=str):
                field, lead_time, bbox = representatives[key]
                lead_ns = int(lead_time.total_seconds() * 1e9)
                self.full_samples.extend(
                    (field, lead_time, bbox, int(target_position))
                    for target_position in self.target_positions[field.dataset, lead_ns]
                )
            if not self.full_samples:
                raise ValueError("full_valid_time_pass found no eligible bundled examples.")

    def __len__(self) -> int:
        if self.full_valid_time_pass:
            # Each model group consumes a different item at the same optimizer
            # step.  Padding by at most group_count-1 examples keeps collective
            # execution synchronized while covering the complete epoch.
            return (len(self.full_samples) + self.sample_group_count - 1) // self.sample_group_count
        return self.length

    @property
    def full_sample_count(self) -> int | None:
        """Number of unique bundled examples in a complete epoch, if enabled."""
        return len(self.full_samples) if self.full_valid_time_pass else None

    def _full_sample(
        self,
        index: int,
    ) -> tuple[CatalogueField, timedelta, tuple[float, float, float, float] | None, int, int]:
        if self._full_permutation_epoch != self.epoch or self._full_permutation is None:
            self._full_permutation = np.random.default_rng(self.seed + self.epoch).permutation(
                len(self.full_samples),
            )
            self._full_permutation_epoch = self.epoch
        global_index = index * self.sample_group_count + self.sample_group_id
        sample_index = int(self._full_permutation[global_index % len(self.full_samples)])
        field, lead_time, bbox, target_position = self.full_samples[sample_index]
        return field, lead_time, bbox, target_position, global_index

    @staticmethod
    def _spherical_xyz(latitudes: Any, longitudes: Any) -> np.ndarray:
        latitude = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
        longitude = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
        return np.column_stack(
            (
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ),
        )

    @staticmethod
    def _same_physical_field(first: Any, second: Any) -> bool:
        """Match physical semantics while deliberately ignoring provenance and grid."""

        def value(field: Any, name: str) -> Any:
            return field.get(name) if isinstance(field, dict) else getattr(field, name)

        names = (
            "variable",
            "level_type",
            "pressure_pa",
            "model_level",
            "height_m",
            "aggregation_type",
            "temporal_aggregation_window_hours",
        )
        return all(value(first, name) == value(second, name) for name in names)

    def _nearest_baseline_indices(
        self,
        dataset: str,
        target_indices: np.ndarray,
        shard_start: int,
        shard_end: int,
        bbox: tuple[float, float, float, float] | None,
    ) -> np.ndarray:
        if self._residual_baseline_tree is None:
            raise RuntimeError("Residual baseline tree was not initialized.")
        key = (dataset, shard_start, shard_end, bbox)
        cached = self._residual_baseline_index_cache.get(key)
        if cached is None:
            reader = self.readers[dataset]
            coordinates = self._spherical_xyz(
                np.asarray(reader.data.latitudes)[target_indices],
                np.asarray(reader.data.longitudes)[target_indices],
            )
            _, cached = self._residual_baseline_tree.query(coordinates, workers=-1)
            cached = np.asarray(cached, dtype=np.int64)
            self._residual_baseline_index_cache[key] = cached
        return cached

    def _forecast_query(
        self,
        field: CatalogueField,
        lead_time: timedelta,
        bbox: tuple[float, float, float, float] | None,
    ) -> ForecastQuery:
        reader = self.readers[field.dataset]
        return ForecastQuery(
            variable=field.variable,
            lead_time=lead_time,
            provenance=field.provenance,
            unit=field.units or "unknown",
            output_frequency=self.task.output_frequency or reader.frequency,
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Satisfy the shared loader contract; worker indices need no partitioning.

        PyTorch's sampler already assigns distinct example indices to workers,
        and each query is deterministically generated from its index. Native
        readers are opened lazily in the worker process.
        """
        del n_workers, worker_id

    def set_comm_group_info(
        self,
        _global_rank: int,
        model_group_id: int,
        model_group_rank: int,
        num_model_groups: int,
        _reader_group_rank: int,
        _reader_group_size: int,
        shard_sizes: dict,
    ) -> None:
        self.model_group_id = model_group_id
        self.model_group_rank = model_group_rank
        self.sample_group_id = model_group_id
        self.sample_group_count = num_model_groups
        self.shard_sizes = shard_sizes

    def set_ens_comm_group_info(
        self,
        ensemble_group_id: int,
        _ensemble_group_rank: int,
        ensemble_group_count: int,
    ) -> None:
        """Make all ensemble members draw exactly the same query and times."""
        self.sample_group_id = ensemble_group_id
        self.sample_group_count = ensemble_group_count

    def _grid_shard(self, dataset: str, grid_size: int) -> tuple[int, int]:
        """Return this model rank's contiguous native-grid interval."""
        if self.shard_sizes is None or dataset not in self.shard_sizes:
            return 0, grid_size
        sizes = self.shard_sizes[dataset]
        if sum(sizes) != grid_size:
            msg = f"Grid shards for {dataset!r} sum to {sum(sizes)}, expected {grid_size}."
            raise ValueError(msg)
        return get_partition_range(sizes, self.model_group_rank)

    def _source_fields(self, source_name: str) -> list[CatalogueField]:
        """Return input fields permitted for one source in the sampled query."""
        fields = list(self.fields_by_dataset[source_name])
        if self.task.target_static_context and source_name not in self.task.global_context_sources:
            fields = [field for field in fields if field.time_invariant]
        return fields

    def _sample_option(
        self,
        rng: np.random.Generator,
        index: int | None = None,
    ) -> tuple[CatalogueField, timedelta, tuple[float, float, float, float] | None]:
        if self.task.sampling_strategy == "provenance_field_cycle":
            if index is None:
                raise ValueError("provenance_field_cycle requires the dataset sample index.")
            provenances = sorted({field.provenance for field in self.target_fields})
            global_position = (
                (self.epoch * self.length + index) * self.sample_group_count
                + self.sample_group_id
            )
            provenance = provenances[global_position % len(provenances)]
            candidates = [
                option for option in self.sampling_options if option[0].provenance == provenance
            ]
            field_position = global_position // len(provenances)
            cycle, offset = divmod(field_position, len(candidates))
            stable_provenance = sum((position + 1) * ord(character) for position, character in enumerate(provenance))
            permutation = np.random.default_rng(
                self.seed + stable_provenance * 10_007 + cycle,
            ).permutation(len(candidates))
            return candidates[int(permutation[offset])]

        if self.task.sampling_strategy == "provenance_variable_level":
            provenances = sorted({field.provenance for field in self.target_fields})
            provenance_weights = np.asarray(
                [
                    self.task.provenance_weights.get(provenance, 1.0)
                    * self.sampling_weights[provenance]
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
            candidates = [option for option in self.sampling_options if option[0].provenance == provenance]
            variables = sorted({option[0].variable for option in candidates})
            variable_weights = np.asarray(
                [self.task.variable_weights.get(name, 1.0) for name in variables],
                dtype=float,
            )
            variable = variables[
                rng.choice(
                    len(variables),
                    p=variable_weights / variable_weights.sum(),
                )
            ]
            candidates = [option for option in candidates if option[0].variable == variable]
            # The remaining options distinguish pressure/model/height levels,
            # accumulation windows, leads and regions. A uniform final draw
            # prevents variables with many levels from dominating the earlier
            # provenance and variable choices.
            return candidates[rng.integers(len(candidates))]

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
        if self.full_valid_time_pass:
            field, lead_time, bbox, target_position, global_sample_index = self._full_sample(index)
            rng = np.random.default_rng(
                self.seed + self.epoch * len(self.full_samples) + global_sample_index,
            )
        else:
            rng = np.random.default_rng(
                self.seed + self.epoch * self.length + index + self.sample_group_id * 1_000_003,
            )
            field, lead_time, bbox = self._sample_option(rng, index=index)
            target_position = None
        fields = [field]
        if self.task.query_all_fields_per_provenance:
            fields = sorted(
                {
                    option[0]
                    for option in self.sampling_options
                    if option[0].provenance == field.provenance
                    and option[1] == lead_time
                    and option[2] == bbox
                },
                key=lambda item: item.field_name,
            )
            field = fields[0]
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
        if target_position is None:
            target_position = int(rng.choice(target_candidates))
        else:
            candidate_index = int(np.searchsorted(target_candidates, target_position))
            if candidate_index == len(target_candidates) or target_candidates[candidate_index] != target_position:
                msg = f"Scheduled target position {target_position} is not eligible for {field.dataset!r}."
                raise RuntimeError(msg)
        valid_time = np.datetime64(target_reader.dates[target_position], "ns")
        origin = valid_time - np.timedelta64(lead_ns, "ns")
        queries = [self._forecast_query(item, lead_time, bbox) for item in fields]
        query = queries[0]

        inputs: dict[str, QueryInput] = {}
        input_context: dict[str, Any] = {}
        all_source_names = list(self.readers)
        if self.task.target_static_context:
            source_names = list(dict.fromkeys([*self.task.global_context_sources, field.dataset]))
        else:
            source_names = list(all_source_names)
        if self.task.source_dropout and len(source_names) > 1 and not self.task.target_static_context:
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

            source_fields = self._source_fields(source_name)
            available_field_count = len(source_fields)
            if not source_fields:
                continue
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
            shard_start, shard_end = self._grid_shard(source_name, reader.grid_size)
            grid_indices = grid_indices[(grid_indices >= shard_start) & (grid_indices < shard_end)]
            # Native readers can push a contiguous slice into Zarr. Integer
            # arrays are applied only after a full-grid read, which defeats
            # model sharding. Read this rank's contiguous interval and apply
            # any geographic context mask locally below.
            loaded = reader.get_sample(
                0,
                np.asarray(positions),
                slice(shard_start, shard_end),
            )[:, 0]
            local_grid_indices = grid_indices - shard_start
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
                    ][local_grid_indices].float()
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
                (shard_end - shard_start, selected_values.shape[-1]),
                dtype=selected_values.dtype,
            )
            mask = torch.zeros_like(values_tensor, dtype=torch.bool)
            values_tensor[local_grid_indices] = torch.nan_to_num(selected_values)
            mask[local_grid_indices] = torch.isfinite(selected_values)
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
                "shard_node_count": shard_end - shard_start,
                "shard_native_index_range": [shard_start, shard_end],
                "native_node_count": int(reader.grid_size),
                "context_bbox_degrees": context_bbox,
            }

        if not inputs:
            msg = f"No source has data available at forecast origin {origin} for target {valid_time}."
            raise ValueError(msg)

        target_nodes = self._bbox_mask(target_reader, bbox)
        target_indices = np.flatnonzero(target_nodes)
        target_shard_start, target_shard_end = self._grid_shard(
            field.dataset,
            target_reader.grid_size,
        )
        target_indices = target_indices[
            (target_indices >= target_shard_start) & (target_indices < target_shard_end)
        ]
        local_target_indices = target_indices - target_shard_start
        if len(fields) == 1:
            target_variable_index = target_reader.name_to_index[field.field_name]
            target_native = np.asarray(
                target_reader.data[
                    slice(target_position, target_position + 1),
                    slice(target_variable_index, target_variable_index + 1),
                    :,
                    slice(target_shard_start, target_shard_end),
                ],
            )
            target = torch.from_numpy(target_native)[0, 0, 0, local_target_indices].float()[None]
        else:
            # A bundled domain step intentionally materialises every enabled
            # target field once on this rank's native-grid shard.
            target_native = target_reader.get_sample(
                0,
                np.asarray([target_position]),
                slice(target_shard_start, target_shard_end),
            )[0, 0]
            variable_indices = [target_reader.name_to_index[item.field_name] for item in fields]
            target = target_native[local_target_indices][:, variable_indices].float().T
        means = torch.tensor([item.mean for item in fields], dtype=target.dtype)[:, None]
        stdevs = torch.tensor([item.stdev for item in fields], dtype=target.dtype)[:, None]
        target = (target - means) / stdevs
        target_mask = torch.isfinite(target)
        target_latitudes = np.asarray(target_reader.data.latitudes)[target_indices]
        target_longitudes = (np.asarray(target_reader.data.longitudes)[target_indices] + 180) % 360 - 180
        output_coordinates = torch.deg2rad(
            torch.from_numpy(
                np.stack((target_latitudes, target_longitudes), axis=-1),
            ).float(),
        )
        query_metadata = np.stack(
            [
                self.catalogue.encode_metadata(item, lead_time.total_seconds() / 3600)
                for item in queries
            ],
        )
        target_geometry = self.catalogue.datasets[field.dataset]
        south, north = target_geometry["latitude_bounds_degrees"]
        west, east = target_geometry["longitude_bounds_degrees"]
        embedded_bbox = query.bbox or (west, south, east, north)
        loss_weight = torch.tensor(
            [
                self.task.loss_weights.get(item.variable, 1.0)
                * self.task.loss_weights.get(item.provenance, 1.0)
                * self.task.loss_weights.get(f"{item.variable}@{item.provenance}", 1.0)
                for item in fields
            ],
            dtype=torch.float32,
        )
        baseline_source = None
        baseline_column = None
        baseline_indices = None
        baseline_multiplier = None
        baseline_offset = None
        configured_baseline = self.task.residual_baseline_source
        if configured_baseline is not None and lead_time == timedelta(0) and configured_baseline in inputs:
            resolved = input_context[configured_baseline]["resolved_fields"]
            columns = []
            multipliers = []
            offsets = []
            matched_any = False
            for target_field in fields:
                matches = [
                    (column, source_field)
                    for column, source_field in enumerate(resolved)
                    if source_field["time_offset_hours"] == 0.0
                    and self._same_physical_field(source_field, target_field)
                    and (source_field.get("units") or "unknown") == (target_field.units or "unknown")
                ]
                if len(matches) > 1:
                    msg = f"Several exact residual baselines match target {target_field}."
                    raise ValueError(msg)
                if not matches:
                    columns.append(-1)
                    multipliers.append(0.0)
                    offsets.append(0.0)
                    continue
                column, source_field = matches[0]
                conversion = 1.0
                if target_field.variable == "tp":
                    source_scale = self.task.precipitation_unit_scale_to_mm.get(configured_baseline)
                    target_scale = self.task.precipitation_unit_scale_to_mm.get(target_field.provenance)
                    if source_scale is None or target_scale is None or source_scale <= 0 or target_scale <= 0:
                        msg = (
                            "Precipitation residual baselines require positive "
                            "precipitation_unit_scale_to_mm entries for source and target."
                        )
                        raise ValueError(msg)
                    conversion = float(source_scale) / float(target_scale)
                matched_any = True
                columns.append(column)
                multipliers.append(float(source_field["stdev"]) * conversion / float(target_field.stdev))
                offsets.append(
                    (float(source_field["mean"]) * conversion - float(target_field.mean))
                    / float(target_field.stdev),
                )
            if matched_any:
                baseline_source = configured_baseline
                baseline_column = torch.tensor(columns, dtype=torch.long)
                baseline_indices = torch.from_numpy(
                    self._nearest_baseline_indices(
                        field.dataset,
                        target_indices,
                        target_shard_start,
                        target_shard_end,
                        bbox,
                    ),
                )
                baseline_multiplier = torch.tensor(multipliers, dtype=torch.float32)
                baseline_offset = torch.tensor(offsets, dtype=torch.float32)
        return QueryBatch(
            inputs=inputs,
            query_metadata=torch.from_numpy(query_metadata),
            query_variable_id=torch.tensor(
                [self.catalogue.variable_to_id[item.variable] for item in fields],
            ),
            query_provenance_id=torch.tensor(
                [self.catalogue.provenance_to_id[item.provenance] for item in fields],
            ),
            query_unit_id=torch.tensor(
                [self.catalogue.unit_to_id[item.units or "unknown"] for item in fields],
            ),
            query_grid_id=torch.tensor(
                [self.catalogue.grid_to_id[item.dataset] for item in fields],
            ),
            target_dataset=field.dataset,
            query=query.as_serialisable_dict(),
            output_coordinates=output_coordinates,
            target=torch.nan_to_num(target),
            target_mask=target_mask,
            loss_weight=loss_weight,
            residual_baseline_source=baseline_source,
            residual_baseline_input_column=baseline_column,
            residual_baseline_indices=baseline_indices,
            residual_baseline_multiplier=baseline_multiplier,
            residual_baseline_offset=baseline_offset,
            diagnostic_context={
                "sample_index": int(index),
                "full_valid_time_pass": bool(self.full_valid_time_pass),
                "full_unique_sample_count": self.full_sample_count,
                "sample_group_id": int(self.sample_group_id),
                "sample_group_count": int(self.sample_group_count),
                "forecast_origin": str(origin),
                "valid_time": str(valid_time),
                "requested_query": query.as_serialisable_dict(),
                "resolved_query": {
                    **query.as_serialisable_dict(),
                    "target_dataset": field.dataset,
                    "target_field_name": field.field_name,
                    "target_units": field.units,
                    "embedded_bbox": embedded_bbox,
                    "embedded_grid": field.dataset,
                },
                "target": {
                    **asdict(field),
                    "field_shape": list(target_reader.data.field_shape),
                    "native_index_count": len(target_indices),
                    "native_index_min": int(target_indices.min()),
                    "native_index_max": int(target_indices.max()),
                    "valid_node_count": int(target_mask.sum()),
                    "selected_node_count": len(target_indices),
                    "shard_node_count": target_shard_end - target_shard_start,
                    "shard_native_index_range": [target_shard_start, target_shard_end],
                },
                "bundle_fields": [asdict(item) for item in fields],
                "bundle_size": len(fields),
                "inputs": input_context,
                "configured_sources": all_source_names,
                "dataset_sampling_weights": dict(self.sampling_weights),
                "active_sources": list(inputs),
                "residual_baseline": {
                    "source": baseline_source,
                    "input_column": None if baseline_column is None else baseline_column.tolist(),
                    "matched": baseline_source is not None,
                },
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
