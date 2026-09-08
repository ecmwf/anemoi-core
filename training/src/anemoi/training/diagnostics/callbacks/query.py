# (C) Copyright 2026 Anemoi contributors.

"""Bounded diagnostics for query-based forecasting."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from anemoi.training.diagnostics.callbacks.plot import BasePlotCallback
from anemoi.training.diagnostics.evaluation.geospatial.maps import Borders
from anemoi.training.diagnostics.evaluation.geospatial.maps import Coastlines
from anemoi.training.diagnostics.evaluation.geospatial.projections import MapProjection
from anemoi.training.query.catalogue import QueryCatalogue
from anemoi.training.query.query import ForecastQuery

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0

if TYPE_CHECKING:
    import pytorch_lightning as pl


def _cpu(value: torch.Tensor | None) -> np.ndarray | None:
    return None if value is None else value.detach().float().cpu().numpy()


def _indices(value: torch.Tensor) -> np.ndarray:
    """Move indices to CPU without the precision loss of a float conversion."""
    return value.detach().cpu().numpy().astype(np.int64, copy=False)


def _bounded(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    # A stride through storage order creates false scan lines on structured and
    # reduced grids. This isolated generator is deterministic and cannot alter
    # either the sampler or model random-number streams.
    rng = np.random.default_rng(size * 1_000_003 + limit)
    return np.sort(rng.choice(size, limit, replace=False))


def _degrees(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _cpu(value) if isinstance(value, torch.Tensor) else np.asarray(value)
    result = np.rad2deg(array)
    result[:, 1] = (result[:, 1] + 180) % 360 - 180
    return result


def _hours(value: str) -> float:
    return float(value[:-1]) / (3600 if value.endswith("s") else 1)


def _cosine(values: np.ndarray) -> np.ndarray:
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float32).eps)
    return values @ values.T


def _distance_km(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    dlat = target[:, 0] - source[:, 0]
    dlon = target[:, 1] - source[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(source[:, 0]) * np.cos(target[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1, np.sqrt(a)))


class QueryDiagnosticsPlot(BasePlotCallback):
    """Inspect the examples and tensors actually used by ``QueryTraining``."""

    def __init__(
        self,
        enabled: bool = False,
        every_n_epochs: int = 1,
        max_cases: int = 2,
        fixed_validation_cases: list[int] | None = None,
        changing_training_case: bool = False,
        domain_plots: bool = True,
        graph_plots: bool = True,
        input_plots: bool = True,
        embedding_plots: bool = True,
        sensitivity_plots: bool = True,
        sampler_plots: bool = True,
        lead_time_hours: list[float] | None = None,
        pressure_levels_hpa: list[float] | None = None,
        target_provenances: list[str] | None = None,
        omit_metadata: list[str] | None = None,
        max_points: int = 20_000,
        max_edges: int = 800,
        max_embedding_items: int = 24,
        max_sweep_values: int = 4,
        dataset_names: list[str] | None = None,
        plotting_settings: Any = None,
    ) -> None:
        super().__init__(dataset_names=dataset_names, plotting_settings=plotting_settings)
        self.enabled = enabled
        self.every_n_epochs = every_n_epochs
        self.max_cases = max_cases
        self.fixed_validation_cases = set([0] if fixed_validation_cases is None else fixed_validation_cases)
        self.changing_training_case = changing_training_case
        self.domain_plots = domain_plots
        self.graph_plots = graph_plots
        self.input_plots = input_plots
        self.embedding_plots = embedding_plots
        self.sensitivity_plots = sensitivity_plots
        self.sampler_plots = sampler_plots
        self.lead_time_hours = lead_time_hours or []
        self.pressure_levels_hpa = pressure_levels_hpa
        self.target_provenances = target_provenances
        self.omit_metadata = omit_metadata or []
        self.max_points = max_points
        self.max_edges = max_edges
        self.max_embedding_items = max_embedding_items
        self.max_sweep_values = max_sweep_values
        self._captured_cases = 0
        self._captured_training_epoch = -1
        self._seen_validation_cases: set[int] = set()
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._metrics: dict[tuple[str, ...], list[float]] = {}
        self._dataset_sampling_weights: dict[str, float] = {}
        self._pca_mean: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._pca_fit_epoch: int | None = None

    @property
    def artifact_subfolder(self) -> str:
        return "query_diagnostics"

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        if self.enabled:
            if not hasattr(pl_module, "enable_query_diagnostics"):
                msg = "QueryDiagnosticsPlot requires QueryTraining."
                raise TypeError(msg)
            pl_module.enable_query_diagnostics()

    def _due(self, trainer: pl.Trainer) -> bool:
        return self.enabled and not trainer.sanity_checking and trainer.current_epoch % self.every_n_epochs == 0

    def on_train_epoch_start(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        if self._due(trainer):
            self._captured_cases = 0
            self._seen_validation_cases.clear()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        if (
            self.changing_training_case
            and self._due(trainer)
            and trainer.is_global_zero
            and batch_idx == 0
            and self._captured_training_epoch != trainer.current_epoch
        ):
            pl_module.request_query_diagnostic_prediction()

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        index = int((batch.diagnostic_context or {}).get("sample_index", -1))
        if (
            self._due(trainer)
            and trainer.is_global_zero
            and self._captured_cases < self.max_cases
            and index in self.fixed_validation_cases
        ):
            self._seen_validation_cases.add(index)
            pl_module.request_query_diagnostic_prediction()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _outputs: Any,
        batch: Any,
        _batch_idx: int,
    ) -> None:
        if not self.enabled:
            return
        step = pl_module.pop_query_diagnostics_step()
        if step is None:
            return
        self._record(batch, step)
        if step["prediction"] is not None and trainer.is_global_zero:
            self.plot(trainer, self._prepare_case(pl_module, batch, step, "train"))
            self._captured_training_epoch = trainer.current_epoch

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _outputs: Any,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        if not self.enabled:
            return
        step = pl_module.pop_query_diagnostics_step()
        if step is not None and step["prediction"] is not None and trainer.is_global_zero:
            self.plot(trainer, self._prepare_case(pl_module, batch, step, "val"))
            self._captured_cases += 1

    def _record(self, batch: Any, step: dict[str, Any]) -> None:
        query, context = batch.query, batch.diagnostic_context or {}
        self._dataset_sampling_weights.update(context.get("dataset_sampling_weights", {}))
        target = context.get("target", {})
        if query.get("level") is not None:
            suffix = "" if query.get("level_unit") is None else f" {query['level_unit']}"
            level = f"{query.get('model_type')} {query['level']}{suffix}"
        else:
            level = str(query.get("model_type") or "sfc")
        variable, provenance, lead = (
            str(query["variable"]),
            str(query["provenance"]),
            f"{_hours(query['lead_time']):g} h",
        )
        for name, key in (("variable", variable), ("provenance", provenance), ("lead", lead), ("level", level)):
            self._counts[name][key] += 1
        self._counts["joint"][(provenance, variable)] += 1
        self._counts["entered"]["samples"] += 1
        self._counts["valid_target"]["samples"] += int(float(step["valid_target_count"].cpu()) > 0)
        for source in context.get("active_sources", []):
            self._counts["active_source"][source] += 1
        for source in context.get("source_dropout_sources", []):
            self._counts["source_dropout"][source] += 1
        for source, details in context.get("inputs", {}).items():
            self._counts["field_available"][source] += int(details.get("available_field_count", 0))
            self._counts["field_selected"][source] += int(details.get("selected_field_count", 0))
            self._counts["history_available"][source] += int(details.get("available_time_count", 0))
            self._counts["history_selected"][source] += int(details.get("selected_time_count", 0))
        key = (variable, provenance, lead, str(target.get("units") or "unit unknown"))
        values = self._metrics.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        values[0] += float(step["normalized_mse"].cpu())
        values[1] += float(step["physical_rmse"].cpu())
        values[2] += float(step["loss_weight"].cpu())
        values[3] += 1

    def on_train_epoch_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        if not self._due(trainer) or not self.sampler_plots:
            return
        local = {"counts": {key: dict(value) for key, value in self._counts.items()}, "metrics": self._metrics}
        gathered: list[Any] = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        if trainer.is_global_zero:
            self.plot(trainer, self._sampler_payload(gathered, trainer))
        self._counts.clear()
        self._metrics.clear()

    def on_validation_epoch_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        if self._due(trainer) and trainer.is_global_zero:
            missing = self.fixed_validation_cases - self._seen_validation_cases
            if missing:
                LOGGER.warning(
                    "Fixed validation sample indices %s were not assigned to global rank zero; "
                    "choose rank-zero indices or use one rank.",
                    sorted(missing),
                )

    def _sampler_payload(self, gathered: list[dict[str, Any]], trainer: pl.Trainer) -> dict[str, Any]:
        counts: dict[str, Counter] = defaultdict(Counter)
        metrics: dict[tuple[str, ...], list[float]] = {}
        for rank in gathered:
            for key, value in rank["counts"].items():
                counts[key].update(value)
            for key, value in rank["metrics"].items():
                total = metrics.setdefault(tuple(key), [0.0] * 4)
                for index in range(4):
                    total[index] += value[index]
        task = trainer.lightning_module.task
        return {
            "kind": "sampler",
            "epoch": trainer.current_epoch,
            "counts": {key: dict(value) for key, value in counts.items()},
            "metrics": metrics,
            "weights": {
                "variable": dict(task.variable_weights),
                "provenance": dict(task.provenance_weights),
                "dataset_sampling": dict(self._dataset_sampling_weights),
            },
            "ranks": len(gathered),
        }

    def _prepare_case(
        self,
        pl_module: pl.LightningModule,
        batch: Any,
        step: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        model, context = pl_module.model, batch.diagnostic_context or {}
        target_info = context.get("target", {})
        mean, stdev = float(target_info.get("mean", 0)), float(target_info.get("stdev", 1))
        payload = {
            "kind": "case",
            "stage": stage,
            "epoch": int(pl_module.current_epoch),
            "sample_index": int(context.get("sample_index", -1)),
            "query": dict(batch.query),
            "context": context,
            "prediction": _cpu(step["prediction"])[0] * stdev + mean,
            "target": None if batch.target is None else _cpu(batch.target)[0] * stdev + mean,
            "target_mask": None if batch.target_mask is None else _cpu(batch.target_mask)[0].astype(bool),
            "coordinates": _degrees(batch.output_coordinates),
            "input": self._representative_input(model, batch, context),
            "input_details": self._input_payload(model, batch, context) if self.input_plots else None,
            "graph": None,
            "embeddings": None,
            "sensitivity": None,
        }
        if self.domain_plots or self.graph_plots:
            payload["graph"] = self._graph_payload(model, batch, context)
        if self.embedding_plots or (self.sensitivity_plots and stage == "val"):
            catalogue = QueryCatalogue.from_snapshot(model.query_catalogue)
            variants = self._variants(catalogue, batch.query, pl_module.task)
            if self.embedding_plots:
                payload["embeddings"] = self._embedding_payload(
                    model,
                    batch,
                    catalogue,
                    variants,
                    int(pl_module.current_epoch),
                )
            if self.sensitivity_plots and stage == "val":
                payload["sensitivity"] = self._sensitivity_payload(
                    model,
                    batch,
                    payload["prediction"],
                    catalogue,
                    variants,
                    pl_module.model_comm_group,
                )
        return payload

    def _input_payload(self, model: Any, batch: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Collect exact adapter inputs and a bounded 2 m-temperature geometry view."""
        graph, hidden_name = model._graph_data, model.hidden_name
        hidden_rad = _cpu(model.node_attributes.get_coordinates(hidden_name))
        hidden = _degrees(hidden_rad)
        selected_hidden = _bounded(len(hidden), self.max_points)
        hidden_lookup = np.full(len(hidden), -1, dtype=np.int64)
        hidden_lookup[selected_hidden] = np.arange(len(selected_hidden))

        temperature_sources = []
        for source, tensor in batch.inputs.items():
            fields = context.get("inputs", {}).get(source, {}).get("resolved_fields", [])
            candidates = [
                (column, field)
                for column, field in enumerate(fields)
                if field.get("variable") == "t" and field.get("level_type") == "height" and field.get("height_m") == 2
            ]
            if not candidates:
                continue
            column, field = max(candidates, key=lambda item: float(item[1].get("time_offset_hours", -math.inf)))
            coordinates_rad = _cpu(model.node_attributes.get_coordinates(source))
            normalized = _cpu(tensor.values)[0, :, column]
            valid = _cpu(tensor.mask)[0, :, column].astype(bool)
            physical = normalized * float(field["stdev"]) + float(field["mean"])
            edges = _indices(graph[(source, "to", hidden_name)].edge_index)
            relevant = valid[edges[0]] & (hidden_lookup[edges[1]] >= 0)
            if relevant.any():
                source_nodes = edges[0, relevant]
                hidden_nodes = edges[1, relevant]
                distances = _distance_km(coordinates_rad[source_nodes], hidden_rad[hidden_nodes])
                local_nodes = hidden_lookup[hidden_nodes]
                order = np.lexsort((distances, local_nodes))
                _, first = np.unique(local_nodes[order], return_index=True)
                chosen = order[first]
                mapping = {
                    "hidden_indices": local_nodes[chosen],
                    "source_indices": source_nodes[chosen],
                    "distance_km": distances[chosen],
                }
            else:
                mapping = {
                    "hidden_indices": np.empty(0, dtype=np.int64),
                    "source_indices": np.empty(0, dtype=np.int64),
                    "distance_km": np.empty(0, dtype=np.float32),
                }
            temperature_sources.append(
                {
                    "source": source,
                    "field": field,
                    "coordinates": _degrees(coordinates_rad),
                    "physical": physical,
                    "valid": valid,
                    "mapping": mapping,
                },
            )

        skipped_temperature_sources = []
        if temperature_sources:
            unit_counts = Counter(source["field"].get("units") or "unknown" for source in temperature_sources)
            shared_unit = unit_counts.most_common(1)[0][0]
            skipped_temperature_sources = [
                source["source"]
                for source in temperature_sources
                if (source["field"].get("units") or "unknown") != shared_unit
            ]
            temperature_sources = [
                source for source in temperature_sources if (source["field"].get("units") or "unknown") == shared_unit
            ]

        mapped = np.full(len(selected_hidden), np.nan, dtype=np.float32)
        mapped_distance = np.full(len(selected_hidden), np.inf, dtype=np.float32)
        mapped_source = np.full(len(selected_hidden), -1, dtype=np.int64)
        for source_index, source in enumerate(temperature_sources):
            mapping = source["mapping"]
            locations = mapping["hidden_indices"]
            use = mapping["distance_km"] < mapped_distance[locations]
            locations = locations[use]
            mapped[locations] = source["physical"][mapping["source_indices"][use]]
            mapped_distance[locations] = mapping["distance_km"][use]
            mapped_source[locations] = source_index
        mapped_distance[~np.isfinite(mapped)] = np.nan

        ifs = next((name for name in batch.inputs if name.upper() == "IFS" or "IFS" in name.upper()), None)
        ifs_field = None
        if ifs is not None:
            tensor = batch.inputs[ifs]
            fields = context.get("inputs", {}).get(ifs, {}).get("resolved_fields", [])
            coordinates = _degrees(model.node_attributes.get_coordinates(ifs))
            context_bbox = context.get("inputs", {}).get(ifs, {}).get("context_bbox_degrees")
            if context_bbox is None:
                selected_geometry = np.ones(len(coordinates), dtype=bool)
            else:
                west, south, east, north = context_bbox
                selected_geometry = (
                    (coordinates[:, 1] >= west)
                    & (coordinates[:, 1] <= east)
                    & (coordinates[:, 0] >= south)
                    & (coordinates[:, 0] <= north)
                )
            normalized = _cpu(tensor.values)[0]
            validity = _cpu(tensor.mask)[0].astype(bool)
            if fields:
                missing_counts = [
                    int(np.sum(selected_geometry & ~validity[:, column])) for column in range(len(fields))
                ]
                column = int(np.argmax(missing_counts))
                field = fields[column]
                physical = normalized[:, column] * float(field["stdev"]) + float(field["mean"])
                ifs_field = {
                    "source": ifs,
                    "field": field,
                    "coordinates": coordinates,
                    "normalized": normalized[:, column],
                    "physical": physical,
                    "valid": validity[:, column],
                    "selected_geometry": selected_geometry,
                    "missing_count": missing_counts[column],
                    "selected_count": int(selected_geometry.sum()),
                    "selected_field_count": int(
                        context.get("inputs", {}).get(ifs, {}).get("selected_field_count", len(fields)),
                    ),
                    "available_field_count": int(
                        context.get("inputs", {}).get(ifs, {}).get("available_field_count", len(fields)),
                    ),
                }
        return {
            "hidden": hidden[selected_hidden],
            "bbox": (context.get("requested_query") or {}).get("bbox"),
            "temperature_sources": temperature_sources,
            "skipped_temperature_sources": skipped_temperature_sources,
            "mapped_temperature": mapped,
            "mapped_distance_km": mapped_distance,
            "mapped_source": mapped_source,
            "ifs_field": ifs_field,
        }

    def _representative_input(self, model: Any, batch: Any, context: dict[str, Any]) -> dict[str, Any] | None:
        target, candidates = context.get("target", {}), []
        for source in batch.inputs:
            fields = context.get("inputs", {}).get(source, {}).get("resolved_fields", [])
            for column, field in enumerate(fields):
                exact = all(
                    field.get(key) == target.get(key)
                    for key in ("variable", "level_type", "pressure_pa", "height_m", "aggregation_type")
                )
                candidates.append((exact, float(field.get("time_offset_hours", -math.inf)), source, column, field))
        if not candidates:
            return None
        exact, _offset, source, column, field = max(candidates)
        tensor = batch.inputs[source]
        return {
            "source": source,
            "field": field,
            "coordinates": _degrees(model.node_attributes.get_coordinates(source)),
            "values": _cpu(tensor.values)[0, :, column] * float(field["stdev"]) + float(field["mean"]),
            "mask": _cpu(tensor.mask)[0, :, column].astype(bool),
            "matches_target": bool(exact and field.get("units") == target.get("units")),
        }

    def _graph_payload(self, model: Any, batch: Any, context: dict[str, Any]) -> dict[str, Any]:
        graph, hidden_name = model._graph_data, model.hidden_name
        hidden_rad = _cpu(model.node_attributes.get_coordinates(hidden_name))
        hidden = np.rad2deg(hidden_rad)
        hidden[:, 1] = (hidden[:, 1] + 180) % 360 - 180
        sources, encoder_edges, nearest = {}, {}, {}
        for name in model.dataset_names:
            coordinates_rad = _cpu(model.node_attributes.get_coordinates(name))
            coordinates = np.rad2deg(coordinates_rad)
            coordinates[:, 1] = (coordinates[:, 1] + 180) % 360 - 180
            active = None if name not in batch.inputs else _cpu(batch.inputs[name].mask)[0].any(axis=-1)
            sources[name] = {"coordinates": coordinates, "coordinates_rad": coordinates_rad, "active": active}
            if active is None:
                continue
            edges = _indices(graph[(name, "to", hidden_name)].edge_index)
            encoder_edges[name] = edges
            usable = active[edges[0]]
            distance = np.full(len(hidden), np.nan, dtype=np.float32)
            if usable.any():
                selected = edges[:, usable]
                values = _distance_km(coordinates_rad[selected[0]], hidden_rad[selected[1]])
                distance.fill(np.inf)
                np.minimum.at(distance, selected[1], values)
                distance[np.isinf(distance)] = np.nan
            nearest[name] = distance
        decoder = None
        if batch.output_coordinates is not None and len(batch.output_coordinates):
            neighbours = model.dynamic_decoder_graph_provider.neighbours
            count = min(len(batch.output_coordinates), max(1, self.max_edges // neighbours))
            output_indices = torch.as_tensor(
                _bounded(len(batch.output_coordinates), count),
                dtype=torch.long,
                device=batch.output_coordinates.device,
            )
            _, edges = model.dynamic_decoder_graph_provider.build_graph(
                model.node_attributes.get_coordinates(hidden_name),
                batch.output_coordinates[output_indices],
            )
            decoder = {"edges": _indices(edges), "output_indices": _indices(output_indices)}
        hidden_masks = {
            str(key): _cpu(graph[hidden_name][key]).reshape(-1).astype(bool)
            for key in graph[hidden_name]
            if str(key).startswith("query_source_coverage_")
        }
        identity = getattr(graph, "query_geometry", {})
        return {
            "hidden": hidden,
            "sources": sources,
            "encoder_edges": encoder_edges,
            "nearest": nearest,
            "decoder": decoder,
            "hidden_masks": hidden_masks,
            "output": _degrees(batch.output_coordinates),
            "output_mask": _cpu(batch.target_mask)[0].astype(bool),
            "bbox": (context.get("requested_query") or {}).get("bbox"),
            "identity": identity,
            "identity_hash": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12],
            "cache": str(model.config.system.input.graph),
        }

    def _variants(  # noqa: C901
        self,
        catalogue: QueryCatalogue,
        query_value: dict[str, Any],
        task: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        base, groups = ForecastQuery.from_dict(query_value), defaultdict(list)

        def add(kind: str, label: str, value: dict[str, Any], status: str) -> None:
            try:
                query = ForecastQuery.from_dict(value)
                mean, stdev = catalogue.normalization(query)
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping unsupported diagnostic request %s/%s: %s", kind, label, exc)
                groups[kind].append({"label": label, "skip": str(exc), "status": "unsupported"})
                return
            try:
                field = catalogue.resolve(query)
            except KeyError:
                field = None
            target_weight = task.variable_weights.get(query.variable, 1.0) * task.provenance_weights.get(
                query.provenance,
                1.0,
            )
            if field is not None and target_weight > 0:
                status = "training-supervised"
            elif field is not None and target_weight <= 0:
                status = "catalogued but target sampling weight is zero"
            groups[kind].append(
                {
                    "label": label,
                    "query": query,
                    "mean": mean,
                    "stdev": stdev,
                    "units": None if field is None else field.units,
                    "status": status,
                },
            )

        trained_leads = {value.total_seconds() / 3600 for value in task.lead_times}
        for value in self.lead_time_hours[: self.max_sweep_values]:
            query = dict(query_value)
            query["lead_time"] = f"{value:g}h"
            add(
                "lead",
                f"lead {value:g} h",
                query,
                "supervised" if value in trained_leads else "unsupported extrapolation",
            )
        pressure_levels = self.pressure_levels_hpa
        if pressure_levels is None and base.level_type == "pressure":
            pressure_levels = sorted(
                {
                    field.pressure_pa / 100
                    for field in catalogue.target_fields
                    if field.variable == base.variable
                    and field.provenance == base.provenance
                    and field.level_type == "pressure"
                    and field.pressure_pa is not None
                },
            )
        for value in (pressure_levels or [])[: self.max_sweep_values]:
            query = dict(query_value)
            query.update(model_type="pl", level=float(value), level_unit="hPa")
            add("pressure", f"{value:g} hPa", query, "interpolated/held-out level")
        provenances = self.target_provenances
        if provenances is None:
            provenances = sorted(
                {
                    field.provenance
                    for field in catalogue.target_fields
                    if field.variable == base.variable
                    and field.level_type == base.level_type
                    and field.pressure_pa == base.pressure_pa
                    and field.model_level == base.model_level
                    and field.height_m == base.height_m
                    and field.aggregation_type == base.aggregation_type
                },
            )
        for value in provenances[: self.max_sweep_values]:
            query = dict(query_value)
            query["provenance"] = value
            add("provenance", value, query, "supervised")
        allowed = {"grid_spacing_km", "spatial_support_km", "output_frequency"}
        for key in self.omit_metadata[: self.max_sweep_values]:
            if key not in allowed:
                groups["omission"].append({"label": f"omit {key}", "skip": "metadata key is not optional"})
                continue
            if query_value.get(key) is None:
                groups["omission"].append(
                    {"label": f"omit {key}", "skip": "metadata is already omitted in the resolved base query"},
                )
                continue
            query = dict(query_value)
            query[key] = None
            observed = any(
                (key == "grid_spacing_km" and field.resolution_km is None)
                or (key == "spatial_support_km" and field.spatial_support_km is None)
                for field in catalogue.target_fields
            )
            status = "training-observed missing state" if observed else "unsupported missing-state extrapolation"
            add("omission", f"omit {key}", query, status)
        return dict(groups)

    def _embedding_payload(
        self,
        model: Any,
        batch: Any,
        catalogue: QueryCatalogue,
        variants: dict[str, list[dict[str, Any]]],
        epoch: int,
    ) -> dict[str, Any]:
        variable_labels = catalogue.variables[: self.max_embedding_items]
        provenance_labels = catalogue.provenances[: self.max_embedding_items]
        unit_labels = catalogue.units[: self.max_embedding_items]
        query_stages = model.query_adapter.diagnostic_stages(
            batch.query_metadata,
            batch.query_variable_id,
            batch.query_provenance_id,
            batch.query_unit_id,
        )
        input_stages = {}
        for source, tensor in batch.inputs.items():
            nodes = torch.nonzero(tensor.mask[0].any(dim=-1), as_tuple=False).flatten()
            if not len(nodes):
                continue
            selected = nodes[torch.as_tensor(_bounded(len(nodes), min(128, self.max_points)), device=nodes.device)]
            stages = model.value_adapter.diagnostic_stages(
                tensor.values,
                tensor.metadata,
                tensor.variable_ids,
                tensor.provenance_ids,
                tensor.unit_ids,
                tensor.mask,
                selected,
            )
            input_stages[source] = {key: _cpu(value) for key, value in stages.items()}

        lead = _hours(batch.query["lead_time"])
        fields = sorted(
            catalogue.target_fields,
            key=lambda field: (field.variable, field.provenance, field.level_type, field.pressure_pa or -1),
        )[: self.max_embedding_items]
        metadata, variable_ids, provenance_ids, labels = [], [], [], []
        for field in fields:
            query = ForecastQuery(
                variable=field.variable,
                lead_time=timedelta(hours=lead),
                provenance=field.provenance,
                unit=field.units or "unknown",
                output_frequency=None if field.cadence_hours is None else timedelta(hours=field.cadence_hours),
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
                temporal_aggregation_window=None
                if field.temporal_aggregation_window_hours is None
                else tuple(timedelta(hours=value) for value in field.temporal_aggregation_window_hours),
                grid=field.dataset,
                grid_spacing_km=field.resolution_km,
                spatial_support_km=field.spatial_support_km,
            )
            metadata.append(catalogue.encode_metadata(query, lead))
            variable_ids.append(catalogue.variable_to_id[field.variable])
            provenance_ids.append(catalogue.provenance_to_id[field.provenance])
            level = f" {field.pressure_pa / 100:g}hPa" if field.pressure_pa else ""
            labels.append(f"{field.variable}@{field.provenance}{level}")
        device = next(model.parameters()).device
        final = _cpu(
            model.query_adapter(
                torch.as_tensor(np.stack(metadata), device=device),
                torch.tensor(variable_ids, device=device),
                torch.tensor(provenance_ids, device=device),
                torch.tensor(
                    [catalogue.unit_to_id[field.units or "unknown"] for field in fields],
                    device=device,
                ),
            ),
        )
        if self._pca_components is None and len(final) >= 2:
            self._pca_mean = final.mean(axis=0)
            _, _, vt = np.linalg.svd(final - self._pca_mean, full_matrices=False)
            self._pca_components = vt[:2]
            self._pca_fit_epoch = epoch
        sweeps = {}
        for kind, items in variants.items():
            valid = [item for item in items if "query" in item]
            if not valid:
                continue
            continuous = np.stack(
                [
                    catalogue.encode_metadata(item["query"], item["query"].lead_time.total_seconds() / 3600)
                    for item in valid
                ],
            )
            vectors = _cpu(
                model.query_adapter(
                    torch.as_tensor(continuous, device=device),
                    torch.tensor([catalogue.variable_to_id[item["query"].variable] for item in valid], device=device),
                    torch.tensor(
                        [catalogue.provenance_to_id[item["query"].provenance] for item in valid],
                        device=device,
                    ),
                    torch.tensor(
                        [catalogue.unit_to_id[item["query"].unit] for item in valid],
                        device=device,
                    ),
                ),
            )
            sweeps[kind] = {
                "labels": [item["label"] for item in valid],
                "status": [item["status"] for item in valid],
                "continuous": continuous,
                "final": vectors,
                "distance": np.linalg.norm(np.diff(vectors, axis=0), axis=1),
                "pca": self._project(vectors),
            }
        return {
            "variable_labels": variable_labels,
            "provenance_labels": provenance_labels,
            "unit_labels": unit_labels,
            "variable": _cpu(model.value_adapter.variable_embedding.weight[: len(variable_labels)]),
            "provenance": _cpu(model.value_adapter.provenance_embedding.weight[: len(provenance_labels)]),
            "unit": _cpu(model.value_adapter.unit_embedding.weight[: len(unit_labels)]),
            "query_stages": {key: _cpu(value) for key, value in query_stages.items()},
            "input_stages": input_stages,
            "labels": labels,
            "pca": self._project(final),
            "pca_fit_epoch": self._pca_fit_epoch,
            "sweeps": sweeps,
        }

    def _project(self, values: np.ndarray) -> np.ndarray:
        if self._pca_components is None or self._pca_mean is None:
            return np.zeros((len(values), 2), dtype=np.float32)
        return (values - self._pca_mean) @ self._pca_components.T

    @torch.inference_mode()
    def _sensitivity_payload(
        self,
        model: Any,
        batch: Any,
        base: np.ndarray,
        catalogue: QueryCatalogue,
        variants: dict[str, list[dict[str, Any]]],
        model_comm_group: Any,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        base_units = (batch.diagnostic_context or {}).get("target", {}).get("units")
        result = {}
        was_training = model.training
        model.eval()
        try:
            for kind, items in variants.items():
                group = {
                    "labels": [],
                    "status": [],
                    "predictions": [],
                    "rmse": [],
                    "skipped": [],
                    "units": base_units or "?",
                }
                for item in items:
                    if "query" not in item:
                        group["skipped"].append(f"{item['label']}: {item.get('skip', 'unsupported')}")
                        continue
                    if base_units and item["units"] and base_units != item["units"]:
                        group["skipped"].append(f"{item['label']}: units {item['units']} != {base_units}")
                        continue
                    query = item["query"]
                    encoded = {
                        "metadata": torch.as_tensor(
                            catalogue.encode_metadata(query, query.lead_time.total_seconds() / 3600),
                            device=device,
                        )[None],
                        "variable_id": torch.tensor([catalogue.variable_to_id[query.variable]], device=device),
                        "provenance_id": torch.tensor([catalogue.provenance_to_id[query.provenance]], device=device),
                        "unit_id": torch.tensor([catalogue.unit_to_id[query.unit]], device=device),
                        "grid": batch.target_dataset,
                    }
                    normalized = model._forward_tensors(
                        batch.inputs,
                        encoded,
                        batch.output_coordinates,
                        model_comm_group=model_comm_group,
                    )
                    physical = _cpu(normalized)[0] * item["stdev"] + item["mean"]
                    group["labels"].append(item["label"])
                    group["status"].append(item["status"])
                    group["predictions"].append(physical)
                    group["rmse"].append(float(np.sqrt(np.nanmean((physical - base) ** 2))))
                result[kind] = group
        finally:
            model.train(was_training)
        return result

    def _plot(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        if payload["kind"] == "sampler":
            self._plot_sampler(trainer, payload)
            return
        self._plot_example(trainer, payload)
        if self.input_plots and payload["input_details"] is not None:
            self._plot_stretched_temperature(trainer, payload)
            self._plot_ifs_input(trainer, payload)
        if self.domain_plots and payload["graph"] is not None:
            self._plot_domain(trainer, payload)
        if self.graph_plots and payload["graph"] is not None:
            self._plot_connectivity(trainer, payload)
        if self.embedding_plots and payload["embeddings"] is not None:
            self._plot_embeddings(trainer, payload)
            self._plot_sweeps(trainer, payload)
        if self.sensitivity_plots and payload["sensitivity"]:
            self._plot_sensitivity(trainer, payload)

    def _map(self, reference: np.ndarray, shape: tuple[int, int]) -> tuple[Any, np.ndarray, MapProjection, Any]:
        projection = MapProjection.from_kind(reference, self.plotting_settings.projection_kind)
        subplot_kw = {"projection": projection.axes_crs()} if projection.axes_crs() is not None else {}
        fig, axes = plt.subplots(*shape, figsize=(5 * shape[1], 3.5 * shape[0]), subplot_kw=subplot_kw)
        data_crs = None
        if projection.axes_crs() is not None:
            import cartopy.crs as ccrs

            data_crs = ccrs.PlateCarree()
        return fig, np.asarray(axes, dtype=object).reshape(shape), projection, data_crs

    @staticmethod
    def _xy(coordinates: np.ndarray, projection: MapProjection) -> tuple[np.ndarray, np.ndarray]:
        return (
            (coordinates[:, 1], coordinates[:, 0])
            if projection.axes_crs() is not None
            else projection.project(coordinates)
        )

    def _finish_map(
        self,
        ax: Any,
        projection: MapProjection,
        data_crs: Any,
        coordinates: np.ndarray,
        extent: list[float] | None = None,
    ) -> None:
        Coastlines(projection).plot_continents(ax, data_crs)
        Borders(projection).plot_borders(ax, data_crs)
        if extent is None:
            south, west = coordinates.min(axis=0)
            north, east = coordinates.max(axis=0)
            extent = [west, east, south, north]
        if data_crs is not None:
            if extent[1] - extent[0] < 340 and extent[3] - extent[2] < 150:
                ax.set_extent(extent, crs=data_crs)
        else:
            x, y = projection(np.asarray(extent[:2]), np.asarray(extent[2:]))
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y), max(y))
        ax.set_xticks([])
        ax.set_yticks([])

    def _scatter(
        self,
        fig: Any,
        ax: Any,
        coordinates: np.ndarray,
        values: np.ndarray,
        projection: MapProjection,
        data_crs: Any,
        title: str,
        *,
        mask: np.ndarray | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        valid = np.isfinite(values) if mask is None else mask & np.isfinite(values)
        available = np.flatnonzero(valid)
        selected = available[_bounded(len(available), self.max_points)]
        x, y = self._xy(coordinates[selected], projection)
        kwargs = {"c": values[selected], "s": 2, "cmap": cmap, "vmin": vmin, "vmax": vmax, "rasterized": True}
        if data_crs is not None:
            kwargs["transform"] = data_crs
        artist = ax.scatter(x, y, **kwargs)
        self._finish_map(ax, projection, data_crs, coordinates[selected] if len(selected) else coordinates)
        ax.set_title(title, fontsize=8)
        fig.colorbar(artist, ax=ax, shrink=0.7, pad=0.02)

    def _plot_example(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        coordinates, prediction = payload["coordinates"], payload["prediction"]
        target, mask = payload["target"], payload["target_mask"]
        fig, axes, projection, data_crs = self._map(coordinates, (2, 3))
        comparable = prediction[mask] if mask is not None else prediction
        if target is not None:
            comparable = np.concatenate((comparable, target[mask]))
        low, high = np.nanpercentile(comparable, [2, 98])
        input_ = payload["input"]
        if input_ is None:
            axes[0, 0].text(0.5, 0.5, "no plottable input", ha="center")
            axes[0, 0].axis("off")
        else:
            field = input_["field"]
            self._scatter(
                fig,
                axes[0, 0],
                input_["coordinates"],
                input_["values"],
                projection,
                data_crs,
                f"input {field['variable']} · {input_['source']} · {field['actual_time']}\n"
                f"{'matching semantics' if input_['matches_target'] else 'representative; not target-equivalent'}",
                mask=input_["mask"],
                vmin=low if input_["matches_target"] else None,
                vmax=high if input_["matches_target"] else None,
            )
        self._scatter(
            fig,
            axes[0, 1],
            coordinates,
            prediction,
            projection,
            data_crs,
            "prediction · physical units",
            vmin=low,
            vmax=high,
        )
        if target is None:
            axes[0, 2].text(0.5, 0.5, "NO REFERENCE TARGET", ha="center")
            axes[1, 0].text(0.5, 0.5, "error unavailable", ha="center")
            axes[0, 2].axis("off")
            axes[1, 0].axis("off")
        else:
            self._scatter(
                fig,
                axes[0, 2],
                coordinates,
                target,
                projection,
                data_crs,
                "exact matching target · physical units",
                mask=mask,
                vmin=low,
                vmax=high,
            )
            error = prediction - target
            limit = max(float(np.nanpercentile(np.abs(error[mask]), 98)), np.finfo(np.float32).eps)
            self._scatter(
                fig,
                axes[1, 0],
                coordinates,
                error,
                projection,
                data_crs,
                "prediction - target",
                mask=mask,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        if input_ is not None:
            self._scatter(
                fig,
                axes[1, 1],
                input_["coordinates"],
                input_["mask"].astype(float),
                projection,
                data_crs,
                "input validity (zero is valid)",
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
        self._scatter(
            fig,
            axes[1, 2],
            coordinates,
            mask.astype(float),
            projection,
            data_crs,
            "target validity / coverage",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
        context = payload["context"]
        query = context.get("resolved_query", payload["query"])
        requested = context.get("requested_query", {})
        times = "; ".join(
            f"{name} [{', '.join(value.get('actual_times', []))}]" for name, value in context.get("inputs", {}).items()
        )
        level = query.get("model_type")
        if query.get("level") is not None:
            level = f"{level}/{query.get('level')} {query.get('level_unit') or ''}".rstrip()
        consumed = payload["query"]
        differences = [
            f"{key}: requested={requested.get(key)!r} -> consumed={consumed.get(key)!r}"
            for key in sorted(set(requested) | set(consumed))
            if requested.get(key) != consumed.get(key)
        ]
        resolution_status = (
            "requested metadata equals consumed metadata"
            if not differences
            else "requested/resolved differences: " + "; ".join(differences)
        )
        annotation = (
            f"consumed: variable={query.get('variable')} level={level} provenance={query.get('provenance')} "
            f"lead={query.get('lead_time')} grid={query.get('grid')} bbox={query.get('bbox')}\n"
            f"origin={context.get('forecast_origin')} valid={context.get('valid_time')} "
            f"output_frequency={query.get('output_frequency')} "
            f"aggregation_type={query.get('aggregation_type')} "
            f"temporal_aggregation_window={query.get('temporal_aggregation_window')} unit={query.get('unit')}\n"
            f"inputs: {times}\n"
            f"{resolution_status} · native-node scatter; no interpolation"
        )
        fig.suptitle(f"Query example · {payload['stage']} sample {payload['sample_index']} · epoch {payload['epoch']}")
        fig.text(0.01, 0.01, annotation, fontsize=7, va="bottom")
        fig.tight_layout(rect=(0, 0.14, 1, 0.94))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_example_{payload['stage']}_case{payload['sample_index']:04d}",
            exp_log_tag="query/example",
        )

    def _plot_stretched_temperature(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        data = payload["input_details"]
        hidden = data["hidden"]
        sources = data["temperature_sources"]
        fig, axes, projection, data_crs = self._map(hidden, (2, 2))
        if not sources:
            for ax in axes.flat:
                ax.text(0.5, 0.5, "No 2 m temperature input in this sampled case", ha="center", va="center")
                ax.axis("off")
        else:
            valid_values = [source["physical"][source["valid"]] for source in sources if source["valid"].any()]
            low, high = (0.0, 1.0) if not valid_values else np.nanpercentile(np.concatenate(valid_values), [2, 98])
            for source in sources:
                valid = np.flatnonzero(source["valid"])
                valid = valid[_bounded(len(valid), max(1, self.max_points // len(sources)))]
                x, y = self._xy(source["coordinates"][valid], projection)
                kwargs = {
                    "c": source["physical"][valid],
                    "s": 2,
                    "cmap": "coolwarm",
                    "vmin": low,
                    "vmax": high,
                    "rasterized": True,
                    "label": f"{source['source']} · {source['field']['actual_time']}",
                }
                if data_crs is not None:
                    kwargs["transform"] = data_crs
                artist = axes[0, 0].scatter(x, y, **kwargs)
            fig.colorbar(
                artist,
                ax=axes[0, 0],
                shrink=0.7,
                pad=0.02,
                label=sources[0]["field"].get("units") or "unit unknown",
            )
            self._finish_map(axes[0, 0], projection, data_crs, hidden)
            axes[0, 0].legend(fontsize=5)
            axes[0, 0].set_title("actual native 2 m-temperature inputs")

            self._scatter(
                fig,
                axes[0, 1],
                hidden,
                data["mapped_temperature"],
                projection,
                data_crs,
                "diagnostic nearest-connected value on hidden nodes",
                cmap="coolwarm",
                vmin=low,
                vmax=high,
            )
            source_ids = data["mapped_source"]
            mapped = source_ids >= 0
            selected = np.flatnonzero(mapped)
            selected = selected[_bounded(len(selected), self.max_points)]
            x, y = self._xy(hidden[selected], projection)
            kwargs = {
                "c": source_ids[selected],
                "s": 2,
                "cmap": "tab10",
                "vmin": -0.5,
                "vmax": max(len(sources) - 0.5, 0.5),
                "rasterized": True,
            }
            if data_crs is not None:
                kwargs["transform"] = data_crs
            axes[1, 0].scatter(x, y, **kwargs)
            self._finish_map(axes[1, 0], projection, data_crs, hidden)
            labels = ", ".join(f"{index}={source['source']}" for index, source in enumerate(sources))
            axes[1, 0].set_title(f"nearest connected source · {labels}", fontsize=8)
            self._scatter(
                fig,
                axes[1, 1],
                hidden,
                data["mapped_distance_km"],
                projection,
                data_crs,
                "source-to-hidden edge distance [km]",
                cmap="magma",
            )
            for ax in axes.flat:
                self._bbox(ax, data["bbox"], projection, data_crs)
        fig.suptitle(
            "Stretched-grid 2 m temperature inspection · actual native values and actual encoder connectivity\n"
            "Hidden-node values are diagnostic nearest-edge assignments; the model pools all fields first"
            + (
                f" · incompatible-unit sources skipped: {', '.join(data['skipped_temperature_sources'])}"
                if data["skipped_temperature_sources"]
                else ""
            ),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_stretched_2m_temperature_case{payload['sample_index']:04d}",
            exp_log_tag="query/stretched_2m_temperature",
        )

    def _plot_ifs_input(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        data = payload["input_details"]["ifs_field"]
        if data is None:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.text(0.5, 0.5, "IFS is not active, or no IFS input field was selected in this case", ha="center")
            ax.axis("off")
        else:
            coordinates = data["coordinates"]
            fig, axes, projection, data_crs = self._map(coordinates, (2, 2))
            field = data["field"]
            selected = data["selected_geometry"]
            valid = data["valid"]
            self._scatter(
                fig,
                axes[0, 0],
                coordinates,
                data["normalized"],
                projection,
                data_crs,
                "exact normalized, zero-filled tensor before QueryValueAdapter",
                mask=selected,
                cmap="RdBu_r",
            )
            self._scatter(
                fig,
                axes[0, 1],
                coordinates,
                data["physical"],
                projection,
                data_crs,
                "physical values at valid input nodes",
                mask=selected & valid,
            )
            self._scatter(
                fig,
                axes[1, 0],
                coordinates,
                valid.astype(float),
                projection,
                data_crs,
                "validity mask entering QueryValueAdapter (zero data remains valid)",
                mask=selected,
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
            state = np.full(len(coordinates), -1.0)
            state[selected] = 0
            state[selected & valid] = 1
            self._scatter(
                fig,
                axes[1, 1],
                coordinates,
                state,
                projection,
                data_crs,
                "location state: -1 excluded, 0 missing, 1 valid",
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
            )
            fig.suptitle(
                f"IFS pre-network field · {field['field_name']} / {field['variable']} · {field['actual_time']} · "
                f"missing inside selected context={data['missing_count']:,}/{data['selected_count']:,}\n"
                f"selected fields={data['selected_field_count']}/{data['available_field_count']} · "
                "unselected fields are absent columns; invalid values retain a separate Boolean mask",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.91))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_ifs_input_case{payload['sample_index']:04d}",
            exp_log_tag="query/ifs_input",
        )

    @staticmethod
    def _bbox(ax: Any, bbox: list[float] | None, projection: MapProjection, data_crs: Any) -> None:
        if bbox is None:
            return
        west, south, east, north = bbox
        lon, lat = np.asarray([west, east, east, west, west]), np.asarray([south, south, north, north, south])
        if projection.axes_crs() is None:
            lon, lat = projection(lon, lat)
        kwargs = {"color": "tab:orange", "lw": 1.5}
        if data_crs is not None:
            kwargs["transform"] = data_crs
        ax.plot(lon, lat, **kwargs)

    def _plot_domain(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:  # noqa: C901
        graph, output = payload["graph"], payload["graph"]["output"]
        hidden, bbox = graph["hidden"], graph["bbox"]
        fig, axes, projection, data_crs = self._map(hidden, (2, 2))
        region = None if bbox is None else [bbox[0] - 3, bbox[2] + 3, bbox[1] - 3, bbox[3] + 3]
        hi, oi = _bounded(len(hidden), self.max_points), _bounded(len(output), self.max_points)
        hx, hy = self._xy(hidden[hi], projection)
        ox, oy = self._xy(output[oi], projection)
        hkw, okw = {"s": 1, "c": "0.65"}, {"s": 3, "c": np.where(graph["output_mask"][oi], "tab:green", "tab:red")}
        if data_crs is not None:
            hkw["transform"] = data_crs
            okw["transform"] = data_crs
        axes[0, 0].scatter(hx, hy, **hkw)
        axes[0, 0].scatter(ox, oy, **okw)
        self._bbox(axes[0, 0], bbox, projection, data_crs)
        self._finish_map(axes[0, 0], projection, data_crs, hidden)
        axes[0, 0].set_title(f"global hidden mesh ({len(hidden):,}) + output ({len(output):,})")
        colours = plt.get_cmap("tab10")
        for number, (name, source) in enumerate(graph["sources"].items()):
            if source["active"] is None:
                continue
            indices = np.flatnonzero(source["active"])
            indices = indices[_bounded(len(indices), max(1, self.max_points // len(graph["sources"])))]
            x, y = self._xy(source["coordinates"][indices], projection)
            kwargs = {
                "s": 2,
                "color": colours(number),
                "label": f"{name} ({source['active'].sum():,}/{len(source['active']):,})",
            }
            if data_crs is not None:
                kwargs["transform"] = data_crs
            axes[0, 1].scatter(x, y, **kwargs)
        self._bbox(axes[0, 1], bbox, projection, data_crs)
        self._finish_map(axes[0, 1], projection, data_crs, output, region)
        axes[0, 1].legend(fontsize=6)
        axes[0, 1].set_title("actual input coverage; context outside bbox retained")
        coverage = np.zeros(len(hidden), dtype=int)
        for value in graph["hidden_masks"].values():
            coverage += value
        kwargs = {"c": coverage[hi], "s": 2, "cmap": "viridis"}
        if data_crs is not None:
            kwargs["transform"] = data_crs
        artist = axes[1, 0].scatter(hx, hy, **kwargs)
        fig.colorbar(artist, ax=axes[1, 0], shrink=0.7, label="regional masks")
        self._finish_map(axes[1, 0], projection, data_crs, hidden, region)
        region_hidden = np.ones(len(hidden), dtype=bool)
        if bbox is not None:
            west, south, east, north = bbox
            region_hidden = (
                (hidden[:, 1] >= west) & (hidden[:, 1] <= east) & (hidden[:, 0] >= south) & (hidden[:, 0] <= north)
            )
        if not region_hidden.any():
            axes[1, 0].text(
                0.5,
                0.5,
                "0 hidden nodes inside requested bbox",
                transform=axes[1, 0].transAxes,
                ha="center",
            )
        axes[1, 0].set_title(
            f"hidden graph masks (0 reveals holes/exclusions) · bbox nodes={region_hidden.sum():,}",
        )
        source_name = next(iter(graph["nearest"]), None)
        if source_name is not None:
            distance = graph["nearest"][source_name]
            indices = np.flatnonzero(np.isfinite(distance))
            indices = indices[_bounded(len(indices), self.max_points)]
            x, y = self._xy(hidden[indices], projection)
            kwargs = {"c": distance[indices], "s": 2, "cmap": "magma"}
            if data_crs is not None:
                kwargs["transform"] = data_crs
            artist = axes[1, 1].scatter(x, y, **kwargs)
            fig.colorbar(artist, ax=axes[1, 1], shrink=0.7, label="km")
            self._finish_map(axes[1, 1], projection, data_crs, hidden, region)
            if not region_hidden.any():
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "no hidden-node distances inside requested bbox",
                    transform=axes[1, 1].transAxes,
                    ha="center",
                )
            axes[1, 1].set_title(f"nearest usable encoder source · {source_name}")
        identity = graph["identity"]
        output_extent = (
            f"lat=[{output[:, 0].min():.3f},{output[:, 0].max():.3f}] "
            f"lon=[{output[:, 1].min():.3f},{output[:, 1].max():.3f}]"
        )
        fig.suptitle(
            f"Forward domain · lat/lon degrees (stored radians) · full graph · cache={graph['cache']} "
            f"id={graph['identity_hash']}\n"
            f"resolved output nodes={len(output):,} extent {output_extent} · "
            f"stretched={identity.get('stretched')} resolution={identity.get('global_resolution')}/"
            f"{identity.get('local_resolution')} "
            f"margin={identity.get('margin_radius_km')} km",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_domain_case{payload['sample_index']:04d}",
            exp_log_tag="query/domain",
        )

    def _plot_connectivity(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        graph, output = payload["graph"], payload["graph"]["output"]
        fig, axes, projection, data_crs = self._map(output, (1, 2))
        source_name = next(iter(graph["encoder_edges"]), None)
        if source_name is not None:
            edges = graph["encoder_edges"][source_name]
            candidates = np.arange(edges.shape[1])
            if graph["bbox"] is not None:
                west, south, east, north = graph["bbox"]
                hidden = graph["hidden"]
                inside = (
                    (hidden[:, 1] >= west) & (hidden[:, 1] <= east) & (hidden[:, 0] >= south) & (hidden[:, 0] <= north)
                )
                candidates = np.flatnonzero(inside[edges[1]])
                if not len(candidates):
                    centre = np.asarray([(south + north) / 2, (west + east) / 2])
                    nearby = np.argsort(np.sum((hidden - centre) ** 2, axis=1))[: min(20, len(hidden))]
                    candidates = np.flatnonzero(np.isin(edges[1], nearby))
            chosen = candidates[_bounded(len(candidates), self.max_edges)]
            self._edge_panel(
                axes[0, 0],
                graph["sources"][source_name]["coordinates"],
                graph["hidden"],
                edges[:, chosen],
                projection,
                data_crs,
                f"encoder {source_name}→hidden · {len(chosen):,}/{edges.shape[1]:,} edges",
            )
        decoder = graph["decoder"]
        if decoder is not None:
            self._edge_panel(
                axes[0, 1],
                graph["hidden"],
                output[decoder["output_indices"]],
                decoder["edges"],
                projection,
                data_crs,
                f"dynamic decoder · {decoder['edges'].shape[1]:,} bounded edges",
            )
        fig.suptitle(
            "Connectivity used by forward path · global indices; decoder recomputed for bounded sampled destinations",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_connectivity_case{payload['sample_index']:04d}",
            exp_log_tag="query/connectivity",
        )

    def _edge_panel(
        self,
        ax: Any,
        source: np.ndarray,
        target: np.ndarray,
        edges: np.ndarray,
        projection: MapProjection,
        data_crs: Any,
        title: str,
    ) -> None:
        for src, dst in edges.T:
            points = np.stack((source[src], target[dst]))
            if abs(points[0, 1] - points[1, 1]) > 180:
                continue
            x, y = self._xy(points, projection)
            kwargs = {"color": "0.4", "lw": 0.25, "alpha": 0.35}
            if data_crs is not None:
                kwargs["transform"] = data_crs
            ax.plot(x, y, **kwargs)
        src, dst = np.unique(edges[0]), np.unique(edges[1])
        sx, sy = self._xy(source[src], projection)
        tx, ty = self._xy(target[dst], projection)
        skw, tkw = {"s": 5, "c": "tab:blue", "label": "source"}, {"s": 8, "c": "tab:red", "label": "destination"}
        if data_crs is not None:
            skw["transform"] = data_crs
            tkw["transform"] = data_crs
        ax.scatter(sx, sy, **skw)
        ax.scatter(tx, ty, **tkw)
        self._finish_map(ax, projection, data_crs, np.concatenate((source[src], target[dst])))
        ax.legend(fontsize=6)
        ax.set_title(title, fontsize=8)

    def _plot_embeddings(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        data = payload["embeddings"]
        fig, axes = plt.subplots(2, 4, figsize=(17, 7))
        for ax, values, labels, title in (
            (axes[0, 0], data["variable"], data["variable_labels"], "variable table norms"),
            (axes[0, 1], data["provenance"], data["provenance_labels"], "provenance table norms"),
            (axes[0, 2], data["unit"], data["unit_labels"], "unit table norms"),
        ):
            ax.bar(range(len(labels)), np.linalg.norm(values, axis=1))
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_title(title)
        for ax, values, labels, title in (
            (axes[0, 3], data["variable"], data["variable_labels"], "variable cosine similarity"),
            (axes[1, 0], data["provenance"], data["provenance_labels"], "provenance cosine similarity"),
            (axes[1, 1], data["unit"], data["unit_labels"], "unit cosine similarity"),
        ):
            image = ax.imshow(_cosine(values), vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
            ax.set_yticks(range(len(labels)), labels, fontsize=5)
            ax.set_title(title)
            fig.colorbar(image, ax=ax, shrink=0.65)
        points = data["pca"]
        axes[1, 2].scatter(points[:, 0], points[:, 1], s=15)
        for point, label in zip(points, data["labels"], strict=False):
            axes[1, 2].annotate(label, point, fontsize=5)
        for kind, sweep in data["sweeps"].items():
            axes[1, 2].plot(sweep["pca"][:, 0], sweep["pca"][:, 1], marker="o", label=kind)
        axes[1, 2].legend(fontsize=6)
        axes[1, 2].set_title(f"catalogue query PCA · fixed basis fit epoch {data['pca_fit_epoch']}")
        query = data["query_stages"]
        axes[1, 3].plot(np.linalg.norm(query["continuous"], axis=-1).reshape(-1), label="query continuous norm")
        axes[1, 3].axhline(np.linalg.norm(query["final"]), color="black", label="query final norm")
        norm_lines = []
        for source, stages in data["input_stages"].items():
            joint = stages["joint"].reshape(-1, stages["joint"].shape[-1])
            pooled = stages["pooled"].reshape(-1, stages["pooled"].shape[-1])
            axes[1, 3].plot(np.std(joint, axis=0), linestyle=":", label=f"{source} joint dimension std")
            axes[1, 3].plot(np.std(pooled, axis=0), label=f"{source} pooled dimension std")
            norm_lines.append(
                f"{source}: joint |.|={np.linalg.norm(joint, axis=1).mean():.2g}, "
                f"pooled |.|={np.linalg.norm(pooled, axis=1).mean():.2g}",
            )
        axes[1, 3].text(0.01, 0.98, "\n".join(norm_lines), transform=axes[1, 3].transAxes, va="top", fontsize=6)
        axes[1, 3].legend(fontsize=6)
        axes[1, 3].set_title("stage norms/variation (input and query branches separate)")
        fig.suptitle(
            "Actual metadata stages: categorical tables + continuous features to final query; "
            "values + metadata to pooled input",
        )
        fig.text(
            0.01,
            0.01,
            "Missing physical pressure is not fabricated: catalogue fields without a physical level are excluded. "
            "Optional known/omitted states appear only when the resolved base metadata supports that intervention.",
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_embeddings_case{payload['sample_index']:04d}",
            exp_log_tag="query/embeddings",
        )

    def _plot_sweeps(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        sweeps = payload["embeddings"]["sweeps"]
        if not sweeps:
            return
        fig, axes = plt.subplots(len(sweeps), 2, figsize=(11, max(3, 2.7 * len(sweeps))), squeeze=False)
        for row, (kind, sweep) in enumerate(sweeps.items()):
            labels = sweep["labels"]
            axes[row, 0].plot(range(len(labels)), sweep["final"][:, : min(6, sweep["final"].shape[1])], marker="o")
            axes[row, 0].set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=7)
            axes[row, 0].set_title(f"{kind}: final-representation components")
            distance = sweep["distance"]
            axes[row, 1].bar(range(len(distance)), distance)
            axes[row, 1].set_xticks(
                range(len(distance)),
                [f"{labels[i]}→{labels[i + 1]}" for i in range(len(distance))],
                rotation=25,
                ha="right",
                fontsize=7,
            )
            axes[row, 1].set_title(f"adjacent L2 distance · {', '.join(sweep['status'])}", fontsize=8)
        fig.suptitle(
            "Metadata sweeps · pressure labels are physical hPa after conversion; PCA uses the fixed basis above",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self._output_figure(
            trainer.logger,
            fig,
            payload["epoch"],
            tag=f"query_metadata_sweeps_case{payload['sample_index']:04d}",
            exp_log_tag="query/metadata_sweeps",
        )

    def _plot_sensitivity(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        coordinates, base = payload["coordinates"], payload["prediction"]
        for kind, group in payload["sensitivity"].items():
            count = len(group["predictions"])
            if not count and not group["skipped"]:
                continue
            fig, axes, projection, data_crs = self._map(coordinates, (max(1, count), 2))
            if not count:
                axes[0, 0].text(0.5, 0.5, "all requests skipped", ha="center")
                axes[0, 1].text(0.5, 0.5, "\n".join(group["skipped"]), ha="center", fontsize=7)
                axes[0, 0].axis("off")
                axes[0, 1].axis("off")
            for row, (prediction, label, status, rmse) in enumerate(
                zip(group["predictions"], group["labels"], group["status"], group["rmse"], strict=False),
            ):
                low, high = np.nanpercentile(np.concatenate((base, prediction)), [2, 98])
                difference = prediction - base
                limit = max(float(np.nanpercentile(np.abs(difference), 98)), np.finfo(np.float32).eps)
                self._scatter(
                    fig,
                    axes[row, 0],
                    coordinates,
                    prediction,
                    projection,
                    data_crs,
                    f"{label} · {status}",
                    vmin=low,
                    vmax=high,
                )
                self._scatter(
                    fig,
                    axes[row, 1],
                    coordinates,
                    difference,
                    projection,
                    data_crs,
                    f"variant - base · RMS={rmse:.4g} {group['units']}",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
            skipped = f" · skipped: {'; '.join(group['skipped'])}" if group["skipped"] else ""
            fig.suptitle(
                f"Controlled {kind} intervention · fixed input and coordinates · deterministic eval{skipped}\n"
                f"origin={payload['context'].get('forecast_origin')} · "
                f"base valid={payload['context'].get('valid_time')}",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            self._output_figure(
                trainer.logger,
                fig,
                payload["epoch"],
                tag=f"query_sensitivity_{kind}_case{payload['sample_index']:04d}",
                exp_log_tag=f"query/sensitivity_{kind}",
            )

    def _plot_sampler(self, trainer: pl.Trainer, payload: dict[str, Any]) -> None:
        counts = payload["counts"]
        fig, axes = plt.subplots(3, 3, figsize=(14, 9))
        for ax, key, title in zip(
            axes.flat[:4],
            ("variable", "provenance", "lead", "level"),
            ("target variable", "target provenance", "lead time", "physical level/bin"),
            strict=False,
        ):
            values = sorted(counts.get(key, {}).items(), key=lambda item: (-item[1], str(item[0])))[:20]
            ax.bar(range(len(values)), [value for _, value in values])
            labels = []
            for name, _value in values:
                weight = payload["weights"].get(key, {}).get(name)
                dataset_weight = payload["weights"]["dataset_sampling"].get(name) if key == "provenance" else None
                suffix = f"\ntask w={weight:g}" if weight is not None else ""
                suffix += f"\ndata w={dataset_weight:g}" if dataset_weight is not None else ""
                labels.append(f"{name}{suffix}")
            ax.set_xticks(range(len(values)), labels, rotation=60, ha="right", fontsize=7)
            ax.set_title(f"observed {title} counts")
        joint = counts.get("joint", {})
        provenances = sorted({key[0] for key in joint})
        variables = sorted({key[1] for key in joint})
        matrix = np.asarray(
            [[joint.get((provenance, variable), 0) for variable in variables] for provenance in provenances],
        )
        image = axes[1, 1].imshow(matrix, aspect="auto", cmap="Blues") if matrix.size else None
        axes[1, 1].set_xticks(range(len(variables)), variables, rotation=60, ha="right", fontsize=7)
        axes[1, 1].set_yticks(range(len(provenances)), provenances, fontsize=7)
        axes[1, 1].set_title("observed provenance x variable")
        if image is not None:
            fig.colorbar(image, ax=axes[1, 1], shrink=0.7)
        names = sorted(set(counts.get("field_available", {})) | set(counts.get("active_source", {})))
        x = np.arange(len(names))
        available = [counts.get("field_available", {}).get(name, 0) for name in names]
        selected = [counts.get("field_selected", {}).get(name, 0) for name in names]
        axes[1, 2].bar(x - 0.2, available, 0.4, label="available field selections")
        axes[1, 2].bar(x + 0.2, selected, 0.4, label="after field dropout")
        source_labels = [
            f"{name}\nactive={counts.get('active_source', {}).get(name, 0)} "
            f"drop={counts.get('source_dropout', {}).get(name, 0)}\n"
            f"history={counts.get('history_selected', {}).get(name, 0)}/"
            f"{counts.get('history_available', {}).get(name, 0)}"
            for name in names
        ]
        axes[1, 2].set_xticks(x, source_labels, rotation=45, ha="right", fontsize=7)
        axes[1, 2].legend(fontsize=7)
        axes[1, 2].set_title("input selection; source/history dropout counted in callback")
        rows = []
        for key, values in sorted(payload["metrics"].items(), key=lambda item: -item[1][3])[:12]:
            count = values[3]
            rows.append(
                [
                    "/".join(key[:3]),
                    f"{count:.0f}",
                    f"{math.sqrt(values[0] / count):.3g}",
                    f"{values[1] / count:.3g} {key[3]}",
                    f"{values[2] / count:.3g}",
                ],
            )
        for ax in axes[2]:
            ax.axis("off")
        table = axes[2, 1].table(
            cellText=rows,
            colLabels=["var/prov/lead", "n", "norm RMSE", "physical RMSE", "loss weight"],
            loc="center",
            cellLoc="left",
            colWidths=[0.35, 0.08, 0.15, 0.27, 0.15],
            bbox=[-1, 0, 3, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        axes[2, 1].set_title("subgroup error; physical units are never pooled")
        entered = counts.get("entered", {}).get("samples", 0)
        valid = counts.get("valid_target", {}).get("samples", 0)
        fig.suptitle(
            f"Actual training examples · {payload['ranks']} coordinated rank(s) · entered={entered}, "
            f"valid-target={valid}; "
            "sample counts ≠ valid counts ≠ loss weights; sampler prefilters unsupported options and has no fallback",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self._output_figure(trainer.logger, fig, payload["epoch"], tag="query_sampler", exp_log_tag="query/sampler")
