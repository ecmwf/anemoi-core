# (C) Copyright 2026 Anemoi contributors.

"""Catalogue built from enabled Anemoi datasets."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from anemoi.models.layers.query_adapter import CONTINUOUS_METADATA
from anemoi.training.query.query import ForecastQuery
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)

PRESSURE_LEVEL_TYPES = {"pl", "pressure", "isobaricInhPa", "isobaricInPa"}
MODEL_LEVEL_TYPES = {"ml", "model", "hybrid", "hybridLevel"}
HEIGHT_LEVEL_TYPES = {"hl", "heightAboveGround", "heightAboveSea", "height"}
SURFACE_LEVEL_TYPES = {"sfc", "surface", "meanSea"}
LAYER_LEVEL_TYPES = {"sol", "soil", "depthBelowLandLayer", "depth_below_land_layer", "layer"}


def _hours(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "h"))
    if isinstance(value, timedelta):
        return value.total_seconds() / 3600
    return frequency_to_timedelta(value).total_seconds() / 3600


def _resolution_km(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    try:
        if text.endswith("km"):
            return float(text[:-2])
        if text.endswith("m"):
            return float(text[:-1]) / 1000
    except ValueError:
        return None
    return None


@dataclass(frozen=True)
class CatalogueField:
    dataset: str
    field_name: str
    variable: str
    provenance: str
    units: str | None
    level_type: str
    pressure_pa: float | None
    height_m: float | None
    processing: str
    interval_hours: tuple[float, float] | None
    cadence_hours: float | None
    resolution_km: float | None
    spatial_support_km: float | None
    mean: float
    stdev: float
    time_invariant: bool = False
    input_supported: bool = True
    target_supported: bool = True
    unsupported_reason: str | None = None

    def matches(self, query: ForecastQuery) -> bool:
        if (self.variable, self.provenance, self.processing) != (
            query.variable,
            query.provenance,
            query.processing,
        ):
            return False
        if self.level_type != (query.level_type or "surface"):
            return False
        if (self.pressure_pa is not None or query.pressure_pa is not None) and (
            self.pressure_pa is None
            or query.pressure_pa is None
            or not math.isclose(
                self.pressure_pa,
                query.pressure_pa,
                rel_tol=0,
                abs_tol=1,
            )
        ):
            return False
        if (self.height_m is not None or query.height_m is not None) and self.height_m != query.height_m:
            return False
        if self.interval_hours is not None or query.interval is not None:
            query_interval = None if query.interval is None else tuple(_hours(v) for v in query.interval)
            if self.interval_hours != query_interval:
                return False
        return True


class QueryCatalogue:
    """Index physical fields without weighting them by archive length."""

    version = 1

    def __init__(  # noqa: C901
        self,
        readers: dict[str, Any],
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.readers = readers
        self.aliases = aliases or {}
        self.fields: list[CatalogueField] = []
        self.excluded_fields: list[CatalogueField] = []
        self.datasets: dict[str, dict[str, Any]] = {}

        for dataset_name, reader in readers.items():
            metadata = reader.metadata
            variables_metadata = metadata.get("variables_metadata") or getattr(
                reader.data,
                "variables_metadata",
                {},
            )
            # The generic Dataset API does not define target provenance. The
            # configured catalogue key is therefore the explicit product identity.
            provenance = dataset_name
            statistics = reader.statistics
            resolution_km = _resolution_km(reader.resolution)
            dates = np.asarray(reader.dates).astype("datetime64[ns]")
            latitudes = np.asarray(reader.data.latitudes)
            longitudes = (np.asarray(reader.data.longitudes) + 180) % 360 - 180
            self.datasets[dataset_name] = {
                "frequency_hours": _hours(reader.frequency),
                "start": str(dates.min()),
                "end": str(dates.max()),
                "grid_size": int(reader.grid_size),
                "latitude_bounds_degrees": [
                    float(latitudes.min()),
                    float(latitudes.max()),
                ],
                "longitude_bounds_degrees": [
                    float(longitudes.min()),
                    float(longitudes.max()),
                ],
                "coordinate_sha256": hashlib.sha256(
                    np.asarray((latitudes, longitudes), dtype="<f8").tobytes(),
                ).hexdigest(),
                "native_resolution": str(reader.resolution),
                "resolution_km": resolution_km,
                "supporting_arrays": sorted(reader.supporting_arrays),
            }

            for field_name in reader.variables:
                index = reader.name_to_index[field_name]
                field_metadata = variables_metadata.get(field_name, {})
                mars = field_metadata.get("mars", {})
                variable = str(
                    mars.get("param") or field_metadata.get("param") or field_name,
                )
                variable = self.aliases.get(variable, variable)
                raw_level_type = mars.get("levtype") or field_metadata.get("level_type") or "sfc"
                level = mars.get("levelist", field_metadata.get("level"))
                pressure_pa = None
                height_m = None
                target_supported = True
                input_supported = True
                unsupported_reason = None

                if raw_level_type in PRESSURE_LEVEL_TYPES:
                    level_type = "pressure"
                    units = str(
                        mars.get("level_units")
                        or field_metadata.get("level_units")
                        or ("Pa" if raw_level_type == "isobaricInPa" else "hPa"),
                    )
                    if level is None:
                        target_supported = False
                        input_supported = False
                        unsupported_reason = "pressure level metadata has no physical level value"
                    elif units.lower() in {"hpa", "mbar"}:
                        pressure_pa = float(level) * 100
                    elif units.lower() == "pa":
                        pressure_pa = float(level)
                    else:
                        target_supported = False
                        input_supported = False
                        unsupported_reason = f"unsupported pressure unit {units!r}"
                elif raw_level_type in MODEL_LEVEL_TYPES:
                    level_type = "model"
                    target_supported = False
                    input_supported = False
                    unsupported_reason = (
                        "the dataset API does not expose a dataset-specific hybrid-coordinate transform, "
                        "its coefficients, and the matching surface-pressure field"
                    )
                elif raw_level_type in HEIGHT_LEVEL_TYPES:
                    level_type = "height"
                    units = str(mars.get("level_units") or field_metadata.get("level_units") or "m")
                    if level is None:
                        target_supported = False
                        input_supported = False
                        unsupported_reason = "height-level metadata has no physical height"
                    elif units.lower() in {"m", "metre", "meter", "metres", "meters"}:
                        height_m = float(level)
                    elif units.lower() in {"km", "kilometre", "kilometer", "kilometres", "kilometers"}:
                        height_m = float(level) * 1000
                    else:
                        target_supported = False
                        input_supported = False
                        unsupported_reason = f"unsupported height unit {units!r}"
                elif raw_level_type in SURFACE_LEVEL_TYPES:
                    level_type = "surface"
                else:
                    level_type = "layer" if raw_level_type in LAYER_LEVEL_TYPES else "unknown"
                    target_supported = False
                    input_supported = False
                    unsupported_reason = f"unsupported physical vertical coordinate {raw_level_type!r}"

                processing = str(field_metadata.get("process") or "instantaneous")
                if processing == "average":
                    processing = "mean"
                period = field_metadata.get("period")
                if period is None:
                    interval_hours = None
                elif isinstance(period, (str, int, float, timedelta, np.timedelta64)):
                    duration = _hours(period)
                    interval_hours = (-duration, 0.0) if duration > 0 else None
                else:
                    step_bounds = tuple(_hours(v) for v in period)
                    # Anemoi stores GRIB startStep/endStep; represent its duration
                    # relative to the field's valid time instead of forecast init.
                    duration = step_bounds[1] - step_bounds[0] if len(step_bounds) == 2 else 0
                    interval_hours = (-duration, 0.0) if duration > 0 else None
                if processing not in {"instantaneous", "accumulation", "mean", "maximum", "minimum"}:
                    target_supported = False
                    input_supported = False
                    unsupported_reason = f"unsupported processing type {processing!r}"
                elif processing != "instantaneous" and interval_hours is None:
                    target_supported = False
                    input_supported = False
                    unsupported_reason = f"{processing} field has no represented interval bounds"
                elif processing == "instantaneous" and interval_hours is not None:
                    target_supported = False
                    input_supported = False
                    unsupported_reason = "instantaneous field unexpectedly declares a represented interval"
                if target_supported and (
                    field_metadata.get("computed_forcing", False) or field_metadata.get("constant_in_time", False)
                ):
                    target_supported = False
                    unsupported_reason = "computed forcings and constant fields are input-only"

                mean = float(
                    np.asarray(statistics.get("mean", np.zeros(len(reader.variables))))[index],
                )
                stdev = float(
                    np.asarray(statistics.get("stdev", np.ones(len(reader.variables))))[index],
                )
                if not np.isfinite(stdev) or stdev <= 0:
                    stdev = 1.0

                field = CatalogueField(
                    dataset=dataset_name,
                    field_name=field_name,
                    variable=variable,
                    provenance=str(provenance),
                    units=mars.get("units") or field_metadata.get("units"),
                    level_type=level_type,
                    pressure_pa=pressure_pa,
                    height_m=height_m,
                    processing=processing,
                    interval_hours=interval_hours,
                    cadence_hours=_hours(reader.frequency),
                    resolution_km=resolution_km,
                    spatial_support_km=field_metadata.get("spatial_support_km"),
                    mean=mean,
                    stdev=stdev,
                    time_invariant=bool(field_metadata.get("constant_in_time", False)),
                    input_supported=input_supported,
                    target_supported=target_supported,
                    unsupported_reason=unsupported_reason,
                )
                self.fields.append(field)
                if not target_supported:
                    self.excluded_fields.append(field)

        self.variables = sorted({field.variable for field in self.fields})
        self.provenances = sorted({field.provenance for field in self.fields})
        self.variable_to_id = {name: index for index, name in enumerate(self.variables)}
        self.provenance_to_id = {name: index for index, name in enumerate(self.provenances)}
        if self.excluded_fields:
            LOGGER.warning(
                "Excluded %d fields from query targets because their physical semantics are incomplete. First: %s",
                len(self.excluded_fields),
                self.excluded_fields[0].unsupported_reason,
            )

    @property
    def target_fields(self) -> list[CatalogueField]:
        return [field for field in self.fields if field.target_supported]

    def restrict_targets(self, variables: list[str] | None) -> list[CatalogueField]:
        if not variables:
            return self.target_fields
        requested = {self.aliases.get(value, value) for value in variables}
        return [field for field in self.target_fields if field.variable in requested]

    def resolve(self, query: ForecastQuery) -> CatalogueField:
        matches = [field for field in self.target_fields if field.matches(query)]
        if not matches:
            msg = f"No supported target matches query {query}."
            raise KeyError(msg)
        if len(matches) > 1:
            matches = sorted(matches, key=lambda field: field.dataset)
        return matches[0]

    def resolve_input(self, value: dict[str, Any], source: str) -> CatalogueField:
        """Resolve an inference input field against persisted physical metadata."""
        query_value = dict(value)
        query_value.setdefault("provenance", source)
        query_value.setdefault("lead_time", "0h")
        query = ForecastQuery.from_dict(query_value)
        matches = [field for field in self.fields if field.input_supported and field.matches(query)]
        if matches:
            return matches[0]
        if query.level_type != "pressure" or query.pressure_pa is None:
            msg = f"No catalogue field matches input metadata {value!r} from source {source!r}."
            raise KeyError(msg)
        mean, stdev = self.normalization(query)
        return CatalogueField(
            dataset=source,
            field_name=str(value.get("field_name", query.variable)),
            variable=query.variable,
            provenance=query.provenance,
            units=value.get("units"),
            level_type="pressure",
            pressure_pa=query.pressure_pa,
            height_m=None,
            processing=query.processing,
            interval_hours=(None if query.interval is None else tuple(_hours(item) for item in query.interval)),
            cadence_hours=self.datasets.get(source, {}).get("frequency_hours"),
            resolution_km=query.grid_spacing_km,
            spatial_support_km=query.spatial_support_km,
            mean=mean,
            stdev=stdev,
        )

    def normalization(self, query: ForecastQuery) -> tuple[float, float]:
        """Return exact statistics or interpolate them in log-pressure space."""
        try:
            field = self.resolve(query)
        except KeyError:
            if query.level_type != "pressure" or query.pressure_pa is None:
                raise
        else:
            return field.mean, field.stdev

        candidates = [
            field
            for field in self.target_fields
            if field.variable == query.variable
            and field.provenance == query.provenance
            and field.processing == query.processing
            and field.level_type == "pressure"
            and field.interval_hours
            == (None if query.interval is None else tuple(_hours(item) for item in query.interval))
            and field.pressure_pa is not None
        ]
        candidates.sort(key=lambda field: field.pressure_pa)
        pressures = np.asarray([field.pressure_pa for field in candidates])
        if len(candidates) < 2 or query.pressure_pa < pressures[0] or query.pressure_pa > pressures[-1]:
            msg = (
                f"Cannot normalize pressure {query.pressure_pa} Pa for {query.variable}/{query.provenance}: "
                "interpolation requires at least two trained levels bracketing the request."
            )
            raise ValueError(
                msg,
            )
        x = np.log(pressures)
        point = np.log(query.pressure_pa)
        return (
            float(np.interp(point, x, [field.mean for field in candidates])),
            float(np.interp(point, x, [field.stdev for field in candidates])),
        )

    def encode_metadata(
        self,
        field: CatalogueField | ForecastQuery,
        time_offset_hours: float,
    ) -> np.ndarray:
        pressure_pa = field.pressure_pa
        height_m = field.height_m
        level_type = field.level_type or "surface"
        interval = (
            field.interval_hours
            if isinstance(field, CatalogueField)
            else (None if field.interval is None else tuple(_hours(v) for v in field.interval))
        )
        resolution = field.resolution_km if isinstance(field, CatalogueField) else field.grid_spacing_km
        support = field.spatial_support_km
        processing = field.processing
        input_cadence = field.cadence_hours if isinstance(field, CatalogueField) else None
        output_cadence = None if isinstance(field, CatalogueField) else field.output_cadence
        values = {
            "log_pressure": (0.0 if pressure_pa is None else math.log(pressure_pa / 100000)),
            "pressure_applies": float(level_type in {"pressure", "model"}),
            "pressure_known": float(pressure_pa is not None),
            "height_m": 0.0 if height_m is None else height_m / 10000,
            "height_applies": float(level_type in {"height", "layer"}),
            "height_known": float(height_m is not None),
            "time_offset_hours": time_offset_hours / 168,
            "input_cadence_hours": 0.0 if input_cadence is None else _hours(input_cadence) / 24,
            "input_cadence_known": float(input_cadence is not None),
            "output_cadence_hours": 0.0 if output_cadence is None else _hours(output_cadence) / 24,
            "output_cadence_known": float(output_cadence is not None),
            "interval_start_hours": 0.0 if interval is None else interval[0] / 24,
            "interval_end_hours": 0.0 if interval is None else interval[1] / 24,
            "interval_applies": float(interval is not None),
            "grid_spacing_km": (0.0 if resolution is None else math.log1p(resolution) / 10),
            "grid_spacing_known": float(resolution is not None),
            "spatial_support_km": 0.0 if support is None else math.log1p(support) / 10,
            "spatial_support_known": float(support is not None),
            "level_surface": float(level_type == "surface"),
            "level_pressure": float(level_type == "pressure"),
            "level_height": float(level_type == "height"),
            "level_layer": float(level_type == "layer"),
            "level_unknown": float(
                level_type not in {"surface", "pressure", "height", "layer"},
            ),
            "processing_instantaneous": float(processing == "instantaneous"),
            "processing_mean": float(processing == "mean"),
            "processing_accumulation": float(processing == "accumulation"),
            "processing_other": float(
                processing not in {"instantaneous", "mean", "accumulation"},
            ),
        }
        return np.asarray(
            [values[name] for name in CONTINUOUS_METADATA],
            dtype=np.float32,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "continuous_metadata": list(CONTINUOUS_METADATA),
            "variables": self.variables,
            "provenances": self.provenances,
            "aliases": self.aliases,
            "datasets": self.datasets,
            "fields": [asdict(field) for field in self.fields],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> QueryCatalogue:
        catalogue = cls.__new__(cls)
        catalogue.readers = {}
        catalogue.aliases = snapshot.get("aliases", {})
        catalogue.datasets = snapshot.get("datasets", {})
        catalogue.fields = [CatalogueField(**value) for value in snapshot["fields"]]
        catalogue.excluded_fields = [field for field in catalogue.fields if not field.target_supported]
        catalogue.variables = list(snapshot["variables"])
        catalogue.provenances = list(snapshot["provenances"])
        catalogue.variable_to_id = {name: index for index, name in enumerate(catalogue.variables)}
        catalogue.provenance_to_id = {name: index for index, name in enumerate(catalogue.provenances)}
        return catalogue
