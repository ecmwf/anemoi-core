# (C) Copyright 2026 Anemoi contributors.

"""Checkpoint-resident query interpretation without access to training archives."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.models.layers.query_adapter import CONTINUOUS_METADATA

MODEL_TYPE_ALIASES = {
    "sfc": "surface",
    "surface": "surface",
    "meanSea": "surface",
    "pl": "pressure",
    "pressure": "pressure",
    "isobaricInhPa": "pressure",
    "isobaricInPa": "pressure",
    "ml": "model",
    "model": "model",
    "hybrid": "model",
    "hybridLevel": "model",
    "height": "height",
    "hl": "height",
    "heightAboveGround": "height",
    "heightAboveSea": "height",
    "layer": "layer",
    "unknown": "unknown",
}


class QueryMetadata:
    """Interpret user metadata using a persisted training catalogue snapshot."""

    def __init__(self, snapshot: dict[str, Any], default_provenance: str) -> None:
        self.snapshot = snapshot
        self.default_provenance = default_provenance
        self.fields = []
        for field_value in snapshot["fields"]:
            field = dict(field_value)
            field.setdefault(
                "aggregation_type", field.pop("processing", "instantaneous")
            )
            field.setdefault(
                "temporal_aggregation_window_hours",
                field.pop("interval_hours", None),
            )
            field.setdefault("model_level", None)
            self.fields.append(field)
        self.variable_to_id = {
            name: index for index, name in enumerate(snapshot["variables"])
        }
        self.provenance_to_id = {
            name: index for index, name in enumerate(snapshot["provenances"])
        }
        self.units = list(snapshot.get("units", ["unknown"]))
        self.unit_to_id = {name: index for index, name in enumerate(self.units)}
        self.aliases = snapshot.get("aliases", {})

    @staticmethod
    def _hours(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, timedelta):
            return value.total_seconds() / 3600
        if isinstance(value, str) and value.lower().endswith("min"):
            value = value[:-3] + "m"
        return frequency_to_timedelta(value).total_seconds() / 3600

    @staticmethod
    def _window_hours(value: Any) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return (-QueryMetadata._hours(value), 0.0)
        if len(value) != 2:
            raise ValueError(
                "temporal_aggregation_window requires [start, end] bounds relative to valid time."
            )
        return tuple(QueryMetadata._hours(item) for item in value)

    def canonicalize(
        self, value: dict[str, Any], default_source: str | None = None
    ) -> dict[str, Any]:
        """Validate preferred query names and return the model's canonical representation."""
        if "variable" not in value:
            raise ValueError("A query or input field requires variable metadata.")
        provenance = value.get("provenance", default_source or self.default_provenance)
        if provenance not in self.provenance_to_id:
            raise KeyError(
                f"Provenance {provenance!r} is absent from the checkpoint vocabulary."
            )
        variable = self.aliases.get(str(value["variable"]), str(value["variable"]))
        if variable not in self.variable_to_id:
            raise KeyError(
                f"Variable {variable!r} is absent from the checkpoint vocabulary."
            )
        raw_units = value.get("unit", value.get("units"))
        if raw_units is None:
            raise ValueError(
                "unit is required; use the physical value unit or explicit 'unknown'."
            )
        units = str(raw_units)
        if units not in self.unit_to_id:
            raise KeyError(
                f"Unit {units!r} is absent from the checkpoint vocabulary {self.units!r}."
            )

        raw_model_type = value.get("model_type", value.get("level_type", "sfc"))
        level_type = MODEL_TYPE_ALIASES.get(raw_model_type)
        if level_type is None:
            raise ValueError(f"Unsupported model_type {raw_model_type!r}.")
        level = value.get("level")
        level_unit = value.get("level_unit", value.get("level_units"))
        pressure_pa = value.get("pressure_pa")
        model_level = value.get("model_level")
        height_m = value.get("height_m")

        if level_type == "pressure" and pressure_pa is None:
            if level is None or level_unit is None:
                raise ValueError(
                    "model_type='pl' requires level and level_unit, or pressure_pa."
                )
            if str(level_unit).lower() in {"hpa", "mbar"}:
                pressure_pa = float(level) * 100
            elif str(level_unit).lower() == "pa":
                pressure_pa = float(level)
            else:
                raise ValueError(
                    f"Unsupported pressure level_unit {level_unit!r}; use Pa or hPa."
                )
        if level_type == "model":
            model_level = level if model_level is None else model_level
            if (
                not isinstance(model_level, int)
                or isinstance(model_level, bool)
                or model_level < 1
            ):
                raise ValueError("model_type='ml' requires a positive integer level.")
            if level_unit not in {None, "1"}:
                raise ValueError(
                    "A model-level index is dimensionless; omit level_unit or use '1'."
                )
        if level_type in {"height", "layer"} and height_m is None:
            if level is None or level_unit is None:
                raise ValueError(
                    f"model_type={raw_model_type!r} requires level and level_unit, or height_m."
                )
            if str(level_unit).lower() in {"m", "metre", "meter", "metres", "meters"}:
                height_m = float(level)
            elif str(level_unit).lower() in {
                "km",
                "kilometre",
                "kilometer",
                "kilometres",
                "kilometers",
            }:
                height_m = float(level) * 1000
            else:
                raise ValueError(
                    f"Unsupported height level_unit {level_unit!r}; use m or km."
                )
        if level_type == "surface" and any(
            item is not None
            for item in (level, level_unit, pressure_pa, model_level, height_m)
        ):
            raise ValueError("model_type='sfc' must not declare level or level_unit.")
        if pressure_pa is not None:
            pressure_pa = float(pressure_pa)
            if (
                level_type != "pressure"
                or not math.isfinite(pressure_pa)
                or pressure_pa <= 0
            ):
                raise ValueError(
                    "pressure_pa must be a positive finite value and applies only to model_type='pl'."
                )
        if height_m is not None:
            height_m = float(height_m)
            if level_type not in {"height", "layer"} or not math.isfinite(height_m):
                raise ValueError(
                    "height_m must be finite and applies only to height or layer coordinates."
                )

        aggregation_type = value.get(
            "aggregation_type",
            value.get(
                "processing",
                "accumulation"
                if value.get("accumulation_interval")
                else "instantaneous",
            ),
        )
        if aggregation_type not in {
            "instantaneous",
            "accumulation",
            "mean",
            "maximum",
            "minimum",
        }:
            raise ValueError(f"Unsupported aggregation_type {aggregation_type!r}.")
        window = self._window_hours(
            value.get(
                "temporal_aggregation_window",
                value.get("interval", value.get("accumulation_interval")),
            ),
        )
        if window is not None and (window[0] >= window[1] or window[1] > 0):
            raise ValueError(
                "temporal_aggregation_window must have start < end <= 0 relative to valid time."
            )
        if aggregation_type != "instantaneous" and window is None:
            raise ValueError(
                f"aggregation_type={aggregation_type!r} requires temporal_aggregation_window."
            )
        if window is not None and aggregation_type == "instantaneous":
            raise ValueError(
                "temporal_aggregation_window does not apply to instantaneous fields."
            )

        resolution = value.get("grid_spacing_km", value.get("resolution"))
        if isinstance(resolution, str):
            if not resolution.lower().endswith("km"):
                raise ValueError(
                    "Grid spacing strings must use kilometres, for example '2.5km'."
                )
            resolution = float(resolution[:-2])
        if resolution is not None:
            resolution = float(resolution)
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError(
                    "Grid spacing must be a positive finite value in kilometres."
                )
        bbox = value.get("area", value.get("bbox"))
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("bbox must contain [west, south, east, north].")
            west, south, east, north = (float(item) for item in bbox)
            bbox = (west, south, east, north)
            if not all(math.isfinite(item) for item in bbox) or not (
                -180 <= west < east <= 180 and -90 <= south < north <= 90
            ):
                raise ValueError("bbox must be [west, south, east, north] in degrees.")
        if (
            resolution is not None
            and value.get("grid") is None
            and value.get("output_coordinates") is None
        ):
            raise ValueError(
                "A bbox and kilometre spacing do not define a projection; supply grid or coordinates."
            )
        lead_time = self._hours(value.get("lead_time", "0h"))
        if not math.isfinite(lead_time) or lead_time < 0:
            raise ValueError("lead_time must be a non-negative finite duration.")
        frequency_value = value.get(
            "output_frequency", value.get("frequency", value.get("output_cadence"))
        )
        output_frequency = (
            None if frequency_value is None else self._hours(frequency_value)
        )
        if output_frequency is not None and (
            not math.isfinite(output_frequency) or output_frequency <= 0
        ):
            raise ValueError("output_frequency must be a positive finite duration.")
        spatial_support = value.get("spatial_support_km")
        if spatial_support is not None:
            spatial_support = float(spatial_support)
            if not math.isfinite(spatial_support) or spatial_support < 0:
                raise ValueError(
                    "Spatial support must be a non-negative finite value in kilometres."
                )
        return {
            "variable": variable,
            "provenance": provenance,
            "units": units,
            "level_type": level_type,
            "pressure_pa": pressure_pa,
            "model_level": model_level,
            "height_m": height_m,
            "aggregation_type": aggregation_type,
            "temporal_aggregation_window_hours": window,
            "resolution_km": resolution,
            "spatial_support_km": spatial_support,
            "lead_time_hours": lead_time,
            "output_frequency_hours": output_frequency,
            "grid": value.get("grid"),
            "bbox": bbox,
        }

    @staticmethod
    def _matches(field: dict[str, Any], value: dict[str, Any]) -> bool:
        scalar_keys = ("variable", "provenance", "level_type", "aggregation_type")
        if any(field.get(key) != value.get(key) for key in scalar_keys):
            return False
        if (field.get("units") or "unknown") != value["units"]:
            return False
        if (
            field.get("pressure_pa") is not None or value["pressure_pa"] is not None
        ) and (
            field.get("pressure_pa") is None
            or value["pressure_pa"] is None
            or not math.isclose(field["pressure_pa"], value["pressure_pa"], abs_tol=1)
        ):
            return False
        field_window = (
            None
            if field.get("temporal_aggregation_window_hours") is None
            else tuple(field["temporal_aggregation_window_hours"])
        )
        return (
            field.get("height_m") == value["height_m"]
            and field.get("model_level") == value["model_level"]
            and field_window == value["temporal_aggregation_window_hours"]
        )

    def normalization(self, value: dict[str, Any]) -> tuple[float, float]:
        matches = [
            field
            for field in self.fields
            if field["target_supported"] and self._matches(field, value)
        ]
        if matches:
            return float(matches[0]["mean"]), float(matches[0]["stdev"])
        candidates = sorted(
            (
                field
                for field in self.fields
                if field["target_supported"]
                and field["variable"] == value["variable"]
                and field["provenance"] == value["provenance"]
                and (field.get("units") or "unknown") == value["units"]
                and field["aggregation_type"] == value["aggregation_type"]
                and field["level_type"] == "pressure"
                and (
                    None
                    if field["temporal_aggregation_window_hours"] is None
                    else tuple(field["temporal_aggregation_window_hours"])
                )
                == value["temporal_aggregation_window_hours"]
                and field["pressure_pa"] is not None
            ),
            key=lambda field: field["pressure_pa"],
        )
        pressures = np.asarray([field["pressure_pa"] for field in candidates])
        point = value["pressure_pa"]
        if (
            point is None
            or len(candidates) < 2
            or point < pressures[0]
            or point > pressures[-1]
        ):
            raise ValueError(
                f"No exact statistics or trained pressure bracket for {value['variable']}/{value['provenance']} "
                f"in {value['units']}."
            )
        x = np.log(pressures)
        return (
            float(np.interp(np.log(point), x, [field["mean"] for field in candidates])),
            float(
                np.interp(np.log(point), x, [field["stdev"] for field in candidates])
            ),
        )

    def resolve_input(self, value: dict[str, Any], source: str) -> dict[str, Any]:
        canonical = self.canonicalize(value, default_source=source)
        matches = [
            field
            for field in self.fields
            if field.get("input_supported", True) and self._matches(field, canonical)
        ]
        if matches:
            return {
                **matches[0],
                "time_offset_hours": self._hours(value.get("time_offset", "0h")),
            }
        mean, stdev = self.normalization(canonical)
        return {
            **canonical,
            "mean": mean,
            "stdev": stdev,
            "time_offset_hours": self._hours(value.get("time_offset", "0h")),
        }

    @staticmethod
    def encode(value: dict[str, Any], time_offset_hours: float) -> np.ndarray:
        pressure = value.get("pressure_pa")
        model_level = value.get("model_level")
        height = value.get("height_m")
        level_type = value.get("level_type") or "surface"
        window = value.get("temporal_aggregation_window_hours")
        resolution = value.get("resolution_km")
        support = value.get("spatial_support_km")
        aggregation_type = value["aggregation_type"]
        encoded = {
            "log_pressure": 0.0 if pressure is None else math.log(pressure / 100000),
            "pressure_applies": float(level_type == "pressure"),
            "pressure_known": float(pressure is not None),
            "model_level": 0.0 if model_level is None else math.log1p(model_level) / 10,
            "model_level_applies": float(level_type == "model"),
            "model_level_known": float(model_level is not None),
            "height_m": 0.0 if height is None else height / 10000,
            "height_applies": float(level_type in {"height", "layer"}),
            "height_known": float(height is not None),
            "time_offset_hours": time_offset_hours / 168,
            "input_cadence_hours": 0.0
            if value.get("cadence_hours") is None
            else value["cadence_hours"] / 24,
            "input_cadence_known": float(value.get("cadence_hours") is not None),
            "output_frequency_hours": (
                0.0
                if value.get("output_frequency_hours") is None
                else value["output_frequency_hours"] / 24
            ),
            "output_frequency_known": float(
                value.get("output_frequency_hours") is not None
            ),
            "temporal_aggregation_window_start_hours": 0.0
            if window is None
            else window[0] / 24,
            "temporal_aggregation_window_end_hours": 0.0
            if window is None
            else window[1] / 24,
            "temporal_aggregation_window_applies": float(window is not None),
            "grid_spacing_km": 0.0
            if resolution is None
            else math.log1p(resolution) / 10,
            "grid_spacing_known": float(resolution is not None),
            "spatial_support_km": 0.0 if support is None else math.log1p(support) / 10,
            "spatial_support_known": float(support is not None),
            "level_surface": float(level_type == "surface"),
            "level_pressure": float(level_type == "pressure"),
            "level_model": float(level_type == "model"),
            "level_height": float(level_type == "height"),
            "level_layer": float(level_type == "layer"),
            "level_unknown": float(
                level_type not in {"surface", "pressure", "model", "height", "layer"}
            ),
            "aggregation_type_instantaneous": float(
                aggregation_type == "instantaneous"
            ),
            "aggregation_type_mean": float(aggregation_type == "mean"),
            "aggregation_type_accumulation": float(aggregation_type == "accumulation"),
            "aggregation_type_other": float(
                aggregation_type not in {"instantaneous", "mean", "accumulation"}
            ),
        }
        return np.asarray(
            [encoded[name] for name in CONTINUOUS_METADATA], dtype=np.float32
        )
