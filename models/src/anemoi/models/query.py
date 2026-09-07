# (C) Copyright 2026 Anemoi contributors.

"""Checkpoint-resident query interpretation without access to training archives."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np

from anemoi.models.layers.query_adapter import CONTINUOUS_METADATA
from anemoi.utils.dates import frequency_to_timedelta


class QueryMetadata:
    """Interpret user metadata using a persisted training catalogue snapshot."""

    def __init__(self, snapshot: dict[str, Any], default_provenance: str) -> None:
        self.snapshot = snapshot
        self.default_provenance = default_provenance
        self.fields = snapshot["fields"]
        self.variable_to_id = {name: index for index, name in enumerate(snapshot["variables"])}
        self.provenance_to_id = {name: index for index, name in enumerate(snapshot["provenances"])}
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

    def canonicalize(self, value: dict[str, Any], default_source: str | None = None) -> dict[str, Any]:
        if "variable" not in value:
            raise ValueError("A query or input field requires variable metadata.")
        provenance = value.get("provenance", default_source or self.default_provenance)
        if provenance not in self.provenance_to_id:
            raise KeyError(f"Provenance {provenance!r} is absent from the checkpoint vocabulary.")
        variable = self.aliases.get(str(value["variable"]), str(value["variable"]))
        if variable not in self.variable_to_id:
            raise KeyError(f"Variable {variable!r} is absent from the checkpoint vocabulary.")

        level_type = value.get("level_type")
        height_m = value.get("height_m")
        if level_type in {"pl", "isobaricInhPa", "isobaricInPa"}:
            level_type = "pressure"
        elif level_type in {"ml", "hybrid", "hybridLevel"}:
            raise ValueError(
                "Native model-level inference requires a persisted dataset-specific pressure transform and "
                "surface pressure; an ML index alone is not a physical coordinate."
            )
        elif level_type in {"sfc", "surface", None}:
            level_type = "surface"
        elif level_type in {"hl", "heightAboveGround", "heightAboveSea", "height"}:
            level_type = "height"
            if height_m is None:
                level = value.get("level")
                units = value.get("level_units")
                if level is None or units is None:
                    raise ValueError("Height metadata requires height_m, or level and level_units.")
                if units.lower() in {"m", "metre", "meter", "metres", "meters"}:
                    height_m = float(level)
                elif units.lower() in {"km", "kilometre", "kilometer", "kilometres", "kilometers"}:
                    height_m = float(level) * 1000
                else:
                    raise ValueError(f"Unsupported height unit {units!r}; use m or km.")
        else:
            raise ValueError(f"Unsupported physical level_type {level_type!r}.")

        pressure_pa = value.get("pressure_pa")
        if level_type == "pressure" and pressure_pa is None:
            level = value.get("level")
            units = value.get("level_units")
            if level is None or units is None:
                raise ValueError("Pressure metadata requires level and level_units, or pressure_pa.")
            if units.lower() in {"hpa", "mbar"}:
                pressure_pa = float(level) * 100
            elif units.lower() == "pa":
                pressure_pa = float(level)
            else:
                raise ValueError(f"Unsupported pressure unit {units!r}; use Pa or hPa.")
        if pressure_pa is not None:
            pressure_pa = float(pressure_pa)
            if not math.isfinite(pressure_pa) or pressure_pa <= 0:
                raise ValueError("Physical pressure must be a positive finite value in Pa.")
        if height_m is not None:
            height_m = float(height_m)
            if not math.isfinite(height_m):
                raise ValueError("Physical height must be finite metres.")
        if pressure_pa is not None and level_type != "pressure":
            raise ValueError("pressure_pa applies only to a pressure level.")
        if height_m is not None and level_type != "height":
            raise ValueError("height_m applies only to a height level.")

        processing = value.get(
            "processing",
            "accumulation" if value.get("accumulation_interval") else "instantaneous",
        )
        if processing not in {"instantaneous", "accumulation", "mean", "maximum", "minimum"}:
            raise ValueError(f"Unsupported processing type {processing!r}.")
        interval = value.get("interval") or value.get("accumulation_interval")
        if interval is not None:
            if isinstance(interval, str):
                interval = (-self._hours(interval), 0.0)
            else:
                if len(interval) != 2:
                    raise ValueError("Represented interval requires [start, end] bounds relative to valid time.")
                interval = tuple(self._hours(v) for v in interval)
            if interval[0] >= interval[1] or interval[1] > 0:
                raise ValueError("Represented interval must have start < end <= 0 relative to valid time.")
        if processing != "instantaneous" and interval is None:
            raise ValueError(f"processing={processing!r} requires represented interval bounds.")
        if interval is not None and processing == "instantaneous":
            raise ValueError("Represented intervals do not apply to instantaneous processing.")

        resolution = value.get("grid_spacing_km", value.get("resolution"))
        if isinstance(resolution, str):
            if not resolution.lower().endswith("km"):
                raise ValueError("Grid spacing strings must use kilometres, for example '2.5km'.")
            resolution = float(resolution[:-2])
        if resolution is not None:
            resolution = float(resolution)
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError("Grid spacing must be a positive finite value in kilometres.")
        bbox = value.get("area", value.get("bbox"))
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("area must contain [west, south, east, north].")
            west, south, east, north = bbox
            if not all(math.isfinite(float(item)) for item in bbox):
                raise ValueError("area bounds must be finite degrees.")
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError("area must be [west, south, east, north] in degrees.")
        if resolution is not None and value.get("grid") is None and value.get("output_coordinates") is None:
            raise ValueError("A bbox and kilometre spacing do not define a projection; supply grid or coordinates.")
        lead_time = self._hours(value.get("lead_time", "0h"))
        if not math.isfinite(lead_time) or lead_time < 0:
            raise ValueError("lead_time must be a non-negative finite duration.")
        cadence = (
            None
            if value.get("frequency", value.get("output_cadence")) is None
            else self._hours(value.get("frequency", value.get("output_cadence")))
        )
        if cadence is not None and (not math.isfinite(cadence) or cadence <= 0):
            raise ValueError("Output cadence must be a positive finite duration.")
        spatial_support = value.get("spatial_support_km")
        if spatial_support is not None:
            spatial_support = float(spatial_support)
            if not math.isfinite(spatial_support) or spatial_support < 0:
                raise ValueError("Spatial support must be a non-negative finite value in kilometres.")
        return {
            "variable": variable,
            "provenance": provenance,
            "units": value.get("units"),
            "level_type": level_type,
            "pressure_pa": pressure_pa,
            "height_m": height_m,
            "processing": processing,
            "interval_hours": interval,
            "resolution_km": resolution,
            "spatial_support_km": spatial_support,
            "lead_time_hours": lead_time,
            "output_cadence_hours": cadence,
            "grid": value.get("grid"),
            "bbox": bbox,
        }

    @staticmethod
    def _matches(field: dict[str, Any], value: dict[str, Any]) -> bool:
        scalar_keys = ("variable", "provenance", "level_type", "processing")
        if any(field[key] != value[key] for key in scalar_keys):
            return False
        if field["pressure_pa"] is not None or value["pressure_pa"] is not None:
            if (
                field["pressure_pa"] is None
                or value["pressure_pa"] is None
                or not math.isclose(field["pressure_pa"], value["pressure_pa"], abs_tol=1)
            ):
                return False
        field_interval = None if field["interval_hours"] is None else tuple(field["interval_hours"])
        return field["height_m"] == value["height_m"] and field_interval == value["interval_hours"]

    def normalization(self, value: dict[str, Any]) -> tuple[float, float]:
        matches = [field for field in self.fields if field["target_supported"] and self._matches(field, value)]
        if matches:
            if (
                value["units"] is not None
                and matches[0].get("units") is not None
                and value["units"] != matches[0]["units"]
            ):
                raise ValueError(
                    f"Output units {value['units']!r} do not match checkpoint units {matches[0]['units']!r}; "
                    "automatic physical-unit conversion is not implemented."
                )
            return float(matches[0]["mean"]), float(matches[0]["stdev"])
        candidates = sorted(
            (
                field
                for field in self.fields
                if field["target_supported"]
                and field["variable"] == value["variable"]
                and field["provenance"] == value["provenance"]
                and field["processing"] == value["processing"]
                and field["level_type"] == "pressure"
                and (None if field["interval_hours"] is None else tuple(field["interval_hours"]))
                == value["interval_hours"]
                and field["pressure_pa"] is not None
            ),
            key=lambda field: field["pressure_pa"],
        )
        pressures = np.asarray([field["pressure_pa"] for field in candidates])
        point = value["pressure_pa"]
        if point is None or len(candidates) < 2 or point < pressures[0] or point > pressures[-1]:
            raise ValueError(
                f"No exact statistics or trained pressure bracket for {value['variable']}/{value['provenance']}."
            )
        if value["units"] is not None and any(
            field.get("units") is not None and field["units"] != value["units"] for field in candidates
        ):
            raise ValueError(
                f"Output units {value['units']!r} do not match the checkpoint pressure-level units; "
                "automatic physical-unit conversion is not implemented."
            )
        x = np.log(pressures)
        return (
            float(np.interp(np.log(point), x, [field["mean"] for field in candidates])),
            float(np.interp(np.log(point), x, [field["stdev"] for field in candidates])),
        )

    def resolve_input(self, value: dict[str, Any], source: str) -> dict[str, Any]:
        canonical = self.canonicalize(value, default_source=source)
        matches = [
            field for field in self.fields if field.get("input_supported", True) and self._matches(field, canonical)
        ]
        if matches:
            requested_units = value.get("units")
            stored_units = matches[0].get("units")
            if requested_units is not None and stored_units is not None and requested_units != stored_units:
                raise ValueError(
                    f"Input units {requested_units!r} do not match checkpoint units {stored_units!r}; "
                    "automatic physical-unit conversion is not implemented."
                )
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
        height = value.get("height_m")
        level_type = value.get("level_type") or "surface"
        interval = value.get("interval_hours")
        resolution = value.get("resolution_km")
        support = value.get("spatial_support_km")
        processing = value["processing"]
        encoded = {
            "log_pressure": 0.0 if pressure is None else math.log(pressure / 100000),
            "pressure_applies": float(level_type in {"pressure", "model"}),
            "pressure_known": float(pressure is not None),
            "height_m": 0.0 if height is None else height / 10000,
            "height_applies": float(level_type in {"height", "layer"}),
            "height_known": float(height is not None),
            "time_offset_hours": time_offset_hours / 168,
            "input_cadence_hours": 0.0 if value.get("cadence_hours") is None else value["cadence_hours"] / 24,
            "input_cadence_known": float(value.get("cadence_hours") is not None),
            "output_cadence_hours": (
                0.0 if value.get("output_cadence_hours") is None else value["output_cadence_hours"] / 24
            ),
            "output_cadence_known": float(value.get("output_cadence_hours") is not None),
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
            "level_unknown": float(level_type not in {"surface", "pressure", "height", "layer"}),
            "processing_instantaneous": float(processing == "instantaneous"),
            "processing_mean": float(processing == "mean"),
            "processing_accumulation": float(processing == "accumulation"),
            "processing_other": float(processing not in {"instantaneous", "mean", "accumulation"}),
        }
        return np.asarray([encoded[name] for name in CONTINUOUS_METADATA], dtype=np.float32)
