# (C) Copyright 2026 Anemoi contributors.

"""Validated query semantics for query-based forecasting."""

from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from anemoi.utils.dates import frequency_to_timedelta


def _duration(value: Any) -> timedelta:
    if isinstance(value, str) and value.lower().endswith("min"):
        value = value[:-3] + "m"
    return frequency_to_timedelta(value)


@dataclass(frozen=True)
class ForecastQuery:
    """One scalar field requested at one valid time.

    Bboxes are ``(west, south, east, north)`` in degrees east/north. ``interval``
    is relative to valid time and is distinct from both lead time and output
    cadence. Geometry is either a registered native grid or explicit coordinates.
    """

    variable: str
    lead_time: timedelta
    provenance: str
    output_cadence: timedelta | None = None
    level_type: str | None = None
    pressure_pa: float | None = None
    height_m: float | None = None
    processing: str = "instantaneous"
    interval: tuple[timedelta, timedelta] | None = None
    bbox: tuple[float, float, float, float] | None = None
    grid: str | None = None
    grid_spacing_km: float | None = None
    spatial_support_km: float | None = None

    def __post_init__(self) -> None:  # noqa: C901
        if not self.variable or not self.provenance:
            msg = "variable and provenance must be resolved before loading a target."
            raise ValueError(msg)
        if self.lead_time < timedelta(0):
            msg = "lead_time must not be negative."
            raise ValueError(msg)
        if self.output_cadence is not None and self.output_cadence <= timedelta(0):
            msg = "output cadence must be positive."
            raise ValueError(msg)
        if self.pressure_pa is not None and self.pressure_pa <= 0:
            msg = "pressure_pa must be positive."
            raise ValueError(msg)
        if self.height_m is not None and self.level_type not in {"height", "layer"}:
            msg = "height_m only applies to height or layer queries."
            raise ValueError(msg)
        if self.level_type == "pressure" and self.pressure_pa is None:
            msg = "Pressure queries require physical pressure, not a model-level index."
            raise ValueError(msg)
        if self.pressure_pa is not None and self.level_type != "pressure":
            msg = "pressure_pa only applies to level_type='pressure'."
            raise ValueError(msg)
        if self.processing not in {"instantaneous", "accumulation", "mean", "maximum", "minimum"}:
            msg = f"Unsupported processing type {self.processing!r}."
            raise ValueError(msg)
        if self.processing != "instantaneous" and self.interval is None:
            msg = f"processing={self.processing!r} requires explicit interval bounds."
            raise ValueError(msg)
        if self.interval is not None and self.processing == "instantaneous":
            msg = "Represented intervals do not apply to instantaneous processing."
            raise ValueError(msg)
        if self.interval is not None and (
            len(self.interval) != 2 or self.interval[1] <= self.interval[0] or self.interval[1] > timedelta(0)
        ):
            msg = "interval must have start < end <= 0 relative to valid time."
            raise ValueError(msg)
        if self.bbox is not None:
            west, south, east, north = self.bbox
            if not all(math.isfinite(item) for item in self.bbox) or not (
                -180 <= west < east <= 180 and -90 <= south < north <= 90
            ):
                msg = "bbox must be (west, south, east, north) in degrees."
                raise ValueError(msg)
        if self.grid_spacing_km is not None and self.grid_spacing_km <= 0:
            msg = "grid_spacing_km must be positive."
            raise ValueError(msg)
        if self.grid_spacing_km is not None and self.grid is None:
            msg = (
                "A kilometre spacing and bbox do not define a grid. Set grid to a registered native geometry "
                "or supply explicit output coordinates at inference."
            )
            raise ValueError(
                msg,
            )
        if self.spatial_support_km is not None and self.spatial_support_km < 0:
            msg = "spatial_support_km must not be negative."
            raise ValueError(msg)

    @classmethod
    def from_dict(  # noqa: C901
        cls,
        value: dict[str, Any],
        default_provenance: str | None = None,
    ) -> ForecastQuery:
        """Canonicalise the convenient user dictionary."""
        provenance = value.get("provenance", default_provenance)
        if provenance is None:
            msg = "Query provenance was omitted and no reference_provenance is configured."
            raise ValueError(msg)

        level_type = value.get("level_type")
        if level_type in {"pl", "isobaricInhPa", "isobaricInPa"}:
            level_type = "pressure"
        elif level_type in {"ml", "hybrid", "hybridLevel"}:
            msg = "A model-level index is not a physical query coordinate."
            raise ValueError(msg)
        elif level_type in {"sfc", "surface", None}:
            level_type = "surface" if value.get("level") is None else level_type

        pressure_pa = value.get("pressure_pa")
        if level_type == "pressure" and pressure_pa is None:
            level = value.get("level")
            units = value.get("level_units")
            if level is None or units is None:
                msg = "Pressure queries require level and level_units, or pressure_pa."
                raise ValueError(msg)
            if units.lower() in {"hpa", "mbar"}:
                pressure_pa = float(level) * 100
            elif units.lower() == "pa":
                pressure_pa = float(level)
            else:
                msg = f"Unsupported pressure units {units!r}; use Pa or hPa."
                raise ValueError(msg)

        interval = value.get("interval") or value.get("accumulation_interval")
        if interval is not None:
            if isinstance(interval, str):
                interval = (-_duration(interval), timedelta(0))
            else:
                interval = tuple(_duration(item) for item in interval)

        resolution = value.get("grid_spacing_km", value.get("resolution"))
        if isinstance(resolution, str):
            if not resolution.lower().endswith("km"):
                msg = "Only kilometre grid spacing strings are accepted; use e.g. '2.5km'."
                raise ValueError(msg)
            resolution = float(resolution[:-2])

        return cls(
            variable=str(value["variable"]),
            lead_time=_duration(value["lead_time"]),
            provenance=str(provenance),
            output_cadence=(
                None
                if value.get("frequency", value.get("output_cadence")) is None
                else _duration(value.get("frequency", value.get("output_cadence")))
            ),
            level_type=level_type,
            pressure_pa=pressure_pa,
            height_m=value.get("height_m"),
            processing=value.get(
                "processing",
                ("accumulation" if value.get("accumulation_interval") else "instantaneous"),
            ),
            interval=interval,
            bbox=(tuple(value["area"]) if value.get("area") is not None else value.get("bbox")),
            grid=value.get("grid"),
            grid_spacing_km=resolution,
            spatial_support_km=value.get("spatial_support_km"),
        )

    def as_serialisable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lead_time"] = f"{int(self.lead_time.total_seconds())}s"
        if self.output_cadence is not None:
            result["output_cadence"] = f"{int(self.output_cadence.total_seconds())}s"
        if self.interval is not None:
            result["interval"] = tuple(f"{int(item.total_seconds())}s" for item in self.interval)
        return result
