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


MODEL_TYPE_ALIASES = {
    "sfc": "sfc",
    "surface": "sfc",
    "meanSea": "sfc",
    "pl": "pl",
    "pressure": "pl",
    "isobaricInhPa": "pl",
    "isobaricInPa": "pl",
    "ml": "ml",
    "model": "ml",
    "hybrid": "ml",
    "hybridLevel": "ml",
    "height": "height",
    "hl": "height",
    "heightAboveGround": "height",
    "heightAboveSea": "height",
    "layer": "layer",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class ForecastQuery:
    """One scalar field requested at one valid time.

    ``model_type`` describes the vertical coordinate. ``level`` is interpreted
    with ``level_unit`` for pressure and height coordinates, and within
    ``provenance`` for model levels. ``unit`` is the physical unit of the
    requested value, not the vertical coordinate.
    """

    variable: str
    lead_time: timedelta
    provenance: str
    unit: str
    output_frequency: timedelta | None = None
    model_type: str = "sfc"
    level: int | float | None = None
    level_unit: str | None = None
    aggregation_type: str = "instantaneous"
    temporal_aggregation_window: tuple[timedelta, timedelta] | None = None
    bbox: tuple[float, float, float, float] | None = None
    grid: str | None = None
    grid_spacing_km: float | None = None
    spatial_support_km: float | None = None

    def __post_init__(self) -> None:  # noqa: C901
        model_type = MODEL_TYPE_ALIASES.get(self.model_type)
        if model_type is None:
            msg = f"Unsupported model_type {self.model_type!r}."
            raise ValueError(msg)
        object.__setattr__(self, "model_type", model_type)
        if not self.variable or not self.provenance or not self.unit:
            msg = "variable, provenance and unit must be resolved before loading a target."
            raise ValueError(msg)
        if self.lead_time < timedelta(0):
            msg = "lead_time must not be negative."
            raise ValueError(msg)
        if self.output_frequency is not None and self.output_frequency <= timedelta(0):
            msg = "output_frequency must be positive."
            raise ValueError(msg)
        if model_type in {"pl", "height"} and (self.level is None or self.level_unit is None):
            msg = f"model_type={model_type!r} requires level and level_unit."
            raise ValueError(msg)
        if model_type == "ml":
            if not isinstance(self.level, int) or self.level < 1:
                msg = "model_type='ml' requires a positive integer level."
                raise ValueError(msg)
            if self.level_unit not in {None, "1"}:
                msg = "A model-level index is dimensionless; omit level_unit or use '1'."
                raise ValueError(msg)
        if model_type == "sfc" and (self.level is not None or self.level_unit is not None):
            msg = "model_type='sfc' must not declare level or level_unit."
            raise ValueError(msg)
        if self.pressure_pa is not None and self.pressure_pa <= 0:
            msg = "Pressure level must be positive."
            raise ValueError(msg)
        if self.aggregation_type not in {"instantaneous", "accumulation", "mean", "maximum", "minimum"}:
            msg = f"Unsupported aggregation_type {self.aggregation_type!r}."
            raise ValueError(msg)
        if self.aggregation_type != "instantaneous" and self.temporal_aggregation_window is None:
            msg = f"aggregation_type={self.aggregation_type!r} requires temporal_aggregation_window."
            raise ValueError(msg)
        if self.temporal_aggregation_window is not None and self.aggregation_type == "instantaneous":
            msg = "temporal_aggregation_window does not apply to instantaneous fields."
            raise ValueError(msg)
        if self.temporal_aggregation_window is not None and (
            len(self.temporal_aggregation_window) != 2
            or self.temporal_aggregation_window[1] <= self.temporal_aggregation_window[0]
            or self.temporal_aggregation_window[1] > timedelta(0)
        ):
            msg = "temporal_aggregation_window must have start < end <= 0 relative to valid time."
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
            raise ValueError(msg)
        if self.spatial_support_km is not None and self.spatial_support_km < 0:
            msg = "spatial_support_km must not be negative."
            raise ValueError(msg)

    @property
    def level_type(self) -> str:
        """Compatibility name used inside the current catalogue and model."""
        return {"sfc": "surface", "pl": "pressure", "ml": "model"}.get(self.model_type, self.model_type)

    @property
    def pressure_pa(self) -> float | None:
        if self.model_type != "pl" or self.level is None or self.level_unit is None:
            return None
        unit = self.level_unit.lower()
        if unit in {"hpa", "mbar"}:
            return float(self.level) * 100
        if unit == "pa":
            return float(self.level)
        msg = f"Unsupported pressure level_unit {self.level_unit!r}; use Pa or hPa."
        raise ValueError(msg)

    @property
    def height_m(self) -> float | None:
        if self.model_type not in {"height", "layer"} or self.level is None or self.level_unit is None:
            return None
        unit = self.level_unit.lower()
        if unit in {"m", "metre", "meter", "metres", "meters"}:
            return float(self.level)
        if unit in {"km", "kilometre", "kilometer", "kilometres", "kilometers"}:
            return float(self.level) * 1000
        msg = f"Unsupported height level_unit {self.level_unit!r}; use m or km."
        raise ValueError(msg)

    @property
    def model_level(self) -> int | None:
        return int(self.level) if self.model_type == "ml" and self.level is not None else None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        default_provenance: str | None = None,
    ) -> ForecastQuery:
        """Canonicalise preferred query names and accepted legacy aliases."""
        provenance = value.get("provenance", default_provenance)
        if provenance is None:
            msg = "Query provenance was omitted and no reference_provenance is configured."
            raise ValueError(msg)
        unit = value.get("unit", value.get("units"))
        if unit is None:
            msg = "Query unit is required; use the physical value unit or explicit 'unknown'."
            raise ValueError(msg)
        model_type = value.get("model_type", value.get("level_type", "sfc"))
        model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
        level = value.get("level")
        level_unit = value.get("level_unit", value.get("level_units"))
        if model_type == "pl" and level is None and value.get("pressure_pa") is not None:
            level, level_unit = value["pressure_pa"], "Pa"
        if model_type in {"height", "layer"} and level is None and value.get("height_m") is not None:
            level, level_unit = value["height_m"], "m"
        if model_type == "ml" and level is None:
            level = value.get("model_level")

        window = value.get(
            "temporal_aggregation_window",
            value.get("interval", value.get("accumulation_interval")),
        )
        if window is not None:
            if isinstance(window, str):
                window = (-_duration(window), timedelta(0))
            else:
                window = tuple(_duration(item) for item in window)
        aggregation_type = value.get(
            "aggregation_type",
            value.get("processing", "accumulation" if value.get("accumulation_interval") else "instantaneous"),
        )
        resolution = value.get("grid_spacing_km", value.get("resolution"))
        if isinstance(resolution, str):
            if not resolution.lower().endswith("km"):
                msg = "Only kilometre grid spacing strings are accepted; use e.g. '2.5km'."
                raise ValueError(msg)
            resolution = float(resolution[:-2])
        frequency = value.get(
            "output_frequency",
            value.get("frequency", value.get("output_cadence")),
        )
        return cls(
            variable=str(value["variable"]),
            lead_time=_duration(value["lead_time"]),
            provenance=str(provenance),
            unit=str(unit),
            output_frequency=None if frequency is None else _duration(frequency),
            model_type=str(model_type),
            level=level,
            level_unit=level_unit,
            aggregation_type=str(aggregation_type),
            temporal_aggregation_window=window,
            bbox=(tuple(value["area"]) if value.get("area") is not None else value.get("bbox")),
            grid=value.get("grid"),
            grid_spacing_km=resolution,
            spatial_support_km=value.get("spatial_support_km"),
        )

    def as_serialisable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lead_time"] = f"{int(self.lead_time.total_seconds())}s"
        if self.output_frequency is not None:
            result["output_frequency"] = f"{int(self.output_frequency.total_seconds())}s"
        if self.temporal_aggregation_window is not None:
            result["temporal_aggregation_window"] = tuple(
                f"{int(item.total_seconds())}s" for item in self.temporal_aggregation_window
            )
        return result
