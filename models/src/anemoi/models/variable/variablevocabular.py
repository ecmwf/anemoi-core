# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from dataclasses import dataclass
from typing import Any, Literal

import torch

VerticalCoordinateType = Literal[
    "surface",
    "pressure",
    "heightaboveground",
    "model",
    "layer",
]

TemporalOperator = Literal[
    "instantaneous",
    "accumulation",
    "mean",
    "maximum",
    "minimum",
]


@dataclass(frozen=True)
class VerticalCoordinate:
    type: VerticalCoordinateType
    level: int | float | None = None
    unit: str | None = None


@dataclass(frozen=True)
class VariableSpecification:
    name: str
    param: str
    vertical_coordinate: VerticalCoordinate
    temporal_operator: TemporalOperator
    temporal_window: int | float | None = None


@dataclass(frozen=True)
class DomainVariableMetadata:
    # Categorical
    param_ids: torch.Tensor
    vertical_type_ids: torch.Tensor
    temporal_operator_ids: torch.Tensor

    # Continuous
    vertical_levels: torch.Tensor
    temporal_windows: torch.Tensor

    # Missing-value masks
    has_vertical_level: torch.Tensor
    has_temporal_window: torch.Tensor


class VariableVocabulary:
    def __init__(
        self,
        specifications: dict[str, VariableSpecification],
        param_to_id: dict[str, int],
        vertical_type_to_id: dict[str, int],
        temporal_operator_to_id: dict[str, int],
    ) -> None:
        self.specification_by_name = specifications

        self.param_to_id = param_to_id
        self.vertical_type_to_id = vertical_type_to_id
        self.temporal_operator_to_id = temporal_operator_to_id

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_temporal_information(
        mars: dict[str, Any],
    ) -> tuple[float | None, TemporalOperator]:
        """Extract temporal semantics from MARS metadata.

        NOTE:
        This is intentionally conservative for now. The exact mapping of
        accumulation / mean / maximum / minimum should eventually use
        explicit metadata rather than guessing from the presence of step.
        """
        time = mars.get("time")
        step = mars.get("step")

        if time == 0 and step == 0:
            return None, "instantaneous"

        if time is None and step is None:
            return None, "instantaneous"

        # TODO:
        # Replace this with explicit temporal-operator metadata when
        # available.
        temporal_operator: TemporalOperator = "accumulation"

        # TODO:
        # Canonicalize the temporal window to a known physical unit
        # such as seconds once the exact semantics of `step` are known.
        temporal_window = float(time) if step is not None else None

        return temporal_window, temporal_operator

    @staticmethod
    def _fetch_height(
        mars: dict[str, Any],
    ) -> tuple[str | None, int | None]:
        """Extract height from surface parameters such as 2t and 10u.

        Examples:
            2t  -> ("t", 2)
            10u -> ("u", 10)
            sp  -> (None, None)
        """
        if mars.get("levtype") != "sfc":
            return None, None

        match = re.match(
            r"^(\d+)(.+)$",
            mars.get("param", ""),
        )

        if match is None:
            return None, None

        height, param = match.groups()

        return param, int(height)

    @staticmethod
    def _fetch_level_and_type(
        mars: dict[str, Any],
    ) -> tuple[
        str | None,
        float | None,
        VerticalCoordinateType,
        str | None,
    ]:
        """Extract the physical vertical-coordinate description."""

        level = mars.get("levelist")
        levtype = mars.get("levtype")

        # --------------------------------------------------------------
        # Pressure-level variable
        #
        # Example:
        #   t_850 -> param=t, type=pressure, level=850, unit=hPa
        # --------------------------------------------------------------
        if levtype == "pl" and level is not None:
            return (
                None,
                float(level),
                "pressure",
                "hPa",
            )

        # --------------------------------------------------------------
        # Surface or height-above-surface variable
        #
        # Examples:
        #   2t  -> param=t, type=height, level=2, unit=m
        #   10u -> param=u, type=height, level=10, unit=m
        #   sp  -> param=sp, type=surface, level=None
        # --------------------------------------------------------------
        if levtype == "sfc":
            param_group, height = VariableVocabulary._fetch_height(mars)

            if height is not None:
                return (
                    param_group,
                    float(height),
                    "heightaboveground",
                    "m",
                )

            return (
                None,
                None,
                "surface",
                None,
            )

        # --------------------------------------------------------------
        # TODO: Explicit handling for model/layer coordinates.
        #
        # Until metadata for those coordinates is available, variables
        # without a known vertical coordinate are treated as surface-like
        # / non-applicable vertical coordinates.
        # --------------------------------------------------------------
        return (
            None,
            None,
            "surface",
            None,
        )

    @staticmethod
    def make_specification(
        name: str,
        metadata: dict[str, dict[str, Any]],
    ) -> VariableSpecification:
        variable_metadata = metadata[name]
        mars = variable_metadata.get(
            "mars",
            variable_metadata,
        )

        if mars.get("levtype") is None:
            # skip forcings
            return None
        (
            param_group,
            level,
            vertical_type,
            unit,
        ) = VariableVocabulary._fetch_level_and_type(mars)

        vertical_coordinate = VerticalCoordinate(
            type=vertical_type,
            level=level,
            unit=unit,
        )

        (
            temporal_window,
            temporal_operator,
        ) = VariableVocabulary.fetch_temporal_information(mars)

        return VariableSpecification(
            name=name,
            param=(param_group if param_group is not None else mars.get("param", name)),
            vertical_coordinate=vertical_coordinate,
            temporal_operator=temporal_operator,
            temporal_window=temporal_window,
        )

    # ------------------------------------------------------------------
    # Foundation vocabulary construction
    # ------------------------------------------------------------------

    @classmethod
    def from_foundation(
        cls,
        data_indices: Any,
        metadata: dict[str, dict[str, Any]],
    ) -> "VariableVocabulary":
        specifications: dict[str, VariableSpecification] = {}

        all_params: set[str] = set()
        all_vertical_types: set[str] = set()
        all_temporal_operators: set[str] = set()

        names: set[str] = set()

        # Foundation vocabulary is the union of model input/output fields.
        for io in ["input", "output"]:
            current_data_indices = getattr(
                data_indices.model,
                io,
            )

            names.update(current_data_indices.name_to_index.keys())

        for name in sorted(names):
            if name not in metadata:
                raise KeyError(f"Missing metadata for variable {name}")

            spec = cls.make_specification(
                name=name,
                metadata=metadata,
            )
            if spec is None:
                continue

            specifications[name] = spec

            all_params.add(spec.param)

            all_vertical_types.add(spec.vertical_coordinate.type)

            all_temporal_operators.add(spec.temporal_operator)

        # IDs are established once for the foundation model.
        #
        # On transfer/resume these mappings should be restored from the
        # checkpoint rather than rebuilt from a domain subset.
        param_to_id = {param: i for i, param in enumerate(sorted(all_params))}

        vertical_type_to_id = {
            vertical_type: i
            for i, vertical_type in enumerate(sorted(all_vertical_types))
        }

        temporal_operator_to_id = {
            temporal_operator: i
            for i, temporal_operator in enumerate(sorted(all_temporal_operators))
        }

        return cls(
            specifications=specifications,
            param_to_id=param_to_id,
            vertical_type_to_id=vertical_type_to_id,
            temporal_operator_to_id=temporal_operator_to_id,
        )

    # ------------------------------------------------------------------
    # Runtime variable selection
    # ------------------------------------------------------------------

    def get_variables(
        self,
        names: str | list[str],
    ) -> DomainVariableMetadata:
        if isinstance(names, str):
            names = [names]

        unknown = [name for name in names if name not in self.specification_by_name]

        if unknown:
            raise KeyError(f"Unknown variable in global vocabulary: {unknown}")

        specs = [self.specification_by_name[name] for name in names]

        return self.tensorize(specs)

    # ------------------------------------------------------------------
    # Tensorization
    # ------------------------------------------------------------------

    def tensorize(
        self,
        specs: list[VariableSpecification],
    ) -> DomainVariableMetadata:
        return DomainVariableMetadata(
            # ----------------------------------------------------------
            # Categorical
            # ----------------------------------------------------------
            param_ids=torch.tensor(
                [self.param_to_id[spec.param] for spec in specs],
                dtype=torch.long,
            ),
            vertical_type_ids=torch.tensor(
                [
                    self.vertical_type_to_id[spec.vertical_coordinate.type]
                    for spec in specs
                ],
                dtype=torch.long,
            ),
            temporal_operator_ids=torch.tensor(
                [
                    self.temporal_operator_to_id[spec.temporal_operator]
                    for spec in specs
                ],
                dtype=torch.long,
            ),
            # ----------------------------------------------------------
            # Continuous
            # ----------------------------------------------------------
            vertical_levels=torch.tensor(
                [
                    (
                        0.0
                        if spec.vertical_coordinate.level is None
                        else float(spec.vertical_coordinate.level)
                    )
                    for spec in specs
                ],
                dtype=torch.float32,
            ),
            temporal_windows=torch.tensor(
                [
                    (
                        0.0
                        if spec.temporal_window is None
                        else float(spec.temporal_window)
                    )
                    for spec in specs
                ],
                dtype=torch.float32,
            ),
            # ----------------------------------------------------------
            # Missing-value masks
            # ----------------------------------------------------------
            has_vertical_level=torch.tensor(
                [spec.vertical_coordinate.level is not None for spec in specs],
                dtype=torch.bool,
            ),
            has_temporal_window=torch.tensor(
                [spec.temporal_window is not None for spec in specs],
                dtype=torch.bool,
            ),
        )

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_to_id": dict(self.param_to_id),
            "vertical_type_to_id": dict(self.vertical_type_to_id),
            "temporal_operator_to_id": dict(self.temporal_operator_to_id),
            "specifications": {
                name: {
                    "name": spec.name,
                    "param": spec.param,
                    "vertical_coordinate": {
                        "type": (spec.vertical_coordinate.type),
                        "level": (spec.vertical_coordinate.level),
                        "unit": (spec.vertical_coordinate.unit),
                    },
                    "temporal_operator": (spec.temporal_operator),
                    "temporal_window": (spec.temporal_window),
                }
                for name, spec in self.specification_by_name.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        state: dict[str, Any],
    ) -> "VariableVocabulary":
        specifications: dict[
            str,
            VariableSpecification,
        ] = {}

        for name, spec in state["specifications"].items():
            vertical_coordinate = VerticalCoordinate(
                type=spec["vertical_coordinate"]["type"],
                level=spec["vertical_coordinate"]["level"],
                unit=spec["vertical_coordinate"]["unit"],
            )

            specifications[name] = VariableSpecification(
                name=spec["name"],
                param=spec["param"],
                vertical_coordinate=vertical_coordinate,
                temporal_operator=spec["temporal_operator"],
                temporal_window=spec["temporal_window"],
            )

        return cls(
            specifications=specifications,
            param_to_id=dict(state["param_to_id"]),
            vertical_type_to_id=dict(state["vertical_type_to_id"]),
            temporal_operator_to_id=dict(state["temporal_operator_to_id"]),
        )
