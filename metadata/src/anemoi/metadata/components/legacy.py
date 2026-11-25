# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Legacy component adapters (component version ``0.1``).

Checkpoints written before ``metadata_inference`` existed (formats F0, the
ai-models era, and F1, the first anemoi-training layout) carry no
``metadata_inference`` block.  They are read through the same data and task
component adapters as every other format: each side is a version reader
subclassing the ``0.2`` one, combined with the data types by
``register_data_version`` (gridded only) and with the task names by
``register_task_version``.  Each overrides only what differs:

* the checkpoint is always a single gridded dataset called ``"data"``;
* variable indices, variable types and the prognostic variables are derived
  from ``data_indices.data.*``, ``data_indices.model.output.*``,
  ``config.data.*`` and the list ``dataset.variables``;
* the ``dataset`` dump is flat (or nested under ``dataset.data``) and may lack
  ``variables_metadata``, ``data_request`` and ``shape``, which are repaired,
  searched for or derived (port of anemoi-inference ``legacy.py``, cited
  below);
* the task has no recorded name or timesteps: it is a forecaster whose step
  counts come from ``config.training.multistep_input``.

Layout of a legacy checkpoint:

* ``data_indices.data.input.full``, ``data_indices.data.output.full``,
  ``data_indices.model.output.full`` and ``.prognostic`` (positions into
  ``dataset.variables``);
* ``config.data.forcing``, ``config.data.diagnostic`` (may be ``null``) and
  ``config.data.timestep``;
* ``config.training.multistep_input`` and ``config.training.precision``;
* ``dataset.variables``, ``variables_metadata``, ``frequency``, ``shape``
  (``[samples, variables, ensemble, grid points]``), ``sources``, ``arguments``
  and a ``data_request`` somewhere in the nested dataset description.
"""

import copy
import logging
import warnings
from collections.abc import Mapping
from typing import Any

from .base import ComponentAdapter
from .data import GRIDDED
from .data import DataReaderV0_2
from .data import _as_dict
from .data import register_data_version
from .task import TaskReaderV0_2
from .task import register_task_version

LOG = logging.getLogger(__name__)

#: The only dataset of a legacy checkpoint.
LEGACY_DATASET_NAME = "data"

#: Timestep assumed when a legacy checkpoint records none (ai-models era checkpoints are 6-hourly).
DEFAULT_TIMESTEP = "6h"

#: Grid points of the grids legacy checkpoints name in their data request (``legacy.py:181``).
GRID_POINTS = {"o96": 40_320, "n320": 542_080}

# Variables whose metadata is known by name (``legacy.py:62-115``).
_TIME_DEPENDENT_FORCINGS = (
    "insolation",
    "cos_solar_zenith_angle",
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
)
_CONSTANT_FORCINGS = ("cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude")
_CONSTANT_SURFACE = ("z", "lsm", "sdor", "slor", "cl", "cvh", "cvl", "slt", "tvh", "tvl")
_ACCUMULATED_SURFACE = ("cp", "tp", "sf", "ro", "ssrd", "strd")
_INSTANTANEOUS_SURFACE = ("sp", "msl", "10u", "10v", "2t", "2d", "skt", "tcw")


def _synthesise_variable(variable: str) -> tuple[dict[str, Any], bool]:
    """Synthesise the ``variables_metadata`` entry of one legacy variable from its name.

    Parameters
    ----------
    variable : str
        Variable name, such as ``"t_850"`` or ``"2t"``.

    Returns
    -------
    tuple[dict[str, Any], bool]
        The entry, and whether the name is a known one (an unknown name is assumed to be an ``an/sfc`` field).
    """
    if variable in _TIME_DEPENDENT_FORCINGS:
        return {"computed_forcing": True, "constant_in_time": False}, True

    if variable in _CONSTANT_FORCINGS:
        return {"computed_forcing": True, "constant_in_time": True}, True

    param_level = variable.split("_")
    if len(param_level) == 2:
        # Assumes levtype=pl for now (legacy.py:79).
        try:
            levelist = int(param_level[1])
        except ValueError:
            pass
        else:
            return {"mars": {"param": param_level[0], "levelist": levelist, "levtype": "pl"}}, True

    if variable in _CONSTANT_SURFACE:
        return {"mars": {"param": variable, "levtype": "sfc"}, "constant_in_time": True}, True

    if variable in _ACCUMULATED_SURFACE:
        mars = {"param": variable, "levtype": "sfc", "type": "fc", "stream": "oper"}
        return {"mars": mars, "process": "accumulation", "period": [0, 6]}, True

    return {"mars": {"param": variable, "levtype": "sfc"}}, variable in _INSTANTANEOUS_SURFACE


class _LegacyMixin(ComponentAdapter):
    """What both legacy components share: one implicit dataset called ``"data"``."""

    def dataset_names(self) -> list[str]:
        """Return ``["data"]``: legacy checkpoints are always single-dataset."""
        return [LEGACY_DATASET_NAME]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


class DataReaderV0_1(DataReaderV0_2):
    """Version reader of legacy checkpoints (no ``metadata_inference``): always one gridded dataset."""

    def __init__(self, raw: Mapping[str, Any], dataset_name: str) -> None:
        super().__init__(raw, dataset_name)
        self._repaired_variables_metadata: dict[str, dict[str, Any]] | None = None

    def dataset_names(self) -> list[str]:
        """Return ``["data"]``: legacy checkpoints are always single-dataset."""
        return [LEGACY_DATASET_NAME]

    # ------------------------------------------------------------------
    # Variables and indices, from data_indices and dataset.variables
    # ------------------------------------------------------------------

    def _variable_names(self) -> list[str]:
        """Return the ordered variable names of ``dataset.variables``."""
        names = self._dataset_section().get("variables")
        return list(names) if isinstance(names, list) else []

    def _indices(self, *path: str) -> list[int]:
        """Return the list at ``data_indices.<path>``, or an empty list."""
        node: Any = self._raw.get("data_indices")
        for key in path:
            node = _as_dict(node).get(key)
        return [i for i in node if isinstance(i, int) and not isinstance(i, bool)] if isinstance(node, list) else []

    def _named_positions(self, full: list[int]) -> dict[str, int]:
        """Map ``dataset.variables[i]`` to the position of ``i`` in *full*."""
        variables = self._variable_names()
        return {variables[index]: pos for pos, index in enumerate(full) if 0 <= index < len(variables)}

    def variable_indices(self) -> dict[str, int]:
        """Return the input variable name to tensor index mapping.

        Built from ``data_indices.data.input.full`` and ``dataset.variables``.
        """
        return self._named_positions(self._indices("data", "input", "full"))

    def output_variable_indices(self) -> dict[str, int]:
        """Return the output variable name to tensor index mapping.

        Built from ``data_indices.data.output.full`` and ``dataset.variables``.
        """
        return self._named_positions(self._indices("data", "output", "full"))

    def variable_types(self) -> dict[str, list[str]]:
        """Return variable names by role.

        Forcing and diagnostic names come from ``config.data``; ``diagnostic:
        null`` gives an empty list.  Prognostic names are derived from
        ``data_indices.model.output.prognostic``.  Target names are all output
        variables.
        """
        variables = self._variable_names()
        config_data = _as_dict(self._section("config").get("data"))
        data_output_full = self._indices("data", "output", "full")
        model_output_full = self._indices("model", "output", "full")

        output_position_to_variable = {
            pos: variables[index] for pos, index in enumerate(data_output_full) if 0 <= index < len(variables)
        }
        prognostic = []
        for model_position in self._indices("model", "output", "prognostic"):
            if model_position < len(model_output_full):
                name = output_position_to_variable.get(model_output_full[model_position])
                if name is not None:
                    prognostic.append(name)

        return {
            "forcing": list(config_data.get("forcing") or []),
            "prognostic": prognostic,
            "diagnostic": list(config_data.get("diagnostic") or []),
            "target": [variables[index] for index in data_output_full if 0 <= index < len(variables)],
        }

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _recorded_grid_size(self) -> int | None:
        """Return the number of grid points.

        As :meth:`DataReaderV0_2._recorded_grid_size` (``dataset.shape``), and
        when there is no recorded shape the point count of the grid named in the
        data request (``O96`` or ``N320``).  ``None`` for any other grid.
        """
        size = super()._recorded_grid_size()
        if size is not None:
            return size
        # anemoi-inference legacy.py:172-183 (_legacy_number_of_grid_points).
        grid = self.grid()
        return GRID_POINTS.get(grid.lower()) if isinstance(grid, str) else None

    # ------------------------------------------------------------------
    # Variables metadata repairs
    # ------------------------------------------------------------------

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata, repaired for legacy checkpoints.

        * When the checkpoint records none, it is synthesised from the variable
          names (``legacy.py:50-124``, ``126-136``), with a warning.
        * A ``param`` without ``mars`` gets a ``mars`` entry; the computed
          forcings (insolation, trigonometric time and position fields) get
          their ``computed_forcing`` and ``constant_in_time`` flags
          (``legacy.py:138-170``).
        * ``accumulated: true`` marks an accumulation (``process`` is set to
          ``"accumulation"`` when absent).

        The repair runs once and is cached; the adapter never changes the raw
        metadata.
        """
        if self._repaired_variables_metadata is None:
            stored = super().variables_metadata()
            self._repaired_variables_metadata = self._synthesise() if not stored else self._repair(stored)
        return copy.deepcopy(self._repaired_variables_metadata)

    def _synthesise(self) -> dict[str, dict[str, Any]]:
        """Build ``variables_metadata`` from the variable names (anemoi-inference ``legacy.py:50-124``)."""
        warnings.warn(
            "Legacy checkpoint without variables_metadata: synthesising it from the variable names; "
            "please try to patch your weights.",
            UserWarning,
            stacklevel=4,
        )
        result: dict[str, dict[str, Any]] = {}
        unknown = []
        for variable in self.variables():
            result[variable], known = _synthesise_variable(variable)
            if not known:
                unknown.append(variable)

        if unknown:
            warnings.warn(f"Unknown variables: {unknown}, assuming an/sfc", UserWarning, stacklevel=4)
        return result

    def _repair(self, variables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Backfill ``mars`` and repair the computed forcings (anemoi-inference ``legacy.py:138-170``)."""
        old_format_seen = False

        def old_format() -> None:
            nonlocal old_format_seen
            if not old_format_seen:
                LOG.warning("Old metadata format detected. Please update your weights.")
                old_format_seen = True

        for variable, metadata in variables.items():
            if "mars" not in metadata and "param" in metadata:
                old_format()
                metadata["mars"] = metadata.copy()

            if variable in _TIME_DEPENDENT_FORCINGS:
                expected = {"computed_forcing": True, "constant_in_time": False}
            elif variable in _CONSTANT_FORCINGS:
                expected = {"computed_forcing": True, "constant_in_time": True}
            else:
                expected = {}
            if any(metadata.get(key) is not value for key, value in expected.items()):
                old_format()
                metadata.update(expected)

            # Old ai-models files flag accumulations with ``accumulated: true``; the
            # components read ``process``.
            if metadata.get("accumulated") is True and "process" not in metadata:
                metadata["process"] = "accumulation"
        return variables


# --------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------


class TaskReaderV0_1(_LegacyMixin, TaskReaderV0_2):
    """Version reader of legacy checkpoints: no recorded timesteps, one input step by default."""

    def _default_n_step_input(self) -> int | None:
        """``config.training.multistep_input``; one input step when it is not recorded."""
        return super()._default_n_step_input() or 1

    def _config_timestep(self) -> Any:
        """``config.data.timestep``; :data:`DEFAULT_TIMESTEP` when it is not recorded."""
        timestep = super()._config_timestep()
        return timestep if isinstance(timestep, str) else DEFAULT_TIMESTEP


# Legacy checkpoints are all gridded.
_DATA_V0_1 = register_data_version("0.1", DataReaderV0_1, names=[GRIDDED])

GriddedV0_1 = _DATA_V0_1[GRIDDED]

# Legacy checkpoints were all trained for the forecaster task; the generic adapter covers any other name.
_V0_1 = register_task_version("0.1", TaskReaderV0_1, names=[None, "forecaster"])

TaskV0_1 = _V0_1[None]
ForecasterV0_1 = _V0_1["forecaster"]
