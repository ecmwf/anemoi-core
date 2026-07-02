# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Version 0.0 metadata schema.

This schema handles pre-``metadata_inference`` checkpoints natively.
It implements the :class:`~anemoi.metadata.base.MetadataContract` by reading
directly from ``data_indices``, ``config.data``, and ``dataset``.

These are legacy checkpoints written before the ``metadata_inference`` /
``schema_version`` fields were introduced.  They are always single-dataset
(the implicit dataset name is ``"data"``).

Key structural characteristics
-------------------------------
* Has ``data_indices.data.input.full``, ``data_indices.data.output.full``,
  ``data_indices.model.output.full``, ``data_indices.model.output.prognostic``
* Has ``config.data.forcing``, ``config.data.diagnostic``,
  ``config.data.timestep``
* Has ``config.training.multistep_input``, ``config.training.precision``
* Has ``dataset.variables`` (or nested ``dataset.data.variables``)
* Has ``dataset.variables_metadata`` (or nested)
* Has ``dataset.data_request`` (or nested)
* Has ``dataset.frequency`` or ``config.data.frequency``
* Has ``dataset.shape`` (format: ``[samples, variables, ensemble, grid_points]``)
* Has ``dataset.sources`` (or nested)
* Has ``dataset.arguments`` (or nested)
* Has **no** ``metadata_inference`` key
* Has **no** ``schema_version`` key
* Always single-dataset named ``"data"``
"""

from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from ..base import MetadataContract
from ..registry import MetadataRegistry

_DATASET_NAME = "data"


@MetadataRegistry.register("0.0")
class MetadataV0(MetadataContract):
    """Version 0.0 metadata schema for pre-``metadata_inference`` checkpoints.

    All fields are permissive (plain dicts) since the raw checkpoint format
    is untyped.

    Attributes
    ----------
    config : dict[str, Any]
        Full training configuration (permissive).
    data_indices : dict[str, Any]
        Raw data-index mappings as written by training.
    dataset : dict[str, Any]
        Dataset provenance and statistics (permissive).
    provenance_training : dict[str, Any]
        Training provenance information (permissive).
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    # All fields are optional / permissive -- the raw dict is untyped.
    config: dict[str, Any] = Field(default_factory=dict)
    data_indices: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    provenance_training: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _ds(self) -> dict[str, Any]:
        """Return the effective dataset section.

        Some legacy checkpoints nest the dataset data under a sub-key named
        after the dataset (e.g. ``dataset.data``).  If that sub-key exists
        and is a dict, use it; otherwise use the top-level ``dataset`` dict.

        Returns
        -------
        dict[str, Any]
            The effective dataset section.
        """
        ds = self.dataset
        nested = ds.get(_DATASET_NAME)
        if isinstance(nested, dict):
            return nested
        return ds

    def _variables(self) -> list[str]:
        """Return the ordered list of variable names from the dataset section.

        Returns
        -------
        list[str]
            Variable names as stored in ``dataset.variables``.
        """
        return self._ds.get("variables", [])

    def _data_input_full(self) -> list[int]:
        """Return ``data_indices.data.input.full``.

        Returns
        -------
        list[int]
            Dataset indices for each input tensor slot.
        """
        return self.data_indices.get("data", {}).get("input", {}).get("full", [])

    def _data_output_full(self) -> list[int]:
        """Return ``data_indices.data.output.full``.

        Returns
        -------
        list[int]
            Dataset indices for each output tensor slot.
        """
        return self.data_indices.get("data", {}).get("output", {}).get("full", [])

    def _model_output(self) -> dict[str, Any]:
        """Return ``data_indices.model.output``.

        Returns
        -------
        dict[str, Any]
            Model output index sub-dict.
        """
        return self.data_indices.get("model", {}).get("output", {})

    # ------------------------------------------------------------------
    # MetadataContract implementation
    # ------------------------------------------------------------------

    def get_dataset_names(self) -> list[str]:
        """Return ``["data"]`` -- V0 is always single-dataset.

        Returns
        -------
        list[str]
            Always ``["data"]``.
        """
        return [_DATASET_NAME]

    def get_task(self) -> str | None:
        """Return ``None`` -- V0 checkpoints have no task label.

        Returns
        -------
        str or None
            Always ``None``.
        """
        return None

    def get_timestep(self, dataset_name: str | None = None) -> str:
        """Return the timestep frequency string.

        Read from ``config.data.timestep``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        str
            Frequency string (e.g. ``"6h"``).
        """
        return self.config.get("data", {}).get("timestep", "6h")

    def get_variable_indices(self, dataset_name: str | None = None) -> dict[str, int]:
        """Return input variable name to tensor index mapping.

        Built from ``data_indices.data.input.full`` and ``dataset.variables``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the input tensor.
        """
        variables = self._variables()
        data_input_full = self._data_input_full()
        result: dict[str, int] = {}
        for pos, dataset_idx in enumerate(data_input_full):
            if dataset_idx < len(variables):
                result[variables[dataset_idx]] = pos
        return result

    def get_output_variable_indices(self, dataset_name: str | None = None) -> dict[str, int]:
        """Return output variable name to tensor index mapping.

        Built from ``data_indices.data.output.full`` and ``dataset.variables``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the output tensor.
        """
        variables = self._variables()
        data_output_full = self._data_output_full()
        result: dict[str, int] = {}
        for pos, dataset_idx in enumerate(data_output_full):
            if dataset_idx < len(variables):
                result[variables[dataset_idx]] = pos
        return result

    def get_variable_types(self, dataset_name: str | None = None) -> dict[str, list[str]]:
        """Return variable categories by role.

        Forcing and diagnostic names come from ``config.data``.  Prognostic
        names are derived from ``data_indices.model.output.prognostic``.
        Target names are all output variables.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, list[str]]
            Dictionary with keys ``"forcing"``, ``"prognostic"``,
            ``"diagnostic"``, and ``"target"``.
        """
        variables = self._variables()
        config_data = self.config.get("data", {})
        forcing_names: list[str] = config_data.get("forcing", [])
        diagnostic_names: list[str] = config_data.get("diagnostic", [])

        data_output_full = self._data_output_full()
        model_output = self._model_output()
        model_output_full: list[int] = model_output.get("full", [])
        model_output_prognostic: list[int] = model_output.get("prognostic", [])

        # Build output-position → variable-name map.
        output_pos_to_var: dict[int, str] = {}
        for pos, dataset_idx in enumerate(data_output_full):
            if dataset_idx < len(variables):
                output_pos_to_var[pos] = variables[dataset_idx]

        # Prognostic: model output prognostic indices → variable names.
        prognostic_names: list[str] = []
        for model_pos in model_output_prognostic:
            if model_pos < len(model_output_full):
                data_out_pos = model_output_full[model_pos]
                if data_out_pos in output_pos_to_var:
                    prognostic_names.append(output_pos_to_var[data_out_pos])

        # Target = all output variables.
        target_names: list[str] = [variables[idx] for idx in data_output_full if idx < len(variables)]

        return {
            "forcing": forcing_names,
            "diagnostic": diagnostic_names,
            "prognostic": prognostic_names,
            "target": target_names,
        }

    def get_input_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return input relative date indices.

        Derived from ``config.training.multistep_input``:
        ``[-n+1, ..., 0]`` where ``n = multistep_input``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        list[int]
            Relative date indices used as model inputs.
        """
        multistep_input: int = self.config.get("training", {}).get("multistep_input", 1)
        return list(range(-(multistep_input - 1), 1))

    def get_output_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return output relative date indices.

        V0 checkpoints always produce a single output step: ``[1]``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        list[int]
            Always ``[1]``.
        """
        return [1]

    def get_tensor_shapes(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return tensor shape metadata.

        Derived from ``dataset.shape`` and ``data_indices``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, Any]
            Shape metadata with keys ``"variables"``, ``"input_timesteps"``,
            ``"ensemble"``, and ``"grid"``.
        """
        shape: list[int] = self._ds.get("shape", [])
        # shape format: [samples, variables, ensemble, grid_points]
        grid: int | None = shape[-1] if len(shape) >= 4 else (shape[1] if len(shape) == 3 else None)
        multistep_input: int = self.config.get("training", {}).get("multistep_input", 1)
        return {
            "variables": len(self._data_input_full()),
            "input_timesteps": multistep_input,
            "ensemble": 1,
            "grid": grid,
        }

    def get_variables_metadata(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata from the dataset section.

        Applies the ``constant_fields`` patch (marks listed variables as
        ``constant_in_time``).

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of variable names to their metadata dicts.
        """
        ds = self._ds
        result = dict(ds.get("variables_metadata", {}))

        # Apply constant_fields patch (old inference behaviour).
        for name in ds.get("constant_fields", []):
            if name in result:
                result[name] = {**result[name], "constant_in_time": True}

        return result

    def get_grid_points(self, dataset_name: str | None = None) -> int | None:
        """Return the number of grid points.

        Read from the last element of ``dataset.shape`` (the 4-element shape
        ``[samples, variables, ensemble, grid_points]``).

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        int or None
            Number of grid points, or ``None`` if not recorded.
        """
        shape: list[int] = self._ds.get("shape", [])
        if len(shape) >= 4:
            return shape[-1]
        if len(shape) == 3:
            return shape[1]
        return None

    def get_data_request(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return data request parameters from the dataset section.

        Read from ``dataset.data_request`` (handles nesting).

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, Any]
            Data request parameters, or an empty dict if absent.
        """
        return self._ds.get("data_request", {})

    def get_precision(self) -> str | None:
        """Return the model precision string.

        Read from ``config.training.precision``.

        Returns
        -------
        str or None
            Precision string (e.g. ``"16-mixed"``), or ``None`` if absent.
        """
        training_cfg = self.config.get("training", {})
        if isinstance(training_cfg, dict):
            return training_cfg.get("precision")
        return None

    def get_provenance(self) -> dict[str, Any]:
        """Return the provenance section.

        Read from ``provenance_training``.

        Returns
        -------
        dict[str, Any]
            Provenance information, or an empty dict if absent.
        """
        return self.provenance_training

    def get_data_frequency(self, dataset_name: str | None = None) -> str | None:
        """Return the data frequency string.

        Read from ``dataset.frequency``; falls back to
        ``config.data.frequency``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        str or None
            Frequency string (e.g. ``"6h"``), or ``None`` if not recorded.
        """
        freq = self._ds.get("frequency")
        if freq is not None:
            return freq
        return self.config.get("data", {}).get("frequency")

    def get_sources(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        """Return source dataset configurations.

        Read from ``dataset.sources`` (handles nesting).

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        list[dict[str, Any]]
            Source dataset configurations, or an empty list if absent.
        """
        return self._ds.get("sources", [])

    def get_open_dataset_args(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return arguments for opening the training dataset.

        Read from ``dataset.arguments`` (handles nesting).

        Parameters
        ----------
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, Any]
            Dataset open arguments, or an empty dict if absent.
        """
        return self._ds.get("arguments", {})

    def get_dataloader_config(
        self,
        partition: str = "training",
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Return dataloader dataset configuration for a given partition.

        Read from ``config.dataloader.<partition>``.  V0 is single-dataset,
        so ``dataset_name`` is ignored.

        Parameters
        ----------
        partition : str, optional
            The partition name, by default ``"training"``.
        dataset_name : str | None, optional
            Ignored (V0 is single-dataset).

        Returns
        -------
        dict[str, Any]
            The dataloader dataset configuration, or an empty dict if absent.
        """
        dataloader = self.config.get("dataloader", {}).get(partition, {})
        if not isinstance(dataloader, dict):
            return {}
        return dataloader
