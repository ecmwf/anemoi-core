# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Version 1.0 metadata schema.

This schema captures the inference-relevant metadata written by training and
consumed by inference.  The top-level container is :class:`MetadataV1`, which
holds a strictly-typed :class:`InferenceMetadata` block alongside permissive
``dict`` sections for training, dataset, environment, and provenance data that
do not need to be validated at this layer.

V1 only handles checkpoints that already have a ``metadata_inference`` block.
Legacy checkpoints (no ``metadata_inference``, no ``schema_version``) are
handled by :class:`~anemoi.metadata.versions.v0.MetadataV0`.  A migration
from V0 to V1 is registered in
:mod:`anemoi.metadata.migrations.v0_to_v1`.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from ..base import MetadataContract
from ..registry import MetadataRegistry


class TimestepConfig(BaseModel):
    """Temporal stepping configuration for a dataset.

    Captures the frequency string and the relative date index arrays that
    describe which input/output time-steps the model was trained with.

    Attributes
    ----------
    timestep : str
        Frequency string, e.g. ``"6h"``.
    input_relative_date_indices : list[int]
        Relative date indices used as model inputs (e.g. ``[-1, 0]``).
    output_relative_date_indices : list[int]
        Relative date indices produced as model outputs (e.g. ``[1]``).
    relative_date_indices_training : list[int]
        Full set of relative date indices seen during training.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestep: str
    input_relative_date_indices: list[int]
    output_relative_date_indices: list[int]
    relative_date_indices_training: list[int]


class DataIndices(BaseModel):
    """Mapping from variable names to tensor indices.

    Attributes
    ----------
    input : dict[str, int]
        Mapping of variable name to its index in the input tensor.
    output : dict[str, int]
        Mapping of variable name to its index in the output tensor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: dict[str, int]
    output: dict[str, int]


class VariableTypes(BaseModel):
    """Categorisation of variables by their role in the model.

    Attributes
    ----------
    forcing : list[str]
        Variables that are provided as external forcings (not predicted).
    target : list[str]
        Variables that the model is trained to predict.
    prognostic : list[str]
        Variables that are both input and output (stepped forward in time).
    diagnostic : list[str]
        Variables that are output-only diagnostics (not fed back as input).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    forcing: list[str] = Field(default_factory=list)
    target: list[str] = Field(default_factory=list)
    prognostic: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)


class TensorShapes(BaseModel):
    """Shape metadata for the model's input/output tensors.

    Attributes
    ----------
    variables : int
        Number of variables (channels) in the tensor.
    input_timesteps : int
        Number of input time-steps stacked along the time dimension.
    ensemble : int
        Ensemble size; defaults to ``1`` for deterministic models.
    grid : int or None
        Number of grid points, or ``None`` when not applicable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variables: int
    input_timesteps: int
    ensemble: int = 1
    grid: int | None = None


class DatasetInferenceConfig(BaseModel):
    """Inference configuration for a single named dataset.

    Bundles together the index mappings, variable categorisation, temporal
    stepping, and tensor-shape information needed by inference for one dataset.

    Attributes
    ----------
    data_indices : DataIndices
        Variable-name-to-tensor-index mappings for input and output.
    variable_types : VariableTypes
        Categorisation of variables by role.
    timesteps : TimestepConfig
        Temporal stepping configuration.
    shapes : TensorShapes
        Tensor shape metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_indices: DataIndices
    variable_types: VariableTypes
    timesteps: TimestepConfig
    shapes: TensorShapes


class InferenceMetadata(BaseModel):
    """Top-level inference metadata written by training.

    This model is the authoritative source of truth consumed by inference at
    runtime.  It supports two input shapes:

    1. **Structured** - a dict that already contains a ``"datasets"`` key
       mapping dataset names to their per-dataset configs.
    2. **Flat** - a dict where scalar fields (``seed``, ``run_id``, ``task``,
       ``dataset_names``) sit alongside per-dataset sub-dicts keyed by the
       names listed in ``dataset_names``.  The ``@model_validator`` reshapes
       this into the structured form before validation.

    Unknown extra keys at the root level are preserved (``extra="allow"``) so
    that future fields added by training do not break existing inference code.

    Attributes
    ----------
    seed : int
        Random seed used during training.
    run_id : str
        Unique identifier for the training run.
    task : str or None
        Optional task label (e.g. ``"forecaster"``).
    dataset_names : list[str]
        Ordered list of dataset names referenced by this checkpoint.
    datasets : dict[str, DatasetInferenceConfig]
        Per-dataset inference configuration, keyed by dataset name.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    seed: int
    run_id: str
    task: str | None = None
    dataset_names: list[str]
    datasets: dict[str, DatasetInferenceConfig]

    @model_validator(mode="before")
    @classmethod
    def _restructure_flat_checkpoint(cls, values: Any) -> Any:
        """Reshape a flat checkpoint dict into the structured form.

        If the incoming data already contains a ``"datasets"`` key the dict is
        returned unchanged.  Otherwise the validator extracts the known scalar
        fields and collects the remaining keys that are listed in
        ``dataset_names`` into a ``"datasets"`` sub-dict.

        Parameters
        ----------
        values : Any
            Raw input passed to the model constructor.

        Returns
        -------
        Any
            Either the original value (if not a dict or already structured) or
            a restructured dict with a ``"datasets"`` key.
        """
        if not isinstance(values, dict):
            return values

        # Already in structured form - nothing to do.
        if "datasets" in values:
            return values

        _SCALAR_KEYS = {"seed", "run_id", "task", "dataset_names"}

        dataset_names: list[str] = values.get("dataset_names", [])

        scalars: dict[str, Any] = {k: v for k, v in values.items() if k in _SCALAR_KEYS}

        missing = [k for k in dataset_names if k not in values]
        if missing:
            msg = (
                f"metadata_inference references datasets {missing} in "
                f"'dataset_names' but no corresponding entries exist in the metadata."
            )
            raise ValueError(msg)

        datasets: dict[str, Any] = {k: values[k] for k in dataset_names}

        # Preserve any extra keys that are neither scalars nor dataset entries
        # so that extra="allow" can capture them at the root level.
        extras: dict[str, Any] = {k: v for k, v in values.items() if k not in _SCALAR_KEYS and k not in dataset_names}

        return {**scalars, "datasets": datasets, **extras}


@MetadataRegistry.register("1.0")
class MetadataV1(MetadataContract):
    """Version 1.0 metadata schema.

    The top-level container for all checkpoint metadata.  The
    ``metadata_inference`` field is strictly validated; all other sections
    (``config``, ``training``, ``dataset``, ``environment``, ``provenance``)
    are stored as plain dicts to avoid breaking changes when training adds new
    fields.

    Extra top-level keys are preserved (``extra="allow"``) for forward
    compatibility.

    V1 only handles checkpoints that already have a ``metadata_inference``
    block.  Legacy checkpoints (no ``metadata_inference``, no
    ``schema_version``) are handled by
    :class:`~anemoi.metadata.versions.v0.MetadataV0`.

    Attributes
    ----------
    created_at : datetime
        Timestamp when the metadata was created.
    metadata_inference : InferenceMetadata
        Strictly-typed inference metadata block.
    config : dict[str, Any]
        Full training configuration (permissive).
    training : dict[str, Any]
        Training run details (permissive).
    dataset : dict[str, Any]
        Dataset provenance and statistics (permissive).
    environment : dict[str, Any]
        Software environment snapshot (permissive).
    provenance : dict[str, Any]
        Code and data provenance information (permissive).
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    created_at: datetime | None = Field(default=None)
    metadata_inference: InferenceMetadata

    # Permissive sections - validated only for presence, not structure.
    config: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dataset(self, dataset_name: str | None) -> DatasetInferenceConfig:
        """Return the :class:`DatasetInferenceConfig` for a named dataset.

        Parameters
        ----------
        dataset_name : str | None
            Dataset name to look up.  When ``None`` the first entry in
            ``metadata_inference.dataset_names`` is used.

        Returns
        -------
        DatasetInferenceConfig
            Per-dataset inference configuration.

        Raises
        ------
        KeyError
            If *dataset_name* is not present in ``metadata_inference.datasets``.
        IndexError
            If ``dataset_names`` is empty and *dataset_name* is ``None``.
        """
        if dataset_name is None:
            dataset_name = self.metadata_inference.dataset_names[0]
        return self.metadata_inference.datasets[dataset_name]

    # ------------------------------------------------------------------
    # MetadataContract contract implementation
    # ------------------------------------------------------------------

    def get_dataset_names(self) -> list[str]:
        """Return the ordered list of dataset names.

        Returns
        -------
        list[str]
            Dataset names as recorded in ``metadata_inference``.
        """
        return self.metadata_inference.dataset_names

    def get_task(self) -> str | None:
        """Return the task label, or ``None`` if not set.

        Returns
        -------
        str or None
            Task label (e.g. ``"forecaster"``).
        """
        return self.metadata_inference.task

    def get_timestep(self, dataset_name: str | None = None) -> str:
        """Return the timestep frequency string for a dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        str
            Frequency string (e.g. ``"6h"``).
        """
        return self._resolve_dataset(dataset_name).timesteps.timestep

    def get_input_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return input relative date indices for a dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[int]
            Relative date indices used as model inputs (e.g. ``[-1, 0]``).
        """
        return self._resolve_dataset(dataset_name).timesteps.input_relative_date_indices

    def get_output_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return output relative date indices for a dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[int]
            Relative date indices produced as model outputs (e.g. ``[1]``).
        """
        return self._resolve_dataset(dataset_name).timesteps.output_relative_date_indices

    def get_variable_indices(self, dataset_name: str | None = None) -> dict[str, int]:
        """Return input variable name to tensor index mapping.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the input tensor.
        """
        return self._resolve_dataset(dataset_name).data_indices.input

    def get_output_variable_indices(self, dataset_name: str | None = None) -> dict[str, int]:
        """Return output variable name to tensor index mapping.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the output tensor.
        """
        return self._resolve_dataset(dataset_name).data_indices.output

    def get_variable_types(self, dataset_name: str | None = None) -> dict[str, list[str]]:
        """Return variable categories by role.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping category names (``"forcing"``, ``"prognostic"``,
            ``"diagnostic"``, ``"target"``) to lists of variable names.
        """
        vt = self._resolve_dataset(dataset_name).variable_types
        return {
            "forcing": list(vt.forcing),
            "prognostic": list(vt.prognostic),
            "diagnostic": list(vt.diagnostic),
            "target": list(vt.target),
        }

    def get_tensor_shapes(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return tensor shape metadata as a plain dict.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            Shape metadata with keys ``"variables"``, ``"input_timesteps"``,
            ``"ensemble"``, and ``"grid"``.
        """
        shapes = self._resolve_dataset(dataset_name).shapes
        return {
            "variables": shapes.variables,
            "input_timesteps": shapes.input_timesteps,
            "ensemble": shapes.ensemble,
            "grid": shapes.grid,
        }

    def get_variables_metadata(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata from the permissive dataset section.

        Replicates the old inference logic:
        1. Look for ``variables_metadata`` at top level or nested under dataset name
        2. Apply ``constant_fields`` patch (marks listed variables as constant_in_time)

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query. Defaults to the first dataset.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of variable names to their metadata dicts, or an empty
            dict if the key is absent.
        """
        if dataset_name is None:
            dataset_name = self.get_dataset_names()[0]

        # Find the per-dataset section in the permissive dataset dict
        ds_section = self.dataset
        if dataset_name in ds_section and isinstance(ds_section.get(dataset_name), dict):
            ds_section = ds_section[dataset_name]

        result = dict(ds_section.get("variables_metadata", {}))

        # Apply constant_fields patch (old inference behaviour)
        for name in ds_section.get("constant_fields", []):
            if name in result:
                result[name] = {**result[name], "constant_in_time": True}

        return result

    def get_grid_points(self, dataset_name: str | None = None) -> int | None:
        """Return the number of grid points from the typed shapes block.

        In V1, read from ``shapes.grid`` of the resolved dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        int or None
            Number of grid points, or ``None`` if not recorded.
        """
        return self._resolve_dataset(dataset_name).shapes.grid

    def get_data_request(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return data request parameters from the permissive dataset section.

        In V1, read from ``dataset.data_request``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Unused in V1 (the permissive ``dataset`` section is not
            per-dataset).  Accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            Data request parameters, or an empty dict if the key is absent.
        """
        return self.dataset.get("data_request", {})

    def get_precision(self) -> str | None:
        """Return the model precision string from the permissive config section.

        In V1, read from ``config.training.precision``.

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

        In V1, this is the top-level ``provenance`` permissive dict.

        Returns
        -------
        dict[str, Any]
            Provenance information (git SHA, hostname, packages, etc.).
            Returns an empty dict if the section is absent.
        """
        return self.provenance

    def get_data_frequency(self, dataset_name: str | None = None) -> str | None:
        """Return the data frequency string from the permissive dataset section.

        In V1, read from ``dataset.frequency``, or if `config.task.output_timestep` is set, that value takes precedence.

        Parameters
        ----------
        dataset_name : str | None, optional
            Unused in V1 (the permissive ``dataset`` section is not
            per-dataset).  Accepted for interface compatibility.

        Returns
        -------
        str or None
            Frequency string (e.g. ``"6h"``), or ``None`` if not recorded.
        """
        output_timestep = (self.config.get("task") or {}).get("output_timestep", None)
        if output_timestep is not None:
            return output_timestep

        freq = self.dataset.get("frequency")
        if freq is not None:
            return freq
        # Fallback: config.data.frequency
        return (self.config.get("data") or {}).get("frequency")

    def get_sources(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        """Return source dataset configurations from the permissive dataset section.

        In V1, read from ``dataset.sources``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Unused in V1 (the permissive ``dataset`` section is not
            per-dataset).  Accepted for interface compatibility.

        Returns
        -------
        list[dict[str, Any]]
            Source dataset configurations, or an empty list if not recorded.
        """
        return self.dataset.get("sources", [])

    def get_open_dataset_args(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return arguments for opening the training dataset.

        In V1, read from ``dataset.arguments``.  The returned dict typically
        contains ``"args"`` and/or ``"kwargs"`` keys that can be passed
        directly to ``open_dataset()``.

        Parameters
        ----------
        dataset_name : str | None, optional
            Unused in V1 (the permissive ``dataset`` section is not
            per-dataset).  Accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            Dataset open arguments, or an empty dict if not recorded.
        """
        ds = self.dataset
        if isinstance(ds, dict) and "arguments" in ds:
            return ds["arguments"]
        return {}

    def get_dataloader_config(
        self,
        partition: str = "training",
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Return dataloader dataset configuration for a given partition.

        Read from ``config.dataloader.<partition>``.  For multi-dataset
        checkpoints, the per-dataset entry under ``datasets.<dataset_name>``
        is returned.  For newer checkpoints, the ``dataset_config`` key is
        unwrapped.

        Parameters
        ----------
        partition : str, optional
            The partition name, by default ``"training"``.
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            The dataloader dataset configuration, or an empty dict if absent.
        """
        if dataset_name is None:
            dataset_name = self.get_dataset_names()[0]

        dataloader = (self.config.get("dataloader") or {}).get(partition, {})
        if not isinstance(dataloader, dict):
            return {}

        # For multi-dataset checkpoints the dataloader has a per-dataset key.
        datasets_section = dataloader.get("datasets", {})
        if isinstance(datasets_section, dict) and dataset_name in datasets_section:
            dataloader = datasets_section[dataset_name]

        # For newer checkpoints the dataset config is under "dataset_config".
        config_val = dataloader.get("dataset_config")
        if config_val is not None:
            if isinstance(config_val, str):
                config_val = {"dataset": config_val}
            if isinstance(config_val, dict):
                # Copy extra dataloader keys that are also open_dataset kwargs.
                for k in ("start", "end"):
                    if k in dataloader:
                        config_val.setdefault(k, dataloader[k])
                return {k: v for k, v in config_val.items() if v is not None}

        return dataloader
