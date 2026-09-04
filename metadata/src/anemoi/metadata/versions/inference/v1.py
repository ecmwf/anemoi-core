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

The nested per-dataset models (:class:`DatasetInferenceConfig`,
:class:`DataIndices`, :class:`VariableTypes`, :class:`TimestepConfig`,
:class:`TensorShapes`) preserve unknown fields for forward compatibility. This
allows newer checkpoint writers to add fields without breaking older readers.
Strict validation will be enforced at write time from V2 onwards.

V1 only handles checkpoints that already have a ``metadata_inference`` block.
Legacy checkpoints (no ``metadata_inference``, no ``schema_version``) are
handled by :class:`~anemoi.metadata.versions.v0.MetadataV0`.  A migration
from V0 to V1 is registered in
:mod:`anemoi.metadata.migrations.v0_to_v1`.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class MetadataBase(BaseModel):
    """Base class for metadata versions."""

    model_config = ConfigDict(frozen=True)


class TimestepConfig(MetadataBase):
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

    timestep: str
    input_relative_date_indices: list[int]
    output_relative_date_indices: list[int]
    relative_date_indices_training: list[int]


class DataIndices(MetadataBase):
    """Mapping from variable names to tensor indices.

    Attributes
    ----------
    input : dict[str, int]
        Mapping of variable name to its index in the input tensor.
    output : dict[str, int]
        Mapping of variable name to its index in the output tensor.
    """

    input: dict[str, int]
    output: dict[str, int]


class VariableTypes(MetadataBase):
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

    forcing: list[str] = Field(default_factory=list)
    target: list[str] = Field(default_factory=list)
    prognostic: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)


class TensorShapes(MetadataBase):
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

    variables: int
    input_timesteps: int
    ensemble: int = 1
    grid: int | None = None


class DatasetInferenceConfig(MetadataBase):
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

    data_indices: DataIndices
    variable_types: VariableTypes
    timesteps: TimestepConfig
    shapes: TensorShapes


class InferenceMetadata(MetadataBase):
    """Top-level inference metadata written by training.

    This model is the authoritative source of truth consumed by inference at
    runtime.  It supports two input shapes:

    1. **Structured** - a dict that already contains a ``"datasets"`` key
       mapping dataset names to their per-dataset configs.
    2. **Flat** - a dict where scalar fields (``seed``, ``run_id``, ``task``,
       ``dataset_names``) sit alongside per-dataset sub-dicts keyed by the
       names listed in ``dataset_names``.  The ``@model_validator`` reshapes
       this into the structured form before validation.

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
