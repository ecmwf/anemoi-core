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

from pydantic import ConfigDict
from pydantic import Field

from ..base import MetadataContract
from ..registry import MetadataRegistry
from .inference import InferenceMetadata
from .inference.contract_fullfillment import InferenceMetadataFullfillment


@MetadataRegistry.register("1.0")
class MetadataV1(InferenceMetadataFullfillment, MetadataContract):
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
