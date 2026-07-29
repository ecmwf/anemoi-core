# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Inference metadata schema version 1.0, to assist in the fullfillment of the metadata contract."""

from typing import Any

from pydantic import ConfigDict

from . import DatasetInferenceConfig
from . import InferenceMetadata


class InferenceMetadataFullfillment:
    """Uses the InferenceMetadata schema to fulfill the MetadataContract interface."""

    model_config = ConfigDict(extra="allow", frozen=True)
    metadata_inference: InferenceMetadata

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

    # def get_variables_metadata(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
    #     """Return per-variable metadata from the permissive dataset section.

    #     Replicates the old inference logic:
    #     1. Look for ``variables_metadata`` at top level or nested under dataset name
    #     2. Apply ``constant_fields`` patch (marks listed variables as constant_in_time)

    #     Parameters
    #     ----------
    #     dataset_name : str | None, optional
    #         Dataset to query. Defaults to the first dataset.

    #     Returns
    #     -------
    #     dict[str, dict[str, Any]]
    #         Mapping of variable names to their metadata dicts, or an empty
    #         dict if the key is absent.
    #     """
    #     raise NotImplementedError("get_variables_metadata is not implemented in inference_metadata")

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

    # def get_data_request(self, dataset_name: str | None = None) -> dict[str, Any]:
    #     """Return data request parameters from the permissive dataset section.

    #     In V1, read from ``dataset.data_request``.

    #     Parameters
    #     ----------
    #     dataset_name : str | None, optional
    #         Unused in V1 (the permissive ``dataset`` section is not
    #         per-dataset).  Accepted for interface compatibility.

    #     Returns
    #     -------
    #     dict[str, Any]
    #         Data request parameters, or an empty dict if the key is absent.
    #     """
    #     raise NotImplementedError("get_data_request is not implemented in inference_metadata")

    # def get_precision(self) -> str | None:
    #     """Return the model precision string from the permissive config section.

    #     In V1, read from ``config.training.precision``.

    #     Returns
    #     -------
    #     str or None
    #         Precision string (e.g. ``"16-mixed"``), or ``None`` if absent.
    #     """
    #     raise NotImplementedError("get_precision is not implemented in inference_metadata")

    # def get_sources(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
    #     """Return source dataset configurations.

    #     Parameters
    #     ----------
    #     dataset_name : str | None, optional

    #     Returns
    #     -------
    #     list[dict[str, Any]]
    #         Source dataset configurations, or an empty list if not recorded.
    #     """
    #     raise NotImplementedError("get_sources is not implemented in inference_metadata")

    # def get_open_dataset_args(self, dataset_name: str | None = None) -> dict[str, Any]:
    #     """Return arguments for opening the training dataset.

    #     In V1, read from ``dataset.arguments``.  The returned dict typically
    #     contains ``"args"`` and/or ``"kwargs"`` keys that can be passed
    #     directly to ``open_dataset()``.

    #     Parameters
    #     ----------
    #     dataset_name : str | None, optional
    #         Unused in V1 (the permissive ``dataset`` section is not
    #         per-dataset).  Accepted for interface compatibility.

    #     Returns
    #     -------
    #     dict[str, Any]
    #         Dataset open arguments, or an empty dict if not recorded.
    #     """
    #     ds = self.dataset
    #     if isinstance(ds, dict) and "arguments" in ds:
    #         return ds["arguments"]
    #     return {}

    # def get_dataloader_config(
    #     self,
    #     partition: str = "training",
    #     dataset_name: str | None = None,
    # ) -> dict[str, Any]:
    #     """Return dataloader dataset configuration for a given partition.

    #     Read from ``config.dataloader.<partition>``.  For multi-dataset
    #     checkpoints, the per-dataset entry under ``datasets.<dataset_name>``
    #     is returned.  For newer checkpoints, the ``dataset_config`` key is
    #     unwrapped.

    #     Parameters
    #     ----------
    #     partition : str, optional
    #         The partition name, by default ``"training"``.
    #     dataset_name : str | None, optional
    #         Dataset to query.  Defaults to the first dataset.

    #     Returns
    #     -------
    #     dict[str, Any]
    #         The dataloader dataset configuration, or an empty dict if absent.
    #     """
    #     if dataset_name is None:
    #         dataset_name = self.get_dataset_names()[0]

    #     dataloader = (self.config.get("dataloader") or {}).get(partition, {})
    #     if not isinstance(dataloader, dict):
    #         return {}

    #     # For multi-dataset checkpoints the dataloader has a per-dataset key.
    #     datasets_section = dataloader.get("datasets", {})
    #     if isinstance(datasets_section, dict) and dataset_name in datasets_section:
    #         dataloader = datasets_section[dataset_name]

    #     # For newer checkpoints the dataset config is under "dataset_config".
    #     config_val = dataloader.get("dataset_config")
    #     if config_val is not None:
    #         if isinstance(config_val, str):
    #             config_val = {"dataset": config_val}
    #         if isinstance(config_val, dict):
    #             # Copy extra dataloader keys that are also open_dataset kwargs.
    #             for k in ("start", "end"):
    #                 if k in dataloader:
    #                     config_val.setdefault(k, dataloader[k])
    #             return {k: v for k, v in config_val.items() if v is not None}

    #     return dataloader
