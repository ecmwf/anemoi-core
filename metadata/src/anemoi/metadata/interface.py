# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""User-facing metadata interface.

This module provides the main ``Metadata`` class that users interact with.
It wraps the raw pydantic schemas and exposes typed properties via the
version-agnostic contract methods defined on :class:`~anemoi.metadata.base.MetadataContract`.
Permissive dict sections are accessible via :meth:`get` or :meth:`__getitem__`.

The interface never reaches into version-specific internal structure directly;
all inference data is accessed through the contract methods that each schema
version implements.
"""

from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from packaging.version import Version

from .mixins import ValidationMixin
from .mixins import VariablesMixin

if TYPE_CHECKING:
    from .base import MetadataContract


class DatasetView:
    """Read-only per-dataset view over a :class:`Metadata` instance.

    Provides convenient access to per-dataset properties and methods
    without needing to pass the dataset name to each call. All properties
    delegate to the underlying contract methods with the bound dataset name.

    The underlying metadata model is frozen (immutable), so all properties
    that do not require parameters are cached using :func:`functools.cached_property`.

    Parameters
    ----------
    raw : MetadataContract
        The underlying pydantic metadata instance.
    name : str
        The dataset name this view is bound to.

    Examples
    --------
    >>> metadata = Metadata.from_checkpoint("model.ckpt")
    >>> view = metadata.dataset("era5_1deg")
    >>> print(view.timestep)
    '6h'
    >>> print(view.multi_step_input)
    2
    >>> print(view.variable_indices)
    {'2t': 0, 'msl': 1, ...}
    """

    def __init__(self, raw: "MetadataContract", name: str) -> None:
        """Initialise with a raw metadata instance and dataset name.

        Parameters
        ----------
        raw : MetadataContract
            The underlying pydantic metadata instance.
        name : str
            The dataset name this view is bound to.
        """
        self._raw = raw
        self._name = name

    @property
    def name(self) -> str:
        """The dataset name this view is bound to.

        Returns
        -------
        str
            Dataset name.
        """
        return self._name

    @cached_property
    def timestep(self) -> str:
        """Model timestep frequency string for this dataset.

        Returns
        -------
        str
            Frequency string (e.g. ``"6h"``).
        """
        return self._raw.get_timestep(self._name)

    @cached_property
    def input_relative_date_indices(self) -> list[int]:
        """Input relative date indices for this dataset.

        Returns
        -------
        list[int]
            Relative date indices used as model inputs (e.g. ``[-1, 0]``).
        """
        return self._raw.get_input_relative_date_indices(self._name)

    @cached_property
    def output_relative_date_indices(self) -> list[int]:
        """Output relative date indices for this dataset.

        Returns
        -------
        list[int]
            Relative date indices produced as model outputs (e.g. ``[1]``).
        """
        return self._raw.get_output_relative_date_indices(self._name)

    @cached_property
    def multi_step_input(self) -> int:
        """Number of input time-steps the model consumes for this dataset.

        Derived from the length of the input relative date indices.

        Returns
        -------
        int
            Count of input time-steps.
        """
        return len(self.input_relative_date_indices)

    @cached_property
    def multi_step_output(self) -> int:
        """Number of output time-steps the model produces for this dataset.

        Derived from the length of the output relative date indices.

        Returns
        -------
        int
            Count of output time-steps.
        """
        return len(self.output_relative_date_indices)

    @cached_property
    def variable_indices(self) -> dict[str, int]:
        """Input variable name to tensor index mapping for this dataset.

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the input tensor.
        """
        return self._raw.get_variable_indices(self._name)

    @cached_property
    def output_variable_indices(self) -> dict[str, int]:
        """Output variable name to tensor index mapping for this dataset.

        Returns
        -------
        dict[str, int]
            Mapping of variable name to its index in the output tensor.
        """
        return self._raw.get_output_variable_indices(self._name)

    @cached_property
    def variable_types(self) -> dict[str, list[str]]:
        """Variable categories by role for this dataset.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping category names (``"forcing"``, ``"prognostic"``,
            ``"diagnostic"``, ``"target"``) to lists of variable names.
        """
        return self._raw.get_variable_types(self._name)

    @cached_property
    def tensor_shapes(self) -> dict[str, Any]:
        """Tensor shape metadata for this dataset.

        Returns
        -------
        dict[str, Any]
            Shape metadata with keys ``"variables"``, ``"input_timesteps"``,
            ``"ensemble"``, and ``"grid"``.
        """
        return self._raw.get_tensor_shapes(self._name)

    @cached_property
    def grid_points(self) -> int | None:
        """Number of grid points for this dataset, or ``None`` if unknown.

        Returns
        -------
        int or None
            Number of grid points, or ``None`` if not recorded.
        """
        return self._raw.get_grid_points(self._name)

    @cached_property
    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Per-variable metadata (MARS keys, flags, units) for this dataset.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of variable names to their metadata dicts.
            Returns an empty dict if not available.
        """
        return self._raw.get_variables_metadata(self._name)

    @cached_property
    def accumulations(self) -> list[str]:
        """Variables that are accumulations (need de-accumulation) for this dataset.

        Returns
        -------
        list[str]
            Variable names that require de-accumulation at inference time.
        """
        return self._raw.get_accumulations(self._name)

    @cached_property
    def computed_forcings(self) -> list[str]:
        """Variables computed at inference time (not retrieved from data) for this dataset.

        Returns
        -------
        list[str]
            Variable names that are computed on the fly during inference.
        """
        return self._raw.get_computed_forcings(self._name)

    @cached_property
    def data_request(self) -> dict[str, Any]:
        """Data request parameters (grid, area, etc.) for this dataset.

        Returns
        -------
        dict[str, Any]
            Data request parameters.  Returns an empty dict if not available.
        """
        return self._raw.get_data_request(self._name)

    @cached_property
    def data_frequency(self) -> str | None:
        """Data frequency string for this dataset, or ``None`` if not recorded.

        Returns
        -------
        str or None
            Frequency string (e.g. ``"6h"``), or ``None`` if not recorded.
        """
        return self._raw.get_data_frequency(self._name)

    @cached_property
    def sources(self) -> list[dict[str, Any]]:
        """Source dataset configurations for this dataset.

        Returns
        -------
        list[dict[str, Any]]
            Source dataset configurations.  Returns an empty list if not
            available.
        """
        return self._raw.get_sources(self._name)

    @cached_property
    def open_dataset_args(self) -> dict[str, Any]:
        """Arguments for opening the training dataset for this dataset.

        Returns
        -------
        dict[str, Any]
            Arguments for opening the training dataset.  Returns an empty dict
            if not available.
        """
        return self._raw.get_open_dataset_args(self._name)

    def dataloader_config(self, partition: str = "training") -> dict[str, Any]:
        """Dataloader dataset configuration for a given partition for this dataset.

        Extracts the dataset configuration from
        ``config.dataloader.<partition>``, handling multi-dataset nesting
        and the ``dataset_config`` key used by newer checkpoints.

        Parameters
        ----------
        partition : str, optional
            The partition name (e.g. ``"training"``, ``"validation"``),
            by default ``"training"``.

        Returns
        -------
        dict[str, Any]
            The dataloader dataset configuration.  Returns an empty dict
            if the partition or config section is absent.
        """
        return self._raw.get_dataloader_config(partition, self._name)

    def __repr__(self) -> str:
        """String representation.

        Returns
        -------
        str
            Repr string showing the dataset name.
        """
        return f"DatasetView(name={self._name!r})"


class Metadata(VariablesMixin, ValidationMixin):
    """User-facing metadata interface.

    Primary API exposes typed properties via the version-agnostic contract
    methods on the underlying :class:`~anemoi.metadata.base.MetadataContract`
    instance.  Permissive sections (``config``, ``training``, ``dataset``,
    ``environment``, ``provenance``) are accessible via :meth:`get` or
    :meth:`__getitem__`.

    This class is the primary interface for users working with checkpoint
    metadata.  It replaces ``anemoi.inference.metadata.Metadata``.

    Parameters
    ----------
    raw : MetadataContract
        The underlying pydantic metadata instance.

    Examples
    --------
    >>> metadata = Metadata.from_checkpoint("model.ckpt")
    >>> print(metadata.schema_version)
    '1.0'
    >>> print(metadata.dataset_names)
    ['data']
    >>> print(metadata.timestep)
    '6h'
    """

    def __init__(self, raw: "MetadataContract") -> None:
        """Initialise with a raw metadata instance.

        Parameters
        ----------
        raw : MetadataContract
            The underlying pydantic metadata instance.
        """
        self._raw = raw

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        migrate: bool = True,
    ) -> "Metadata":
        """Load metadata from a checkpoint file.

        Parameters
        ----------
        path : str | Path
            Path to the checkpoint file.
        migrate : bool, optional
            If ``True`` (default), auto-migrate to the latest schema version.

        Returns
        -------
        Metadata
            Loaded metadata instance.

        Raises
        ------
        CheckpointError
            If the file is invalid or metadata is missing.

        Examples
        --------
        >>> m = Metadata.from_checkpoint("model.ckpt")
        >>> m = Metadata.from_checkpoint("old_model.ckpt", migrate=False)
        """
        from .checkpoint import load_metadata

        raw = load_metadata(path, migrate=migrate)
        return cls(raw)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        migrate: bool = True,
    ) -> "Metadata":
        """Load metadata from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Raw metadata dictionary.
        migrate : bool, optional
            If ``True`` (default), auto-migrate to the latest schema version.

        Returns
        -------
        Metadata
            Loaded metadata instance.

        Examples
        --------
        >>> data = {"schema_version": "1.0", ...}
        >>> m = Metadata.from_dict(data)
        """
        from .registry import MetadataRegistry

        raw = MetadataRegistry.load(data, migrate=migrate)
        return cls(raw)

    # ------------------------------------------------------------------
    # Envelope properties
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> str | None:
        """The schema version string, or ``None`` if not set.

        Returns
        -------
        str or None
            Semantic version string (e.g., ``"1.0"``), or ``None`` if the
            schema version is not set.
        """
        return self._raw.schema_version

    @property
    def original_schema_version(self) -> str | None:
        """The original schema version string, or ``None`` if not set.

        Returns
        -------
        str or None
            Semantic version string (e.g., ``"1.0"``), or ``None`` if the
            original schema version is not set.
        """
        return self._raw.original_schema_version

    @property
    def is_legacy(self) -> bool:
        """Whether this metadata is a V0 (pre-``metadata_inference``) checkpoint.

        Returns ``True`` when the checkpoint is a legacy V0 instance (version
        ``"0.0"``) that has no ``metadata_inference`` block. Returns ``False``
        for native V1+ checkpoints and for hand-constructed instances with no
        version information.

        Note that the registry's ``detect_version`` assigns ``"0.0"`` to true
        legacy data on load, so ``None`` only occurs for hand-constructed
        instances without version metadata.

        Returns
        -------
        bool
            ``True`` if this is a legacy V0 checkpoint, ``False`` otherwise.
        """
        ver = self._raw.original_version
        return ver is not None and ver < Version("1.0")

    @cached_property
    def created_at(self) -> datetime | None:
        """When the metadata was created.

        Returns
        -------
        datetime | None
            Creation timestamp, or ``None`` if not recorded.
        """
        return getattr(self._raw, "created_at", None)

    @property
    def raw(self) -> "MetadataContract":
        """Access the underlying pydantic model.

        Users who need version-specific internals (e.g. the typed
        ``InferenceMetadata`` block from V1) can reach them via this
        property: ``metadata.raw.metadata_inference``.

        Returns
        -------
        MetadataContract
            The raw pydantic metadata instance.
        """
        return self._raw

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the metadata, suitable for JSON
            serialisation.
        """
        return self._raw.to_dict()

    # ------------------------------------------------------------------
    # Typed properties via version-agnostic contract methods
    # ------------------------------------------------------------------

    @cached_property
    def dataset_names(self) -> list[str]:
        """Ordered list of dataset names referenced by this checkpoint.

        Returns
        -------
        list[str]
            Dataset names as recorded in the inference metadata.
        """
        return self._raw.get_dataset_names()

    def dataset(self, name: str | None = None) -> DatasetView:
        """Return a per-dataset view for convenient access to dataset-specific properties.

        When *name* is ``None`` the view is bound to the first dataset.
        Multi-dataset consumers can use this method to access per-dataset
        properties without needing to pass the dataset name to each call.

        Parameters
        ----------
        name : str | None, optional
            Dataset name to query.  If ``None`` (default), the first entry
            in :attr:`dataset_names` is used.

        Returns
        -------
        DatasetView
            Read-only view over the metadata for the specified dataset.

        Raises
        ------
        IndexError
            If *name* is ``None`` and :attr:`dataset_names` is empty.
        KeyError
            If *name* is not in :attr:`dataset_names`.

        Examples
        --------
        >>> metadata = Metadata.from_checkpoint("model.ckpt")
        >>> view = metadata.dataset("era5_1deg")
        >>> print(view.timestep)
        '6h'
        >>> print(view.multi_step_input)
        2

        >>> # Default to first dataset
        >>> first = metadata.dataset()
        >>> print(first.name)
        'era5_1deg'
        """
        if name is None:
            if not self.dataset_names:
                msg = "Cannot get default dataset: dataset_names is empty"
                raise IndexError(msg)
            name = self.dataset_names[0]
        elif name not in self.dataset_names:
            msg = f"Dataset {name!r} not found. Available datasets: {self.dataset_names!r}"
            raise KeyError(msg)
        return DatasetView(self._raw, name)

    @cached_property
    def datasets(self) -> dict[str, DatasetView]:
        """Mapping of all dataset names to their views.

        Iteration order follows :attr:`dataset_names`.

        Returns
        -------
        dict[str, DatasetView]
            Dictionary mapping dataset names to their read-only views.

        Examples
        --------
        >>> metadata = Metadata.from_checkpoint("multi_dataset.ckpt")
        >>> for name, view in metadata.datasets.items():
        ...     print(f"{name}: {view.timestep}")
        era5_1deg: 6h
        cerra_025deg: 1h
        """
        return {name: DatasetView(self._raw, name) for name in self.dataset_names}

    @cached_property
    def task(self) -> str | None:
        """Optional task label for this checkpoint.

        Returns
        -------
        str or None
            Task label (e.g. ``"forecaster"``), or ``None`` if not set.
        """
        return self._raw.get_task()

    @cached_property
    def timestep(self) -> str:
        """Model timestep frequency string.

        Returns
        -------
        str
            Frequency string (e.g. ``"6h"``).
        """
        return self._raw.get_timestep()

    @cached_property
    def multi_step_input(self) -> int:
        """Number of input time-steps the model consumes.

        Derived from the length of the input relative date indices for the
        first dataset.

        Returns
        -------
        int
            Count of input time-steps.
        """
        return len(self._raw.get_input_relative_date_indices())

    @cached_property
    def multi_step_output(self) -> int:
        """Number of output time-steps the model produces.

        Derived from the length of the output relative date indices for the
        first dataset.

        Returns
        -------
        int
            Count of output time-steps.
        """
        return len(self._raw.get_output_relative_date_indices())

    @cached_property
    def grid_points(self) -> int | None:
        """Number of grid points, or ``None`` if unknown.

        Returns
        -------
        int or None
            Number of grid points for the first dataset, or ``None`` if not
            recorded.
        """
        return self._raw.get_grid_points()

    @cached_property
    def precision(self) -> str | None:
        """Model precision string, or ``None`` if not recorded.

        Returns
        -------
        str or None
            Precision string (e.g. ``"16-mixed"``, ``"bf16-mixed"``,
            ``"32"``), or ``None`` if not recorded.
        """
        return self._raw.get_precision()

    def variables_metadata(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
        """Per-variable metadata (MARS keys, flags, units).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of variable names to their metadata dicts.
            Returns an empty dict if not available.
        """
        return self._raw.get_variables_metadata(dataset_name)

    def accumulations(self, dataset_name: str | None = None) -> list[str]:
        """Variables that are accumulations (need de-accumulation).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[str]
            Variable names that require de-accumulation at inference time.
        """
        return self._raw.get_accumulations(dataset_name)

    def computed_forcings(self, dataset_name: str | None = None) -> list[str]:
        """Variables computed at inference time (not retrieved from data).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[str]
            Variable names that are computed on the fly during inference.
        """
        return self._raw.get_computed_forcings(dataset_name)

    def data_request(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Data request parameters (grid, area, etc.).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            Data request parameters.  Returns an empty dict if not available.
        """
        return self._raw.get_data_request(dataset_name)

    def provenance(self) -> dict[str, Any]:
        """Code and data provenance information.

        Returns
        -------
        dict[str, Any]
            Provenance information (git SHA, hostname, packages, etc.).
            Returns an empty dict if not available.
        """
        return self._raw.get_provenance()

    def data_frequency(self, dataset_name: str | None = None) -> str | None:
        """Data frequency string, or ``None`` if not recorded.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        str or None
            Frequency string (e.g. ``"6h"``), or ``None`` if not recorded.
        """
        return self._raw.get_data_frequency(dataset_name)

    def sources(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        """Source dataset configurations.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[dict[str, Any]]
            Source dataset configurations.  Returns an empty list if not
            available.
        """
        return self._raw.get_sources(dataset_name)

    def open_dataset_args(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Arguments for opening the training dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            Arguments for opening the training dataset.  Returns an empty dict
            if not available.
        """
        return self._raw.get_open_dataset_args(dataset_name)

    def dataloader_config(
        self,
        partition: str = "training",
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Dataloader dataset configuration for a given partition.

        Extracts the dataset configuration from
        ``config.dataloader.<partition>``, handling multi-dataset nesting
        and the ``dataset_config`` key used by newer checkpoints.

        Parameters
        ----------
        partition : str, optional
            The partition name (e.g. ``"training"``, ``"validation"``),
            by default ``"training"``.
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            The dataloader dataset configuration.  Returns an empty dict
            if the partition or config section is absent.
        """
        return self._raw.get_dataloader_config(partition, dataset_name)

    # ------------------------------------------------------------------
    # Safe access for permissive sections
    # ------------------------------------------------------------------

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Access metadata sections and keys safely.

        Retrieves a top-level section from the raw metadata (e.g. ``"config"``,
        ``"training"``, ``"dataset"``, ``"environment"``, ``"provenance"``).
        When *key* is provided the value is looked up within that section.

        Parameters
        ----------
        section : str
            Top-level attribute name on the raw metadata model.
        key : str | None, optional
            Key to look up within the section.  When ``None`` the entire
            section is returned.
        default : Any, optional
            Value to return when the section or key is absent.  Defaults to
            ``None``.

        Returns
        -------
        Any
            The section, the value at *key* within the section, or *default*.

        Examples
        --------
        >>> metadata.get("config")
        {'data': {...}, 'training': {...}}
        >>> metadata.get("config", "data")
        {...}
        >>> metadata.get("missing_section", default=42)
        42
        """
        section_data = self._raw.get_section(section)
        if section_data is None:
            return default
        if key is None:
            return section_data
        if isinstance(section_data, dict):
            return section_data.get(key, default)
        return getattr(section_data, key, default)

    def __getitem__(self, section: str) -> Any:
        """Return a top-level metadata section by name.

        Parameters
        ----------
        section : str
            Top-level attribute name on the raw metadata model (e.g.
            ``"config"``, ``"training"``).

        Returns
        -------
        Any
            The section value.

        Raises
        ------
        KeyError
            If *section* does not exist on the raw metadata model.

        Examples
        --------
        >>> metadata["config"]
        {'data': {...}}
        """
        try:
            return getattr(self._raw, section)
        except AttributeError:
            raise KeyError(section) from None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation.

        Returns
        -------
        str
            Repr string showing schema version and dataset names.
        """
        return f"Metadata(version={self.schema_version!r}, " f"datasets={self.dataset_names!r})"
