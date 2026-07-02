# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Base metadata class and version gating decorator."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import TypeVar

from packaging.version import Version
from pydantic import BaseModel
from pydantic import ConfigDict

from .exceptions import VersionError

F = TypeVar("F", bound=Callable[..., Any])


def requires_version(min_version: str) -> Callable[[F], F]:
    """Decorator to gate methods by minimum schema version.

    Use this decorator on methods that are only available from a certain
    schema version onwards. The method will raise VersionError if called
    on metadata with an older schema version.

    Parameters
    ----------
    min_version : str
        Minimum schema version required (e.g., "2.0").

    Returns
    -------
    Callable[[F], F]
        Decorator that wraps the function with version checking.

    Raises
    ------
    VersionError
        If the metadata version is less than min_version.

    Examples
    --------
    >>> @requires_version("2.0")
    ... def get_provenance(self) -> ProvenanceInfo:
    ...     return self._raw.provenance
    """
    min_ver = Version(min_version)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            # Support both MetadataContract (has .version) and Metadata interface
            # (has ._raw.version).
            if hasattr(self, "version"):
                current = self.version
            elif hasattr(self, "_raw"):
                current = self._raw.version
            else:
                raise VersionError(
                    f"{func.__name__} requires schema version >= {min_version}, "
                    f"but could not determine the current version"
                )
            if current is None or current < min_ver:
                raise VersionError(
                    f"{func.__name__} requires schema version >= {min_version}, "
                    f"but this metadata is version {current}"
                )
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class MetadataContract(BaseModel, ABC):
    """Abstract base class for all metadata versions.

    All versioned metadata schemas inherit from this class. Each version
    is a complete, standalone schema - no inheritance between versions.
    This allows V2 to completely redefine fields from V1 if needed.

    The model is configured as frozen (immutable) to ensure data integrity.
    Migrations between versions create new instances rather than modifying
    existing ones.

    Attributes
    ----------
    schema_version : str
        The semantic version string (e.g., "1.0").

    Examples
    --------
    >>> @MetadataRegistry.register("1.0")
    ... class MetadataV1(MetadataContract):
    ...     schema_version: str
    ...     model_name: str
    ...     ...etc...
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: str | None = None
    original_schema_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialisation.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataContract":
        """Deserialise from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation of metadata.

        Returns
        -------
        MetadataContract
            Validated metadata instance.
        """
        return cls.model_validate(data)

    @property
    def version(self) -> Version | None:
        """Parsed version for comparison.

        Returns
        -------
        Version or None
            Schema version as a :class:`packaging.version.Version` object
        """
        if self.schema_version is None:
            return None
        return Version(self.schema_version)

    @property
    def original_version(self) -> Version | None:
        """Return the original schema_version string.

        Returns
        -------
        Version or None
            Original schema_version as a :class:`packaging.version.Version` object
        """
        if self.original_schema_version is None:
            return self.version
        return Version(self.original_schema_version)

    def copy_with(self, **overrides: Any) -> "MetadataContract":
        """Create a modified copy of this instance (models are frozen).

        Dumps the current model to a dict, applies *overrides*, then
        re-validates through the same class.  This is the idiomatic way to
        produce a slightly-different instance of the **same** version without
        repeating every field.

        For **cross-version** migrations, construct the target class directly
        (e.g. ``MetadataV2.model_validate({...})``).

        Parameters
        ----------
        **overrides : Any
            Field values to replace.  Keys must be valid field names for this
            model class.

        Returns
        -------
        MetadataContract
            A new instance of the same class with the overridden fields.

        Raises
        ------
        pydantic.ValidationError
            If the overrides produce an invalid model.

        Examples
        --------
        >>> v1 = MetadataV1.model_validate(data)
        >>> v1_patched = v1.copy_with(schema_version="1.1")
        """
        data = self.model_dump()
        data.update(overrides)
        return type(self).model_validate(data)

    # ------------------------------------------------------------------
    # Contract methods — each version must implement these so the
    # interface layer never reaches into version-specific internals.
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dataset_names(self) -> list[str]:
        """Return the ordered list of dataset names.

        Returns
        -------
        list[str]
            Dataset names as recorded in the inference metadata.
        """
        ...

    @abstractmethod
    def get_task(self) -> str | None:
        """Return the task label, or ``None`` if not set.

        Returns
        -------
        str or None
            Task label (e.g. ``"forecaster"``).
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def get_variables_metadata(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata (MARS keys, flags, units, etc.).

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
        ...

    def get_accumulations(self, dataset_name: str | None = None) -> list[str]:
        """Return variable names that are accumulations.

        Accumulations require de-accumulation at inference time.  Derived from
        :meth:`get_variables_metadata` entries where ``process`` is
        ``"accumulated"`` or ``"accumulation"``.

        Versions may override this if they use a different convention.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[str]
            Variable names whose ``process`` is ``"accumulated"``.
            Returns an empty list if not available.
        """
        vm = self.get_variables_metadata(dataset_name)
        return [name for name, meta in vm.items() if meta.get("process") in ("accumulated", "accumulation")]

    def get_computed_forcings(self, dataset_name: str | None = None) -> list[str]:
        """Return variable names that are computed at inference time.

        Computed forcings are not retrieved from data but derived on the fly
        during inference (e.g. solar zenith angle).  Derived from
        :meth:`get_variables_metadata` entries where ``computed_forcing`` is
        ``True``.

        Versions may override this if they use a different convention.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[str]
            Variable names flagged as computed forcings.
            Returns an empty list if not available.
        """
        vm = self.get_variables_metadata(dataset_name)
        return [name for name, meta in vm.items() if meta.get("computed_forcing", False)]

    @abstractmethod
    def get_grid_points(self, dataset_name: str | None = None) -> int | None:
        """Return the number of grid points, or ``None`` if unknown.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        int or None
            Number of grid points, or ``None`` when not recorded.
        """
        ...

    @abstractmethod
    def get_data_request(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return data request parameters (grid, area, etc.).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            Data request parameters.  Returns an empty dict if not available.
        """
        ...

    @abstractmethod
    def get_precision(self) -> str | None:
        """Return the model precision string, or ``None`` if not recorded.

        Returns
        -------
        str or None
            Precision string (e.g. ``"16-mixed"``, ``"bf16-mixed"``, ``"32"``),
            or ``None`` if not recorded.
        """
        ...

    @abstractmethod
    def get_provenance(self) -> dict[str, Any]:
        """Return provenance information (git, packages, etc.).

        Returns
        -------
        dict[str, Any]
            Provenance information.  Returns an empty dict if not available.
        """
        ...

    @abstractmethod
    def get_data_frequency(self, dataset_name: str | None = None) -> str | None:
        """Return the data frequency string, or ``None`` if not recorded.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        str or None
            Frequency string (e.g. ``"6h"``), or ``None`` if not recorded.
        """
        ...

    @abstractmethod
    def get_sources(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        """Return source dataset configurations.

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
        ...

    @abstractmethod
    def get_open_dataset_args(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return arguments for opening the training dataset.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Any]
            Arguments for opening the training dataset (e.g. ``"args"`` and
            ``"kwargs"`` keys).  Returns an empty dict if not available.
        """
        ...

    @abstractmethod
    def get_dataloader_config(
        self,
        partition: str = "training",
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Return dataloader dataset configuration for a given partition.

        Extracts the dataset configuration from the training config's
        dataloader section for the requested partition (e.g. ``"training"``,
        ``"validation"``), handling multi-dataset nesting and the
        ``dataset_config`` key used by newer checkpoints.

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
        ...

    _MISSING = object()

    def get_section(self, name: str) -> dict[str, Any] | None:
        """Return a permissive top-level section by name, or ``None``.

        Uses a sentinel to distinguish between an attribute that genuinely
        holds ``None`` and an attribute that does not exist at all.

        Parameters
        ----------
        name : str
            Attribute name on the concrete metadata model (e.g. ``"config"``,
            ``"training"``).

        Returns
        -------
        dict[str, Any] or None
            The section value if it exists (may itself be ``None``), or
            ``None`` if the attribute does not exist.
        """
        value = getattr(self, name, self._MISSING)
        if value is self._MISSING:
            return None
        return value  # type: ignore[return-value]
