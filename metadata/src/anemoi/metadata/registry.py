# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Version registry for metadata schemas."""

import warnings
from typing import TYPE_CHECKING

from packaging.version import Version

from .exceptions import UnknownVersionError

if TYPE_CHECKING:
    from .base import MetadataContract


class MetadataRegistry:
    """Registry for metadata versions.

    Manages registration and retrieval of versioned metadata schemas.
    Versions are sorted using proper semantic versioning comparison.

    This class uses class methods exclusively - it is not meant to be
    instantiated. All state is stored in class variables.

    Examples
    --------
    >>> @MetadataRegistry.register("1.0")
    ... class MetadataV1(MetadataContract):
    ...     schema_version: str
    ...     model_name: str

    >>> MetadataRegistry.list_versions()
    ['1.0']

    >>> cls = MetadataRegistry.get_version("1.0")
    >>> cls.__name__
    'MetadataV1'
    """

    _versions: dict[str, type["MetadataContract"]] = {}
    _sorted_versions: list[str] | None = None

    def __init__(self) -> None:
        raise RuntimeError("MetadataRegistry is a static class and cannot be instantiated")

    @classmethod
    def register(cls, version: str):
        """Decorator to register a metadata version.

        Parameters
        ----------
        version : str
            Semantic version string (e.g., "1.0").

        Returns
        -------
        Callable
            Decorator that registers the metadata class.

        Raises
        ------
        ValueError
            If the version is already registered.

        Examples
        --------
        >>> @MetadataRegistry.register("1.0")
        ... class MetadataV1(MetadataContract):
        ...     schema_version: str
        """

        def decorator(
            metadata_cls: type["MetadataContract"],
        ) -> type["MetadataContract"]:
            if version in cls._versions:
                raise ValueError(f"Version {version} already registered")
            cls._versions[version] = metadata_cls
            cls._sorted_versions = None
            return metadata_cls

        return decorator

    @classmethod
    def register_minor(cls, version: str) -> None:
        """Register a minor version that shares its schema class with a base version.

        Use this to add additive minor versions (e.g. "1.1") that reuse the
        same schema class as a previously registered version (e.g. "1.0").
        The base version is derived automatically by replacing the last version
        component with "0" (e.g. "1.1" -> "1.0").
        The schema class must use ``extra="allow"`` so that new optional fields
        are accepted without breaking validation.

        This is the recommended pattern for backwards-compatible additions that
        don't warrant a new schema class.  Migration from the base version is
        handled automatically by the migrator (a no-op version bump), so no
        migration registration is needed for schema-sharing minors unless you
        need to populate or transform fields.

        Parameters
        ----------
        version : str
            New minor version string to register (e.g. "1.1").  The base
            version is derived by replacing the final component with "0".
            Must match the pattern ``^\\d+\\.\\d+$`` (major.minor, digits only).

        Raises
        ------
        ValueError
            If *version* is already registered, the derived base version
            is not registered, or the version format is invalid.

        Examples
        --------
        >>> MetadataRegistry.register_minor("1.1")
        """
        import re

        if not re.match(r"^\d+\.\d+$", version):
            raise ValueError(
                f"Version '{version}' does not match the required format 'major.minor' "
                f"(e.g. '1.1', '2.3'). Version must be two numeric components separated by a dot."
            )

        if version in cls._versions:
            raise ValueError(f"Version {version} already registered")

        base_version = ".".join(version.split(".")[:-1]) + ".0"

        if base_version not in cls._versions:
            raise ValueError(
                f"Base version {base_version} not registered; "
                f"register it first with @MetadataRegistry.register('{base_version}')"
            )
        cls._versions[version] = cls._versions[base_version]
        cls._sorted_versions = None

    @classmethod
    def get_version(cls, version: str) -> type["MetadataContract"]:
        """Get metadata class for specific version.

        Parameters
        ----------
        version : str
            Version string to look up.

        Returns
        -------
        type[MetadataContract]
            The metadata class for the specified version.

        Raises
        ------
        UnknownVersionError
            If version is not registered.
        """
        if version not in cls._versions:
            raise UnknownVersionError(f"Unknown metadata version: {version}")
        return cls._versions[version]

    @classmethod
    def get_latest(cls) -> type["MetadataContract"]:
        """Get the latest registered metadata version.

        Returns
        -------
        type[MetadataContract]
            The metadata class for the latest version.

        Raises
        ------
        RuntimeError
            If no versions are registered.
        """
        if not cls._versions:
            raise RuntimeError("No metadata versions registered")
        return cls._versions[cls.latest_version()]

    @classmethod
    def latest_version(cls) -> str:
        """Get latest version string.

        Returns
        -------
        str
            The latest registered version string.
        """
        return cls._list_versions()[-1]

    @classmethod
    def _list_versions(cls) -> list[str]:
        """List all versions sorted by semver (internal).

        Returns
        -------
        list[str]
            List of version strings sorted in ascending order.
        """
        if cls._sorted_versions is None:
            cls._sorted_versions = sorted(cls._versions.keys(), key=Version)
        return cls._sorted_versions

    @classmethod
    def list_versions(cls) -> list[str]:
        """List all available versions.

        Returns
        -------
        list[str]
            List of version strings sorted in ascending order.
        """
        return cls._list_versions().copy()

    @classmethod
    def detect_version(cls, data: dict) -> str:
        """Detect version from metadata dict.

        Checkpoints that pre-date the ``schema_version`` field are classified
        by their content:

        * If ``metadata_inference`` is present but ``schema_version`` is absent,
          the checkpoint is a transitional V1 (written after the
          ``metadata_inference`` block was introduced but before
          ``schema_version`` was added).  These are routed to ``"1.0"``.
        * If neither ``schema_version`` nor ``metadata_inference`` is present,
          the checkpoint is a true legacy V0 (pre-``metadata_inference``).
          These are routed to ``"0.0"``.

        Parameters
        ----------
        data : dict
            Raw metadata dictionary.

        Returns
        -------
        str
            The schema_version from the data, normalised to a registered
            version string.
        """
        if "schema_version" not in data:
            warnings.warn(
                "Metadata missing 'schema_version' field. "
                "This may indicate an older version of metadata. "
                "Please ensure that the metadata is up-to-date.",
                UserWarning,
            )
            # Distinguish between transitional V1 (has metadata_inference) and
            # true legacy V0 (no metadata_inference, no schema_version).
            if "metadata_inference" in data:
                return "1.0"
            return "0.0"
        version = data["schema_version"]
        return version

    @classmethod
    def load(cls, data: dict, *, migrate: bool = True, allow_stop: bool = False) -> "MetadataContract":
        """Load metadata with automatic version detection and optional migration.

        This is the primary entry point for loading raw metadata dictionaries.
        It detects the schema version, validates the data, and optionally
        migrates to the latest version.

        Parameters
        ----------
        data : dict
            Raw metadata dict.
        migrate : bool, optional
            If True (default), auto-migrate to latest version.
        allow_stop : bool, optional
            If True, allow migration to stop at the latest version
            that is compatible with the current version. If False
            (default), migration will always attempt to reach the
            absolute latest version.

        Returns
        -------
        MetadataContract
            Loaded (and optionally migrated) metadata instance.
        """
        version = cls.detect_version(data)
        if "schema_version" not in data:
            data = {**data, "schema_version": version}
        metadata = cls.get_version(version).from_dict(data)

        if migrate:
            from .migration import MetadataMigrator

            return MetadataMigrator.migrate(metadata, cls.latest_version(), allow_stop=allow_stop)
        return metadata
