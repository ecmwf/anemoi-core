# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sequential migration system for metadata versions.

Migrations are defined between adjacent versions only (v1→v2, v2→v3).
Multi-version jumps are handled by chaining migrations automatically.

Version Policy
--------------
- Versions use major.minor only (no patch): "1.0", "2.0", "2.1".
- Migrations are defined between any adjacent registered versions.
- Downgrades are not supported - use ``migrate=False`` to preserve original.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from .exceptions import MigrationError

if TYPE_CHECKING:
    from .base import MetadataContract


class MetadataMigrator:
    """Migrate metadata between versions using sequential migrations.

    This class manages version migrations by chaining registered migration
    functions. Only adjacent version migrations need to be registered;
    multi-version jumps (e.g., v1→v3) are handled automatically by
    running v1→v2 then v2→v3.

    Examples
    --------
    >>> @MetadataMigrator.register_migration("1.0", "2.0")
    ... def migrate_v1_to_v2(old: MetadataV1) -> MetadataV2:
    ...     return MetadataV2(
    ...         schema_version="2.0",
    ...         created_at=old.created_at,
    ...         # ... transform fields
    ...     )

    >>> v1_metadata = MetadataV1(...)
    >>> v2_metadata = MetadataMigrator.migrate(v1_metadata, "2.0")
    """

    _migrations: dict[tuple[str, str], Callable[["MetadataContract"], "MetadataContract"]] = {}

    def __init__(self) -> None:
        raise RuntimeError("MetadataMigrator is a static class and cannot be instantiated")

    @classmethod
    def register_migration(cls, from_version: str, to_version: str):
        """Register a migration function between adjacent versions.

        Migration functions receive the old metadata instance and return
        a new instance of the target version. Since metadata models are
        frozen, migrations always create new instances.

        Parameters
        ----------
        from_version : str
            Source version string (e.g., "1.0").
        to_version : str
            Target version string (e.g., "2.0").

        Returns
        -------
        Callable
            Decorator that registers the migration function.

        Raises
        ------
        ValueError
            If migration is already registered for this pair.

        Examples
        --------
        >>> @MetadataMigrator.register_migration("1.0", "2.0")
        ... def migrate_v1_to_v2(old: MetadataV1) -> MetadataV2:
        ...     return MetadataV2(...)
        """

        def decorator(
            func: Callable[["MetadataContract"], "MetadataContract"],
        ) -> Callable[["MetadataContract"], "MetadataContract"]:
            key = (from_version, to_version)
            if key in cls._migrations:
                raise ValueError(f"Migration from {from_version} to {to_version} already registered")
            cls._migrations[key] = func
            return func

        return decorator

    @classmethod
    def _get_migration_path(cls, from_ver: str, to_ver: str) -> list[str]:
        """Get ordered list of versions to migrate through.

        Parameters
        ----------
        from_ver : str
            Starting version.
        to_ver : str
            Target version.

        Returns
        -------
        list[str]
            List of versions forming the migration path.

        Raises
        ------
        ValueError
            If migration direction is invalid (downgrade).
        MigrationError
            If versions are not in the registry.
        """
        from .registry import MetadataRegistry

        all_versions = MetadataRegistry._list_versions()

        try:
            from_idx = all_versions.index(from_ver)
            to_idx = all_versions.index(to_ver)
        except ValueError as e:
            raise MigrationError(f"Version not found in registry: {e}") from e

        if from_idx >= to_idx:
            raise ValueError(f"Cannot migrate from {from_ver} to {to_ver} (downgrades not supported)")

        return all_versions[from_idx : to_idx + 1]

    @classmethod
    def migrate(cls, metadata: "MetadataContract", target_version: str, allow_stop: bool = False) -> "MetadataContract":
        """Migrate metadata to target version through sequential migrations.

        This method chains together registered migrations to move metadata
        from its current version to the target version. For example, migrating
        from v1 to v3 will run v1→v2 then v2→v3.

        Parameters
        ----------
        metadata : MetadataContract
            Source metadata instance.
        target_version : str
            Target version string.
        allow_stop : bool, optional
            If True, allows migration to stop at the last registered version
            before the target if no direct migration exists. Default is False.

        Returns
        -------
        MetadataContract
            Migrated metadata instance at target version.

        Raises
        ------
        ValueError
            If attempting a downgrade.
        MigrationError
            If a required migration is not registered.
        """
        current_version = metadata.schema_version

        if current_version is None:
            raise MigrationError("Cannot migrate metadata without a schema_version")

        if current_version == target_version:
            return metadata

        path = cls._get_migration_path(current_version, target_version)

        result = metadata
        for i in range(len(path) - 1):
            from_v, to_v = path[i], path[i + 1]
            migration_fn = cls._migrations.get((from_v, to_v))
            if migration_fn is None:
                if allow_stop:
                    break
                raise MigrationError(f"No migration registered from {from_v} to {to_v}")
            result = migration_fn(result)

            if result.schema_version != to_v:
                raise MigrationError(
                    f"Migration from {from_v} to {to_v} returned wrong version: " f"{result.schema_version}"
                )

        return result

    @classmethod
    def migrate_to_latest(cls, metadata: "MetadataContract") -> "MetadataContract":
        """Migrate metadata to the latest version.

        Convenience method that migrates to whatever the current latest
        registered version is.

        Parameters
        ----------
        metadata : MetadataContract
            Source metadata instance.

        Returns
        -------
        MetadataContract
            Migrated metadata instance at latest version.
        """
        from .registry import MetadataRegistry

        return cls.migrate(metadata, MetadataRegistry.latest_version())

    @classmethod
    def migrate_as_possible(cls, metadata: "MetadataContract", target_version: str) -> "MetadataContract":
        """Migrate metadata to the target version as far as possible.

        This method attempts to migrate metadata to the target version,
        but will stop at the last registered version if no direct migration
        exists. This is useful for cases where you want to migrate as far
        as possible without failing.

        Parameters
        ----------
        metadata : MetadataContract
            Source metadata instance.
        target_version : str
            Target version string.

        Returns
        -------
        MetadataContract
            Migrated metadata instance at the furthest possible version.
        """
        return cls.migrate(metadata, target_version, allow_stop=True)

    @classmethod
    def has_migration(cls, from_version: str, to_version: str) -> bool:
        """Check if a direct migration exists between two versions.

        Parameters
        ----------
        from_version : str
            Source version.
        to_version : str
            Target version.

        Returns
        -------
        bool
            True if a migration is registered for this pair.
        """
        return (from_version, to_version) in cls._migrations

    @staticmethod
    def remap_fields(data: dict, renames: dict[str, str]) -> dict:
        """Rename keys in a dictionary.

        Useful in migration functions for field renames between versions.
        Keys listed in *renames* that are absent from *data* are silently
        ignored.  The original dict is not mutated.

        Parameters
        ----------
        data : dict
            Source dictionary.
        renames : dict[str, str]
            Mapping of old key names to new key names.

        Returns
        -------
        dict
            New dictionary with keys renamed.
        """
        result = dict(data)
        for old_key, new_key in renames.items():
            if old_key in result:
                result[new_key] = result.pop(old_key)
        return result
