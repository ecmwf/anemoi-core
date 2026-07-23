# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sequential migration system for metadata versions.

Migrations can be defined between any versions (direct multi-hop jumps
or adjacent steps). Multi-version migrations are resolved by preferring
the longest registered direct jump that doesn't overshoot, with automatic
no-op version bumps for schema-sharing minors (registered via
``MetadataRegistry.register_minor``).

Version Policy
--------------
- Versions use major.minor only (no patch): "1.0", "2.0", "2.1".
- Migrations are defined between any registered versions.
- Downgrades are not supported - use ``migrate=False`` to preserve original.
- Schema-sharing minors are automatically bridged with no-op version bumps.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from .exceptions import MigrationError

if TYPE_CHECKING:
    from .base import MetadataContract


class MetadataMigrator:
    """Migrate metadata between versions using sequential migrations.

    This class manages version migrations by chaining registered migration
    functions. Migrations can be registered between any versions (not just
    adjacent ones). Multi-version jumps are resolved by preferring the
    longest direct registered migration that doesn't overshoot the target.
    Schema-sharing minor versions (registered via ``register_minor``) are
    automatically bridged with no-op version bumps.

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
    def _plan_steps(cls, from_ver: str, to_ver: str) -> tuple[list[tuple[str, str, bool]], tuple[str, str] | None]:
        """Plan the migration steps from from_ver to to_ver.

        Prefers the longest registered direct migration that doesn't overshoot.
        Falls back to no-op version bumps for schema-sharing minors.  When the
        path cannot be completed, the resolvable prefix is returned together
        with the point at which planning got stuck, so callers can decide
        whether to migrate as far as possible or fail.

        Parameters
        ----------
        from_ver : str
            Starting version.
        to_ver : str
            Target version.

        Returns
        -------
        steps : list[tuple[str, str, bool]]
            List of (from_version, to_version, is_noop) steps. is_noop is True
            for schema-sharing version bumps, False for registered migrations.
        unresolved : tuple[str, str] or None
            ``None`` when the plan reaches *to_ver*.  Otherwise the
            ``(stuck_version, next_needed_version)`` pair at which no
            registered migration or no-op bridge was available.

        Raises
        ------
        ValueError
            If migration direction is invalid (downgrade).
        MigrationError
            If either version is not in the registry.
        """
        from packaging.version import Version

        from .registry import MetadataRegistry

        all_versions = MetadataRegistry._list_versions()

        if from_ver not in all_versions:
            raise MigrationError(f"Version not found in registry: {from_ver}")
        if to_ver not in all_versions:
            raise MigrationError(f"Version not found in registry: {to_ver}")

        if Version(from_ver) >= Version(to_ver):
            raise ValueError(f"Cannot migrate from {from_ver} to {to_ver} (downgrades not supported)")

        steps: list[tuple[str, str, bool]] = []
        current = from_ver

        while current != to_ver:
            # Prefer the registered migration that jumps farthest without overshooting.
            candidates = [
                t
                for (f, t) in cls._migrations
                if f == current and t in all_versions and Version(current) < Version(t) <= Version(to_ver)
            ]

            if candidates:
                # Jump to the farthest reachable version.
                nxt = max(candidates, key=Version)
                steps.append((current, nxt, False))
                current = nxt
                continue

            # Fall back: synthesise a no-op step to the next registered version
            # if it shares the same schema class.
            current_idx = all_versions.index(current)

            # Find the next version in sorted order that doesn't overshoot.
            nxt = None
            for idx in range(current_idx + 1, len(all_versions)):
                candidate = all_versions[idx]
                if Version(candidate) <= Version(to_ver):
                    nxt = candidate
                break

            if nxt is None:
                # No next version within range.
                return steps, (current, to_ver)

            # Check if current and nxt share the same schema class.
            if MetadataRegistry.get_version(current) is MetadataRegistry.get_version(nxt):
                steps.append((current, nxt, True))
                current = nxt
                continue

            # Unresolvable from here: no migration and schemas differ.
            return steps, (current, nxt)

        return steps, None

    @classmethod
    def migrate(cls, metadata: "MetadataContract", target_version: str, allow_stop: bool = False) -> "MetadataContract":
        """Migrate metadata to target version through sequential migrations.

        This method chains together registered migrations to move metadata
        from its current version to the target version. Prefers the longest
        direct registered migration that doesn't overshoot. Automatically
        bridges schema-sharing minor versions with no-op version bumps.

        Parameters
        ----------
        metadata : MetadataContract
            Source metadata instance.
        target_version : str
            Target version string.
        allow_stop : bool, optional
            If True, migrate as far as possible along the resolvable prefix of
            the path and stop (with a ``UserWarning``) instead of raising when
            no complete path to the target exists. Default is False.

        Returns
        -------
        MetadataContract
            Migrated metadata instance at target version (or furthest reachable
            version if allow_stop is True).

        Raises
        ------
        ValueError
            If attempting a downgrade.
        MigrationError
            If a required migration is not registered and allow_stop is False.

        Warns
        -----
        UserWarning
            If allow_stop is True and migration stopped short of the target.
        """
        current_version = metadata.schema_version

        if current_version is None:
            raise MigrationError("Cannot migrate metadata without a schema_version")

        if current_version == target_version:
            return metadata

        # Plan the migration steps.  Unknown versions and downgrades always
        # raise; an incomplete path is reported via `unresolved` so we can
        # still execute the resolvable prefix when allow_stop is True.
        steps, unresolved = cls._plan_steps(current_version, target_version)

        if unresolved is not None and not allow_stop:
            stuck_at, next_needed = unresolved
            raise MigrationError(f"No migration registered from {stuck_at} to {next_needed}")

        result = metadata
        for from_v, to_v, is_noop in steps:
            if is_noop:
                # Schema-sharing version bump: use copy_with.
                result = result.copy_with(schema_version=to_v)
            else:
                # Registered migration.
                migration_fn = cls._migrations[(from_v, to_v)]
                result = migration_fn(result)

                if result.schema_version != to_v:
                    raise MigrationError(
                        f"Migration from {from_v} to {to_v} returned wrong version: " f"{result.schema_version}"
                    )

        if unresolved is not None:
            stuck_at, next_needed = unresolved
            warnings.warn(
                f"Migration stopped at version {stuck_at}: no migration "
                f"registered from {stuck_at} to {next_needed} "
                f"(target was {target_version}).",
                UserWarning,
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
        executing the resolvable prefix of the migration path and stopping
        at the furthest reachable version when no complete path exists.
        A ``UserWarning`` is emitted when migration stops short of the
        target. This is useful for cases where you want to migrate as far
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

        Warns
        -----
        UserWarning
            If migration stopped short of the target version.
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
        ignored.  A shallow copy is made: top-level keys are not mutated,
        but nested values are shared with the input.

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
