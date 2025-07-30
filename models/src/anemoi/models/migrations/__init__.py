# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from collections.abc import MutableMapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from enum import auto
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any
from typing import TypedDict

import cloudpickle

MIGRATION_PATH = Path(__file__).parent

_ckpt_migration_key = "migrations"

LOGGER = logging.getLogger(__name__)


class MissingMigrationException(BaseException):
    """The checkpoint is missing a migration that cannot be added (wrong order)."""

    pass


class IncompatibleCheckpointException(BaseException):
    """The provided checkpoitn cannot be migrated because it is to old/recent."""

    pass


CkptType = MutableMapping[str, Any]


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-models": str})


@dataclass
class MigrationMetadata:
    versions: MigrationVersions
    final: bool = False


class SerializedMigration(TypedDict):
    name: str
    rollback: Callable[[CkptType], CkptType] | None
    metadata: MigrationMetadata


class _SerializedRollback:
    """Use cloudpickle to serialize the rollback function by value and not reference.
    When doing rollbacks, migration files might not exist anymore, and we need to
    execute the migration from the checkpoint directly.
    """

    def __init__(self, rollback_bytes: bytes):
        self._rollback_bytes = rollback_bytes

    @cached_property
    def rollback(self) -> Callable[[CkptType], CkptType]:
        return cloudpickle.loads(self._rollback_bytes)

    def __call__(self, ckpt: CkptType) -> CkptType:
        return self.rollback(ckpt)

    def __reduce__(self) -> tuple[Callable[[bytes], _SerializedRollback], tuple[bytes]]:
        return self.__class__, (self._rollback_bytes,)


@dataclass
class Migration:
    """Represents a migration"""

    name: str
    """Name of the migration"""
    migrate: Callable[[CkptType], CkptType] | None
    """Callback to execute the migration"""
    rollback: Callable[[CkptType], CkptType] | None
    """Callback to execute a migration rollback"""
    metadata: MigrationMetadata
    """Tracked metadata"""

    @classmethod
    def from_serialized(cls, migration: SerializedMigration) -> Migration:
        return Migration(migration["name"], None, migration["rollback"], migration["metadata"])

    def serialize(self) -> SerializedMigration:
        serialized_rollback = None
        if self.rollback is not None:
            cloudpickle.register_pickle_by_value(sys.modules[self.rollback.__module__])
            rollback_bytes = cloudpickle.dumps(self.rollback)
            serialized_rollback = _SerializedRollback(rollback_bytes)
        return {"name": self.name, "rollback": serialized_rollback, "metadata": self.metadata}


class OpType(Enum):
    migration = auto()
    rollback = auto()


def _migrations_from_path(location: str | PathLike, package: str) -> list[Migration]:
    """Returns the migrations from a given folder

    Parameters
    ----------
    location : str | PathLike
        Path to the migration folder
    package : str
        Reference package for the import of the migrations

    Returns
    -------
    list[Migration]
        The migrations from the given path
    """
    migrations: list[Migration] = []

    for file in sorted(Path(location).iterdir()):
        if not file.is_file() and file.suffix != ".py" or file.name == "__init__.py":
            continue
        LOGGER.debug("Loading migration .%s from %s", file.stem, package)
        try:
            migration = importlib.import_module(f".{file.stem}", package)
        except ImportError as e:
            LOGGER.warning("Error loading %s: %s", file.name, str(e))
            continue

        args: dict[str, Any] = dict(name=file.stem, migrate=None, rollback=None, metadata=migration.metadata)
        if hasattr(migration, "migrate"):
            args["migrate"] = migration.migrate
        if hasattr(migration, "rollback"):
            args["rollback"] = migration.rollback
        migrations.append(Migration(**args))
    return migrations


class Migrator:
    def __init__(self, migrations: Sequence[Migration] | None = None, raise_missing_migrations: bool = True) -> None:
        """Create the migrator object

        Parameters
        ----------
        migrations : Sequence[Migration] | None, default None
            List of migration to execute. If None, get migrations from the current folder.
        raise_missing_migrations : bool
            Whether to check if there are out of order migrations missing from the checkpoint
        """

        if migrations is None:
            migrations = _migrations_from_path(MIGRATION_PATH, __name__)

        # Compatibility groups. Checkpoints cannot be migrated past their
        # own group. This is useful to indicate when migrating checkpoints is no longer
        # supported.
        self._grouped_migrations: list[list[Migration]] = []
        current_group: list[Migration] = []
        for migration in migrations:
            if migration.metadata.final:
                self._grouped_migrations.append(current_group)
                current_group = []
            current_group.append(migration)
        self._grouped_migrations.append(current_group)

        self._raise_missing_migrations = raise_missing_migrations

    @classmethod
    def from_path(cls, location: str | PathLike, package: str) -> Migrator:
        """Load from a given folder

        Parameters
        ----------
        location : str | PathLike
            Path to the migration folder
        package : str
            Reference package for the import of the migrations

        Returns
        -------
        A Migrator instance
        """
        return cls(_migrations_from_path(location, package))

    def is_compatible_ckpt(self, ckpt: CkptType) -> bool:
        """Checks whether the ckpt is compatible with the current version.

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint

        Returns
        -------
        bool
            Whether it is compatible
        """

        # No migration means checkpoint too old, no migrations available.
        if _ckpt_migration_key not in ckpt:
            return False
        # If empty, means first group
        if not len(ckpt[_ckpt_migration_key]):
            if len(self._grouped_migrations) > 1:
                return False
            else:
                return True

        first_migration = ckpt[_ckpt_migration_key][0]["name"]
        # Compare the first migration of the last group
        # Migrations that are not in the first group must always have at least the previous "final" migration registered.
        if self._grouped_migrations[-1][0].name == first_migration:
            return True
        return False

    def _resolve_ops(self, ckpt: CkptType, migrations: list[Migration]) -> list[tuple[OpType, Migration]]:
        ckpt_migrations = self.registered_migrations(ckpt)
        ops: list[tuple[OpType, Migration]] = []
        n_ckpt_migrations = len(ckpt_migrations)
        for k, ckpt_migration in enumerate(reversed(ckpt_migrations), 1):
            if (
                len(migrations) > n_ckpt_migrations - k
                and migrations[n_ckpt_migrations - k].name == ckpt_migration.name
            ):
                break

            if ckpt_migration.rollback is None:
                raise IncompatibleCheckpointException(
                    f"{ckpt_migration.name} cannot bo rollbacked. Missing rollback function."
                )
            ops.append((OpType.rollback, ckpt_migration))

        num_rollbacks = len(ops)
        for k, migration in enumerate(migrations):
            if (
                len(ckpt_migrations[: len(ckpt_migrations) - num_rollbacks]) > k
                and migration.name == ckpt_migrations[k].name
            ):
                continue
            if migration.migrate is None:
                raise IncompatibleCheckpointException(
                    f"Migration {migration.name} cannot be executed. Missing migrate function."
                )
            ops.append((OpType.migration, migration))
        return ops

    def sync(self, ckpt: CkptType, steps: int | None = None) -> tuple[CkptType, list[tuple[OpType, Migration]]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to migrate.
        steps : int | None, default None
            Number of steps to execute. Cannot be negative.

        Returns
        -------
        tuple[CkptType, list[tuple[OpType, Migration]]]
            * The migrated checkpoint
            * The list of migrations or rollbacks
        """
        ckpt = deepcopy(ckpt)

        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]
        ops = self._resolve_ops(ckpt, compatible_migrations)
        if steps is not None and steps < 0:
            raise ValueError("steps should be positive.")
        if steps is not None:
            ops = ops[:steps]
        for op_type, callback in ops:
            if op_type is OpType.rollback:
                assert callback.rollback is not None
                ckpt = callback.rollback(ckpt)
                ckpt[_ckpt_migration_key].pop()
            else:
                assert callback.migrate is not None
                ckpt = callback.migrate(ckpt)
                ckpt[_ckpt_migration_key].append(callback.serialize())
        return ckpt, ops

    def inspect(self, ckpt: CkptType) -> tuple[list[Migration], list[Migration], list[Migration]]:
        """Inspect migration information in checkpoint

        Parameters
        ----------
        ckpt : CkptType
            The chekpoint to inspect

        Returns
        -------
        tuple[list[Migration], list[Migration], list[Migration]]
            * The list of already executed migrations
            * The list of missing migrations
            * The list of extra migrations in the checkpoint (to rollback)
        """
        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]
        registered_migrations = self.registered_migrations(ckpt)
        ops = self._resolve_ops(ckpt, compatible_migrations)
        missing_migrations: list[Migration] = []
        extra_migrations: list[Migration] = []
        for op_type, op in ops:
            if op_type is OpType.rollback:
                extra_migrations.append(op)
                registered_migrations.pop()
            else:
                missing_migrations.append(op)
        return registered_migrations, missing_migrations, extra_migrations

    def registered_migrations(self, ckpt: CkptType) -> list[Migration]:
        """Registered migrations in a ckpt

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint

        Returns
        -------
        list[str]
            The names of registered migrations
        """
        if _ckpt_migration_key not in ckpt:
            return []
        return [Migration.from_serialized(migration) for migration in ckpt[_ckpt_migration_key]]

    def register_migrations(self, ckpt: CkptType) -> CkptType:
        """Registers a list of migration to the checkpoint.
        Note: this does not execute any migration. It only registers them in the migration
        key of the checkpoint.

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint

        Returns
        -------
        CkptType
            Checkpoint with registered migrations
        """
        if _ckpt_migration_key not in ckpt:
            ckpt[_ckpt_migration_key] = []
        for migration in self._grouped_migrations[-1]:
            ckpt[_ckpt_migration_key].append(migration.serialize())
        return ckpt


__all__ = [
    "CkptType",
    "IncompatibleCheckpointException",
    "Migration",
    "Migrator",
    "MigrationMetadata",
    "MigrationVersions",
    "MIGRATION_PATH",
    "MissingMigrationException",
    "OpType",
    "SerializedMigration",
]
