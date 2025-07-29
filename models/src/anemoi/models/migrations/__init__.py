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

    def serialize(self) -> dict[str, Any]:
        serialized_rollback = None
        if self.rollback is not None:
            cloudpickle.register_pickle_by_value(sys.modules[self.rollback.__module__])
            rollback_bytes = cloudpickle.dumps(self.rollback)
            serialized_rollback = _SerializedRollback(rollback_bytes)
        return {"name": self.name, "rollback": serialized_rollback}


def registered_migrations(ckpt: CkptType) -> list[dict[str, Any]]:
    """Return all registered migrations from a checkpoint.
    Parameters
    ----------
    ckpt : CkptType
        The checkpoint

    Returns
    -------
    list[dict[str, Any]]
        The registered migrations
    """
    if _ckpt_migration_key not in ckpt:
        return []
    return ckpt[_ckpt_migration_key]


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

    def _get_missing_migrations(self, ckpt: CkptType, migrations: list[Migration]) -> list[Migration]:
        """Get missing migrations from a checkpoint

        Parameters
        ----------
        ckpt : CkptType
            The loaded checkpoint
        migrations : Sequence[Migration]
            List of migration to execute

        Returns
        -------
        list[Migration]
                Missing migrations from the checkpoint to execute
        """
        if _ckpt_migration_key not in ckpt:
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        done_migrations = [mig["name"] for mig in ckpt[_ckpt_migration_key]]
        # Migration should be done in order, we look for the the last done migration and
        # execute the rest. This is to allow havind removed migrations in a checkpoint and
        # not complain.
        key_rest_migration = 0
        num_migrations = len(migrations)
        for k, mig in enumerate(reversed(migrations)):
            if mig.name in done_migrations:
                key_rest_migration = num_migrations - k
                break

        if self._raise_missing_migrations:
            for migration in migrations[:key_rest_migration]:
                if migration.name not in done_migrations:
                    raise MissingMigrationException(
                        f"{migration.name} is not part of the checkpoint but cannot be executed (out of order)."
                    )
        return migrations[key_rest_migration:]

    def sync(self, ckpt: CkptType, steps: int | None = None) -> tuple[CkptType, list[str], list[str]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to migrate.
        steps : int | None, default None
            Number of relative migration step to execute. If negative, will rollback the provided number of steps.
            Mutually exclusive with steps and target. Defaults to migrate all missing migrations.

        Returns
        -------
        tuple[CkptType, list[str], list[str]]
            * The migrated checkpoint
            * The list of migrations that were applied to the checkpoint
            * The list of rollbacks that were applied
        """
        ckpt = deepcopy(ckpt)

        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]

        if len(compatible_migrations) < len(ckpt[_ckpt_migration_key]):
            # We should rollback, set a negative steps
            steps = len(compatible_migrations) - len(ckpt[_ckpt_migration_key])

        if steps is not None and steps <= 0:
            ckpt, rollbacks = self._rollback(ckpt, -steps)
            return ckpt, [], rollbacks
        ckpt, missing_migrations = self._migrate(ckpt, compatible_migrations, steps)
        return ckpt, missing_migrations, []

    def _migrate(
        self,
        ckpt: CkptType,
        compatible_migrations: list[Migration],
        steps: int | None = None,
    ) -> tuple[CkptType, list[str]]:
        """Rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to migrate.
        steps : int | None, default None
            Number of migration step to execute. Defaults to all missing migrations.
        Returns
        -------
        tuple[CkptType, list[str]]
            * The migrated checkpoint
            * The list of migrations that were applied to the checkpoint
        """
        if steps is not None and steps < 0:
            raise ValueError("Cannot migrate negative number of steps. Use sync instead")

        assert _ckpt_migration_key in ckpt
        missing_migrations = self._get_missing_migrations(ckpt, compatible_migrations)
        if steps is not None:
            if steps > len(missing_migrations):
                raise IncompatibleCheckpointException(
                    f"Checkpoint cannot be migrated {steps} steps. (Max: {len(missing_migrations)})."
                )
            missing_migrations = missing_migrations[:steps]
        migrated: list[str] = []
        for migration in missing_migrations:
            if migration.migrate is None:
                raise IncompatibleCheckpointException(
                    f"Migration {migration.name} cannot be executed. Missing migrate function."
                )
            ckpt = migration.migrate(ckpt)
            ckpt[_ckpt_migration_key].append(migration.serialize())
            migrated.append(migration.name)
        return ckpt, migrated

    def _rollback(self, ckpt: CkptType, steps: int) -> tuple[CkptType, list[str]]:
        """Rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to rollback.
        steps : int
            Number of rollback steps to execute.

        Returns
        -------
        Tuple[CkptType, Sequence[Migration], List[str]]
            * The migrated checkpoint
            * The list of rollbacks that were applied
        """
        if steps is not None and steps < 0:
            raise ValueError("Cannot rollback negative number of steps. Use sync instead")

        assert _ckpt_migration_key in ckpt
        rollbacks: list[str] = []
        for _ in range(steps):
            migration = ckpt[_ckpt_migration_key].pop()
            rollbacks = [migration["name"]] + rollbacks
            if migration["rollback"] is None:
                raise IncompatibleCheckpointException(f"{migration['name']} cannot bo rollbacked.")
            ckpt = migration["rollback"](ckpt)
        return ckpt, rollbacks

    def inspect(self, ckpt: CkptType) -> tuple[list[Migration], list[Migration], list[str]]:
        """Inspect migration information in checkpoint

        Parameters
        ----------
        ckpt : CkptType
            The chekpoint to inspect

        Returns
        -------
        tuple[list[Migration], list[Migration], list[str]]
            * The list of already executed migrations
            * The list of missing migrations
            * The list of extra migrations in the checkpoint (to rollback)
        """
        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]
        registered_migrations = self.registered_migrations(ckpt)
        common_migrations: list[Migration] = []
        extra_migrations: list[str] = []
        missing_migrations: list[Migration] = []
        k = 0
        for migration in compatible_migrations:
            if migration.name in registered_migrations:
                common_migrations.append(migration)
            else:
                missing_migrations.append(migration)
            k += 1
        if len(registered_migrations) > k:
            for migration in registered_migrations[k:]:
                extra_migrations.append(migration)
        return common_migrations, missing_migrations, extra_migrations

    def registered_migrations(self, ckpt: CkptType) -> list[str]:
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
        return [migration["name"] for migration in ckpt[_ckpt_migration_key]]

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


__all__ = ["MIGRATION_PATH", "CkptType", "Migration", "Migrator", "registered_migrations"]
