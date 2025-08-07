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
import types
from collections.abc import Callable
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from enum import auto
from functools import cache
from functools import cached_property
from os import PathLike
from pathlib import Path
from pickle import Unpickler
from typing import Any
from typing import TypedDict

import cloudpickle

MIGRATION_PATH = Path(__file__).parent

_ckpt_migration_key = "migrations"

LOGGER = logging.getLogger(__name__)


class MissingMigrationException(BaseException):
    """The checkpoint is missing a migration that cannot be added (wrong order)."""


class IncompatibleCheckpointException(BaseException):
    """The provided checkpoitn cannot be migrated because it is to old/recent."""


CkptType = MutableMapping[str, Any]


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-models": str})


class MigrationContext:
    def __init__(self) -> None:
        self.module_paths: dict[str, str] = {}
        self.attribute_paths: dict[str, str] = {}

    def move_attribute(self, path_start: str, path_end: str) -> None:
        if path_start in self.attribute_paths:
            path_start = self.attribute_paths.pop(path_start)
        self.attribute_paths[path_end] = path_start

    def move_module(self, path_start: str, path_end: str) -> None:
        if path_start in self.module_paths:
            path_start = self.module_paths.pop(path_start)
        self.module_paths[path_end] = path_start


@dataclass
class MigrationMetadata:
    versions: MigrationVersions
    final: bool = False


class SerializedMigration(TypedDict):
    name: str
    metadata: MigrationMetadata
    rollback: Callable[[CkptType], CkptType] | None
    rollback_setup: Callable[[MigrationContext], None] | None


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


class _SerializedRollbackSetup:
    """Use cloudpickle to serialize the rollback_setup function by value and not reference."""

    def __init__(self, rollback_setup_bytes: bytes):
        self._rollback_setup_bytes = rollback_setup_bytes

    @cached_property
    def rollback_setup(self) -> Callable[[MigrationContext], None]:
        return cloudpickle.loads(self._rollback_setup_bytes)

    def __call__(self, context: MigrationContext) -> None:
        return self.rollback_setup(context)

    def __reduce__(self) -> tuple[Callable[[bytes], _SerializedRollbackSetup], tuple[bytes]]:
        return self.__class__, (self._rollback_setup_bytes,)


@dataclass
class Migration:
    """Represents a migration"""

    name: str
    """Name of the migration"""
    metadata: MigrationMetadata
    """Tracked metadata"""
    migrate: Callable[[CkptType], CkptType] | None = None
    """Callback to execute the migration"""
    migrate_setup: Callable[[MigrationContext], None] | None = None
    """Setup function to execute before loading the checkpoint. This can be used to
    mock missing modules or Attributes."""
    rollback: Callable[[CkptType], CkptType] | None = None
    """Callback to execute a migration rollback"""
    rollback_setup: Callable[[MigrationContext], None] | None = None
    """Setup function to execute before loading the checkpoint for rollback. This can be used to
    mock missing modules or Attributes."""

    @classmethod
    def from_serialized(cls, migration: SerializedMigration) -> Migration:
        return Migration(
            migration["name"], migration["metadata"], None, None, migration["rollback"], migration["rollback_setup"]
        )

    def serialize(self) -> SerializedMigration:
        serialized_rollback: _SerializedRollback | None = None
        serialized_rollback_setup: _SerializedRollbackSetup | None = None
        if self.rollback is not None:
            cloudpickle.register_pickle_by_value(sys.modules[self.rollback.__module__])
            rollback_bytes = cloudpickle.dumps(self.rollback)
            serialized_rollback = _SerializedRollback(rollback_bytes)
        if self.rollback_setup is not None:
            cloudpickle.register_pickle_by_value(sys.modules[self.rollback_setup.__module__])
            rollback_setup_bytes = cloudpickle.dumps(self.rollback_setup)
            serialized_rollback_setup = _SerializedRollbackSetup(rollback_setup_bytes)
        return {
            "name": self.name,
            "metadata": self.metadata,
            "rollback": serialized_rollback,
            "rollback_setup": serialized_rollback_setup,
        }


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

        args: dict[str, Any] = dict(name=file.stem, metadata=migration.metadata)
        if hasattr(migration, "migrate"):
            args["migrate"] = migration.migrate
        if hasattr(migration, "migrate_setup"):
            args["migrate_setup"] = migration.migrate_setup
        if hasattr(migration, "rollback"):
            args["rollback"] = migration.rollback
        if hasattr(migration, "rollback_setup"):
            args["rollback_setup"] = migration.rollback_setup
        migrations.append(Migration(**args))
    return migrations


class MissingAttribute: ...


class LenientUnpickler(Unpickler):
    def find_class(self, module_name: str, global_name: str, /) -> Any:
        try:
            return super().find_class(module_name, global_name)
        except (ImportError, AttributeError):
            LOGGER.debug("Missing attribute %s.%s is checkpoint. Ignoring.", module_name, global_name)
            return MissingAttribute


class LenientPicklerModule:
    Unpickler = LenientUnpickler


@cache
def _load_ckpt(path: str | PathLike, lenient: bool = True) -> CkptType:
    import torch

    pickle_module: Any = None
    if lenient:
        pickle_module = LenientPicklerModule
    ckpt = torch.load(path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    # TODO: remove this. Only for testing
    if _ckpt_migration_key not in ckpt:
        ckpt[_ckpt_migration_key] = []
    return ckpt


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

    def _resolve_ops(
        self, ckpt: CkptType, migrations: list[Migration], steps: int | None = None
    ) -> tuple[list[Callable[[MigrationContext], None]], list[tuple[OpType, Migration]]]:
        ckpt_migrations = self.registered_migrations(ckpt)
        setups: list[Callable[[MigrationContext], None]] = []
        ops: list[tuple[OpType, Migration]] = []
        n_ckpt_migrations = len(ckpt_migrations)
        for k, ckpt_migration in enumerate(reversed(ckpt_migrations), 1):
            if steps is not None and len(ops) == steps:
                break
            if (
                len(migrations) > n_ckpt_migrations - k
                and migrations[n_ckpt_migrations - k].name == ckpt_migration.name
            ):
                break

            if ckpt_migration.rollback is None:
                raise IncompatibleCheckpointException(
                    f"{ckpt_migration.name} cannot bo rollbacked. Missing rollback function."
                )
            if ckpt_migration.rollback_setup is not None:
                setups.append(ckpt_migration.rollback_setup)
            ops.append((OpType.rollback, ckpt_migration))

        num_rollbacks = len(ops)
        for k, migration in enumerate(migrations):
            if steps is not None and len(ops) == steps:
                break
            if (
                len(ckpt_migrations[: len(ckpt_migrations) - num_rollbacks]) > k
                and migration.name == ckpt_migrations[k].name
            ):
                continue
            if migration.migrate is None:
                raise IncompatibleCheckpointException(
                    f"Migration {migration.name} cannot be executed. Missing migrate function."
                )
            if migration.migrate_setup is not None:
                setups.append(migration.migrate_setup)
            ops.append((OpType.migration, migration))
        return setups, ops

    def _resolve_context(self, context: MigrationContext) -> None:
        for full_attribute_path_end, attribute_path_start in context.attribute_paths.items():
            attribute_path_start, _, mod_name_start = attribute_path_start.rpartition(".")
            attribute_path_end, _, mod_name_end = full_attribute_path_end.rpartition(".")
            LOGGER.debug(
                "Move attribute %s from %s to %s.", mod_name_start, attribute_path_start, full_attribute_path_end
            )
            mod_end = importlib.import_module(attribute_path_end, __name__)
            attr_end = getattr(mod_end, mod_name_end)
            mod_start = sys.modules[attribute_path_start]
            setattr(mod_start, mod_name_start, attr_end)
        for module_path_end, module_path_start in context.module_paths.items():
            LOGGER.debug("Move module %s to %s.", module_path_start, module_path_end)
            sys.modules[module_path_start] = sys.modules[module_path_end]

    def sync(
        self, path: str | PathLike | CkptType, steps: int | None = None
    ) -> tuple[CkptType, list[tuple[OpType, Migration]]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        path : str | PathLike | CkptType
            The checkpoint to migrate.
        steps : int | None, default None
            Number of steps to execute. Cannot be negative.

        Returns
        -------
        tuple[CkptType, list[tuple[OpType, Migration]]]
            * The migrated checkpoint
            * The list of migrations or rollbacks
        """
        if isinstance(path, str | PathLike):
            ckpt = _load_ckpt(path)
        else:
            ckpt = path

        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]
        if steps is not None and steps < 0:
            raise ValueError("steps should be positive.")
        setups, ops = self._resolve_ops(ckpt, compatible_migrations, steps)
        # Setups are useful only of we load the checkpoint from path
        # Otherwise, this means that the checkpoint could already be loaded.
        if isinstance(path, str | PathLike) and len(setups):
            context = MigrationContext()
            for setup in setups:
                setup(context)
            self._resolve_context(context)
            # Force reloading checkpoint without obfuscating import issues.
            ckpt = _load_ckpt(path, lenient=False)
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

    def inspect(self, path: str | PathLike) -> tuple[list[Migration], list[Migration], list[Migration]]:
        """Inspect migration information in checkpoint

        Parameters
        ----------
        path : str | PathLike
            Path to the chekpoint to inspect

        Returns
        -------
        tuple[list[Migration], list[Migration], list[Migration]]
            * The list of already executed migrations
            * The list of missing migrations
            * The list of extra migrations in the checkpoint (to rollback)
        """
        ckpt = _load_ckpt(path)
        if not self.is_compatible_ckpt(ckpt):
            raise IncompatibleCheckpointException("This checkpoint is too old and cannot be migrated.")
        compatible_migrations = self._grouped_migrations[-1]
        registered_migrations = self.registered_migrations(ckpt)
        _, ops = self._resolve_ops(ckpt, compatible_migrations)
        missing_migrations: list[Migration] = []
        extra_migrations: list[Migration] = []
        for op_type, op in ops:
            if op_type is OpType.rollback:
                extra_migrations.append(op)
                registered_migrations.pop()
            else:
                missing_migrations.append(op)
        return registered_migrations, missing_migrations, extra_migrations

    def registered_migrations(self, ckpt: str | PathLike | CkptType) -> list[Migration]:
        """Registered migrations in a ckpt

        Parameters
        ----------
        ckpt : str | PathLike | CkptType
            The checkpoint

        Returns
        -------
        list[str]
            The names of registered migrations
        """
        if isinstance(ckpt, str | PathLike):
            ckpt = _load_ckpt(ckpt)
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
    "MigrationContext",
    "MigrationMetadata",
    "MigrationVersions",
    "MIGRATION_PATH",
    "MissingMigrationException",
    "OpType",
    "SerializedMigration",
]
