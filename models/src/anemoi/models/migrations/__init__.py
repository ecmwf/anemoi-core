# (C) Copyright 2025 Anemoi contributors.
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
    """A context object allowing setup callbacks to access some utilities:
    * ``context.move_attribute("pkg.start.MyClass", "pkg.end.MyRenamedClass")`` to update paths
        to attributes.
    * ``context.move_module("pkg.start", "pkg.end")`` to move a full module.
    * ``context.delete_attribute("pkg.mod.MyClass")`` to remove a class you can use "*" as
        a wildcard for the attribute name: ``context.delete_attribute("pkg.mod.*")`` will remove
        all attribute from the module.
    """

    def __init__(self) -> None:
        self.attribute_paths: dict[str, str] = {}
        self.module_paths: dict[str, str] = {}
        self.deleted_attributes: list[str] = []

    def delete_attribute(self, path: str) -> None:
        """Indicate that an attribute has been deleted. Any class referencing this module will
        be replace by a ``MissingAttribute`` object.

        Parameters
        ----------
        path : str
            Path to the attribute. For example ``pkg.mod.MyClass``.
        """
        self.deleted_attributes.append(path)

    def move_attribute(self, path_start: str, path_end: str) -> None:
        """Move and rename an attribute between modules.

        Parameters
        ----------
        path_start : str
            Starting module path
        path_end : str
            End module path
        """
        if path_start in self.attribute_paths:
            path_start = self.attribute_paths.pop(path_start)
        self.attribute_paths[path_end] = path_start

    def move_module(self, path_start: str, path_end: str) -> None:
        """Move a module.

        Parameters
        ----------
        path_start : str
            Starting module path
        path_end : str
            End module path
        """
        if path_start in self.module_paths:
            path_start = self.module_paths.pop(path_start)
        self.module_paths[path_end] = path_start


@dataclass
class MigrationMetadata:
    """Metadata object of the migration."""

    versions: MigrationVersions
    """ Migration and anemoi-model versions. """
    final: bool = False
    """ Whether the migration is final."""


class SerializedMigration(TypedDict):
    """The serialized migration stored in the checkpoint"""

    name: str
    """ Name of the migration """
    metadata: MigrationMetadata
    rollback: Callable[[CkptType], CkptType] | None
    """ The rollback function stored as value """
    rollback_setup: Callable[[MigrationContext], None] | None
    """ The setup callback for the rollback method """


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
        """Alt init to load the migration from the serialized migration dict in the checkpoint
        This migration does not contain the ``migrate`` or ``migrate_setup`` callbacks as
        they are not serialized.

        Parameters
        ----------
        migration : SerializedMigration
            The serialized migration dict

        Returns
        -------
        Migration
            The migration.
        """
        return Migration(
            migration["name"], migration["metadata"], None, None, migration["rollback"], migration["rollback_setup"]
        )

    def serialize(self) -> SerializedMigration:
        """Serialize this migration

        Returns
        -------
        SerializedMigration
            The serialized dict to store in the checkpoint.
        """
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


@dataclass
class BaseOp:
    """Base class for operations."""

    run: Callable[[CkptType], CkptType]
    migration: Migration


@dataclass
class MigrationOp(BaseOp):
    """Migration Operation"""


@dataclass
class RollbackOp(BaseOp):
    """Rollback Operation"""


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


class MissingAttribute:
    """Placeholder type when encountering ImportError or AttributeError in Unpickler.find_class"""


def get_unpickler(replace_attrs: list[str] | bool = False):
    """Get the Unpickler

    Parameters
    ----------
    replace_attrs : list[str] | bool, default False
        Replace the provided attrs by a ``MissingAttribute`` object. If False, Fill not
        try to replace attributes. If True, will replace every missing attribute. You can use
        * as a wildcard to be replaced by any attribute in a module.

    Returns
    -------
    Any
        An Unpickler wrapper for torch.load.
    """

    class _Unpickler(Unpickler):
        """And Unpickler that does not fail when the pickle object has some reference to non-existing attributes.
        This is useful to load the "migrations" key from the checkpoint regardless of import issues.
        """

        def find_class(self, module_name: str, global_name: str, /) -> Any:
            try:
                return super().find_class(module_name, global_name)
            except (ImportError, AttributeError) as e:
                attr_name = f"{module_name}.{global_name}"
                wild_name = f"{module_name}.*"
                if replace_attrs is False:
                    raise e
                if replace_attrs is True or attr_name in replace_attrs or wild_name in replace_attrs:
                    LOGGER.debug("Missing attribute %s.%s is checkpoint. Ignoring.", module_name, global_name)
                    return MissingAttribute
                raise e

    class UnpicklerWrapper:
        """For torch.load's pickle_module argument.
        A "module" with the LenientUnpickler as Unpickler.
        """

        Unpickler = _Unpickler

    return UnpicklerWrapper


def _load_ckpt(path: str | PathLike, replace_attrs: list[str] | bool = False) -> CkptType:
    """Loads a checkpoint

    Parameters
    ----------
    path : str | PathLike
        Checkpoint path
    replace_attrs : list[str] | bool, default False
        Replace the provided attrs by a ``MissingAttribute`` object. If False, Fill not
        try to replace attributes. If True, will replace every missing attribute. You can use
        * as a wildcard to be replaced by any attribute in a module.

    Returns
    -------
    CkptType

    """
    import torch

    pickle_module = get_unpickler(replace_attrs)
    ckpt = torch.load(path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    if "pytorch-lightning_version" not in ckpt:
        raise ValueError(
            "You can only migrate training checkpoint. If you need a migrated inference checkpoint, fisrt "
            "migrate the training checkpoint, then regenerate the inference one with `anemoi-training checkpoint inference`."
        )
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

    def _get_group(self, ckpt: CkptType) -> int:
        """Get the compatibility group of the checkpoint. Note that if the compatibility
        group is not the latest group, then the checkpoint cannot be migrated.

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to get the group

        Returns
        -------
        int
            Index of the compatibility group
        """
        if _ckpt_migration_key not in ckpt:
            raise ValueError("Checkpoint is not compatible")

        if not len(ckpt[_ckpt_migration_key]):
            return 0
        first_migration = ckpt[_ckpt_migration_key][0]["name"]
        for k, group in enumerate(self._grouped_migrations[1:], 1):
            if group[0].name == first_migration:
                return k
        raise ValueError("Checkpoint is not compatible")

    def get_first_incompatible_version(self, ckpt: CkptType) -> str | None:
        """Get the first version where you cannot update the checkpoint

        Parameters
        ----------
        ckpt : CkptType
            the checkpoint to check

        Returns
        -------
        str | None
            If None, no incompatibility (you can update to any version). Otherwise,
            the first anemoi-models version where your checkpoint would not be compatible.
        """
        group = self._get_group(ckpt)
        if group == len(self._grouped_migrations) - 1:
            return None
        return self._grouped_migrations[group + 1][0].metadata.versions["anemoi-models"]

    def _resolve_operations(
        self, ckpt: CkptType, migrations: list[Migration]
    ) -> tuple[list[Callable[[MigrationContext], None]], list[BaseOp]]:
        """Resolves the list of operations to execute to migrate the checkpoint.
        If it contains migrations and rollbacks, first rollbacked are applied (starting
        from the end), then migrations are applied (starting from the beginning).

        The migrations in the checkpoint are compared with the ones in the ``migrations`` argument.

        For example for the migrations...
        in ``migrations``  | in the checkpoint
        A                  | A
        C                  | E (extra)
        D (extra)          |

        First backward with the checkpoint as reference:
        * we need to rollback E
        * A is ok because it's already synchronized
        Then forward with ``migrations`` as reference:
        * A is already sync
        * then apply C and D.

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint
        migrations : list[Migration]
            The reference migration list

        Returns
        -------
        tuple[list[Callable[[MigrationContext], None]], list[BaseOp]]
            The resolved operation (in order)
            * the list of setup callbacks to execute
            * the list of operations (migrate or rollback) to execute.
        """
        ckpt_migrations = self.registered_migrations(ckpt)
        setups: list[Callable[[MigrationContext], None]] = []
        ops: list[BaseOp] = []
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
            if ckpt_migration.rollback_setup is not None:
                setups.append(ckpt_migration.rollback_setup)
            ops.append(RollbackOp(ckpt_migration.rollback, ckpt_migration))

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
            if migration.migrate_setup is not None:
                setups.append(migration.migrate_setup)
            ops.append(MigrationOp(migration.migrate, migration))
        return setups, ops

    def _resolve_context(self, context: MigrationContext) -> None:
        """Resolves the final context object after all setup callbacks have been executed.

        It first tries to move all modules, then moves all attributes.

        Parameters
        ----------
        context : MigrationContext
            The context object
        """
        for module_path_end, module_path_start in context.module_paths.items():
            LOGGER.debug("Move module %s to %s.", module_path_start, module_path_end)
            sys.modules[module_path_start] = sys.modules[module_path_end]
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

    def sync(self, path: str | PathLike) -> tuple[CkptType, CkptType, list[BaseOp]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        path : str | PathLike
            The checkpoint to migrate.

        Returns
        -------
        tuple[CkptType, list[BaseOp]]
            * The original checkpoint (might have obfuscated attributes with `MissingAttribute`
                if it cannot be imported
            * The migrated checkpoint
            * The list of migrations or rollbacks
        """
        # First load the checkpoint and obfuscate any import issue, just to get the
        # migrations from the checkpoint. The real checkpoint is reloaded afterwards.
        old_ckpt = _load_ckpt(path, replace_attrs=True)
        ckpt = deepcopy(old_ckpt)

        if not self.is_compatible_ckpt(ckpt):
            first_incompatible_version = self.get_first_incompatible_version(ckpt)
            raise IncompatibleCheckpointException(
                "No compatible migration available: the checkpoint is too old. "
                f"Use a version of anemoi-models < {first_incompatible_version}."
            )
        compatible_migrations = self._grouped_migrations[-1]
        setups, ops = self._resolve_operations(ckpt, compatible_migrations)
        replace_attrs: list[str] = []
        if len(setups):
            context = MigrationContext()
            for setup in setups:
                setup(context)
            self._resolve_context(context)
            replace_attrs = context.deleted_attributes
        # Force reloading checkpoint without obfuscating import issues.
        ckpt = _load_ckpt(path, replace_attrs)
        for op in ops:
            if isinstance(op, RollbackOp):
                ckpt = op.run(ckpt)
                ckpt[_ckpt_migration_key].pop()
            else:
                ckpt = op.run(ckpt)
                ckpt[_ckpt_migration_key].append(op.migration.serialize())
        return old_ckpt, ckpt, ops

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
        ckpt = _load_ckpt(path, replace_attrs=True)
        if not self.is_compatible_ckpt(ckpt):
            first_incompatible_version = self.get_first_incompatible_version(ckpt)
            raise IncompatibleCheckpointException(
                "No compatible migration available: the checkpoint is too old. "
                f"Use a version of anemoi-models < {first_incompatible_version}."
            )
        compatible_migrations = self._grouped_migrations[-1]
        registered_migrations = self.registered_migrations(ckpt)
        _, ops = self._resolve_operations(ckpt, compatible_migrations)
        missing_migrations: list[Migration] = []
        extra_migrations: list[Migration] = []
        for op in ops:
            if isinstance(op, RollbackOp):
                extra_migrations.append(op.migration)
                registered_migrations.pop()
            elif isinstance(op, MigrationOp):
                missing_migrations.append(op.migration)
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


class SaveCkpt:
    """Useful for testing. Used in the save_ckpt fixture."""

    def __init__(self, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir

    def __call__(self, ckpt: CkptType, migrations: list[dict[str, Any]], name: str = "model.ckpt") -> Path:
        import torch

        ckpt_migrations: list[SerializedMigration] = []
        for migration in migrations:
            ckpt_migrations.append(
                {
                    "name": migration.get("name", "dummy_name"),
                    "rollback": migration.get("rollback", None),
                    "rollback_setup": migration.get("rollback_setup", None),
                    "metadata": migration.get(
                        "metadata", {"versions": {"migration": "1.0.0", "anemoi-models": "x.x.x"}}
                    ),
                }
            )
        ckpt["migrations"] = ckpt_migrations
        path = self.ckpt_dir / name
        torch.save(ckpt, path)
        return path


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
    "MigrationOp",
    "RollbackOp",
    "SaveCkpt",
    "SerializedMigration",
]
