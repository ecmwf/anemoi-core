# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import logging
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypedDict
from typing import Union

import semver

MIGRATION_PATH = Path(__file__).parent

_ckpt_migration_key = "migrations"

LOGGER = logging.getLogger(__name__)


class MissingMigrationException(BaseException):
    """The checkpoint is missing a migration that cannot be added (wrong order)."""

    pass


class FinalMigrationException(BaseException):
    """This migration is final and older checkpoints cannot be migrated past this."""

    pass


CkptType = MutableMapping[str, Any]


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-models": str})


@dataclass
class MigrationMetadata:
    versions: MigrationVersions
    final: bool = False


@dataclass
class Migration:
    """Represents a migration"""

    name: str
    """Name of the migration"""
    migrate: Callable[[CkptType], CkptType]
    """Callback to execute the migration"""
    rollback: Callable[[CkptType], CkptType]
    """Callback to execute a migration rollback"""
    metadata: MigrationMetadata
    """Tracked metadata"""

    def serialize(self) -> str:
        return self.name


def registered_migrations(ckpt: CkptType) -> List[Dict[str, Any]]:
    """Return all registered migrations from a checkpoint.
    Parameters
    ----------
    ckpt : CkptType
        The checkpoint

    Returns
    -------
    List[Dict[str, Any]]
        The registered migrations
    """
    if _ckpt_migration_key not in ckpt:
        return []
    return ckpt[_ckpt_migration_key]


def _migrations_from_path(location: Union[str, PathLike], package: str) -> List[Migration]:
    """Returns the migrations from a given folder

    Parameters
    ----------
    location : Union[str, PathLike]
        Path to the migration folder
    package : str
        Reference package for the import of the migrations

    Returns
    -------
    List[Migration]
        The migrations from the given path
    """
    migrations: List[Migration] = []

    for file in sorted(Path(location).iterdir()):
        if not file.is_file() and file.suffix != ".py" or file.name == "__init__.py":
            continue
        LOGGER.debug("Loading migration .%s from %s", file.stem, package)
        try:
            migration = importlib.import_module(f".{file.stem}", package)
        except ImportError as e:
            LOGGER.warning("Error loading %s: %s", file.name, str(e))
            continue

        migrations.append(
            Migration(
                name=file.stem, migrate=migration.migrate, rollback=migration.rollback, metadata=migration.metadata
            )
        )
    return migrations


class Migrator:
    def __init__(self, migrations: Optional[Sequence[Migration]] = None, raise_missing_migrations: bool = True) -> None:
        """Create the migrator object

        Parameters
        ----------
        migrations : Optional[Sequence[Migration]], default None
            List of migration to execute. If None, get migrations from the current folder.
        raise_missing_migrations : bool
            Whether to check if there are out of order migrations missing from the checkpoint
        """

        if migrations is None:
            self._migrations = _migrations_from_path(MIGRATION_PATH, __name__)
        else:
            self._migrations = migrations
        self._raise_missing_migrations = raise_missing_migrations
        self._migration_map = {migration.name: migration for migration in self._migrations}

    @classmethod
    def from_path(cls, location: Union[str, PathLike], package: str) -> "Migrator":
        """Load from a given folder

        Parameters
        ----------
        location : Union[str, PathLike]
            Path to the migration folder
        package : str
            Reference package for the import of the migrations

        Returns
        -------
        A Migrator instance
        """
        return cls(_migrations_from_path(location, package))

    def _get_first_missing_migration(self, ckpt: CkptType) -> int:
        """Get first missing migrations from a checkpoint

        Parameters
        ----------
        ckpt : CkptType
            The loaded checkpoint
        migrations : Sequence[Migration]
            List of migration to execute
        raise_missing_migrations : bool
            Whether to check if there are out of order migrations missing from the checkpoint

        Returns
        -------
        int
            Index of first missing migrations from the checkpoint to execute
        """
        if _ckpt_migration_key not in ckpt:
            return 0
        done_migrations = ckpt[_ckpt_migration_key]
        # Migration should be done in order, we look for the the last done migration and
        # execute the rest. This is to allow having removed migrations in a checkpoint and
        # not complain.
        key_rest_migration = 0
        num_migrations = len(self._migrations)
        for k, mig in enumerate(reversed(self._migrations)):
            if mig.name in done_migrations:
                key_rest_migration = num_migrations - k
                break

        if self._raise_missing_migrations:
            for migration in self._migrations[:key_rest_migration]:
                if migration.name not in done_migrations:
                    raise MissingMigrationException(
                        f"{migration.name} is not part of the checkpoint but cannot be executed (out of order)."
                    )
        return key_rest_migration

    def _get_steps_from_target(self, done_migrations: Sequence[str], target_version: str) -> int:
        """Returns the number of migration steps to execute given a target version

        Parameters
        ----------
        migrations : Sequence[Migration]
            All possible migrations
        done_migrations : Sequence[str]
            Names of already done migrations in the checkpoint
        target_version : str
            The target version

        Returns
        -------
        int
            The number of steps
        """
        if not len(done_migrations):
            current_version = "0.0.0"
        else:
            current_version = self._migration_map[done_migrations[-1]].metadata.versions["anemoi-models"]
        current = semver.Version.parse(current_version)
        target = semver.Version.parse(target_version)
        steps = 0
        if current > target:
            # rollback
            for migration_name in reversed(done_migrations):
                migration = self._migration_map.get(migration_name, None)
                if migration is None:
                    raise MissingMigrationException(f"{migration_name} does not exist anymore.")
                if semver.Version.parse(migration.metadata.versions["anemoi-models"]) <= target_version:
                    return steps
                steps -= 1
        else:
            # migrate
            for migration in self._migrations:
                if migration.name in done_migrations:
                    continue
                if semver.Version.parse(migration.metadata.versions["anemoi-models"]) > target_version:
                    return steps
                steps += 1
        return steps

    def sync(
        self,
        ckpt: CkptType,
        steps: Optional[int] = None,
        n_migrations: Optional[int] = None,
        target: Optional[str] = None,
    ) -> Tuple[CkptType, Sequence[Migration], List[Migration]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to migrate.
        steps : Optional[int], default None
            Number of relative migration step to execute. If negative, will rollback the provided number of steps.
            Mutually exclusive with steps and target.
            Defaults to migrate all missing migrations.
        n_migrations : Optional [int]
            Number migrations to be executed. Cannot be negative. Will migrate or rollback
            to have exactly n_migrations migrations executed.
            Mutually exclusive with rel_steps and target.
            Defaults to migrate all migrations.
        target : Optional[str], default None
            Target version of anemoi-models. Will migrate or rollback accordingly.
            Mutually exclusive with steps and n_migrations.
            Defaults to latest version.

        Returns
        -------
        Tuple[CkptType, Sequence[Migration], List[Migration]]
            * The migrated checkpoint
            * The list of migrations that were applied to the checkpoint
            * The list of rollbacks that were applied
        """
        if steps is not None and target is not None:
            raise ValueError("Cannot have both rel_steps and target")
        if steps is not None and n_migrations is not None:
            raise ValueError("Cannot have both rel_steps and steps")
        if n_migrations is not None and target is not None:
            raise ValueError("Cannot have both steps and target")

        ckpt = deepcopy(ckpt)

        if _ckpt_migration_key not in ckpt:
            ckpt[_ckpt_migration_key] = []
        done_migrations = ckpt[_ckpt_migration_key]
        # we convert steps and target to rel_steps and
        # do computation with rel_step.
        if n_migrations is not None:
            steps = n_migrations - len(done_migrations)
        if target is not None:
            steps = self._get_steps_from_target(done_migrations, target)
        if steps is not None and steps <= 0:
            ckpt, rollbacks = self.rollback(ckpt, -steps, copy_ckpt=False)
            return ckpt, [], rollbacks
        ckpt, missing_migrations = self.migrate(ckpt, steps, copy_ckpt=False)
        return ckpt, missing_migrations, []

    def _next_final_migration(self, start_migration: int) -> Optional[int]:
        """Returns the next final migration from the given start_migration"""
        for k, migration in enumerate(self._migrations[start_migration:], start_migration):
            if migration.metadata.final:
                return k

    def migrate(
        self, ckpt: CkptType, steps: Optional[int] = None, *, copy_ckpt: bool = True
    ) -> Tuple[CkptType, Sequence[Migration]]:
        """Rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to migrate.
        steps : Optional[int], default None
            Number of migration step to execute. Defaults to all missing migrations.
        copy_ckpt : bool, default True
            Whether to deepcopy the checkpoint before applying migrations before applying migrations.
            If False, the input ckpt cannot be used after migration.
        Returns
        -------
        Tuple[CkptType, Sequence[Migration], List[Migration]]
            * The migrated checkpoint
            * The list of migrations that were applied to the checkpoint
        """
        if steps is not None and steps < 0:
            raise ValueError("Cannot migrate negative number of steps. Use sync instead")
        if copy_ckpt:
            ckpt = deepcopy(ckpt)

        if _ckpt_migration_key not in ckpt:
            ckpt[_ckpt_migration_key] = []
        key_missing_migration = self._get_first_missing_migration(ckpt)
        next_final = self._next_final_migration(key_missing_migration)
        if next_final is not None and steps is not None and next_final < steps:
            raise FinalMigrationException("The checkpoint is too old. Migration is not supported.")
        if steps is None and next_final is not None:
            steps = next_final - key_missing_migration
        missing_migrations = self._migrations[key_missing_migration:]
        if steps is not None:
            missing_migrations = missing_migrations[:steps]
        for migration in missing_migrations:
            ckpt = migration.migrate(ckpt)
            ckpt[_ckpt_migration_key].append(migration.serialize())
        return ckpt, missing_migrations

    def _prev_final_migration(self, migrations: List[str]) -> int:
        """Returns the previously stored final migration."""
        for k, migration in enumerate(migrations):
            if self._migration_map[migration].metadata.final:
                return k
        return len(migrations)

    def rollback(
        self, ckpt: CkptType, steps: Optional[int] = None, *, copy_ckpt: bool = True
    ) -> Tuple[CkptType, List[Migration]]:
        """Rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint to rollback.
        steps : Optional[int], default None
            Number of rollback steps to execute. Defaults to rollback everything
        copy_ckpt : bool, default True
            Whether to deepcopy the checkpoint before applying migrations before applying migrations.
            If False, the input ckpt cannot be used after migration.

        Returns
        -------
        Tuple[CkptType, Sequence[Migration], List[Migration]]
            * The migrated checkpoint
            * The list of rollbacks that were applied
        """
        if steps is not None and steps < 0:
            raise ValueError("Cannot rollback negative number of steps. Use sync instead")
        if copy_ckpt:
            ckpt = deepcopy(ckpt)

        if _ckpt_migration_key not in ckpt:
            ckpt[_ckpt_migration_key] = []
        rollbacks: List[Migration] = []
        prev_final = self._prev_final_migration(ckpt[_ckpt_migration_key])
        if steps is not None and steps > prev_final:
            raise FinalMigrationException("The checkpoint is too recent. Migration is not supported.")
        if steps is None:
            steps = len(ckpt[_ckpt_migration_key]) - prev_final - 1
        for _ in range(steps):
            migration_name = ckpt[_ckpt_migration_key].pop()
            last_migration = self._migration_map.get(migration_name, None)
            if last_migration is None:
                raise MissingMigrationException(f"Migration {migration_name} does not exist anymore. Cannot rollback.")
            rollbacks = [last_migration] + rollbacks
            ckpt = last_migration.rollback(ckpt)
        return ckpt, rollbacks

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
        for migration in self._migrations:
            ckpt[_ckpt_migration_key].append(migration.serialize())
        return ckpt


__all__ = ["MIGRATION_PATH", "CkptType", "Migration", "Migrator", "registered_migrations"]
