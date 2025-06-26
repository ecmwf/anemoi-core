# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeAlias
from typing import TypedDict
from typing import Union

import semver

from anemoi.utils.cli import importlib

MIGRATION_PATH = Path(__file__).parent

_ckpt_migration_key = "migrations"

LOGGER = logging.getLogger(__name__)


class MissingMigrationException(BaseException):
    pass


CkptType: TypeAlias = MutableMapping[str, Any]
MigrationCallback: TypeAlias = Callable[[CkptType], CkptType]

# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
Versions = TypedDict("Versions", {"migration": str, "anemoi-models": str})


@dataclass
class Migration:
    """Represents a migration"""

    name: str
    """Name of the migration"""
    migrate: MigrationCallback
    """Callback to execute the migration"""
    rollback: MigrationCallback
    """Callback to execute a migration rollback"""
    versions: Versions
    """Tracked versions"""

    def serialize(self) -> Dict[str, Any]:
        return {"name": self.name, "versions": self.versions}


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
                name=file.stem,
                migrate=migration.migrate,
                rollback=migration.rollback,
                versions=migration.versions,
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

    def _get_missing_migrations(self, ckpt: CkptType) -> Sequence[Migration]:
        """Get missing migrations from a checkpoint

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
        Sequence[Migration]
            Missing migrations from the checkpoint to execute
        """
        if _ckpt_migration_key not in ckpt:
            return self._migrations
        done_migrations = ckpt[_ckpt_migration_key]
        done_migration_names = [migration["name"] for migration in done_migrations]
        # Migration should be done in order, we look for the the last done migration and
        # execute the rest. This is to allow havind removed migrations in a checkpoint and
        # not complain.
        key_rest_migration = 0
        for k, mig in reversed(list(enumerate(self._migrations))):
            if mig.name in done_migration_names:
                key_rest_migration = k + 1

        if self._raise_missing_migrations:
            for migration in self._migrations[:key_rest_migration]:
                if migration.name not in done_migration_names:
                    raise MissingMigrationException(
                        f"{migration.name} is not part of the checkpoint but cannot be executed (out of order)."
                    )
        return self._migrations[key_rest_migration:]

    def _get_steps_from_target(self, done_migrations: Sequence[Mapping[str, Any]], target_version: str) -> int:
        """Returns the number of migration steps to execute given a target version

        Parameters
        ----------
        migrations : Sequence[Migration]
            All possible migrations
        done_migrations : Sequence[Mapping[str, Any]]
            Already done migrations
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
            current_version = done_migrations[-1]["versions"]["anemoi-models"]
        current = semver.Version.parse(current_version)
        target = semver.Version.parse(target_version)
        done_migration_names = [migration["name"] for migration in done_migrations]
        steps = 0
        if current > target:
            # rollback
            for migration in reversed(done_migrations):
                if semver.Version.parse(migration["versions"]["anemoi-models"]) <= target_version:
                    return steps
                steps -= 1
        else:
            # migrate
            for migration in self._migrations:
                if migration.name in done_migration_names:
                    continue
                if semver.Version.parse(migration.versions["anemoi-models"]) > target_version:
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
        missing_migrations = self._get_missing_migrations(ckpt)
        if steps is not None:
            missing_migrations = missing_migrations[:steps]
        for migration in missing_migrations:
            ckpt = migration.migrate(ckpt)
            ckpt[_ckpt_migration_key].append(migration.serialize())
        return ckpt, missing_migrations

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
        if steps is None:
            steps = len(ckpt[_ckpt_migration_key])
        for _ in range(steps):
            last_migration = ckpt[_ckpt_migration_key].pop()
            rollbacks = [last_migration] + rollbacks
            ckpt = self._migration_map[last_migration["name"]].rollback(ckpt)
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
