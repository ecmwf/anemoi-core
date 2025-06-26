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
LOGGER.setLevel("DEBUG")


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


def _get_steps_from_target(
    migrations: Sequence[Migration], done_migrations: Sequence[Mapping[str, Any]], target_version: str
) -> int:
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
        for migration in migrations:
            if migration.name in done_migration_names:
                continue
            if semver.Version.parse(migration.versions["anemoi-models"]) > target_version:
                return steps
            steps += 1
    return steps


def get_missing_migrations(
    ckpt: CkptType, migrations: Sequence[Migration], raise_missing_migrations: bool = True
) -> Sequence[Migration]:
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
        return migrations
    done_migrations = ckpt[_ckpt_migration_key]
    done_migration_names = [migration["name"] for migration in done_migrations]
    # Migration should be done in order, we look for the the last done migration and
    # execute the rest. This is to allow havind removed migrations in a checkpoint and
    # not complain.
    key_rest_migration = 0
    for k, mig in reversed(list(enumerate(migrations))):
        if mig.name in done_migration_names:
            key_rest_migration = k + 1

    if raise_missing_migrations:
        for migration in migrations[:key_rest_migration]:
            if migration.name not in done_migration_names:
                raise MissingMigrationException(
                    f"{migration.name} is not part of the checkpoint but cannot be executed (out of order)."
                )
    return migrations[key_rest_migration:]


def migrate_ckpt(
    ckpt: CkptType, migrations: Sequence[Migration], steps: Optional[int] = None, target: Optional[str] = None
) -> Tuple[CkptType, Sequence[Migration], List[Migration]]:
    """Migrate checkpoint using provided migrations

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint to migrate.
    migrations : Sequence[Migration]
        The list of migrations to perform.
    steps : Optional[int], default None
        Number of migration step to execute. If negative, will rollback the provided number of steps.
        Defaults to all missing migrations.
    target : Optional[str], default None
        Target version of anemoi-models. Will migrate or rollback accordingly.
        Defaults to latest.

    Returns
    -------
    Tuple[CkptType, Sequence[Migration], List[Migration]]
        * The migrated checkpoint
        * The list of migrations that were applied to the checkpoint
        * The list of rollbacks that were applied
    """
    ckpt = deepcopy(ckpt)
    if steps is not None and target is not None:
        raise ValueError("Cannot have both steps and target")

    if _ckpt_migration_key not in ckpt:
        ckpt[_ckpt_migration_key] = []
    done_migrations = ckpt[_ckpt_migration_key]
    if target is not None:
        steps = _get_steps_from_target(migrations, done_migrations, target)
    if steps is not None and steps <= 0:
        migration_map = {migration.name: migration for migration in migrations}
        rollbacks: List[Migration] = []
        for _ in range(-steps):
            last_migration = ckpt[_ckpt_migration_key].pop()
            rollbacks = [last_migration] + rollbacks
            ckpt = migration_map[last_migration["name"]].rollback(ckpt)
        return ckpt, [], rollbacks
    missing_migrations = get_missing_migrations(ckpt, migrations)
    if steps is not None:
        assert steps > 0
        missing_migrations = missing_migrations[:steps]
    for migration in missing_migrations:
        ckpt = migration.migrate(ckpt)
        ckpt[_ckpt_migration_key].append(migration.serialize())
    return ckpt, missing_migrations, []


def migrations_from_path(location: Union[str, PathLike], package: str) -> Tuple[List[Migration], List[str]]:
    """Load the migrations from a given folder

    Parameters
    ----------
    location : Union[str, PathLike]
        Path to the migration folder
    package : str
        Reference package for the import of the migrations

    Returns
    -------
    Tuple[List[Migration], List[str]]
        * The list of migrations to execute from this module
        * The list of migrations that failed to load
    """
    migrations: List[Migration] = []
    failed_migration_imports: List[str] = []

    for file in sorted(Path(location).iterdir()):
        if not file.is_file() and file.suffix != ".py" or file.name == "__init__.py":
            continue
        LOGGER.debug("Loading migration .%s from %s", file.stem, package)
        try:
            migration = importlib.import_module(f".{file.stem}", package)
        except ImportError as e:
            LOGGER.debug("Error loading %s: %s", file.name, str(e))
            failed_migration_imports.append(file.name)
            continue

        migrations.append(
            Migration(
                name=file.stem,
                migrate=migration.migrate,
                rollback=migration.rollback,
                versions=migration.versions,
            )
        )
    return migrations, failed_migration_imports


def load_migrations() -> Tuple[List[Migration], List[str]]:
    """Load the migrations from this folder

    Returns
    -------
    Tuple[List[Migration], List[str]]
        * The list of migrations to execute from this module
        * The list of migrations that failed to load
    """
    return migrations_from_path(MIGRATION_PATH, __name__)


def register_migrations_to_ckpt(ckpt: CkptType, migrations: Sequence[Migration]) -> CkptType:
    """Registers a list of migration to the checkpoint.
    Note: this does not execute any migration. It only registers them in the migration
    key of the checkpoint.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint
    migrations : Sequence[Migration]
        Sequence of migrations to add to the checkpoint migration key

    Returns
    -------
    CkptType
        Checkpoint with registered migrations
    """
    if _ckpt_migration_key not in ckpt:
        ckpt[_ckpt_migration_key] = []
    for migration in migrations:
        ckpt[_ckpt_migration_key].append(migration.serialize())
    return ckpt


__all__ = [
    "MIGRATION_PATH",
    "CkptType",
    "Migration",
    "get_missing_migrations",
    "migrate_ckpt",
    "migrations_from_path",
    "load_migrations",
    "register_migrations_to_ckpt",
]
