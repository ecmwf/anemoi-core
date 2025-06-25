import logging
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeAlias
from typing import TypedDict
from typing import Union

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
    """Represent a migration"""

    name: str
    """ Name of the migration """
    upgrade: MigrationCallback
    """ Callback to perform when upgrading the migration """
    downgrade: MigrationCallback
    """ Callback to perform when downgrading the migration """
    versions: Versions
    """ Tracked versions """


def get_missing_migrations(
    ckpt: CkptType,
    migrations: Sequence[Migration],
    raise_missing_migrations: bool = True,
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
    # Migration should be done in order, we look for the the last done migration and
    # execute the rest. This is to allow havind removed migrations in a checkpoint and
    # not complain.
    key_rest_migration = 0
    for k, mig in reversed(list(enumerate(migrations))):
        if mig.name in done_migrations:
            key_rest_migration = k + 1

    if raise_missing_migrations:
        for migration in migrations[:key_rest_migration]:
            if migration.name not in done_migrations:
                raise MissingMigrationException(
                    f"{migration.name} is not part of the checkpoint but cannot be executed (out of order)."
                )
    return migrations[key_rest_migration:]


def _add_migration_to_ckpt(ckpt: CkptType, migration: Migration) -> CkptType:
    """Add migration to ckpt"""
    if _ckpt_migration_key not in ckpt.keys():
        ckpt[_ckpt_migration_key] = []
    ckpt[_ckpt_migration_key].append(migration.name)
    return ckpt


def migrate_ckpt(
    ckpt: CkptType, migrations: Sequence[Migration], steps: Optional[int] = None, target: Optional[str] = None
) -> Tuple[CkptType, Sequence[Migration]]:
    """Migrate checkpoint using provided migrations

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint to migrate.
    migrations : Sequence[Migration]
        The list of migrations to perform.
    steps : Optional[int], default None
        Number of migration step to execute. Defaults to all missing migrations.
    target : Optional[str], default None
        Target version of anemoi-models. Will upgrade or downgrade accordingly.
        Defaults to latest.

    Returns
    -------
    Tuple[CkptType, Sequence[Migration]]
        * The migrated checkpoint
        * The list of migrations that were applied to the checkpoint
    """
    ckpt = deepcopy(ckpt)
    missing_migrations = get_missing_migrations(ckpt, migrations)
    if steps is not None:
        missing_migrations = missing_migrations[:steps]
    for migration in missing_migrations:
        ckpt = migration.upgrade(ckpt)
        ckpt = _add_migration_to_ckpt(ckpt, migration)
    return ckpt, missing_migrations


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
                upgrade=migration.upgrade,
                downgrade=migration.downgrade,
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


__all__ = [
    "MIGRATION_PATH",
    "CkptType",
    "Migration",
    "get_missing_migrations",
    "migrate_ckpt",
    "migrations_from_path",
    "load_migrations",
]
