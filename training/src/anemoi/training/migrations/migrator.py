# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import ast
import hashlib
import logging
from copy import deepcopy
from functools import cached_property
from inspect import getsource
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict

from anemoi.utils.migrations import (
    IncompatibleObjectError,
    IncompleteMigrationScriptError,
    Migration,
    MigrationMetadata,
    Migrator,
)

from anemoi.training import __version__
from anemoi.training.migrations.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

MIGRATION_PATH = Path(__file__).parent / "scripts"

_CONFIG_MIGRATION_KEY = "migration_state"

LOGGER = logging.getLogger(__name__)


class IncompatibleConfigError(IncompatibleObjectError):
    """The provided config cannot be migrated because it is to old/recent."""


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-training": str})


def _get_code_digest(content: str) -> str:
    """Get a digest for some python code.

    This does not take indentations, comments (except docstrings) and is based on
    the code's ast.

    Parameters
    ----------
    content : str
        Some valid python code

    Returns
    -------
    str
        The digest of the code
    """
    code = ast.dump(ast.parse(content), include_attributes=False)
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class ConfigMigration(Migration[Config, Config, MigrationVersions]):
    """Represents a config migration."""

    @classmethod
    def from_migration(cls, name: str, migration: ModuleType) -> Self:
        if not hasattr(migration, "metadata") or not isinstance(migration.metadata, MigrationMetadata):
            msg = "Migration script is missing metadata."
            raise IncompleteMigrationScriptError(msg)

        metadata = migration.metadata
        signature = _get_code_digest(getsource(migration))

        migrate = None

        if hasattr(migration, "migrate"):
            migrate = migration.migrate

        if metadata.versions["anemoi-training"] == "%NEXT_ANEMOI_TRAINING_VERSION%":
            metadata.versions["anemoi-training"] = __version__

        return cls(name, metadata, signature, migrate)

    @cached_property
    def name_hash(self) -> str:
        """Hash of the migration name."""
        return hashlib.sha256(self._name.encode()).hexdigest()[:8]


class ConfigMigrator(Migrator[ConfigMigration, Config]):
    def __init__(
        self,
        migrations: Sequence[ConfigMigration] | None = None,
        obj_migration_key: str | None = None,
        update_summary: bool = False,
    ) -> None:
        """Create the migrator object.

        Parameters
        ----------
        migrations : Sequence[ConfigMigration] | None, default None
            List of migration to execute. If None, get migrations from the current folder.
        obj_migration_key : str | None, default None
            The migration key to use.
        update_summary : bool, defaut False
            Whether to update the top yaml comment to inclued the migration summaries.
        """
        if migrations is None:
            # remove the ".migrator" at the end to get parent folder as migration package
            migration_pkg, _, _ = __name__.rpartition(".")
            migrations = self._migrations_from_path(ConfigMigration, MIGRATION_PATH, f"{migration_pkg}.scripts")

        self._migration_hash_to_name = {migration.name_hash: migration.name for migration in migrations}
        self._update_summary = update_summary
        super().__init__(migrations, obj_migration_key or _CONFIG_MIGRATION_KEY)

    def _migration_state(self, obj: Config) -> list[str] | None:
        """The migration state of the object.

        Parameters
        ----------
        obj : Config
            The config to extract the migration state from.

        Returns
        -------
        list[str] | None
            The migration state. It contains the migration already executed.
            If None, the object doesn't have any migration state and is assumed too
            old to be migratable.

        """
        if self._obj_migration_key not in obj:
            return None
        return obj[self._obj_migration_key].value

    def sync(self, path: str | Path) -> tuple[Config, Config, list[ConfigMigration]]:
        """Migrate or rollbacks the config using provided migrations.

        Parameters
        ----------
        path : str | PathLike
            The config to migrate.

        Returns
        -------
        tuple[Config, Config, list[ConfigMigration]]
            * The original config (might have obfuscated attributes with `MissingAttribute`
                if it cannot be imported
            * The migrated config
            * The list of executed migrations
        """
        old_config = Config.from_path(path, self._update_summary)
        config = deepcopy(old_config)

        if not self.is_compatible(config):
            first_incompatible_migration = self.get_first_incompatible_migration(config)
            assert first_incompatible_migration is not None
            first_incompatible_version = first_incompatible_migration.metadata.versions["anemoi-training"]
            msg = (
                "No compatible migration available: the config is too old. "
                f"Use a version of anemoi-training < {first_incompatible_version}."
            )
            raise IncompatibleConfigError(msg)
        missing_migrations = self.missing_migrations(config)
        for migration in missing_migrations:
            if migration.migrate is None:
                msg = (f"Migration {migration.name} cannot be executed. Missing migrate function.",)
                raise IncompatibleConfigError(msg)
            config.set_migration(migration)
            config = migration.migrate(config)
            migration_state = config[self._obj_migration_key].value
            migration_state.append(migration.name_hash)
            config[self._obj_migration_key] = migration_state
        return old_config, config, missing_migrations

    def inspect(self, path: str | Path) -> tuple[list[Migration], list[Migration]]:
        """Inspect migration information in config.

        Parameters
        ----------
        path : str | PathLike
            Path to the chekpoint to inspect

        Returns
        -------
        tuple[list[Migration], list[Migration]]
            * The list of already executed migrations,
            * the list of missing migrations,
        """
        config = Config.from_path(path)
        if not self.is_compatible(config):
            first_incompatible_migration = self.get_first_incompatible_migration(config)
            assert first_incompatible_migration is not None
            first_incompatible_version = first_incompatible_migration.metadata.versions["anemoi-training"]
            msg = (
                "No compatible migration available: the config is too old. "
                f"Use a version of anemoi-training < {first_incompatible_version}."
            )
            raise IncompatibleConfigError(msg)
        return list(self.registered_migrations(config)), list(self.missing_migrations(config))
