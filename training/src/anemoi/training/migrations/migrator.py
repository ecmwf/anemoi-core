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
from collections.abc import Sequence
from copy import deepcopy
from functools import cached_property
from inspect import getsource
from pathlib import Path
from types import ModuleType
from typing import Self
from typing import TypedDict

from anemoi.training import __version__
from anemoi.training.migrations.config import Config
from anemoi.utils.migrations import IncompleteMigrationScript
from anemoi.utils.migrations import Migration
from anemoi.utils.migrations import MigrationMetadata
from anemoi.utils.migrations import Migrator

MIGRATION_PATH = Path(__file__).parent / "scripts"

_CONFIG_MIGRATION_KEY = "migration_state"

LOGGER = logging.getLogger(__name__)


class IncompatibleConfigException(BaseException):
    """The provided config cannot be migrated because it is to old/recent."""


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-training": str})


def _get_code_digest(content: str) -> str:
    """Get a digest for some python code. This does not take indentations, comments
    (except docstrings) and is based on the code's ast.

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
    """Represents a config migration"""

    @classmethod
    def from_migration(cls, name: str, migration: ModuleType) -> Self:
        if not hasattr(migration, "metadata") or not isinstance(migration.metadata, MigrationMetadata):
            raise IncompleteMigrationScript("Migration script is missing metadata.")

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
        self, migrations: Sequence[ConfigMigration] | None = None, obj_migration_key: str | None = None
    ) -> None:
        """Create the migrator object

        Parameters
        ----------
        migrations : Sequence[ConfigMigration] | None, default None
            List of migration to execute. If None, get migrations from the current folder.
        """

        if migrations is None:
            # remove the ".migrator" at the end to get parent folder as migration package
            migration_pkg, _, _ = __name__.rpartition(".")
            migrations = self._migrations_from_path(ConfigMigration, MIGRATION_PATH, f"{migration_pkg}.scripts")

        self._migration_hash_to_name = {migration.name_hash: migration.name for migration in migrations}
        super().__init__(migrations, obj_migration_key or _CONFIG_MIGRATION_KEY)

    def _current_migration_name(self, obj: Config) -> str | None:
        if self._obj_migration_key not in obj:
            raise IncompatibleConfigException("config is not compatible")
        migration_state = obj[self._obj_migration_key].value
        if migration_state is not None and not isinstance(migration_state, str):
            raise TypeError("The migration state should be None or str.")

        if migration_state is None:
            return

        if migration_state not in self._migration_hash_to_name:
            raise IncompatibleConfigException("The config's migration state is not a valid migration.")

        return self._migration_hash_to_name[migration_state]

    def _current_group(self, obj: Config) -> int:
        """Get the compatibility group of the config. Note that if the compatibility
        group is not the latest group, then the config cannot be migrated.

        Parameters
        ----------
        config : Config
            The config to get the group

        Returns
        -------
        int
            Index of the compatibility group
        """
        current_migration_name = self._current_migration_name(obj)
        if current_migration_name is None:
            return 0

        if current_migration_name not in self._migration_groups:
            raise IncompatibleConfigException("config is not compatible")
        return self._migration_groups[current_migration_name]

    def is_compatible(self, obj: Config) -> bool:
        """Checks whether the object is compatible with the current version.

        Parameters
        ----------
        obj : _T
            The object

        Returns
        -------
        bool
            Whether it is compatible
        """
        # No migration means checkpoint too old, no migrations available.
        if self._obj_migration_key not in obj:
            return False
        # If empty, means first group
        if obj[self._obj_migration_key].value is None:
            return True

        obj_compat_group = self._current_group(obj)
        return obj_compat_group == len(self._compatibility_groups) - 1

    def registered_migrations(self, obj: Config) -> list[ConfigMigration]:
        compat_group = self._compatibility_groups[self._current_group(obj)]
        current_migration_name = self._current_migration_name(obj)
        if current_migration_name is None:
            return []
        migrations: list[ConfigMigration] = []
        for migration in compat_group:
            migrations.append(migration)
            if migration.name == current_migration_name:
                break
        return migrations

    def missing_migrations(self, obj: Config) -> list[ConfigMigration]:
        compat_group = self._compatibility_groups[self._current_group(obj)]
        current_migration_name = self._current_migration_name(obj)
        if current_migration_name is None:
            return compat_group
        if current_migration_name not in self._migration_refs:
            raise IncompatibleConfigException(
                f"config cannot be migrated. Extra migrations are registered. ({current_migration_name})"
            )
        last_registered_migration = self._migration_refs[current_migration_name]
        return compat_group[last_registered_migration + 1 :]

    def get_first_incompatible_version(self, config: Config) -> str | None:
        """Get the first version where you cannot update the config

        Parameters
        ----------
        config : Config
            the config to check

        Returns
        -------
        str | None
            If None, no incompatibility (you can update to any version). Otherwise,
            the first anemoi-models version where your config would not be compatible.
        """
        group = self._current_group(config)
        if group == len(self._compatibility_groups) - 1:
            return None
        return self._compatibility_groups[group + 1][0].metadata.versions["anemoi-training"]

    def sync(self, path: str | Path) -> tuple[Config, Config, list[ConfigMigration]]:
        """Migrate or rollbacks the config using provided migrations

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
        old_config = Config.from_path(path)
        config = deepcopy(old_config)

        if not self.is_compatible(config):
            first_incompatible_version = self.get_first_incompatible_version(config)
            raise IncompatibleConfigException(
                "No compatible migration available: the config is too old. "
                f"Use a version of anemoi-training < {first_incompatible_version}."
            )
        missing_migrations = self.missing_migrations(config)
        for migration in missing_migrations:
            if migration.migrate is None:
                raise IncompatibleConfigException(
                    f"Migration {migration.name} cannot be executed. Missing migrate function."
                )
            config = migration.migrate(config)
            config[_CONFIG_MIGRATION_KEY] = migration.name_hash
        return old_config, config, missing_migrations

    def inspect(self, path: str | Path) -> tuple[list[Migration], list[Migration]]:
        """Inspect migration information in config

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
            first_incompatible_version = self.get_first_incompatible_version(config)
            raise IncompatibleConfigException(
                "No compatible migration available: the config is too old. "
                f"Use a version of anemoi-training < {first_incompatible_version}."
            )
        return list(self.registered_migrations(config)), list(self.missing_migrations(config))
