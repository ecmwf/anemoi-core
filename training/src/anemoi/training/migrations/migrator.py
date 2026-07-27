# (C) Copyright 2025 Anemoi contributors.
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
import importlib
import logging
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import getsource
from os import PathLike
from pathlib import Path
from typing import Any
from typing import TypedDict

from omegaconf import DictConfig
from omegaconf import ListConfig
from ruamel.yaml import YAML

from anemoi.training import __version__

MIGRATION_PATH = Path(__file__).parent / "scripts"

LOGGER = logging.getLogger(__name__)

_version_key = "version"


class IncompleteMigrationScript(BaseException):
    """The migration script is missing some mandatory content (metadata)."""


class IncompatibleCheckpointException(BaseException):
    """The provided checkpoint cannot be migrated because it is to old/recent."""


CfgType = dict[str, Any]


def select(cfg: Any, keys: str, create: bool = False) -> Any:
    if keys == "":
        return cfg
    parts = keys.split(".")
    for part in parts:
        if isinstance(cfg, dict) and part in cfg:
            cfg = cfg[part]
        elif isinstance(cfg, dict) and create:
            cfg[part] = {}
            cfg = cfg[part]
        elif isinstance(cfg, list) and part.isdigit() and len(cfg) > int(part):
            cfg = cfg[int(part)]
        elif isinstance(cfg, list) and part.isdigit() and len(cfg) == int(part) and create:
            cfg.append({})
            cfg = cfg[-1]
        else:
            raise ValueError(f"{keys} not in config.")
    return cfg


def update(config: Any, keys: str, val: Any) -> None:
    parts = keys.split(".")
    parent = select(config, ".".join(parts[:-1]), create=True)
    key = parts[-1]
    if isinstance(parent, dict):
        parent[key] = val
    elif isinstance(parent, list) and len(parent) > int(key):
        parent[int(key)] = val
    elif isinstance(parent, list) and len(parent) == int(key):
        parent.append(val)
    else:
        raise ValueError(f"{keys} not in config.")


def delete(config: Any, keys: str) -> None:
    parts = keys.split(".")
    parent = select(config, ".".join(parts[:-1]))
    key = parts[-1]
    if isinstance(parent, dict):
        del parent[key]
    elif isinstance(parent, list) and key.isdigit():
        parent.pop(int(key))
    else:
        raise ValueError(f"{keys} not in config.")


class Op:
    pass


@dataclass
class AddOp(Op):
    target: str
    value: Any


@dataclass
class RemoveOp(Op):
    source: list[str]


@dataclass
class MoveOp(Op):
    source: str
    target: str


@dataclass
class TransformOp(Op):
    source: list[str]
    func: Callable[..., Any]


def _nest_in_list(_: DictConfig | ListConfig, val: Any) -> Any:
    return [val]


class MigrationManifest:
    def __init__(self):
        self._ops: list[Op] = []

    def add(self, target: str, value: Any = "???") -> None:
        self._ops.append(AddOp(target, value))

    def remove(self, source: str | list[str]) -> None:
        if isinstance(source, str):
            source = [source]
        self._ops.append(RemoveOp(source))

    def move(self, source: str, target: str) -> None:
        self._ops.append(MoveOp(source, target))

    def transform(self, source: str | list[str], func: Callable[..., Any]) -> None:
        if isinstance(source, str):
            source = [source]
        self._ops.append(TransformOp(source, func))

    def nest_in_list(self, source: str | list[str]) -> None:
        return self.transform(source, _nest_in_list)

    def execute(self, cfg: Any):
        for op in self._ops:
            self._exec_step(cfg, op)
        return cfg

    def _exec_step(self, cfg: Any, op: Op):
        if isinstance(op, AddOp):
            update(cfg, op.target, op.value)
            return
        if isinstance(op, RemoveOp):
            for source in op.source:
                delete(cfg, source)
        elif isinstance(op, MoveOp):
            val = select(cfg, op.source)
            self._exec_step(cfg, RemoveOp([op.source]))
            self._exec_step(cfg, AddOp(op.target, val))
        elif isinstance(op, TransformOp):
            for source in op.source:
                new_val = op.func(cfg, select(cfg, source))
                update(cfg, source, new_val)


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-training": str})


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
    signature: str
    """ The signature of the script. Can be used to detect if a script changed. """


@dataclass
class Migration:
    """Represents a migration"""

    name: str
    """Name of the migration"""
    metadata: MigrationMetadata
    """Tracked metadata"""
    signature: str
    """Signature of the migration. Can be used to detect if the script changed"""
    migrate: Callable[[MigrationManifest], None]
    """Callback to execute the migration"""


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
        migration = importlib.import_module(f".{file.stem}", package)
        if not hasattr(migration, "metadata"):
            raise IncompleteMigrationScript("Migration script is missing metadata.")
        if not hasattr(migration, "migrate"):
            raise IncompleteMigrationScript("Migration script is missing migrate.")

        args: dict[str, Any] = dict(
            name=file.stem,
            metadata=migration.metadata,
            signature=_get_code_digest(getsource(migration)),
            migrate=migration.migrate,
        )

        if not isinstance(args["metadata"], MigrationMetadata):
            raise IncompleteMigrationScript("Migration script is missing metadata.")

        if args["metadata"].versions["anemoi-training"] == "%NEXT_ANEMOI_TRAINING_VERSION%":
            args["metadata"].versions["anemoi-training"] = __version__

        migrations.append(Migration(**args))
    return migrations


class Migrator:
    def __init__(self, migrations: Sequence[Migration] | None = None) -> None:
        """Create the migrator object

        Parameters
        ----------
        migrations : Sequence[Migration] | None, default None
            List of migration to execute. If None, get migrations from the current folder.
        """

        if migrations is None:
            # remove the ".migrator" at the end to get parent folder as migration package
            migration_pkg, _, _ = __name__.rpartition(".")
            migrations = _migrations_from_path(MIGRATION_PATH, f"{migration_pkg}.scripts")

        self._migrations = migrations

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

    def is_compatible(self, config: CfgType) -> bool:
        """Checks whether the config is compatible with the current version.

        Parameters
        ----------
        config : CfgType
            The config

        Returns
        -------
        bool
            Whether it is compatible
        """

        # No migration means checkpoint too old, no migrations available.
        if _version_key not in config:
            return False
        if config[_version_key] > len(self._migrations):
            raise ValueError(f"version cannot be higher than {len(self._migrations)}.")

        for migration in self._migrations[config[_version_key] :]:
            if migration.metadata.final:
                return False
        return True

    def sync(self, migrated_config: str | Path) -> CfgType:
        """Migrate the checkpoint using provided migrations

        Parameters
        ----------
        config : CfgType
            The config to migrate.

        Returns
        -------
        CkptType
            The migrated checkpoint
        """
        path = Path(migrated_config)
        yaml = YAML()
        config = yaml.load(path.read_text())

        manifest = MigrationManifest()

        if not self.is_compatible(config):
            raise IncompatibleCheckpointException("No compatible migration available")
        version = config[_version_key]
        for migration in self._migrations[version:]:
            migration.migrate(manifest)
            version += 1
        migrated_config = manifest.execute(config)
        migrated_config[_version_key] = version
        return migrated_config
