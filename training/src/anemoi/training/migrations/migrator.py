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
import importlib
import io
import logging
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import getsource
from os import PathLike
from pathlib import Path
from typing import Any
from typing import TypedDict

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedBase
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

from anemoi.training import __version__

MIGRATION_PATH = Path(__file__).parent / "scripts"

LOGGER = logging.getLogger(__name__)

_version_key = "version"


class IncompleteMigrationScript(BaseException):
    """The migration script is missing some mandatory content (metadata)."""


class IncompatibleCheckpointException(BaseException):
    """The provided checkpoint cannot be migrated because it is to old/recent."""


CfgType = dict[str, Any]


def set_before_comment(parent: CommentedBase, key: Any, comments: list[str]):
    """Custom function to set comment before a key in the yaml. Reimplemented
    because ruamel functions don't work.

    Parameters
    ----------
    parent : CommentedBase
        The parent object.
    key : Any
        The key to add comment before.
    comment : str
        The comment.
    """
    comment = "".join([f"# {c}\n" for c in comments])
    ca = parent.ca.items.setdefault(key, [None, None, None, None])
    if ca[1] is None:
        ca[1] = []
    ca[1].append(CommentToken(comment, CommentMark(0)))


def set_after_comment(parent: CommentedBase, key: Any, comments: list[str]):
    """Custom function to set comment after a key in the yaml. Reimplemented
    because ruamel functions don't work.

    Parameters
    ----------
    parent : CommentedBase
        The parent object.
    key : Any
        The key to add comment after.
    comment : str
        The comment.
    """
    ca = parent.ca.items.setdefault(key, [None, None, None, None])
    existing_comments = ""
    if ca[2] is not None:
        existing_comments = f"\n{ca[2].value}"

    comment = "".join([f"\n# {c}" for c in comments]) + existing_comments
    ca[2] = CommentToken(f"{comment}\n", CommentMark(0))


def select(cfg: Any, keys: str, create: bool = False, new_comment_before: list[str] | None = None) -> Any:
    if keys == "":
        return cfg
    has_before_comment = False
    parts = keys.split(".")
    parent = cfg
    for part in parts:
        if isinstance(cfg, dict) and part in cfg:
            parent, cfg = cfg, cfg[part]
        elif isinstance(cfg, dict) and create:
            cfg[part] = CommentedMap()
            if new_comment_before is not None and not has_before_comment:
                set_before_comment(cfg, part, new_comment_before)
                has_before_comment = True
            parent, cfg = cfg, cfg[part]
        elif isinstance(cfg, list) and part.isdigit() and len(cfg) > int(part):
            parent, cfg = cfg, cfg[int(part)]
        elif isinstance(cfg, list) and part.isdigit() and len(cfg) == int(part) and create:
            cfg.append(CommentedMap())
            if new_comment_before is not None and not has_before_comment:
                set_before_comment(parent, part, new_comment_before)
                has_before_comment = True
            parent, cfg = cfg, cfg[-1]
        else:
            raise ValueError(f"{keys} not in config.")

    if new_comment_before is not None and not has_before_comment:
        set_before_comment(parent, part, new_comment_before)
    return cfg


def update(
    config: Any,
    keys: str,
    val: Any,
    new_comment_before: list[str] | None = None,
    new_comment_after: list[str] | None = None,
) -> None:
    parts = keys.split(".")
    parent = select(config, ".".join(parts[:-1]), create=True, new_comment_before=new_comment_before)
    key = parts[-1]
    if isinstance(parent, CommentedMap):
        parent[key] = val
        if len(parts) == 1 and new_comment_before is not None:
            set_before_comment(parent, key, new_comment_before)
        if new_comment_after is not None:
            set_after_comment(parent, key, new_comment_after)
    elif isinstance(parent, list) and len(parent) > int(key):
        parent[int(key)] = val
        if len(parts) == 1 and new_comment_before is not None:
            set_before_comment(parent, int(key), new_comment_before)
        if new_comment_after is not None:
            set_after_comment(parent, int(key), new_comment_after)
    elif isinstance(parent, list) and len(parent) == int(key):
        parent.append(val)
        if len(parts) == 1 and new_comment_before is not None:
            set_before_comment(parent, int(key), new_comment_before)
        if new_comment_after is not None:
            set_after_comment(parent, int(key), new_comment_after)
    else:
        raise ValueError(f"{keys} not in config.")


def delete(config: Any, keys: str, new_comment: list[str] | None = None) -> None:
    parts = keys.split(".")
    parent = select(config, ".".join(parts[:-1]), new_comment_before=new_comment)
    key = parts[-1]
    if isinstance(parent, dict):
        del parent[key]
    elif isinstance(parent, list) and key.isdigit():
        parent.pop(int(key))
    else:
        raise ValueError(f"{keys} not in config.")


def parent_key(keys: str) -> tuple[str, str]:
    parts = keys.split(".")
    return ".".join(parts[:-1]), parts[-1]


class Op:
    comment: str | None


@dataclass
class AddOp(Op):
    target: str
    value: Any
    comment: list[str] | None


@dataclass
class RemoveOp(Op):
    source: list[str]
    comment: list[str] | None


@dataclass
class MoveOp(Op):
    source: str
    target: str
    comment: list[str] | None


@dataclass
class TransformOp(Op):
    source: list[str]
    func: Callable[..., Any]
    comment: list[str] | None


def _nest_in_list(_: Any, val: Any) -> Any:
    return [val]


class MigrationManifest:
    def __init__(self, version: str):
        self._ops: list[Op] = []
        self._version = version

    def add(self, target: str, value: Any = "???", comment: list[str] | None = None) -> None:
        self._ops.append(AddOp(target, value, comment))

    def remove(self, source: str | list[str], comment: list[str] | None = None) -> None:
        if isinstance(source, str):
            source = [source]
        self._ops.append(RemoveOp(source, comment))

    def move(self, source: str, target: str, comment: list[str] | None = None) -> None:
        self._ops.append(MoveOp(source, target, comment))

    def transform(self, source: str | list[str], func: Callable[..., Any], comment: list[str] | None = None) -> None:
        if isinstance(source, str):
            source = [source]
        self._ops.append(TransformOp(source, func, comment))

    def nest_in_list(self, source: str | list[str], comment: list[str] | None = None) -> None:
        return self.transform(source, _nest_in_list, comment)

    def execute(self, cfg: Any):
        for op in self._ops:
            self._exec_step(cfg, op)
        return cfg

    def _exec_step(self, cfg: Any, op: Op):
        start_comment = f"<<< MIGRATION {self._version}"
        end_comment = f">>> MIGRATION {self._version}"
        op_comment = [] if op.comment is None else op.comment
        if isinstance(op, AddOp):
            update(
                cfg,
                op.target,
                op.value,
                new_comment_before=[f"{start_comment}: added key {op.target}.", *op_comment],
                new_comment_after=[end_comment],
            )
        elif isinstance(op, RemoveOp):
            for source in op.source:
                delete(
                    cfg, source, new_comment=[f"{start_comment}: removed key {op.source}.", *op_comment, end_comment],
                )
        elif isinstance(op, MoveOp):
            parent, key = parent_key(op.source)
            parent = select(cfg, parent)
            ca_comments_before = []
            ca_comments_after = []
            if isinstance(parent, CommentedBase):
                ca = parent.ca.items.setdefault(key, [None, None, None, None])
                if ca[1] is not None:
                    ca_comments_before = [c.value.strip().lstrip("# ") for c in ca[1]]
                if ca[2] is not None:
                    ca_comments_after = [ca[2].value.strip().lstrip("# ")]

            val = select(cfg, op.source)
            self._exec_step(cfg, RemoveOp([op.source], comment=None))
            update(
                cfg,
                op.target,
                val,
                new_comment_before=[
                    *ca_comments_before,
                    f"{start_comment}: moved key from {op.source} to {op.target}.",
                    *op_comment,
                ],
                new_comment_after=[end_comment, *ca_comments_after],
            )
        elif isinstance(op, TransformOp):
            for source in op.source:
                new_val = op.func(cfg, select(cfg, source))
                update(
                    cfg,
                    source,
                    new_val,
                    new_comment_before=[f"{start_comment}: key changed.", *op_comment],
                    new_comment_after=[end_comment],
                )


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
        if (not file.is_file() and file.suffix != ".py") or file.name == "__init__.py":
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
        # Migration key to None means compatible but without any migration registered.
        if config[_version_key] is None:
            return True

        key = None
        for k, migration in enumerate(self._migrations):
            if migration.metadata.final:
                return False

            if migration.signature[:8] == config[_version_key]:
                key = k
                break
        if key is None:
            raise ValueError(f"The config migration version {config[_version_key]} does not exist.")

        return True

    def sync(self, config: str) -> str:
        """Migrate the checkpoint using provided migrations

        Parameters
        ----------
        config : str
            The yaml config.

        Returns
        -------
        str
            The migrated yaml
        """
        yaml = YAML()
        yaml_config = yaml.load(config)

        if not self.is_compatible(yaml_config):
            raise IncompatibleCheckpointException("No compatible migration available")
        version = yaml_config[_version_key]
        for migration in self._migrations[version:]:
            manifest = MigrationManifest(migration.signature[:8])
            migration.migrate(manifest)
            migrated_config = manifest.execute(yaml_config)
            migrated_config[_version_key] = migration.signature[:8]
        buf = io.StringIO()
        yaml.dump(migrated_config, buf)
        return buf.getvalue()
