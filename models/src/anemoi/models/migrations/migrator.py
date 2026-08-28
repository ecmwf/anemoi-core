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
import importlib.util
import logging
import sys
from collections.abc import Callable
from collections.abc import MutableMapping
from collections.abc import Sequence
from copy import deepcopy
from inspect import getsource
from os import PathLike
from pathlib import Path
from pickle import Unpickler
from types import ModuleType
from typing import Any
from typing import Self
from typing import TypedDict

from anemoi.models import __version__
from anemoi.models.migrations.setup_context import MigrationContext
from anemoi.utils.migrations import IncompleteMigrationScript
from anemoi.utils.migrations import Migration
from anemoi.utils.migrations import MigrationMetadata
from anemoi.utils.migrations import Migrator
from anemoi.utils.migrations import SerializedMigration

MIGRATION_PATH = Path(__file__).parent / "scripts"

_CKPT_MIGRATION_KEY = "migrations"

LOGGER = logging.getLogger(__name__)


class IncompatibleCheckpointException(BaseException):
    """The provided checkpoint cannot be migrated because it is to old/recent."""


CkptType = MutableMapping[str, Any]


# migration is the version of the migration module to allow future update of
# the script and keep backward compatibility
MigrationVersions = TypedDict("MigrationVersions", {"migration": str, "anemoi-models": str})


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


class CkptMigration(Migration[CkptType, CkptType, MigrationVersions]):
    """Represents a ckpt migration"""

    def __init__(
        self,
        name: str,
        metadata: MigrationMetadata[MigrationVersions],
        signature: str,
        migrate: Callable[[CkptType], CkptType] | None = None,
        migrate_setup: Callable[[MigrationContext], None] | None = None,
    ) -> None:
        self._migrate_setup = migrate_setup
        super().__init__(name, metadata, signature, migrate)

    @classmethod
    def from_migration(cls, name: str, migration: ModuleType) -> Self:
        if not hasattr(migration, "metadata") or not isinstance(migration.metadata, MigrationMetadata):
            raise IncompleteMigrationScript("Migration script is missing metadata.")

        metadata = migration.metadata
        signature = _get_code_digest(getsource(migration))

        migrate = None
        migrate_setup = None

        if hasattr(migration, "migrate"):
            migrate = migration.migrate
        if hasattr(migration, "migrate_setup"):
            migrate_setup = migration.migrate_setup

        if metadata.versions["anemoi-models"] == "%NEXT_ANEMOI_MODELS_VERSION%":
            metadata.versions["anemoi-models"] = __version__

        return cls(name, metadata, signature, migrate, migrate_setup)

    @property
    def migrate_setup(self) -> Callable[[MigrationContext], None] | None:
        """Setup function to execute before loading the checkpoint. This can be used to
        mock missing modules or Attributes.
        """
        return self._migrate_setup


class MissingAttribute:
    """Placeholder type when encountering ImportError or AttributeError in Unpickler.find_class"""

    def __init__(self, *args, **kwargs): ...


def _get_unpickler(replace_attrs: dict[str, list[str]] | bool = False):
    """Get the Unpickler

    Parameters
    ----------
    replace_attrs : dict[str,list[str]] | bool, default False
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
            except (ImportError, AttributeError):
                attr_name = f"{module_name}.{global_name}"
                wild_name = f"{module_name}.*"

                # For retro-compatibility with checkpoints with rollbacks (now removed from code)
                if attr_name == "anemoi.models.migrations.migrator._SerializedRollback":
                    return MissingAttribute

                deleted_modules: list[str] = []
                deleted_attributes: list[str] = []

                # --- Normalize replace_attrs ---
                if isinstance(replace_attrs, dict):
                    deleted_modules = replace_attrs.get("deleted_modules", [])
                    deleted_attributes = replace_attrs.get("deleted_attributes", [])

                if replace_attrs is False:
                    raise
                if (
                    replace_attrs is True
                    or attr_name in deleted_attributes
                    or module_name in deleted_modules
                    or wild_name in replace_attrs
                ):
                    LOGGER.debug(
                        "Missing attribute %s.%s is checkpoint. Ignoring.",
                        module_name,
                        global_name,
                    )
                    return MissingAttribute
                raise

    class UnpicklerWrapper:
        """For torch.load's pickle_module argument.
        A "module" with the LenientUnpickler as Unpickler.
        """

        Unpickler = _Unpickler

    return UnpicklerWrapper


def _load_ckpt(path: str | PathLike, replace_attrs: dict[str, list[str]] | bool = False) -> CkptType:
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

    pickle_module = _get_unpickler(replace_attrs)
    ckpt = torch.load(path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    if "pytorch-lightning_version" not in ckpt:
        raise ValueError(
            "You can only migrate training checkpoint. If you need a migrated inference checkpoint, fisrt "
            "migrate the training checkpoint, then regenerate the inference one with `anemoi-training checkpoint inference`."
        )
    return ckpt


class CkptMigrator(Migrator[CkptMigration, CkptType]):
    def __init__(self, migrations: Sequence[CkptMigration] | None = None, obj_migration_key: str | None = None) -> None:
        """Create the migrator object

        Parameters
        ----------
        migrations : Sequence[CkptMigration] | None, default None
            List of migration to execute. If None, get migrations from the current folder.
        """

        if migrations is None:
            # remove the ".migrator" at the end to get parent folder as migration package
            migration_pkg, _, _ = __name__.rpartition(".")
            migrations = self._migrations_from_path(CkptMigration, MIGRATION_PATH, f"{migration_pkg}.scripts")

        super().__init__(migrations, obj_migration_key or _CKPT_MIGRATION_KEY)

    def _current_group(self, obj: CkptType) -> int:
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
        if self._obj_migration_key not in obj:
            raise IncompatibleCheckpointException("Checkpoint is not compatible")

        if not len(obj[self._obj_migration_key]):
            return 0
        first_migration = obj[self._obj_migration_key][0]["name"]
        for k, group in enumerate(self._compatibility_groups):
            if group[0].name == first_migration:
                return k
        raise IncompatibleCheckpointException("Checkpoint is not compatible")

    def registered_migrations(self, obj: CkptType) -> list[CkptMigration]:
        migrations: list[CkptMigration] = []
        compat_group = self._compatibility_groups[self._current_group(obj)]
        for registered_migration in obj[self._obj_migration_key]:
            if registered_migration["name"] not in self._migration_refs:
                raise IncompatibleCheckpointException(
                    f"Checkpoint cannot be migrated. Extra migrations are registered. ({registered_migration['name']})"
                )
            migrations.append(compat_group[self._migration_refs[registered_migration["name"]]])
        return migrations

    def missing_migrations(self, obj: CkptType) -> list[CkptMigration]:
        compat_group = self._compatibility_groups[self._current_group(obj)]
        if not len(obj[self._obj_migration_key]):
            return compat_group
        if obj[self._obj_migration_key][-1]["name"] not in self._migration_refs:
            raise IncompatibleCheckpointException(
                f"Checkpoint cannot be migrated. Extra migrations are registered. ({obj[self._obj_migration_key][-1]['name']})"
            )
        last_registered_migration = self._migration_refs[obj[self._obj_migration_key][-1]["name"]]
        return compat_group[last_registered_migration + 1 :]

    def _check_registered_script_changed(self, ckpt: CkptType) -> bool:
        """Checks whether the checkpoint has run a migration that was changed.
        We use the signature stored in the history to detect it.

        Parameters
        ----------
        ckpt : CkptType
            The checkpoint

        Returns
        -------
        bool
            Whether one script in the history has been modified.
        """
        migration_signatures = {migration.name: migration.signature for migration in self.missing_migrations(ckpt)}
        history = ckpt.get("hyper_parameters", {}).get("metadata", {}).get("migrations", {}).get("history", [])
        has_run_modified_migrations = False
        for executed_migration in history:
            if executed_migration["signature"] != migration_signatures[executed_migration["name"]]:
                LOGGER.warning(
                    "Your checkpoint has executed migration %s, but the script has changed. "
                    "Re-run the migrations if possible to use the new updated script.",
                    executed_migration["name"],
                )
                has_run_modified_migrations = True
        return has_run_modified_migrations

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
        group = self._current_group(ckpt)
        if group == len(self._compatibility_groups) - 1:
            return None
        return self._compatibility_groups[group + 1][0].metadata.versions["anemoi-models"]

    def _resolve_context(self, context: MigrationContext) -> None:
        """Resolves the final context object after all setup callbacks have been executed.

        It first tries to move all modules, then moves all attributes.

        Parameters
        ----------
        context : MigrationContext
            The context object
        """
        for module_path in getattr(context, "deleted_modules", []):
            if module_path in sys.modules:
                LOGGER.debug("Delete module %s.", module_path)
                del sys.modules[module_path]

        for module_path_end, module_path_start in context.module_paths.items():
            LOGGER.debug("Move module %s to %s.", module_path_start, module_path_end)
            importlib.import_module(module_path_end)
            sys.modules[module_path_start] = sys.modules[module_path_end]
        for (
            full_attribute_path_end,
            attribute_path_start,
        ) in context.attribute_paths.items():
            attribute_path_start, _, mod_name_start = attribute_path_start.rpartition(".")
            attribute_path_end, _, mod_name_end = full_attribute_path_end.rpartition(".")
            LOGGER.debug(
                "Move attribute %s from %s to %s.",
                mod_name_start,
                attribute_path_start,
                full_attribute_path_end,
            )
            mod_end = importlib.import_module(attribute_path_end, __name__)
            attr_end = getattr(mod_end, mod_name_end)
            mod_start = sys.modules[attribute_path_start]
            setattr(mod_start, mod_name_start, attr_end)

    def sync(self, path: str | PathLike) -> tuple[CkptType, CkptType, list[CkptMigration]]:
        """Migrate or rollbacks the checkpoint using provided migrations

        Parameters
        ----------
        path : str | PathLike
            The checkpoint to migrate.

        Returns
        -------
        tuple[CkptType, CkptType, list[CkptMigration]]
            * The original checkpoint (might have obfuscated attributes with `MissingAttribute`
                if it cannot be imported
            * The migrated checkpoint
            * The list of executed migrations
        """
        # First load the checkpoint and obfuscate any import issue, just to get the
        # migrations from the checkpoint. The real checkpoint is reloaded afterwards.
        old_ckpt = _load_ckpt(path, replace_attrs=True)
        ckpt = deepcopy(old_ckpt)

        if not self.is_compatible(ckpt):
            first_incompatible_version = self.get_first_incompatible_version(ckpt)
            raise IncompatibleCheckpointException(
                "No compatible migration available: the checkpoint is too old. "
                f"Use a version of anemoi-models < {first_incompatible_version}."
            )
        self._check_registered_script_changed(ckpt)
        replace_attrs: dict[str, list[str]] = {}
        missing_migrations = self.missing_migrations(ckpt)
        context = MigrationContext()
        for migration in missing_migrations:
            if migration.migrate_setup is not None:
                migration.migrate_setup(context)
        self._resolve_context(context)
        if len(context.deleted_modules):
            replace_attrs["deleted_modules"] = context.deleted_modules
        if len(context.deleted_attributes):
            replace_attrs["deleted_attributes"] = context.deleted_attributes
        # Force reloading checkpoint without obfuscating import issues.
        ckpt = _load_ckpt(path, replace_attrs)
        ckpt["hyper_parameters"]["metadata"].setdefault("migrations", {}).setdefault("history", [])
        for migration in missing_migrations:
            if migration.migrate is None:
                raise IncompatibleCheckpointException(
                    f"Migration {migration.name} cannot be executed. Missing migrate function."
                )
            ckpt = migration.migrate(ckpt)
            ckpt[_CKPT_MIGRATION_KEY].append(migration.serialize())
            ckpt["hyper_parameters"]["metadata"]["migrations"]["history"].append(
                {"type": "migrate", "name": migration.name, "signature": migration.signature}
            )
        return old_ckpt, ckpt, missing_migrations

    def inspect(self, path: str | PathLike) -> tuple[list[Migration], list[Migration]]:
        """Inspect migration information in checkpoint

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
        ckpt = _load_ckpt(path, replace_attrs=True)
        if not self.is_compatible(ckpt):
            first_incompatible_version = self.get_first_incompatible_version(ckpt)
            raise IncompatibleCheckpointException(
                "No compatible migration available: the checkpoint is too old. "
                f"Use a version of anemoi-models < {first_incompatible_version}."
            )
        return list(self.registered_migrations(ckpt)), list(self.missing_migrations(ckpt))

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
        if self._obj_migration_key not in ckpt:
            ckpt[self._obj_migration_key] = []
        for migration in self._compatibility_groups[-1]:
            ckpt[self._obj_migration_key].append(migration.serialize())
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
                    "metadata": migration.get(
                        "metadata",
                        {"versions": {"migration": "1.0.0", "anemoi-models": "x.x.x"}},
                    ),
                    "signature": migration.get("signature", migration.get("name", "")),
                }
            )
        ckpt["migrations"] = ckpt_migrations
        ckpt["pytorch-lightning_version"] = ""
        ckpt.setdefault("hyper_parameters", {}).setdefault("metadata", {})
        path = self.ckpt_dir / name
        torch.save(ckpt, path)
        return path
