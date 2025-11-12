# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict


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
        self.deleted_modules: list[str] = []

    def delete_attribute(self, path: str) -> None:
        """Indicate that an attribute has been deleted. Any class referencing this module will
        be replace by a ``MissingAttribute`` object.

        Parameters
        ----------
        path : str
            Path to the attribute. For example ``pkg.mod.MyClass``.
        """
        self.deleted_attributes.append(path)

    def delete_module(self, path: str) -> None:
        """Mark a module for deletion."""
        self.deleted_modules.append(path)

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


class SerializedMigrationContext(TypedDict):
    """Serialized migration context"""

    attribute_paths: dict[str, str]
    module_paths: dict[str, str]
    deleted_attributes: list[str]
    deleted_modules: list[str]


def serialize_setup_callback(setup: Callable[[MigrationContext], None]) -> SerializedMigrationContext:
    """Serialize a setup callback. It runs the callback with a dummy context and
    returns the serialized context.

    Parameters
    ----------
    setup : Callable[[MigrationContext], None]
        The setup callback.

    Returns
    -------
    _SerializedMigrationContext
        The serialized migration context.
    """
    ctx = MigrationContext()
    setup(ctx)
    return {
        "attribute_paths": ctx.attribute_paths,
        "module_paths": ctx.module_paths,
        "deleted_attributes": ctx.deleted_attributes,
        "deleted_modules": ctx.deleted_modules,
    }


class DeserializeMigrationContext:
    """Deserializes the serialized migratoin context into a setup callback"""

    def __init__(self, ctx: SerializedMigrationContext) -> None:
        self._ctx = ctx

    def __call__(self, context: MigrationContext) -> None:
        for deleted_attribute in self._ctx["deleted_attributes"]:
            context.delete_attribute(deleted_attribute)
        for path_end, path_start in self._ctx["attribute_paths"].items():
            context.move_attribute(path_start, path_end)
        for deleted_module in self._ctx["deleted_modules"].items():
            context.delete_module(deleted_module)
        for path_end, path_start in self._ctx["module_paths"].items():
            context.move_module(path_start, path_end)


class ReversedSetupCallback:
    """Reverses a setup callback.
    When called with __call__, it creates a dummy context, runs the callback, then reverse the output of the
    context into the provided context.
    """

    def __init__(self, callback: Callable[[MigrationContext], None]) -> None:
        self._callback = callback

    def __call__(self, context: MigrationContext) -> None:
        new_ctx = MigrationContext()
        # apply the callback on a dummy context
        self._callback(new_ctx)
        # then reverse everything that was registered
        # Note context.delete_attribute is not present because items are
        # not deleted when going back
        for path_end, path_start in new_ctx.attribute_paths.items():
            context.move_attribute(path_end, path_start)
        for path_end, path_start in new_ctx.module_paths.items():
            context.move_module(path_end, path_start)
