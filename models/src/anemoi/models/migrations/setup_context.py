# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations


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
