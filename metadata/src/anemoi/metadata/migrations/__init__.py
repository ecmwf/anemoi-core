# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE 2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Migration functions between schema versions.

Each module in this package defines one migration step between adjacent
schema versions.  Migrations are registered automatically via the
``@MetadataMigrator.register_migration`` decorator; no manual wiring is
needed beyond dropping a file in this directory.

How to add a migration
----------------------
1. Create a new file named ``v{X}_to_v{Y}.py`` in this directory.
2. Import the source and target schema classes.
3. Decorate your migration function::

       @MetadataMigrator.register_migration("1.0", "2.0")
       def migrate_v1_to_v2(old: MetadataV1) -> MetadataV2:
           ...

4. That's it -- the auto-import below handles registration.

The auto-import loop at the bottom of this file imports every module in
the package when ``anemoi.metadata.migrations`` is first imported.  This
mirrors the pattern used by ``versions/__init__.py`` for schema classes.
Modules whose names start with an underscore (e.g. ``_example.py``) are
skipped to allow for templates and private utilities.

Notes
-----
* Migration functions must be **pure** -- no side effects, no I/O.
* They receive a fully-validated *old* instance and must return a fully-
  validated *new* instance.  Pydantic handles validation on both ends.
* Multi-step jumps (e.g. V1 → V3) are chained automatically by
  :class:`~anemoi.metadata.migration.MetadataMigrator`.
"""

import importlib
import pkgutil

# Auto-import all modules in this package to trigger @register_migration
# decorators.  New migration files are picked up without any changes here.
# Skip underscore-prefixed modules (templates, examples, private utilities).
for _importer, _name, _ispkg in pkgutil.iter_modules(__path__):
    if not _name.startswith("_"):
        importlib.import_module(f".{_name}", __name__)
