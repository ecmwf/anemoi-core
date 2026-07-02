# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""EXAMPLE migration template -- not imported (underscore prefix).

Copy this file to ``v1_to_v2.py`` (dropping the underscore) when creating
a real migration.  The auto-import in ``__init__.py`` will pick it up.

Writing a migration
-------------------
1. Copy this file to ``v{X}_to_v{Y}.py``.
2. Import the source and target schema classes.
3. Decorate with ``@MetadataMigrator.register_migration("X.0", "Y.0")``.
4. Implement the function: receive validated old, return validated new.

Minor version migrations
------------------------
For minor versions (e.g. 1.0 -> 1.1) that share the same schema class,
register the minor version with::

    MetadataRegistry.register_minor("1.1")

The migrator automatically bridges schema-sharing versions with a no-op
version bump (using ``copy_with(schema_version="1.1")``), so no explicit
migration function is needed **unless** you need to populate or transform
fields.  If you do need to transform fields, register a migration function
as usual::

    @MetadataMigrator.register_migration("1.0", "1.1")
    def migrate_v1_0_to_v1_1(old: MetadataV1) -> MetadataV1:
        return old.copy_with(
            schema_version="1.1",
            new_optional_field="default_value",  # populate new field
        )

Tips
----
* Migrations must be **pure** -- no I/O, no side effects.
* Use ``old.model_dump()`` to get a dict you can manipulate.
* Use ``TargetClass.model_validate(data)`` to validate the output.
* Use ``old.copy_with(field=new_value)`` for same-version tweaks.
* Access extra top-level keys via ``old.model_extra``.
* ``old.metadata_inference.model_dump()`` gives you the inference block
  as a plain dict for restructuring.

Version scheme
--------------
Versions use major.minor only (no patch). Examples: "1.0", "2.0", "2.1".
Register with the version string exactly as it appears in the registry.
"""

# === Major version migration example ===
#
# from anemoi.metadata.migration import MetadataMigrator
# from anemoi.metadata.versions.v1 import MetadataV1
# from anemoi.metadata.versions.v2 import MetadataV2
#
#
# @MetadataMigrator.register_migration("1.0", "2.0")
# def migrate_v1_to_v2(old: MetadataV1) -> MetadataV2:
#     """Migrate V1 to V2.
#
#     Parameters
#     ----------
#     old : MetadataV1
#         Validated V1 instance.
#
#     Returns
#     -------
#     MetadataV2
#         Validated V2 instance.
#     """
#     # Step 1: Extract what you need from the old schema
#     dataset = dict(old.dataset)
#     new_field = dataset.pop("some_key", {})
#
#     # Step 2: Carry forward extra top-level keys
#     extras = dict(old.model_extra or {})
#
#     # Step 3: Build and validate the new schema
#     return MetadataV2.model_validate({
#         "schema_version": "2.0",
#         "created_at": old.created_at.isoformat() if old.created_at else None,
#         "metadata_inference": old.metadata_inference.model_dump(),
#         "new_typed_field": new_field,
#         "config": dict(old.config),
#         "training": dict(old.training),
#         "dataset": dataset,
#         "environment": dict(old.environment),
#         "provenance": dict(old.provenance),
#         **extras,
#     })

# === Minor version migration example ===
#
# from anemoi.metadata.migration import MetadataMigrator
# from anemoi.metadata.versions.v1 import MetadataV1
#
#
# @MetadataMigrator.register_migration("1.0", "1.1")
# def migrate_v1_0_to_v1_1(old: MetadataV1) -> MetadataV1:
#     """Migrate V1.0 to V1.1 (add new_optional_field).
#
#     Parameters
#     ----------
#     old : MetadataV1
#         Validated V1.0 instance.
#
#     Returns
#     -------
#     MetadataV1
#         V1.1 instance with schema_version bumped.
#     """
#     return old.copy_with(schema_version="1.1")
