# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Versioned metadata schemas.

This module auto-imports all version modules to ensure they register
with the MetadataRegistry upon package import.

Adding a new major version
--------------------------
1. Create ``v{N}.py`` in this directory.
2. Decorate the schema class with ``@MetadataRegistry.register("N.0")``.
3. Add the import below so the class is registered on package import.
4. Export it in ``__all__`` for convenient access.
5. Add a migration in ``../migrations/v{N-1}_to_v{N}.py``.

Adding a minor version (backwards-compatible additions)
-------------------------------------------------------
Minor versions reuse the same schema class as their base version.  The
schema class must use ``extra="allow"`` so new optional fields pass
validation.

1. After the base class import/registration, call::

       MetadataRegistry.register_minor("1.1", base_version="1.0")

2. If the minor requires data transforms on upgrade, add a migration in
   ``../migrations/v1_0_to_v1_1.py``.  Otherwise no migration is needed
   (the schema already accepts the new fields via ``extra="allow"``).
"""

from ..registry import MetadataRegistry
from .v0 import MetadataV0
from .v1 import MetadataV1

__all__ = [
    "MetadataRegistry",
    "MetadataV0",
    "MetadataV1",
]

# -- Minor version registrations --
# Add backwards-compatible minor versions here.  Example:
# MetadataRegistry.register_minor("1.1", base_version="1.0")
