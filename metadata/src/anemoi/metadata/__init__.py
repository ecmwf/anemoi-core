# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Anemoi Metadata Package.

Versioned metadata models for ML checkpoints with a two-layer architecture.

Layer 1 (Raw Schema): Frozen pydantic models for data validation.
Layer 2 (User API): Metadata class with computed properties and helpers.

Examples
--------
>>> from anemoi.metadata import Metadata
>>>
>>> # Load from checkpoint
>>> m = Metadata.from_checkpoint("model.ckpt")
>>>
>>> # Access properties
>>> print(m.variables)
>>> print(m.schema_version)
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "999"

# Import migrations to register migration functions with MetadataMigrator.
# This must come after versions so that all schema classes are available
# when the migration modules are imported.
# Import versions to register them with the registry
from . import migrations  # noqa: F401
from . import versions  # noqa: F401

# Public API - Layer 1 (for advanced use cases)
from .base import MetadataContract
from .base import requires_version

# Checkpoint I/O
from .checkpoint import extract_metadata_dict
from .checkpoint import has_metadata
from .checkpoint import load_metadata
from .checkpoint import remove_metadata
from .checkpoint import replace_metadata
from .checkpoint import save_metadata

# Exceptions
from .exceptions import CheckpointError
from .exceptions import MetadataError
from .exceptions import MigrationError
from .exceptions import UnknownVersionError
from .exceptions import VersionError

# Public API - Layer 2 (recommended for most users)
from .interface import DatasetView
from .interface import Metadata

# Migration
from .migration import MetadataMigrator
from .registry import MetadataRegistry

# Version schemas (for type hints and advanced usage)
from .versions import MetadataV0
from .versions import MetadataV1

__all__ = [
    # Version
    "__version__",
    # Layer 2 - User API
    "Metadata",
    "DatasetView",
    # Layer 1 - Raw API
    "MetadataContract",
    "MetadataRegistry",
    "requires_version",
    # Version schemas
    "MetadataV0",
    "MetadataV1",
    # Checkpoint I/O
    "load_metadata",
    "save_metadata",
    "replace_metadata",
    "remove_metadata",
    "has_metadata",
    "extract_metadata_dict",
    # Migration
    "MetadataMigrator",
    # Exceptions
    "MetadataError",
    "VersionError",
    "UnknownVersionError",
    "MigrationError",
    "CheckpointError",
]
