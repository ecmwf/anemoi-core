# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Anemoi Metadata Package.

Layered, component-based reading of ML checkpoint metadata, in two layers.

Layer 1 (adapters): a :class:`CheckpointAdapter` holds the raw metadata
dictionary and composes one adapter per versioned component (``task``,
``lineage``, and ``data`` per dataset; registered in :class:`ComponentRegistry`) plus an
unversioned reader for the raw sections.  A new format or component version
needs only a version reader that subclasses the nearest one and overrides what
differs.
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
>>> print(m.version)
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "999"

# Public API - Layer 1 (for advanced use cases)
# Public API - Layer 2 (recommended for most users)
from .arrays import DatasetArrays
from .base import CheckpointAdapter

# Checkpoint I/O
from .checkpoint import extract_metadata_dict
from .checkpoint import has_metadata
from .checkpoint import load_metadata
from .checkpoint import remove_metadata
from .checkpoint import replace_metadata
from .checkpoint import save_metadata
from .components.base import ComponentAdapter
from .components.base import requires

# Exceptions
from .exceptions import CheckpointError
from .exceptions import MetadataError
from .exceptions import UnknownVersionError
from .exceptions import VersionError
from .interface import DatasetSelectionError
from .interface import DatasetView
from .interface import Metadata
from .registry import ComponentRegistry

# Strict write
from .write_schema import COMPONENT_VERSIONS
from .write_schema import CURRENT_VERSION
from .write_schema import WriteValidationError
from .write_schema import validate_for_write

__all__ = [
    "COMPONENT_VERSIONS",
    "CURRENT_VERSION",
    # Layer 1 - Adapters
    "CheckpointAdapter",
    "CheckpointError",
    "ComponentAdapter",
    "ComponentRegistry",
    "DatasetArrays",
    "DatasetSelectionError",
    "DatasetView",
    # Layer 2 - User API
    "Metadata",
    # Exceptions
    "MetadataError",
    "UnknownVersionError",
    "VersionError",
    "WriteValidationError",
    # Version
    "__version__",
    "extract_metadata_dict",
    "has_metadata",
    # Checkpoint I/O
    "load_metadata",
    "remove_metadata",
    "replace_metadata",
    "requires",
    "save_metadata",
    "validate_for_write",
]
