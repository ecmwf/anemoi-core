# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint pipeline infrastructure for Anemoi training.

This module provides a flexible, extensible pipeline for checkpoint
handling including acquisition from various sources, loading strategies,
and model modifications. The system leverages Hydra for configuration
and instantiation, providing a clean integration with the existing
Anemoi ecosystem.

Key components:
- CheckpointContext: Carries state through pipeline stages
- PipelineStage: Base class for all pipeline stages
- CheckpointPipeline: Orchestrates stage execution
- ComponentCatalog: Lightweight registry for component discovery
"""

from .base import CheckpointContext
from .base import PipelineStage
from .catalog import ComponentCatalog
from .exceptions import CheckpointConfigError
from .exceptions import CheckpointError
from .exceptions import CheckpointIncompatibleError
from .exceptions import CheckpointLoadError
from .exceptions import CheckpointNotFoundError
from .exceptions import CheckpointSourceError
from .exceptions import CheckpointTimeoutError
from .exceptions import CheckpointValidationError
from .pipeline import CheckpointPipeline
from .utils import calculate_checksum
from .utils import compare_state_dicts
from .utils import download_with_retry
from .utils import estimate_checkpoint_memory
from .utils import format_size
from .utils import get_checkpoint_metadata
from .utils import validate_checkpoint

__all__ = [
    "CheckpointConfigError",
    # Core classes
    "CheckpointContext",
    # Exceptions
    "CheckpointError",
    "CheckpointIncompatibleError",
    "CheckpointLoadError",
    "CheckpointNotFoundError",
    "CheckpointPipeline",
    "CheckpointSourceError",
    "CheckpointTimeoutError",
    "CheckpointValidationError",
    "ComponentCatalog",
    "PipelineStage",
    "calculate_checksum",
    "compare_state_dicts",
    # Utilities
    "download_with_retry",
    "estimate_checkpoint_memory",
    "format_size",
    "get_checkpoint_metadata",
    "validate_checkpoint",
]

__version__ = "0.1.0"
