# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Core checkpoint pipeline base classes."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn
    from omegaconf import DictConfig
    from torch.optim import Optimizer


@dataclass
class CheckpointContext:
    """Carries state through pipeline stages."""

    checkpoint_path: Path | None = None
    checkpoint_data: dict[str, Any] | None = None
    model: nn.Module | None = None
    optimizer: Optimizer | None = None
    scheduler: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    config: DictConfig | None = None
    checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"] | None = None
    pl_module: Any | None = None

    def has_checkpoint_data(self) -> bool:
        """Check if checkpoint data is loaded."""
        return self.checkpoint_data is not None and bool(self.checkpoint_data)

    def update_metadata(self, **kwargs) -> None:
        """Update metadata dictionary with new values."""
        self.metadata.update(kwargs)


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Process the context and return updated context."""
