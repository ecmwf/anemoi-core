# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Core abstractions for checkpoint pipeline.

This module provides the fundamental building blocks for the checkpoint
pipeline system. It defines the context object that carries state through
the pipeline and the abstract base class for all pipeline stages.

The checkpoint pipeline follows a simple pattern:
1. A CheckpointContext is created with initial state
2. The context is passed through a series of PipelineStage implementations
3. Each stage modifies the context and passes it to the next stage
4. The final context contains the result of all transformations

Example
-------
>>> from anemoi.training.checkpoint import CheckpointContext, PipelineStage
>>>
>>> class MyStage(PipelineStage):
...     async def process(self, context: CheckpointContext) -> CheckpointContext:
...         context.update_metadata(processed_by='MyStage')
...         return context
>>>
>>> context = CheckpointContext()
>>> stage = MyStage()
>>> # In async context:
>>> # result = await stage.process(context)
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    import torch.nn as nn
    from omegaconf import DictConfig
    from torch.optim import Optimizer


@dataclass
class CheckpointContext:
    """Carries state through pipeline stages.

    This context object is passed through each stage of the checkpoint
    pipeline, accumulating state and metadata as it progresses. It serves
    as a shared data structure that allows stages to communicate and build
    upon each other's work.

    The context supports:
    - Checkpoint data management (path and loaded data)
    - Model and optimizer state tracking
    - Metadata accumulation for debugging and logging
    - Configuration passing for stage behavior

    Parameters
    ----------
    checkpoint_path : Path, optional
        Local path to checkpoint file (if fetched from remote source)
    checkpoint_data : dict, optional
        Loaded checkpoint data dictionary containing model weights,
        optimizer state, training metadata, etc.
    model : nn.Module, optional
        PyTorch model being modified by the pipeline. Can be either
        AnemoiModelInterface (pure PyTorch) or extracted from GraphForecaster
    optimizer : Optimizer, optional
        Optional optimizer to restore state to (for warm starts)
    scheduler : Any, optional
        Optional learning rate scheduler to restore state to
    metadata : dict
        Dictionary of accumulated metadata from pipeline stages.
        Each stage can add information here for tracking.
    config : DictConfig, optional
        Hydra configuration object containing pipeline settings
    checkpoint_format : str, optional
        Format of the checkpoint: 'lightning', 'pytorch', 'safetensors', or 'state_dict'
    pl_module : pl.LightningModule, optional
        Lightning module (GraphForecaster) if loading from Lightning checkpoint.
        This preserves the full Lightning context for training resumption

    Examples
    --------
    >>> import torch.nn as nn
    >>> from pathlib import Path
    >>>
    >>> # Create context with initial state
    >>> context = CheckpointContext(
    ...     checkpoint_path=Path('/tmp/model.ckpt'),
    ...     model=nn.Linear(10, 5),
    ...     metadata={'source': 'local'}
    ... )
    >>>
    >>> # Update metadata during processing
    >>> context.update_metadata(stage='loading', status='success')
    >>> print(context.metadata)
    {'source': 'local', 'stage': 'loading', 'status': 'success'}
    """

    checkpoint_path: Path | None = None
    checkpoint_data: dict[str, Any] | None = None
    model: nn.Module | None = None
    optimizer: Optimizer | None = None
    scheduler: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    config: DictConfig | None = None
    checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"] | None = None
    pl_module: Any | None = None  # Type hint as Any to avoid circular imports, actually pl.LightningModule

    def __post_init__(self):
        """Validate context after initialization."""
        if self.checkpoint_path is not None and not isinstance(self.checkpoint_path, Path):
            self.checkpoint_path = Path(self.checkpoint_path)

    def update_metadata(self, **kwargs) -> None:
        """Update metadata dictionary with new values.

        This method allows stages to add information to the shared metadata
        dictionary. Existing keys will be overwritten with new values.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to add to metadata

        Examples
        --------
        >>> context = CheckpointContext()
        >>> context.update_metadata(epoch=10, loss=0.5)
        >>> context.update_metadata(epoch=11)  # Updates existing key
        >>> print(context.metadata)
        {'epoch': 11, 'loss': 0.5}
        """
        self.metadata.update(kwargs)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get value from metadata with optional default.

        Safe method to retrieve metadata values without raising KeyError.

        Parameters
        ----------
        key : str
            Metadata key to retrieve
        default : Any, optional
            Default value if key not found (default: None)

        Returns
        -------
        Any
            Metadata value or default if key doesn't exist

        Examples
        --------
        >>> context = CheckpointContext(metadata={'epoch': 10})
        >>> context.get_metadata('epoch')
        10
        >>> context.get_metadata('missing', 'default_value')
        'default_value'
        """
        return self.metadata.get(key, default)

    def has_checkpoint_data(self) -> bool:
        """Check if checkpoint data is loaded.

        This method verifies that checkpoint data exists and is not empty,
        which is useful for stages that require checkpoint data to operate.

        Returns
        -------
        bool
            True if checkpoint_data is not None and not empty,
            False otherwise

        Examples
        --------
        >>> context = CheckpointContext()
        >>> context.has_checkpoint_data()
        False
        >>> context.checkpoint_data = {'state_dict': {}}
        >>> context.has_checkpoint_data()
        True
        """
        return self.checkpoint_data is not None and len(self.checkpoint_data) > 0

    def __repr__(self) -> str:
        """String representation of context."""
        parts = []
        if self.checkpoint_path:
            parts.append(f"path={self.checkpoint_path.name}")
        if self.model:
            parts.append(f"model={type(self.model).__name__}")
        if self.metadata:
            parts.append(f"metadata_keys={list(self.metadata.keys())}")
        return f"CheckpointContext({', '.join(parts)})"


class PipelineStage(ABC):
    """Base class for all pipeline stages.

    Each stage in the checkpoint pipeline must inherit from this class
    and implement the process method. Stages are composable and can be
    chained together in a pipeline to create complex checkpoint processing
    workflows.

    Pipeline stages follow these principles:
    - Single Responsibility: Each stage does one thing well
    - Composability: Stages can be combined in different orders
    - Immutability: Stages should not have mutable state between calls
    - Error Handling: Stages should raise appropriate exceptions on failure

    Examples
    --------
    >>> class ValidationStage(PipelineStage):
    ...     async def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         if not context.has_checkpoint_data():
    ...             raise CheckpointError("No checkpoint data to validate")
    ...         # Perform validation
    ...         context.update_metadata(validated=True)
    ...         return context
    >>>
    >>> # Use in a pipeline
    >>> from anemoi.training.checkpoint import CheckpointPipeline
    >>> pipeline = CheckpointPipeline([ValidationStage()])
    """

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Process the checkpoint context.

        This method should perform the stage's operation on the context
        and return the modified context for the next stage. The method
        is async to support I/O operations like downloading checkpoints
        or loading from disk.

        Implementations should:
        - Validate required context fields exist
        - Perform their specific operation
        - Update context with results
        - Add relevant metadata for tracking
        - Handle errors appropriately

        Parameters
        ----------
        context : CheckpointContext
            Current checkpoint context containing accumulated state
            from previous stages

        Returns
        -------
        CheckpointContext
            Modified checkpoint context with this stage's contributions.
            This should typically be the same object, modified in place.

        Raises
        ------
        CheckpointError
            Base exception for checkpoint operations
        CheckpointNotFoundError
            If a required checkpoint file doesn't exist
        CheckpointLoadError
            If checkpoint data cannot be loaded
        CheckpointIncompatibleError
            If checkpoint is incompatible with the model

        Notes
        -----
        The process method is async to support:
        - Network operations (downloading from S3, HTTP, etc.)
        - Large file I/O operations
        - Concurrent processing when needed

        If your stage doesn't need async operations, you can still
        implement it as an async method that doesn't use await.
        """

    def __repr__(self) -> str:
        """String representation of stage."""
        return f"{self.__class__.__name__}()"

    async def __call__(self, context: CheckpointContext) -> CheckpointContext:
        """Allow stage to be called directly.

        This convenience method allows stages to be used as callables,
        making them more intuitive to use outside of a pipeline.

        Parameters
        ----------
        context : CheckpointContext
            Current checkpoint context

        Returns
        -------
        CheckpointContext
            Modified checkpoint context

        Examples
        --------
        >>> # Instead of:
        >>> result = await stage.process(context)
        >>>
        >>> # You can write:
        >>> result = await stage(context)
        """
        return await self.process(context)
