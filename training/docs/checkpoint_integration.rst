.. _checkpoint_integration:

#################################
 Checkpoint Pipeline Integration
#################################

This guide covers the checkpoint pipeline infrastructure for Anemoi
training. The pipeline provides a foundation for building checkpoint
loading workflows.

.. note::

   This documents Phase 1 (Pipeline Infrastructure). Sources, loaders,
   and modifiers are implemented in subsequent phases.

**************
 Core Classes
**************

CheckpointContext
=================

The ``CheckpointContext`` carries state through pipeline stages:

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext

   # Create context with a model
   context = CheckpointContext(
       model=my_model,
       config=my_config,  # Optional OmegaConf config
   )

   # Access and update metadata
   context.update_metadata(source="local", loaded=True)
   print(context.metadata)

**Attributes:**

-  ``model``: PyTorch model
-  ``optimizer``: Optional optimizer
-  ``scheduler``: Optional learning rate scheduler
-  ``checkpoint_path``: Path to checkpoint file
-  ``checkpoint_data``: Loaded checkpoint dictionary
-  ``metadata``: Dictionary for tracking state
-  ``config``: Optional Hydra configuration
-  ``checkpoint_format``: Detected format (lightning, pytorch,
   state_dict)

PipelineStage
=============

Base class for implementing pipeline stages:

.. code:: python

   from anemoi.training.checkpoint import PipelineStage, CheckpointContext


   class MyCustomStage(PipelineStage):
       def __init__(self, param: str):
           self.param = param

       async def process(self, context: CheckpointContext) -> CheckpointContext:
           # Implement your logic here
           context.update_metadata(custom_param=self.param)
           return context

CheckpointPipeline
==================

Orchestrates execution of multiple stages:

.. code:: python

   from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext

   # Build pipeline with stages
   pipeline = CheckpointPipeline(
       stages=[stage1, stage2, stage3],
       async_execution=True,
       continue_on_error=False,
   )

   # Execute
   context = CheckpointContext(model=my_model)
   result = await pipeline.execute(context)

**From Hydra configuration:**

.. code:: python

   from omegaconf import OmegaConf

   config = OmegaConf.create({
       "stages": [
           {"_target_": "my_module.MyStage", "param": "value"},
       ],
       "async_execution": True,
   })

   pipeline = CheckpointPipeline.from_config(config)

****************
 Error Handling
****************

The checkpoint module provides a hierarchy of exceptions:

.. code:: python

   from anemoi.training.checkpoint import (
       CheckpointError,           # Base exception
       CheckpointNotFoundError,   # File not found
       CheckpointLoadError,       # Loading failed
       CheckpointValidationError, # Validation failed
       CheckpointSourceError,     # Source fetch failed
       CheckpointTimeoutError,    # Operation timed out
       CheckpointConfigError,     # Configuration error
       CheckpointIncompatibleError,  # Model/checkpoint mismatch
   )

   try:
       result = await pipeline.execute(context)
   except CheckpointNotFoundError as e:
       print(f"Checkpoint not found: {e.path}")
   except CheckpointLoadError as e:
       print(f"Failed to load: {e}")
   except CheckpointError as e:
       print(f"Checkpoint error: {e}")

*******************
 Utility Functions
*******************

Format Detection
================

.. code:: python

   from anemoi.training.checkpoint.formats import (
       detect_checkpoint_format,
       load_checkpoint,
       extract_state_dict,
   )

   # Auto-detect format
   fmt = detect_checkpoint_format("/path/to/checkpoint.ckpt")
   # Returns: "lightning", "pytorch", or "state_dict"

   # Load checkpoint
   data = load_checkpoint("/path/to/checkpoint.ckpt")

   # Extract state dict from various formats
   state_dict = extract_state_dict(data)

Checkpoint Utilities
====================

.. code:: python

   from anemoi.training.checkpoint import (
       get_checkpoint_metadata,
       validate_checkpoint,
       calculate_checksum,
       compare_state_dicts,
       estimate_checkpoint_memory,
       format_size,
   )

   # Get metadata without loading full checkpoint
   metadata = get_checkpoint_metadata(Path("model.ckpt"))

   # Validate checkpoint structure
   validate_checkpoint(checkpoint_data)

   # Calculate file checksum
   checksum = calculate_checksum(Path("model.ckpt"), algorithm="sha256")

   # Compare state dictionaries
   missing, unexpected, mismatched = compare_state_dicts(source_dict, target_dict)

   # Estimate memory usage
   bytes_needed = estimate_checkpoint_memory(checkpoint_data)
   print(format_size(bytes_needed))  # e.g., "1.5 GB"

*********************
 Component Discovery
*********************

The ``ComponentCatalog`` provides discovery of available pipeline
components:

.. code:: python

   from anemoi.training.checkpoint import ComponentCatalog

   # List available components
   print(ComponentCatalog.list_sources())    # Available source types
   print(ComponentCatalog.list_loaders())    # Available loading strategies
   print(ComponentCatalog.list_modifiers())  # Available model modifiers

   # Get Hydra target path for a component
   target = ComponentCatalog.get_source_target("local")

********************
 Configuration YAML
********************

Example pipeline configuration:

.. code:: yaml

   # config/training/checkpoint_pipeline.yaml
   training:
     checkpoint_pipeline:
       stages:
         # Each stage uses Hydra _target_ pattern
         - _target_: my_module.sources.LocalSource
           path: /path/to/checkpoint.ckpt

         - _target_: my_module.loaders.WeightsOnlyLoader
           strict: false

       async_execution: true
       continue_on_error: false

**Execution Patterns:**

The pipeline supports two execution approaches:

#. **Standalone (recommended)**: Execute during model initialization

   .. code:: python

      pipeline = CheckpointPipeline.from_config(config)
      context = CheckpointContext(model=model)
      result = await pipeline.execute(context)
      model = result.model

#. **Lightning callback**: Integrate with PyTorch Lightning lifecycle
   for coordinated checkpoint operations.

************
 Next Steps
************

This infrastructure enables subsequent phases:

-  **Phase 2**: Loading strategies (weights-only, transfer learning,
   warm/cold start)
-  **Phase 3**: Integration with model modifiers and legacy migration

See :ref:`checkpoint_pipeline_configuration` for configuration details
and :ref:`checkpoint_troubleshooting` for common issues.
