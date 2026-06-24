.. _checkpoint_integration:

#################################
 Checkpoint Pipeline Integration
#################################

This guide covers the checkpoint pipeline infrastructure for Anemoi
training. The pipeline provides a foundation for building checkpoint
loading workflows.

.. note::

   The trainer builds and runs this pipeline via the opt-in
   ``training.checkpoint`` configuration surface (sources, loaders, and
   modifiers). See :ref:`checkpoint_pipeline_configuration` for the
   configuration details.

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

**From configuration:**

The trainer builds the pipeline from the ``training.checkpoint`` surface
via :func:`anemoi.training.checkpoint.builder.build_checkpoint_pipeline`,
which assembles stages in a fixed order (source, then loading, then
modifiers). See :ref:`checkpoint_pipeline_configuration` for the
configuration namespace.

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

The pipeline is configured under the ``training.checkpoint`` namespace
with three optional blocks read by
:func:`anemoi.training.checkpoint.builder.build_checkpoint_pipeline`.
The builder assembles stages in a fixed order: source, then loading,
then modifiers (list order). Any block may be absent; an absent block
means no stage of that kind.

.. code:: yaml

   training:
     checkpoint:
       source: # OPTIONAL — a single Hydra _target_ (acquisition stage)
         _target_: anemoi.training.checkpoint.sources.local.LocalSource

       loading: # OPTIONAL — a single Hydra _target_ (loading strategy)
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

       modifiers: # OPTIONAL — a LIST of stages, applied in order after loading
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder, "processor.0"]
           strict: false
           validate_gradients: true

**Hydra group selection:**

The source, loading, and modifier blocks are wired as opt-in default
groups. The default is null (no pipeline). Select a group to opt in:

.. code:: bash

   anemoi-training train training/checkpoint/loading=weights_only

   anemoi-training train training/checkpoint/source=local \
       training/checkpoint/loading=transfer_learning

See :ref:`checkpoint_pipeline_configuration` for the full list of group
options and per-stage parameters.

************
 Next Steps
************

This infrastructure underpins the full checkpoint pipeline:

-  **Loading strategies**: weights-only, transfer learning, warm start,
   cold start (under ``training.checkpoint.loading``)
-  **Model modifiers**: post-loading transformations such as freezing
   (under ``training.checkpoint.modifiers``)

See :ref:`checkpoint_pipeline_configuration` for configuration details
and :ref:`checkpoint_troubleshooting` for common issues.
