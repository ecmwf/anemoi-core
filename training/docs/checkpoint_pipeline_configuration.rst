.. _checkpoint_pipeline_configuration:

###################################
 Checkpoint Pipeline Configuration
###################################

This guide covers configuration of the checkpoint pipeline system.

**********
 Overview
**********

The checkpoint pipeline provides a composable system for:

-  **Applying loading strategies** (weights-only, transfer learning,
   warm/cold start)
-  **Loading checkpoints** from various sources (local, HTTP, S3)
-  **Modifying models** after loading (freezing, adapters) — planned

*****************
 Basic Structure
*****************

.. code:: yaml

   training:
     checkpoint_pipeline:
       stages:
         # Pipeline stages using Hydra _target_ pattern
         - _target_: path.to.SourceStage
           param: value

         - _target_: path.to.LoaderStage
           strict: false

       # Pipeline settings
       async_execution: true
       continue_on_error: false

**Key settings:**

-  ``stages``: List of pipeline stages with Hydra ``_target_`` pattern
-  ``async_execution``: Use async I/O (default: true)
-  ``continue_on_error``: Continue on stage failures (default: false)

************************
 Configuration Sections
************************

Checkpoint Sources
==================

Sources define where to fetch checkpoints.

.. note::

   Source implementations (LocalSource, S3Source, HTTPSource) are part
   of this package (checkpoint acquisition layer, PR #464 / issue #458).
   Wiring the pipeline into the trainer is the Phase 3 integration work
   (issue #495).

Local Files
-----------

``LocalSource`` reads its path from ``checkpoint_path`` on the pipeline
context rather than from a stage argument:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.LocalSource

Amazon S3
---------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.S3Source
       url: s3://my-models/checkpoints/model-v1.ckpt

HTTP/HTTPS
----------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.HTTPSource
       url: https://models.example.com/checkpoint.ckpt

Loading Strategies
==================

Strategies define how to apply checkpoint data to your model. All four
strategies below are implemented in
``anemoi.training.checkpoint.loading.strategies``.

Weights-Only Loading
--------------------

Load model weights, discard optimizer/scheduler state:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
       strict: false

**Use cases:** Fine-tuning pretrained models, composing inside a larger
pipeline where another stage owns training-progress state

Transfer-Learning Loading
-------------------------

Flexible loading with mismatch handling. Keys missing in the target or
with mismatched shapes are filtered out rather than raising:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
       skip_mismatched: true

Set ``skip_mismatched: false`` to raise ``CheckpointIncompatibleError``
on a shape mismatch instead of skipping it.

**Use cases:** Loading from different architectures, partial model
loading

Warm Start
----------

Resume training with full state restoration (weights, optimizer,
scheduler, epoch and global step). Uses ``strict=True`` because an exact
architecture match is expected:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loading.strategies.WarmStartLoader

**Use cases:** Resume interrupted training, continue from checkpoint

Cold Start
----------

Fresh training from pretrained weights. Loads weights like
``WeightsOnlyLoader``, then resets ``epoch`` and ``global_step`` to zero
and records ``pretrained_from`` provenance in the context metadata:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
       strict: false

**Use cases:** New task with pretrained backbone

Model Modifiers
===============

Modifiers transform the model after checkpoint loading.

.. note::

   Modifiers are not yet wired into the checkpoint pipeline. The model
   modifier system itself lives in ``anemoi.training.train.modify``
   (e.g. ``FreezingModelModifier``, PR #410 / #442); integrating it as
   pipeline stages is Phase 3 (issue #495). The ``_target_`` below is
   illustrative.

Parameter Freezing (Planned)
----------------------------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.modifiers.FreezingModifier
       layers: [encoder, processor.0]

*******************
 Complete Examples
*******************

Simple Pipeline
===============

.. code:: yaml

   training:
     checkpoint_pipeline:
       stages:
         - _target_: my_module.MySource
           path: /pretrained/model.ckpt

         - _target_: my_module.MyLoader
           strict: false

       async_execution: true

Custom Stage Implementation
===========================

.. code:: python

   from anemoi.training.checkpoint import PipelineStage, CheckpointContext
   import torch


   class MyLoader(PipelineStage):
       """Custom checkpoint loader."""

       def __init__(self, strict: bool = True):
           self.strict = strict

       async def process(self, context: CheckpointContext) -> CheckpointContext:
           if context.checkpoint_data and context.model:
               state_dict = context.checkpoint_data.get("state_dict", {})
               context.model.load_state_dict(state_dict, strict=self.strict)
               context.update_metadata(loading_strategy="custom", strict=self.strict)
           return context

.. tip::

   The example reads ``state_dict`` directly for brevity. For real
   checkpoints, use ``extract_state_dict`` from
   ``anemoi.training.checkpoint.formats`` to handle the various
   Lightning / PyTorch / raw ``state_dict`` shapes robustly.

Then use in configuration:

.. code:: yaml

   stages:
     - _target_: my_module.MyLoader
       strict: false

*************************************
 Migration from Legacy Configuration
*************************************

The checkpoint pipeline replaces several legacy configuration options:

.. list-table:: Legacy to Modern Migration
   :header-rows: 1

   -  -  Legacy Setting
      -  Modern Equivalent
      -  Notes

   -  -  ``load_weights_only: true``
      -  ``WeightsOnlyLoader`` stage
      -  More flexible with strict parameter

   -  -  ``transfer_learning: true``
      -  ``TransferLearningLoader`` stage
      -  Better mismatch handling

   -  -  ``resume_from_checkpoint: path``
      -  Source + ``WarmStartLoader`` stages
      -  Supports multiple sources

   -  -  ``submodules_to_freeze: [...]``
      -  ``FreezingModifier`` stage
      -  More modifier types available

****************
 Best Practices
****************

**Performance:**

-  Use ``async_execution: true`` for better I/O performance
-  Cache remote checkpoints locally when possible

**Reliability:**

-  Set reasonable timeouts for remote sources
-  Use retry logic for network operations

**Development:**

-  Start with simple configurations

-  Enable debug logging for troubleshooting:

   .. code:: python

      import logging
      logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

**********************
 Pipeline Composition
**********************

**Recommended ordering** (a convention, not enforced by the pipeline):
source stages first, then a loader stage, then any modifier stages.

.. code:: yaml

   stages:
     # 1. Source - fetch checkpoint
     - _target_: my_module.LocalSource
       path: /checkpoint.ckpt

     # 2. Loader - apply to model
     - _target_: my_module.WeightsOnlyLoader
       strict: false

     # 3. Modifier - transform model
     - _target_: my_module.FreezingModifier
       layers: [encoder]

**What the pipeline actually validates.** Stage ordering is not checked,
but two validation hooks run automatically:

-  Before execution, ``CheckpointPipelineValidator`` checks the runtime
   environment (Python, PyTorch, optional dependencies) and the shape of
   the ``training.checkpoint`` config (that a ``source`` or ``loading``
   block is present and carries a ``_target_``).
-  After execution, ``validate_pipeline_health`` checks that every stage
   completed, that the model reports ``weights_initialized`` when a
   source stage ran, and that the context's fields are mutually coherent
   (optimizer implies model, scheduler implies optimizer, ``pl_module``
   implies a Lightning checkpoint format).

See :ref:`checkpoint_integration` for implementation details and
:ref:`checkpoint_troubleshooting` for common issues.
