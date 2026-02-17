.. _checkpoint_pipeline_configuration:

###################################
 Checkpoint Pipeline Configuration
###################################

This guide covers configuration of the checkpoint pipeline system.

**********
 Overview
**********

The checkpoint pipeline provides a composable system for:

-  **Loading checkpoints** from various sources
-  **Applying loading strategies** (weights-only, transfer learning,
   warm/cold start)
-  **Modifying models** after loading (freezing, adapters)

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

   Source implementations (LocalSource, S3Source, HTTPSource) are
   planned for Phase 2. The configuration patterns below show the
   intended API.

Local Files (Planned)
---------------------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.LocalSource
       path: /path/to/checkpoint.ckpt

Amazon S3 (Planned)
-------------------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.S3Source
       bucket: my-models
       key: checkpoints/model-v1.ckpt
       region: us-east-1

HTTP/HTTPS (Planned)
--------------------

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.sources.HTTPSource
       url: https://models.example.com/checkpoint.ckpt

Loading Strategies
==================

Strategies define how to apply checkpoint data to your model.

.. note::

   Loading strategy implementations are planned for Phase 2.

Weights-Only Loading (Planned)
------------------------------

Load model weights, discard optimizer/scheduler state:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loaders.WeightsOnlyLoader
       strict: false

**Use cases:** Fine-tuning pretrained models, fresh training with
pretrained weights

Transfer Learning (Planned)
---------------------------

Flexible loading with mismatch handling:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loaders.TransferLearningLoader
       strict: false
       skip_mismatched: true

**Use cases:** Loading from different architectures, partial model
loading

Warm Start (Planned)
--------------------

Resume training with full state restoration:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loaders.WarmStartLoader

**Use cases:** Resume interrupted training, continue from checkpoint

Cold Start (Planned)
--------------------

Fresh training from pretrained weights:

.. code:: yaml

   stages:
     - _target_: anemoi.training.checkpoint.loaders.ColdStartLoader
       strict: false
       reset_layers: [classifier]

**Use cases:** New task with pretrained backbone

Model Modifiers
===============

Modifiers transform the model after checkpoint loading.

.. note::

   Modifier implementations are planned for Phase 3 (integration with PR
   #410).

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

The pipeline validates stage composition and warns about common issues:

-  Source stages should come before loader stages
-  Modifier stages should come after loader stages
-  Multiple loaders may conflict

Example of well-composed pipeline:

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

See :ref:`checkpoint_integration` for implementation details and
:ref:`checkpoint_troubleshooting` for common issues.
