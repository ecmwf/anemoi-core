.. _checkpoint_pipeline_configuration:

###################################
 Checkpoint Pipeline Configuration
###################################

This guide covers configuration of the checkpoint pipeline system.

**********
 Overview
**********

The checkpoint pipeline provides a composable system for:

-  **Loading checkpoints** from various sources (local, HTTP, S3)
-  **Applying loading strategies** (weights-only, transfer learning,
   warm/cold start)
-  **Modifying models** after loading (freezing, adapters)

The trainer builds and runs this pipeline from the ``training.checkpoint``
configuration via
``anemoi.training.checkpoint.builder.build_checkpoint_pipeline``. Stage order
is fixed by the builder: source, then loading, then modifiers (in list
order). Every block is optional; an absent block contributes no stage.

*****************
 Basic Structure
*****************

The blessed namespace is ``training.checkpoint``, with three optional
blocks: ``source`` (a single Hydra ``_target_``), ``loading`` (a single
Hydra ``_target_``), and ``modifiers`` (a *list* of Hydra ``_target_``
stages applied in order after loading):

.. code:: yaml

   training:
     checkpoint:
       source: # OPTIONAL — single _target_ (acquisition stage)
         _target_: anemoi.training.checkpoint.sources.local.LocalSource

       loading: # OPTIONAL — single _target_ (loading strategy)
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

       modifiers: # OPTIONAL — list of _target_ stages, applied in order
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder, "processor.0"]
           strict: false
           validate_gradients: true

**Key points:**

-  ``source``, ``loading`` and ``modifiers`` are each optional; an absent
   block contributes no stage.
-  Stage order is fixed by the builder: source, then loading, then
   modifiers (in list order).

Easy path — Hydra group selection
==================================

The ``source``, ``loading`` and ``modifiers`` blocks are wired as opt-in
default groups in every shipped training preset. The default is ``null``
(no pipeline — a fresh run with no checkpoint loaded). Select a group to opt in:

.. code:: bash

   anemoi-training train training/checkpoint/loading=weights_only

   anemoi-training train \
     training/checkpoint/source=local \
     training/checkpoint/loading=transfer_learning

Available group options:

-  ``training/checkpoint/source`` = ``local`` | ``http`` | ``s3``
-  ``training/checkpoint/loading`` = ``weights_only`` |
   ``transfer_learning`` | ``warm_start`` | ``cold_start``
-  ``training/checkpoint/modifiers`` = ``freezing``

************************
 Configuration Sections
************************

Checkpoint Sources
==================

Sources define where to fetch checkpoints. Configure a single source as
the ``training.checkpoint.source`` block.

Local Files
-----------

``LocalSource`` resolves its path from the run lineage / pipeline context
rather than from a stage argument:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource

Amazon S3
---------

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.s3.S3Source
         url: s3://my-models/checkpoints/model-v1.ckpt

HTTP/HTTPS
----------

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.http.HTTPSource
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

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

**Use cases:** Fine-tuning pretrained models, composing inside a larger
pipeline where another stage owns training-progress state

Transfer-Learning Loading
-------------------------

Flexible loading with mismatch handling. Keys missing in the target or
with mismatched shapes are filtered out rather than raising:

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
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

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WarmStartLoader

**Use cases:** Resume interrupted training, continue from checkpoint

Cold Start
----------

Fresh training from pretrained weights. Loads weights like
``WeightsOnlyLoader``, then resets ``epoch`` and ``global_step`` to zero
and records ``pretrained_from`` provenance in the context metadata:

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
         strict: false

**Use cases:** New task with pretrained backbone

Model Modifiers
===============

Modifiers transform the model after checkpoint loading. The
``training.checkpoint.modifiers`` block is a *list* of stages, applied in
order after the loading strategy.

Parameter Freezing
------------------

``FreezingModifierStage`` freezes the named submodules (dot-paths) by
setting ``requires_grad=False``:

.. code:: yaml

   training:
     checkpoint:
       modifiers:
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder, "processor.0"]
           strict: false
           validate_gradients: true

-  ``submodules_to_freeze``: list of dot-paths to freeze.
-  ``strict``: raise if a named submodule is not found (default: false).
-  ``validate_gradients``: assert frozen parameters accumulate no
   gradient.

*******************
 Complete Examples
*******************

Simple Pipeline
===============

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource

       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

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

Then use it as the loading strategy in configuration:

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: my_module.MyLoader
         strict: false

*************************************
 Migration from Legacy Configuration
*************************************

The ``training.checkpoint`` surface replaces several legacy configuration
options. The legacy keys have been **removed**: each is rejected at config
validation with an error naming its replacement. Use the
``training.checkpoint`` surface for all checkpoint acquisition, loading and
model modification.

.. list-table:: Legacy to Modern Migration
   :header-rows: 1

   -  -  Removed Setting
      -  Modern Equivalent
      -  Notes

   -  -  ``training.load_weights_only: true``
      -  ``training.checkpoint.source`` + ``training.checkpoint.loading`` with
         ``WeightsOnlyLoader``
      -  Rejected at config validation; set the source to the run/file to load.

   -  -  ``training.transfer_learning: true``
      -  ``training.checkpoint.source`` + ``training.checkpoint.loading`` with
         ``TransferLearningLoader``
      -  Rejected at config validation (use ``skip_mismatched: true``).

   -  -  ``training.submodules_to_freeze: [...]``
      -  ``training.checkpoint.modifiers`` with ``FreezingModifierStage``
      -  Rejected at config validation.

   -  -  ``training.run_id`` / ``training.fork_run_id``
      -  ``training.checkpoint.source`` with ``RunSource`` (``fork: true`` to fork)
      -  Rejected at config validation.

   -  -  ``system.input.warm_start``
      -  ``training.checkpoint.source`` with ``LocalSource`` (explicit file)
      -  Rejected at config validation.

.. note::

   Resume / fork / warm-start are now expressed through the same
   ``training.checkpoint.source`` surface: a ``RunSource`` (``fork: false`` to
   resume, ``fork: true`` to fork) or a ``LocalSource`` (an explicit file). The
   source stage records the resolved checkpoint path, which the trainer hands to
   Lightning's ``ckpt_path`` for the warm-start full-state resume.

****************
 Best Practices
****************

**Performance:**

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

**Stage order** is fixed by the builder: the source runs first, then the
loading strategy, then any modifiers (in list order). You only declare the
blocks; the builder assembles the pipeline in this order.

.. code:: yaml

   training:
     checkpoint:
       # 1. Source — fetch checkpoint
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource

       # 2. Loading — apply to model
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

       # 3. Modifiers — transform model (applied in list order)
       modifiers:
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder]

**What the pipeline actually validates.** Stage order is set by the
builder rather than validated, but two validation hooks run
automatically:

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
