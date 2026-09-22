.. _checkpoint_integration:

#################################
 Checkpoint Pipeline (Developer Guide)
#################################

This is the **developer-level companion** to
:ref:`checkpoint_pipeline_configuration`. The configuration guide tells you how
to *use* the checkpoint pipeline from YAML; this page explains the **Python
building blocks** underneath, so you can read the code, debug it, or write your
own stages.

If you just want to configure a run, you do not need this page — start with
:ref:`checkpoint_pipeline_configuration`.

.. contents:: On this page
   :local:
   :depth: 2

***************
 The big picture
***************

A checkpoint load is a small **assembly line** of *stages*. Each stage receives
a shared :class:`CheckpointContext`, changes it, and passes it on:

.. code:: text

   CheckpointContext  ──▶  source  ──▶  loading  ──▶  modifier(s)  ──▶  CheckpointContext
   (model in)              (fetch)      (apply)        (adjust)         (model out)

-  **Sources** put the checkpoint onto the context (``checkpoint_path`` /
   ``checkpoint_data``).
-  **Loading strategies** apply the checkpoint to ``context.model``.
-  **Modifiers** change ``context.model`` after loading.

All three are :class:`PipelineStage` subclasses with a single ``async process``
method. The :class:`CheckpointPipeline` runs them in order. In a training run,
the trainer assembles the pipeline from config via
:func:`~anemoi.training.checkpoint.builder.build_checkpoint_pipeline`.

**************
 Core classes
**************

CheckpointContext
=================

The object that carries state through the stages:

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext

   context = CheckpointContext(
       model=my_model,
       config=my_config,  # optional OmegaConf config
   )

   context.update_metadata(source="local", loaded=True)
   print(context.metadata)

Fields:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   -  -  Field
      -  Meaning

   -  -  ``model``
      -  The model being loaded into / modified. Set by you before execution.

   -  -  ``checkpoint_path``
      -  Local path to the checkpoint file, if there is one. A source sets this;
         the trainer reads it afterwards to hand to Lightning on a resume.

   -  -  ``checkpoint_data``
      -  The loaded checkpoint dictionary.

   -  -  ``checkpoint_format``
      -  Auto-detected format: ``"lightning"``, ``"pytorch"``, or
         ``"state_dict"``.

   -  -  ``metadata``
      -  A free-form dict stages use to record what they did (e.g.
         ``loading_strategy``, ``modifiers_applied``, per-stage status).

   -  -  ``config``
      -  Optional Hydra config, used by stages that read run settings.

PipelineStage
=============

The base class for everything that goes in the assembly line:

.. code:: python

   from anemoi.training.checkpoint import PipelineStage, CheckpointContext


   class MyCustomStage(PipelineStage):
       def __init__(self, param: str) -> None:
           self.param = param

       async def process(self, context: CheckpointContext) -> CheckpointContext:
           context.update_metadata(custom_param=self.param)
           return context

``process`` is ``async`` so stages can do I/O (downloads, large reads) without
blocking. A stage that needs no ``await`` is still written ``async`` — just
return the context.

In practice you usually subclass one of the three **layer base classes**
instead of ``PipelineStage`` directly, because that is how the pipeline knows
which station a stage belongs to (see :ref:`cp_dev_extending`):

-  ``anemoi.training.checkpoint.sources.base.CheckpointSource``
-  ``anemoi.training.checkpoint.loading.base.LoadingStrategy``
-  ``anemoi.training.checkpoint.modifiers.base.ModelModifier``

CheckpointPipeline
==================

Runs the stages in order:

.. code:: python

   from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext

   pipeline = CheckpointPipeline(
       stages=[source, loader, modifier],   # instances OR {_target_: ...} configs
       continue_on_error=False,
   )

   context = CheckpointContext(model=my_model)
   result = await pipeline.execute(context)        # async
   # or, from synchronous code:
   result = pipeline.execute_sync(context)         # manages the event loop for you

Two things happen automatically when you construct a pipeline:

#. **Stage instantiation.** Items that are dicts/``DictConfig`` with a
   ``_target_`` are instantiated via Hydra; already-built ``PipelineStage``
   objects pass through unchanged. So you can mix config and live objects.

#. **Composition validation.** The order is checked, and **invalid
   compositions raise** :class:`CheckpointConfigError` immediately:

   -  a source after a loader,
   -  a modifier before a loader,
   -  more than one loading strategy.

   Multiple sources, or a station with no companion, only log a warning/hint.
   Classification is by ``isinstance`` against the three layer base classes, so
   a correctly-subclassed custom stage is placed and validated like a built-in.

After execution, a safety check (``_verify_weights_loaded``) **raises**
:class:`CheckpointLoadError` if a source ran but the model never reported
``weights_initialized`` — this prevents silently training a randomly-initialised
model when ``continue_on_error=True`` swallowed a loading failure. A resume is
exempt: its resolve-only source marks the context, because the weights arrive
at ``Trainer.fit(ckpt_path=)`` rather than from a pipeline stage.

From configuration
==================

The trainer does not build pipelines by hand; it calls the builder:

.. code:: python

   from anemoi.training.checkpoint import build_checkpoint_pipeline

   pipeline = build_checkpoint_pipeline(cfg)   # reads cfg.training.checkpoint.*

The builder reads ``training.checkpoint.{source,loading,modifiers}`` and emits
stages in the fixed order source → loading → modifiers. Absent blocks
contribute nothing; an empty config yields a no-op pipeline. The builder also
accepts ``parent_run_server2server`` / ``fork_run_server2server`` keyword
arguments, which it merges **only** onto a ``RunIdSource`` config — this is how
the trainer injects cross-server resume/fork lineage that cannot be written
statically in YAML.

.. _cp_dev_warmstart:

***********************************
 How loading splits with Lightning
***********************************

This is the most important behaviour to understand when reading the loading
code: a checkpoint has exactly one loader.

**Resume.** When the configured loader declares ``restores_training_state``
(``WarmStartLoader``), or when no ``training.checkpoint.loading`` block is
configured at all, PyTorch Lightning is the loader. The builder
(:func:`~anemoi.training.checkpoint.builder.resumes_via_lightning`) wraps the
source in a ``ResolveOnlySource`` — the source's ``resolve`` step makes the
checkpoint reachable as a local file and publishes ``context.checkpoint_path``;
nothing is ``torch.load``-ed — emits no loading stage, and keeps the modifier
stages. The trainer reads the resolved path back from the executed context
(``AnemoiTrainer.last_checkpoint``) and hands it to
``trainer.fit(ckpt_path=...)``. Lightning then reads the file once, calls
``BaseTrainingModule.on_load_checkpoint`` on the dict it is about to load —
which is where
:func:`~anemoi.training.checkpoint.loading.base.apply_checkpoint_corrections`
runs for a resume — and restores weights, optimizer, scheduler and loop
progress together.

**Pipeline load.** Every other strategy (``WeightsOnlyLoader``,
``ColdStartLoader``, ``TransferLearningLoader``) applies the weights at
model-build time and runs the same corrections itself; the trainer withholds
``ckpt_path`` (``_skip_lightning_restore``) so Lightning does not load the file
a second time, and training state starts fresh.

Any source works for a resume. ``LocalSource`` and ``RunIdSource`` resolve to an
existing file (``~`` expanded, canonicalised); ``HTTPSource`` and ``S3Source``
download to a node-local temporary file, keep it, and register it on
``context.temporary_files``, which the trainer deletes once ``fit()`` returns
(``atexit`` is the backstop). A resume with no source at all is rejected at
build time by
:func:`~anemoi.training.checkpoint.builder.reject_unsupported_warm_start`.

Shared parity helpers
=====================

Every loading strategy applies the same set of "parity" steps so that a
checkpoint loads **identically** whether it goes through the pipeline (at build
time) or through Lightning's own ``on_load_checkpoint`` hook. These live as
module-level functions in ``anemoi.training.checkpoint.loading.base`` and are
the single home for that logic:

-  ``apply_checkpoint_corrections`` — the one entry point for the three steps
   that change the checkpoint, in one fixed order: format migrations, then the
   trainable-edge-permutation migration, then the processor refresh. The
   pipeline strategies call it (``LoadingStrategy._apply_corrections``) and so
   does ``on_load_checkpoint`` on a resume. It composes:

   -  ``apply_checkpoint_format_migrations`` — apply anemoi-models format
      migrations (e.g. chunking fix) if available; no-op otherwise.
   -  ``apply_trainable_edge_perm_migration`` — apply the model-dependent
      trainable-edge-permutation migration if available.
   -  ``refresh_checkpoint_processors`` — rebuild stale pre/post-processor
      weights from the current model when
      ``training.update_ds_stats_on_ckpt_load.{states,tendencies}`` is set.
-  ``preserve_anemoi_metadata`` — restore ``model._ckpt_model_name_to_index``
   from the checkpoint's ``data_indices`` (multi-dataset aware).
-  ``extract_checkpoint_variables_metadata`` — populate
   ``model._ckpt_variables_metadata``.
-  ``warn_on_hparams_divergence`` — warn if the checkpoint's stored model config
   differs from the current run's, since weight loading keeps the current
   architecture.

If you write a custom loader, call ``self._apply_corrections(context)`` and the
metadata helpers (the built-ins do) so your strategy stays consistent with the
rest of the system.

****************
 Error handling
****************

The checkpoint module exposes a small exception hierarchy:

.. code:: python

   from anemoi.training.checkpoint import (
       CheckpointError,             # base exception
       CheckpointNotFoundError,     # file not found
       CheckpointLoadError,         # loading failed
       CheckpointValidationError,   # validation failed
       CheckpointSourceError,       # source fetch failed
       CheckpointTimeoutError,      # operation timed out
       CheckpointConfigError,       # configuration error
       CheckpointIncompatibleError, # model/checkpoint mismatch
   )

   try:
       result = await pipeline.execute(context)
   except CheckpointNotFoundError as e:
       print(f"Checkpoint not found: {e.path}")
   except CheckpointLoadError as e:
       print(f"Failed to load: {e}")
   except CheckpointError as e:
       print(f"Checkpoint error: {e}")

Stage failures are wrapped with pipeline context (which stage, its index, the
total count) before being re-raised, so a traceback tells you exactly where in
the assembly line things went wrong.

*******************
 Utility functions
*******************

Format detection
================

.. code:: python

   from anemoi.training.checkpoint.formats import (
       detect_checkpoint_format,
       load_checkpoint,
       extract_state_dict,
   )

   fmt = detect_checkpoint_format("/path/to/checkpoint.ckpt")  # "lightning" | "pytorch" | "state_dict"
   data = load_checkpoint("/path/to/checkpoint.ckpt")
   state_dict = extract_state_dict(data)   # handles the various checkpoint shapes

Checkpoint utilities
===================

.. code:: python

   from pathlib import Path
   from anemoi.training.checkpoint import (
       get_checkpoint_metadata,
       validate_checkpoint,
       calculate_checksum,
       compare_state_dicts,
       estimate_checkpoint_memory,
       format_size,
   )

   metadata = get_checkpoint_metadata(Path("model.ckpt"))     # without loading the full file
   validate_checkpoint(checkpoint_data)
   checksum = calculate_checksum(Path("model.ckpt"), algorithm="sha256")
   missing, unexpected, mismatched = compare_state_dicts(source_dict, target_dict)
   print(format_size(estimate_checkpoint_memory(checkpoint_data)))   # e.g. "1.5 GB"

``compare_state_dicts`` is especially handy when debugging a shape mismatch: it
tells you exactly which keys are missing, unexpected, or the wrong shape.

.. _cp_dev_extending:

*********************
 Component discovery
*********************

``ComponentCatalog`` lists the built-in stages, for tooling and for humans who
want to see what is available:

.. code:: python

   from anemoi.training.checkpoint import ComponentCatalog

   ComponentCatalog.list_sources()     # e.g. ['local', 'run', 's3', 'http']
   ComponentCatalog.list_loaders()     # loading strategies
   ComponentCatalog.list_modifiers()   # model modifiers
   ComponentCatalog.get_source_target("local")  # full _target_ path

.. important::

   The catalog is **discovery-only**. It scans the built-in
   ``sources``/``loading``/``modifiers`` packages and caches the result. It is
   **not** consulted when the pipeline builds, validates, or runs. A custom stage
   that lives in your own package will not appear in these lists, yet works
   perfectly in the pipeline — because the pipeline classifies and runs stages by
   their base class and ``process`` method, never by catalog membership. You do
   not need to register anything.

******************************
 Writing your own stage (recap)
******************************

The configuration guide has a full worked example
(:ref:`cp_extensibility`). In short:

#. Subclass the base class for the station you want
   (``CheckpointSource`` / ``LoadingStrategy`` / ``ModelModifier``).
#. Implement ``async def process(self, context) -> CheckpointContext``.
#. Mutate the context (load weights, change the model, set metadata) and return
   it. A source sets ``checkpoint_path``/``checkpoint_data``; a loader sets
   ``context.model.weights_initialized = True``; a modifier appends to
   ``context.metadata["modifiers_applied"]``.
#. Reference it from config with ``_target_``, or pass an instance directly when
   building a pipeline in code.

************
 Next steps
************

-  :ref:`checkpoint_pipeline_configuration` — the configuration guide and the
   custom-stage walkthrough.
-  :ref:`checkpoint_troubleshooting` — error-by-error solutions.
