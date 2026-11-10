.. _checkpoint_troubleshooting:

#####################################
 Checkpoint Pipeline Troubleshooting
#####################################

Something went wrong loading a checkpoint? This page lists the errors you are
likely to see, in plain language, with the fix for each. Each entry shows the
**message**, **what it means**, and **how to fix it**.

For how the pipeline is meant to be configured, see
:ref:`checkpoint_pipeline_configuration`.

.. contents:: On this page
   :local:
   :depth: 2

*******************
 Quick diagnostics
*******************

See what is available
=====================

List the built-in stages you can use:

.. code:: python

   from anemoi.training.checkpoint import ComponentCatalog

   print("Sources:  ", ComponentCatalog.list_sources())
   print("Loaders:  ", ComponentCatalog.list_loaders())
   print("Modifiers:", ComponentCatalog.list_modifiers())

(Custom stages in your own package will not appear here, but still work — see
:ref:`cp_extensibility`.)

Look inside a checkpoint
========================

Inspect a file without fully loading it:

.. code:: python

   from pathlib import Path
   from anemoi.training.checkpoint import get_checkpoint_metadata

   metadata = get_checkpoint_metadata(Path("model.ckpt"))
   print(metadata)

Turn on detailed logging
========================

The single most useful debugging step:

.. code:: python

   import logging
   logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

*********************************
 Configuration errors
*********************************

"... has been removed" (an old setting)
=======================================

**Message** (example):

.. code:: text

   ValueError: training.load_weights_only has been removed. Set
   training.checkpoint.source to the run or file to load (...) and
   training.checkpoint.loading to {_target_: ...WeightsOnlyLoader}.

**What it means:** the older checkpoint settings have been replaced by the
``training.checkpoint`` pipeline. The run stops at config load — even if the old
key is set to ``null``, and even if you turned config validation off — and the
message names the exact replacement.

**How to fix it:** remove the old key and use the modern equivalent. The full
mapping is in :ref:`checkpoint_pipeline_configuration` ("Migrating from the old
settings"). The removed keys are ``training.run_id``, ``training.fork_run_id``,
``system.input.warm_start``, ``training.load_weights_only``,
``training.transfer_learning``, and ``training.submodules_to_freeze``.

Invalid pipeline order
======================

**Message** (example):

.. code:: text

   CheckpointConfigError: Invalid checkpoint pipeline ['FreezingModifierStage',
   'WeightsOnlyLoader']: use the order source -> one loader -> modifiers. (at
   training.checkpoint)

**What it means:** the stages are in an impossible order. The pipeline enforces
**source → loading → modifiers** and refuses three combinations:

-  a source after a loader,
-  a modifier before a loader,
-  more than one loading strategy.

**How to fix it:** if you configure via ``training.checkpoint`` the builder
always produces a valid order, so this almost always means a hand-assembled
pipeline in code. Reorder the stages, or remove the extra loading strategy
(there can be only one).

Resume has nothing to resume from
=================================

**Messages** (examples):

.. code:: text

   CheckpointConfigError: Warm start resumes through Trainer.fit(ckpt_path=), which
   needs a checkpoint to read, but no training.checkpoint.source is configured. ...

   CheckpointConfigError: Resume configured, but RunIdSource resolved no checkpoint
   file to hand Trainer.fit(ckpt_path=). Check the training.checkpoint.source
   configuration (a RunIdSource needs run_id), or add a loading strategy if you
   meant to start fresh training state.

   CheckpointConfigError: RunIdSource requires a run_id:
   training.checkpoint.source.run_id is unset. ...

   CheckpointConfigError: S3Source cannot resume a run: Trainer.fit(ckpt_path=)
   needs a local file. Download the checkpoint and use LocalSource, or add a
   loading strategy (WeightsOnlyLoader / TransferLearningLoader / ColdStartLoader)
   to load only the weights.

   CheckpointNotFoundError: Checkpoint not found: <checkpoints.root>/<run_id>/last.ckpt

**What it means:** a ``training.checkpoint.source`` with no ``loading`` block,
or with ``WarmStartLoader``, is a resume. PyTorch Lightning reads the file the
source resolves and restores weights, optimizer, scheduler and loop progress in
one pass at ``fit()``; the pipeline loads nothing itself. The checkpoint runner
resolves the source to a local file before the pipeline runs. Each message
names the step that had nothing to hand over: no source at all, a source that
resolved no file (rank 0 raises, the other ranks defer to it), the
``RunIdSource`` preset used with its ``run_id`` left at ``null``, a remote
source, or a run checkpoint that does not exist.

Only ``LocalSource`` and ``RunIdSource`` can resume: they resolve an existing
path. ``S3Source`` and ``HTTPSource`` delete their download straight after
loading, so a resume from them is rejected when the pipeline is built (the
message names ``HTTPSource`` for an HTTP source). A ``RunIdSource`` whose
``<run_id>/last.ckpt`` is missing raises ``CheckpointNotFoundError`` on rank 0;
an unreadable one raises ``CheckpointLoadError``.

**How to fix it:** either

-  add the source: ``training/checkpoint/source=run
   +training.checkpoint.source.run_id=<id>`` for a run, or
   ``training/checkpoint/source=local +training.checkpoint.source.path=<file>``
   for a file, or
-  pass the run id the ``run`` preset ships as ``null``, or check that the run
   id and ``system.output.checkpoints.root`` point at an existing run, or
-  for a remote checkpoint, download it and use ``LocalSource``, or
-  if you meant to start fresh training state from those weights, add a loading
   strategy (``WeightsOnlyLoader`` / ``TransferLearningLoader`` /
   ``ColdStartLoader``).

Could not build a stage from config
===================================

**Message:**

.. code:: text

   CheckpointConfigError: Failed to instantiate pipeline stage ... from configuration

On a resume, the source is built on its own, so the messages are:

.. code:: text

   CheckpointConfigError: Failed to instantiate the resume checkpoint source '<target>': ...

   CheckpointConfigError: The resume checkpoint source '<target>' is not a
   CheckpointSource (got <type>) (at training.checkpoint.source)

**What it means:** Hydra could not create one of your stages from its
``_target_``, or (on a resume) the ``_target_`` is not a ``CheckpointSource``.
Usually the ``_target_`` path is wrong, a required parameter is missing, or the
target module fails to import.

**How to fix it:**

#. Double-check the ``_target_`` string against the configuration guide.

#. Confirm all required parameters are present (e.g. ``HTTPSource`` needs
   ``url``; ``LocalSource`` needs ``path`` for the everyday case).

#. Try importing it yourself to surface the real error:

   .. code:: python

      from my_module import MyStage   # does this raise?
      stage = MyStage(param="value")

Model trained on random weights (safety net)
============================================

**Message:**

.. code:: text

   CheckpointLoadError: A checkpoint source stage was configured but the model's
   weights were never loaded (weights_initialized is False). ... Refusing to
   proceed with random weights.

**What it means:** you configured a source (so you clearly intended to load
something), but no loading strategy actually applied weights. This guard stops
you from accidentally training a brand-new random model when you meant to start
from a checkpoint.

**How to fix it:** add a ``training.checkpoint.loading`` block (e.g.
``WeightsOnlyLoader``). If you wrote a custom loader, make sure it sets
``context.model.weights_initialized = True`` after loading.

*********************************
 Loading and compatibility errors
*********************************

Keys do not match (strict load)
===============================

**Message** (example):

.. code:: text

   CheckpointLoadError: ... Missing key(s) / Unexpected key(s) in state_dict

**What it means:** with ``strict: true`` the checkpoint's layer names must match
your model's exactly, and they do not.

**How to fix it:** set ``strict: false`` to tolerate small differences, or use
``TransferLearningLoader`` to load only the parts that fit:

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

To see exactly which keys differ:

.. code:: python

   import torch
   from anemoi.training.checkpoint import compare_state_dicts

   checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")
   checkpoint_dict = checkpoint.get("state_dict", checkpoint)
   missing, unexpected, mismatched = compare_state_dicts(checkpoint_dict, model.state_dict())
   print("Missing:   ", missing)
   print("Unexpected:", unexpected)
   print("Mismatched:", mismatched)

Shape mismatch with ``skip_mismatched: false``
==============================================

**Message** (example):

.. code:: text

   CheckpointIncompatibleError: Shape mismatches found and skip_mismatched=False: {...}

**What it means:** you used ``TransferLearningLoader`` with
``skip_mismatched: false``, and some layers have the right name but the wrong
shape.

**How to fix it:** set ``skip_mismatched: true`` to skip those layers (they will
be reported in the run metadata as ``skipped_params``), or adjust your model so
the shapes match.

Size mismatch on a weights-only or cold-start load
==================================================

**Message** (example):

.. code:: text

   CheckpointLoadError: ... size mismatch for model.model.decoder...: copying a param
   with shape torch.Size([100, ...]) from checkpoint, the shape in current model is
   torch.Size([90, ...]).

**What it means:** the checkpoint was trained on a different set of variables
from the dataset you are training on, so the variable-dependent layers have a
different width. ``strict: false`` does **not** cover this: PyTorch reports a
size mismatch whether or not ``strict`` is set, because ``strict`` governs the
key set rather than tensor shapes.

**How to fix it**, in order of likelihood:

#. It is the wrong checkpoint. Check the run id or path first; this error is
   the main thing that catches that mistake.

#. The reduction is deliberate (fine-tuning onto fewer variables, issue #838).
   Set both:

   .. code:: yaml

      training:
        allow_variable_subset: True     # relaxes the variable-order and units checks
        checkpoint:
          loading:
            _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
            skip_mismatched: true       # skips the shape-mismatched parameters

   ``ColdStartLoader`` takes the same option. The skipped parameters stay at
   their initialised values, are logged at WARNING, and are listed in the run
   metadata as ``skipped_params``. Read that list: nothing downstream will
   notice that those parts of the model are untrained.

#. You want the broader tolerance of ``TransferLearningLoader``, which also
   drops keys the current model does not have at all.

Warm start architecture mismatch
================================

**Message** (example):

.. code:: text

   RuntimeError: Error(s) in loading state_dict for ...:
       Missing key(s) in state_dict: ... Unexpected key(s) in state_dict: ...

**What it means:** on a resume (``WarmStartLoader``, or no ``loading`` block)
Lightning loads the state dict strictly, so the resumed model has to match the
checkpoint exactly (resuming assumes the same architecture).

**How to fix it:** resume only models that match the checkpoint. If the
architecture genuinely changed, you are not resuming — use
``TransferLearningLoader`` (to reuse what fits) or ``ColdStartLoader`` (to start
fresh from the weights) instead.

Cannot find the model weights in the file
=========================================

**Message** (example):

.. code:: text

   CheckpointValidationError: Cannot find model state in checkpoint

**What it means:** the file does not look like a checkpoint the loader
recognises (non-standard structure or unusual key names).

**How to fix it:** inspect and detect the format:

.. code:: python

   import torch
   from anemoi.training.checkpoint.formats import detect_checkpoint_format

   checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")
   print("Top-level keys:", list(checkpoint.keys())[:10])
   print("Detected format:", detect_checkpoint_format("checkpoint.ckpt"))

Checkpoint from before multi-dataset support
============================================

**Message:**

.. code:: text

   TypeError: Checkpoint hyper_parameters.data_indices is a single-dataset
   IndexCollection from a pre-multi-dataset anemoi-core, incompatible with the
   current loaders. Run `anemoi-models migration sync <checkpoint>` to upgrade it
   to the multi-dataset format, or re-export it with current anemoi-core.

**What it means:** the checkpoint was written by an anemoi version from before
multi-dataset support. The load applies the ``chunking_fix`` and
``trainable_edge_perm_fix`` migrations in memory, but it does not convert this
older ``data_indices`` format.

**How to fix it:**

#. Migrate the file on disk with ``anemoi-models migration sync <checkpoint>``,
   then run again. The command rewrites the checkpoint in place and leaves a
   full-size backup beside it, so check the disk space first.

#. For a remote source (``S3Source`` / ``HTTPSource``) there is no file to
   rewrite in place: download the checkpoint, migrate the copy, and load it
   with ``LocalSource`` (or upload the migrated copy).

*********************************
 Source and download errors
*********************************

File not found (local)
======================

**Message:**

.. code:: text

   CheckpointNotFoundError: Checkpoint not found: /path/to/checkpoint.ckpt

**How to fix it:**

.. code:: python

   from pathlib import Path
   p = Path("/path/to/checkpoint.ckpt")
   print("exists:", p.exists(), "| is file:", p.is_file(), "| parent:", p.parent.exists())

Use an absolute path, and remember that ``LocalSource`` does **not** understand
``s3://`` or ``https://`` strings — use the matching source for those.

S3 downloads need an extra package
==================================

**Message:**

.. code:: text

   CheckpointSourceError: S3Source requires anemoi-utils[s3]: pip install 'anemoi-utils[s3]'

**How to fix it:** install the S3 extra. HTTP(S) downloads need no extra package.

.. code:: bash

   pip install anemoi-training[s3]

Download failed (HTTP)
======================

**Message** (example):

.. code:: text

   CheckpointSourceError: HTTP 404 client error downloading checkpoint from https://...

**How to fix it:** check connectivity, then consider downloading once and using
a local source:

.. code:: bash

   curl -I https://host.example/model.ckpt          # is it reachable?
   wget https://host.example/model.ckpt -O model.ckpt

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: ./model.ckpt

S3 access denied
================

**Message** (example):

.. code:: text

   CheckpointSourceError: S3 download failed for s3://bucket/model.ckpt: <reason>

**How to fix it:** ``S3Source`` reads credentials and endpoints from your
anemoi-utils settings (``~/.config/anemoi/settings.toml``), not the training
config. Verify your access works, e.g.:

.. code:: bash

   aws s3 ls s3://bucket/

*********************************
 Memory errors
*********************************

Out of GPU memory while loading
===============================

**Message:**

.. code:: text

   RuntimeError: CUDA out of memory

**Notes:** sources load checkpoints onto **CPU** (``map_location="cpu"``), so the
load itself should not exhaust GPU memory — pressure usually comes from the model
already on the GPU. Use a weights-only strategy (no optimizer state to hold), and
free cached memory between steps:

.. code:: python

   import torch
   torch.cuda.empty_cache()

********************
 Advanced debugging
********************

Trace what each stage did
=========================

After a run, every stage records its status in the context metadata:

.. code:: python

   result = pipeline.execute(context)
   for key, value in result.metadata.items():
       if key.startswith("stage_"):
           print(key, "->", value)

Test one stage in isolation
===========================

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext

   context = CheckpointContext(model=my_model)
   try:
       result = my_stage.process(context)
       print("Stage succeeded")
   except Exception as e:
       import traceback
       print("Stage failed:", e)
       traceback.print_exc()

**************
 Getting help
**************

When reporting a checkpoint problem, please include:

#. the full error message and traceback,
#. your (sanitised) ``training.checkpoint`` config,
#. environment info and the available components:

   .. code:: python

      import sys, torch, anemoi.training
      from anemoi.training.checkpoint import ComponentCatalog

      print("Python:", sys.version)
      print("PyTorch:", torch.__version__)
      print("Anemoi Training:", anemoi.training.__version__)
      print("Sources:", ComponentCatalog.list_sources())
      print("Loaders:", ComponentCatalog.list_loaders())
      print("Modifiers:", ComponentCatalog.list_modifiers())
