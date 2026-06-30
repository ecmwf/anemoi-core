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

   CheckpointConfigError: Invalid checkpoint pipeline composition:
     - a modifier stage at position 0 comes before a loader stage at position 1; ...

**What it means:** the stages are in an impossible order. The pipeline enforces
**source → loading → modifiers** and refuses three combinations:

-  a source after a loader,
-  a modifier before a loader,
-  more than one loading strategy.

**How to fix it:** if you configure via ``training.checkpoint`` the builder
always produces a valid order, so this almost always means a hand-assembled
pipeline in code. Reorder the stages, or remove the extra loading strategy
(there can be only one).

Warm start with a remote source
===============================

**Message** (example):

.. code:: text

   CheckpointConfigError: Warm start restores optimizer and epoch state via
   Lightning's ckpt_path, which requires a checkpoint reachable as a local file.
   The configured training.checkpoint.source (S3Source) does not provide one ...

**What it means:** ``WarmStartLoader`` resumes the optimizer and epoch through
PyTorch Lightning, which reads the checkpoint from disk. ``S3Source`` and
``HTTPSource`` do not provide a local file, so the resume could not happen.
Anemoi refuses rather than silently dropping your optimizer/epoch state.

**How to fix it:** either

-  use ``LocalSource`` (an explicit ``path``) or ``RunSource`` (a run id) for
   warm start, or
-  download the remote checkpoint to a local file first and point ``LocalSource``
   at it, or
-  if you only need the *weights* (not the optimizer/epoch), switch
   ``training.checkpoint.loading`` to ``WeightsOnlyLoader`` /
   ``TransferLearningLoader`` / ``ColdStartLoader``, which work with any source.

Could not build a stage from config
===================================

**Message:**

.. code:: text

   CheckpointConfigError: Failed to instantiate pipeline stage ... from configuration

**What it means:** Hydra could not create one of your stages from its
``_target_``. Usually the ``_target_`` path is wrong, a required parameter is
missing, or the target module fails to import.

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

Warm start architecture mismatch
================================

**Message** (example):

.. code:: text

   CheckpointIncompatibleError: WarmStart requires exact model match: ...

**What it means:** ``WarmStartLoader`` requires the resumed model to match the
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

Remote downloads need an extra package
======================================

**Message:**

.. code:: text

   ImportError: aiohttp is required for remote checkpoint downloads.
   Install with: pip install anemoi-training[remote]

**How to fix it:** install the remote extra for HTTP(S):

.. code:: bash

   pip install anemoi-training[remote]

For S3, install the S3 extra instead:

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

   result = await pipeline.execute(context)
   for key, value in result.metadata.items():
       if key.startswith("stage_"):
           print(key, "->", value)

Test one stage in isolation
===========================

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext

   context = CheckpointContext(model=my_model)
   try:
       result = await my_stage.process(context)
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
