.. _checkpoint_pipeline_configuration:

###################################
 Checkpoint Pipeline Configuration
###################################

This is the everyday guide to telling Anemoi **which trained model to start
from, and how**. You do not need to be a software engineer to use it: most of
the time you copy one of the recipes below, change a path or a run id, and you
are done.

If you only read one thing, read :ref:`cp_recipes`.

.. contents:: On this page
   :local:
   :depth: 2

***********************
 What is a checkpoint?
***********************

A **checkpoint** is a saved snapshot of a model taken during training. It is a
single file (often called ``last.ckpt`` or similar) that contains:

-  the **weights** — the numbers the model has learned (this is "the model"),
-  the **optimizer state** — the bookkeeping the training algorithm uses to
   keep improving smoothly (think: momentum, running averages),
-  the **training progress** — which epoch and step the run had reached,
-  some **metadata** — e.g. which variables the model was trained on.

Starting a new training run *from* a checkpoint is extremely common. You might
want to:

-  pick up an interrupted run exactly where it stopped,
-  fine-tune a published model on your own data,
-  reuse a model's weights but start the training clock from zero,
-  freeze part of the model so it does not change while you train the rest.

The **checkpoint pipeline** is the one place where you describe what you want,
and Anemoi does the rest.

*****************************
 The pipeline in one picture
*****************************

Think of loading a checkpoint as a short **assembly line with three stations**.
A checkpoint travels left to right; each station is optional.

.. code:: text

   ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
   │  1. SOURCE  │ ──▶ │  2. LOADING  │ ──▶ │  3. MODIFIERS  │
   │  get the    │     │  apply it    │     │  adjust the    │
   │  checkpoint │     │  to the model│     │  model         │
   └─────────────┘     └──────────────┘     └───────────────┘
     where from?         which parts?          freeze? adapt?
     (file, run,         (weights only,        (one or more,
      S3, URL)            warm start, …)        in order)

-  **Station 1 — Source**: *where* the checkpoint comes from (a local file, a
   previous run, an S3 bucket, an HTTP URL).
-  **Station 2 — Loading strategy**: *how* the checkpoint is applied to your
   model (just the weights? a full resume? skip parts that do not fit?).
-  **Station 3 — Modifiers**: *what to change* about the model afterwards
   (e.g. freeze the encoder). This is a **list** — you can apply several, in
   order.

You configure these three stations under one place in your config:
``training.checkpoint``. You only declare the stations you want; Anemoi always
runs them in the fixed order **source → loading → modifiers**.

.. code:: yaml

   training:
     checkpoint:
       source: { ... }      # OPTIONAL — exactly one
       loading: { ... }     # OPTIONAL — exactly one
       modifiers: [ ... ]   # OPTIONAL — a list, applied in order

If you leave ``training.checkpoint`` out entirely (or set it to ``null``), you
get a **fresh run** with a randomly-initialised model — the normal "train from
scratch" behaviour.

.. _cp_recipes:

************************
 "I just want to…" recipes
************************

Copy a block, change the path/run id, and go. Each recipe shows the YAML; below
each one is the equivalent command line using Hydra "group selection" (see
:ref:`cp_turning_on`), which is often shorter.

Fine-tune from a pretrained checkpoint file
===========================================

Load the published weights, then train with a **fresh optimizer** starting at
epoch 0:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
         strict: false

Resume an interrupted run (continue exactly where it stopped)
=============================================================

Restore weights **and** the optimizer/epoch state so training continues
seamlessly. Use the run's id:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.run.RunSource
         run_id: a1b2c3d4-...        # the run you want to resume
         fork: false                 # false = resume the SAME run
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WarmStartLoader

.. code:: bash

   anemoi-training train \
     training/checkpoint/source=run +training.checkpoint.source.run_id=a1b2c3d4-... \
     training/checkpoint/loading=warm_start

Transfer-learn when the architecture changed a little
=====================================================

Load only the parts that still fit; skip layers that changed shape or no longer
exist, instead of crashing:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
         skip_mismatched: true

Freeze the encoder while fine-tuning
====================================

Load the weights, then freeze part of the model so it stays fixed during
training (add a modifier after any loading strategy):

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
         strict: false
       modifiers:
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder]

Load weights from S3 or a URL
=============================

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.s3.S3Source
         url: s3://my-bucket/path/to/model.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

Replace ``S3Source`` + ``url: s3://…`` with ``HTTPSource`` + ``url:
https://…`` to download over HTTP(S).

.. note::

   **Warm start (resume) needs a local file.** ``WarmStartLoader`` can only be
   paired with ``LocalSource`` or ``RunSource``, because resuming the optimizer
   and epoch state is handled by PyTorch Lightning, which needs the checkpoint
   as a file on disk. Pairing warm start with ``S3Source``/``HTTPSource`` is
   rejected with a clear error (see :ref:`cp_warmstart_ownership`). To use a
   remote checkpoint, either download it first to a local path, or use a
   weights-only / transfer-learning / cold-start strategy.

Start a fresh run (no checkpoint)
=================================

Just leave ``training.checkpoint`` unset (the default). Nothing to configure.

*****************************
 The vocabulary (plain words)
*****************************

.. list-table::
   :header-rows: 1
   :widths: 22 78

   -  -  Term
      -  What it means

   -  -  **Checkpoint**
      -  A saved snapshot file of a model (weights + optimizer + progress +
         metadata).

   -  -  **Weights**
      -  The learned numbers — effectively "the model". Loading weights makes
         your model behave like the saved one.

   -  -  **Optimizer state**
      -  The training algorithm's bookkeeping. You need it to continue training
         smoothly; you do *not* need it to just reuse a model's knowledge.

   -  -  **Epoch / step**
      -  How far a run had progressed. Resuming keeps these; starting fresh
         resets them to 0.

   -  -  **Source**
      -  Station 1 — *where* the checkpoint is fetched from.

   -  -  **Loading strategy**
      -  Station 2 — *how* the checkpoint is applied to your model.

   -  -  **Modifier**
      -  Station 3 — a post-loading change to the model (e.g. freezing).

   -  -  **Stage**
      -  The general word for any one station (a source, a loading strategy, or
         a modifier). They are the building blocks the pipeline runs in order.

   -  -  ``_target_``
      -  A Hydra setting that names the exact Python class to use, e.g.
         ``anemoi.training.checkpoint.sources.local.LocalSource``. You can think
         of it as "which tool to put in this station".

   -  -  **Run / run id**
      -  A previous training run, identified by a unique id (from the
         experiment tracker, e.g. MLflow).

.. _cp_turning_on:

**********************
 Two ways to turn it on
**********************

There are two equivalent ways to fill the three stations. They produce the same
result — pick whichever you find clearer.

Easy path — Hydra group selection
==================================

Every shipped training preset wires the three stations as **opt-in groups**
that default to ``null`` (no checkpoint = fresh run). Select a ready-made group
on the command line to switch a station on:

.. code:: bash

   # Just load weights from a local checkpoint
   anemoi-training train training/checkpoint/source=local \
       +training.checkpoint.source.path=/path/to/model.ckpt \
       training/checkpoint/loading=weights_only

   # Resume a run, restoring optimizer + epoch
   anemoi-training train \
       training/checkpoint/source=run +training.checkpoint.source.run_id=<id> \
       training/checkpoint/loading=warm_start

The available groups are:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   -  -  Group selector
      -  Options

   -  -  ``training/checkpoint/source=``
      -  ``local`` · ``run`` · ``s3`` · ``http``

   -  -  ``training/checkpoint/loading=``
      -  ``weights_only`` · ``transfer_learning`` · ``warm_start`` ·
         ``cold_start``

   -  -  ``training/checkpoint/modifiers=``
      -  ``freezing``

Explicit path — write the blocks in YAML
========================================

Or write the three blocks directly into your config, as in every recipe above.
This is clearer when you want to set several parameters at once or keep the
whole pipeline in one file. Each block is a Hydra ``_target_`` plus its
parameters.

************************************************
 Station 1 — Sources (where the checkpoint comes from)
************************************************

A source fetches the checkpoint and hands it to the next station. Configure
**exactly one** under ``training.checkpoint.source``.

.. list-table:: Which source?
   :header-rows: 1
   :widths: 18 30 52

   -  -  Source
      -  Use it when…
      -  Key parameters

   -  -  ``LocalSource``
      -  the checkpoint is a file on disk you can point to
      -  ``path`` (the file path)

   -  -  ``RunSource``
      -  you want to resume or fork a previous **run** by its id
      -  ``run_id``, ``fork``

   -  -  ``S3Source``
      -  the checkpoint lives in an S3-compatible bucket
      -  ``url`` (``s3://bucket/key``)

   -  -  ``HTTPSource``
      -  the checkpoint is downloadable over HTTP(S)
      -  ``url``, ``expected_checksum`` (optional)

.. important::

   There is **no automatic detection** of where a checkpoint lives. You choose
   the source explicitly. For example, giving ``LocalSource`` an ``s3://…``
   string will *not* fetch from S3 — it will look for a local file literally
   named that and fail. Pick the source that matches your checkpoint's location.

Local files
===========

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/checkpoint.ckpt

``path`` understands ``~`` (your home directory) and follows symbolic links. If
the file is not there you get a clear ``CheckpointNotFoundError``.

.. tip::

   ``path`` is optional. If you omit it, ``LocalSource`` uses a path that an
   earlier step put on the pipeline (this is how :class:`RunSource` works
   internally). For everyday "load this file" use, **set** ``path``.

Previous runs (resume or fork)
==============================

``RunSource`` finds the checkpoint belonging to a previous run and loads it.
The single ``fork`` flag chooses what kind of continuation you want:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.run.RunSource
         run_id: a1b2c3d4-...
         fork: false   # false = resume the SAME run; true = start a NEW run from these weights

-  ``fork: false`` (**resume**) — keep the same run identity. Combined with
   ``WarmStartLoader``, training picks up exactly where it left off.
-  ``fork: true`` (**fork**) — mint a *new* run that starts from the old run's
   weights. Use this to branch an experiment without overwriting the original.

See :ref:`cp_resume_fork` for the full story, including how this replaces the
old ``run_id`` / ``fork_run_id`` settings.

Amazon S3 (and S3-compatible storage)
=====================================

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.s3.S3Source
         url: s3://my-bucket/path/to/model.ckpt

Credentials and endpoint settings are read from your anemoi-utils settings
(``~/.config/anemoi/settings.toml``), not from the training config. This needs
the S3 extra installed (``anemoi-training[s3]`` / ``anemoi-utils[s3]``).

HTTP / HTTPS
============

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.http.HTTPSource
         url: https://host.example/path/to/checkpoint.ckpt
         expected_checksum: null   # optional sha256; if set, the download is verified

.. list-table:: ``HTTPSource`` parameters
   :header-rows: 1
   :widths: 26 14 60

   -  -  Parameter
      -  Default
      -  Meaning

   -  -  ``url``
      -  *(required)*
      -  The ``http://`` or ``https://`` URL. Checked immediately when the
         config loads.

   -  -  ``max_retries``
      -  ``3``
      -  How many times to retry the download before giving up.

   -  -  ``timeout``
      -  ``300``
      -  Per-attempt timeout, in seconds.

   -  -  ``expected_checksum``
      -  ``None``
      -  Optional SHA-256 checksum. If set, the downloaded file is verified and
         a mismatch raises an error. If left unset, integrity is not checked.

*********************************************
 Station 2 — Loading strategies (how it is applied)
*********************************************

A loading strategy decides *how much* of the checkpoint to apply. Configure
**exactly one** under ``training.checkpoint.loading``.

.. list-table:: Which loading strategy?
   :header-rows: 1
   :widths: 22 40 38

   -  -  Strategy
      -  Use it when…
      -  Optimizer / epoch

   -  -  ``WeightsOnlyLoader``
      -  you want the weights and nothing else, and another part of your setup
         owns training progress
      -  discarded; progress left untouched

   -  -  ``ColdStartLoader``
      -  you want to **fine-tune from pretrained weights** as a brand-new run
      -  discarded; progress **reset to 0**

   -  -  ``TransferLearningLoader``
      -  the architecture changed and some layers no longer fit
      -  discarded; only matching weights loaded

   -  -  ``WarmStartLoader``
      -  you are **resuming** the same run and want it to continue seamlessly
      -  **restored** (by Lightning — see below)

Weights-only
============

Loads the weights and explicitly throws away optimizer/scheduler state. It does
**not** touch training-progress numbers (epoch, step) — that is left to whatever
else is configured. It is the simplest building block, ideal when composing
larger pipelines.

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false   # false = tolerate small key differences; true = require an exact match

For the common "fine-tune from pretrained" case, prefer **cold start** below,
which is weights-only *plus* an explicit reset of the training clock.

Cold start
==========

Fresh training **from** pretrained weights. It loads weights exactly like
weights-only, then resets ``epoch`` and ``global_step`` to ``0`` and records
where the weights came from (``pretrained_from``) so loggers and checkpoint
names can trace the lineage.

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
         strict: false

Transfer learning
=================

Loads only the weights that are **compatible** with your current model — same
name and same shape — and skips the rest, instead of failing. Useful when you
changed the architecture (added variables, resized a layer) but still want to
reuse what you can.

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
         skip_mismatched: true

-  ``skip_mismatched: true`` (default) — quietly skip layers whose shapes do
   not match, and report which were transferred vs. skipped in the run metadata.
-  ``skip_mismatched: false`` — treat a shape mismatch as an error
   (``CheckpointIncompatibleError``) instead of skipping it.

The original checkpoint file is never modified; the filtering builds a new,
in-memory subset. To also **freeze** the transferred layers, add a freezing
modifier after this loader (Station 3).

Warm start
==========

Resume an interrupted run so it continues exactly where it stopped. It loads
weights with an **exact-match requirement** (resuming assumes the same
architecture) and applies the compatibility steps. It takes no parameters.

.. code:: yaml

   training:
     checkpoint:
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WarmStartLoader

.. _cp_warmstart_ownership:

Who restores what (the warm-start ownership rule)
-------------------------------------------------

This is the one subtlety worth understanding, because it explains an error you
might hit.

When you resume, two different things need restoring:

#. the **weights** — done by the checkpoint pipeline, at the moment the model is
   built;
#. the **optimizer, scheduler, and training clock** (epoch/step) — done by
   **PyTorch Lightning**, when ``trainer.fit()`` starts.

Why the split? The pipeline runs early, while the model is being constructed —
before the optimizer and the training loop even exist, so it *cannot* restore
them. Lightning restores them later, from the same checkpoint file, through its
built-in ``ckpt_path`` mechanism.

Two consequences for you:

-  **Only ``WarmStartLoader`` triggers the Lightning restore.** The other
   strategies start with a fresh optimizer and epoch 0. (Internally each loader
   declares this via a ``restores_training_state`` flag that the trainer reads.)
-  **Warm start needs a local file.** Lightning's restore reads the checkpoint
   from disk, so warm start only works with ``LocalSource`` or ``RunSource``.
   Configuring it with ``S3Source`` or ``HTTPSource`` is rejected up front with
   an explanatory error, rather than silently dropping your optimizer and epoch
   state.

.. note::

   Warm start also records the checkpoint's epoch/step onto the run metadata for
   inspection (via a small ``TrainingState`` record). That is informational
   only — the value that actually drives the resumed run is Lightning's restore,
   not this metadata.

*********************************************
 Station 3 — Modifiers (adjust the model after loading)
*********************************************

Modifiers change the model *after* the weights are loaded. Unlike the other two
stations, ``training.checkpoint.modifiers`` is a **list** — you can apply
several, and they run **in the order you list them**.

Freezing
========

``FreezingModifierStage`` freezes named parts of the model so their weights stay
fixed during training (it sets ``requires_grad=False``):

.. code:: yaml

   training:
     checkpoint:
       modifiers:
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: [encoder, "processor.0"]
           strict: false
           validate_gradients: true

.. list-table:: ``FreezingModifierStage`` parameters
   :header-rows: 1
   :widths: 26 14 60

   -  -  Parameter
      -  Default
      -  Meaning

   -  -  ``submodules_to_freeze``
      -  ``[]``
      -  Names of the parts to freeze, in **dot notation**. ``encoder`` freezes
         the submodule called ``encoder``; ``processor.0`` reaches one level
         deeper. Names are exact — there are no wildcards.

   -  -  ``strict``
      -  ``false``
      -  If ``true``, a name that does not exist raises an error. If ``false``,
         it logs a warning and continues with the rest.

   -  -  ``validate_gradients``
      -  ``true``
      -  After freezing, double-check that the frozen parts really are frozen
         (by inspecting their ``requires_grad`` flags) and warn if not.

.. tip::

   To find the right names, print your model's submodules
   (``print(dict(model.named_modules()).keys())``) and copy the dotted paths.

Freezing is the only modifier shipped today. Adding more (adapters, LoRA,
quantisation) is exactly the extensibility story in the next section.

.. _cp_resume_fork:

***************************************
 Resume, fork, and warm start in detail
***************************************

"Resume / fork / warm start" used to be a handful of separate, easy-to-confuse
settings. They are now all expressed through **the source + loading stations**,
which makes the intent explicit:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   -  -  Goal
      -  Source
      -  Loading

   -  -  Continue the same run seamlessly
      -  ``RunSource`` with ``fork: false``
      -  ``WarmStartLoader``

   -  -  Branch a new run from an old one's weights, fresh optimizer
      -  ``RunSource`` with ``fork: true``
      -  ``ColdStartLoader`` (or ``WeightsOnlyLoader``)

   -  -  Start from a specific checkpoint file
      -  ``LocalSource`` with ``path``
      -  any loading strategy

Behind the scenes, choosing ``RunSource`` also sets the run identity the
experiment tracker and output paths expect (resume keeps the run id; fork mints
a new one), so the behaviour matches the old ``run_id`` / ``fork_run_id`` flow
exactly — you just express it in one consistent place now.

.. _cp_extensibility:

***************************************
 Write your own stage (the powerful part)
***************************************

Here is what makes the pipeline genuinely flexible: **every station is open**.
The four sources, four loading strategies, and the freezing modifier that ship
with Anemoi are not special — they are simply the built-in examples of three
open "shapes". You can write your own and drop it into any station, and it
behaves *exactly* like a built-in.

Three things make this work, and they are worth knowing because they remove a
lot of would-be confusion:

#. **A stage is recognised by what it *is*, not where it lives or what it is
   named.** The pipeline classifies a stage as a source, a loader, or a
   modifier by checking which base class it inherits from — not by its filename,
   class name, or package. A loader you write in your own project, called
   anything you like, is treated identically to ``WarmStartLoader`` as long as
   it subclasses the loading base class.

#. **You can supply a stage as config *or* as a ready-made object.** In a normal
   training run you point ``_target_`` at your class (config). In a script or a
   test you can hand the pipeline an already-built Python object. Both work.

#. **You do not have to "register" anything.** There is a discovery helper
   (``ComponentCatalog``) that *lists* the built-ins for convenience, but the
   pipeline never consults it to run. Nothing breaks if your class is not in it.

The minimum contract
====================

Pick the base class for the station you are filling, write one ``async
process`` method, change the context, and return it:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   -  -  Station
      -  Subclass
      -  Your ``process`` should…

   -  -  Source
      -  ``...checkpoint.sources.base.CheckpointSource``
      -  set ``context.checkpoint_path`` / ``context.checkpoint_data``

   -  -  Loading
      -  ``...checkpoint.loading.base.LoadingStrategy``
      -  load weights into ``context.model`` and set
         ``context.model.weights_initialized = True``

   -  -  Modifier
      -  ``...checkpoint.modifiers.base.ModelModifier``
      -  change ``context.model`` and append a record to
         ``context.metadata["modifiers_applied"]``

A complete custom modifier
==========================

This modifier scales the weights of named submodules — something not built in.
It lives in *your* project, e.g. ``my_project/modifiers.py``:

.. code:: python

   from anemoi.training.checkpoint.modifiers.base import ModelModifier
   from anemoi.training.checkpoint import CheckpointContext


   class WeightScalingModifier(ModelModifier):
       """Multiply the weights of the named submodules by a constant."""

       def __init__(self, submodules: list[str], factor: float = 0.5) -> None:
           self.submodules = submodules
           self.factor = factor

       async def process(self, context: CheckpointContext) -> CheckpointContext:
           for name in self.submodules:
               module = context.model.get_submodule(name)
               for param in module.parameters():
                   param.data.mul_(self.factor)
           # Append (do not overwrite) so several modifiers can compose.
           context.metadata.setdefault("modifiers_applied", []).append(
               {"type": "weight_scaling", "submodules": self.submodules, "factor": self.factor}
           )
           return context

Use it exactly like a built-in — point ``_target_`` at it:

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false
       modifiers:
         - _target_: my_project.modifiers.WeightScalingModifier
           submodules: [decoder]
           factor: 0.5

The same recipe works for a custom **source** (subclass ``CheckpointSource``) or
custom **loading strategy** (subclass ``LoadingStrategy``). Because the pipeline
identifies stages by their base class, your class slots into the correct station
and the ordering checks below protect it just like a built-in.

.. tip::

   When writing a custom loader, reuse the shared "parity" helpers in
   ``anemoi.training.checkpoint.loading.base`` (format migrations, processor
   refresh, metadata preservation). The built-in strategies call them so that a
   checkpoint loads identically whether through the pipeline or through
   Lightning. See :ref:`checkpoint_integration` for the developer-level details.

.. note::

   A training run can only take stages **as config** (``_target_`` strings) —
   Hydra cannot serialise a live Python object into the config tree. Handing the
   pipeline a ready-made object is for scripts and tests, where you build the
   pipeline yourself.

*****************************************
 What the pipeline checks for you
*****************************************

You do not have to assemble the stations in the right order — when you use the
config, Anemoi always builds them **source → loading → modifiers**. On top of
that, the pipeline guards against mistakes. Some problems are **errors** (they
stop the run with a clear message), and some are just **warnings**.

**Errors** (raise ``CheckpointConfigError`` when the pipeline is built):

-  a source placed *after* a loader,
-  a modifier placed *before* a loader,
-  more than one loading strategy.

These only happen if you assemble a pipeline by hand in code; the config builder
always produces a valid order.

**A safety net at the end** (raises ``CheckpointLoadError``): if a source was
configured but the model's weights were never actually loaded, the run refuses
to continue rather than silently training a random model.

**Warnings / hints** (logged, never fatal): more than one source, or a station
that looks like it is missing a companion (e.g. a loader with no source).

There are also lightweight pre-run checks (Python/PyTorch versions, that your
``training.checkpoint`` block has a ``_target_``) and a post-run health check.
These record their findings but do not stop a run on their own.

.. tip::

   Seeing more detail helps when something is off. Turn on debug logging:

   .. code:: python

      import logging
      logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

*************************************
 Migrating from the old settings
*************************************

The older, separate settings have been **removed**. If your config still
contains one, the run stops immediately at config load with a message naming the
exact replacement (this happens even if the value is ``null``, and even if you
disabled config validation). Here is the mapping:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   -  -  Old setting (now removed)
      -  Modern replacement

   -  -  ``training.run_id``
      -  ``training.checkpoint.source`` = ``RunSource`` with ``run_id`` and
         ``fork: false``

   -  -  ``training.fork_run_id``
      -  ``training.checkpoint.source`` = ``RunSource`` with ``run_id`` and
         ``fork: true``

   -  -  ``system.input.warm_start``
      -  ``training.checkpoint.source`` = ``LocalSource`` with ``path``

   -  -  ``training.load_weights_only``
      -  ``training.checkpoint.source`` + ``training.checkpoint.loading`` =
         ``WeightsOnlyLoader``

   -  -  ``training.transfer_learning``
      -  ``training.checkpoint.source`` + ``training.checkpoint.loading`` =
         ``TransferLearningLoader`` (``skip_mismatched: true``)

   -  -  ``training.submodules_to_freeze``
      -  ``training.checkpoint.modifiers`` with ``FreezingModifierStage``

****************
 Where to next
****************

-  :ref:`checkpoint_integration` — the Python API and how stages, the context,
   and the trainer fit together (for developers writing custom stages or using
   the pipeline in code).
-  :ref:`checkpoint_troubleshooting` — error-by-error solutions.
