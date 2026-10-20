..
   (C) Copyright 2026 Anemoi contributors.

   This software is licensed under the terms of the Apache Licence Version 2.0
   which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

   In applying this licence, ECMWF does not waive the privileges and immunities
   granted to it by virtue of its status as an intergovernmental organisation
   nor does it submit to any jurisdiction.

.. _checkpoint_pipeline:

#####################
 Checkpoint Pipeline
#####################

The checkpoint pipeline sets the checkpoint that a training run starts
from, and how the run uses it.

.. contents:: On this page
   :local:
   :depth: 2

**********
 Overview
**********

A checkpoint is a file that contains the model weights, the optimizer
state, the training progress (epoch and step) and metadata.

The pipeline has three stages. Each stage is optional. The stages always
run in this order:

.. code:: text

   source  ──▶  loading  ──▶  modifiers
   (where)      (how)         (change the model)

-  **Source**: where the checkpoint comes from (file, run, S3, URL).
   Maximum one.
-  **Loading strategy**: how the checkpoint is applied to the model.
   Maximum one.
-  **Modifiers**: changes to the model after loading, for example
   freezing. A list, applied in order.

Configure the stages under ``training.checkpoint``:

.. code:: yaml

   training:
     checkpoint:
       source: { ... }
       loading: { ... }
       modifiers: [ ... ]

Each stage is a Hydra ``_target_`` block. If ``training.checkpoint`` is
not set, the run starts from random weights.

.. _cp_recipes:

*********
 Recipes
*********

Fine-tune from a checkpoint file
================================

Load the weights. Start a new optimizer at epoch 0.

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.ColdStartLoader
         strict: false

Resume a run
============

Restore the weights, optimizer, scheduler and training progress.

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.run.RunIdSource
         run_id: a1b2c3d4-...
         fork: false
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WarmStartLoader

.. code:: bash

   anemoi-training train \
     training/checkpoint/source=run +training.checkpoint.source.run_id=a1b2c3d4-... \
     training/checkpoint/loading=warm_start

Transfer learning after an architecture change
==============================================

Load only the weights that fit. Skip the others.

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
         skip_mismatched: true

Freeze the encoder
==================

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
           submodules_to_freeze: ["model.model.encoder.0"]

Load from S3 or HTTP
====================

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.s3.S3Source
         url: s3://my-bucket/path/to/model.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader
         strict: false

For HTTP(S), use ``HTTPSource`` with an ``https://`` URL.

.. _cp_groups:

*********************
 Hydra config groups
*********************

The training presets have a config group for each stage. The default is
``null``. Select a group on the command line:

.. code:: bash

   anemoi-training train training/checkpoint/source=local \
       +training.checkpoint.source.path=/path/to/model.ckpt \
       training/checkpoint/loading=weights_only

.. list-table::
   :header-rows: 1
   :widths: 35 65

   -  -  Group
      -  Options

   -  -  ``training/checkpoint/source=``
      -  ``local``, ``run``, ``s3``, ``http``

   -  -  ``training/checkpoint/loading=``
      -  ``weights_only``, ``transfer_learning``, ``warm_start``,
         ``cold_start``

   -  -  ``training/checkpoint/modifiers=``
      -  ``freezing``

*********
 Sources
*********

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   -  -  Source
      -  Use
      -  Parameters

   -  -  ``LocalSource``
      -  A file on disk.
      -  ``path``

   -  -  ``RunIdSource``
      -  Resume or fork a previous run.
      -  ``run_id``, ``fork``

   -  -  ``S3Source``
      -  An S3 bucket.
      -  ``url`` (``s3://bucket/key``)

   -  -  ``HTTPSource``
      -  An HTTP(S) URL.
      -  ``url``, ``max_retries`` (3), ``timeout`` (300 s),
         ``expected_checksum`` (SHA-256, optional)

.. important::

   Anemoi does not detect the location type. Select the source that
   agrees with the location. For example, ``LocalSource`` does not read
   ``s3://`` paths.

**LocalSource.** ``path`` accepts ``~`` and symbolic links. A relative
path starts from the launch directory. Use an absolute path.

**RunIdSource.** The checkpoint is
``<system.output.checkpoints.root>/../<run_id>/last.ckpt``.
``fork: false`` resumes the same run. ``fork: true`` starts a new run from
the weights. The source also sets the run ID for MLflow and for the output
paths.

**S3Source.** Credentials and the endpoint come from the anemoi-utils
settings (``~/.config/anemoi/settings.toml``). Install
``anemoi-training[s3]``.

**HTTPSource.** If you set ``expected_checksum``, the download is
verified.

********************
 Loading strategies
********************

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   -  -  Strategy
      -  Use
      -  Optimizer and progress

   -  -  ``WeightsOnlyLoader``
      -  Load the weights only.
      -  Optimizer discarded. Progress not changed.

   -  -  ``ColdStartLoader``
      -  Fine-tune from pretrained weights as a new run.
      -  Optimizer discarded. Progress set to 0.

   -  -  ``TransferLearningLoader``
      -  Some layers changed shape or do not exist.
      -  Optimizer discarded. Only matching weights load.

   -  -  ``WarmStartLoader``
      -  Resume the same run.
      -  Restored by Lightning.

``strict``
   ``true`` requires the same set of keys. ``false`` accepts extra or
   missing keys. ``strict`` does not apply to tensor shapes: a size
   mismatch always gives an error.

``skip_mismatched``
   Skip parameters with a different shape. ``WeightsOnlyLoader`` and
   ``ColdStartLoader`` accept it (default ``false``).
   ``TransferLearningLoader`` uses it by default (``true``). With
   ``false``, a shape mismatch raises ``CheckpointIncompatibleError``.
   Skipped parameters keep their initial values. The run logs them at
   WARNING and records them in the run metadata as ``skipped_params``.

**ColdStartLoader** also records the source of the weights as
``pretrained_from``.

**TransferLearningLoader** loads the parameters that have the same name
and shape. For example, if ``num_channels`` changes, only the tensors
without a channel dimension load. The checkpoint file does not change.

To fine-tune on fewer variables than the checkpoint has, set
``training.allow_variable_subset: true`` and ``skip_mismatched: true``.

.. _cp_warm_start:

Warm start (resume)
===================

``WarmStartLoader`` has no parameters. If a source is set and no
``loading`` block is set, the result is the same.

On a resume, PyTorch Lightning loads the checkpoint through
``trainer.fit(ckpt_path=...)``. It restores the weights, optimizer,
scheduler and training progress in one load. The pipeline only makes the
checkpoint available as a local file and applies the modifiers.

A resume needs ``LocalSource`` or ``RunIdSource``. ``S3Source`` and
``HTTPSource`` delete the download after loading. Thus they cannot resume a
run. To resume from a remote checkpoint, download it and use ``LocalSource``.

Lightning loads the state dict strictly. Thus the model must agree with
the checkpoint. If the architecture changed, use ``WeightsOnlyLoader``
with ``skip_mismatched: true`` or ``TransferLearningLoader``.

With all other strategies, the pipeline loads the weights when it builds
the model. The trainer does not give ``ckpt_path`` to Lightning.

***********
 Modifiers
***********

Freezing
========

``FreezingModifierStage`` sets ``requires_grad=False`` on the specified
submodules.

.. code:: yaml

   training:
     checkpoint:
       modifiers:
         - _target_: anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage
           submodules_to_freeze: ["model.model.encoder.0", "model.model.processor.proc.0"]
           strict: false

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   -  -  Parameter
      -  Default
      -  Description

   -  -  ``submodules_to_freeze``
      -  ``[]``
      -  Dot-notation paths from the LightningModule. No wildcards. The
         graph model is ``model.model``. The encoder and decoder keys come
         from ``model.encoders`` and ``model.decoders`` (``encoder.0`` in
         ``graphtransformer.yaml``). Processor layers are under
         ``processor.proc``.

   -  -  ``strict``
      -  ``false``
      -  If ``true``, an unknown name gives an error. If ``false``, it
         gives a warning. If no names resolve, the stage always gives an
         error.

To find the names, print ``dict(module.named_modules()).keys()`` for the
LightningModule.

***************************
 Migrate from old settings
***************************

The old settings are removed. If the config contains one, the run stops
at config load and the error gives the replacement. This also occurs if
the value is ``null``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   -  -  Old setting
      -  Replacement

   -  -  ``training.run_id``
      -  ``RunIdSource`` with ``fork: false``

   -  -  ``training.fork_run_id``
      -  ``RunIdSource`` with ``fork: true``

   -  -  ``system.input.warm_start``
      -  ``LocalSource`` with ``path``

   -  -  ``training.load_weights_only``
      -  A source and ``WeightsOnlyLoader``

   -  -  ``training.transfer_learning``
      -  A source and ``TransferLearningLoader``

   -  -  ``training.submodules_to_freeze``
      -  ``FreezingModifierStage`` in ``training.checkpoint.modifiers``

.. _cp_custom_stages:

***********************
 Write a custom stage
***********************

Each stage is a subclass of one of three base classes. The pipeline uses
the base class to find the stage type. It does not use the name or the
package. You do not have to register the class.

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   -  -  Stage
      -  Base class
      -  ``process`` must

   -  -  Source
      -  ``anemoi.training.checkpoint.sources.base.CheckpointSource``
      -  Set ``context.checkpoint_path`` or ``context.checkpoint_data``.

   -  -  Loading
      -  ``anemoi.training.checkpoint.loading.base.LoadingStrategy``
      -  Load weights into ``context.model``. Set
         ``context.model.weights_initialized = True``.

   -  -  Modifier
      -  ``anemoi.training.checkpoint.modifiers.base.ModelModifier``
      -  Change ``context.model``. Append a record to
         ``context.metadata["modifiers_applied"]``.

Example:

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext
   from anemoi.training.checkpoint.modifiers.base import ModelModifier


   class WeightScalingModifier(ModelModifier):
       """Multiply the weights of the specified submodules by a factor."""

       def __init__(self, submodules: list[str], factor: float = 0.5) -> None:
           self.submodules = submodules
           self.factor = factor

       def process(self, context: CheckpointContext) -> CheckpointContext:
           for name in self.submodules:
               for param in context.model.get_submodule(name).parameters():
                   param.data.mul_(self.factor)
           context.metadata.setdefault("modifiers_applied", []).append(
               {"type": "weight_scaling", "submodules": self.submodules, "factor": self.factor}
           )
           return context

.. code:: yaml

   training:
     checkpoint:
       modifiers:
         - _target_: my_project.modifiers.WeightScalingModifier
           submodules: [decoder]
           factor: 0.5

A custom loader must call ``apply_checkpoint_corrections`` and
``preserve_anemoi_metadata`` from ``anemoi.training.checkpoint.loading.base``.
Then the loader gives the same result as the built-in loaders.

A training config accepts stages only as ``_target_`` blocks. A script or
a test can also give stage instances to ``CheckpointPipeline``.

**************
 Python API
**************

.. code:: python

   from anemoi.training.checkpoint import (
       CheckpointContext,
       CheckpointPipeline,
       build_checkpoint_pipeline,
   )

   pipeline = build_checkpoint_pipeline(cfg)   # reads cfg.training.checkpoint
   # or: CheckpointPipeline(stages=[source, loader, modifier])

   context = CheckpointContext(model=my_model, config=cfg)
   result = pipeline.execute(context)

``CheckpointContext`` fields:

-  ``model``: the model.
-  ``checkpoint_path``: the local checkpoint path. A source sets it.
-  ``checkpoint_data``: the loaded checkpoint dict.
-  ``checkpoint_format``: ``"lightning"``, ``"pytorch"`` or
   ``"state_dict"``.
-  ``metadata``: a dict where each stage records what it did.
-  ``config``: the Hydra config (optional).

``CheckpointPipeline`` accepts instances or ``_target_`` configs. The
stages must be in the order source, loading, modifiers.

Utilities:

.. code:: python

   from anemoi.training.checkpoint import (
       ComponentCatalog,          # lists the built-in stages only
       compare_state_dicts,       # missing, unexpected, mismatched keys
       get_checkpoint_metadata,   # read metadata without a full load
   )
   from anemoi.training.checkpoint.formats import detect_checkpoint_format

Exceptions (all are subclasses of ``CheckpointError``):
``CheckpointNotFoundError``, ``CheckpointLoadError``,
``CheckpointValidationError``, ``CheckpointSourceError``,
``CheckpointTimeoutError``, ``CheckpointConfigError``,
``CheckpointIncompatibleError``.

*****************
 Troubleshooting
*****************

To get more log data:

.. code:: python

   import logging

   logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

Removed setting
===============

``ValueError: training.load_weights_only has been removed. ...``

Replace the setting. See the table in `Migrate from old settings`_.

Invalid pipeline order
======================

``CheckpointConfigError: Invalid checkpoint pipeline [...]: use the order source -> one loader -> modifiers.``

The pipeline does not accept a source after a loader, a modifier before
a loader, or more than one loader. The config builder always uses the
correct order. Thus this error occurs only for a pipeline built in code.

Resume without a checkpoint
===========================

``CheckpointConfigError: Warm start resumes through Trainer.fit(ckpt_path=), ...``

``CheckpointConfigError: Resume configured, but RunIdSource resolved no checkpoint file ...``

``CheckpointConfigError: RunIdSource requires a run_id ...``

A resume needs a source that resolves to a checkpoint. Do one of these:

-  Add a source, for example ``training/checkpoint/source=run
   +training.checkpoint.source.run_id=<id>``.
-  Set ``run_id``. The ``run`` preset sets it to ``null``.
-  To start with a new optimizer, add a loading strategy.

Remote source cannot resume
===========================

``CheckpointConfigError: S3Source cannot resume a run ...``

Download the checkpoint and use ``LocalSource``. Or add a loading strategy
to load only the weights.

Stage cannot be built
=====================

``CheckpointConfigError: Failed to instantiate pipeline stage ...``

Hydra cannot make the stage from ``_target_``. Make sure that the path is
correct and that all required parameters are set. To see the cause,
import the class and make an instance in Python.

Key mismatch
============

``CheckpointIncompatibleError: strict=True and the checkpoint key set does not match the model's. ...``

``strict: true`` requires the same keys. Set ``strict: false`` or use
``TransferLearningLoader``. To find the different keys, use
``compare_state_dicts``.

Size mismatch
=============

``CheckpointIncompatibleError: Parameter shapes differ between checkpoint and model. ...``

The checkpoint has a different set of variables. ``strict: false`` does
not prevent this error.

#. Make sure that the checkpoint is correct.
#. To fine-tune on fewer variables, set
   ``training.allow_variable_subset: true`` and ``skip_mismatched: true``.
   Read the ``skipped_params`` list: these parameters are not trained.
#. Or use ``TransferLearningLoader``.

Resume after an architecture change
===================================

``Error(s) in loading state_dict ... Missing key(s) / Unexpected key(s)``

A resume needs the same architecture. Use ``TransferLearningLoader`` or
``ColdStartLoader``. See :ref:`cp_warm_start`.

Model state not found
=====================

``CheckpointValidationError: Cannot find model state in checkpoint``

The file structure is not known. Use ``detect_checkpoint_format`` to
examine the file.

Old checkpoint format
=====================

If a checkpoint from an older anemoi-models fails with a checkpoint-format
error, migrate it with ``anemoi-models migration sync <path>``. This
command changes the file and writes a full-size backup. Make sure that
there is sufficient disk space. For a remote source, download the file,
migrate it, and load it with ``LocalSource``.

File not found
==============

``CheckpointNotFoundError: Checkpoint not found: <path>``

Make sure that the file exists. Use an absolute path. Use ``S3Source`` or
``HTTPSource`` for remote files.

Missing package
===============

``CheckpointSourceError: S3Source requires anemoi-utils[s3] ...``

Install ``anemoi-training[s3]``.

Download failed
===============

``CheckpointSourceError: HTTP 404 ...`` or ``CheckpointSourceError: S3
download failed ...``

Make sure that the URL is correct and that you have access. For S3, look
at the credentials in ``~/.config/anemoi/settings.toml``. You can also
download the file and use ``LocalSource``.

Out of GPU memory
=================

Sources load checkpoints to the CPU. Thus the GPU memory is usually used
by the model. Use a weights-only strategy, because it does not keep the
optimizer state.
