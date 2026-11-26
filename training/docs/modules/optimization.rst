#############
 Optimization
#############

Optimization in ``anemoi-training`` covers the optimizer, learning rate,
and LR scheduler. All three are configured together under the
``config.training.optimization`` config group.

**************************
 Configuration Structure
**************************

The optimization config group lives at
``training/config/training/optimization/`` and is composed of three
parts:

.. code:: text

   optimization/
   ├── default.yaml          # top-level defaults list + lr + pl_lr_scheduler
   ├── optimizer/
   │   ├── adamw.yaml        # default optimizer
   │   ├── ademamix.yaml     # AdEMAMix preset
   │   └── zero.yaml         # ZeroRedundancyOptimizer preset
   └── lr_scheduler/
       └── cosine_scheduler.yaml  # default scheduler (timm CosineLRScheduler)

The top-level ``default.yaml`` selects sub-configs through Hydra's
defaults list and sets the learning rates and Lightning scheduler
integration options:

.. code:: yaml

   defaults:
     - optimizer: adamw
     - lr_scheduler: cosine_scheduler
     - _self_

   learning_rate: 1.0e-3
   learning_rate_per_replica: null
   learning_rate_minimum: 3.0e-7
   learning_rate_per_replica_minimum: null

   # Lightning scheduler integration settings.
   # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
   pl_lr_scheduler:
     interval: step

All preset configs are activated by adding ``- optimization: default``
to the defaults list of a training config, which is the case for all
built-in training configs (``default``, ``autoencoder``, ``ensemble``,
etc.).

**************************
 Configuring an Optimizer
**************************

The active optimizer is selected via the ``optimization/optimizer``
config group. The default is AdamW:

.. code:: yaml

   # optimization/optimizer/adamw.yaml
   _target_: torch.optim.AdamW
   betas: [0.9, 0.95]

The learning rate is set separately in ``config.training.optimization``
and is not part of the optimizer config. There are two learning rates,
each set with one of two keys:

-  the peak learning rate, reached at the end of the scheduler's
   warmup: ``learning_rate`` or ``learning_rate_per_replica``.
-  the minimum learning rate the scheduler decays to:
   ``learning_rate_minimum`` or ``learning_rate_per_replica_minimum``.

``learning_rate`` and ``learning_rate_minimum`` are used as they are.
``learning_rate_per_replica`` and ``learning_rate_per_replica_minimum``
are multiplied by the number of model replicas, the model copies that
train side by side, each on its own batch:

.. code::

   num_model_replicas = num_nodes * num_gpus_per_node / num_gpus_per_model

For ensemble training, ``num_gpus_per_ensemble`` takes the place of
``num_gpus_per_model``. Keeping a per-replica value fixed makes the
learning rate grow with the number of GPUs, in step with the effective
batch size.

Set one key of each pair and the other to ``null``. The two pairs can be
mixed, for example a peak learning rate for the whole run with a
per-replica minimum:

.. code:: yaml

   training:
     optimization:
       learning_rate: 1.0e-3
       learning_rate_per_replica: null
       learning_rate_minimum: null
       learning_rate_per_replica_minimum: 3.0e-8

The peak learning rate is required. ``BaseGraphModule`` passes it as the
``lr`` argument when instantiating the optimizer. The minimum learning
rate is passed to the scheduler as its ``lr_min`` argument. Set both
minimum keys to ``null`` for a scheduler that takes no ``lr_min``, and
give its minimum in the scheduler config instead.

To override the optimizer at the command line:

.. code:: bash

   anemoi-training train training/optimization/optimizer=ademamix

Or inline in a training config override:

.. code:: yaml

   training:
     optimization:
       optimizer:
         _target_: torch.optim.AdamW
         betas: [0.9, 0.95]
         weight_decay: 0.1

**************************
 Learning Rate Schedulers
**************************

The LR scheduler is configured under ``optimization/lr_scheduler``.
The default is ``CosineLRScheduler`` from ``timm.scheduler``:

.. code:: yaml

   # optimization/lr_scheduler/cosine_scheduler.yaml
   _target_: timm.scheduler.CosineLRScheduler
   t_initial: ${training.max_steps}
   warmup_t: 1000
   t_in_epochs: false  # t_initial and warmup_t are in steps, not epochs

Any scheduler compatible with the ``timm`` scheduler interface or the
standard PyTorch ``LRScheduler`` interface can be used by specifying a
``_target_``. To use a different scheduler:

.. code:: bash

   anemoi-training train training/optimization/lr_scheduler=<your_preset>

To disable the scheduler entirely (constant LR), set ``lr_scheduler``
to ``null``:

.. code:: yaml

   training:
     optimization:
       lr_scheduler: null

``BaseGraphModule.configure_optimizers`` returns just the optimizer when
``lr_scheduler`` is absent or null. When a scheduler is configured, it
returns a tuple of ``([optimizer], [{"scheduler": scheduler,
**pl_lr_scheduler}])``, which is the format Lightning expects when both
an optimizer and a scheduler are used.

************************************
 Lightning Scheduler Integration
************************************

PyTorch Lightning wraps the scheduler in its own configuration layer
that controls *how* Lightning calls the scheduler during training —
independently of the scheduler's own parameters. This is configured
via ``pl_lr_scheduler``:

.. code:: yaml

   pl_lr_scheduler:
     interval: step  # "step" or "epoch"

The most important field is ``interval``, which tells Lightning whether
to step the scheduler after every batch (``step``) or after every epoch
(``epoch``). The default is ``step``, which is appropriate for the
default cosine scheduler where ``t_initial`` and ``warmup_t`` are given
in steps.

Any additional key accepted by Lightning's ``LRSchedulerConfig`` can
be added here — for example, ``monitor: val/loss`` when using
``ReduceLROnPlateau``. See the `Lightning docs
<https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers>`_
for the full list of options.

For ``timm`` schedulers, the stepping is handled by a custom
``lr_scheduler_step`` in ``BaseGraphModule`` that calls
``scheduler.step_update`` instead of the standard Lightning path, so
``interval: step`` works correctly with ``timm`` schedulers out of the
box.

********************
 Available Presets
********************

Optimizer Presets
=================

``optimization/optimizer: adamw`` (default)
   Standard AdamW. Parameters: ``betas``.

``optimization/optimizer: ademamix``
   :ref:`AdEMAMix <ademamix>` optimizer. Parameters: ``betas``,
   ``alpha``, ``beta3_warmup``, ``alpha_warmup``, ``weight_decay``.

``optimization/optimizer: zero``
   ``ZeroRedundancyOptimizer`` wrapping AdamW for memory-efficient
   distributed training. Parameters: ``betas``.

LR Scheduler Presets
====================

``optimization/lr_scheduler: cosine_scheduler`` (default)
   Cosine annealing with warm-up via ``timm.scheduler.CosineLRScheduler``.

********************
 AdEMAMix Optimizer
********************

.. _ademamix:

``AdEMAMix`` is a custom optimizer implemented in
``anemoi.training.optimizers.AdEMAMix`` and taken from the `Apple ML
AdEMAMix project <https://github.com/apple/ml-ademamix>`_. It combines
elements of Adam and exponential moving average (EMA) mixing for
improved stability and generalization.

The optimizer maintains **three exponential moving averages (EMAs)** of
the gradients. See <https://arxiv.org/abs/2409.03137> for more details.

To activate it, use the preset or override inline:

.. code:: bash

   anemoi-training train training/optimization/optimizer=ademamix

Or inline:

.. code:: yaml

   training:
     optimization:
       optimizer:
         _target_: anemoi.training.optimizers.AdEMAMix.AdEMAMix
         betas: [0.9, 0.95, 0.9999]
         alpha: 8.0
         beta3_warmup: 260000
         alpha_warmup: 260000
         weight_decay: 0.01

**************************
 Implementation Reference
**************************

.. automodule:: anemoi.training.optimizers.AdEMAMix
   :members:
   :no-undoc-members:
   :show-inheritance:
