##########
Evaluation
##########

While we can run validation during training, some plotting callbacks take longer to
execute and it can be desirable to run a full validation pass on a saved checkpoint
in a decoupled way. For example, to compute metrics on a held-out period, regenerate
diagnostic plots, or benchmark a model before deployment — *without* resuming training.

.. code:: bash

   anemoi-training evaluate --config-name <config>

This runs one complete validation epoch using the same Hydra configuration as
training, so all data loading, normalisation, and diagnostics callbacks behave
identically. No optimizer state is created and no gradients are computed.

.. warning::

   A checkpoint **must** be specified via ``training.run_id``,
   ``training.fork_run_id``, or ``system.input.warm_start``. Omitting
   all three raises a :exc:`RuntimeError` immediately — evaluation on a
   randomly-initialised model is almost certainly a user error.

***********************
 Differences from train
***********************

The evaluator reuses :class:`~anemoi.training.train.train.AnemoiTrainer`
for all setup steps (datamodule, graph, model, callbacks, loggers,
strategy), but replaces the final ``trainer.fit()`` call with
``trainer.validate()``. Key behavioural differences:

- ``limit_val_batches`` controls how many batches to run (``config.dataloader.limit_batches.validation``).
- Arguments that only apply to training — ``max_epochs``, ``max_steps``,
  ``gradient_clip_val``, ``accumulate_grad_batches``, etc. — are not passed
  to the evaluator trainer.
- **DDP model wrapping is skipped**: Lightning's ``DDPStrategy`` only wraps
  the model in ``DistributedDataParallel`` during ``fit()``, not ``validate()``,
  because there are no gradients to reduce. The strategy handles this
  transparently — hardware and communication groups are set up as normal.
- Checkpointing and weight-averaging callbacks are automatically disabled
  (see below).

**************************
 Checkpoint loading
**************************

A checkpoint source must be configured before evaluation starts. Three
cases are recognised, in priority order:

1. **``system.input.warm_start``** — load from an explicit file path.
   Raises :exc:`FileNotFoundError` if the file does not exist. Takes
   precedence over ``run_id`` / ``fork_run_id`` when both are set.

2. **``training.run_id`` or ``training.fork_run_id``** — resolve the
   last checkpoint automatically as
   ``<checkpoints.root>/<run_id>/last.ckpt``.  Raises
   :exc:`RuntimeError` if the file is not found.

3. **Neither set** — raises :exc:`RuntimeError` immediately with a
   descriptive message (unlike training, where a fresh start is valid).

Once a checkpoint path is resolved, two loading modes are available:

- **``load_weights_only: True``** (recommended for evaluation) — model
  weights are loaded once during model initialisation; ``ckpt_path=None``
  is passed to ``trainer.validate()`` to avoid a redundant second load
  and to skip restoring optimizer/scheduler state.
- **``load_weights_only: False``** — PyTorch Lightning restores the full
  training state (weights, optimizer, epoch counter) before validation.

**************************
 Checkpointing and weight averaging
**************************

Checkpointing callbacks (:class:`~anemoi.training.diagnostics.callbacks.checkpoint.AnemoiCheckpoint`)
and weight-averaging callbacks (SWA / EMA) are **automatically disabled**
during evaluation regardless of what the diagnostics config says.
Evaluation is a read-only operation on a trained model and should never
write new checkpoint files or update model weights.

***********************************
 Config and CLI overrides
***********************************

``anemoi-training evaluate`` works exactly like ``anemoi-training train``
for config selection and Hydra overrides. Pass ``--config-name`` to select
a config file and any Hydra overrides as positional arguments:

.. code:: bash

   anemoi-training evaluate \
       --config-name evaluate_ana_short \
       training.run_id=<run_id>

You can also override individual keys without a dedicated config file:

.. code:: bash

   anemoi-training evaluate \
       --config-name debug_ana_short \
       training.run_id=<run_id> \
       training.load_weights_only=true \
       dataloader.limit_batches.validation=10 \
       diagnostics.plot.enabled=true

A minimal evaluation config that pairs with a training config is shown below.
It inherits the same defaults and overrides only the evaluation-specific keys:

.. literalinclude:: yaml/config_evaluate.yaml
   :language: yaml

***********************************
 Distributed evaluation
***********************************

The evaluator supports multi-GPU and multi-node evaluation via the same
``DDPGroupStrategy`` / ``DDPEnsGroupStrategy`` strategies used during
training. Set the hardware configuration as usual:

.. code:: yaml

   system:
     hardware:
       num_gpus_per_node: 4
       num_nodes: 2

The hidden script entry point ``.anemoi-training-evaluate`` is
registered alongside ``.anemoi-training-train`` so that Lightning's
interactive DDP can spawn rank > 0 processes correctly. Note that DDP
wrapping is not applied during validation (Lightning only wraps the model
for ``fit()``), but communication groups and sharding are set up
identically to training.
