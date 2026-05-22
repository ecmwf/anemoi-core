##########
Evaluation
##########

While we can run validation during training, some plotting callbacks take longer to
execute and it can desirable to be able to run a full validation pass on a saved checkpoint in a decouple way.
For example, we might want to compute metrics on a held-out period and regenerate diagnostic plots — *without* resuming training.

.. code:: bash

   anemoi-training evaluate

This runs one complete validation epoch using exactly the same Hydra
configuration as training, so all data loading, normalisation, and
diagnostics callbacks behave identically.  No optimizer state is
created and no gradients are computed.

***********************
 Differences from train
***********************

The evaluator reuses :class:`~anemoi.training.train.train.AnemoiTrainer`
for all setup steps (datamodule, graph, model, callbacks, loggers,
strategy), but replaces the final ``trainer.fit()`` call with
``trainer.validate()``.  The only trainer arguments that are relevant
are therefore:

- ``limit_val_batches`` — limits the number of validation batches (same
  as during training, controlled by
  ``config.dataloader.limit_batches.validation``).
- Logging, callbacks, and hardware settings — identical to a training
  run so that diagnostic plots and experiment-tracking entries are
  produced in the same way.

Arguments that only apply to training — ``max_epochs``, ``max_steps``,
``gradient_clip_val``, ``accumulate_grad_batches``, etc. — are not
passed to the evaluator trainer.

**************************
 Checkpoint loading
**************************

The evaluator follows the same checkpoint-loading logic as training:

- If ``config.training.load_weights_only`` is ``True``, only the model
  weights are restored from the checkpoint.  The Lightning checkpoint
  connector is not called again during ``trainer.validate()``, avoiding
  a redundant second load.
- Otherwise, ``ckpt_path`` is set to ``self.last_checkpoint`` and
  PyTorch Lightning restores the full training state (model weights,
  optimizer state, epoch counter, etc.) before validation begins.

To point the evaluator at a specific checkpoint set
``config.system.input.warm_start`` as you would for a training restart.

***********************************
 Passing overrides via the CLI
***********************************

``anemoi-training evaluate`` accepts arbitrary Hydra overrides just like
the ``train`` command:

.. code:: bash

   anemoi-training evaluate \
       dataloader.limit_batches.validation=10 \
       diagnostics.plot.enabled=true

This makes it straightforward to, for example, enable plots that were
disabled during training, or evaluate on a different date range by
overriding the dataset split.

***********************************
 Distributed evaluation
***********************************

The evaluator supports multi-GPU and multi-node evaluation via the same
``DDPGroupStrategy`` / ``DDPEnsGroupStrategy`` strategies used during
training.  Set the hardware configuration as usual:

.. code:: yaml

   system:
     hardware:
       num_gpus_per_node: 4
       num_nodes: 2

The hidden script entry point ``.anemoi-training-evaluate`` is
registered alongside ``.anemoi-training-train`` so that Lightning's
interactive DDP can spawn rank > 0 processes correctly.
