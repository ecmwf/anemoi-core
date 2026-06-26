.. _rollout:

################################
 Rollout Training
################################

The ``Forecaster`` task supports **rollout training**: the
model is initially trained to predict only one step ahead, and the
prediction horizon is gradually extended across epochs. This lets the
model learn short-range dynamics first before being exposed to the
harder, compounding-error problem of longer forecasts.

*********************
 How rollout works
*********************

Each training sample loads ``multistep_input`` input frames followed by
enough output frames to cover the *maximum* rollout window. During
forward passes the training loop iterates over the *current* rollout
window, producing one autoregressive step per iteration.

The current rollout window is controlled by ``RolloutConfig``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``start``
     - Number of rollout steps at epoch 0. Must be ≥ 1.
   * - ``epoch_increment``
     - The rollout window is extended by one step at the end of every
       ``epoch_increment``-th epoch. Set to ``0`` to keep the window
       fixed.
   * - ``maximum``
     - The rollout window never grows beyond this value.

At epoch zero the effective rollout length is ``start``. After every
``epoch_increment`` epochs ``RolloutConfig.step`` is incremented by one
(capped at ``maximum``).

Example: with ``start=1``, ``epoch_increment=1`` and ``maximum=4`` the
rollout window grows as follows:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Epoch
     - Rollout window
   * - 0
     - 1 step
   * - 1
     - 2 steps
   * - 2
     - 3 steps
   * - 3 and beyond
     - 4 steps (capped at ``maximum``)

*****************
 Configuration
*****************

Configure rollout under the ``task`` key in your Hydra YAML:

.. code:: yaml

   task:
     _target_: anemoi.training.tasks.Forecaster
     multistep_input: 2
     multistep_output: 1
     timestep: "6H"
     rollout:
       start: 1
       epoch_increment: 1
       maximum: 12
     validation_rollout: 1

``validation_rollout`` controls the rollout length used during
validation. It is independent of the training curriculum and does not
change across epochs.

To disable the curriculum and train at a fixed rollout length from the
start, set ``epoch_increment: 0``.

**********************************
 Persisting rollout across resumes
**********************************

``RolloutConfig.step`` is mutable runtime state. Prior to
`PR #1109 <https://github.com/ecmwf/anemoi-core/pull/1109>`_ this value
was not written to checkpoints, so a resumed job always restarted the
curriculum from ``start``.

With the fix, the rollout step is saved into the Lightning checkpoint
under the ``task_state`` key by ``BaseTrainingModule.on_save_checkpoint``
and restored by ``on_load_checkpoint``. Because ``task_state`` is stored
at the **Lightning checkpoint level** rather than inside the model's own
``state_dict``, it is invisible to inference code (e.g.
``anemoi-inference``), which only reads the model weights.

The serialisation path is:

.. code::

   Forecaster.training_runtime_state_dict()
       ↓
   BaseTrainingModule.on_save_checkpoint(checkpoint)
       → checkpoint["task_state"] = task.training_runtime_state_dict()

   BaseTrainingModule.on_load_checkpoint(checkpoint)
       → task.load_training_runtime_state_dict(checkpoint.get("task_state", {}))
       ↓
   Forecaster.load_training_runtime_state_dict(state)

Old checkpoints that predate this change have no ``task_state`` key; the
``checkpoint.get("task_state", {})`` fallback passes an empty dict to
``load_training_runtime_state_dict``, which leaves the rollout step at
``start``. This maintains full backward compatibility.

Double-increment guard
======================

PyTorch Lightning fires ``on_train_epoch_end`` once during the
checkpoint-restore phase *before* any new epoch actually trains. Without
a guard this would increment the rollout step one epoch too early on
every resume.

The fix tracks the last epoch that triggered an increment in
``last_increased_epoch`` (also persisted in ``task_state``). The
``on_train_epoch_end`` callback becomes a no-op when
``current_epoch == last_increased_epoch``, so the replayed hook during
restore is safely ignored.

Restarting rollout training
===========================

Because the rollout curriculum step is now persisted in every
Lightning checkpoint, restarting an interrupted run requires no manual
adjustment of ``rollout.start``.

The recommended restart recipe is:

1. Restart from any Lightning checkpoint (end-of-epoch or mid-epoch).
2. Keep ``rollout.start``, ``epoch_increment``, and ``maximum``
   **unchanged** in your configuration.
3. Set ``training.max_epochs`` to the new total epoch count and
   ``training.run_id`` to the run ID of the interrupted job.

.. code:: yaml

   # First run
   training:
     max_epochs: 2
     run_id: null   # new run

   # Resumed run — same YAML, only max_epochs and run_id change
   training:
     max_epochs: 4
     run_id: <run_id_from_first_run>

On resume, Anemoi reads the saved ``task_state`` from the checkpoint
and restores ``rollout.step`` and ``last_increased_epoch`` automatically.
The double-increment guard (see above) ensures that if the checkpoint
was written at an epoch boundary, the rollout is not incremented twice
for that epoch.

.. note::

   Checkpoints written before PR #1109 have no ``task_state`` key.
   Resuming from such a checkpoint falls back to ``rollout.start``,
   so you would need to set ``start`` to the rollout value that was
   active when that checkpoint was saved.

The rollout metric logged to MLflow will continue from where it left
off rather than restarting from ``start``.

*********
 Logging
*********

The ``Forecaster`` task logs the current ``rollout`` value once per
epoch (``on_epoch=True``). This is constant within an epoch, so
per-step logging would produce sparse or missing entries in MLflow
depending on the ``log_every_n_steps`` setting.

You can observe the rollout progression in any experiment tracker that
receives the ``rollout`` metric (MLflow, Weights & Biases, etc.).

See also :doc:`tasks` for a full description of the ``Forecaster`` task
and :doc:`training` for general training configuration.
