#######
 Tasks
#######

The ``anemoi.training.tasks`` module defines the temporal structure of a
training sample independently of the Lightning module. Each task
specifies which time steps are loaded as inputs and which are used as
targets, referred as offsets as they are defined as relative positions in time compared to a reference point, and provides helpers for mapping those offsets to batch
positions.

This separation lets plotting callbacks, dataloaders, and the trainer
remain task-agnostic — they call ``task.get_batch_input_indices`` and
``task.get_batch_output_indices`` without knowing anything about the
specific workflow.

*********
 BaseTask
*********

:class:`~anemoi.training.tasks.base.BaseTask` is the abstract root of
the hierarchy. It is constructed from two lists of
:class:`~datetime.timedelta` objects:

- ``input_offsets`` — time offsets of the model inputs relative to the
  analysis time, e.g. ``[-6H, 0H]`` for a two-step input.
- ``output_offsets`` — time offsets of the model targets, e.g. ``[6H]``
  for a single-step forecast.

The union of these two lists (``task.offsets``) tells the datamodule
which time steps to load for each sample.

Key properties and methods:

- ``input_offsets`` / ``output_offsets`` — sorted timedelta lists.
- ``offsets`` — full sorted union, used by the datamodule.
- ``num_input_timesteps`` / ``num_output_timesteps`` — lengths of the
  offset lists.
- ``steps`` — iterable of per-step dicts (e.g. ``{"rollout_step": 0}``);
  the training loop iterates over these.
- ``get_batch_input_indices(**kwargs)`` — positions of input offsets
  within the full batch tensor's time dimension.
- ``get_batch_output_indices(**kwargs)`` — positions of output offsets
  within the full batch tensor's time dimension.
- ``get_inputs`` / ``get_targets`` — extract and index-select the
  appropriate slices from a batch dict.

.. automodule:: anemoi.training.tasks.base
   :members:
   :no-undoc-members:
   :show-inheritance:

********************
 BaseSingleStepTask
********************

:class:`~anemoi.training.tasks.base.BaseSingleStepTask` is a convenience
subclass for tasks with a single training step (no rollout). Both
:class:`~anemoi.training.tasks.temporal_downscaling.TemporalDownscaler`
and :class:`~anemoi.training.tasks.timeless.BaseTimelessTask` inherit
from it.

************
 Forecaster
************

:class:`~anemoi.training.tasks.forecasting.Forecaster` implements
autoregressive rollout training. It is constructed with:

- ``multistep_input`` — number of input time steps (e.g. ``2`` for
  ``[-6H, 0H]``).
- ``multistep_output`` — number of output time steps per rollout step
  (e.g. ``1``).
- ``timestep`` — the model timestep as a frequency string (e.g.
  ``"6H"``).
- ``rollout`` — optional dict configuring the rollout schedule (see
  :class:`~anemoi.training.tasks.forecasting.RolloutConfig`).
- ``validation_rollout`` — number of rollout steps used during
  validation (default ``1``).

RolloutConfig
=============

:class:`~anemoi.training.tasks.forecasting.RolloutConfig` encapsulates
the progressive rollout schedule:

- ``start`` — initial number of rollout steps at epoch 0.
- ``epoch_increment`` — increase the rollout window by one every this
  many epochs (``0`` disables progression).
- ``maximum`` — the rollout window is never increased beyond this value.

The current step count is stored in ``rollout.step`` and is increased
by calling ``rollout.increase()``, which is triggered by the trainer at
the end of each epoch via ``on_train_epoch_end``.

.. automodule:: anemoi.training.tasks.forecasting
   :members:
   :no-undoc-members:
   :show-inheritance:


Multistep Input and Output
==========================

The forecaster task uses ``multistep_input`` and ``multistep_output`` to control how many time
steps the model ingests as input and predicts in a single forward pass.

-  ``multistep_input``: number of past timesteps provided as model input. When set to 1, only `t_{0}` is used.
-  ``multistep_output``: number of future timesteps predicted per forward pass.

Set ``multistep_output`` greater than 1 to enable multi-output prediction. This
reduces the number of forward passes needed to cover a rollout horizon.

Example:

.. code:: yaml

  task:
    _target_: anemoi.training.tasks.Forecaster
    multistep_input: 3
    multistep_output: 2
    timestep: "6H"
    rollout:
      start: 1
      epoch_increment: 1
      maximum: 6


Rollout behavior:

-  When time indices are inferred, the dataloader uses
   ``multistep_input + rollout * multistep_output`` to determine how many timesteps
   to load.
-  If ``multistep_output`` is greater than ``multistep_input``, only the most recent
   ``multistep_input`` outputs are fed into the next rollout step.


*********************
 TemporalDownscaler
*********************

:class:`~anemoi.training.tasks.temporal_downscaling.TemporalDownscaler`
downscales to higher temporal resolution by generating intermediate time steps between two input
times. It is constructed with:

- ``input_timestep`` — coarse time resolution (e.g. ``"6H"``).
- ``output_timestep`` — target fine resolution (e.g. ``"3H"``).
  Must evenly divide ``input_timestep``.
- ``output_left_boundary`` — if ``True``, include the ``t=0`` frame in
  the output targets (default ``False``).
- ``output_right_boundary`` — if ``True``, include the final
  ``t=input_timestep`` frame in the output targets (default ``False``).

Example: ``input_timestep="6H"``, ``output_timestep="3H"``,
``output_left_boundary=True`` produces output offsets
``[0H, 3H]`` and input offsets ``[0H, 6H]``.

.. automodule:: anemoi.training.tasks.temporal_downscaling
   :members:
   :no-undoc-members:
   :show-inheritance:

*************
 Autoencoder
*************

:class:`~anemoi.training.tasks.timeless.Autoencoder` is a timeless task:
both input and output are a single snapshot at ``t=0``. It inherits from
:class:`~anemoi.training.tasks.timeless.BaseTimelessTask` which itself
inherits from :class:`~anemoi.training.tasks.base.BaseSingleStepTask`.

.. automodule:: anemoi.training.tasks.timeless
   :members:
   :no-undoc-members:
   :show-inheritance:
