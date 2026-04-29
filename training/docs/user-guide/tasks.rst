.. _tasks target:

#######
 Tasks
#######

A **task** defines the temporal I/O structure of a training sample: which
time steps are loaded as model inputs and which are used as prediction
targets. Tasks are defined in ``anemoi.training.tasks`` and are
configured under the ``task`` key. The task is independent of
the model architecture and the training method.

The three built-in tasks are:

``Forecaster``
   Autoregressive rollout training. Inputs are ``multistep_input``
   consecutive frames ending at ``t=0``; outputs are
   ``multistep_output`` frames per rollout step. The rollout window
   grows progressively from ``rollout.start`` up to ``rollout.maximum``
   every ``rollout.epoch_increment`` epochs.

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

``TemporalDownscaler``
   Generates a dense sequence of intermediate time steps between two
   coarse input frames. The output resolution must evenly divide the
   input resolution.

   .. code:: yaml

      task:
        _target_: anemoi.training.tasks.TemporalDownscaler
        input_timestep: "6H"
        output_timestep: "3H"
        output_left_boundary: true   # include t=0 in targets

``Autoencoder``
   Single-snapshot reconstruction: both input and output are at
   ``t=0``. No temporal structure required.

   .. code:: yaml

      task:
        _target_: anemoi.training.tasks.Autoencoder

For full API details see :doc:`../modules/tasks`.
