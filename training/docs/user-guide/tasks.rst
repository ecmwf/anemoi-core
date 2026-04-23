.. _tasks target:

#######
 Tasks
#######

A **task** defines the temporal I/O structure of a training sample: which
time steps are loaded as model inputs and which are used as prediction
targets. Tasks are defined in ``anemoi.training.tasks`` and are
configured under the ``task`` key. The task is independent of
the model architecture and the training method.

All tasks inherit from ``BaseTask``, which defines the interface that
the training loop relies on:

* ``get_inputs()`` / ``get_targets()`` — slice the loaded batch into
  model inputs and targets.
* ``steps()`` — an iterable of per-step keyword dicts consumed by the
  training loop (one dict = one forward pass).
* ``advance_input()`` — update model inputs between rollout steps
  (only relevant for multi-step tasks).


************
 Our tasks
************

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

For the full API reference see :doc:`../modules/tasks`.

************************
 Writing a custom task
************************

This section walks through adding a new task to Anemoi Training, using
a **backward forecaster** as a concrete example. A backward forecaster
predicts the *previous* time step given one or two consecutive input weather states.
This can be useful for predictability studies where one wants to
understand how well a model can reconstruct the past from a given state.

The temporal layout is:

* **Inputs:** :math:`t = 0` and :math:`t = +\Delta t` (two consecutive
  frames).
* **Output:** :math:`t = -\Delta t` (the step immediately before the
  input window).


 **Step 1** — Implement the task class
===================================

Create a new file
``src/anemoi/training/tasks/backward_forecaster.py``:

.. code:: python

   # (C) Copyright 2026- Anemoi contributors.
   #
   # This software is licensed under the terms of the Apache Licence Version 2.0
   # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

   from datetime import timedelta

   from anemoi.training.tasks.base import BaseSingleStepTask
   from anemoi.utils.dates import frequency_to_timedelta


   class BackwardForecaster(BaseSingleStepTask):
       """Backward forecasting task.

       Predicts the previous time step from two consecutive input frames.
       Useful for predictability studies.

       Temporal layout (example with timestep = 6 H):

       * Inputs:  t = 0, t = +6 H
       * Output:  t = −6 H
       """

       name: str = "backward-forecaster"

       def __init__(self, timestep: str, **kwargs) -> None:
           self.timestep = frequency_to_timedelta(timestep)

           input_offsets = [timedelta(0), self.timestep]
           output_offsets = [-self.timestep]

           super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)


Key points:

* The class extends ``BaseSingleStepTask``, so ``steps()`` returns a
  single empty dict and ``advance_input()`` is a no-op. These methods should
  be overridden if you want to support backward rollout training.
* ``input_offsets`` and ``output_offsets`` are plain lists of
  ``timedelta`` objects. The base class sorts them and builds the
  combined offset list that the datamodule uses to decide how many
  time steps to load per sample.


 **Step 2** — Add a Hydra configuration file
==========================================

Create ``src/anemoi/training/config/task/backward_forecaster.yaml``:

.. code:: yaml

   _target_: anemoi.training.tasks.BackwardForecaster
   timestep: "6H"

This file becomes a Hydra **config group option**. Users can select it
on the command line or in a recipe YAML with ``task: backward_forecaster``
(Hydra resolves the file name without the ``.yaml`` extension).


 **Step 3** — Add a Pydantic validation schema
=================================================

Open ``src/anemoi/training/schemas/tasks.py`` and add:

.. code:: python

   class BackwardForecasterSchema(BaseModel):
       """Configuration for the backward forecasting task."""

       target_: Literal["anemoi.training.tasks.BackwardForecaster"] = Field(
           ..., alias="_target_"
       )
       "Task class path."
       timestep: str = Field(example="6H")
       "Timestep string (e.g. '6H')."

Then include the new schema in the ``TaskSchema`` discriminated union at
the bottom of the same file:

.. code:: python

   TaskSchema = Annotated[
       ForecasterSchema
       | AutoencoderTaskSchema
       | TemporalDownscalerSchema
       | BackwardForecasterSchema,
       Discriminator("target_"),
   ]

For more details on how Pydantic schemas and Hydra instantiation work
together in Anemoi Training, see :doc:`../contributing`.


 **Step 4** — Register the task in the package
=================================================

Open ``src/anemoi/training/tasks/__init__.py`` and add the new task:

.. code:: python

   from .backward_forecaster import BackwardForecaster
   from .forecaster import Forecaster
   from .temporal_downscaler import TemporalDownscaler
   from .timeless import Autoencoder

   __all__ = [
       "Autoencoder",
       "BackwardForecaster",
       "Forecaster",
       "TemporalDownscaler",
   ]

This is only needed to import it directly from ``anemoi.training.tasks``. 


 **Step 5** — Use the new task
============================

Override the task in your training command:

.. code:: bash

   anemoi-training train task=backward_forecaster

Or set it directly in a recipe YAML:

.. code:: yaml

   defaults:
     - task: backward_forecaster

   task:
     timestep: "6H"

The training loop will:

#. Load three time steps per sample (``t = −6 H``, ``t = 0``,
   ``t = +6 H``).
#. Feed ``t = 0`` and ``t = +6 H`` as model inputs.
#. Compare the model output against ``t = −6 H``.
#. Compute the loss. 
#. Backpropagate the gradients.
#. Update the model parameters.
