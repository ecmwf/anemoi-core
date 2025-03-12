############
 Schedulers
############

This module provides custom schedulers for use in training.

It provides implementations of step or epoch based schedulers, as well
as those that have complex step to epoch relations.

.. automodule:: anemoi.training.schedulers.schedulers
   :members:
   :no-undoc-members:
   :show-inheritance:

********************
 Rollout Schedulers
********************

Using the base scheduler class, rollout can be scheduled as a product of
step or epoch.

By default, the rollout scheduler is set to be an epoch based scheduler,

.. code:: yaml

   rollout:
      _target_: anemoi.training.schedulers.rollout.stepped.EpochStepped
      minimum: 1
      maximum: 12
      # increase rollout every n epochs
      every_n_epochs: 1
      # Control the incrementing of the rollout window
      increment:
         step:
            0: 0
            260000: 1 # After 200k steps, increment by 1 every 1 epoch

This will step the rollout every epoch up to 12, for every epoch after
step 260000.

The following rollout schedulers are also available:

.. automodule:: anemoi.training.schedulers.rollout
   :members:
   :no-undoc-members:
   :show-inheritance:

Indexed
=======

.. automodule:: anemoi.training.schedulers.rollout.indexed
   :members:
   :no-undoc-members:
   :show-inheritance:

Randomised
==========

.. automodule:: anemoi.training.schedulers.rollout.randomise
   :members:
   :no-undoc-members:
   :show-inheritance:

Stepped
=======

.. automodule:: anemoi.training.schedulers.rollout.stepped
   :members:
   :no-undoc-members:
   :show-inheritance:
