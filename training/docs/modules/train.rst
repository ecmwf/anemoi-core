#######
 Train
#######

This module defines the training process for the neural network model.
By setting the flag ``task``, you can choose which task to optimize your
model over. Any model that inherits from the ``GraphForecaster`` object
can be used. Examples are: GraphAutoEncoder, GraphInterpolator, etc..

The default task is set to the ``GraphForecaster`` object in
``forecaster.py`` which is responsible for the forward pass of the model
itself.

The key-functions in all tasks that users may want to adapt to their own
applications are:

-  ``advance_input``, which defines how the model iterates forward in
   forecast time
-  ``_step``, where the forward pass of the model happens both during
   training and validation

``AnemoiTrainer`` in ``train.py`` is the object from which the training
of the model is controlled. It also contains functions that enable the
user to profile the training of the model (``profiler.py``).

.. automodule:: anemoi.training.train.forecaster
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.train
   :members:
   :no-undoc-members:
   :show-inheritance:
