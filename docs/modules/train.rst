#######
 Train
#######

This module defines the training process for the neural network model.
The ``GraphForecaster`` object in ``forecaster.py`` is responsible for
the forward pass of the model itself. The key-functions in the
forecaster that users may want to adapt to their own applications are:

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
