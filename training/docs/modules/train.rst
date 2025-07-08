#######
 Train
#######

The ``GraphForecaster`` and ``AnemoiTrainer`` define the training
process for the neural network model. While the ``GraphForecaster``
defines the ``LightningModule`` that defines the model task, the
``AnemoiTrainer`` module calls the training function.

************
 Forecaster
************

The different model tasks are reflected in different forecasters:

#. Deterministic Forecasting (GraphForecaster)
#. Ensemble Forecasting (GraphEnsForecaster)
#. Time Interpolation (GraphInterpolator)

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

.. automodule:: anemoi.training.train.forecaster.forecaster
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.forecaster.ensforecaster
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.forecaster.interpolator
   :members:
   :no-undoc-members:
   :show-inheritance:

*********
 Trainer
*********

The ``AnemoiTrainer`` object in ``train.py`` is responsible for calling
the training function.

.. automodule:: anemoi.training.train.train
   :members:
   :no-undoc-members:
   :show-inheritance:
