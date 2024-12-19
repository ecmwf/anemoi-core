########
 Losses
########

This module is used to define the loss function used to train the model.

Anemoi-training exposes a couple of loss functions by default to be
used, all of which are subclassed from ``BaseWeightedLoss``. This class
enables scalar multiplication, and graph node weighting.

.. automodule:: anemoi.training.losses.weightedloss
   :members:
   :no-undoc-members:
   :show-inheritance:

************************
 Default Loss Functions
************************

By default anemoi-training trains the model using a latitude-weighted
mean-squared-error, which is defined in the ``WeightedMSELoss`` class in
``anemoi/training/losses/mse.py``. The loss function can be configured
in the config file at ``config.training.training_loss``, and
``config.training.validation_metrics``.

The following loss functions are available by default:

-  ``WeightedMSELoss``: Latitude-weighted mean-squared-error.
-  ``WeightedMAELoss``: Latitude-weighted mean-absolute-error.
-  ``WeightedHuberLoss``: Latitude-weighted Huber loss.
-  ``WeightedLogCoshLoss``: Latitude-weighted log-cosh loss.
-  ``WeightedRMSELoss``: Latitude-weighted root-mean-squared-error.
-  ``CombinedLoss``: Combined component weighted loss.

These are available in the ``anemoi.training.losses`` module, at
``anemoi.training.losses.{short_name}.{class_name}``.

So for example, to use the ``WeightedMSELoss`` class, you would
reference it in the config as follows:

.. code:: yaml

   # loss function for the model
   training_loss:
      # loss class to initialise
      _target_: anemoi.training.losses.mse.WeightedMSELoss
      # loss function kwargs here

*********
 Scalars
*********

In addition to node scaling, the loss function can also be scaled by a
scalar. These are provided by the ``Forecaster`` class, and a user can
define whether to include them in the loss function by setting
``scalars`` in the loss config dictionary.

.. code:: yaml

   # loss function for the model
   training_loss:
      # loss class to initialise
      _target_: anemoi.training.losses.mse.WeightedMSELoss
      scalars: ['scalar1', 'scalar2']

Currently, the following scalars are available for use:

-  ``variable``: Scale by the feature/variable weights as defined in the
   config ``config.training.variable_loss_scaling``.

********************
 Validation Metrics
********************

Validation metrics as defined in the config file at
``config.training.validation_metrics`` follow the same initialisation
behaviour as the loss function, but can be a list. In this case all
losses are calculated and logged as a dictionary with the corresponding
name

Scaling Validation Losses
=========================

Validation metrics can **not** by default be scaled by scalars across
the variable dimension, but can be by all other scalars. If you want to
scale a validation metric by the variable weights, it must be added to
`config.training.scale_validation_metrics`.

These metrics are then kept in the normalised, preprocessed space, and
thus the indexing of scalars aligns with the indexing of the tensors.

By default, only `all` is kept in the normalised space and scaled.

.. code:: yaml

   # List of validation metrics to keep in normalised space, and scalars to be applied
   # Use '*' in reference all metrics, or a list of metric names.
   # Unlike above, variable scaling is possible due to these metrics being
   # calculated in the same way as the training loss, within the internal model space.
   scale_validation_metrics:
   scalars_to_apply: ['variable']
   metrics:
      - 'all'
      # - "*"

***********************
 Custom Loss Functions
***********************

Additionally, you can define your own loss function by subclassing
``BaseWeightedLoss`` and implementing the ``forward`` method, or by
subclassing ``FunctionalWeightedLoss`` and implementing the
``calculate_difference`` function. The latter abstracts the scaling, and
node weighting, and allows you to just specify the difference
calculation.

.. code:: python

   from anemoi.training.losses.weightedloss import FunctionalWeightedLoss

   class MyLossFunction(FunctionalWeightedLoss):
      def calculate_difference(self, pred, target):
         return (pred - target) ** 2

Then in the config, set ``_target_`` to the class name, and any
additional kwargs to the loss function.

*****************
 Combined Losses
*****************

Building on the simple single loss functions, a user can define a
combined loss, one that weights and combines multiple loss functions.

This can be done by referencing the ``CombinedLoss`` class in the config
file, and setting the ``losses`` key to a list of loss functions to
combine. Each of those losses is then initalised just like the other
losses above.

.. code:: yaml

   training_loss:
      __target__: anemoi.training.losses.combined.CombinedLoss
      losses:
         - __target__: anemoi.training.losses.mse.WeightedMSELoss
         - __target__: anemoi.training.losses.mae.WeightedMAELoss
      scalars: ['variable']
      loss_weights: [1.0,0.5]

All kwargs passed to ``CombinedLoss`` are passed to each of the loss
functions, and the loss weights are used to scale the individual losses
before combining them.

.. automodule:: anemoi.training.losses.combined
   :members:
   :no-undoc-members:
   :show-inheritance:

*******************
 Utility Functions
*******************

There is also generic functions that are useful for losses in
``anemoi/training/losses/utils.py``.

``grad_scaler`` is used to automatically scale the loss gradients in the
loss function using the formula in https://arxiv.org/pdf/2306.06079.pdf,
section 4.3.2. This can be switched on in the config by setting the
option ``config.training.loss_gradient_scaling=True``.

``ScaleTensor`` is a class that can record and apply arbitrary scaling
factors to tensors. It supports relative indexing, combining multiple
scalars over the same dimensions, and is only constructed at
broadcasting time, so the shape can be resolved to match the tensor
exactly.

.. automodule:: anemoi.training.losses.utils
   :members:
   :no-undoc-members:
   :show-inheritance:
