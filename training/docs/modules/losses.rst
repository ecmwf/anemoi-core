########
 Losses
########

This module is used to define the loss function used to train the model.

Anemoi-training exposes a couple of loss functions by default to be
used, all of which are subclassed from ``BaseLoss``. This class enables
scaler multiplication, and graph node weighting.

.. automodule:: anemoi.training.losses.weightedloss
   :members:
   :no-undoc-members:
   :show-inheritance:

******************************
 Deterministic Loss Functions
******************************

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

******************************
 Probabilistic Loss Functions
******************************

The following probabilistic loss functions are available by default:

-  ``KernelCRPSLoss``: Kernel CRPS loss.
-  ``AlmostFairKernelCRPSLoss``: Almost fair Kernel CRPS loss see `Lang
   et al. (2024) <http://arxiv.org/abs/2412.15832>`_.

The config for these loss functions is the same as for the
deterministic:

.. code:: yaml

   # loss function for the model
   training_loss:
      # loss class to initialise
      _target_: anemoi.training.losses.kcrps.KernelCRPSLoss
      # loss function kwargs here

*********
 Scalers
*********

In addition to node scaling, the loss function can also be scaled by a
scaler. These are provided by the ``Forecaster`` class, and a user can
define whether to include them in the loss function by setting
``scalers`` in the loss config dictionary.

.. code:: yaml

   # loss function for the model
   training_loss:
      # loss class to initialise
      _target_: anemoi.training.losses.mse.WeightedMSELoss
      scalers: ['scaler1', 'scaler2']

Scalers can be added as options for the loss functions using the
`scaler` builders in `config.training.scaler`.

``*`` is a valid entry to use all `scalers` given, if a scaler is to be
excluded add `!scaler_name`, i.e. ``['*', '!scaler_1']``, and
``scaler_1`` will not be added.

Tendency Scalers
================

Tendency scalers allow the scaling of prognostic losses by the standard
deviation or variance of the variable tendencies (e.g. the 6-hourly
differences in the data). To floating point precision, this loss scaling
is equivalent to training on tendencies rather than the forecasts
themselves. This approach is particularly useful when training models
that include both slow-evolving variables (e.g., Land/Ocean) and
fast-evolving variables (e.g., Atmosphere), ensuring balanced
contributions to the loss function. When using this option, it is
recommended to set the `general_variable` scaling values close to 1.0
for all prognostic variables to maintain consistency and avoid
unintended bias in the training process.

.. code:: yaml

   stdev_tendency:
      _target_: anemoi.training.losses.scalers.StdevTendencyScaler
   var_tendency:
     _target_: anemoi.training.losses.scalers.VarTendencyScaler

Variable Level Scalers
======================

Variable level scalers allow the user to scale variables by its level,
i.e. model or pressure levels for upper air variables. The variable
level scalers are applied to groups that are defined under
`scalers.variable_groups`.

For a pressure level scaler applied to all pressure level variables the
configuration would look like this:

.. code:: yaml

   pressure_level:
      # Variable level scaler to be used
      _target_: anemoi.training.losses.scalers.ReluVariableLevelScaler
      group: pl
      y_intercept: 0.2
      slope: 0.001

This will scale all variables in the `pl` group by max(0.2, 0.001 *
level), where `level` is the pressure level of the variable.

Variable Groups
===============

Define a default group and a list of groups to be used in the variable
level scalers.

.. code:: yaml

   # Variable groups to be used in the variable level scalers
   variable_groups:
      default: sfc
      pl: [q, t, u, v, w, z]

If working with upper-air variables from variable levels, the
temperature fields start with the variable reference `t` followed by the
level, i.e. `t_500`, `t_850`, etc. Since `t` is specified under variable
group `pl`, all temperature fields are considered group `pl`. If the
datasets are build from mars the variable reference is etracted from
metadata, otherwise by splitting the variable name by `_` and taking the
first part, see class
`anemoi.training.utils.ExtractVariableGroupAndLevel`.

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

Validation metrics can **not** by default be scaled by scalers across
the variable dimension, but can be by all other scalers. If you want to
scale a validation metric by the variable weights, it must be added to
`config.training.scale_validation_metrics`.

These metrics are then kept in the normalised, preprocessed space, and
thus the indexing of scalers aligns with the indexing of the tensors.

By default, only `all` is kept in the normalised space and scaled.

.. code:: yaml

   # List of validation metrics to keep in normalised space, and scalers to be applied
   # Use '*' in reference all metrics, or a list of metric names.
   # Unlike above, variable scaling is possible due to these metrics being
   # calculated in the same way as the training loss, within the internal model space.
   scale_validation_metrics:
   scalers_to_apply: ['variable']
   metrics:
      - 'all'
      # - "*"

***********************
 Custom Loss Functions
***********************

Additionally, you can define your own loss function by subclassing
``BaseLoss`` and implementing the ``forward`` method, or by subclassing
``BaseLoss`` and implementing the ``calculate_difference`` function. The
latter abstracts the scaling, and node weighting, and allows you to just
specify the difference calculation.

.. code:: python

   from anemoi.training.losses.weightedloss import BaseLoss

   class MyLossFunction(FunctionalLoss):
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
      _target_: anemoi.training.losses.combined.CombinedLoss
      losses:
         - __target__: anemoi.training.losses.mse.WeightedMSELoss
         - __target__: anemoi.training.losses.mae.WeightedMAELoss
      scalers: ['variable']
      loss_weights: [1.0,0.5]
      scalars: ['variable']

All extra kwargs passed to ``CombinedLoss`` are passed to each of the
loss functions, and the loss weights are used to scale the individual
losses before combining them.

If ``scalars`` is not given in the underlying loss functions, all the
scalars given to the ``CombinedLoss`` are used.

If different scalars are required for each loss, the root level scalars
of the ``CombinedLoss`` should contain all the scalars required by the
individual losses. Then the scalars for each loss can be set in the
individual loss config.

.. code:: yaml

   training_loss:
      _target_: anemoi.training.losses.combined.CombinedLoss
      losses:
            - _target_: anemoi.training.losses.mse.WeightedMSELoss
              scalars: ['variable']
            - _target_: anemoi.training.losses.mae.WeightedMAELoss
              scalars: ['loss_weights_mask']
      loss_weights: [1.0, 1.0]
      scalars: ['*']

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
scalers over the same dimensions, and is only constructed at
broadcasting time, so the shape can be resolved to match the tensor
exactly.

.. automodule:: anemoi.training.losses.utils
   :members:
   :no-undoc-members:
   :show-inheritance:
