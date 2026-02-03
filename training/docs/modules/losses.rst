########
 Losses
########

This module is used to define the loss function used to train the model.

Anemoi-training exposes a couple of loss functions by default to be
used, all of which are subclassed from ``BaseLoss``. This class enables
scaler multiplication, and graph node weighting.

.. automodule:: anemoi.training.losses.base
   :members:
   :no-undoc-members:
   :show-inheritance:

******************************
 Deterministic Loss Functions
******************************

By default anemoi-training trains the model using a mean-squared-error,
which is defined in the ``MSELoss`` class in
``anemoi/training/losses/mse.py``. The loss function can be configured
in the config file at ``config.training.training_loss``, and
``config.training.validation_metrics``.

The following loss functions are available by default:

-  ``MSELoss``: mean-squared-error.
-  ``RMSELoss``: root mean-squared-error.
-  ``MAELoss``: mean-absolute-error.
-  ``HuberLoss``: Huber loss.
-  ``LogCoshLoss``: log-cosh loss.
-  ``CombinedLoss``: Combined component weighted loss.

All the above losses by default are averaged across the grid nodes,
ensemble dimension and batch size. Losses can also consider specific
weighting either spatial, vertical or specific to the variables used.
Those weights are defined via `scalers`. For example spatial scaling
based on the area of the nodes needs is done using the ``node_weights``
as a scaler. For more details on the loss function scaling please refer
to :ref:`loss-function-scaling`.

These are available in the ``anemoi.training.losses`` module, at
``anemoi.training.losses.{short_name}.{class_name}``.

So for example, to use the ``WeightedMSELoss`` class, you would
reference it in the config as follows:

.. code:: yaml

   # loss function for the model
   training_loss:
      datasets:
         your_dataset_name:
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
-  ``WeightedMSELoss`` : is the MSELoss used for the diffussion model to
   handle noise weights

The config for these loss functions is the same as for the
deterministic:

.. code:: yaml

   # loss function for the model
   training_loss:
      datasets:
         your_dataset_name:
            # loss class to initialise
            _target_: anemoi.training.losses.kcrps.KernelCRPSLoss
            # loss function kwargs here

***************************
 Multiscale Loss Functions
***************************

The `MultiscaleLossWrapper` implements the multiscale loss formulation
presented in <https://arxiv.org/abs/2506.10868>. It wraps around loss
functions such as the `AlmostFairKernelCRPSLoss` to provide scale-aware
model training.

The config for the multiscale loss functions is the following:

.. code:: yaml

   training_loss:
      datasets:
         your_dataset_name:
            _target_: anemoi.training.losses.MultiscaleLossWrapper
            loss_matrices_path: ${system.input.loss_matrices_path}
            loss_matrices: ["matrix.npz", null]
            weights:
               - 1.0
               - 1.0

            per_scale_loss:
               _target_: anemoi.training.losses.kcrps.AlmostFairKernelCRPS
               scalers: ['node_weights']
               ignore_nans: False
               no_autocast: True
               alpha: 1.0

************************
Spectral loss functions
************************

Some loss functions operate in spectral space rather than directly in grid-point space.
This is useful when the error characteristics are better expressed by scale (wavenumber)
than by location, or when the loss should emphasise/regularise specific ranges of scales.

In Anemoi, spectral losses follow the same API as other losses (scalars/node weights, reduction,
etc.), but they additionally require a *spectral transform* configuration.

Spectral transforms
-------------------

Spectral losses rely on a transform that maps grid-point fields to spectral coefficients.

Supported transforms include:

* ``FFT2D``: 2D FFT for regular latitude/longitude grids (or any regular 2D field) with
  known ``x_dim`` and ``y_dim``.
* ``DCT2D``: 2D Discrete Cosine Transform for regular 2D fields. This transform requires
  the optional dependency ``torch-dct``.
* ``EcTransOctahedralSHT``: Spherical Harmonic Transform (SHT) on the *octahedral reduced
  Gaussian grid* using ecTrans assets (via precomputed ``npz`` assets, or generated with
  ``ectrans4py`` if available).

.. note::

   SHT-based transforms expect a flattened reduced-grid ordering:
   ``[batch, ensemble, grid_points, variables]`` and return spectral coefficients with
   shape ``[batch, ensemble, l, m, variables]`` where ``l = truncation + 1``.

Spectral kernel CRPS
--------------------

``SpectralCRPSLoss`` computes a CRPS-style probabilistic loss in spectral space.
Conceptually, it applies a spectral transform to both forecast ensemble and target,
then evaluates a kernel-CRPS over the resulting spectral representation (typically
interpreted as scale-dependent coefficients).

This loss is intended for *ensemble* training (``ensemble > 1``). For deterministic
training, consider spectral distance losses instead.

Example configuration (FFT2D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this for limited-area or other regular 2D fields that can be reshaped to
``[y_dim, x_dim]``:

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.spectral.SpectralCRPSLoss
         # Transform selection / geometry
         transform: fft2d
         x_dim: 256
         y_dim: 128

Example configuration (octahedral SHT via ecTrans assets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this for global models on the octahedral reduced grid:

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.spectral.SpectralCRPSLoss
         transform: ectrans_octahedral_sht
         truncation: 127
         # Path to precomputed Legendre polynomials / weights.
         # If the file does not exist, assets can optionally be generated when `ectrans4py`
         # is installed and the path is writable.
         filepath: /path/to/ectrans_assets_T127.npz
         # Optional: control transform dtype
         dtype: float32

Combining spectral and grid-point losses
----------------------------------------

Spectral losses can be combined with standard grid-point losses through
``CombinedLoss``:

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.combined.CombinedLoss
         losses:
           - _target_: anemoi.training.losses.mse.WeightedMSELoss
           - _target_: anemoi.training.losses.spectral.SpectralCRPSLoss
             transform: fft2d
             x_dim: 256
             y_dim: 128
         loss_weights: [1.0, 0.1]


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
      datasets:
         your_dataset_name:
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
group `pl`, all temperature fields are considered group `pl`.

If the datasets are built from mars the variable reference is extracted
from metadata, otherwise it is found by splitting the variable name by
`_` and taking the first part, see class
`anemoi.training.utils.ExtractVariableGroupAndLevel`.

If more complex variable groups are required, it is possible to define
the group values as a dictionary, such that the variable's metadata must
contain the key and value pair. See
`anemoi.transforms.variable.Variable` for the metadata attributes that
are available.

.. code:: yaml

   variable_groups:
      datasets:
         your_dataset_name:
            default: sfc
            pl:
               is_pressure_level: True
            z_ml:
               is_model_level: True
               param: 'z'

The list of available metadata attributes is:

-  ``is_pressure_level``: whether the variable is a pressure level,
-  ``is_model_level``: whether the variable is a model level,
-  ``is_surface_level``: whether the variable is on the surface,
-  ``level``: the level of the variable,
-  ``is_constant_in_time``: whether the variable is constant in time,
-  ``is_instantanous``: whether the variable is instantaneous,
-  ``is_valid_over_a_period``: whether the variable is valid over a
   period,
-  ``time_processing``: the time processing type of the variable,
-  ``period``: the variable's period as a timedelta,
-  ``is_accumulation``: whether the variable is an accumulation,
-  ``param``: the parameter name of the variable,
-  ``grib_keys``: the GRIB keys for the variable,
-  ``is_computed_forcing``: whether if the variable is a computed
   forcing,
-  ``is_from_input``: whether the variable is from input.

For example, to set a different scaler coefficient for a particular
level, several groups can be defined:

.. code:: yaml

   variable_groups:
      datasets:
         your_dataset_name:
            default: sfc
            pl:
               is_pressure_level: True
            l_50:  # this needs to come first to take priority
               param: ["z"]
               level: [50]
            l:
               param: ["z"]

If metadata is not available, complex variable groups cannot be defined,
and an error will be raised.

If multiple groups are defined for a variable, the first group in the
`variable_groups` is used. If the variable is not in any group, it is
assigned to the default group.

Custom Scalars
==============

To create a custom scalar, subclass the ``BaseScaler`` and implement the
``get_scaling_values`` method. This method should return an array of the
scaling values. Set ``scale_dims`` to the dimensions that the scaling
values should be applied to.

.. code:: python

   from anemoi.training.losses.scalers import BaseScaler
   from anemoi.training.utils.enums import TensorDim

   class CustomScaler(BaseScaler):
      scale_dims = [TensorDim.GRID]
      def get_scaling_values(self):
         # Custom scaling logic here
         return scaling_values

This scalar will only be instantiated once at the start of training, and
thus cannot adapt throughout batches and epochs.

Custom Updating Scalars
-----------------------

If you want a scalar that adapts throughout the training process, you
can subclass the ``BaseUpdatingScaler``.

As with the ``BaseScaler``, set the initial scalar values at the start
of training by implementing the ``get_scaling_values`` method.
Currently, two callbacks to update at are available, at the start of
training, and at the start of every batch.

Implementing any of these updating methods will allow for the scaler
values to be changed at the specified point. If ``None`` is returned by
these methods, it indicates that the scaler values should not be updated
at that time.

An example of this updating scaler is the
``anemoi.training.losses.scalers.loss_weights_mask.NaNMaskScaler``,
which updates the loss weights based on the presence of NaN values in
the input.

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
   # calculated in the same way as the training loss, within the model space.
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
``FunctionalLoss`` and implementing the ``calculate_difference``
function. The latter abstracts the scaling, and node weighting, and
allows you to just specify the difference calculation.

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
      datasets:
         your_dataset_name:
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
      datasets:
         your_dataset_name:
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
