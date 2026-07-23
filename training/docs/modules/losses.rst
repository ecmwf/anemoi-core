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

-  ``CRPS``: Kernel CRPS loss for ensemble predictions. ``alpha=0`` gives
   standard CRPS, ``alpha=1`` gives fair CRPS, and values between 0 and 1
   give the almost fair CRPS formulation (`Lang et al. (2024)
   <http://arxiv.org/abs/2412.15832>`_). The default ``alpha: 0.95``
   combines 5% standard CRPS with 95% fair CRPS.
   The ``backend`` option can be set to:

   - ``naive``: simple loop over unordered ensemble-member pairs, avoiding
     materialization of the full pairwise tensor.
   - ``stable``: materializes pairwise tensors and uses the numerically
     stable all-pairs formulation.
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
            _target_: anemoi.training.losses.CRPS
            # loss function kwargs here

.. _multiscale-loss-functions:

***************************
 Time Aggregate Loss Functions
***************************

These loss functions encourage the model to produce **temporally consistent** outputs
i.e. output sequences that are internally coherent over
time, not just accurate at each individual step.

:class:`~anemoi.training.losses.aggregate.TimeAggregateLossWrapper`
addresses this by applying a base loss function to *time-aggregated*
versions of the prediction and target, rather than step-by-step. The
following aggregations are supported:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   -  -  Aggregation
      -  Description

   -  -  ``mean``
      -  Mean over the output time window — penalises bias in the
         temporal average.

   -  -  ``max``
      -  Maximum over the output time window — penalises errors in peak
         values.

   -  -  ``min``
      -  Minimum over the output time window — penalises errors in
         minimum values.

   -  -  ``diff``
      -  Consecutive step-to-step differences
         (``pred[:, 1:] - pred[:, :-1]``) — penalises unrealistic
         temporal transitions and discontinuities.

The wrapper accumulates the specified loss function evaluated on each aggregation in
turn and returns the average. Because the ``time_steps`` scaler is
intentionally excluded from the inner ``loss_fn`` (temporal aggregation
collapses the time dimension), only spatial and variable scalers should
be listed there.

.. note::

   ``TimeAggregateLossWrapper`` requires an output time dimension
   greater than one, as it is not
   meaningful for single-step tasks.

We strongly recommend using the time aggregate loss when training any
temporal downscaler. The pre-built config variants ``single_MSE_aggregation``
and ``ensemble_multiscale_aggregation`` combine it with the primary loss inside a
:class:`~anemoi.training.losses.combined.CombinedLoss`.

***************************
 Multiscale Loss Functions
***************************

The ``MultiscaleLossWrapper`` implements the multiscale loss formulation
presented in <https://arxiv.org/abs/2506.10868>. It wraps any base loss
(e.g. ``CRPS``) and evaluates it at multiple spatial scales by
progressively smoothing both predictions and targets. Each scale loss is
computed on the *residual* between successive smoothing levels, so
coarser scales capture large-scale errors and finer scales capture
small-scale structure.

The number of weights must equal the number of smoothing levels. A final
``null`` entry in ``loss_matrices`` (or the implicit full-resolution
scale appended when using on-the-fly generation) represents the
unsmoothed field.

All smoothing configuration is provided through the single
``multiscale_config`` key, which supports two modes:

On-the-fly mode (builds smoothing matrices from the graph at runtime):

.. code:: yaml

   training_loss:
      datasets:
         your_dataset_name:
            _target_: anemoi.training.losses.MultiscaleLossWrapper
            weights: [0.5, 0.25, 0.15, 0.1]   # num_scales + 1 entries
            multiscale_config:
               num_scales: 3                   # 3 smoothed + 1 full-res appended automatically
               base_num_nearest_neighbours: 4
               base_sigma: 0.1
               scale_factor: 2                 # neighbours and sigma double each level
            per_scale_loss:
               _target_: anemoi.training.losses.CRPS
               scalers: ['node_weights']
               ignore_nans: False
               no_autocast: True
               alpha: 0.95

File-based mode (load precomputed ``.npz`` matrices from disk):

.. code:: yaml

   training_loss:
      datasets:
         your_dataset_name:
            _target_: anemoi.training.losses.MultiscaleLossWrapper
            weights: [0.5, 0.25, 0.15, 0.1]   # must match number of loss_matrices entries
            multiscale_config:
               loss_matrices_path: /path/to/truncation-matrices
               loss_matrices:
                  - filter_O96_w=gaussian_d=8.0x.npz   # coarsest scale
                  - filter_O96_w=gaussian_d=4.0x.npz
                  - filter_O96_w=gaussian_d=2.0x.npz
                  - null                                # full resolution (no smoothing)
            per_scale_loss:
               _target_: anemoi.training.losses.CRPS
               scalers: ['node_weights']
               ignore_nans: False
               no_autocast: True
               alpha: 1.0

.. note::

   The top-level ``loss_matrices_path`` and ``loss_matrices`` kwargs are
   still accepted for backward compatibility but are deprecated. Move
   them inside ``multiscale_config``.

************************
Spectral loss functions
************************

Some loss functions operate in spectral space rather than directly in grid-point space.
This is useful when the error characteristics are better expressed by scale (wavenumber)
than by location, or when the loss should emphasise/regularise specific ranges of scales.

In Anemoi, spectral losses follow the same API as other losses (scalers and reduction),
but they additionally require a *spectral transform* configuration. Spatial node weights
cannot be used after the transform; use a spectral-dimension scaler instead.

Spectral transforms
===================


Spectral losses rely on a transform that maps grid-point fields to spectral coefficients.

Supported transforms include:

* ``FFT2D``: 2D FFT for regular latitude/longitude grids (or any regular 2D field) with
  known ``x_dim`` and ``y_dim``.
* ``DCT2D``: 2D Discrete Cosine Transform for regular 2D fields. This transform requires
  the optional dependency ``torch-dct``.
* ``ReducedSHT``: Spherical harmonic transform (SHT) on ECMWF's traditional reduced Gaussian grid. This can handle the
  native grid of ERA5 such as N320.
* ``OctahedralSHT``: Spherical harmonic transform (SHT) on the octahedral reduced Gaussian grid.

.. note::

   SHT-based transforms expect a flattened reduced-grid ordering:
   ``[batch, ensemble, grid_points, variables]`` and return spectral coefficients with
   shape ``[batch, ensemble, l, m, variables]`` where ``l = truncation + 1``.

.. note::

   ``ReducedSHT`` and ``OctahedralSHT`` both perform a spherical harmonic transform on a reduced Gaussian grid.
   By default, a naive Fourier transform is performed in the meridional direction which is very inefficient when
   executed on GPUs. Therefore an optimised version using graphs is provided, which can be switched on by setting
   ``use_graphed_rfft=True`` in the section of the config file corresponding to your spectral loss. This can provide
   significant speedups, but may not be supported on all devices and can have higher memory usage.

.. note::

   Before the transform is applied the grid can be subset to a subgrid by setting the optional ``subgrid`` argument.
   This can be a slice represented by a tuple, e.g. ``(0, 100)``, to select the first 100 gridpoints, or the string ``output_mask``
   that will restrict the grid to the region specified by the ``output_mask`` of LAM models.

   ``subgrid`` is only supported for the Cartesian transforms (``FFT2D`` / ``DCT2D``). Spherical harmonic
   transforms (``ReducedSHT`` / ``OctahedralSHT``) compute the spectra over the whole domain and reject an
   explicit ``subgrid``.

   For example, to restrict an ``FFT2D`` loss to the first 100 gridpoints (a tuple) or to the LAM output
   region (``output_mask``):

   .. code-block:: yaml

      training_loss:
        datasets:
          your_dataset_name:
            _target_: anemoi.training.losses.spectral.LogSpectralDistance
            transform: fft2d
            x_dim: 10
            y_dim: 10
            subgrid: [0, 100]   # or, for a LAM model:  subgrid: output_mask


Spectral projections
--------------------

Before the spectral transform is applied, but after the grid has been subset, an optional sparse projection can remap
the input field from its native (possibly unstructured) grid to the regular 2D grid
expected by the transform. This is configured via the ``projection_config`` key and
works with *any* spectral loss class (``SpectralCRPSLoss``,
``LogSpectralDistance``, ``FourierCorrelationLoss``, …).

Two modes are available:

- **From file** (``matrix_path``): load a precomputed sparse projection matrix from
  an ``.npz`` file. This is the most efficient option when the same projection is
  reused across many training runs.
- **From graph**: derive the projection at training startup from the model graph.
  The target grid can come from an existing edge set (``edges_name``) or be built
  from scratch using any ``anemoi.graphs`` node builder (``node_builder`` +
  ``num_nearest_neighbours`` + ``sigma``).

Example: spectral CRPS with a precomputed projection matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The typical setup for a limited-area model whose native grid is unstructured: the
projection matrix (generated offline) maps grid points to the ``[y_dim, x_dim]``
regular array expected by FFT2D.

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.spectral.SpectralCRPSLoss
         transform: fft2d
         x_dim: 256
         y_dim: 128
         projection_config:
           matrix_path: /path/to/projection.npz
         # subgrid: [0, 32768] # stretched-grid case, need to select y*x points first

Example: spectral L2 loss with a graph-derived projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The projection can also be built at training startup directly from the model graph.
Use ``edges_name`` to reuse an existing edge set, or ``node_builder`` to define the
target grid from scratch (here a regular lat/lon grid) and let Anemoi compute
Gaussian-weighted nearest-neighbour weights.

.. code-block:: yaml

   # Option A: reuse an existing graph edge set
   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.spectral.LogSpectralDistance  # any spectral loss
         transform: fft2d
         x_dim: 256
         y_dim: 128
         projection_config:
           edges_name: data/to/target_grid  # "src/rel/dst" or [src, rel, dst]

.. code-block:: yaml

   # Option B: build the target grid on the fly with a node builder
   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.spectral.LogSpectralDistance  # any spectral loss
         transform: fft2d
         x_dim: 256
         y_dim: 128
         projection_config:
           node_builder:
             _target_: anemoi.graphs.nodes.LatLonNodes
             # latitudes/longitudes define the regular target grid, e.g.:
             #   import numpy as np
             #   lats = np.repeat(np.linspace(90, -90, y_dim), x_dim)
             #   lons = np.tile(np.linspace(0, 360, x_dim, endpoint=False), y_dim)
             latitudes: [...]   # y_dim * x_dim values
             longitudes: [...]  # y_dim * x_dim values
             name: projection_target
           num_nearest_neighbours: 4
           sigma: 0.5
           row_normalize: false

Spectral kernel CRPS
====================

``SpectralCRPSLoss`` computes a CRPS-style probabilistic loss in spectral space.
Conceptually, it applies a spectral transform to both forecast ensemble and target,
then evaluates a kernel-CRPS over the resulting spectral representation (typically
interpreted as scale-dependent coefficients).

This loss is intended for *ensemble* training (``ensemble > 1``). For deterministic
training, consider spectral distance losses instead.

Example configuration (FFT2D)
-----------------------------

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

Example configuration (reduced Gaussian grid SHT)
-------------------------------------------------

Use this for global models on the reduced Gaussian grid (only N320 supported):

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.SpectralCRPSLoss
         transform: reduced_sht
         grid: n320

Truncation is by default set to 319 for n320 grids, but can be set to a higher or lower value in the config file.
This truncation parameter defines how many wave numbers are included in the spectral representation.

Power Spectrum Loss
===================

``PowerSpectrumLoss`` (PSL) is a spectral loss that compares
the *power spectrum* (energy per total wavenumber) of the prediction against
that of the target, rather than comparing complex spectral coefficients
directly. This emphasises that the model reproduces the correct distribution
of variance across spatial scales.

Given spectral coefficients :math:`\hat{F}_{lm}` (prediction) and
:math:`F_{lm}` (target), with total wavenumber :math:`l` and zonal
wavenumber :math:`m`, the loss is

.. math::

   \mathcal{L} = \sum_l \Bigl( \sum_m |\hat{F}_{lm}|^2
                              - \sum_m |F_{lm}|^2 \Bigr)^{2}.

The chosen spectral transform must provide a ``power_spectral_density``
method, so ``PowerSpectrumLoss`` currently supports the SHT-based
transforms (``reduced_sht``, ``octahedral_sht``). For these,
:math:`l` is the total wavenumber and :math:`m` the zonal wavenumber.

.. note::

   Because the loss operates on power per wavenumber, any scaler registered
   on the grid dimension must be a
   :class:`~anemoi.training.losses.scalers.SpectralDimensionScaler` (sized to
   the spectral dimension), not a spatial weight such as ``node_weights``.

Example configuration (octahedral SHT)
--------------------------------------

.. code-block:: yaml

   training_loss:
     datasets:
       your_dataset_name:
         _target_: anemoi.training.losses.PowerSpectrumLoss
         transform: octahedral_sht
         truncation: 192
         nlat: 192
         scalers: ['pressure_level', 'general_variable', 'spectral_dim_mean']

Since a power-spectrum-only loss does not constrain the *phase* of the
field, ``PowerSpectrumLoss`` is most useful when combined with a
grid-point loss (e.g. ``MSELoss``) via ``CombinedLoss``, where it acts as
a spectral regulariser that discourages spectral blurring.

Combining spectral and grid-point losses
========================================

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
---------------

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

Spectral Loss Scalers
=====================

Spectral scalers optionally weight or normalise the spectral dimension of
spectral losses. Without one, reduction sums over spectral modes. In spectral
space the grid dimension holds spectral modes rather than grid points, so
spatial scalers such as ``node_weights`` are not valid even when their length
happens to equal the number of spectral modes.

Anemoi-training records whether each grid scaler applies to spatial points or
spectral modes and rejects a scaler from the wrong domain when it is attached
to a loss. It also checks the scaler length against the global spectral
dimension at runtime. During distributed execution, the global spectral scaler
is sliced to the modes owned by each rank after the spectral all-to-all
transpose.

``n_spectral_modes`` is the number of spectral modes (for spherical
harmonic transforms this is the number of total wavenumbers
``L = truncation + 1``) considered by the loss.

.. note::

   You must know the size of the spectral dimension (``spectral_dims``) produced by your
   loss before configuring these scalers. A mismatch between
   the shape of the scaler and the loss tensor will be caught at
   runtime by the spectral-loss compatibility check, but only after
   model instantiation. The configured size is always the global spectral
   size, including when the loss tensor is sharded across ranks.

The following spectral scaler is available:

-  :class:`~anemoi.training.losses.scalers.SpectralDimensionScaler`:
   uniform scaling by ``1 / n_spectral_modes`` of ``spectral_dims``-dimensional tensor.

Example: averaging over total wavenumbers for an SHT-based loss that
reduces over the zonal wavenumber (e.g. ``PowerSpectrumLoss`` or
``SpectralAMSELoss``) with ``truncation = 192``
(so :math:`L = 193`):

.. code:: yaml

   # config.training.scalers
   spectral_dim_mean:
      _target_: anemoi.training.losses.scalers.SpectralDimensionScaler
      n_spectral_modes: 193

Then reference it from the loss:

.. code:: yaml

   training_loss:
      datasets:
         your_dataset_name:
            _target_: anemoi.training.losses.PowerSpectrumLoss
            transform: octahedral_sht
            truncation: 192
            nlat: 192 # o96 grid
            scalers: ['pressure_level', 'general_variable', 'spectral_dim_mean']

Custom Scalers
==============

To create a custom scaler, subclass the ``BaseScaler`` and implement the
``get_scaling_values`` method. This method should return an array of the
scaling values. Set ``scale_dims`` to the dimensions that the scaling
values should be applied to. A scaler that includes ``TensorDim.GRID`` must
also set ``grid_domain`` to ``ScalerDomain.SPATIAL`` or
``ScalerDomain.SPECTRAL``.

.. code:: python

   from anemoi.training.losses.scalers import ScalerDomain
   from anemoi.training.losses.scalers.base_scaler import BaseScaler
   from anemoi.training.utils.enums import TensorDim

   class CustomScaler(BaseScaler):
      scale_dims = [TensorDim.GRID]
      grid_domain = ScalerDomain.SPATIAL

      def get_scaling_values(self):
         # Custom scaling logic here
         return scaling_values

This scaler will only be instantiated once at the start of training, and
thus cannot adapt throughout batches and epochs.

Custom Updating Scalers
-----------------------

If you want a scaler that adapts throughout the training process, you
can subclass the ``BaseUpdatingScaler``.

As with the ``BaseScaler``, set the initial scaler values at the start
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
name.

Validation metrics can **not** be scaled by scalers across
the variable dimension, but can be by all other scalers.

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

All extra kwargs passed to ``CombinedLoss`` are passed to each of the
loss functions, and the loss weights are used to scale the individual
losses before combining them.

If ``scalers`` is not given in the underlying loss functions, all the
scalers given to the ``CombinedLoss`` are used.

If different scalers are required for each loss, the root level scalers
of the ``CombinedLoss`` should contain all the scalers required by the
individual losses. Then the scalers for each loss can be set in the
individual loss config.

.. code:: yaml

   training_loss:
      datasets:
         your_dataset_name:
            _target_: anemoi.training.losses.combined.CombinedLoss
            losses:
                  - _target_: anemoi.training.losses.mse.WeightedMSELoss
                  scalers: ['variable']
                  - _target_: anemoi.training.losses.mae.WeightedMAELoss
                  scalers: ['loss_weights_mask']
            loss_weights: [1.0, 1.0]
            scalers: ['*']

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
