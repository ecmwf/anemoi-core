###############
 Preprocessing
###############

The preprocessing module is used to pre- and post-process the data.
Preprocessors are applied to the input data before it is passed to the
model, and postprocessors are applied to the output data after it has
been produced by the model and (in training) after the training loss has
been calculated. The module contains the following classes:

.. automodule:: anemoi.models.preprocessing
   :members:
   :no-undoc-members:
   :show-inheritance:

************
 Normalizer
************

The normalizer module is used to normalize the data. The module contains
the following classes:

.. automodule:: anemoi.models.preprocessing.normalizer
   :members:
   :no-undoc-members:
   :show-inheritance:

*********
 Imputer
*********

Machine learning models cannot process **missing values (NaNs)**
directly, so missing values in input data and the target must be handled
before being handled by the model. The **Imputer** module in
anemoi-models handles missing values (NaNs) before the data is input to
the model and after the model's output is handled by the training loss.

For each input batch, the module identifies NaN locations and replaces
the NaNs with a configured imputation value, as specified in the
configuration file. If a variable is present in the output data, the
imputed values are restored to NaN at the original NaN locations from
the first timestep of the input.

The imputer provides the nan mask as a **loss scaler**
``anemoi.training.losses.scalers.loss_weights_mask.NaNMaskScaler`` to
the loss function, if the scaler is included in
``config.training.training_loss``. Then the training loss function uses
the nan mask to ignore the imputed values in the loss calculation. This
mask is updated for every batch during training.

During training, diagnostic variables are included in each batch, and
therefore at the input timesteps. Any NaNs in the target data are
weighted by zero to enable proper loss computation. During inference,
however, NaN locations for diagnostic variables are not available (those
fields aren not part of the model input) so the imputer cannot
reintroduces NaNs into the diagnostic output. To insert NaNs into
diagnostic variables, the postprocessor
``anemoi.models.preprocessing.postprocessor.ConditionalNaNPostprocessor``
has to be used. This masks diagnostic variable entries by setting them
to NaN wherever the chosen (prognostic) masking variable is NaN.

The dynamic imputers are used to impute NaNs in the input data and do
not replace the imputed values with NaNs in the output data. Therefore,
the nan mask is not provided as a scaler to the loss function either.

The module contains the following classes:

.. automodule:: anemoi.models.preprocessing.imputer
   :members:
   :no-undoc-members:
   :show-inheritance:

****************
 ZeroOverwriter
****************

Overwrite selected timesteps of specified input variables with zero.

This preprocessor operates on inputs before the model and is
model-independent. It is useful whenever a variable should be reset to
zero at certain timesteps within each input window (for example, for
accumulated or windowed variables).

Example
=======

To include in data config (Hydra/YAML) to set var_a and var_b to zero at
timesteps 0 and 3, and var_c and var_d to zero at timesteps 0 and 4.

.. code:: yaml

   processors:
     zero_overwriter:
       _target_: anemoi.models.preprocessing.overwriter.ZeroOverwriter
       config:
         groups:
           - vars:
               - "var_a"
               - "var_b"
             time_indices: [0, 3]
           - vars:
               - "var_c"
               - "var_d"
             time_indices: [0, 4]

API
===

.. autoclass:: anemoi.models.preprocessing.overwriter.ZeroOverwriter
   :members:
   :no-undoc-members:
   :show-inheritance:
