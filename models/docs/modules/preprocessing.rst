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

For diagnostic variables, NaN locations are not available during
inference, as these fields are not included in the model input. As a
result, no NaNs are reintroduced into the diagnostic output fields. In
contrast, during training, diagnostic variables are included in the
batch, also for input timesteps. Therefore, any NaNs in the target data
are imputed to enable proper loss computation.

The dynamic imputers are used to impute NaNs in the input data and do
not replace the imputed values with NaNs in the output data. Therefore,
the nan mask is not provided as a scaler to the loss function either.

The module contains the following classes:

.. automodule:: anemoi.models.preprocessing.imputer
   :members:
   :no-undoc-members:
   :show-inheritance:
