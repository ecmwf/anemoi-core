###############
 Preprocessing
###############

The preprocessing module is used to pre- and post-process the data. The
module contains the following classes:

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

The imputer module is used to impute NaNs. For every input batch, the
module extracts the NaN locations and replaces the NaNs with a value
that is specified in the configuration file. In the output data, the
imputed values are replaced with NaNs again at the locations of NaNs in
the input data at the first timestep.

The imputer can provide the nan mask as a loss scaler
``anemoi.training.losses.scalers.loss_weights_mask.NaNMaskScaler`` to
the loss function. Then the loss function uses the nan mask to ignore
the imputed values in the loss calculation. This mask is updated for
every batch during training.

For diagnostic variables, in inference no NaN locations are available in
the input data, so no NaNs are introduced to diagnostic output fields.

The dynamic imputers are used to impute NaNs in the input data and do
not replace the imputed values with NaNs in the output data.

The module contains the following classes:

.. automodule:: anemoi.models.preprocessing.imputer
   :members:
   :no-undoc-members:
   :show-inheritance:
