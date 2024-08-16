########
 Losses
########

This module is used to define the loss function used to train the model.
By default anemoi-training trains the model using a latitude-weighted
mean-squared-error, which is defined in the ``WeightedMSELoss`` class in
``aifs/losses/mse.py``.

The user can define their own loss function using the same structure as
the ``WeightedMSELoss`` class.

.. automodule:: anemoi.training.losses.mse
   :members:
   :no-undoc-members:
   :show-inheritance:

There is also generic functions that are useful for losses in
``aifs/losses/utils.py``.

``grad_scaler`` is used to automatically scale the loss gradients in the
loss function using the formula in https://arxiv.org/pdf/2306.06079.pdf,
section 4.3.2. This can be switched on in the config by setting the
option ``config.training.loss_gradient_scaling=True``.
