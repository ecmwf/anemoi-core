#############
 Diagnostics
#############

The diagnostics module in anemoi-training is used to monitor progress
during training. It is split into two parts:

   #. tracking training to a standard machine learning tracking tool.
      This monitors the training and validation losses and uploads the
      plots created by the callbacks.

   #. a series of callbacks, evaluated on the validation dataset,
      including plots of example forecasts and power spectra plots;

**Trackers**

By default, anemoi-training uses MLFlow tracker, but it includes
functionality to use both Weights & Biases and Tensorboard.

**Callbacks**

The callbacks can also be used to evaluate forecasts over longer
rollouts beyond the forecast time that the model is trained on. The
number of rollout steps (or forecast iteration steps) is set using
``config.eval.rollout = *num_of_rollout_steps*``.

Note the user has the option to evaluate the callbacks asynchronously
(using the following config option
``config.diagnostics.plot.asynchronous``, which means that the model
training doesn't stop whilst the callbacks are being evaluated).
However, note that callbacks can still be slow, and therefore the
plotting callbacks can be switched off by setting
``config.diagnostics.plot.enabled`` to ``False`` or all the callbacks
can be completely switched off by setting
``config.diagnostics.eval.enabled`` to ``False``.

Below is the documentation for the default callbacks provided, but it is
also possible for users to add callbacks using the same structure:

.. automodule:: anemoi.training.diagnostics.callbacks
   :members:
   :no-undoc-members:
   :show-inheritance:
