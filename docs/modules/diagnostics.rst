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
number of rollout steps for verification (or forecast iteration steps)
is set using ``config.dataloader.validation_rollout =
*num_of_rollout_steps*``.

Callbacks are configured in the config file under the
``config.diagnostics`` key.

For regular callbacks, they can be provided as a list of dictionaries
underneath the ``config.diagnostics.callbacks`` key. Each dictionary
must have a ``_target`` key which is used by hydra to instantiate the
callback, any other kwarg is passed to the callback's constructor.

.. code:: yaml

   callbacks:
      - _target_: anemoi.training.diagnostics.callbacks.evaluation.RolloutEval
      rollout: ${dataloader.validation_rollout}
      frequency: 20

Plotting callbacks are configured in a similar way, but they are
specified underneath the ``config.diagnostics.plot.callbacks`` key.

This is done to ensure seperation and ease of configuration between
experiments.

``config.diagnostics.plot`` is a broader config file specifying the
parameters to plot, as well as the plotting frequency, and
asynchronosity.

Setting ``config.diagnostics.plot.asynchronous``, means that the model
training doesn't stop whilst the callbacks are being evaluated. This is
useful for large models where the plotting can take a long time. The
plotting module uses asynchronous callbacks via `asyncio` and
`concurrent.futures.ThreadPoolExecutor` to handle plotting tasks without
blocking the main application. A dedicated event loop runs in a separate
background thread, allowing plotting tasks to be offloaded to worker
threads. This setup keeps the main thread responsive, handling
plot-related tasks asynchronously and efficiently in the background.

There is an additional flag in the plotting callbacks to control the
rendering method for geospatial plots, offering a trade-off between
performance and detail. When `datashader` is set to True, Datashader is
used for rendering, which accelerates plotting through efficient
hexbining, particularly useful for large datasets. This approach can
produce smoother-looking plots due to the aggregation of data points. If
`datashader` is set to False, matplotlib.scatter is used, which provides
sharper and more detailed visuals but may be slower for large datasets.

**Note** - this asynchronous behaviour is only available for the
plotting callbacks.

.. code:: yaml

   plot:
      asynchronous: True # Whether to plot asynchronously
      datashader: True # Whether to use datashader for plotting (faster)
      frequency: # Frequency of the plotting
      batch: 750
      epoch: 5

      # Parameters to plot
         parameters:
         - z_500
         - t_850
         - u_850

         # Sample index
         sample_idx: 0

         # Precipitation and related fields
         precip_and_related_fields: [tp, cp]

         callbacks:
         - _target_: anemoi.training.diagnostics.callbacks.plot.PlotLoss
            # group parameters by categories when visualizing contributions to the loss
            # one-parameter groups are possible to highlight individual parameters
            parameter_groups:
               moisture: [tp, cp, tcw]
               sfc_wind: [10u, 10v]
         - _target_: anemoi.training.diagnostics.callbacks.plot.PlotSample
            sample_idx: ${diagnostics.plot.sample_idx}
            per_sample : 6
            parameters: ${diagnostics.plot.parameters}

Below is the documentation for the default callbacks provided, but it is
also possible for users to add callbacks using the same structure:

.. automodule:: anemoi.training.diagnostics.callbacks.checkpoint
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.diagnostics.callbacks.evaluation
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.diagnostics.callbacks.optimiser
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.diagnostics.callbacks.plot
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.diagnostics.callbacks.provenance
   :members:
   :no-undoc-members:
   :show-inheritance:
