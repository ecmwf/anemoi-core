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
must have a ``_target_`` key which is used by hydra to instantiate the
callback, any other kwarg is passed to the callback's constructor.

.. code:: yaml

   callbacks:
      - _target_: anemoi.training.diagnostics.callbacks.evaluation.RolloutEval
      rollout:
      - ${dataloader.validation_rollout}
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

Plot adapter compatibility
==========================

Task-specific plot adapters normalize output handling so plotting
callbacks can use the same interface across task types:

- forecaster tasks use ``ForecasterPlotAdapter``;
- autoencoder tasks use ``AutoencoderPlotAdapter``;
- temporal downscaler tasks use ``TemporalDownscalerPlotAdapter``.

These adapters rely on the shared task ``_step`` return format
``(loss, metrics, predictions)`` where ``predictions`` is always a list
of dataset-keyed dictionaries.

**Focus Area**

Plotting callbacks (such as ``PlotSample`` and ``PlotLoss``) support a ``focus_area`` parameter. This allows you to restrict the geographic scope of plots to specific regions or masks. A focus area can be defined in two ways:

* **Mask Name**: A ``mask_attr_name`` string referencing a boolean mask defined within the graph data.
* **Lat/Lon Bounds**: A ``latlon_bbox`` list specifying a bounding box: ``[lat_min, lon_min, lat_max, lon_max]``.

When a focus area is applied, the plot filenames and experiment log tags will automatically include a suffix (e.g., ``_mask_attr_name`` or ``_latlon_bbox``) to distinguish them from global plots.

.. code:: yaml

   # Example: Focusing on multiple specific geographic region
   focus_areas:
      europe:
         latlon_bbox: [30.0, -20.0, 60.0, 40.0]
      china:
         latlon_bbox: [18.0, 73.0, 54.0, 135.0]

**Rendering Methods**

There is an additional flag in the plotting callbacks to control the
rendering method for geospatial plots, offering a trade-off between
performance and detail.

* When `datashader` is set to True, Datashader is
   used for rendering, which accelerates plotting through efficient
   hexbining, particularly useful for large datasets. This approach can
   produce smoother-looking plots due to the aggregation of data points.
* If `datashader` is set to False, matplotlib.scatter is used, which provides
   sharper and more detailed visuals but may be slower for large datasets.

**Projection**

Plotting callbacks also support ``config.diagnostics.plot.projection_kind``
to control the map projection used for geospatial figures.

- ``equirectangular`` (default): regular axes, no Cartopy dependency.
- ``lambert_conformal``: regional Lambert Conformal projection fitted to
  the plotted latitude/longitude domain (requires Cartopy).

When ``datashader: True`` is enabled, plotting is forced to
``equirectangular`` because Datashader rendering does not support
Cartopy transforms.

**Note** - this asynchronous behaviour is only available for the
plotting callbacks.

**Progress Bar**

The progress bar callback can be configured to control how training
progress is displayed. This is particularly useful on HPC systems with
SLURM where output is written to files, as the default RichProgressBar
in PyTorch Lightning 2.6+ may not work correctly. The progress bar is controlled by two configuration options:

-  ``enable_progress_bar``: A boolean flag to enable or disable the
   progress bar entirely
-  ``progress_bar``: Configuration for which progress bar callback to
   use

.. code:: yaml

   enable_progress_bar: True
   progress_bar:
     _target_: pytorch_lightning.callbacks.TQDMProgressBar
     refresh_rate: 1

Lightning 2.6+ supports the
(https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichProgressBar.html#lightning.pytorch.callbacks.RichProgressBar)[RichProgressBar],
which is recommended for interactive terminals and
(https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.TQDMProgressBar.html#lightning.pytorch.callbacks.TQDMProgressBar)[TQDMProgressBar]
, that should be used with SLURM.

.. code:: yaml

   plot:
      asynchronous: True # Whether to plot asynchronously
      datashader: True # Whether to use datashader for plotting (faster)
      projection_kind: equirectangular # or lambert_conformal (requires Cartopy)
      frequency: # Frequency of the plotting
      batch: 750
      epoch: 5

      # Parameters to plot
      parameters:
         - z_500
         - t_850
         - u_850

      # Sample index
      sample_idx: 0

      # Precipitation and related fields
      precip_and_related_fields: [tp, cp]

      datasets_to_plot: ["data"]

      focus_areas:
         europe:
            latlon_bbox: [30.0, -20.0, 60.0, 40.0]
         china:
            latlon_bbox: [18.0, 73.0, 54.0, 135.0]

      callbacks:
         - _target_: anemoi.training.diagnostics.callbacks.plot.PlotLoss
            dataset_names: ["your_dataset_name"]
            # group parameters by categories when visualizing contributions to the loss
            # one-parameter groups are possible to highlight individual parameters
            parameter_groups:
               moisture: [tp, cp, tcw]
               sfc_wind: [10u, 10v]

         - _target_: anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot
            tag_infix: sample
            dataset_names: ["your_dataset_name"]
            sample_idx: ${diagnostics.plot.sample_idx}
            parameters: ${diagnostics.plot.parameters}
            plot_fn:
               _target_: anemoi.training.diagnostics.evaluation.plotting.spatial_map.sample_plot_fn
               _partial_: true
               per_sample: 6

Pluggable spatial-map plots
===========================

Spatial map–style callbacks (samples, spectra, histograms, ensembles) are
all driven by a single :class:`~anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot`
callback that iterates over datasets, samples and focus areas, and delegates
the actual figure rendering to a pluggable ``plot_fn``.

The bundled adapters live in
``anemoi.training.diagnostics.evaluation.plotting.spatial_map``:

- ``sample_plot_fn`` — multi-level forecast sample plot (former ``PlotSample``).
- ``spectrum_plot_fn`` — power spectrum (former ``PlotSpectrum``).
- ``histogram_plot_fn`` — per-variable histograms (former ``PlotHistogram``).
- ``ensemble_plot_fn`` — ensemble member plot (former ``PlotEnsSample``).

Each ``plot_fn`` follows the
:class:`~anemoi.training.diagnostics.evaluation.plotting.spatial_map.SpatialMapPlotFn`
protocol and receives keyword-only arguments (``x``, ``y_true``, ``y_pred``,
``latlons``, ``auxiliary``, ``settings``, plus any plot-specific kwargs bound
in YAML via ``_partial_: true``).

.. code:: yaml

   # Same callback, different plot_fn → different figures
   - _target_: anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot
     tag_infix: spectrum
     sample_idx: 0
     parameters: ${diagnostics.plot.parameters}
     plot_fn:
       _target_: anemoi.training.diagnostics.evaluation.plotting.spatial_map.spectrum_plot_fn
       _partial_: true
       min_delta: 0.01

Plot function contracts
-----------------------

Three callbacks accept a pluggable ``plot_fn``. Each defines a fixed
keyword-only signature; anything else in the ``plot_fn:`` YAML block is
bound as a partial kwarg via ``_partial_: true`` and forwarded to the
function. All three contracts accept a ``settings`` argument (a
``PlottingSettings`` instance carrying ``datashader``, ``projection_kind``,
``colormaps``, ``precip_and_related_fields``, ``asynchronous``) and
``**kwargs`` so future additions do not break existing implementations.

Layers and data flow
....................

Each of the three pluggable callbacks (``PlotLoss``, ``SpatialMapPlot``,
``GraphTrainableFeaturesPlot``) has the same three-layer shape:

.. code:: text

    YAML config
      │  (Hydra + `_partial_: true` build a functools.partial)
      ▼
    Callback._plot(pl_module, ...)                     (callbacks/plot.py)
      │
      │  extract_<family>_inputs(pl_module, ...)       (evaluation/plotting/
      │  returns a dict of kwargs                       model_introspection.py)
      ▼
    self.plot_fn(**inputs, **extras)                   ← call site
      │
      │  signature declared by
      ▼
    <Family>PlotFn (Protocol)                          (evaluation/plotting/
                                                        {loss,graph,spatial_map}.py)

Concretely:

- ``Callback._plot`` is the Lightning glue. It reads the batch and calls
  the matching ``extract_*_inputs`` helper.
- ``model_introspection.extract_*_inputs`` is the **only** place that pokes
  at ``pl_module`` (data indices, metadata, graph). It returns a plain
  ``dict`` whose keys match the corresponding Protocol's kwargs.
- The callback splats the dict into ``plot_fn`` alongside per-step extras
  (loss array, ``x``/``y_true``/``y_pred``, ``settings``, …).
- The Protocol declares the ``plot_fn`` signature so mypy / IDEs can verify
  custom plot functions.

Family-to-artifact mapping:

+---------------------------------+----------------------------+--------------------------+
| Callback                        | ``extract_*_inputs`` helper| Protocol                 |
+=================================+============================+==========================+
| ``PlotLoss``                    | ``extract_loss_inputs``    | ``LossPlotFn``           |
+---------------------------------+----------------------------+--------------------------+
| ``SpatialMapPlot``              | ``extract_spatial_inputs`` | ``SpatialMapPlotFn``     |
+---------------------------------+----------------------------+--------------------------+
| ``GraphTrainableFeaturesPlot``  | ``extract_graph_inputs``   | ``GraphPlotFn``          |
+---------------------------------+----------------------------+--------------------------+

``SpatialMapPlot`` — per-sample, per-dataset figures
....................................................

Callback: :class:`~anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot`.
Protocol:
:class:`~anemoi.training.diagnostics.evaluation.plotting.spatial_map.SpatialMapPlotFn`.

.. code:: python

   def plot_fn(
       parameters: dict[int, tuple[str, bool]],
       *,
       x: np.ndarray,                       # (n_gridpoints, n_input_vars)
       y_true: np.ndarray | None,           # (n_gridpoints, n_output_vars) or None
       y_pred: np.ndarray,                  # (n_gridpoints, n_output_vars)
       latlons: np.ndarray | None = None,   # (n_gridpoints, 2), [lat, lon]
       auxiliary: np.ndarray | None = None, # only if with_auxiliary=True
       settings: PlottingSettings | None = None,
       **kwargs,                            # plot-specific kwargs from YAML
   ) -> matplotlib.figure.Figure: ...

Contract notes:

- ``parameters`` is a mapping ``{output_index: (variable_name, is_diagnostic)}``
  restricted to the intersection of ``diagnostics.plot.parameters`` and the
  model's output variables.
- The sample dimension and any leading batch/rollout dims have already been
  reduced by ``SpatialMapPlot``'s ``process`` step (using ``sample_idx``).
  ``y_true`` / ``y_pred`` are 2-D arrays over grid points × output variables.
- ``latlons`` is pre-masked to the active ``focus_area`` when one is set.
- Return a ``matplotlib.figure.Figure`` — returning ``None`` is not allowed.

``PlotLoss`` — grouped per-variable loss bar chart
..................................................

Callback: :class:`~anemoi.training.diagnostics.callbacks.plot.PlotLoss`.
Protocol:
:class:`~anemoi.training.diagnostics.evaluation.plotting.loss.LossPlotFn`.
Default: ``anemoi.training.diagnostics.evaluation.plotting.loss.loss_plot_fn``.

.. code:: python

   def plot_fn(
       loss: np.ndarray,                      # (n_parameters,), in model-output order
       *,
       parameter_names: list[str],            # names in model-output order
       parameter_groups: dict[str, list[str]] | None = None,
       metadata_variables: dict | None = None,  # from model.metadata["dataset"]
       settings: PlottingSettings | None = None,
       **kwargs,
   ) -> matplotlib.figure.Figure: ...

Contract notes:

- ``PlotLoss`` supplies the **raw** per-variable loss in the model's output
  order together with the naming/grouping context, and does **not** apply any
  presentation-specific reordering, grouping or colouring itself.
- The default ``loss_plot_fn`` reproduces the historic behaviour by calling
  :func:`argsort_variablename_variablelevel` (sort by variable + level),
  then :func:`sort_and_color_by_parameter_group` (group + colour), then
  :func:`plot_loss` (bar-chart render). A custom ``plot_fn`` is free to
  replace or skip any of these steps.
- ``loss`` and ``parameter_names`` share the same length and ordering.

``GraphTrainableFeaturesPlot`` — graph node/edge feature plots
..............................................................

Callback:
:class:`~anemoi.training.diagnostics.callbacks.plot.GraphTrainableFeaturesPlot`.
Protocol:
:class:`~anemoi.training.diagnostics.evaluation.plotting.graph.GraphPlotFn`.
Default: ``anemoi.training.diagnostics.evaluation.plotting.graph.graph_plot_fn``.

.. code:: python

   def plot_fn(
       *,
       dataset_name: str,
       node_attributes: NamedNodesAttributes,
       node_trainable_tensors: dict[str, torch.Tensor],
       edge_trainable_modules: dict[tuple[str, str], torch.nn.Module],
       q_extreme_limit: float = 0.05,
       settings: PlottingSettings | None = None,
       **kwargs,
   ) -> Iterable[tuple[matplotlib.figure.Figure, str]]: ...

Contract notes:

- ``plot_fn`` is a **generator** (or any iterable) yielding
  ``(figure, tag)`` pairs. This lets a single call emit multiple figures
  (e.g. one for node features, one for edge features) under distinct
  logging tags.
- ``GraphTrainableFeaturesPlot`` extracts the graph artifacts once via
  ``extract_graph_inputs`` (unwrapping any DDP wrapper) and passes them
  as keyword arguments; the plot function never sees the raw model.
- ``edge_trainable_modules`` is empty for hierarchical models (they carry
  no trainable edge parameters); ``node_trainable_tensors`` is empty when
  no trainable node attributes are defined. The default ``graph_plot_fn``
  logs a warning and skips the corresponding figure in that case.

Adding a new spatial-map plot
-----------------------------

To add a new plot type, write a function matching the ``SpatialMapPlotFn``
signature and reference it from YAML — no new callback class or Pydantic
schema is required.

1. Implement the plot function (in your project or in
   ``anemoi.training.diagnostics.evaluation.plotting``):

   .. code:: python

      # my_project/plots.py
      import matplotlib.pyplot as plt
      import numpy as np
      from matplotlib.figure import Figure


      def bias_map_plot_fn(
          parameters: dict[int, tuple[str, bool]],
          *,
          x,
          y_true,
          y_pred,
          latlons,
          auxiliary=None,
          settings=None,
          cmap: str = "RdBu_r",
          vmax: float | None = None,
          **_kwargs,
      ) -> Figure:
          """Scatter-map of ``y_pred - y_true`` per plotted parameter."""
          fig, axes = plt.subplots(1, len(parameters), figsize=(4 * len(parameters), 3))
          axes = np.atleast_1d(axes)
          bias = np.asarray(y_pred) - np.asarray(y_true)
          for ax, (idx, (name, _)) in zip(axes, parameters.items()):
              limit = vmax if vmax is not None else np.nanpercentile(np.abs(bias[..., idx]), 99)
              ax.scatter(
                  latlons[:, 1], latlons[:, 0],
                  c=bias[..., idx], cmap=cmap, vmin=-limit, vmax=limit, s=1,
              )
              ax.set_title(name)
          return fig

2. Wire it in via the ``plot_fn`` block (any additional keys are bound as
   partial kwargs):

   .. code:: yaml

      - _target_: anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot
        tag_infix: bias
        sample_idx: 0
        parameters: ${diagnostics.plot.parameters}
        plot_fn:
          _target_: my_project.plots.bias_map_plot_fn
          _partial_: true
          cmap: seismic
          vmax: 5.0

Example: pressure–latitude (or longitude) cross-section
-------------------------------------------------------

``SpatialMapPlot`` does not assume the figure is a map — it just delivers
``x``, ``y_true``, ``y_pred``, ``latlons`` and ``parameters`` and stores
whatever ``Figure`` the ``plot_fn`` returns. That makes it a good fit for
vertical cross-sections too.

For a pressure–latitude (or pressure–longitude) plot you need:

- multiple pressure levels of the same variable in
  ``diagnostics.plot.parameters`` (e.g. ``t_50, t_100, ..., t_1000``);
- a ``plot_fn`` that groups those parameters by variable prefix, reads the
  pressure level from the suffix, and bins the values along
  ``latlons[:, 0]`` (lat) or ``latlons[:, 1]`` (lon).

.. code:: python

   # my_project/plots.py
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.figure import Figure


   def zonal_cross_section_plot_fn(
       parameters: dict[int, tuple[str, bool]],
       *,
       x,
       y_true,
       y_pred,
       latlons,
       auxiliary=None,
       settings=None,
       variable: str = "t",     # prefix, e.g. "t" for t_500, t_700, ...
       axis: str = "lat",       # "lat" or "lon"
       n_bins: int = 90,
       **_kwargs,
   ) -> Figure:
       """Pressure vs latitude/longitude cross-section for a single variable."""
       # 1. pick output indices belonging to `variable` and extract pressure levels
       items = [
           (idx, int(name.split("_")[-1]))
           for idx, (name, _) in parameters.items()
           if name.startswith(f"{variable}_")
       ]
       if not items:
           msg = f"No parameters matching prefix '{variable}_' were requested."
           raise ValueError(msg)
       items.sort(key=lambda kv: kv[1])
       idxs, levels = zip(*items)
       levels = np.asarray(levels)

       # 2. average onto lat (or lon) bins
       coord = latlons[:, 0] if axis == "lat" else latlons[:, 1]
       bin_edges = np.linspace(coord.min(), coord.max(), n_bins + 1)
       which = np.clip(np.digitize(coord, bin_edges) - 1, 0, n_bins - 1)

       def zonal(field: np.ndarray) -> np.ndarray:
           field = np.asarray(field)
           out = np.full((len(idxs), n_bins), np.nan)
           for b in range(n_bins):
               mask = which == b
               if mask.any():
                   out[:, b] = field[mask][:, list(idxs)].mean(axis=0)
           return out

       truth = zonal(y_true)
       pred = zonal(y_pred)

       fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
       centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
       for ax, field, title in zip(
           axes, [truth, pred, pred - truth], ["truth", "pred", "pred - truth"],
       ):
           im = ax.pcolormesh(centres, levels, field, shading="auto")
           ax.invert_yaxis()  # pressure decreasing upward
           ax.set_xlabel(axis)
           ax.set_title(title)
           fig.colorbar(im, ax=ax)
       axes[0].set_ylabel("pressure [hPa]")
       return fig

Wire it in as:

.. code:: yaml

   - _target_: anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot
     tag_infix: zonal_t
     sample_idx: 0
     parameters: [t_50, t_100, t_250, t_500, t_700, t_850, t_1000]
     plot_fn:
       _target_: my_project.plots.zonal_cross_section_plot_fn
       _partial_: true
       variable: t
       axis: lat        # or "lon" for meridional cross-section
       n_bins: 90

Notes:

- If the variable/level naming in your dataset does not encode the
  pressure level in the suffix, replace the ``split("_")`` step with a
  small explicit mapping.
- ``focus_area`` still applies: pass a ``latlon_bbox`` and the
  cross-section is restricted to that region for free.

The ``PlotLoss`` and ``GraphTrainableFeaturesPlot`` callbacks follow the
same pattern (see ``loss_plot_fn`` and ``graph_plot_fn`` in
``anemoi.training.diagnostics.evaluation.plotting``), so custom loss-bar
or graph-feature renderings can be swapped in the same way.

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

.. automodule:: anemoi.training.diagnostics.callbacks.plot_adapter
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.diagnostics.callbacks.provenance
   :members:
   :no-undoc-members:
   :show-inheritance:
