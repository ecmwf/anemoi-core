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

Plotting callbacks (such as ``BatchOutputPlot`` and ``LossCurvePlot``) support a ``focus_area`` parameter. This allows you to restrict the geographic scope of plots to specific regions or masks. A focus area can be defined in two ways:

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

**Rendering settings**

Shared rendering options are grouped under ``diagnostics.plot.settings``
and map 1:1 to :class:`~anemoi.training.diagnostics.callbacks.plot.PlottingSettings`.
All fields are **optional** — defaults are defined in code so you only need
to specify values that differ from them.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Default
     - Description
   * - ``datashader``
     - ``true``
     - Use Datashader for rendering (fast hexbinning, recommended for large grids).
       When ``false``, ``matplotlib.scatter`` is used — slower but sharper.
   * - ``projection_kind``
     - ``equirectangular``
     - Map projection. ``lambert_conformal`` fits the domain automatically
       (requires Cartopy). Any ``cartopy.crs`` class in snake_case is also accepted.
       Forced to ``equirectangular`` when ``datashader: true``.
   * - ``asynchronous``
     - ``true``
     - Run plotting in a background thread so it does not block training.
   * - ``precip_and_related_fields``
     - ``null``
     - Variable names that use precipitation-specific colour scaling.
       Shared across all callbacks so you don't need to repeat the list per callback.
   * - ``colormaps``
     - ``null``
     - Variable-specific colormaps keyed by ``default``, ``error``, or a variable
       group name. Shared across all callbacks.

Example — override only what differs from the defaults:

.. code:: yaml

   plot:
     settings:
       datashader: false               # switch to matplotlib scatter
       projection_kind: lambert_conformal
       precip_and_related_fields: [tp, cp]
       colormaps:
         precip:
           _target_: anemoi.training.utils.custom_colormaps.MatplotlibColormapClevels
           clevels: [0, 0.5, 1, 2, 5, 10, 25, 50, 100]
           variables: ${diagnostics.plot.settings.precip_and_related_fields}

**Note** - asynchronous plotting is only available for the plotting callbacks.

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
      # Rendering settings — all optional, code-defined defaults apply.
      settings:
         datashader: true
         projection_kind: equirectangular
         precip_and_related_fields: [tp, cp]

      frequency:
         batch: 750
         epoch: 5

      parameters:
         - z_500
         - t_850
         - u_850

      sample_idx: 0
      datasets_to_plot: ["data"]

      focus_areas:
         europe:
            latlon_bbox: [30.0, -20.0, 60.0, 40.0]

      callbacks:
         - _target_: anemoi.training.diagnostics.callbacks.plot.LossCurvePlot
            dataset_names: ${diagnostics.plot.datasets_to_plot}
            parameter_groups:
               moisture: [tp, cp, tcw]
               sfc_wind: [10u, 10v]
            every_n_batches: ${diagnostics.plot.frequency.batch}

         - _target_: anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot
            tag_infix: sample
            dataset_names: ${diagnostics.plot.datasets_to_plot}
            sample_idx: ${diagnostics.plot.sample_idx}
            parameters: ${diagnostics.plot.parameters}
            every_n_batches: ${diagnostics.plot.frequency.batch}
            plot_fn:
               _target_: anemoi.training.diagnostics.evaluation.plotting.batch_output.sample_plot_fn
               _partial_: true
               # per_sample, accumulation_levels_plot etc. are optional — omit to use defaults

Pluggable batch-output plots
============================

Batch-output–style callbacks (samples, spectra, histograms, ensembles) are
all driven by a single :class:`~anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot`
callback that iterates over datasets, samples and focus areas, and delegates
the actual figure rendering to a pluggable ``plot_fn``.

The bundled plot functions live in
``anemoi.training.diagnostics.evaluation.plotting.batch_output``:

- ``sample_plot_fn`` — multi-level forecast sample plot (replaces ``PlotSample``).
- ``spectrum_plot_fn`` — power spectrum (replaces ``PlotSpectrum``).
- ``histogram_plot_fn`` — per-variable histograms (replaces ``PlotHistogram``).
- ``ensemble_plot_fn`` — ensemble member plot (replaces ``PlotEnsSample``).

Each ``plot_fn`` receives keyword-only arguments (``x``, ``y_true``,
``y_pred``, ``latlons``, ``auxiliary``, ``settings``, plus any plot-specific
kwargs bound in YAML via ``_partial_: true``). The full signature is
documented in `Plot function contracts`_ below and enforced at callback
initialisation time by :func:`~anemoi.training.diagnostics.evaluation.plotting.protocols.validate_plot_fn`.

.. code:: yaml

   # Same callback, different plot_fn → different figures
   - _target_: anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot
     tag_infix: spectrum
     sample_idx: 0
     parameters: ${diagnostics.plot.parameters}
     plot_fn:
       _target_: anemoi.training.diagnostics.evaluation.plotting.batch_output.spectrum_plot_fn
       _partial_: true
       min_delta: 0.01

Since a single run can register many instances of ``BatchOutputPlot``,
``LossCurvePlot`` or ``GraphFeaturePlot`` (one per ``plot_fn``), MLflow
artifacts are grouped by ``plot_fn`` name rather than callback class name —
e.g. figures from ``sample_plot_fn`` and ``histogram_plot_fn`` land under
separate folders instead of all being dumped into one ``BatchOutputPlot`` folder.

Plot function contracts
-----------------------

Three callbacks accept a pluggable ``plot_fn``. Each defines a fixed
keyword-only signature; anything else in the ``plot_fn:`` YAML block is
bound as a partial kwarg via ``_partial_: true`` and forwarded to the
function. All three contracts accept a ``settings`` argument (a
``PlottingSettings`` instance carrying ``datashader``, ``projection_kind``,
``colormaps``, ``precip_and_related_fields``, ``asynchronous``) and
``**kwargs`` so future additions do not break existing implementations.

The contracts are enforced at callback ``__init__`` time via
:func:`~anemoi.training.diagnostics.evaluation.plotting.protocols.validate_plot_fn`,
which checks that the resolved callable accepts the required parameters.
The full API reference for the three Protocols is at the bottom of this page.

Layers and data flow
....................

Each of the three pluggable callbacks (``LossCurvePlot``, ``BatchOutputPlot``,
``GraphFeaturePlot``) has the same three-layer shape:

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

Concretely:

- ``Callback._plot`` is the Lightning glue. It reads the batch and calls
  the matching ``extract_*_inputs`` helper.
- ``model_introspection.extract_*_inputs`` is the **only** place that pokes
  at ``pl_module`` (data indices, metadata, graph). It returns a plain
  ``dict`` whose keys match the ``plot_fn`` signature below.
- The callback splats the dict into ``plot_fn`` alongside per-step extras
  (loss array, ``x``/``y_true``/``y_pred``, ``settings``, …).

Family-to-artifact mapping:

+----------------------+-----------------------------+
| Callback             | ``extract_*_inputs`` helper |
+======================+=============================+
| ``LossCurvePlot``    | ``extract_loss_inputs``     |
+----------------------+-----------------------------+
| ``BatchOutputPlot``  | ``extract_spatial_inputs``  |
+----------------------+-----------------------------+
| ``GraphFeaturePlot`` | ``extract_graph_inputs``    |
+----------------------+-----------------------------+

``BatchOutputPlot`` — per-sample, per-dataset figures
.....................................................

Callback: :class:`~anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot`.
Protocol: :class:`~anemoi.training.diagnostics.evaluation.plotting.protocols.BatchOutputPlotFn`.

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
  reduced by ``BatchOutputPlot``'s ``process`` step (using ``sample_idx``).
  ``y_true`` / ``y_pred`` are 2-D arrays over grid points × output variables.
- ``latlons`` is pre-masked to the active ``focus_area`` when one is set.
- Return a ``matplotlib.figure.Figure`` — returning ``None`` is not allowed.

``LossCurvePlot`` — grouped per-variable loss bar chart
.......................................................

Callback: :class:`~anemoi.training.diagnostics.callbacks.plot.LossCurvePlot`.
Protocol: :class:`~anemoi.training.diagnostics.evaluation.plotting.protocols.LossPlotFn`.
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

- ``LossCurvePlot`` supplies the **raw** per-variable loss in the model's output
  order together with the naming/grouping context, and does **not** apply any
  presentation-specific reordering, grouping or colouring itself.
- The default ``loss_plot_fn`` reproduces the historic behaviour by calling
  :func:`argsort_variablename_variablelevel` (sort by variable + level),
  then :func:`sort_and_color_by_parameter_group` (group + colour), then
  :func:`plot_loss` (bar-chart render). A custom ``plot_fn`` is free to
  replace or skip any of these steps.
- ``loss`` and ``parameter_names`` share the same length and ordering.

``GraphFeaturePlot`` — graph node/edge feature plots
....................................................

Callback: :class:`~anemoi.training.diagnostics.callbacks.plot.GraphFeaturePlot`.
Protocol: :class:`~anemoi.training.diagnostics.evaluation.plotting.protocols.GraphPlotFn`.
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
- ``GraphFeaturePlot`` extracts the graph artifacts once via
  ``extract_graph_inputs`` (unwrapping any DDP wrapper) and passes them
  as keyword arguments; the plot function never sees the raw model.
- ``edge_trainable_modules`` is empty for hierarchical models (they carry
  no trainable edge parameters); ``node_trainable_tensors`` is empty when
  no trainable node attributes are defined. The default ``graph_plot_fn``
  logs a warning and skips the corresponding figure in that case.

Adding a new batch-output plot
------------------------------

To add a new plot type, write a function matching the ``BatchOutputPlot``
``plot_fn`` signature (see `Plot function contracts`_) and reference it from
YAML — no new callback class or Pydantic schema is required. The callback
will validate the signature at initialisation time.

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
   partial kwargs). Because ``_target_`` is not one of the built-in options,
   set ``_partial_: true`` and pass any extra kwargs directly:

   .. code:: yaml

      - _target_: anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot
        tag_infix: bias
        sample_idx: 0
        parameters: ${diagnostics.plot.parameters}
        plot_fn:
          _target_: my_project.plots.bias_map_plot_fn
          _partial_: true
          cmap: seismic
          vmax: 5.0

   .. note::

      Custom ``_target_`` values are not in the Pydantic ``Literal`` for
      ``plot_fn._target_``, so schema validation will reject them. Override
      ``BatchOutputPlotFnSchema`` in your own schema, or set
      ``model_config = {"extra": "allow"}`` on a subclass, to permit them.

Example: pressure–latitude cross-section
-----------------------------------------

``BatchOutputPlot`` does not assume the figure is a map — it just delivers
``x``, ``y_true``, ``y_pred``, ``latlons`` and ``parameters`` and stores
whatever ``Figure`` the ``plot_fn`` returns. That makes it a good fit for
vertical cross-sections too.

.. code:: yaml

   - _target_: anemoi.training.diagnostics.callbacks.plot.BatchOutputPlot
     tag_infix: zonal_t
     sample_idx: 0
     parameters: [t_50, t_100, t_250, t_500, t_700, t_850, t_1000]
     plot_fn:
       _target_: my_project.plots.zonal_cross_section_plot_fn
       _partial_: true
       variable: t
       axis: lat

The ``LossCurvePlot`` and ``GraphFeaturePlot`` callbacks follow the
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

.. automodule:: anemoi.training.diagnostics.evaluation.plotting.protocols
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
