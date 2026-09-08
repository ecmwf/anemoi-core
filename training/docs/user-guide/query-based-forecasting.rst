#########################
 Query-based forecasting
#########################

This experimental path samples a requested physical field first, then selects a
supported target and the sources available at its forecast origin. It uses a
shared value/metadata adapter and scalar decoder instead of archive-specific
channel heads.

Configuration
=============

Use ``training/src/anemoi/training/config/multidomain.yaml``. In
``dataloader/query_based_forecasting.yaml``, set ``dataset_config.dataset`` for
each archive to its Anemoi dataset path and set ``enabled: true`` for the
archives to use. Every supplied name (ARRA, CARA2, ERA5, IFS, ICON-FORCE and
MEPS) deliberately starts with an empty path. An enabled empty path is an
error. Training and validation must enable the same geometries; use their
``start`` and ``end`` values for a reproducible validation period.

ERA5 is the default global context and stretched-mesh reference. To use IFS,
enable IFS and set ``model.query.stretched_grid.context_source=IFS`` and
``task.reference_provenance=IFS``. The
processor mesh remains global but is refined over
``model.query.stretched_grid.area``. Query targets are decoded only on their
sampled AOI. Set ``task.target_regions_by_provenance`` when products have
different native domains; regions outside the refined AOI use the coarser part
of the processor mesh. The graph cache records source bounds and all stretched
geometry settings and refuses an incompatible existing graph.
Source coverage is represented by a longitude/latitude bounding rectangle;
irregular domains and dateline-crossing domains need a more precise mask.

The catalogue discovers fields, physical levels, units, native frequency,
coordinates, statistics, support metadata and configured product provenance
from enabled datasets. It persists the native resolution identifier and uses a
physical spacing only when the dataset exposes metres or kilometres; identifiers
such as ``o96`` remain an explicitly unknown spacing. ``target_variables`` and
``input_variables`` are optional intersections with that catalogue. The
configured archive key is the explicit provenance because the generic dataset
API has no target-product provenance field.

Run
===

From the repository root, after activating the environment containing the
editable Anemoi packages:

.. code-block:: bash

   anemoi-training train --config-name=multidomain

Training diagnostics
====================

The query configuration contains an opt-in ``QueryDiagnosticsPlot`` block at
``diagnostics.plot.callbacks[0]``. Enable it for a short run with:

.. code-block:: bash

   anemoi-training train --config-name=multidomain \
       system.output.root=/path/to/multidomain/diagnostics \
       system.output.plots=. \
       diagnostics.plot.callbacks.0.enabled=true \
       training.max_steps=4 task.samples_per_epoch=4 task.validation_samples=2 \
       diagnostics.enable_checkpointing=false

Set ``fixed_validation_cases`` to deterministic ``QueryDataset`` sample
indices. ``max_cases`` bounds the number rendered, while ``max_points``,
``max_edges`` and ``max_embedding_items`` cap plotting work. Expensive query
interventions occur only during scheduled validation. ``lead_time_hours``,
``pressure_levels_hpa``, ``target_provenances`` and ``omit_metadata`` control
one-at-a-time interventions. Use ``null`` for automatic compatible
level/provenance discovery and ``[]`` to disable one of those sweeps. An
unsupported request is logged and shown as skipped; no nearest level, product
or grid is substituted.

Figures are written below ``${system.output.plots}/plots`` (therefore below the
chosen ``system.output.root``) and logged as MLflow artifacts under
``query_diagnostics`` when MLflow is enabled. Expected names
include ``query_example_val_case0000_epoch000.jpg``,
``query_stretched_2m_temperature_case0000_epoch000.jpg``,
``query_ifs_input_case0000_epoch000.jpg``,
``query_domain_case0000_epoch000.jpg``,
``query_connectivity_case0000_epoch000.jpg``,
``query_embeddings_case0000_epoch000.jpg``,
``query_metadata_sweeps_case0000_epoch000.jpg``,
``query_sensitivity_lead_case0000_epoch000.jpg`` and
``query_sampler_epoch000.jpg``.

The example plot checks physical inverse normalization, exact valid times,
native-node alignment, reference masks and missing-versus-zero semantics. The
input plots show native 2 m-temperature inputs mapped to a bounded set of
hidden nodes through their nearest actual encoder edge, plus the exact
normalized, zero-filled IFS value channel and separate Boolean validity mask
immediately before the value adapter. The hidden-node temperature view is
labelled as diagnostic mapping: the network itself first pools all input
fields at each source node. It is never used for scoring or fed back to the
model. The selected IFS channel maximizes missing locations inside the selected
context for that example; excluded context and missing values are shown as
different states. The domain/connectivity plots expose the full graph
coordinates, requested bbox,
actual input coverage, hidden masks, a nearest usable input distance and only a
bounded sample of the encoder/dynamic-decoder edges. Embedding plots keep
categorical, query and pooled-input spaces distinct and reuse the first PCA
basis across epochs. Sensitivity maps hold the input state and output nodes
fixed and annotate whether each change has direct training supervision. The
sampler plot uses examples that reached the loss and separates sample counts,
valid-target counts, loss weights and unit-specific physical errors.

Diagnostics do not interpolate fields for display or scoring. Map panels use
native unstructured coordinates and the existing coastline/border and
projection utilities. With distributed data parallelism sampler counters are
gathered by every rank before rank-zero rendering; the graph shown is the full
pre-sharding graph and is labelled as such.

What is actually conditioned
----------------------------

The current query adapter consumes categorical variable, configured archive
provenance and physical-unit IDs plus the continuous vector listed in
``anemoi.models.layers.query_adapter.CONTINUOUS_METADATA``: physical pressure
(log-scaled) and its applicability/known flags, provenance-relative model-level
index and flags, height and flags, lead/time offset, input frequency, output
frequency, temporal aggregation window, grid spacing, spatial support,
vertical-coordinate one-hot flags and aggregation-type flags. Model-level
indices are supported as exact identities within a provenance; no cross-model
equivalence or pressure interpolation is inferred from the integer index.
Pressure labels in diagnostics are Pa converted to hPa for display, never
model-level numbers. Archives without physical unit metadata use the explicit
``unknown`` category rather than an invented unit.

The training sampler chooses a supported variable/provenance/lead/region query
before it chooses a target time and before it loads or drops input fields.
There is no nearest-target fallback. Bbox and explicit output coordinates
select geometry; the bbox itself and the grid name are not embedded. During
training, ``output_coordinates`` is always supplied, so the dynamic spherical
KNN decoder is used and the registered decoder graph selected by ``grid`` is
not. Provenance changes the query embedding and target catalogue selection but
does not select graph connectivity. Source masks control value pooling and
encoder coverage; target masks and optional cosine latitude weights control the
spatial loss. The domain plots intentionally show these effective tensors and
edges rather than planned query fields that the forward method ignores.

The public query spelling separates value units from the vertical coordinate.
Two-metre temperature is therefore expressed as ``variable: t``,
``model_type: height``, ``level: 2``, ``level_unit: m`` and ``unit: K``.
Likewise, ``provenance: ICON``, ``model_type: ml`` and ``level: 1`` identifies
ICON model level 1; ``level_unit`` is omitted because the index is
dimensionless. ``aggregation_type`` describes instantaneous, accumulated or
statistically aggregated values, while ``temporal_aggregation_window`` gives
the aggregation window relative to valid time. ``output_frequency`` is the
spacing between requested output valid times. Older names remain accepted when
reading queries, but sampled training queries and diagnostics use these names.

The default run is 32 optimizer steps. Useful overrides include
``task.lead_times``, ``task.input_history``, ``task.source_dropout``,
``task.field_dropout``, ``task.history_dropout``, ``task.variable_weights``,
``task.provenance_weights``, ``task.target_regions`` and
``task.spatial_weighting`` (``uniform`` or ``cosine_latitude``), and
``training.max_steps``. Sampling is variable-first and provenance-balanced;
archive length does not set the request distribution. Each query loss is a
masked spatial mean, so denser native grids do not receive a larger weight just
because they contain more points.
Training targets are selected only from existing native nodes. Requested finer
coordinates are an inference geometry and never synthesize finer supervision.

Inference input
===============

The inference checkpoint accepts ``model(inputs, query)``. ``inputs`` is a
dictionary keyed by a source registered in the checkpoint. Each value has a
``values`` tensor of shape ``(native_nodes, fields)`` in physical units and a
same-length ``fields`` list. Node order must match that source's checkpoint
geometry. Put each source's array in ``VALUES_NPY`` and its field dictionaries
in ``FIELDS_JSON``. For example, ``era5_fields.json`` may contain:

.. code-block:: json

   [
     {
       "variable": "t",
       "model_type": "pl",
       "level": 500,
       "level_unit": "hPa",
       "unit": "K",
       "time_offset": "-6h"
     },
     {"variable": "lsm", "model_type": "sfc", "unit": "1", "time_offset": "0h"}
   ]

Create the input file with the runnable converter (repeat ``--source`` for
MEPS or other registered sources):

.. code-block:: bash

   python training/examples/prepare_query_inputs.py \
       --source ERA5 era5_values.npy era5_fields.json \
       --output forecast_inputs.pt

These tensors must come from real fields at or before the forecast origin under
the configured retrospective availability lag. Include the weather and static
fields needed by the experiment; the model does not reopen training archives
or create missing static data at inference.

Save a query as JSON, then run the supplied entry point:

.. code-block:: bash

   python training/examples/query_based_inference.py \
       --checkpoint query_based_forecasting_run/checkpoint/<run-id>/inference-last.ckpt \
       --inputs forecast_inputs.pt --query query.json --output prediction.npy

An unseen pressure level is permitted only between two trained physical levels
for the same variable and requested provenance. For example:

.. code-block:: json

   {
     "variable": "u",
     "lead_time": "3h",
     "output_frequency": "1h",
     "model_type": "pl",
     "level": 775,
     "level_unit": "hPa",
     "unit": "m s-1",
     "aggregation_type": "instantaneous",
     "grid": "ERA5",
     "area": [-15.0, 55.0, 35.0, 72.0],
     "provenance": "ERA5",
     "below_ground_policy": "unmasked"
   }

For new coordinates, save an ``(nodes, 2)`` NumPy array containing latitude
and longitude in degrees and add ``--output-coordinates norway_1km.npy``. The
query may then declare ``"resolution": "1km"``; the coordinates, not the bbox
and spacing alone, define the projection and grid. This changes output geometry
but does not create finer supervision.

An ICON-FORCE-conditioned Norway request can use the same Norwegian coordinates
or a registered MEPS grid:

.. code-block:: json

   {
     "variable": "u",
     "lead_time": "3h",
     "output_frequency": "1h",
     "model_type": "pl",
     "level": 775,
     "level_unit": "hPa",
     "unit": "m s-1",
     "aggregation_type": "instantaneous",
     "grid": "MEPS",
     "area": [-15.0, 55.0, 35.0, 72.0],
     "provenance": "ICON-FORCE",
     "below_ground_policy": "unmasked"
   }

This requires ICON-FORCE targets for ``u`` with pressure levels bracketing 775
hPa (for example 700 and 850 hPa) during training, plus a registered MEPS
geometry or explicit output coordinates. Supply full native ERA5/IFS global
context and any available local
Norwegian weather/static fields in ``inputs``. ICON-FORCE conditioning remains
independent of the decoder geometry, but product effects and regional climate
can remain confounded when products have no overlapping supervision.
Pressure queries must explicitly choose ``below_ground_policy: unmasked`` or
provide an ``output_validity_mask`` aligned with the requested coordinates.
The latter must not be derived from unavailable future reference pressure.

Current limits
==============

The first implementation handles one requested field and lead per example and
one GPU per model group. Horizon queries evaluate their leads independently.
Only native time-series readers and an explicit retrospective availability
policy are supported. Forecast-trajectory initialization/availability metadata
is not inferred yet. Model-level fields are excluded unless a future dataset
API supplies a dataset-specific hybrid-coordinate transform, coefficients and
matching surface pressure; no pressure is invented from a model-level index.
Input values must use the exact units recorded in the catalogue. Automatic unit
conversion, pressure extrapolation, automatic below-ground masking and native
future model-level output geometry are unsupported. Dynamic decoder edges are
built in bounded chunks but very large output grids still require memory
proportional to their node and edge counts.
