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
       "level_type": "pl",
       "level": 500,
       "level_units": "hPa",
       "time_offset": "-6h"
     },
     {"variable": "lsm", "level_type": "sfc", "time_offset": "0h"}
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
     "level_type": "pl",
     "level": 775,
     "level_units": "hPa",
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
     "level_type": "pl",
     "level": 775,
     "level_units": "hPa",
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
