############################
 Time Interpolator training
############################

This guide explains how to train the Time Interpolator model, including
the configuration changes for mass-conserving time interpolation using
accumulated inputs.

*************************
 Changes in model config
*************************

Use the specialized interpolator architecture:

.. code:: yaml

   model:
     _target_: anemoi.models.models.interpolator.AnemoiModelEncProcDecInterpolator
     # typical args for encoder/processor/decoder omitted for brevity

Set the requested output cadence:

.. code:: yaml

   data:
     frequency: 1h
     timestep: 1h

****************************
 Changes in training config
****************************

Use the deterministic forecaster task (no ensemble specialization
required):

.. code:: yaml

   training:
     task:
       _target_: anemoi.training.train.tasks.GraphForecaster
     # other training params as per your deterministic setup

Control interpolation windows with ``training.explicit_times``. Indices
in ``explicit_times`` are expressed in units of ``data.timestep``: -
**input**: the temporal boundary indices you interpolate between. -
**target**: the sequence of intermediate temporal indices to predict.

.. code:: yaml

   training:
     explicit_times:
       input: [0, 6]
       target: [1, 2, 3, 4, 5]

Examples:

-  6h → 1h with 1h data

      -  ``data.frequency: 1h``, ``data.timestep: 1h``
      -  ``training.explicit_times: input: [0, 6], target: [1, 2, 3, 4,
         5]``

-  24h → 6h with 6h data

   -  ``data.frequency: 6h``, ``data.timestep: 6h``
   -  ``training.explicit_times: input: [0, 4], target: [1, 2, 3]``

-  24h → 6h with 1h data

   -  ``data.frequency: 1h``, ``data.timestep: 6h``
   -  ``training.explicit_times: input: [0, 4], target: [1, 2, 3]``

******************************************
 Mass-conserving time interpolation setup
******************************************

To enforce mass conservation while interpolating in time (e.g., from
6-hour to 1-hour precipitation):

-  **Required datasets**:

   -  Base dataset at the target resolution (e.g., 1h fields).
   -  Aggregated dataset that provides the same variables over a longer
      window (e.g., 6h or 24h accumulations).

-  **Dataloader rename**:

   -  Import accumulated variables (e.g., cp, tp) from the aggregated
      dataset. - Rename them to non-conflicting inputs (e.g., cp_accum,
      tp_accum).

-  **Data config (model side)**:

   -  Put accumulated inputs (tp_accum, cp_accum) under ``forcing``.

   -  Keep their high-frequency counterparts (tp, cp) under
      ``diagnostic`` so they are predicted, not used as inputs.

   -  Normalize accumulated inputs with ``std`` only to preserve zeros.

   -  Remap statistics for diagnostic variables to their accumulated
      counterparts.

   -  Zero the first timestep of each input window for accumulated
      inputs using ``ZeroOverwriter``. This provides compatibility with
      anemoi-inference whereby the 0'th timestep index of an accumulated
      field of forecast state is always zero. Therefore this
      ZeroOverwriter allows training and inference to have aligned zero
      values at the left temporal boundary (index 0) of each input
      window.

Example (data config entries):

.. code:: yaml

   data:
     frequency: 1h
     timestep: 1h

     forcing:
       - "tp_accum"
       - "cp_accum"
       # other forcing fields (topography, solar, etc.)

     diagnostic:
       - "tp"
       - "cp"

     normalizer:
       default: "mean-std"
       remap:
         tp: "tp_accum"
         cp: "cp_accum"
       std:
         - "tp_accum"
         - "cp_accum"
         - "tp"
         - "cp"

     processors:
       normalizer:
         _target_: anemoi.models.preprocessing.normalizer.InputNormalizer
         config: ${data.normalizer}

       zero_overwriter:
         _target_: anemoi.models.preprocessing.overwriter.ZeroOverwriter
         config:
           groups:
             - vars:
                 - "tp_accum"
                 - "cp_accum"
               time_index: [0]

******************************
 Changes in dataloader config
******************************

Provide the base dataset at the target frequency and join an aggregated
dataset that contains the same variables accumulated over a coarser
window (e.g., 6h → 1h or 24h → 6h). Rename the accumulated variables to
distinct names (e.g., tp_accum, cp_accum) so they can be referenced in
the model configuration.

This example details a 24hr to 6hr. The accumulated variables dataset
must have accumulations over 24 hours but defined at 6hr intervals.
Example (adapt to your paths):

interpolation:

.. code:: yaml

   dataloader:
     dataset: ${hardware.paths.data}${hardware.files.dataset}            # base dataset at target frequency
     dataset_24_accums: ${hardware.paths.data}${hardware.files.dataset_24_accums}  # aggregated dataset

     training:
       dataset:
         join:
           - dataset: ${dataloader.dataset}
             start: ${dataloader.train_start}
             end: ${dataloader.train_end}
             frequency: ${data.frequency}
           - dataset: ${dataloader.dataset_24_accums}
             start: ${dataloader.train_start}
             end: ${dataloader.train_end}
             select: ${dataloader.select_24_accums}
             rename: ${dataloader.rename_24_accums}

     # variables to pull from the aggregated dataset
     select_24_accums:
       - cp
       - tp
       # add others as needed (e.g., sf, ssrd, strd, ttr)

     # rename to dedicated accumulated names to avoid clashes
     rename_24_accums:
       cp: cp_accum
       tp: tp_accum
       # add others similarly

****************
 Example recipe
****************

Putting it together:

#. Dataloader joins the aggregated dataset and renames its variables: -
   ``cp -> cp_accum``, ``tp -> tp_accum``.

#. Model config uses the interpolator architecture and sets target
   cadence: - ``data.frequency: 1h``, ``data.timestep: 1h``.

#. Data config wires mass conservation: - Accumulated inputs as forcing,
   diagnostic predictions for instantaneous outputs, stats remap,
   ``std`` normalization for accumulations, and ``ZeroOverwriter`` with
   ``time_index: [0]``.

#. Dataloader joins the aggregated dataset and renames its variables:

   -  cp -> cp_accum
   -  tp -> tp_accum

#. Model config uses the interpolator architecture and sets target
   cadence.

   -  ``data.frequency: 1h``, ``data.timestep: 1h``

#. Data config wires mass conservation:

   -  Accumulated inputs as forcing, Corresponding dissagregated outputs
      as diagnostic
   -  Stats remap for tp/cp -> tp_accum/cp_accum
   -  Std normalization for accumulations
   -  ZeroOverwriter with time_index: [0]

.. note::

   See training/src/anemoi/training/config/data/zarr_interpolator.yaml
   for a compact reference of the mass-conserving normalization and
   preprocessing setup
