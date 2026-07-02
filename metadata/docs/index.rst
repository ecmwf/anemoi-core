.. (C) Copyright 2026- Anemoi contributors.
..
.. This software is licensed under the terms of the Apache Licence Version 2.0
.. which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
..
.. In applying this licence, ECMWF does not waive the privileges and immunities
.. granted to it by virtue of its status as an intergovernmental organisation
.. nor does it submit to any jurisdiction.

anemoi-metadata
===============

**anemoi-metadata** provides versioned metadata schemas for Anemoi ML checkpoints.

It offers:

- **Versioned Schemas**: Pydantic-based metadata models with semantic versioning
- **Automatic Migration**: Seamless migration between schema versions
- **Checkpoint I/O**: Read and write metadata from/to checkpoint files
- **User-Friendly Interface**: High-level ``Metadata`` class with computed properties

Quick Start
-----------

Load metadata from a checkpoint:

.. code-block:: python

   from anemoi.metadata import Metadata

   # Load from checkpoint (handles legacy checkpoints without schema_version)
   metadata = Metadata.from_checkpoint("model.ckpt")

   # Typed inference properties
   print(metadata.dataset_names)     # ['data']
   print(metadata.timestep)          # '6h'
   print(metadata.multi_step_input)  # 3
   print(metadata.variables[:5])     # ['10u', '10v', '2d', '2t', ...]

   # Category-based variable selection
   prognostic = metadata.select_variables(include=["prognostic"])

   # Per-dataset access
   cfg = metadata.dataset_config("data")
   print(cfg.shapes.variables)     # 309
   print(cfg.timesteps.timestep)   # '6h'

   # Permissive section access
   lr = metadata.get("config", "training")

Create metadata programmatically:

.. code-block:: python

   from datetime import datetime, timezone
   from anemoi.metadata.versions.v1 import MetadataV1, InferenceMetadata

   raw = MetadataV1(
       schema_version="1.0.0",
       created_at=datetime.now(timezone.utc),
       metadata_inference={
           "seed": 42,
           "run_id": "train-abc",
           "dataset_names": ["data"],
           "data": {
               "data_indices": {"input": {"2t": 0, "msl": 1}, "output": {"2t": 0}},
               "variable_types": {"forcing": [], "target": [], "prognostic": ["2t", "msl"], "diagnostic": []},
               "timesteps": {"timestep": "6h", "input_relative_date_indices": [0, 1], "output_relative_date_indices": [2], "relative_date_indices_training": [0, 1, 2]},
               "shapes": {"variables": 2, "input_timesteps": 2},
           },
       },
       config={"lr": 0.001},
   )
   metadata = Metadata(raw)


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   versions
   api
   cli


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
