.. (C) Copyright 2026- Anemoi contributors.
..
.. This software is licensed under the terms of the Apache Licence Version 2.0
.. which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
..
.. In applying this licence, ECMWF does not waive the privileges and immunities
.. granted to it by virtue of its status as an intergovernmental organisation
.. nor does it submit to any jurisdiction.

Schema Versions
===============

This page documents the metadata schema versions. Each version is a Pydantic model
that defines the structure of checkpoint metadata.

Version Policy
--------------

- **Major.Minor only**: Schema versions use ``MAJOR.MINOR`` (no patch). Examples: ``"1.0"``, ``"2.0"``, ``"2.1"``.
- **Migrations**: Defined between adjacent registered versions.
- **Downgrades**: Not supported; use ``migrate=False`` to preserve original version.


Version 1.0.0
-------------

The initial metadata schema. Its design follows a two-section principle:

- **Strictly typed section** (``metadata_inference``): A fully validated
  :class:`~anemoi.metadata.versions.v1.InferenceMetadata` block that captures
  everything inference needs at runtime — dataset names, variable index mappings,
  variable categories, timestep configuration, and tensor shapes.  This section
  is frozen and validated by Pydantic. The nested per-dataset models preserve
  unknown fields for forward compatibility, allowing newer checkpoint writers to
  add fields without breaking older readers. Strict validation will be enforced
  at write time from V2 onwards.

- **Permissive dict sections** (``config``, ``training``, ``dataset``,
  ``environment``, ``provenance``): Plain ``dict`` fields that accept arbitrary
  content written by training.  They are stored as-is without structural
  validation, so training can evolve these sections freely without requiring a
  schema migration.

This split means inference code only depends on the typed contract, while the
full training configuration is preserved for reproducibility and debugging.

MetadataV1
^^^^^^^^^^

.. autopydantic_model:: anemoi.metadata.versions.v1.MetadataV1
   :members:
   :show-inheritance:
   :model-show-json:
   :model-show-config-summary:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


Supporting Models
^^^^^^^^^^^^^^^^^

The following Pydantic models are used as nested fields within the
``metadata_inference`` block of :class:`~anemoi.metadata.versions.v1.MetadataV1`.

InferenceMetadata
"""""""""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.InferenceMetadata
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


DatasetInferenceConfig
""""""""""""""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.DatasetInferenceConfig
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


DataIndices
"""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.DataIndices
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


VariableTypes
"""""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.VariableTypes
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


TimestepConfig
""""""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.TimestepConfig
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


TensorShapes
""""""""""""

.. autopydantic_model:: anemoi.metadata.versions.v1.TensorShapes
   :members:
   :model-show-json:
   :model-show-field-summary:
   :field-show-constraints:
   :field-show-default:


Future Versions
---------------

As the metadata schema evolves, new versions will be added here. Migrations
between versions are handled automatically by the :class:`~anemoi.metadata.migration.MetadataMigrator`.

To check if a migration path exists:

.. code-block:: python

   from anemoi.metadata.migration import MetadataMigrator

   # Check if direct migration exists
   if MetadataMigrator.has_migration("1.0.0", "2.0.0"):
       print("Migration available")

   # Migrate to latest
   from anemoi.metadata import Metadata

   metadata = Metadata.from_checkpoint("old_model.ckpt", migrate=True)
   print(f"Migrated to: {metadata.schema_version}")
