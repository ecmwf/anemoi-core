.. (C) Copyright 2026- Anemoi contributors.
..
.. This software is licensed under the terms of the Apache Licence Version 2.0
.. which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
..
.. In applying this licence, ECMWF does not waive the privileges and immunities
.. granted to it by virtue of its status as an intergovernmental organisation
.. nor does it submit to any jurisdiction.

API Reference
=============

This page provides the API reference for anemoi-metadata.  The first sections
are the interface most users need.  The later ones are for code that reads
other formats or adds a component version (see :doc:`versions`).

High-level interface
--------------------

The recommended entry point.

Metadata
^^^^^^^^

.. autoclass:: anemoi.metadata.interface.Metadata
   :members:
   :undoc-members:
   :show-inheritance:


DatasetView
^^^^^^^^^^^

.. autoclass:: anemoi.metadata.interface.DatasetView
   :members:
   :undoc-members:
   :show-inheritance:


DatasetData
^^^^^^^^^^^

The ``data`` component of one dataset.

.. autoclass:: anemoi.metadata.interface.DatasetData
   :members:
   :undoc-members:
   :show-inheritance:


DatasetArrays
^^^^^^^^^^^^^

The supporting arrays of one dataset.

.. autoclass:: anemoi.metadata.arrays.DatasetArrays
   :members:
   :undoc-members:
   :show-inheritance:


TaskView
^^^^^^^^

The task of the checkpoint.

.. autoclass:: anemoi.metadata.interface.TaskView
   :members:
   :undoc-members:
   :show-inheritance:


TaskTiming
^^^^^^^^^^

The timing of the task for one dataset, returned by ``task.timing``,
``task.for_dataset(name)`` and ``task.timings()``.

.. autoclass:: anemoi.metadata.components.task.TaskTiming
   :members:
   :undoc-members:
   :show-inheritance:


LineageView
^^^^^^^^^^^

Where the checkpoint comes from.  A value the lineage version does not record
raises :class:`~anemoi.metadata.exceptions.VersionError`.

.. autoclass:: anemoi.metadata.interface.LineageView
   :members:
   :undoc-members:
   :show-inheritance:


SectionsView
^^^^^^^^^^^^

The raw, unversioned sections.

.. autoclass:: anemoi.metadata.interface.SectionsView
   :members:
   :undoc-members:
   :show-inheritance:


Checkpoint I/O
--------------

Functions for reading and writing metadata from and to checkpoint files.

.. automodule:: anemoi.metadata.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:


Strict write
------------

Validation of the metadata this package writes (format 3.1).  See
:ref:`lenient-and-strict`.

.. autodata:: anemoi.metadata.write_schema.CURRENT_VERSION

.. autodata:: anemoi.metadata.write_schema.COMPONENT_VERSIONS

.. autofunction:: anemoi.metadata.write_schema.validate_for_write

.. autoclass:: anemoi.metadata.write_schema.WriteValidationError
   :show-inheritance:

.. autofunction:: anemoi.metadata.write_schema.register_write_schema


Adapters
--------

The layered adapters that read every format.  The decorator
:func:`~anemoi.metadata.components.base.requires` gates an adapter method on
the version of its component (see :ref:`version-gate`).

.. autofunction:: anemoi.metadata.components.base.requires

CheckpointAdapter
^^^^^^^^^^^^^^^^^

.. automodule:: anemoi.metadata.base
   :members:
   :undoc-members:
   :show-inheritance:


Component registry
^^^^^^^^^^^^^^^^^^

.. autoclass:: anemoi.metadata.registry.ComponentRegistry
   :members:
   :undoc-members:
   :show-inheritance:


Format detection
^^^^^^^^^^^^^^^^

.. automodule:: anemoi.metadata.formats
   :members:
   :show-inheritance:


Component adapters
^^^^^^^^^^^^^^^^^^

.. autoclass:: anemoi.metadata.components.base.ComponentAdapter
   :members:
   :show-inheritance:

.. automodule:: anemoi.metadata.components.data
   :members:
   :show-inheritance:

.. automodule:: anemoi.metadata.components.task
   :members:
   :show-inheritance:

.. automodule:: anemoi.metadata.components.lineage
   :members:
   :show-inheritance:

.. automodule:: anemoi.metadata.components.legacy
   :members:
   :show-inheritance:


Sections reader
^^^^^^^^^^^^^^^

.. autoclass:: anemoi.metadata.sections.SectionsReader
   :members:
   :undoc-members:
   :show-inheritance:


Exceptions
----------

Custom exceptions for the metadata package.

.. automodule:: anemoi.metadata.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: anemoi.metadata.interface.DatasetSelectionError
   :show-inheritance:
