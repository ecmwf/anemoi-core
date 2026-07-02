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

This page provides the complete API reference for anemoi-metadata.

High-Level Interface
--------------------

The recommended entry point for most users.

Metadata
^^^^^^^^

.. autoclass:: anemoi.metadata.interface.Metadata
   :members:
   :undoc-members:
   :show-inheritance:


Checkpoint I/O
--------------

Functions for reading and writing metadata from/to checkpoint files.

.. automodule:: anemoi.metadata.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:


Registry
--------

Version registry for metadata schemas.

.. autoclass:: anemoi.metadata.registry.MetadataRegistry
   :members:
   :undoc-members:
   :show-inheritance:


Base Classes
------------

Base classes and decorators for metadata schemas.

.. automodule:: anemoi.metadata.base
   :members:
   :undoc-members:
   :show-inheritance:


Migration
---------

Sequential migration system for metadata versions.

.. autoclass:: anemoi.metadata.migration.MetadataMigrator
   :members:
   :undoc-members:
   :show-inheritance:


Mixins
------

Shared functionality for the Metadata interface.

VariablesMixin
^^^^^^^^^^^^^^

.. autoclass:: anemoi.metadata.mixins.variables.VariablesMixin
   :members:
   :undoc-members:
   :show-inheritance:


ValidationMixin
^^^^^^^^^^^^^^^

.. autoclass:: anemoi.metadata.mixins.validation.ValidationMixin
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
