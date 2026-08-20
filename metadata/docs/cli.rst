.. (C) Copyright 2026- Anemoi contributors.
..
.. This software is licensed under the terms of the Apache Licence Version 2.0
.. which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
..
.. In applying this licence, ECMWF does not waive the privileges and immunities
.. granted to it by virtue of its status as an intergovernmental organisation
.. nor does it submit to any jurisdiction.

Command Line Interface
======================

The ``anemoi-metadata`` command provides tools for inspecting, navigating,
and managing checkpoint metadata from the command line.

Installation
------------

The CLI is installed automatically with the package:

.. code-block:: bash

   pip install anemoi-metadata


Commands
--------

info
^^^^

Show a human-readable summary of checkpoint metadata.

**Usage**:

.. code-block:: bash

   anemoi-metadata info <checkpoint> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--no-migrate``
   Do not migrate metadata to the latest schema version before displaying.

**Example**:

.. code-block:: bash

   anemoi-metadata info model.ckpt

**Example output**:

.. code-block:: text

   Schema version: 1.0.0
   Created at:     2026-01-15 10:30:00+00:00

   Datasets:           ['data']
   Task:               None
   Timestep:           6h
   Multi-step input:   3
   Multi-step output:  2

   Variables (101 total, first 10):
     - 10u
     - 10v
     - 2d
     - 2t
     - lsm
     - msl
     - q
     - sp
     - t
     - tp
     ... and 91 more

   Full metadata:
   {
     "schema_version": "1.0.0",
     ...
   }


dump
^^^^

Extract raw metadata from a checkpoint and print it as JSON or YAML,
without pydantic validation or migration.

**Usage**:

.. code-block:: bash

   anemoi-metadata dump <checkpoint> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--output FILE``
   Write output to *FILE* instead of stdout.

``--yaml``
   Output in YAML format (requires PyYAML).

``--json``
   Output in JSON format (default).

**Examples**:

Print raw JSON to stdout:

.. code-block:: bash

   anemoi-metadata dump model.ckpt

Save raw YAML to a file:

.. code-block:: bash

   anemoi-metadata dump model.ckpt --yaml --output metadata.yaml


load
^^^^

Load metadata from a JSON or YAML file into a checkpoint, replacing any
existing metadata.

**Usage**:

.. code-block:: bash

   anemoi-metadata load <checkpoint> --input <file> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--input FILE``
   *(Required.)* JSON or YAML file containing the new metadata.

``--yaml``
   Treat the input file as YAML (requires PyYAML).

``--json``
   Treat the input file as JSON (default; also inferred from ``.json`` extension).

**Example**:

.. code-block:: bash

   anemoi-metadata load model.ckpt --input updated_metadata.json


edit
^^^^

Open checkpoint metadata in an external editor and write any changes back
to the checkpoint on exit.  The checkpoint is left unmodified if no changes
are made.

**Usage**:

.. code-block:: bash

   anemoi-metadata edit <checkpoint> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--editor EDITOR``
   Editor command to use.  Defaults to the ``$EDITOR`` environment variable,
   falling back to ``vi``.

``--yaml``
   Edit in YAML format (requires PyYAML).

``--json``
   Edit in JSON format (default).

**Example**:

.. code-block:: bash

   anemoi-metadata edit model.ckpt --editor nano


view
^^^^

Open checkpoint metadata in a pager for read-only browsing.  The checkpoint
is never modified.

**Usage**:

.. code-block:: bash

   anemoi-metadata view <checkpoint> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--pager PAGER``
   Pager command to use.  Defaults to the ``$PAGER`` environment variable,
   falling back to ``less``.

``--yaml``
   View in YAML format (requires PyYAML).

``--json``
   View in JSON format (default).

**Example**:

.. code-block:: bash

   anemoi-metadata view model.ckpt


remove
^^^^^^

Remove metadata (and supporting arrays) from a checkpoint.  Either modifies
the checkpoint in-place or writes a cleaned copy to a new file.

**Usage**:

.. code-block:: bash

   anemoi-metadata remove <checkpoint> (--inplace | --output <file>)

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Options**:

``--inplace``
   Remove metadata from the source checkpoint in-place.

``--output FILE``
   Write the cleaned checkpoint to *FILE* instead of modifying in-place.

One of ``--inplace`` or ``--output`` is required.

**Examples**:

Remove in-place:

.. code-block:: bash

   anemoi-metadata remove model.ckpt --inplace

Write a cleaned copy:

.. code-block:: bash

   anemoi-metadata remove model.ckpt --output model_no_metadata.ckpt


get
^^^

Navigate metadata via a dot-separated key path and print the value.
Use ``'.'`` as the key to list all top-level keys.  Append a trailing dot
to any path segment to list keys at that level.

**Usage**:

.. code-block:: bash

   anemoi-metadata get <checkpoint> <path> [options]

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

``path``
   Dot-separated key path into the metadata dict
   (e.g. ``config.model.type``).  Use ``'.'`` to list top-level keys.

**Options**:

``--yaml``
   Print dict/list values in YAML format (requires PyYAML).

``--json``
   Print dict/list values in JSON format (default).

**Examples**:

List top-level keys:

.. code-block:: bash

   anemoi-metadata get model.ckpt .

Navigate to a nested value:

.. code-block:: bash

   anemoi-metadata get model.ckpt config.training

List keys at a given level (trailing dot):

.. code-block:: bash

   anemoi-metadata get model.ckpt config.


arrays
^^^^^^

Print the name, shape, and dtype of every supporting array stored in the
checkpoint.  Supporting arrays are numpy arrays stored alongside the metadata
(e.g. latitudes, longitudes, land-sea mask).

**Usage**:

.. code-block:: bash

   anemoi-metadata arrays <checkpoint>

**Arguments**:

``checkpoint``
   Path to the checkpoint file.

**Example**:

.. code-block:: bash

   anemoi-metadata arrays model.ckpt

**Example output**:

.. code-block:: text

   latitudes: shape=(1038240,) dtype=float32
   longitudes: shape=(1038240,) dtype=float32
   lsm: shape=(1038240,) dtype=float32
