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


Global options
--------------

``--version``, ``-V``
   Show the version and exit.

``--debug``, ``-d``
   Debug mode.

``--rich``
   Use ``rich`` for logging.

These go before the command name, for example
``anemoi-metadata --debug info model.ckpt``.  Every command takes the path of
the checkpoint as its first argument.  Reading commands work on any metadata
format.  The commands that change a checkpoint (``load``, ``edit``, ``remove``)
do not validate the metadata: see :ref:`lenient-and-strict`.


Commands
--------

info
^^^^

Show a summary of checkpoint metadata: the format version, the version of each
component, the task and its timing, and for each dataset the data type, grid
size and variables.  It ends with the full metadata as JSON.  When the
datasets have different timings, the timing of each dataset is listed.

**Usage**:

.. code-block:: bash

   anemoi-metadata info <path>

**Arguments**:

``path``
   Path to the checkpoint file.

**Example**:

.. code-block:: bash

   anemoi-metadata info model.ckpt

**Example output** (a checkpoint with a gridded and a tabular dataset):

.. code-block:: text

   Version:            3.1
   Component versions: {'data': '1.0', 'task': '1.0'}
   Created at:         2026-11-23 00:00:00+00:00

   Task:               forecaster
   Timestep:           6h
   Steps in / out:     2 / 1

   Datasets:           ['data', 'obs']

   [data]
     Data type:        gridded
     Grid size:        40320
     Variables:        101 (99 input, 88 output)
       - 10u
       - 10v
       - 2d
       - 2t
       - cos_julian_day
       - cos_latitude
       - cos_local_time
       - cos_longitude
       - insolation
       - lsm
       ... and 91 more

   [obs]
     Data type:        tabular
     Grid size:        None
     Variables:        6 (5 input, 4 output)
       - 2t
       - 2d
       - msl
       - cos_julian_day
       - sin_julian_day
       - tp

   Full metadata:
   {
     "version": "3.1",
     ...
   }


dump
^^^^

Extract raw metadata from a checkpoint and print it as JSON or YAML,
exactly as stored.

**Usage**:

.. code-block:: bash

   anemoi-metadata dump <path> [options]

**Arguments**:

``path``
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
existing metadata.  The file is written as it is, without strict validation
and without changing its ``version``.  If the checkpoint has no metadata yet,
a warning is logged and the metadata is added.

**Usage**:

.. code-block:: bash

   anemoi-metadata load <path> --input <file> [options]

**Arguments**:

``path``
   Path to the checkpoint file.

**Options**:

``--input FILE``
   *(Required.)* JSON or YAML file containing the new metadata.

``--yaml``
   Treat the input file as YAML (requires PyYAML).

``--json``
   Treat the input file as JSON (default).  The format is also taken from the
   extension: ``.json``, ``.yaml`` or ``.yml``.  Any other extension needs
   ``--json`` or ``--yaml``.

**Example**:

.. code-block:: bash

   anemoi-metadata load model.ckpt --input updated_metadata.json


edit
^^^^

Open checkpoint metadata in an external editor and write any changes back
to the checkpoint on exit.  The checkpoint is left unmodified if no changes
are made.  The edited metadata is written as it is, without strict
validation.

**Usage**:

.. code-block:: bash

   anemoi-metadata edit <path> [options]

**Arguments**:

``path``
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

   anemoi-metadata view <path> [options]

**Arguments**:

``path``
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

   anemoi-metadata remove <path> (--inplace | --output FILE)

**Arguments**:

``path``
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

Navigate the raw metadata via a dot-separated key path and print the value.
Use ``'.'`` as the key to list all top-level keys.  Append a trailing dot
to any path segment to list keys at that level.  A key that is not found
raises a ``KeyError``.

**Usage**:

.. code-block:: bash

   anemoi-metadata get <path> <key> [options]

**Arguments**:

``path``
   Path to the checkpoint file.

``key``
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

Arrays grouped by dataset are listed as ``<dataset>/<name>``.

**Usage**:

.. code-block:: bash

   anemoi-metadata arrays <path>

**Arguments**:

``path``
   Path to the checkpoint file.

**Example**:

.. code-block:: bash

   anemoi-metadata arrays model.ckpt

**Example output**:

.. code-block:: text

   latitudes: shape=(1038240,) dtype=float32
   longitudes: shape=(1038240,) dtype=float32
   lsm: shape=(1038240,) dtype=float32
