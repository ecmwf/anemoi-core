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

**anemoi-metadata** reads and writes the metadata stored inside Anemoi
checkpoints.

Checkpoints have been written in several metadata formats over time.  This
package opens all of them through one interface, so that code such as
inference does not need to know which format it was given.  It also defines
the format it writes itself, and checks it strictly before writing.

It offers:

- **One interface for every format**: the :class:`~anemoi.metadata.interface.Metadata`
  class gives the same answers for old and new checkpoints.
- **Layered adapters**: each structured part of the metadata (a *component*)
  has its own version and is read by a small adapter, built from a version
  reader and a behaviour (the task name, or the ``data_type`` of each dataset:
  gridded or tabular).  A new version only needs the code that differs.
- **Lenient read, strict write**: opening a checkpoint never fails on unknown
  or missing optional keys.  Writing checks the metadata against a schema.
- **Checkpoint I/O and a command line tool**: read, save, replace and remove
  the metadata of a checkpoint file.


Quick start
-----------

Open a checkpoint and read the task, the datasets and the supporting arrays:

.. code-block:: python

   from anemoi.metadata import Metadata

   metadata = Metadata.from_checkpoint("model.ckpt")

   # Checkpoint level
   print(metadata.version)             # '2.0'
   print(metadata.component_versions)  # {'data': '0.2', 'task': '0.2', 'lineage': '1.0'}
   print(metadata.dataset_names)       # ['data']
   print(metadata.task.task_name)      # 'forecaster'
   print(metadata.task.timestep())     # '6h'
   print(metadata.task.n_step_input)   # 2
   print(metadata.task.n_step_output)  # 1

   # Per dataset
   dataset = metadata.dataset("data")
   print(dataset.data.grid_size())                             # 40320
   print(dataset.data.variable_indices())                      # {'10u': 0, '10v': 1, ...}
   print(dataset.data.select_variables(include=["prognostic"]))
   print(dataset.arrays.latitudes)                             # numpy array, or None

   # Where the checkpoint comes from
   print(metadata.lineage.uuid)
   print(metadata.lineage.parent_uuid)                         # None for a fresh run

   # Raw sections, as stored
   print(metadata.sections.precision)                          # '16-mixed'
   print(metadata["config"]["data"]["frequency"])              # '6h'

The values above come from a checkpoint with one dataset.  The next page,
:doc:`versions`, explains where the metadata is stored and how the formats are
told apart.  The sections below describe the interface.


The interface
-------------

:class:`~anemoi.metadata.interface.Metadata` is organised by component.

**Checkpoint level**

``version``
   The top-level ``version`` string, or ``None`` for the oldest checkpoints.

``component_versions``
   The version of each structured component (``data``, ``task`` and
   ``lineage``).  The values are stamped in the metadata when present,
   otherwise inferred.

``dataset_names``
   The dataset names, in order.

``task``
   The task of the checkpoint (a :class:`~anemoi.metadata.interface.TaskView`).
   There is one task per checkpoint.  It gives ``task_name``, ``timestep()``,
   ``n_step_input``, ``n_step_output``, ``lagged``, ``input_offsets``,
   ``output_offsets``, ``rollout_shift``, ``advance_map``, the relative
   date indices and ``input_explicit_times`` and ``target_explicit_times`` (the
   explicit times of the training config, else the relative date indices).  For
   the temporal downscaler and time interpolator the offsets are the ones
   anemoi-training defines from ``config.task`` (``input_timestep``,
   ``output_timestep``, ``output_left_boundary``, ``output_right_boundary``):
   ``input_offsets`` and ``lagged`` are ``[0, input_timestep]``,
   ``output_offsets`` are the output timesteps inside the input interval
   (plus the boundaries when set) and ``rollout_shift`` is ``input_timestep``.
   Older files without those keys give ``lagged`` as the input explicit times
   in timesteps of the data frequency (it looks ahead), ``output_offsets`` as
   the target explicit times in the same unit, and ``rollout_shift`` as the
   last input offset.  A value is ``None`` when the task does not define it or
   the metadata cannot give it.

``lineage``
   Where the checkpoint comes from (a
   :class:`~anemoi.metadata.interface.LineageView`): ``uuid``, ``parent_uuid``
   (``None`` when the checkpoint has no parent) and ``entries`` (the
   ``checkpoint_lineage`` list, empty for a fresh run).  A value that the
   checkpoint's lineage version does not record raises
   :class:`~anemoi.metadata.exceptions.VersionError` instead of returning
   ``None``; see :doc:`versions`.

``sections``
   The raw dumps (a :class:`~anemoi.metadata.interface.SectionsView`):
   ``config``, ``provenance``, ``timestamp``, ``seeds``, ``run_id``,
   ``tracker``, ``training``, ``precision`` and ``environment``.  Use
   ``sections.get(name, default)`` for any other top-level key.

``to_dict()``
   A copy of the raw metadata, unchanged.  ``metadata["config"]`` returns a
   copy of one stored top-level key and raises ``KeyError`` for a key that is
   not stored.

**Datasets**

``metadata.dataset(name)`` returns a
:class:`~anemoi.metadata.interface.DatasetView`.  ``metadata.datasets`` is a
dictionary of all views.  The name may be omitted only when the checkpoint
has exactly one dataset.  A view has:

``name``
   The dataset name.

``data``
   The data component for this dataset: data type, grid size, variables and
   their index maps, variable types and categories, prognostic masks,
   ``select_variables``, ``typed_variables``, ``variables_metadata``,
   ``accumulations``, ``computed_forcings``, ``data_request``, ``grid``,
   ``area``, ``sources``, ``open_dataset_args``, ``data_frequency`` and
   ``dataloader_config``.  See :class:`~anemoi.metadata.interface.DatasetData`.

``arrays``
   The supporting arrays of the dataset
   (:class:`~anemoi.metadata.arrays.DatasetArrays`): ``latitudes``,
   ``longitudes``, ``grid_indices``, ``output_mask`` and the named ``masks``.
   Each is ``None`` when absent.

``number_of_grid_points``
   The number of grid points: the length of ``arrays.grid_indices`` when the
   dataset has it (a limited-area model keeps a subset of the source grid),
   else the length of ``arrays.latitudes``, else ``data.grid_size()``.
   ``None`` when none of them is known.

A tabular dataset has no grid.  Its grid values (``grid_size()``, ``grid()``,
``area()``, ``data_request()``) are ``None``, which means *not applicable*.
An empty value means *not recorded*.  The checkpoint below has a gridded
dataset and a tabular one:

.. code-block:: python

   metadata = Metadata.from_checkpoint("multi.ckpt")
   print(metadata.dataset_names)                         # ['data', 'obs']
   print(metadata.dataset("data").data.data_type())      # 'gridded'
   print(metadata.dataset("data").data.grid_size())      # 40320
   print(metadata.dataset("obs").data.data_type())       # 'tabular'
   print(metadata.dataset("obs").data.grid_size())       # None
   print(metadata.dataset("obs").arrays.latitudes)       # None

**Single-dataset shortcuts**

``metadata.variables``, ``metadata.grid_size``, ``metadata.latitudes`` and
``metadata.longitudes`` are shortcuts for checkpoints with one dataset.  With
several datasets they raise
:class:`~anemoi.metadata.interface.DatasetSelectionError`.  They never pick the first
dataset for you.

.. code-block:: python

   from anemoi.metadata import DatasetSelectionError

   try:
       metadata.variables
   except DatasetSelectionError as error:
       print(error)  # A dataset name is required: the checkpoint has 2 datasets (['data', 'obs'])

**Task timing with several datasets**

Training copies the same timing into every dataset.  The task reads all of
them.  If they agree, ``metadata.task.timestep()`` gives the shared value.
If they differ, it raises
:class:`~anemoi.metadata.components.task.TaskTimingError`.  Ask for one
dataset instead:

.. code-block:: python

   metadata.task.timestep("obs")           # '12h'
   metadata.task.for_dataset("obs")        # the full timing of that dataset
   metadata.task.timings()                 # {'data': ..., 'obs': ...}

**Overrides**

``with_overrides(patch)`` returns a new :class:`~anemoi.metadata.interface.Metadata`
built from the raw metadata with *patch* merged in.  Dictionaries merge
recursively and any other value (a list, for example) replaces the stored
one.  The supporting arrays are kept and the original is not changed.

Pass ``supporting_arrays={dataset_name: {array_name: array}}`` to replace or
add arrays of some datasets, for example the grid built by an external graph
runner.  The arrays are copied; the other arrays and datasets are kept, and a
dataset name the metadata does not have raises ``KeyError``.

.. code-block:: python

   patched = metadata.with_overrides({"config": {"data": {"frequency": "3h"}}})
   print(patched.sections.config["data"]["frequency"])   # '3h'
   print(metadata.sections.config["data"]["frequency"])  # '6h'

   # Replace the grid of one dataset
   regridded = metadata.with_overrides({}, supporting_arrays={"data": {"latitudes": lats, "longitudes": lons}})
   print(regridded.dataset("data").number_of_grid_points)

**Building metadata from a dictionary**

.. code-block:: python

   import json
   from anemoi.metadata import Metadata

   with open("metadata.json") as f:
       metadata = Metadata.from_dict(json.load(f))

   # Supporting arrays are optional: keyed by dataset name, or flat for one dataset.
   metadata = Metadata.from_dict(raw, supporting_arrays={"data": {"latitudes": lats, "longitudes": lons}})

Everything the interface returns (lists, dictionaries and arrays) is a copy.
Changing it never changes the :class:`~anemoi.metadata.interface.Metadata` it came from.


Reading and writing checkpoints
-------------------------------

.. code-block:: python

   from anemoi.metadata import (
       extract_metadata_dict, has_metadata, load_metadata,
       remove_metadata, replace_metadata, save_metadata,
   )

   has_metadata("model.ckpt")                       # True or False
   raw = extract_metadata_dict("model.ckpt")        # the stored dict, untouched
   adapter = load_metadata("model.ckpt")            # a CheckpointAdapter
   adapter, arrays = load_metadata("model.ckpt", supporting_arrays=True)

   save_metadata("model.ckpt", metadata_dict)       # checkpoint has no metadata yet
   replace_metadata("model.ckpt", metadata_dict)    # checkpoint already has metadata
   remove_metadata("model.ckpt")

By default the two writers validate strictly and stamp the metadata as format
3.1.  Pass ``validate=False`` to write a dictionary as it is.  See
:ref:`lenient-and-strict`.


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
