.. (C) Copyright 2026- Anemoi contributors.
..
.. This software is licensed under the terms of the Apache Licence Version 2.0
.. which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
..
.. In applying this licence, ECMWF does not waive the privileges and immunities
.. granted to it by virtue of its status as an intergovernmental organisation
.. nor does it submit to any jurisdiction.

Formats, components and versions
================================

This page explains where the metadata lives in a checkpoint, which formats
exist, how a format and the version of each component are detected, how
reading and writing differ, and how to add support for a new version.


Where the metadata is stored
----------------------------

A checkpoint is a zip file with one top-level directory.  The metadata is a
JSON file inside it, with the supporting arrays next to it:

.. code-block:: text

   <top-dir>/anemoi-metadata/anemoi.json
   <top-dir>/anemoi-metadata/<dataset>/latitudes.numpy
   <top-dir>/anemoi-metadata/<dataset>/longitudes.numpy
   ...

Older checkpoints name the file ``ai-models.json``.  The package finds the
file by name, so the name of the top-level directory does not matter.  The
list of array files is stored in the JSON under ``supporting_arrays_paths``.
Arrays are grouped by dataset name in the newer formats.  Older
single-dataset checkpoints store them flat, and the package assigns them to
the only dataset.

Reading and writing the file itself is done with ``anemoi-utils``.  Use
:func:`~anemoi.metadata.checkpoint.load_metadata`,
:func:`~anemoi.metadata.checkpoint.save_metadata` and the other functions of
:mod:`anemoi.metadata.checkpoint`, or the :doc:`command line tool <cli>`.


Formats
-------

The top-level ``version`` key names the format.  Policy: a **minor** change is
additive (new optional keys).  A **major** change alters the layout.  The
oldest checkpoints have no ``version``.

.. list-table::
   :header-rows: 1
   :widths: 12 12 76

   * - Name
     - ``version``
     - What it is
   * - F0
     - absent
     - The ai-models era.  No ``metadata_inference`` block.  May lack
       ``variables_metadata``, ``data_request`` or ``dataset.shape``.
   * - F1
     - ``"1.0"``
     - Single-dataset anemoi-training.  No ``metadata_inference`` block.
   * - F2 to F6
     - ``"2.0"``
     - Multi-dataset layout with a ``metadata_inference`` block (dataset
       names and one block per dataset).  The variants share the number and
       are told apart by content: F2 is the first multi-dataset form; F3 adds
       the input and output relative date indices; F4 has ``task`` at the top
       level; F5 adds flexible timestepping (forecaster offsets); F6 is what
       training wrote before the richer batch (``metadata_inference.task``,
       ``base_seed``, ``uuid``, ``checkpoint_lineage``, ``tracker`` and more;
       ``parent_uuid`` after a warm start).
   * - F8
     - ``"3.0"``
     - Richer batch, what training writes today.  F6 plus a ``data_type``
       (``"gridded"`` or ``"tabular"``) and a ``grid_size`` per dataset, and no
       ``shapes``.  Tabular datasets have an empty ``dataset`` entry and no
       supporting arrays.
   * - F9
     - ``"3.1"``
     - Written by this package.  The 3.0 layout plus ``component_versions``
       (``data``, ``task`` and ``lineage``).

The names F0 to F9 are only labels.  The golden test fixtures in
``tests/data`` are named after them (``f6_main.json``, ``f8_richer_batch.json``
and so on).

Training does not write 3.1 yet.  Checkpoints from training are read leniently
(see :ref:`lenient-and-strict`) in whatever format they have.


Components
----------

Some parts of the metadata have a known structure.  Each of these is a
*component* with a version of its own, so that a change in one does not affect
the others.

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - Component
     - Level
     - Covers
   * - ``data``
     - per dataset
     - Dataset names, ``data_indices``, ``variable_types``, ``data_type`` and
       ``grid_size`` (3.0 and later).  It also reads the unversioned
       ``dataset`` and ``config`` dumps for the data request, sources,
       variable metadata and frequency.
   * - ``task``
     - checkpoint
     - ``metadata_inference.task`` and the per-dataset ``timesteps``.  The
       adapter is chosen by task name too (``forecaster``,
       ``offset-forecaster``, ``temporal-downscaler``, ``timeless``,
       ``autoencoder``).  ``time-interpolator`` is an alias of
       ``temporal-downscaler``: older checkpoints record temporal downscalers
       under that name and are read with the same adapter.  An unknown task
       name uses the generic task adapter and warns.
   * - ``lineage``
     - checkpoint
     - ``uuid``, ``parent_uuid`` and ``checkpoint_lineage``: where the
       checkpoint comes from.  Exposed as ``metadata.lineage``.

All other top-level keys are **raw sections** and carry no version:
``config``, ``dataset``, ``provenance_training``, ``training``, ``tracker``,
``model``, ``seed``, ``base_seed``, ``run_id`` and ``timestamp``.  They are
read as stored through the sections reader (``metadata.sections``).

Component versions are not the same numbers as the format version:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Component version
     - Layout
     - Used for
   * - ``0.1``
     - legacy
     - F0 and F1 (no ``metadata_inference`` block).
   * - ``0.2``
     - 2.0
     - F2 to F6.
   * - ``1.0``
     - 3.x
     - F8 and F9.

The ``data`` and ``task`` numbers follow the layout above.  The ``lineage``
numbers follow what the file records, not the format:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``lineage`` version
     - What the file records
   * - ``0.1``
     - Nothing: no ``uuid``, ``parent_uuid`` or ``checkpoint_lineage`` (F0,
       some F1).
   * - ``0.2``
     - A ``uuid``, and a ``parent_uuid`` after a warm start; no
       ``checkpoint_lineage`` (F1 to F5).
   * - ``1.0``
     - ``checkpoint_lineage``, a list that is empty for a fresh run (F6 and
       later).

``component_versions``
^^^^^^^^^^^^^^^^^^^^^^

Format 3.1 stamps the version of each component in a top-level map:

.. code-block:: json

   {
     "version": "3.1",
     "component_versions": {"data": "1.0", "task": "1.0", "lineage": "1.0"}
   }

Stamps are used when present.  Entries for unknown components, and values that
are not strings, are ignored.


How versions are detected
-------------------------

1. The stamp in ``component_versions`` wins, one component at a time.
2. Without a stamp, the version is inferred from ``version`` and the content:

   - no ``metadata_inference`` block: ``0.1``;
   - ``version`` absent, ``"1.x"`` or ``"2.x"``: ``0.2``;
   - ``version`` ``"3.x"``: ``1.0``;
   - ``version`` absent but a dataset block has ``data_type`` or
     ``grid_size``: ``1.0``;
   - ``lineage`` ignores ``version`` and follows the content:
     ``checkpoint_lineage`` present (a list) is ``1.0``, else ``uuid`` or
     ``parent_uuid`` present is ``0.2``, else ``0.1``.

3. The task name comes from ``metadata_inference.task``, else the top-level
   ``task`` (F4), else ``"forecaster"`` (what inference assumes).
4. The registry then picks the adapter.  An exact match is used.  Otherwise
   the nearest older registered version is used.  A version newer than any
   known one (a checkpoint from a newer package) is read with the newest
   known adapter and a warning.  A newer ``version`` without a
   ``metadata_inference`` block also warns; there is nothing newer to read it
   with, so it is read as legacy.

.. code-block:: python

   import json
   from anemoi.metadata import Metadata

   with open("tests/data/f8_richer_batch.json") as f:
       metadata = Metadata.from_dict(json.load(f))

   print(metadata.version)             # '3.0'
   print(metadata.component_versions)  # {'data': '1.0', 'task': '1.0', 'lineage': '1.0'}

The adapter classes are in :mod:`anemoi.metadata.components`; the detection
code is in :mod:`anemoi.metadata.formats`.


How the adapters are layered
----------------------------

A :class:`~anemoi.metadata.base.CheckpointAdapter` holds a copy of the raw
dictionary.  It composes one task adapter, one lineage adapter, one data adapter
**per dataset** and the sections reader, and exposes them as ``.task``,
``.lineage``, ``.data(name)`` (with ``.dataset_names``) and ``.sections``.
Each component module owns how its adapters are chosen and built, through a
``build_*`` function that takes the raw dictionary and the component version:
:func:`~anemoi.metadata.components.task.build_task_adapter`,
:func:`~anemoi.metadata.components.data.build_data_adapters` (also the
dataset names) and
:func:`~anemoi.metadata.components.lineage.build_lineage_adapter`.  The
checkpoint adapter only calls them, with the versions from
:mod:`anemoi.metadata.formats`.
:class:`~anemoi.metadata.interface.Metadata` is the user-facing layer on top of it.

Each adapter implements its getters against one layout.  An adapter for a
newer or older layout **subclasses the nearest one and overrides only what
differs**.  Adapters register themselves with
:class:`~anemoi.metadata.registry.ComponentRegistry`.

Data and task adapters vary on two axes: the component version and a name.
Each is built from a *version reader*, which knows where the values are
stored, and a *behaviour*, which knows what the kind of thing derives or stores.
Where both define the same method, the behaviour wins.

* Task: the name is the task (``forecaster``, ``temporal-downscaler``, ...).
  Readers are ``TaskReaderV0_2``, ``TaskReaderV1_0``, ... and behaviours are
  ``ForecasterBehaviour``, ...;
  :func:`~anemoi.metadata.components.task.register_task_version` combines one
  reader with every behaviour and registers the results under
  ``("task", version, name)``.  There is one task adapter per checkpoint.
* Data: the name is the dataset's ``data_type`` (``gridded`` or ``tabular``).
  Readers are ``DataReaderV0_1`` (legacy), ``DataReaderV0_2`` and
  ``DataReaderV1_0``; behaviours are ``GriddedBehaviour`` and
  ``TabularBehaviour`` (a tabular dataset has no grid, data request, area or
  grid size).
  :func:`~anemoi.metadata.components.data.register_data_version` registers the
  combinations under ``("data", version, data_type)``.  A data adapter is built
  for one dataset, because a checkpoint can mix gridded and tabular datasets
  (``f8_richer_batch`` does), so its methods take no dataset name.  The
  behaviour is chosen from the ``data_type`` the version reader reports.

A change for a new version is therefore one reader that reaches every task name
or data type.  The lineage component has no second axis: one adapter per
version (``LineageV0_1``, ``LineageV0_2``, ``LineageV1_0``), each subclassing
the previous one.


.. _version-gate:

The version gate
----------------

A value that older component versions do not record, and that has no fallback
for them, is **gated**.  On a checkpoint whose component is too old, a gated
accessor raises :class:`~anemoi.metadata.exceptions.VersionError` instead of
returning ``None`` or an empty value, because that would read as *not set*
when the truth is *not recorded*.  The error names the component, the version
required and the version the checkpoint has.

The gate is the decorator :func:`~anemoi.metadata.components.base.requires`
on a component adapter method; the version compared is that of the adapter
(stamped or inferred).  Put it under ``@property`` for a property:

.. code-block:: python

   from anemoi.metadata import requires

   class LineageV0_2(LineageV0_1):
       @property
       @requires("0.2")
       def uuid(self): ...

The lineage component uses it.  ``uuid`` and ``parent_uuid`` need ``0.2`` and
``entries`` needs ``1.0``, so on an F0 file all three raise and on an F3 file
``entries`` raises:

.. code-block:: python

   from anemoi.metadata import Metadata, VersionError

   metadata = Metadata.from_checkpoint("f3.ckpt")   # lineage 0.2
   metadata.lineage.uuid                            # the uuid
   try:
       metadata.lineage.entries
   except VersionError as error:
       print(error)   # lineage.entries requires lineage version >= 1.0, but ...

``parent_uuid`` reads ``None`` from ``0.2`` on when the file has no parent: a
fresh run records ``checkpoint_lineage == []`` and no ``parent_uuid``, and that
means *no parent*, not *unknown*.  Only ``0.1``, which records nothing, raises.
From ``1.0`` on, when ``parent_uuid`` is missing but the last
``checkpoint_lineage`` entry records a parent, that parent is returned; the
strict write rejects such a file.

Values with a fallback are not gated: the data and task adapters derive or
look up what older layouts lack (for example the grid size from
``dataset.shape``, or the offsets of a forecaster from its timestep), and a
value that is genuinely absent reads as ``None`` or empty, documented as *not
recorded*.


.. _lenient-and-strict:

Lenient read, strict write
--------------------------

**Reading is lenient.**  Opening a checkpoint never fails on unknown keys or on
missing optional keys.  Unknown keys are kept, and
:meth:`~anemoi.metadata.interface.Metadata.to_dict` returns the content unchanged.
Values that cannot be known come back as ``None`` or as an empty value.

**Writing is strict.**  :func:`~anemoi.metadata.checkpoint.save_metadata` and
:func:`~anemoi.metadata.checkpoint.replace_metadata` call
:func:`~anemoi.metadata.write_schema.validate_for_write` by default.  It:

- requires ``component_versions`` and checks each component against the
  strict schema of its stamped version (``lineage`` ``1.0`` requires ``uuid``
  and ``checkpoint_lineage``, and a ``parent_uuid`` that matches the last
  entry, an omitted one counting as ``None``);
- checks that ``config`` and a ``dataset`` entry for each dataset exist (the
  inner structure of raw sections is never checked);
- rejects unknown keys in dataset blocks, and datasets with different
  ``timesteps`` (the task is per checkpoint);
- rejects inconsistent task timing: input and output relative date indices
  that are not among the training indices (which may repeat an index, so they
  are compared as a set), and, for the offset forecaster, offset lists whose
  length differs from their indices;
- sets ``version`` to ``CURRENT_VERSION`` (``"3.1"``).  A missing ``version``
  or ``"3.0"`` is relabelled.  Any other version, such as ``"2.0"``, and any
  unknown stamp is rejected.  Versions are never changed silently.

All problems are collected and raised together as
:class:`~anemoi.metadata.write_schema.WriteValidationError`.

.. code-block:: python

   import copy
   import json
   from anemoi.metadata import (
       COMPONENT_VERSIONS, CURRENT_VERSION, WriteValidationError, validate_for_write,
   )

   with open("tests/data/f8_richer_batch.json") as f:
       raw = json.load(f)

   try:
       validate_for_write(raw)                 # no stamps yet
   except WriteValidationError as error:
       print(error.errors)                     # ['component_versions: required ...']

   stamped = copy.deepcopy(raw)
   stamped["component_versions"] = dict(COMPONENT_VERSIONS)   # {'data': '1.0', 'task': '1.0', 'lineage': '1.0'}
   ready = validate_for_write(stamped)
   print(ready["version"])                      # '3.1'

An older format such as F6 cannot go through the strict path.  To write
metadata as it is, for example to repair an old checkpoint, pass
``validate=False``:

.. code-block:: python

   from anemoi.metadata import replace_metadata

   replace_metadata("old.ckpt", edited_dict, validate=False)

The ``load`` and ``edit`` commands of the :doc:`command line tool <cli>` write
with ``validate=False``, because they must handle every format.


Adding a component version or a task
------------------------------------

Say a new training release stores the timestep of the task as an integer
number of hours, ``timestep_hours``, in ``timesteps``.  The recipe is the same
for a new data version or a new task name.

1. **Add a golden fixture.**  Put a trimmed real metadata file in
   ``tests/data/`` (see ``tests/data/README.md`` and ``build_fixtures.py``)
   and add it to the expectation table in ``tests/unit/test_golden.py``.
   Real metadata, not hand-written, keeps the tests in step with what
   training writes.

2. **Subclass the nearest version reader and register it.**

   * *Data*: subclass the nearest data reader and register it for every data
     type with ``register_data_version(version, reader)``.
   * *Task*: subclass the nearest task reader and register it for every task
     name with ``register_task_version(version, reader)``.

3. **Override only what differs, and gate what has no fallback.**  A data reader overrides what it reads:
   ``recorded_data_type``, ``variable_indices``, ``output_variable_indices``,
   ``variable_types``, ``variables_metadata``, ``_dataset_section`` and
   ``_recorded_grid_size``.  A task version reader overrides the reading hooks
   ``_timesteps``, ``_config_timestep``, ``_default_n_step_input`` and
   ``_default_n_step_output``.

   .. code-block:: python

      from anemoi.metadata.components.task import TaskReaderV1_0
      from anemoi.metadata.components.task import register_task_version
      from anemoi.metadata.write_schema import register_write_schema


      class TaskReaderV1_1(TaskReaderV1_0):
          """Reader of a task that stores the timestep as ``timestep_hours``."""

          def _timesteps(self, dataset_name):
              timesteps = dict(super()._timesteps(dataset_name))
              if "timestep_hours" in timesteps:
                  timesteps["timestep"] = f"{timesteps['timestep_hours']}h"
              return timesteps


      register_task_version("1.1", TaskReaderV1_1)

   Put the class in the component's module (``components/task.py`` or
   ``components/data.py``) and make the call there, next to the existing
   ones.  The module's ``build_*`` function already selects the new adapters
   by version.  ``components/__init__.py`` imports those modules, so the
   adapter is registered when the package is imported.  A new *component*
   (not a new version) provides its own ``build_<component>_adapter`` function
   in its module, which ``CheckpointAdapter`` calls, and its module must be
   added to the imports there.  The task reader reaches forecaster, offset
   forecaster, temporal downscaler, time interpolator, timeless and
   autoencoder alike; a data reader reaches gridded and tabular datasets alike.

   .. code-block:: python

      from anemoi.metadata.components.data import DataReaderV1_0
      from anemoi.metadata.components.data import register_data_version


      class DataReaderV1_1(DataReaderV1_0):
          """Reader of a data layout that stores the grid size as ``points``."""

          def _recorded_grid_size(self):
              stored = self._block().get("points")
              return stored if isinstance(stored, int) else super()._recorded_grid_size()


      register_data_version("1.1", DataReaderV1_1)

   Use the gate (:ref:`version-gate`) for **new values without a fallback**: a
   value the new version adds that older versions cannot give is a method
   decorated with ``@requires("<new version>")``, so older files raise
   ``VersionError`` rather than read as ``None``.  A value older versions can
   derive gets the fallback instead.

4. **Check detection.**  A stamped file (``"component_versions": {"task":
   "1.1"}``) selects the new adapters at once.  If unstamped files with the
   new layout exist, teach :func:`anemoi.metadata.formats.inferred_version`
   to recognise them.

5. **Add the write schema and bump the stamp.**  Register a strict schema for
   the new component version, then update the constants in
   ``write_schema.py``.  A schema is a function that takes the raw dictionary
   and a list of error messages and appends the problems it finds:

   .. code-block:: python

      @register_write_schema("task", "1.1")
      def _validate_task_v1_1(metadata, errors):
          ...   # append a readable message to ``errors`` for each problem

   Then set ``COMPONENT_VERSIONS["task"] = "1.1"`` and raise
   ``CURRENT_VERSION`` (a minor step if the change is additive, a major step
   if the layout changes).  ``validate_for_write`` accepts only a missing
   ``version``, ``"3.0"`` and ``CURRENT_VERSION`` as input, so add the
   previous current version to ``_RELABELLED_VERSIONS`` if it must stay
   writable.  Other components and the older schemas are not touched.

6. **Test it.**  Run the golden tests.  A test that registers an adapter
   should use the ``registry_state`` fixture from ``tests/conftest.py``, which
   restores the registry afterwards.  The same applies in a notebook or
   script: ``ComponentRegistry.snapshot()`` and ``ComponentRegistry.restore()``
   isolate your registrations.

   .. code-block:: python

      raw["component_versions"] = {"data": "1.0", "task": "1.1"}
      metadata = Metadata.from_dict(raw)
      print(metadata.task.timestep())   # '12h' when timestep_hours is 12

A new task name works the same way on the other axis: write a behaviour class
in ``components/task.py`` (it overrides ``_extend`` to add what the task
derives or stores) and add it to ``TASK_BEHAVIOURS`` under the task name.  The
existing ``register_task_version`` calls then register it for every component
version (``0.2`` and ``1.0``).  A new data type is a behaviour class in
``components/data.py`` added to ``DATA_BEHAVIOURS`` under the ``data_type``.
