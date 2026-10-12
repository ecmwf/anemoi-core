######
 Data
######

This module is used to initialise datasets (constructed using
anemoi-datasets) and load data into the model. It performs
validation checks, such as ensuring that the training dataset end date is
before the start date of the validation dataset.

The dataset files contain functions which define how datasets get
split between workers (``worker_init_func``) and how datasets are
iterated across to produce data batches that get fed as input into
the model (``__iter__``).

Dataset Architecture
====================

The data module provides three types of dataset readers that wrap
anemoi-datasets data:

Gridded Data Reader
-------------------

The ``GriddedDataReader`` class is used for standard atmospheric data
on a native grid. It provides a simple interface for reading data samples
at specified time indices.

Tabular Data Reader
-------------------

The ``TabularDataReader`` class reads tabular (observation) datasets,
selected when the dataset configuration has a ``window`` and a
``frequency``. Each sample holds the observations of its time windows,
with per-observation coordinates and timedeltas. The timedeltas are in
seconds from the sample's reference time, so observations in earlier
windows have more negative values; during rollout they are measured from
the forecast time of the current step.

Trajectory Data Reader
----------------------

The ``TrajectoryDataReader`` class extends ``GriddedDataReader`` to read
5-D ``trajectories``-layout datasets (forecast initialisations x steps),
selected when the dataset configuration has a ``trajectory`` section.
Each forecast initialisation is an independent sequence and the forecast
step is the position within it, so a training sample never crosses
initialisation boundaries.

The optional ``trajectory.sampling.stride`` sets the spacing between
sample anchors within a sequence: ``null`` (the default) gives
non-overlapping windows and ``1`` keeps every valid position.

Multi-Dataset
-------------

The ``MultiDataset`` class provides a higher-level wrapper that can
synchronize and combine multiple datasets (``GriddedDataReader``,
``TabularDataReader`` or ``TrajectoryDataReader`` instances). This is the primary interface used
for training and supports:

* Synchronizing samples across multiple datasets with different grids
* Managing distributed data loading across workers and communication groups
* Shuffling and batching data for training
* Handling grid sharding for distributed training

.. note::

   Users wishing to change the format of the batch input into the model
   should sub-class ``MultiDataset`` and override the ``__iter__``
   method or the ``get_sample`` method.

API Reference
=============

Dataset Readers
---------------

.. automodule:: anemoi.training.data.data_reader
   :members:
   :no-undoc-members:
   :show-inheritance:

Multi-Dataset API
-----------------

.. automodule:: anemoi.training.data.multidataset
   :members:
   :no-undoc-members:
   :show-inheritance:
