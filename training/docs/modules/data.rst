######
 Data
######

This module is used to initialise datasets (constructed using
anemoi-datasets) and load data into the model. It performs
validation checks, such as ensuring that the training dataset end date is
before the start date of the validation dataset.

Each training, validation, and test dataset is a map-style dataset:
an index selects one sample, which ``__getitem__`` reads from all data
readers. The datamodule pairs each dataset with a sampler that decides
which samples a rank loads and in which order, and with a dataloader
that records its position so that training can resume in the middle of
an epoch.

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
* Handling grid sharding for distributed training

Shuffling and the split of samples between groups of ranks are done by the
sampler that the datamodule builds, not by the dataset.

.. note::

   Users wishing to change the format of the batch input into the model
   should sub-class ``MultiDataset`` and override the ``get_sample``
   method.

Data Loading Policy
--------------------

The ``DataLoadingPolicy`` pairs the dataset construction above with the
batch sampler that orders, shards and batches its samples, so the two always
stay consistent with each other. It is selected via the
``dataloader.loading_policy`` config entry. The default,
``SynchronizedLoadingPolicy``, builds one synchronized ``MultiDataset`` and
splits the samples of a distributed sampler into batches with torchdata's
``BatchSampler``, which remembers its position so that training can resume
in the middle of an epoch.

API Reference
=============

Dataset Readers
---------------

.. automodule:: anemoi.training.data.readers
   :members:
   :no-undoc-members:
   :show-inheritance:

Multi-Dataset API
-----------------

.. automodule:: anemoi.training.data.multidataset
   :members:
   :no-undoc-members:
   :show-inheritance:

Data Loading Policy API
------------------------

.. automodule:: anemoi.training.data.loading
   :members:
   :no-undoc-members:
   :show-inheritance:
