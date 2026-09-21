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

The data module provides two types of dataset readers that wrap
anemoi-datasets data:

Native Grid Dataset
-------------------

The ``NativeGridDataset`` class is used for standard atmospheric data
on a native grid. It provides a simple interface for reading data samples
at specified time indices.

Trajectory Dataset
------------------

The ``TrajectoryDataset`` class extends ``NativeGridDataset`` to support
trajectory-based sampling, where data is organized into temporal
trajectories. This is useful for tracking atmospheric features over time
or for specialized training strategies that require trajectory awareness.

Trajectories are defined by:

* **Trajectory start**: The reference datetime from which trajectories begin
* **Trajectory length**: The number of time steps in each trajectory

Each sample in the dataset is associated with a trajectory ID, ensuring
that samples are correctly grouped and that trajectory boundaries are
respected during training.

Multi-Dataset
-------------

The ``MultiDataset`` class provides a higher-level wrapper that can
synchronize and combine multiple datasets (either ``NativeGridDataset``
or ``TrajectoryDataset`` instances). This is the primary interface used
for training and supports:

* Synchronizing samples across multiple datasets with different grids
* Managing distributed data loading across workers and communication groups
* Shuffling and batching data for training
* Handling grid sharding for distributed training

.. note::

   Users wishing to change sample selection or the format of the batch input
   should subclass the configured iteration (``BaseIteration`` or
   ``CrossDatasetIteration``) and select it in ``dataloader.iteration``.
   Override ``MultiDataset.__iter__`` only when replacing the complete sampling
   workflow.

Multi-Domain
------------

``MultiDataset`` combines independent domains when configured with
``CrossDatasetIteration``. The default ``BaseIteration``
returns synchronized data from every reader in each sample, whereas
cross-dataset sampling returns data from one reader at a time. The readers may
have different grids and date ranges. Each domain is partitioned independently
across distributed sample groups and data-loader workers.

The data module selects multi-domain sampling through the iteration configuration::

   dataloader:
     iteration:
       _target_: anemoi.training.data.iteration.CrossDatasetIteration
     check_dataset_units: true
     batch_size:
       training: 1
       validation: 1
       test: 1

A batch size of one is currently required because each sample contains one
domain key.

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

Iteration API
-------------

.. automodule:: anemoi.training.data.iteration
   :members:
   :no-undoc-members:
   :show-inheritance:
