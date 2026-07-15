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

   Users wishing to change the format of the batch input into the model
   should sub-class ``MultiDataset`` and override the ``__iter__``
   method or the ``get_sample`` method.

Multi-Domain Dataset
--------------------
The ``MultiDomainDataset`` class provides an alternative to ``MultiDataset``
for synchronizing and combining multiple datasets that have different grids
and time indices, but share similar data structures. The dataset indices are
shuffled and a single one is passed forward.

Advantages of this training strategy include:
* Optimal utilization of dataset similarities, avoiding the need for
unnecessary additional encoders
* Easier to achieve cross-domain or cross-resolution generalization as the
model is able to continuously see different samples
* Avoiding additional transfer learning steps when fine-tuning to a new
domain, which can cause catastrophic forgetting.
* Easier to get a good prediction on an unseen domain

Similar to ``MultiDataset``, distributed data loading across workers and
communication groups is supported, as well as shuffling and batching data
for training.

API Reference
=============

Dataset Readers
---------------

.. automodule:: anemoi.training.data.dataset
   :members:
   :no-undoc-members:
   :show-inheritance:

Multi-Dataset
-------------

.. automodule:: anemoi.training.data.multidataset
   :members:
   :no-undoc-members:
   :show-inheritance:

Multi-Domain Dataset
--------------------

.. automodule:: anemoi.training.data.multidomain
   :members:
   :no-undoc-members:
   :show-inheritance:
