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

Multi-Domain
------------

``CrossDatasetLoadingPolicy`` builds one independently indexed
``MultiDataset`` per domain. Its batch sampler visits domains in a random (seeded)
order and keeps every batch domain-homogeneous, including when the configured
batch size is greater than one. The readers may use different grids and date
ranges.

Select it in the dataloader configuration::

   dataloader:
     loading_policy:
       _target_: anemoi.training.data.loading.CrossDatasetLoadingPolicy

Common variables are checked for metadata compatibility when training starts.
Flat encoder-processor-decoder models can route each domain through its own
hidden mesh while sharing the encoder, processor, and decoder weights. Each
mapped hidden node set must have corresponding domain-to-hidden,
hidden-to-hidden, and hidden-to-domain edges in the graph. Hidden node and edge
feature dimensions must match because the model weights remain shared.

The model routing names the hidden mesh for each domain and assigns the same
encoder and decoder group to both domains::

   model:
     model:
       hidden_nodes_name:
         data: hidden
         datb: hidden_b
     encoders:
       shared:
         source_datasets: [data, datb]
         dataset_fusing_strategy: not_supported
         mapper:
           _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
           # ... mapper configuration
     decoders:
       shared:
         target_datasets: [data, datb]
         target_node_features: [encoded_data]
         mapper:
           _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
           # ... mapper configuration

The ``shared`` key is user-defined; datasets listed in the same group use the
same mapper weights. ``hidden`` and ``hidden_b`` must be nodes in the
graph configuration, with ``data -> hidden -> data`` and
``datb -> hidden_b -> datb`` edges. The complete example is the top-level
``multidomain`` configuration, which builds on the single-domain stretched-grid
configurations (first domain ``data``) and adds the second domain ``datb``; its
graph is defined in the ``graph=multidomain`` group.

With the current distributed strategy, domain-specific trainable node and edge
features must be disabled because the active domain, and therefore the active
graph-specific parameters, changes between batches.

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

Data Loading Policy API
------------------------

.. automodule:: anemoi.training.data.loading
   :members:
   :no-undoc-members:
   :show-inheritance:
