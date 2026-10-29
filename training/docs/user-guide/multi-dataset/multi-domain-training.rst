###########################################
Multi-Domain Training (Shared Architecture)
###########################################

Multi-domain training enables multiple datasets to share a single encoder, processor, and decoder within one training workflow. Unlike the multiple encoder–decoder approach, in which each dataset has its own encoder and decoder, this setup uses shared model parameters across datasets. It supports training with different spatial resolutions, grid structures, and geographical domains, making it useful for applications such as downscaling, combining global and regional datasets, and improving model generalisation across domains and resolutions.

Overview
********

In a shared encoder–decoder configuration, the same model processes datasets with different spatial grids and numbers of nodes. Each training iteration selects data from one of the configured datasets and uses its corresponding graph. The shared encoder maps the input to a latent representation, the shared processor operates on that representation, and the shared decoder produces predictions on the selected dataset's grid. All datasets contribute to updating the same model parameters, allowing the model to learn across different domains and resolutions.

Datasets may cover different geographical regions and do not need to overlap spatially. They can also have dataset-specific readers, preprocessing settings, and normalisation statistics, provided the resulting inputs and outputs are compatible with the shared architecture.

.. warning::

   Datasets must use compatible input and output variable definitions and consistent variable ordering. Dataset-specific preprocessing and normalisation do not remove this requirement.

Training Procedure
******************

Multi-domain training aims to learn a shared representation across spatial domains and resolutions rather than training separate encoders and decoders for individual datasets. During training, data and the corresponding graph are selected from the configured datasets according to the sampling strategy. The shared encoder, processor, and decoder then process the batch, after which the loss is calculated and the shared parameters are updated. The selected grid, resolution, and number of spatial nodes may change between iterations, while the model parameters remain shared.

The following illustration shows multi-domain training with datasets that differ in their spatial characteristics.

.. image:: ../../images/multi-dataset/multi-domain-sg-lam.png
   :alt: Multi-domain training with different spatial grids and resolutions
   :width: 600px
   :align: center

The animation below shows datasets passing through the shared encoder–decoder architecture during training.

.. image:: ../../images/multi-dataset/multi-domain.gif
   :alt: Animation showing multi-domain training with a shared encoder and decoder
   :width: 400px
   :align: center

Multi-domain models can be trained using deterministic objectives, probabilistic objectives such as the Continuous Ranked Probability Score (CRPS), or transport objectives such as EDM diffusion.

Dataset Configuration
*********************

The shared encoder–decoder approach follows the same general dataset reader structure as the multiple encoder–decoder approach. Each dataset is configured under ``dataloader.training.datasets``, where its data source, time range, and optional processing settings are specified. For example, a single ERA5 dataset can be configured as follows:

.. code-block:: yaml

   dataloader:
       training:
           datasets:
               era5:
                   dataset_config:
                       dataset: ${system.input.dataset}
                       frequency: ${data.frequency}
                       drop: []
                       select: []
                       statistics: /path/to/statistics.zarr
                   start: 1985
                   end: 2020
                   trajectory: null

The ``dataset_config`` dictionary is passed to ``open_dataset`` to specify how the dataset is opened and processed. Conceptually, this corresponds to:

.. code-block:: python

   open_dataset({
       "dataset": ...,
       "frequency": ...,
       "drop": ...,
       "select": ...,
       "statistics": ...,
   })

Additional datasets can be introduced by adding entries under ``dataloader.training.datasets``. Each entry can specify its own data source, temporal coverage, and preprocessing options while following the same configuration structure.

**Example multi-dataset configuration**

The following example illustrates two global ERA5 datasets on different grids, ``era5_o96`` (O96) and ``era5_n320`` (N320), with different date ranges:

.. code-block:: yaml

   dataloader:
       training:
           datasets:
               era5_o96:
                   dataset_config:
                       dataset: /path/to/era5_o96.zarr
                       frequency: ${data.frequency}
                       drop: [u_600, z]
                   start: 1985
                   end: 2020
               era5_n320:
                   dataset_config:
                       dataset: /path/to/era5_n320.zarr
                       frequency: ${data.frequency}
                       drop: ${dataloader.training.datasets.era5_o96.dataset_config.drop}
                   start: 1979
                   end: 2000

The second dataset reuses the first dataset's ``drop`` setting through a configuration reference. This helps keep the variable selection consistent across datasets, even when their grids and time ranges differ.

Loading Policy and Model Configuration
**************************************

Multi-domain training is enabled by selecting the cross-dataset loading policy, which builds one independently indexed dataset per domain and keeps every batch domain-homogeneous:

.. code-block:: yaml

   dataloader:
       loading_policy:
           _target_: anemoi.training.data.loading.CrossDatasetLoadingPolicy

The model must then name the hidden mesh of each dataset under ``model.model.hidden_nodes_name`` and list the datasets that share an encoder or decoder group under ``source_datasets`` and ``target_datasets``. The configuration is described in :doc:`/modules/data`, and a complete example is the top-level ``multidomain`` configuration.

The following constraints apply:

- Hierarchical models do not support per-dataset hidden meshes.
- All datasets must have the same variables in the same order, because the shared encoder and decoder weights are tied to the variable position.
- Hidden node and edge feature dimensions must match across the hidden meshes, because the weights are shared.
- Trainable node and edge parameters must be set to 0, because the active graph-specific parameters would change between batches.

Graph Creation
**************

Multi-domain training requires graph definitions that support the spatial structure of each dataset. Since datasets may differ in resolution, geographical domain, and number of spatial nodes, their graph structures can also vary. The graph recipe must define the data nodes associated with each dataset, using the dataset names to identify the corresponding nodes. For example, when training with ``era5_o96`` and ``era5_n320``, the graph configuration must define the data nodes, hidden meshes, and corresponding data-to-hidden, hidden-to-hidden, and hidden-to-data edges for both datasets.

During training, the graph associated with the selected dataset is used to encode and decode its spatial representation, while the encoder, processor, and decoder parameters remain shared across the different graph structures.

One of the key features of multi-domain training is its ability to support different hidden mesh sizes and structures, provided they are compatible with the shared model architecture. For example, training can alternate between batches from global datasets at O96 and N320 resolutions, each with its own hidden mesh resolution. This can be configured by defining separate data and hidden nodes for each dataset in the graph configuration file:

.. code-block:: yaml

   nodes:
       o96:
           node_builder:
               _target_: anemoi.graphs.nodes.AnemoiDatasetNodes
               dataset: ${dataloader.data_o96}
           attributes: ${graph.attributes.nodes}
       hidden_o96:
           node_builder:
               _target_: anemoi.graphs.nodes.TriNodes
               resolution: 5
       n320:
           node_builder:
               _target_: anemoi.graphs.nodes.AnemoiDatasetNodes
               dataset: ${dataloader.data_n320}
           attributes: ${graph.attributes.nodes}
       hidden_n320:
           node_builder:
               _target_: anemoi.graphs.nodes.TriNodes
               resolution: 7

This configuration creates separate data and hidden nodes for each grid, allowing the O96 and N320 datasets to use different hidden mesh resolutions. The corresponding hidden nodes must then be mapped to their respective datasets in the model configuration, located under ``config/model/`` (see the processor section):

.. code-block:: yaml

   model:
       num_channels: ...
       condition_on_residual: ...
       cpu_offload: ...
       keep_batch_sharded: ...
       model:
           _target_: ...
           hidden_nodes_name:
               o96: hidden_o96
               n320: hidden_n320
