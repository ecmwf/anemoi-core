#######################################
Shared Encoder–Decoder (Multi-Domain)
#######################################

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

Multi-domain models can be trained using deterministic objectives or probabilistic objectives such as the Continuous Ranked Probability Score (CRPS). Support for diffusion-based training is planned for a future release.

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

The following example illustrates two global ERA5 datasets on different grids: ``era5_o96`` (O96) and ``era5_n320`` (N320). Both are intended to use the shared model architecture. The example retains the dataset selection and diagnostic settings from the original configuration.

.. code-block:: yaml

   data:
       datasets:
           era5_o96:
               dataset_config:
                   dataset: ${system.input.dataset.era5_o96}
                   frequency: ${data.frequency}
                   drop: [u_600, z]
                   select: [2t, tp, u_500, tp, cp]
                   reorder: sort
                   statistics: /path/to/statistics.zarr
                   start: 1985
                   end: 2020
                   trajectory: null
               diagnostic: [tp, cp]
           era5_n320:
               dataset: ${system.input.dataset.era5_n320}
               frequency: ${data.frequency}
               drop: ${data.datasets.era5_o96.dataset_config.drop}
               select: ${data.datasets.era5_o96.dataset_config.select}
               reorder: sort
               start: 1979
               end: 2000
               trajectory: null
               diagnostic: ${data.datasets.era5_o96.diagnostic}

The second dataset reuses the first dataset's ``drop``, ``select``, and ``diagnostic`` settings through configuration references. This helps keep the variable selection consistent across datasets, even when their grids and time ranges differ.

.. note::

   Check the nesting of ``dataset_config`` and the placement of ``start``, ``end``, and ``trajectory`` against your project's configuration schema before using this example. The structure above preserves the supplied example; it has not been validated against the training code.

Graph Creation
**************

Multi-domain training requires graph definitions that support each dataset's spatial structure. Since datasets may have different resolutions, geographical domains, and numbers of spatial nodes, their graphs can differ. The graph recipe must define the data nodes associated with each dataset, using the dataset names to identify the corresponding nodes. For example, when training with ``era5_o96`` and ``era5_n320``, the graph configuration must provide appropriate data-node definitions for both datasets.

During training, the graph associated with the selected dataset is used to encode and decode its spatial representation, while the shared model parameters are reused across the different graph structures.

.. note::

   Multi-domain training can support different hidden mesh sizes and structures, provided the graph configuration and shared model architecture are compatible.