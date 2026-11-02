Multi-Dataset Training
======================

Since ``anemoi-training`` version 0.4, the framework supports training with multiple datasets within a single training workflow. This enables applications such as downscaling, combining global and regional datasets, and leveraging diverse data sources to improve model generalisation.

There are two ways to load multiple datasets during training:

- **Synchronized multi-dataset training:** Every sample contains all configured datasets. Datasets may use separate encoder and decoder groups, or compatible datasets may use the same group. Their encoded representations are combined in the latent space before processing.

- **Multi-domain training:** Every batch contains samples from one domain. Training alternates domain-homogeneous batches, allowing the active grid and graph to change between batches while the model parameters remain shared.

The loading policy is independent of how datasets are assigned to encoder and decoder groups. In particular, synchronized loading can also use one encoder for multiple datasets with compatible variables. The following sections describe both loading modes and their configuration requirements.

.. toctree::
   :maxdepth: 1
   :hidden:

   multi-dataset/multiple-encoder-decoder
   multi-dataset/multi-domain-training
