Training with Multiple Datasets
===============================

Since ``anemoi-training`` version 0.4, the framework supports training with multiple datasets within a single training workflow. This enables applications such as downscaling, combining global and regional datasets, and leveraging diverse data sources to improve model generalisation.

There are two ways of training a model with multiple datasets, selected by the data loading policy (``dataloader.loading_policy``):

- **Synchronized loading (default):** every sample contains data from all configured datasets, so the model handles the same datasets at every iteration. Each dataset has its own encoder and decoder (datasets with the same variables can also share one), the encoded representations are combined in a shared latent space, and a common processor operates on it. This supports datasets with different variables and preprocessing requirements. See :doc:`multi-dataset/multiple-encoder-decoder`.

- **Multi-domain loading:** each batch is drawn from a single dataset, a 'domain', i.e. the spatial structure of a dataset (its grid, resolution and extent). The same model is therefore trained on different domains in different iterations, across different spatial resolutions, grid structures, and geographical extents. All domains must share the same variables, with a consistent variable ordering. See :doc:`multi-dataset/multi-domain-training`.

.. toctree::
   :maxdepth: 1
   :hidden:

   multi-dataset/multiple-encoder-decoder
   multi-dataset/multi-domain-training
