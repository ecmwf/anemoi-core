Multi-Dataset Training
======================

Since ``anemoi-training`` version 0.4, the framework supports training with multiple datasets within a single training workflow. This enables applications such as downscaling, combining global and regional datasets, and leveraging diverse data sources to improve model generalisation.

Two multi-dataset architectures are supported:

- **Multiple Encoders and Decoders:** Each dataset has its own encoder and decoder, allowing dataset-specific input and output representations. The encoded representations are combined in a shared latent space and processed by a common processor. This approach supports datasets with different variables and preprocessing requirements.

- **Multi-Domain Training (Shared Architecture):** Multiple datasets with common variables (i.e. 'domains') share the same encoder, processor, and decoder. This approach enables training across different spatial resolutions, grid structures, and geographical domains while using shared model parameters. Datasets must have compatible input and output variable definitions and consistent variable ordering.

Both approaches enable learning from multiple datasets, but differ in how dataset-specific information is encoded, processed, and decoded. The following sections describe their architectures, training procedures, and configuration requirements.

.. toctree::
   :maxdepth: 1
   :hidden:

   multi-dataset/multiple-encoder-decoder
   multi-dataset/multi-domain-training
