###############
 Normalization
###############

The normalization module provides custom normalization layers used
throughout the network. These layers are designed to be flexible and
efficient, supporting various normalization strategies.

*******
 Usage
*******

These normalization layers can be used in two ways:

#. Directly in model implementations
#. Through the layer kernels configuration system

Example configuration using layer kernels:

.. code:: yaml

   layer_kernels:
     processor:
       LayerNorm:
         _target_: anemoi.models.layers.normalization.AutocastLayerNorm
         bias: False

The normalization layers are particularly useful when:

#. Working with mixed precision training
#. Implementing ensemble models with noise injection
#. Requiring specialized normalization behavior in specific parts of the
   model

******************
 Available Layers
******************

.. automodule:: anemoi.models.layers.normalization
   :members:
   :no-undoc-members:
   :show-inheritance:
