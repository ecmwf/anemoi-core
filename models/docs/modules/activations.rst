#############
 Activations
#############

The activations module provides custom activation layers used throughout
the network.

*******
 Usage
*******

These activation layers can be used in two ways:

#. Directly in model implementations
#. Through the layer kernels configuration system

Example configuration using layer kernels:

.. code:: yaml

   layer_kernels:
     processor:
       Activation:
         _target_: anemoi.models.layers.activations.GLU
         dim: 1024

******************
 Available Layers
******************

.. automodule:: anemoi.models.layers.activations
   :members:
   :no-undoc-members:
   :show-inheritance:
