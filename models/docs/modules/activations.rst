#############
 Activations
#############

The activations module provides custom activation layers used throughout
the network.

*******
 Usage
*******

These activation layers can be used through the layer kernels configuration
system. For example, to use the ``Sine`` activation function:

.. code:: yaml

   layer_kernels:
     Activation:
       _target_: anemoi.models.layers.activations.Sine

For gated variants (GLU, SwiGLU, GEGLU, ReGLU), use ``mlp_implementation``
instead of ``layer_kernels.Activation``:

.. code:: yaml

   processor:
     mlp_implementation: swiglu  # options: glu, swiglu, geglu, reglu
     mlp_hidden_ratio: 2.67  # recommended ratio for gated variants

******************
 Available Layers
******************

.. automodule:: anemoi.models.layers.activations
   :members:
   :no-undoc-members:
   :show-inheritance:
