.. _usage-create-model:

#########################
 Create your first model
#########################

This section describes how to create an existing model from the
``anemoi-models`` package.

In this example we show how to create an instance of the
Encoder-Processor-Decoder that uses a Graph Transformer for the encoder
and decoder and a sliding window transformer [#f1]_ for the processor.

Our implemented models are instantiated by omegaconf [#f2]_ and hydra
[#f3]_. Commonly used model configurations can be found in
``configs/models`` (see :doc:`anemoi-training:user-guide/hydra-intro`).

*********************
 Model Configuration
*********************

First, let's take the model configuration ``transformer.yaml``:

.. code:: yaml

   model:
     _target_: anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec

   num_channels: 1024

   processor:
     _target_: anemoi.models.layers.processor.TransformerProcessor
     num_layers: 16
     num_chunks: 2

   encoder:
     _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
     trainable_size: 8
     sub_graph_edge_attributes: ${model.attributes.edges}
     num_chunks: 1
     mlp_hidden_ratio: 4
     num_heads: 16

   decoder:
     _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
     trainable_size: 8
     sub_graph_edge_attributes: ${model.attributes.edges}
     num_chunks: 1
     mlp_hidden_ratio: 4
     num_heads: 16

   attributes:
     edges:
     - edge_length
     - edge_dirs
     nodes: []

Typically the model is instantiated in :doc:`Anemoi Training
<anemoi-training:index>` or :doc:`Anemoi Inference
<anemoi-inference:index>`. For this example we will load the model
configuration by itself to understand the different components needed to
create a model.

.. code:: python

   from omegaconf import OmegaConf

   model_config = OmegaConf.load("transformer.yaml")

*******************************************************
 Define statistics, data indices and supporting arrays
*******************************************************

As described in :ref:`overview`, we want to create a model interface
that can be used for training and inference. For that we need to create
the statistics, data indices and supporting arrays which is required for
the pre- and postprocessing. These attributes are provided by the
:doc:`anemoi-datasets:index`.

Statistics
==========

The **statistics** are simply stored in a dictionary with the mean,
stdev, maximum and minimum of the variables. They are usually loaded
from the dataset, i.e. ``ds.statistics``:

.. code:: python

   statistics = {
       "mean": [0.5, 1.1, 0.0],
       "stdev": [0.1, 0.1, 0.1],
       "maximum": [1.0, 1.0, 1.0],
       "minimum": [0.0, 0.0, 0.0],
   }

Data Indices
============

**Data indices** is a dictionary with the forcing and diagnostic
variables. They are usually created from the dataset, i.e.
``ds.name_to_index``:

.. code:: python

   from anemoi.models.data_indices.collection import IndexCollection

   name_to_index = {"10u": 0, "10v": 1, "2d": 2, "2t": 3}

   # This part is usually defined in the config/data/zarr.yaml file.
   data_config = dict(
       data={
           "forcing": ["cos_latitude"],
           "diagnostics": ["tp", "cp"],
           "remapper": [],
       }
   )
   data_indices = IndexCollection(data_config, name_to_index)

Supporting Arrays
=================

**Supporting arrays** is a dictionary with the latitudes and longitudes
of the grid and naturally comes from the dataset, i.e.
``ds.supporting_arrays``.

.. code:: python

   supporting_arrays = {
       "latitudes": [90.0, 89.0, 88.0],
       "longitudes": [0.0, 1.0, 2.0]
   }

********************
 Creating the Graph
********************

All our currently implemented models are based on a graph encoder and
decoder. The graph is created by the ``GraphCreator`` class which is
part of :doc:`Anemoi Graphs <anemoi-graphs:index>`.

.. code:: python

   from anemoi.models.graphs.create import GraphCreator

   graph_config = OmegaConf.load("graph.yaml")
   graph_data = GraphCreator(config=graph_config).create()

************************
 Initializing the Model
************************

Now that we have all the pieces needed to create the model, we can call
the ``AnemoiModelInterface`` class.

.. code:: python

   from anemoi.models.interface import AnemoiModelInterface

   model_interface = AnemoiModelInterface(
       statistics=statistics,
       data_indices=data_indices,
       supporting_arrays=supporting_arrays,
       graph_data=graph_data,
       config=model_config,
   )

The model interface includes the preprocessor, postprocessor and the
actual model (see :ref:`overview`).

.. code:: python

   model_interface.preprocessor
   model_interface.postprocessor
   model_interface.model

.. note::

   During training the forward pass is done by the
   ``model_interface.forward`` method while during inference the
   ``model_interface.predict_step``. Their difference is that the
   forward function assumes an already normalized state and predicts the
   normalized state while the predict_step performs the pre- and
   post-processing in addition to the forward step.

   -  ``y_norm = model_interface.forward(x_norm)`` with ``x_in`` and
      ``y_pred`` are normalized.
   -  ``y = model_interface.predict_step(x)`` with ``x`` and ``y`` are
      absolute values.

*******************
 The PyTorch Model
*******************

The model architecture is in ``model_interface.model`` which is a
``pytorch.nn.Module``. The model therefore has a ``forward()`` function
and inherits all the important features for training.

In this example, ``model_interface.model`` is the following:

.. code:: python

   AnemoiModelEncProcDec(
     (encoder): GraphTransformerForwardMapper(
       (trainable): TrainableTensor()
       (proc): GraphTransformerMapperBlock(
         (lin_key): Linear(in_features=1024, out_features=1024, bias=True)
         ...
       )
     )
     (processor): TransformerProcessor(
       ...
     )
     (decoder): GraphTransformerBackwardMapper(
       (proc): GraphTransformerMapperBlock(
         (lin_key): Linear(in_features=1024, out_features=1024, bias=True)
         ...
     )
   )

.. _layer-kernels:

**************************************
 Layer Kernels - Switching out Layers
**************************************

The model interface allows switching out layers in the model. For
example, if you want to use a different activation function, you can
simply change the activation function in the model configuration. Anemoi
will automatically train the model with the new activation function.

This functionality is optional and can be used to test different layers
and architectures. The model interface will automatically create the new
model with the new layer. For example, if you want to use the ``Sine``
activation function instead of the ``GELU`` activation function, you can
simply change the activation function in a model component, like in the
processor below:

.. code:: yaml

   processor:
     _target_: anemoi.models.layers.processor.TransformerProcessor
     num_layers: 16
     num_chunks: 2
     layer_kernels:
       Activation:
         _target_: anemoi.models.layers.activation.GLU

Available Layer Kernels
=======================

This is entirely optional and uses sensible defaults for each layer.
Currently, you can switch out the following layers (with a given key):

-  **Activation function** (``Activation``): Default ``torch.nn.GELU``
-  **Linear layers** (``Linear``): Default ``torch.nn.Linear``
-  **Layer Normalisation** (``LayerNorm``): Default
   ``torch.nn.LayerNorm``
-  **Query Normalisation** (``QueryNorm``): Default
   ``anemoi.models.layers.normalization.AutocastLayerNorm``
-  **Key Normalisation** (``KeyNorm``): Default
   ``anemoi.models.layers.normalization.AutocastLayerNorm``

These layers can technically accept any type of PyTorch ``nn.Module``
that implements a forward pass. The default layers are chosen to be
compatible with the model architecture and the training process.

Suitable Alternatives
=====================

Examples for suitable alternatives within Anemoi are:

**Normalisation Layers** (see :doc:`modules/normalization`):

-  ``anemoi.models.layers.normalization.AutocastLayerNorm``
-  ``anemoi.models.layers.normalization.ConditionalLayerNorm``

**Activation functions** (see :doc:`modules/activations`):

-  ``anemoi.models.layers.activation.GLU``
-  ``anemoi.models.layers.activation.SwiGLU``
-  ``anemoi.models.layers.activation.Sine``

The ``_target_`` can be any local or installed class (see Hydra
documentation [#f4]_).

When to Use Layer Kernels
=========================

Layer kernels are particularly useful when:

#. You need to use specialized implementations for efficiency
#. You want to experiment with different normalization techniques
#. You need to customize the behaviour of specific layers in different
   parts of the model

.. rubric:: Footnotes

.. [#f1]

   https://arxiv.org/abs/2004.05150v2

.. [#f2]

   https://omegaconf.readthedocs.io/en/latest/

.. [#f3]

   https://hydra-documentation.readthedocs.io/en/latest/

.. [#f4]

   https://hydra.cc/docs/advanced/instantiate_objects/overview/
