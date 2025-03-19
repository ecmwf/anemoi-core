#########################
 Create your first model
#########################

This section describes how to create an existing model from the
``anemoi-models`` package.

In this example we show how to create an instance of the
Encoder-Processor-Decoder that uses a Graphtransformer for the encoder
and decoder and a sliding window transformer [#f1]_ for the processor.

Our implemented models are instantiated by hydra [#f2]_. Commonly used
model configurations can be found in ``configs/models`` (see generating
configs).

*****************************
 Loading Model Configuration
*****************************

First, let's take a model configuration:

.. code:: yaml

   model:
     _target_: anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec

   layer_kernels:
       LayerNorm:
         _target_: torch.nn.LayerNorm
         _partial_: True
       Linear:
         _target_: torch.nn.Linear
         _partial_: True

   num_channels: 1024

   processor:
     _target_: anemoi.models.layers.processor.TransformerProcessor
     activation: GELU
     num_layers: 16
     num_chunks: 2

   encoder:
     _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
     trainable_size: 8
     sub_graph_edge_attributes: ${model.attributes.edges}
     activation: GELU
     num_chunks: 1
     mlp_hidden_ratio: 4
     num_heads: 16

   decoder:
     _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
     trainable_size: 8
     sub_graph_edge_attributes: ${model.attributes.edges}
     activation: GELU
     num_chunks: 1
     mlp_hidden_ratio: 4
     num_heads: 16

   attributes:
     edges:
     - edge_length
     - edge_dirs
     nodes: []

Typically the model is instantiated in ``anemoi-training`` or
``anemoi-inference``. For this example we will load the model
configuration for itself to understand the different bits needed to
create a model.

.. code:: python

   from omegaconf import OmegaConf

   model_config = OmegaConf.load("transformer.yaml")

********************************************************
 Getting statistics, data indices and supporting arrays
********************************************************

As described in :ref:`overview` we want to create a model interface that
can be used for training and inference. For that we need to create the
statistics, data indices and supporting arrays which is required for the
pre- and postprocessing.

The **statistics** are usually loaded from the dataset, i.e
``datamodule.statistics``. The statistics is simply a dictionary with
the mean, stdev, maximum and minimum of the variables.

.. code:: python

   statistics = {
       "mean": [0.5, 1.1, 0.0],
       "stdev": [0.1, 0.1, 0.1],
       "maximum": [1.0, 1.0, 1.0],
       "minimum": [0.0, 0.0, 0.0],
   }

The **data indices** are usually loaded from the dataset, i.e
``datamodule.data_indices``. The data indices is a dictionary with the
forcing and diagnostic variables.

.. code:: python

   from anemoi.data.index import IndexCollection

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

The **supporting arrays** are usually loaded from the dataset with
``datamodule.supporting_arrays``. The ``supporting_arrays`` is a
dictionary with the latitudes and longitudes of the grid.

.. code:: python

   supporting_arrays = {"latitudes": [90.0, 89.0, 88.0], "longitudes": [0.0, 1.0, 2.0]}

********************
 Creating the Graph
********************

All our currently implemented models are based on a graph encoder and
decoder. The graph is created by the ``GraphCreator`` class, see
:ref:`anemoi-graphs`.

.. code:: python

   from anemoi.graphs.create import GraphCreator

   graph_config = OmegaConf.load("graph.yaml")
   graph_data = GraphCreator(config=graph_config).create()

**********************************
 Initializing the Model Interface
**********************************

Now that we all the pieces needed to create the model interface, we can
initialize the model interface.

.. code:: python

   from anemoi.models.interface import AnemoiModelInterface

   model_interface = AnemoiModelInterface(
       statistics=statistics,
       data_indices=data_indices,
       supporting_arrays=supporting_arrays,
       graph_data=graph_data,
       config=model_config,
   )

The model interface includes all parts needed for training and
inference, i.e. preprocessor, postprocessor and model.

.. code:: python

   model_interface.preprocessor
   model_interface.postprocessor
   model_interface.model

.. note::

   During training the forward pass is done by the ``model.forward``
   method while during inference the ``model.predict_step``. Their
   difference is that the forward function assumes an already normalized
   state and predicts the normalized state while the predict_step
   performs the pre- and post-processing in addition to the forward
   step.

   -  ``y_norm = model_interface.forward(x_norm)`` with ``x_in`` and
      ``y_pred`` are normalized
   -  ``y = model_interface.predict_step(x)`` with ``x`` and ``y`` are
      in physical space

********************
 Model Architecture
********************

The actual model architecture is in ``model_interface.model`` which is a
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

.. rubric:: Footnotes

.. [#f1]

   https://arxiv.org/abs/2004.05150v2

.. [#f2]

   https://hydra-documentation.readthedocs.io/en/latest/
