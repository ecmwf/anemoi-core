########
 Layers
########

***********************
 Environment Variables
***********************

``ANEMOI_INFERENCE_NUM_CHUNKS``
===============================

This environment variable controls the number of chunks used in the
`Mapper` and `Processor` during inference. Setting this variable allows
the model to split large computations into a specified number of smaller
chunks, reducing memory overhead. If not set, it falls back to the
default value of, 1 i.e. no chunking. See pull request `#46
<https://github.com/ecmwf/anemoi-models/pull/46>`_. For finer control,
set the following environment variables:

``ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER``
======================================

This environment variable controls the number of chunks used in the
`Mapper` during inference.

``ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR``
=========================================

This environment variable controls the number of chunks used in the
`Processor` during inference.

*********
 Mappers
*********

.. automodule:: anemoi.models.layers.mapper
   :members:
   :no-undoc-members:
   :show-inheritance:

************
 Processors
************

.. automodule:: anemoi.models.layers.processor
   :members:
   :no-undoc-members:
   :show-inheritance:

********
 Chunks
********

.. automodule:: anemoi.models.layers.chunk
   :members:
   :no-undoc-members:
   :show-inheritance:

********
 Blocks
********

.. automodule:: anemoi.models.layers.block
   :members:
   :no-undoc-members:
   :show-inheritance:

*******
 Graph
*******

.. automodule:: anemoi.models.layers.graph
   :members:
   :no-undoc-members:
   :show-inheritance:

******
 Conv
******

.. automodule:: anemoi.models.layers.conv
   :members:
   :no-undoc-members:
   :show-inheritance:

***********
 Attention
***********

.. automodule:: anemoi.models.layers.attention
   :members:
   :no-undoc-members:
   :show-inheritance:

************************
 Multi-Layer Perceptron
************************

.. automodule:: anemoi.models.layers.mlp
   :members:
   :no-undoc-members:
   :show-inheritance:

*******
 Utils
*******

.. automodule:: anemoi.models.layers.utils
   :members:
   :no-undoc-members:
   :show-inheritance:
