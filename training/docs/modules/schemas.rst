#########
 Schemas
#########

This module defines pydantic schemas, which are used to validate the
configuration before a training run is started. The top-level config
yaml matches the BaseSchema.

.. automodule:: anemoi.training.schemas.base_schema
   :members:
   :no-undoc-members:
   :show-inheritance:

The below schemas are organised below identically to the training config
files,

******
 Data
******

.. automodule:: anemoi.training.schemas.data
   :members:
   :no-undoc-members:
   :show-inheritance:

************
 Dataloader
************

.. automodule:: anemoi.training.schemas.dataloader
   :members:
   :no-undoc-members:
   :show-inheritance:

**********
 Hardware
**********

.. automodule:: anemoi.training.schemas.hardware
   :members:
   :no-undoc-members:
   :show-inheritance:

*******
 Graph
*******

.. automodule:: anemoi.training.schemas.graphs.basegraph
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.schemas.graphs.node_schemas
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.schemas.graphs.edge_schemas
   :members:
   :no-undoc-members:
   :show-inheritance:

*******
 Model
*******

.. automodule:: anemoi.training.schemas.models.models
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.schemas.models.processor
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.schemas.models.encoder
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.schemas.models.decoder
   :members:
   :no-undoc-members:
   :show-inheritance:

**********
 Training
**********

.. automodule:: anemoi.training.schemas.training
   :members:
   :no-undoc-members:
   :show-inheritance:
