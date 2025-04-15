##########
 Strategy
##########

.. _strategy target:

This module defines the strategy for parallelising the model training
across GPUs. It also seeds the random number generators for the rank.
The strategy used is a Distributed Data Parallel strategy with group
communication. This strategy implements data parallelism at the module
level which can also run on multiple GPUs, and is a standard strategy
within PyTorch `DDP Strategy
<https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__.

.. note::

   Generally you should not need to change this module, as it is
   independent of the system being used for training.

Anemoi Training provides different sharding strategies for the
deterministic or ensemble based model tasks.

For deterministic models, the ``DDPGroupStrategy`` is used while for
ensemble models, the ``DDPEnsGroupStrategy`` is used which in addition
to sharding the model also distributes the ensemble members across GPUs.

******************
 DDPGroupStrategy
******************

.. autoclass:: anemoi.training.distributed.strategy.DDPGroupStrategy
   :members:
   :no-undoc-members:
   :show-inheritance:

*********************
 DDPEnsGroupStrategy
*********************

.. autoclass:: anemoi.training.distributed.strategy.DDPEnsGroupStrategy
   :members:
   :no-undoc-members:
   :show-inheritance:
