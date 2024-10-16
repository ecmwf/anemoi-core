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

Generally you should not need to change this module, as it is
independent of the computer being used for training.
