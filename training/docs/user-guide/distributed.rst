#################
 Parallelisation
#################

There are two types of parallelisation which users can use in
anemoi-training:

#. Data Distributed
#. Model Sharding

These can either be used individually or both at the same time.

******************
 Data-Distributed
******************

Data-parallel training (DDP) is the default and most common way to
scale Anemoi Training across multiple GPUs. Each GPU holds a *full
replica* of the model and processes a distinct shard of every mini-batch
in parallel. After the backward pass the gradients are averaged across
all replicas with a collective ``all-reduce`` before the optimiser step,
so every replica stays in sync.

Data parallelism is enabled automatically whenever the number of
data-parallel replicas

.. math::

   N_{\text{data}} = \frac{\texttt{num\_nodes} \times \texttt{num\_gpus\_per\_node}}{\texttt{num\_gpus\_per\_model}}

is greater than 1. With the default ``num_gpus_per_model=1`` this simply
means "one replica per GPU". Combining data parallelism with model
sharding is supported — see :ref:`Model Sharding <model-sharding>` below.

How to run on multiple GPUs
===========================

The relevant knobs live under ``config.system.hardware`` and
``config.dataloader.batch_size``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Config key
     - Default
     - Meaning
   * - ``system.hardware.num_gpus_per_node``
     - ``1``
     - GPUs used on each node.
   * - ``system.hardware.num_nodes``
     - ``1``
     - Number of nodes participating in the job.
   * - ``system.hardware.num_gpus_per_model``
     - ``1``
     - GPUs a single model replica is sharded across. Keep at ``1``
       for pure data parallelism.
   * - ``dataloader.batch_size.training``
     - ``2``
     - Per-replica (per-GPU when ``num_gpus_per_model=1``) batch size.

The **effective (global) batch size** used per optimiser step is:

.. math::

   B_{\text{eff}} = \texttt{batch\_size.training} \times N_{\text{data}}

Single node with multiple GPUs
------------------------------

To use all 4 GPUs on a single workstation and keep an effective batch
size of 16, override the hardware and batch-size values on the command
line:

.. code:: bash

   anemoi-training train \
       system.hardware.num_gpus_per_node=4 \
       system.hardware.num_nodes=1 \
       system.hardware.num_gpus_per_model=1 \
       dataloader.batch_size.training=4

Lightning launches one Python process per GPU under the hood, so no
``torchrun`` or ``python -m torch.distributed.launch`` wrapper is
needed — just the plain ``anemoi-training train`` entrypoint.

Multiple nodes (SLURM)
----------------------

For multi-node jobs a SLURM configuration group is shipped that reads
the number of GPUs and nodes from the environment variables SLURM sets
for each job step, so you do not have to hard-code them:

.. code:: yaml

   # config/system/hardware/slurm.yaml
   num_gpus_per_node: ${oc.decode:${oc.env:SLURM_GPUS_PER_NODE}}
   num_nodes:         ${oc.decode:${oc.env:SLURM_NNODES}}
   num_gpus_per_model: 1

A typical batch script for 2 nodes with 4 GPUs each (giving 8 data-parallel
replicas) looks like:

.. code:: bash

   #!/bin/bash
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=4       # one task per GPU
   #SBATCH --gpus-per-node=4
   #SBATCH --cpus-per-task=32        # give the dataloader workers enough cores
   #SBATCH --mem=0                   # request each node's full memory

   srun anemoi-training train system/hardware=slurm

``srun`` starts one process per GPU across all nodes; Lightning picks
these up as the DDP world and wires up the collective communication.
See the :doc:`performance-optimisation` guide for related SLURM
resource-allocation guidance.

.. note::

   ``num_gpus_per_node`` in the Anemoi config must match the number of
   GPUs the launcher actually makes available on each node
   (``--gpus-per-node`` and ``--ntasks-per-node`` under SLURM, or
   ``CUDA_VISIBLE_DEVICES`` when launching locally). A mismatch
   typically manifests as ranks hanging during process-group
   initialisation or as out-of-memory errors on the "extra" ranks.

How data parallelism works under the hood
=========================================

Anemoi Training uses the ``DDPGroupStrategy`` from
:mod:`anemoi.training.distributed.strategy`, a thin wrapper around
PyTorch Lightning's ``DDPStrategy``:

-  **One process per GPU.** Each rank owns a full copy of the model
   parameters and optimiser state.
-  **Disjoint data shards.** The training dataloader uses a distributed
   sampler that partitions the dataset so every rank sees a different
   subset of samples each epoch. Together the replicas cover the whole
   dataset exactly once per epoch.
-  **Synchronised gradients.** Immediately after ``loss.backward()``,
   DDP performs an ``all-reduce`` on the gradients so every replica
   sees the same averaged gradient before ``optimizer.step()``. The
   forward and backward passes themselves run entirely locally on each
   GPU.

Because each replica processes ``batch_size.training`` samples per
step, adding more data-parallel replicas increases the *effective*
batch size linearly. The default configs assume ``B_eff = 16``; if you
change either ``batch_size.training`` or the number of replicas the
per-step learning-rate scale changes with it. Anemoi rescales the base
learning rate as

.. math::

   \texttt{global\_lr} = \texttt{local\_lr} \times
       \frac{\texttt{num\_nodes} \times \texttt{num\_gpus\_per\_node}}
            {\texttt{num\_gpus\_per\_model}}

so if you want to keep the *global* learning rate fixed while changing
the number of GPUs, adjust ``local_lr`` accordingly. See
``config/training/single.yaml`` for the reference values.

Trade-offs and tips
===================

-  **Memory.** Every replica stores the full model, optimiser state and
   activations for its local batch. If the model does not fit on a
   single GPU, switch to (or combine with) :ref:`Model Sharding
   <model-sharding>` by setting ``num_gpus_per_model > 1``.
-  **Communication.** DDP only synchronises once per step
   (gradient all-reduce), so it scales very well across nodes as long as
   the interconnect is not saturated. Prefer data parallelism over model
   sharding whenever the model fits in a single GPU's memory.
-  **Dataloader throughput** often becomes the limiting factor before
   compute does when scaling out. Allocate enough CPU cores per task,
   tune ``dataloader.num_workers``, and consider ``read_group_size`` for
   sharded reads — see the :doc:`performance-optimisation` guide.
-  **Reproducibility.** Because ordering of the ``all-reduce`` is
   deterministic per-world-size, changing the number of GPUs will
   produce numerically different (but statistically equivalent) runs
   even at the same effective batch size.

.. _model-sharding:

****************
 Model Sharding
****************

It is also possible to shard a single model across multiple GPUs.
To use model sharding, set ``config.system.hardware.num_gpus_per_model``
to the number of GPUs you wish to shard the model across. Set
``config.model.keep_batch_sharded=True`` to also keep batches fully
sharded throughout training and dataloading, reducing memory usage for
large inputs or long rollouts.

Note that model sharding comes with communication overhead, so it is
recommended to first maximise data parallelism before sharding the
model.

When using model sharding, the global grid is partitioned across GPUs:
each GPU owns a contiguous sub-region of the globe. For GNN-based layers
(including the GraphTransformer), the edges are split accordingly: each
GPU only stores and computes edges whose destination node belongs to its
partition. To allow information to flow across partition boundaries,
nodes from neighbouring partitions that are sources of cross-partition
edges are exchanged as *halo* nodes before each layer's message passing,
and discarded afterwards. This is illustrated below.

.. figure:: ../images/graph-partitioning.png
   :width: 500
   :align: center

   Graph partitioning across two GPUs. Each colour represents one
   partition (inner nodes). Halo nodes received from the neighbouring
   partition are shown in red/green along the partition boundary.

This *edge sharding* strategy is enabled by default for the GraphTransformer:

.. code:: yaml

   model:
     encoder:
       shard_strategy: edges
     processor:
       shard_strategy: edges
     decoder:
       shard_strategy: edges

Dense attention layers without a sparse graph (e.g. the sliding-window
processor) instead shard the attention heads across GPUs, as shown in
the figure below. This requires expensive all-to-all communication
as opposed to the more local point-to-point communication used in edge sharding.
Head sharding can also be selected for the GraphTransformer:

.. code:: yaml

   model:
     encoder:
       shard_strategy: heads
     processor:
       shard_strategy: heads
     decoder:
       shard_strategy: heads

This may be beneficial when the graph is very densely connected, but
note that head sharding requires an all-to-all communication at every
layer and quickly becomes the communication bottleneck. It is therefore
most suitable for layers where attention is already close to dense (e.g.
sliding-window attention) rather than sparse GNN message passing.

.. figure:: ../images/transformer-head-sharding.png
   :width: 500
   :align: center

   Head sharding (source: `Jacobs et al. (2023) <https://arxiv.org/pdf/2309.14509>`_)

Anemoi Training provides different sharding strategies depending if the
model task is deterministic or ensemble based.

For deterministic models, the ``DDPGroupStrategy`` is used:

.. code:: yaml

   strategy:
      _target_: anemoi.training.distributed.strategy.DDPGroupStrategy
      num_gpus_per_model: ${system.hardware.num_gpus_per_model}
      read_group_size: ${dataloader.read_group_size}

When using model sharding, ``config.dataloader.read_group_size`` allows
for sharded data loading in subgroups. This should be set to the number
of GPUs per model for optimal performance.

For ensemble models, the ``DDPEnsGroupStrategy`` is used which in
addition to sharding the model also distributes the ensemble members
across GPUs:

.. code:: yaml

   strategy:
     _target_: anemoi.training.distributed.strategy.DDPEnsGroupStrategy
     num_gpus_per_model: ${system.hardware.num_gpus_per_model}
     read_group_size: ${dataloader.read_group_size}

This requires setting ``config.system.hardware.num_gpus_per_ensemble``
to the number of GPUs you wish to parallelise the ensemble members
across and ``config.training.ensemble_size_per_device`` to the number of
ensemble members per GPU.

**********
 Examples
**********

Suppose the job is running on 8 nodes each with 4 GPUs and that
``config.system.hardware.num_gpus_per_model=2`` and
``config.dataloader.batch_size.training=1``. Then each model will be
sharded across 2 GPUs and the data sharded across ``total number of
GPUs/num_gpus_per_model=32/2=16``. This means the effective batch size
is 16.

Alternatively, with no model sharding on a single node with 4 GPUs, setting
``config.system.hardware.num_gpus_per_model=1`` and
``config.dataloader.batch_size.training=4`` gives 4-way data
parallelism on 4 GPUs, again yielding an effective batch size of 16.
