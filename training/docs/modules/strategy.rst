##########
 Strategy
##########

.. _strategy target:

This module defines the strategy for parallelising model training
across GPUs. It also seeds the random number generators for each rank
to control stochastic parts of a run. This improves repeatability, but
does not guarantee exact reproducibility because floating-point numerics
and distributed reductions can vary across environments. The implementation builds on the PyTorch Lightning
`DDP Strategy <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html>`__
but layers several communication groups on top of vanilla DDP so that
models, readers, and ensemble members can coordinate work explicitly.

.. note::

   Generally you should not need to change this module, as it is
   independent of the system being used for training.

Anemoi Training provides different sharding strategies for the
deterministic or ensemble-based model tasks.

-------------------------
 Base strategy template
-------------------------

The :class:`BaseDDPStrategy` bundles the common logic shared by all
strategies:

* initializes the different process group layouts (model, reader, ensemble)
* configures DDP and injects per-parameter gradient scaling hooks
* exposes the ``shard_shapes`` that dataloaders need to produce correctly
   partitioned batches
* seeds ``torch``, ``numpy`` and PyTorch Lightning RNGs in a controlled way

To implement a new strategy inherit from :class:`BaseDDPStrategy` and
override two methods:

* ``_setup_communication_groups``: define how the ranks are split across
   model, reader, and optional ensemble groups. The method must return the
   model communication group ID for the current rank. Most strategies use
   the helpers in :mod:`anemoi.training.distributed.groups` to stay
   consistent with the existing layouts.
* ``process_dataloader``: forward any group metadata that the underlying
   dataset requires. The default implementation already calls the parent
   DDP logic, so derived strategies only need to pass along additional
   information (for example the ensemble group IDs).

Understanding communication groups
==================================

``BaseDDPStrategy`` composes three complementary layouts:

* **Model communication group**: shards a single model across
   ``num_gpus_per_model`` ranks. Parameters that do not receive a full
   batch on each rank are rescaled during gradient reduction to preserve
   balanced updates.
* **Reader layout and groups**: within each model group, ranks are
   subdivided into reader groups of size ``read_group_size``. The
   :class:`~anemoi.training.distributed.groups.ReaderLayout` decides which
   rank loads which shard, and the dataset receives that information via
   ``set_comm_group_info`` when ``process_dataloader`` runs.
* **Ensemble groups** (optional): needed only when training ensemble
   models. ``_setup_communication_groups`` builds a second hierarchy that
   spreads ensemble members across GPUs and exposes both the coarse group
   (all ranks holding different members) and subgroups (used for member
   specific reductions).

Because the layouts are computed centrally inside the strategy, models
and dataloaders receive a consistent view of the world even when the
hardware topology changes between runs.

For deterministic models, the ``DDPGroupStrategy`` is used while for
ensemble models, the ``DDPEnsGroupStrategy`` is used which in addition
to sharding the model also distributes the ensemble members across GPUs.

******************
 DDPGroupStrategy
******************

``DDPGroupStrategy`` is the default choice for deterministic models.
It extends :class:`BaseDDPStrategy` by:

* building model and reader layouts during ``_setup_communication_groups``
* wiring the resulting communication groups into the model via
   ``set_model_comm_group`` and ``set_reader_groups``
* passing the reader layout information to datasets via
   ``process_dataloader`` so each rank knows which shard to load and how
   to size data windows using ``shard_shapes``

This strategy is best suited when every rank should work on the same
model parameters but potentially different spatial shards of the input
data.

.. autoclass:: anemoi.training.distributed.strategy.DDPGroupStrategy
   :members:
   :no-undoc-members:
   :show-inheritance:

*********************
 DDPEnsGroupStrategy
*********************

``DDPEnsGroupStrategy`` starts from ``DDPGroupStrategy`` and adds a
second set of communication groups so that ensemble members can be
distributed across GPUs. The strategy keeps three invariants in place:

* Each model shard still runs inside a model communication group, so the
   encoder/processor/decoder stack remains data parallel.
* Each reader group continues to coordinate file access, ensuring that
   ensemble training does not multiply I/O contention.
* Ensemble groups and subgroups expose APIs such as
   ``set_ens_comm_group`` and ``set_ens_comm_subgroup`` that models use
   to collect statistics across ensemble members (for example mean and
   spread) without interfering with the gradient exchange streams.

Use this strategy whenever the task configuration defines
``num_gpus_per_ensemble`` greater than one or when ensemble-specific
metrics must be aggregated online.

.. autoclass:: anemoi.training.distributed.strategy.DDPEnsGroupStrategy
   :members:
   :no-undoc-members:
   :show-inheritance:
