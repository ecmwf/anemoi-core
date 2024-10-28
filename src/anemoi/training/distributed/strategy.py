# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn

from anemoi.training.utils.seeding import get_base_seed

LOGGER = logging.getLogger(__name__)


class DDPGroupStrategy(DDPStrategy):
    """Distributed Data Parallel strategy with group communication."""

    def __init__(self, num_gpus_per_model: int, **kwargs: dict) -> None:
        """Initialize the distributed strategy.

        Parameters
        ----------
        num_gpus_per_model : int
            Number of GPUs per model to shard over.
        **kwargs : dict
            Additional keyword arguments.

        """
        super().__init__(**kwargs)
        self.model_comm_group_size = num_gpus_per_model

    def setup(self, trainer: pl.Trainer) -> None:
        assert self.accelerator is not None, "Accelerator is not initialized for distributed strategy"
        self.accelerator.setup(trainer)

        # determine the model groups that work together:

        assert self.world_size % self.model_comm_group_size == 0, (
            f"Total number of GPUs ({self.world_size}) must be divisible by the number of GPUs "
            f"per model ({self.model_comm_group_size})."
        )

        model_comm_group_ranks = np.split(
            np.arange(self.world_size, dtype=int),
            int(self.world_size / self.model_comm_group_size),
        )
        model_comm_groups = [
            torch.distributed.new_group(x) for x in model_comm_group_ranks
        ]  # every rank has to create all of these

        model_comm_group_id, model_comm_group_nr, model_comm_group_rank = self.get_my_model_comm_group(
            self.model_comm_group_size,
        )
        model_comm_group = model_comm_groups[model_comm_group_id]
        self.model.set_model_comm_group(model_comm_group)
        LOGGER.debug(
            "Rank %d model_comm_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
            self.global_rank,
            str(model_comm_group_nr),
            model_comm_group_id,
            model_comm_group_rank,
            str(model_comm_group_ranks[model_comm_group_id]),
        )

        # register hooks for correct gradient reduction
        self.register_parameter_hooks()

        # move the model to the correct device
        self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None, "Model is not initialized for distributed strategy"
            self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            # do not wrap with DDP if not fitting as there's no gradients to reduce
            self.configure_ddp()

            # set up optimizers after the wrapped module has been moved to the device
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()
        else:
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            assert self.model is not None, "Model is not initialized for distributed strategy"
            _sync_module_states(self.model)

        # seed ranks
        self.seed_rnd(model_comm_group_id)

    def get_my_model_comm_group(self, num_gpus_per_model: int) -> tuple[int, np.ndarray, int]:
        """Determine tasks that work together and from a model group.

        Parameters
        ----------
        num_gpus_per_model : int
            Number of GPUs per model to shard over.

        Returns
        -------
        tuple[int, np.ndarray, int]
            Model_comm_group id, Model_comm_group Nr, Model_comm_group rank
        """
        model_comm_groups = np.arange(0, self.world_size, dtype=np.int32)
        model_comm_groups = np.split(model_comm_groups, self.world_size / num_gpus_per_model)

        model_comm_group_id = None
        for i, model_comm_group in enumerate(model_comm_groups):
            if self.global_rank in model_comm_group:
                model_comm_group_id = i
                model_comm_group_nr = model_comm_group
                model_comm_group_rank = np.ravel(np.asarray(model_comm_group == self.global_rank).nonzero())[0]
        return model_comm_group_id, model_comm_group_nr, model_comm_group_rank

    def seed_rnd(self, model_comm_group_id: int) -> None:
        """Seed the random number generators for the rank."""
        base_seed = get_base_seed()
        initial_seed = base_seed * (model_comm_group_id + 1)
        rnd_seed = pl.seed_everything(initial_seed)  # note: workers are seeded independently in dataloader
        np_rng = np.random.default_rng(rnd_seed)
        sanity_rnd = (torch.rand(1), np_rng.random())
        LOGGER.debug(
            (
                "Strategy: Rank %d, model comm group id %d, base seed %d, seeded with %d, "
                "running with random seed: %d, sanity rnd: %s"
            ),
            int(os.environ.get("SLURM_PROCID", "0")),
            model_comm_group_id,
            base_seed,
            initial_seed,
            rnd_seed,
            sanity_rnd,
        )

    def register_parameter_hooks(self) -> None:
        """Register parameter hooks for gradient reduction.

        Here, we rescale parameters that only see a subset of the input on each rank
        -> these are still divided by the total number of GPUs in DDP as if each rank would see a full set of inputs
        note: the trainable parameters are added before the split across GPUs and are therefore not rescaled.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad is True and "trainable" not in name:
                param.register_hook(lambda grad: grad * float(self.model_comm_group_size))
