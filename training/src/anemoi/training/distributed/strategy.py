# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn

from anemoi.training.utils.seeding import get_base_seed

LOGGER = logging.getLogger(__name__)


def register_gradient_scaling_hooks(
    model: torch.nn.Module,
    model_comm_group_size: float,
    skip_grad_scaling: list[str] | None = None,
) -> None:
    """Register parameter hooks for gradient reduction.

    Here, we rescale parameters that only see a subset of the input on each rank
    -> these are still divided by the total number of GPUs in DDP as if each rank would see a full set of inputs
    note: the trainable parameters are added before the split across GPUs and are therefore not rescaled.

    Parameters
    ----------
    model : torch.nn.Module
        The model to register hooks on.
    model_comm_group_size : float
        The size of the model communication group for scaling.
    skip_grad_scaling : list[str] | None
        List of parameter name patterns to skip gradient scaling.
        Defaults to ["trainable", "no_gradscaling"].
    """
    if skip_grad_scaling is None:
        skip_grad_scaling = ["trainable", "no_gradscaling"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(skip_name in name for skip_name in skip_grad_scaling):
            continue
        param.register_hook(lambda grad: grad * float(model_comm_group_size))


def seed_rnd(model_comm_group_id: int, global_rank: int) -> None:
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
        global_rank,
        model_comm_group_id,
        base_seed,
        initial_seed,
        rnd_seed,
        sanity_rnd,
    )


def get_my_model_comm_group(num_gpus_per_model: int, global_rank: int, world_size: int) -> tuple[int, int, int]:
    """Determine tasks that work together and from a model group.

    Parameters
    ----------
    num_gpus_per_model : int
        Number of GPUs per model to shard over.

    Returns
    -------
    tuple[int, int, int]
        Model_comm_group id, Model_comm_group rank, Number of model_comm_groups
    """
    model_comm_group_id = global_rank // num_gpus_per_model
    model_comm_group_rank = global_rank % num_gpus_per_model
    model_comm_num_groups = world_size // num_gpus_per_model

    return model_comm_group_id, model_comm_group_rank, model_comm_num_groups


def get_my_reader_group(model_comm_group_rank: int, read_group_size: int, global_rank: int) -> tuple[int, int, int]:
    """Determine tasks that work together and from a reader group.

    Parameters
    ----------
    model_comm_group_rank : int
        Rank within the model communication group.
    read_group_size : int
        Number of dataloader readers per model group.

    Returns
    -------
    tuple[int, int, int]
        Reader_group id, Reader_group rank, Reader_group root (global rank)
    """
    reader_group_id = model_comm_group_rank // read_group_size
    reader_group_rank = model_comm_group_rank % read_group_size
    reader_group_size = read_group_size
    reader_group_root = (global_rank // read_group_size) * read_group_size

    return reader_group_id, reader_group_rank, reader_group_size, reader_group_root


class DDPGroupStrategy(DDPStrategy):
    """Distributed Data Parallel strategy with group communication."""

    def __init__(self, num_gpus_per_model: int, read_group_size: int, **kwargs: dict) -> None:
        """Initialize the distributed strategy.

        Parameters
        ----------
        num_gpus_per_model : int
            Number of GPUs per model to shard over.
        read_group_size : int
            Number of GPUs per reader group.
        **kwargs : dict
            Additional keyword arguments.

        """
        super().__init__(**kwargs)
        self.model_comm_group_size = num_gpus_per_model
        self.read_group_size = read_group_size

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

        model_comm_group_id, model_comm_group_rank, model_comm_num_groups = get_my_model_comm_group(
            self.model_comm_group_size,
            self.global_rank,
            self.world_size,
        )
        model_comm_group = model_comm_groups[model_comm_group_id]
        self.model.set_model_comm_group(
            model_comm_group,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            self.model_comm_group_size,
        )

        # set up reader groups by further splitting model_comm_group_ranks with read_group_size:

        assert self.model_comm_group_size % self.read_group_size == 0, (
            f"Number of GPUs per model ({self.model_comm_group_size}) must be divisible by read_group_size "
            f"({self.read_group_size})."
        )

        reader_group_ranks = np.array(
            [
                np.split(group_ranks, int(self.model_comm_group_size / self.read_group_size))
                for group_ranks in model_comm_group_ranks
            ],
        )  # Shape: (num_model_comm_groups, model_comm_grp_size/read_group_size, read_group_size)
        reader_groups = [[torch.distributed.new_group(x) for x in group_ranks] for group_ranks in reader_group_ranks]
        reader_group_id, reader_group_rank, reader_group_size, reader_group_root = get_my_reader_group(
            model_comm_group_rank,
            self.read_group_size,
            self.global_rank,
        )
        # get all reader groups of the current model group
        model_reader_groups = reader_groups[model_comm_group_id]
        self.model.set_reader_groups(
            model_reader_groups,
            reader_group_id,
            reader_group_rank,
            reader_group_size,
        )

        LOGGER.debug(
            "Rank %d model_comm_group_id: %d model_comm_group: %s model_comm_group_rank: %d "
            "reader_group_id: %d reader_group: %s reader_group_rank: %d reader_group_root (global): %d",
            self.global_rank,
            model_comm_group_id,
            str(model_comm_group_ranks[model_comm_group_id]),
            model_comm_group_rank,
            reader_group_id,
            reader_group_ranks[model_comm_group_id, reader_group_id],
            reader_group_rank,
            reader_group_root,
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
        seed_rnd(model_comm_group_id, self.global_rank)

    def process_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Pass communication group information to the dataloader for distributed training.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader to process.

        Returns
        -------
        torch.utils.data.DataLoader
            Processed dataloader.

        """
        dataloader = super().process_dataloader(dataloader)

        # pass model and reader group information to the dataloaders dataset
        model_comm_group_id, model_comm_group_rank, model_comm_num_groups = get_my_model_comm_group(
            self.model_comm_group_size,
            self.global_rank,
            self.world_size,
        )
        _, reader_group_rank, _, _ = get_my_reader_group(
            model_comm_group_rank,
            self.read_group_size,
            self.global_rank,
        )

        dataloader.dataset.set_comm_group_info(
            self.global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            self.read_group_size,
        )

        return dataloader

    def register_parameter_hooks(self) -> None:
        """Register parameter hooks for gradient reduction."""
        register_gradient_scaling_hooks(self.model, self.model_comm_group_size)


class DDPEnsGroupStrategy(DDPStrategy):
    """Distributed Data Parallel strategy with group communication for ensembles."""

    def __init__(self, num_gpus_per_model: int, num_gpus_per_ensemble: int, read_group_size: int, **kwargs) -> None:
        """Initialize the distributed strategy.

        Parameters
        ----------
        num_gpus_per_model : int
            Number of GPUs per model to shard over.
        read_group_size : int
            Number of GPUs per reader group.
        **kwargs : dict
            Additional keyword arguments.

        """
        super().__init__(**kwargs)
        self.model_comm_group_size = num_gpus_per_model
        self.read_group_size = read_group_size
        self.ens_comm_group_size = num_gpus_per_ensemble

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

        model_comm_group_id, model_comm_group_rank, model_comm_num_groups = get_my_model_comm_group(
            self.model_comm_group_size,
            self.global_rank,
            self.world_size,
        )
        model_comm_group = model_comm_groups[model_comm_group_id]
        self.model.set_model_comm_group(
            model_comm_group,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            self.model_comm_group_size,
        )

        # set up reader groups by further splitting model_comm_group_ranks with read_group_size:

        assert self.model_comm_group_size % self.read_group_size == 0, (
            f"Number of GPUs per model ({self.model_comm_group_size}) must be divisible by read_group_size "
            f"({self.read_group_size})."
        )

        reader_group_ranks = np.array(
            [
                np.split(group_ranks, int(self.model_comm_group_size / self.read_group_size))
                for group_ranks in model_comm_group_ranks
            ],
        )  # Shape: (num_model_comm_groups, model_comm_grp_size/read_group_size, read_group_size)
        reader_groups = [[torch.distributed.new_group(x) for x in group_ranks] for group_ranks in reader_group_ranks]
        reader_group_id, reader_group_rank, reader_group_size, reader_group_root = get_my_reader_group(
            model_comm_group_rank,
            self.read_group_size,
            self.global_rank,
        )
        # get all reader groups of the current model group
        model_reader_groups = reader_groups[model_comm_group_id]
        self.model.set_reader_groups(
            model_reader_groups,
            reader_group_id,
            reader_group_rank,
            reader_group_size,
        )

        LOGGER.info(
            "Rank %d model_comm_group_id: %d model_comm_group: %s "
            "model_comm_group_rank: %d model_comm_group.size(): %d "
            "reader_group_id: %d reader_group: %s reader_group_rank: %d "
            "reader_group_root (global): %d "
            "model_reader_groups: %s reader_groups: %s",
            self.global_rank,
            model_comm_group_id,
            str(model_comm_group_ranks[model_comm_group_id]),
            model_comm_group_rank,
            model_comm_group.size(),
            reader_group_id,
            reader_group_ranks[model_comm_group_id, reader_group_id],
            reader_group_rank,
            reader_group_root,
            model_reader_groups,
            reader_groups,
        )

        # determine the ensemble groups that work together:
        assert self.world_size % self.ens_comm_group_size == 0, (
            f"Total number of GPUs ({self.world_size}) must be divisible by the number of GPUs "
            f"per ensemble ({self.ens_comm_group_size})."
        )
        assert self.ens_comm_group_size % self.model_comm_group_size == 0, (
            f"Number of GPUs per ensemble ({self.ens_comm_group_size}) must be divisible by the number of GPUs "
            f"per model ({self.model_comm_group_size})."
        )

        ens_comm_group_ranks = np.split(
            np.arange(self.world_size, dtype=int),
            int(self.world_size / self.ens_comm_group_size),
        )
        ens_comm_groups = [torch.distributed.new_group(x) for x in ens_comm_group_ranks]

        ens_comm_group_id, ens_comm_group_rank, ens_comm_num_groups = get_my_model_comm_group(
            self.ens_comm_group_size,
            self.global_rank,
            self.world_size,
        )

        ens_comm_group = ens_comm_groups[ens_comm_group_id]
        self.model.set_ens_comm_group(
            ens_comm_group,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.ens_comm_group_size,
        )

        # ens_comm_subgroup: subgroup of same model_comm_group ranks inside the ensemble group
        spacing = self.model_comm_group_size
        ens_comm_subgroup_ranks = [
            ens_comm_group[offset::spacing] for ens_comm_group in ens_comm_group_ranks for offset in range(spacing)
        ]

        ens_comm_subgroups = [torch.distributed.new_group(x) for x in ens_comm_subgroup_ranks]

        ens_comm_subgroup_size = self.ens_comm_group_size // self.model_comm_group_size
        ens_comm_subgroup_id = ens_comm_group_id * self.model_comm_group_size + model_comm_group_rank
        ens_comm_subgroup_rank = ens_comm_group_rank // self.model_comm_group_size
        ens_comm_num_subgroups = self.world_size // ens_comm_subgroup_size

        ens_comm_subgroup = ens_comm_subgroups[ens_comm_subgroup_id]
        self.model.set_ens_comm_subgroup(
            ens_comm_subgroup,
            ens_comm_subgroup_id,
            ens_comm_subgroup_rank,
            ens_comm_num_subgroups,
            ens_comm_subgroup_size,
        )

        LOGGER.info(
            "Rank %d ens_comm_group_id: %d ens_comm_group: %s ens_comm_group_rank: %d "
            "ens_comm_group_size: %d ens_comm_group.size(): %d ens_comm_subgroup_id: %d "
            "ens_comm_subgroup: %s ens_comm_subgroup_rank: %d ens_comm_subgroup.size(): %d ",
            self.global_rank,
            ens_comm_group_id,
            str(ens_comm_group_ranks[ens_comm_group_id]),
            ens_comm_group_rank,
            self.ens_comm_group_size,
            ens_comm_group.size(),
            ens_comm_subgroup_id,
            str(ens_comm_subgroup_ranks[ens_comm_subgroup_id]),
            ens_comm_subgroup_rank,
            ens_comm_subgroup_size,
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
        seed_rnd(model_comm_group_id, self.global_rank)

    def process_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Pass communication group information to the dataloader for distributed training.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader to process.

        Returns
        -------
        torch.utils.data.DataLoader
            Processed dataloader.

        """
        dataloader = super().process_dataloader(dataloader)

        # pass model and reader group information to the dataloaders dataset
        model_comm_group_id, model_comm_group_rank, model_comm_num_groups = get_my_model_comm_group(
            self.model_comm_group_size,
            self.global_rank,
            self.world_size,
        )
        _, reader_group_rank, _, _ = get_my_reader_group(
            model_comm_group_rank,
            self.read_group_size,
            self.global_rank,
        )
        ens_comm_group_id, ens_comm_group_rank, ens_comm_num_groups = get_my_model_comm_group(
            self.ens_comm_group_size,
            self.global_rank,
            self.world_size,
        )

        dataloader.dataset.set_comm_group_info(
            self.global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            self.read_group_size,
        )

        dataloader.dataset.set_ens_comm_group_info(
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
        )

        return dataloader

    def register_parameter_hooks(self) -> None:
        """Register parameter hooks for gradient reduction."""
        register_gradient_scaling_hooks(self.model, self.model_comm_group_size)
