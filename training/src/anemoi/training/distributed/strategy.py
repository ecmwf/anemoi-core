# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy

from anemoi.training.distributed.groups import build_ensemble_layout
from anemoi.training.distributed.groups import build_model_layout
from anemoi.training.distributed.groups import build_reader_layout
from anemoi.training.distributed.groups import create_ensemble_process_groups
from anemoi.training.distributed.groups import create_model_process_groups
from anemoi.training.distributed.groups import create_reader_process_groups
from anemoi.training.distributed.groups import get_my_ensemble_comm_group
from anemoi.training.distributed.groups import get_my_model_comm_group
from anemoi.training.distributed.groups import get_my_reader_group
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
    sanity_rnd = (torch.rand(1)[0], np_rng.random())
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


class BaseDDPStrategy(DDPStrategy):
    """Base DDP strategy with common functionality for group communication strategies."""

    def __init__(self, num_gpus_per_model: int, read_group_size: int, **kwargs: dict) -> None:
        """Initialise the distributed strategy.

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
        self.shard_shapes: dict | None = None

    @abstractmethod
    def _setup_communication_groups(self) -> int:
        """Set up communication groups for distributed training.

        Returns
        -------
        int
            The model communication group ID for this rank.
        """
        raise NotImplementedError

    def setup(self, trainer: pl.Trainer) -> None:
        model_comm_group_id = self._setup_communication_groups()

        super().setup(trainer)

        self.shard_shapes = self._setup_shard_shapes(trainer)
        self.shard_shapes = self._setup_shard_shapes(trainer)
        seed_rnd(model_comm_group_id, self.global_rank)

    def configure_ddp(self) -> None:
        """Configure DDP with custom gradient hooks."""
        self.register_parameter_hooks()
        super().configure_ddp()

    def _setup_shard_shapes(self, trainer: pl.Trainer) -> dict:
        """Set up shard shapes for the dataloader.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.

        Returns
        -------
        dict
            A dictionary containing the shard shapes for each dataset.
        """
        shard_shapes = trainer.model.module.shard_shapes
        assert shard_shapes is not None, "Shard shapes should be set after setup"
        return shard_shapes

    def register_parameter_hooks(self) -> None:
        """Register parameter hooks for gradient reduction."""
        register_gradient_scaling_hooks(self.model, self.model_comm_group_size)


class DDPGroupStrategy(BaseDDPStrategy):
    """Distributed Data Parallel strategy with group communication."""

    def _setup_communication_groups(self) -> int:
        """Set up model and reader communication groups.

        Returns
        -------
        int
            The model communication group ID for this rank.
        """
        model_layout = build_model_layout(
            world_size=self.world_size,
            global_rank=self.global_rank,
            model_comm_group_size=self.model_comm_group_size,
        )
        reader_layout = build_reader_layout(
            model_comm_group_ranks=model_layout.model_comm_group_ranks,
            model_comm_group_size=self.model_comm_group_size,
            read_group_size=self.read_group_size,
            model_comm_group_rank=model_layout.model_comm_group_rank,
            global_rank=self.global_rank,
        )
        model_comm_groups = create_model_process_groups(model_layout.model_comm_group_ranks)
        reader_groups = create_reader_process_groups(reader_layout.reader_group_ranks)
        model_comm_group = model_comm_groups[model_layout.model_comm_group_id]
        model_reader_groups = reader_groups[model_layout.model_comm_group_id]

        self.model.set_model_comm_group(
            model_comm_group,
            model_layout.model_comm_group_id,
            model_layout.model_comm_group_rank,
            model_layout.model_comm_num_groups,
            self.model_comm_group_size,
        )
        self.model.set_reader_groups(
            model_reader_groups,
            reader_layout.reader_group_id,
            reader_layout.reader_group_rank,
            reader_layout.reader_group_size,
        )

        LOGGER.debug(
            "Rank %d model_comm_group_id: %d model_comm_group: %s model_comm_group_rank: %d "
            "reader_group_id: %d reader_group: %s reader_group_rank: %d reader_group_root (global): %d",
            self.global_rank,
            model_layout.model_comm_group_id,
            str(model_layout.model_comm_group_ranks[model_layout.model_comm_group_id]),
            model_layout.model_comm_group_rank,
            reader_layout.reader_group_id,
            reader_layout.reader_group_ranks[model_layout.model_comm_group_id, reader_layout.reader_group_id],
            reader_layout.reader_group_rank,
            reader_layout.reader_group_root,
        )

        return model_layout.model_comm_group_id

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
            self.shard_shapes,
            self.shard_shapes,
        )

        return dataloader


class DDPEnsGroupStrategy(BaseDDPStrategy):
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
        super().__init__(num_gpus_per_model=num_gpus_per_model, read_group_size=read_group_size, **kwargs)
        self.ens_comm_group_size = num_gpus_per_ensemble
        self.shard_shapes: dict | None = None

    def _setup_communication_groups(self) -> int:
        """Set up model, reader, and ensemble communication groups.

        Returns
        -------
        int
            The model communication group ID for this rank.
        """
        model_layout = build_model_layout(
            world_size=self.world_size,
            global_rank=self.global_rank,
            model_comm_group_size=self.model_comm_group_size,
        )
        reader_layout = build_reader_layout(
            model_comm_group_ranks=model_layout.model_comm_group_ranks,
            model_comm_group_size=self.model_comm_group_size,
            read_group_size=self.read_group_size,
            model_comm_group_rank=model_layout.model_comm_group_rank,
            global_rank=self.global_rank,
        )
        model_comm_groups = create_model_process_groups(model_layout.model_comm_group_ranks)
        reader_groups = create_reader_process_groups(reader_layout.reader_group_ranks)
        model_comm_group = model_comm_groups[model_layout.model_comm_group_id]
        model_reader_groups = reader_groups[model_layout.model_comm_group_id]

        self.model.set_model_comm_group(
            model_comm_group,
            model_layout.model_comm_group_id,
            model_layout.model_comm_group_rank,
            model_layout.model_comm_num_groups,
            self.model_comm_group_size,
        )
        self.model.set_reader_groups(
            model_reader_groups,
            reader_layout.reader_group_id,
            reader_layout.reader_group_rank,
            reader_layout.reader_group_size,
        )

        LOGGER.info(
            "Rank %d model_comm_group_id: %d model_comm_group: %s "
            "model_comm_group_rank: %d model_comm_group.size(): %d "
            "reader_group_id: %d reader_group: %s reader_group_rank: %d "
            "reader_group_root (global): %d "
            "model_reader_groups: %s reader_groups: %s",
            self.global_rank,
            model_layout.model_comm_group_id,
            str(model_layout.model_comm_group_ranks[model_layout.model_comm_group_id]),
            model_layout.model_comm_group_rank,
            model_comm_group.size(),
            reader_layout.reader_group_id,
            reader_layout.reader_group_ranks[model_layout.model_comm_group_id, reader_layout.reader_group_id],
            reader_layout.reader_group_rank,
            reader_layout.reader_group_root,
            model_reader_groups,
            reader_groups,
        )

        ensemble_layout = build_ensemble_layout(
            world_size=self.world_size,
            global_rank=self.global_rank,
            ens_comm_group_size=self.ens_comm_group_size,
            model_comm_group_size=self.model_comm_group_size,
            model_comm_group_rank=model_layout.model_comm_group_rank,
        )
        ensemble_groups = create_ensemble_process_groups(ensemble_layout)

        self.model.set_ens_comm_group(
            ensemble_groups.ens_comm_group,
            ensemble_layout.ens_comm_group_id,
            ensemble_layout.ens_comm_group_rank,
            ensemble_layout.ens_comm_num_groups,
            self.ens_comm_group_size,
        )
        self.model.set_ens_comm_subgroup(
            ensemble_groups.ens_comm_subgroup,
            ensemble_layout.ens_comm_subgroup_id,
            ensemble_layout.ens_comm_subgroup_rank,
            ensemble_layout.ens_comm_num_subgroups,
            ensemble_layout.ens_comm_subgroup_size,
        )

        LOGGER.info(
            "Rank %d ens_comm_group_id: %d ens_comm_group: %s ens_comm_group_rank: %d "
            "ens_comm_group_size: %d ens_comm_group.size(): %d ens_comm_subgroup_id: %d "
            "ens_comm_subgroup: %s ens_comm_subgroup_rank: %d ens_comm_subgroup.size(): %d ",
            self.global_rank,
            ensemble_layout.ens_comm_group_id,
            str(ensemble_layout.ens_comm_group_ranks[ensemble_layout.ens_comm_group_id]),
            ensemble_layout.ens_comm_group_rank,
            self.ens_comm_group_size,
            ensemble_groups.ens_comm_group.size(),
            ensemble_layout.ens_comm_subgroup_id,
            str(ensemble_layout.ens_comm_subgroup_ranks[ensemble_layout.ens_comm_subgroup_id]),
            ensemble_layout.ens_comm_subgroup_rank,
            ensemble_layout.ens_comm_subgroup_size,
        )

        return model_layout.model_comm_group_id

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
        ens_comm_group_id, ens_comm_group_rank, ens_comm_num_groups = get_my_ensemble_comm_group(
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
            self.shard_shapes,
            self.shard_shapes,
        )

        dataloader.dataset.set_ens_comm_group_info(
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
        )

        return dataloader
