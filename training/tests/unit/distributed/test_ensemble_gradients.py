# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare ensemble-sharded DDP gradients with an unsharded reference on CPU."""

from datetime import timedelta
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.training.distributed.groups import get_my_ensemble_comm_group
from anemoi.training.distributed.strategy import DDPEnsGroupStrategy


class _EnsembleModel(pl.LightningModule):
    """Small test model."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        # Test parameters excluded from the default model scaling hook.
        self.trainable = torch.nn.Parameter(torch.linspace(0.1, 0.4, 4, dtype=torch.float64))
        self.no_gradscaling = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

        self.model_comm_group = None
        self.ens_comm_subgroup = None

    def set_model_comm_group(self, group: dist.ProcessGroup, *_metadata: int) -> None:
        self.model_comm_group = group

    def set_reader_groups(self, groups: list, *_metadata: int) -> None:
        """No dataloader is needed for the test inputs."""

    def set_ens_comm_group(self, group: dist.ProcessGroup, *_metadata: int) -> None:
        """Only the ensemble subgroup is used for gathering predictions."""

    def set_ens_comm_subgroup(self, group: dist.ProcessGroup, *_metadata: int) -> None:
        self.ens_comm_subgroup = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs contain local ensemble members and all four grid points.
        model_shards = self.model_comm_group.size() if self.model_comm_group is not None else 1
        # The two biases receive full-grid gradients through the backward gather in shard_tensor
        # Hence, only weight needs the model scaling factor.
        x = shard_tensor(
            x + self.trainable + self.no_gradscaling,
            dim=1,
            sizes=[4 // model_shards] * model_shards,
            mgroup=self.model_comm_group,
        )
        # Apply the weight after sharding across the  grid dimension, so its gradient covers only local points.
        return x * self.weight

    def loss(self, prediction: torch.Tensor) -> torch.Tensor:
        # Full predictions have shape [4 ensemble members, 4 grid points].
        # A missing group means the reference is unsharded along that dimension.
        ensemble_shards = self.ens_comm_subgroup.size() if self.ens_comm_subgroup is not None else 1
        model_shards = self.model_comm_group.size() if self.model_comm_group is not None else 1
        # Gather the ensemble dimension (0) across the ensemble subgroup:
        # collect all members while retaining this rank's grid partition.
        prediction = gather_tensor(
            prediction.clone(),  # Match the ensemble method's handling of checkpointed predictions.
            dim=0,
            sizes=[4 // ensemble_shards] * ensemble_shards,
            mgroup=self.ens_comm_subgroup,
        )
        # Gather the grid dimension (1) across the model group:
        # collect all grid partitions so that we get the full [4, 4] prediction.
        prediction = gather_tensor(
            prediction,
            dim=1,
            sizes=[4 // model_shards] * model_shards,
            mgroup=self.model_comm_group,
        )
        # Average over members: one ensemble mean per grid point.
        ensemble_mean = prediction.mean(dim=0)
        # Compare the ensemble mean with the target, then average over grid points.
        ensemble_mean_loss = (ensemble_mean - 0.3).square().mean()
        # Square each member's prediction before averaging over members and grid points.
        per_member_loss = prediction.square().mean()
        return ensemble_mean_loss + 0.2 * per_member_loss


def _check_ensemble_gradients(rank: int, init_file: str) -> None:
    # Initialize dist backend
    torch.set_num_threads(1)
    world_size = 4
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        # Give the Lightning strategy this worker's global rank and total rank count.
        environment = LightningEnvironment()
        environment.set_global_rank(rank)
        environment.set_world_size(world_size)
        # Ensemble-only, model + ensemble, data + ensemble, and no ensemble split.
        for model_shards, ensemble_shards in [(1, 4), (2, 2), (1, 2), (2, 1)]:
            # One ensemble group contains all member and grid partitions for one input.
            ensemble_group_size = model_shards * ensemble_shards
            # Identify this rank's data group.
            data_rank, _, data_groups = get_my_ensemble_comm_group(ensemble_group_size, rank, world_size)
            # Each data group gets different values for its full [member, grid] input.
            inputs = [torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10 + i for i in range(data_groups)]
            # Compute expected gradients without sharding, hooks, or DDP.
            reference = _EnsembleModel()
            # Average independent data-group losses, each evaluated on all members and points.
            data_group_losses = torch.stack([reference.loss(reference(x)) for x in inputs])
            reference_loss = data_group_losses.mean()
            reference_loss.backward()

            # Create an identically model for the distributed calculation.
            model = _EnsembleModel()
            strategy = DDPEnsGroupStrategy(
                num_gpus_per_model=model_shards,
                num_gpus_per_ensemble=ensemble_group_size,
                read_group_size=1,
                cluster_environment=environment,
            )
            # Let the strategy attach its model, reader, and ensemble groups.
            strategy.connect(model)
            strategy._setup_communication_groups()
            strategy.register_parameter_hooks()
            # DDP averages parameter gradients across all four ranks during backward.
            model_ddp = DistributedDataParallel(model)
            # Select this subgroup rank's members; forward() will split their grid points.
            ensemble_rank = model.ens_comm_subgroup.rank()
            local_input = inputs[data_rank].chunk(ensemble_shards, dim=0)[ensemble_rank].contiguous()

            # Run both ordinary backward and the checkpointed loss used in training.
            for checkpoint_loss in [False, True]:
                model_ddp.zero_grad(set_to_none=True)
                prediction = model_ddp(local_input)
                loss = (
                    checkpoint(model.loss, prediction, use_reentrant=False)
                    if checkpoint_loss
                    else model.loss(prediction)
                )
                # Compare loss to reference loss.
                torch.testing.assert_close(loss, reference.loss(reference(inputs[data_rank])), rtol=1e-12, atol=1e-12)

                # Run the gather/split backward, scaling hooks, and DDP averaging.
                loss.backward()
                # Every parameter must match the gradient averaged over all reference inputs,
                # including the parameters excluded from model scaling.
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        reference.get_parameter(name).grad,
                        rtol=1e-12,
                        atol=1e-12,
                        msg=(
                            f"{name}: rank={rank}, model_shards={model_shards}, "
                            f"ensemble_shards={ensemble_shards}, checkpoint={checkpoint_loss}"
                        ),
                    )
    finally:
        # Release all communication groups even if a gradient comparison fails.
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
def test_ensemble_ddp_gradients_match_unsharded_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DDP must average data groups while summing model and ensemble contributions."""
    # Limit thread pools before the spawned workers import numerical libraries.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    # Launch four CPU ranks and propagate any worker's assertion failure to pytest.
    mp.spawn(_check_ensemble_gradients, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)
