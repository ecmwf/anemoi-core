# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Check ensemble gathering against an unsharded model's parameter gradients."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import gather_ensemble
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor


class EnsembleModel(torch.nn.Module):
    """Small nonlinear model with parameters before and after grid sharding."""

    def __init__(self, device: torch.device, model_group: dist.ProcessGroup | None = None) -> None:
        super().__init__()
        self.model_group = model_group
        self.weight = torch.nn.Parameter(torch.tensor([[0.7], [-0.3]], device=device, dtype=torch.float64))
        self.trainable_bias = torch.nn.Parameter(torch.tensor([0.2], device=device, dtype=torch.float64))

        if model_group is not None:
            # Apply the strategy's model-sharding scaling rule to the weight,
            # which only sees a grid shard. The bias is added before sharding,
            # so shard_tensor's backward gathers its full-grid contribution.
            self.weight.register_hook(lambda grad: grad * model_group.size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.trainable_bias
        if self.model_group is not None:
            sizes = get_balanced_partition_sizes(x.size(2), self.model_group.size())
            x = shard_tensor(x, dim=2, sizes=sizes, mgroup=self.model_group)
        return torch.tanh(x @ self.weight)


def ensemble_loss(prediction: torch.Tensor) -> torch.Tensor:
    """Use a loss that couples ensemble members."""
    return prediction.mean(dim=1).square().mean() + 0.3 * prediction.var(dim=1, unbiased=False).mean()


def _test_gather_ensemble_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    data_parallel: bool,
    model_group_size: int,
    use_checkpoint: bool,
) -> None:
    num_data_groups = 2 if data_parallel else 1
    ensemble_group_size = world_size // num_data_groups
    ensemble_subgroup_size = ensemble_group_size // model_group_size
    data_group_id = rank // ensemble_group_size
    ensemble_rank = (rank % ensemble_group_size) // model_group_size
    model_rank = rank % model_group_size

    model_group = None
    if model_group_size > 1:
        model_groups = [
            dist.new_group(ranks=list(range(start, start + model_group_size)))
            for start in range(0, world_size, model_group_size)
        ]
        model_group = model_groups[rank // model_group_size]

    ensemble_subgroup = group
    if data_parallel or model_group_size > 1:
        ensemble_subgroups = [
            dist.new_group(ranks=list(range(start + offset, start + ensemble_group_size, model_group_size)))
            for start in range(0, world_size, ensemble_group_size)
            for offset in range(model_group_size)
        ]
        ensemble_subgroup = ensemble_subgroups[data_group_id * model_group_size + model_rank]

    members_per_rank = 2
    generator = torch.Generator().manual_seed(71)
    inputs = torch.randn(
        num_data_groups,
        members_per_rank * ensemble_subgroup_size,
        3,
        2,
        dtype=torch.float64,
        generator=generator,
    ).to(device)
    reference = EnsembleModel(device)
    reference_prediction = reference(inputs)
    ensemble_loss(reference_prediction).backward()

    model = EnsembleModel(device, model_group=model_group)
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    member_start = ensemble_rank * members_per_rank
    local_input = inputs[data_group_id : data_group_id + 1, member_start : member_start + members_per_rank]
    grid_shard_sizes = get_balanced_partition_sizes(inputs.size(2), model_group_size)

    def gather_and_score(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = gather_ensemble(
            prediction.clone(),
            dim=1,
            sizes=[members_per_rank] * ensemble_subgroup_size,
            mgroup=ensemble_subgroup,
        )
        gathered = gather_tensor(
            gathered.clone(),
            dim=2,
            sizes=grid_shard_sizes,
            mgroup=model_group,
        )
        return ensemble_loss(gathered), gathered

    prediction = ddp(local_input)
    if use_checkpoint:
        loss, gathered = checkpoint(gather_and_score, prediction, use_reentrant=False)
    else:
        loss, gathered = gather_and_score(prediction)
    expected_prediction = reference_prediction[data_group_id : data_group_id + 1]
    torch.testing.assert_close(gathered, expected_prediction, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loss, ensemble_loss(expected_prediction), rtol=1e-12, atol=1e-12)
    loss.backward()

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=1e-12, atol=1e-12)


@pytest.mark.distributed
@pytest.mark.parametrize("data_parallel", [False, True], ids=["data_groups_1", "data_groups_2"])
@pytest.mark.parametrize("model_group_size", [1, 2], ids=["model_shards_1", "model_shards_2"])
@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["no_checkpoint", "checkpoint"])
def test_gather_ensemble_matches_unsharded_parameter_gradients(
    data_parallel: bool,
    model_group_size: int,
    use_checkpoint: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    num_data_groups = 2 if data_parallel else 1
    if distributed_world_size % (num_data_groups * model_group_size):
        pytest.skip("World size must be divisible by the number of data groups times the model group size.")
    run_distributed_test(
        _test_gather_ensemble_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        data_parallel=data_parallel,
        model_group_size=model_group_size,
        use_checkpoint=use_checkpoint,
    )


def test_gather_ensemble_without_group_preserves_gradients() -> None:
    model = EnsembleModel(torch.device("cpu"))
    inputs = torch.tensor([[[[1.0, -2.0]], [[3.0, 0.5]]]], dtype=torch.float64)
    prediction = model(inputs)
    gathered = gather_ensemble(prediction, dim=1, sizes=[2], mgroup=None)
    actual_gradients = torch.autograd.grad(ensemble_loss(gathered), tuple(model.parameters()), retain_graph=True)
    expected_gradients = torch.autograd.grad(ensemble_loss(prediction), tuple(model.parameters()))
    torch.testing.assert_close(gathered, prediction)
    torch.testing.assert_close(actual_gradients, expected_gradients)
