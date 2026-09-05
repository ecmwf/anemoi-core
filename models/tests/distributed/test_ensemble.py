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

from anemoi.models.distributed.graph import gather_ensemble


class EnsembleModel(torch.nn.Module):
    """Small nonlinear model with parameters upstream of the ensemble gather."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.7], [-0.3]], device=device, dtype=torch.float64))
        self.trainable_bias = torch.nn.Parameter(torch.tensor([0.2], device=device, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.weight + self.trainable_bias)


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
    use_checkpoint: bool,
) -> None:
    num_data_groups = 2 if data_parallel else 1
    ensemble_group_size = world_size // num_data_groups
    data_group_id = rank // ensemble_group_size
    ensemble_rank = rank % ensemble_group_size
    ensemble_group = group
    if data_parallel:
        for start in range(0, world_size, ensemble_group_size):
            ranks = list(range(start, start + ensemble_group_size))
            subgroup = dist.new_group(ranks=ranks)
            if rank in ranks:
                ensemble_group = subgroup

    members_per_rank = 2
    generator = torch.Generator().manual_seed(71)
    inputs = torch.randn(
        num_data_groups,
        members_per_rank * ensemble_group_size,
        3,
        2,
        dtype=torch.float64,
        generator=generator,
    ).to(device)
    reference = EnsembleModel(device)
    reference_prediction = reference(inputs)
    ensemble_loss(reference_prediction).backward()

    model = EnsembleModel(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    member_start = ensemble_rank * members_per_rank
    local_input = inputs[data_group_id : data_group_id + 1, member_start : member_start + members_per_rank]

    def gather_and_score(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = gather_ensemble(
            prediction.clone(),
            dim=1,
            sizes=[members_per_rank] * ensemble_group_size,
            mgroup=ensemble_group,
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
@pytest.mark.parametrize("data_parallel", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_gather_ensemble_matches_unsharded_parameter_gradients(
    data_parallel: bool,
    use_checkpoint: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    if data_parallel and distributed_world_size % 2:
        pytest.skip("Two data-parallel groups require an even world size.")
    run_distributed_test(
        _test_gather_ensemble_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        data_parallel=data_parallel,
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
