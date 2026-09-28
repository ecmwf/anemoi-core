# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace
from unittest.mock import PropertyMock

import pytest
from pytest_mock import MockFixture

from anemoi.training.distributed.strategy import BaseDDPStrategy
from anemoi.training.distributed.strategy import DDPEnsGroupStrategy
from anemoi.training.distributed.strategy import DDPGroupStrategy
from anemoi.training.distributed.strategy import seed_rnd
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed


def test_seed_rnd_uses_bounded_model_seed(mocker: MockFixture) -> None:
    base_seed = 19525198
    model_comm_group_id = 219
    expected_seed = derive_seed(base_seed, SeedContext.MODEL, model_comm_group_id)

    mocker.patch("anemoi.training.distributed.strategy.get_base_seed", return_value=base_seed)
    seed_everything = mocker.patch(
        "anemoi.training.distributed.strategy.pl.seed_everything",
        return_value=expected_seed,
    )
    mocker.patch("anemoi.training.distributed.strategy.torch.rand", return_value=[0.0])

    seed_rnd(model_comm_group_id, global_rank=0)

    seed_everything.assert_called_once_with(expected_seed)
    assert 0 <= expected_seed <= 2**32 - 1


def _place_strategy(mocker: MockFixture, strategy: BaseDDPStrategy, global_rank: int, world_size: int) -> None:
    mocker.patch.object(type(strategy), "global_rank", new_callable=PropertyMock, return_value=global_rank)
    mocker.patch.object(type(strategy), "world_size", new_callable=PropertyMock, return_value=world_size)


@pytest.mark.parametrize(
    ("strategy", "expected_groups"),
    [
        pytest.param(
            DDPGroupStrategy(num_gpus_per_model=2, read_group_size=1),
            [0, 0, 1, 1, 2, 2, 3, 3],
            id="model-groups",
        ),
        pytest.param(
            DDPEnsGroupStrategy(num_gpus_per_model=2, num_gpus_per_ensemble=4, read_group_size=1),
            [0, 0, 0, 0, 1, 1, 1, 1],
            id="ensemble-groups",
        ),
    ],
)
def test_distributed_sampler_kwargs_split_by_sample_group(
    mocker: MockFixture,
    strategy: BaseDDPStrategy,
    expected_groups: list[int],
) -> None:
    """Ranks that train on the same samples form one sampler replica."""
    world_size = len(expected_groups)
    num_groups = len(set(expected_groups))

    for global_rank, expected_group in enumerate(expected_groups):
        _place_strategy(mocker, strategy, global_rank, world_size)
        assert strategy.distributed_sampler_kwargs == {"num_replicas": num_groups, "rank": expected_group}


def test_process_dataloader_passes_reader_group_info(mocker: MockFixture) -> None:
    """The dataset learns which part of the grid this rank reads."""
    strategy = DDPGroupStrategy(num_gpus_per_model=4, read_group_size=2)
    strategy.shard_sizes = {"data": [5, 5]}
    _place_strategy(mocker, strategy, global_rank=7, world_size=8)
    dataloader = SimpleNamespace(dataset=mocker.Mock())

    assert strategy.process_dataloader(dataloader) is dataloader

    # Global rank 7 is rank 3 of its model group, which is rank 1 of its reader group.
    dataloader.dataset.set_reader_group_info.assert_called_once_with(1, {"data": [5, 5]})
