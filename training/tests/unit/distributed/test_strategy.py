# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pytest_mock import MockFixture

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
