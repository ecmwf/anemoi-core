# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from anemoi.training.distributed.single import SingleDeviceStrategy
from anemoi.training.distributed.strategy import DDPGroupStrategy
from anemoi.training.train.train import AnemoiTrainer

DDP_TARGET = "anemoi.training.distributed.strategy.DDPGroupStrategy"
SINGLE_TARGET = "anemoi.training.distributed.single.SingleDeviceStrategy"


def _build_strategy(data_types: list[str], target: str, **ddp_settings: bool) -> object:
    """Run ``AnemoiTrainer.strategy`` for readers of the given data types and a strategy config."""
    strategy_config = {"_target_": target, "num_gpus_per_model": 1, "read_group_size": 1, **ddp_settings}
    if target == SINGLE_TARGET:
        strategy_config["device"] = "cpu"
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {"strategy": strategy_config}}),
        datamodule=SimpleNamespace(
            ds_train=SimpleNamespace(
                data_readers={f"data_{i}": SimpleNamespace(data_type=dt) for i, dt in enumerate(data_types)},
            ),
        ),
    )
    return AnemoiTrainer.strategy.func(trainer)


@pytest.mark.parametrize(
    "ddp_settings",
    [
        pytest.param({"static_graph": True, "find_unused_parameters": True}, id="static-graph"),
        pytest.param({"static_graph": False, "find_unused_parameters": False}, id="no-unused-search"),
        pytest.param({}, id="ddp-defaults"),
    ],
)
def test_tabular_data_rejects_ddp_settings_that_assume_all_parameters_train(ddp_settings: dict) -> None:
    with pytest.raises(ValueError, match="tabular datasets"):
        _build_strategy(["gridded", "tabular"], DDP_TARGET, **ddp_settings)


def test_tabular_data_accepts_dynamic_graph_settings() -> None:
    strategy = _build_strategy(["gridded", "tabular"], DDP_TARGET, static_graph=False, find_unused_parameters=True)

    assert isinstance(strategy, DDPGroupStrategy)
    assert strategy._ddp_kwargs["static_graph"] is False
    assert strategy._ddp_kwargs["find_unused_parameters"] is True


def test_gridded_data_keeps_configured_static_graph() -> None:
    strategy = _build_strategy(["gridded"], DDP_TARGET, static_graph=True, find_unused_parameters=False)

    assert strategy._ddp_kwargs["static_graph"] is True
    assert strategy._ddp_kwargs["find_unused_parameters"] is False


def test_single_device_ignores_ddp_settings_with_tabular_data() -> None:
    strategy = _build_strategy(["tabular"], SINGLE_TARGET, static_graph=True, find_unused_parameters=False)

    assert isinstance(strategy, SingleDeviceStrategy)
