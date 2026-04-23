# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from anemoi.training.schemas.training import OptimizerSchema
from anemoi.training.schemas.training import TimeAggregateLossConfigSchema


_TIME_AGG_CFG = {
    "time_aggregation_types": ["mean", "diff"],
    "weight": 0.5,
    "loss_fn": {
        "_target_": "anemoi.training.losses.MSELoss",
        "scalers": ["node_weights"],
    },
}


def test_time_aggregate_loss_config_valid() -> None:
    """TimeAggregateLossConfigSchema accepts a valid config."""
    schema = TimeAggregateLossConfigSchema(**_TIME_AGG_CFG)
    assert schema.time_aggregation_types == ["mean", "diff"]
    assert schema.weight == 0.5


def test_time_aggregate_loss_config_default_weight() -> None:
    """Weight defaults to 1.0 when not specified."""
    cfg = {k: v for k, v in _TIME_AGG_CFG.items() if k != "weight"}
    schema = TimeAggregateLossConfigSchema(**cfg)
    assert schema.weight == 1.0


def test_time_aggregate_loss_config_invalid_agg_type() -> None:
    """Unknown aggregation type is rejected."""
    cfg = {**_TIME_AGG_CFG, "time_aggregation_types": ["sum"]}
    with pytest.raises(ValidationError):
        TimeAggregateLossConfigSchema(**cfg)


def test_time_aggregate_loss_config_empty_agg_types() -> None:
    """Empty aggregation list is rejected (min_length=1)."""
    cfg = {**_TIME_AGG_CFG, "time_aggregation_types": []}
    with pytest.raises(ValidationError):
        TimeAggregateLossConfigSchema(**cfg)


def test_optimizer_schema_allows_extra_keys() -> None:
    """Test that the OptimizerSchema allows extra keys."""
    # Explicitly test for the issue present in (anemoi-core/#885)[https://github.com/ecmwf/anemoi-core/pull/885]
    optimizer_config = {
        "_target_": "torch.optim.AdamW",
        "lr": 0.001,
        "weight_decay": 0.01,
        "extra_key": "extra_value",  # This key is not defined in the schema
    }
    optimizer_schema = OptimizerSchema(**optimizer_config)
    assert optimizer_schema.target_ == "torch.optim.AdamW"
    assert optimizer_schema.lr == 0.001
    assert optimizer_schema.weight_decay == 0.01
    assert optimizer_schema.extra_key == "extra_value"

    model_dump = optimizer_schema.model_dump(by_alias=True)
    assert model_dump["_target_"] == "torch.optim.AdamW"
    assert model_dump["lr"] == 0.001
    assert model_dump["weight_decay"] == 0.01
    assert model_dump["extra_key"] == "extra_value"
