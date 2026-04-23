# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.schemas.training import OptimizerSchema

_AGGREGATE_LOSS_CFG = {
    "_target_": "anemoi.training.losses.TimeAggregateLossWrapper",
    "time_aggregation_types": ["mean", "diff"],
    "loss_fn": {
        "_target_": "anemoi.training.losses.MSELoss",
        "scalers": ["node_weights"],
    },
    "scalers": [],
}

_MSE_CFG = {
    "_target_": "anemoi.training.losses.MSELoss",
    "scalers": ["node_weights"],
}


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


def test_time_aggregate_loss_rejected_as_standalone() -> None:
    """TimeAggregateLossWrapper must not be usable as a top-level training loss."""
    ta = TypeAdapter(LossSchemas)
    with pytest.raises(ValidationError):
        ta.validate_python(_AGGREGATE_LOSS_CFG)


def test_time_aggregate_loss_accepted_inside_combined_loss() -> None:
    """TimeAggregateLossWrapper must be valid as a child of CombinedLoss."""
    combined_cfg = {
        "_target_": "anemoi.training.losses.combined.CombinedLoss",
        "scalers": [],
        "losses": [_MSE_CFG, _AGGREGATE_LOSS_CFG],
    }
    schema = CombinedLossSchema(**combined_cfg)
    assert len(schema.losses) == 2


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
