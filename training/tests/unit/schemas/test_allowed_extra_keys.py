# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.training.schemas.training import OptimizerSchema


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
