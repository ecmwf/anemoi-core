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

from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import MultiscaleConfigDiskSchema
from anemoi.training.schemas.training import MultiscaleConfigOnTheFlySchema
from anemoi.training.schemas.training import MultiScaleLossSchema
from anemoi.training.schemas.training import OptimizerSchema
from anemoi.training.schemas.training import TimeAggregateLossWrapperSchema

_TIME_AGG_CFG = {
    "_target_": "anemoi.training.losses.aggregate.TimeAggregateLossWrapper",
    "time_aggregation_types": ["mean", "diff"],
    "loss_fn": {
        "_target_": "anemoi.training.losses.MSELoss",
        "scalers": ["node_weights"],
    },
}


def test_time_aggregate_loss_config_valid() -> None:
    """TimeAggregateLossWrapperSchema accepts a valid config."""
    schema = TimeAggregateLossWrapperSchema(**_TIME_AGG_CFG)
    assert schema.time_aggregation_types == ["mean", "diff"]


def test_time_aggregate_loss_config_invalid_agg_type() -> None:
    """Unknown aggregation type is rejected."""
    cfg = {**_TIME_AGG_CFG, "time_aggregation_types": ["sum"]}
    with pytest.raises(ValidationError):
        TimeAggregateLossWrapperSchema(**cfg)


def test_time_aggregate_loss_config_empty_agg_types() -> None:
    """Empty aggregation list is rejected (min_length=1)."""
    cfg = {**_TIME_AGG_CFG, "time_aggregation_types": []}
    with pytest.raises(ValidationError):
        TimeAggregateLossWrapperSchema(**cfg)


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


_MULTISCALE_BASE = {
    "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
    "per_scale_loss": {"_target_": "anemoi.training.losses.MSELoss", "scalers": []},
    "weights": [0.5, 0.5],
}

_ON_THE_FLY_MULTISCALE_CONFIG = {
    "num_scales": 4,
    "base_num_nearest_neighbours": 16,
    "base_sigma": 0.01570,
}


def test_multiscale_config_disk_valid() -> None:
    MultiscaleConfigDiskSchema(loss_matrices=["filter.npz", None])


def test_multiscale_config_disk_requires_loss_matrices() -> None:
    with pytest.raises(ValidationError):
        MultiscaleConfigDiskSchema()


def test_multiscale_config_on_the_fly_valid() -> None:
    MultiscaleConfigOnTheFlySchema(**_ON_THE_FLY_MULTISCALE_CONFIG)


def test_multiscale_config_on_the_fly_smoothers_valid() -> None:
    MultiscaleConfigOnTheFlySchema(smoothers={"smooth_2x": {"num_nearest_neighbours": 16, "sigma": 0.01570}})


def test_multiscale_config_on_the_fly_requires_num_scales_or_smoothers() -> None:
    with pytest.raises(ValidationError):
        MultiscaleConfigOnTheFlySchema()


def test_multiscale_config_on_the_fly_requires_base_parameters_with_num_scales() -> None:
    with pytest.raises(ValidationError, match=r"base_num_nearest_neighbours.*base_sigma"):
        MultiscaleConfigOnTheFlySchema(num_scales=4)


def test_multiscale_loss_disk_mode_valid() -> None:
    MultiScaleLossSchema(**{**_MULTISCALE_BASE, "multiscale_config": {"loss_matrices": ["filter.npz", None]}})


def test_multiscale_loss_on_the_fly_valid() -> None:
    MultiScaleLossSchema(**{**_MULTISCALE_BASE, "multiscale_config": _ON_THE_FLY_MULTISCALE_CONFIG})


def test_multiscale_loss_mixed_mode_rejected() -> None:
    """loss_matrices (disk) and num_scales (on-the-fly) must not coexist in multiscale_config."""
    with pytest.raises(ValidationError):
        MultiScaleLossSchema(
            **{
                **_MULTISCALE_BASE,
                "multiscale_config": {"loss_matrices": [None], "num_scales": 4},
            },
        )


def test_multiscale_loss_deprecated_loss_matrices_with_on_the_fly_config_rejected() -> None:
    """Deprecated top-level loss_matrices must not be combined with on-the-fly multiscale_config."""
    with pytest.raises(ValidationError):
        MultiScaleLossSchema(
            **{
                **_MULTISCALE_BASE,
                "loss_matrices": [None],
                "multiscale_config": _ON_THE_FLY_MULTISCALE_CONFIG,
            },
        )


def test_multiscale_loss_deprecated_loss_matrices_path_with_on_the_fly_config_rejected() -> None:
    """Deprecated top-level loss_matrices_path must not be combined with on-the-fly multiscale_config."""
    with pytest.raises(ValidationError):
        MultiScaleLossSchema(
            **{
                **_MULTISCALE_BASE,
                "loss_matrices_path": "/some/path",
                "multiscale_config": _ON_THE_FLY_MULTISCALE_CONFIG,
            },
        )


_COMBINED_LOSS_BASE = {
    "_target_": "anemoi.training.losses.combined.CombinedLoss",
}


def test_combined_loss_with_scalers_valid() -> None:
    CombinedLossSchema(
        **{
            **_COMBINED_LOSS_BASE,
            "losses": [
                {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["nan_mask_weights"]},
                {"_target_": "anemoi.training.losses.MAELoss", "scalers": ["nan_mask_weights"]},
            ],
            "loss_weights": [1.0, 1.0],
        },
    )


def test_combined_loss_with_multiscale_valid() -> None:
    CombinedLossSchema(
        **{
            **_COMBINED_LOSS_BASE,
            "losses": [
                {**_MULTISCALE_BASE, "multiscale_config": _ON_THE_FLY_MULTISCALE_CONFIG},
            ],
        },
    )


def test_combined_loss_with_multiscale_mixed_mode_rejected() -> None:
    """MultiscaleLossWrapper nested in CombinedLoss must still reject mixed mode."""
    with pytest.raises(ValidationError):
        CombinedLossSchema(
            **{
                **_COMBINED_LOSS_BASE,
                "losses": [
                    {
                        **_MULTISCALE_BASE,
                        "multiscale_config": {"loss_matrices": [None], "num_scales": 4},
                    },
                ],
            },
        )
