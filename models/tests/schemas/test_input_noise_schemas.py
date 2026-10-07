# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.models.schemas.models import SphericalInputConditionedNoiseSchema
from anemoi.models.schemas.models import SphericalInputNoiseSchema
from anemoi.models.schemas.models import SphericalInputNoiseUnion

KT = [1.2322e-4, 4.9289e-4, 1.9716e-3, 7.8862e-3]


def conditioned(**modulation) -> dict:
    return {
        "_target_": "anemoi.models.layers.ensemble.SphericalInputConditionedNoise",
        "grid": "n320",
        "dataset": "era5",
        "n_channels": 12,
        "noise": {"type": "diffusion", "sigma": 1.0, "lmax": 640, "kT": KT * 3},
        "modulation": {
            "groups": {"A": ["t", "skt", "2t"], "B": ["u", "v", "z", "msl", "sp", "10u", "10v"], "C": ["q", "2d"]},
            "channel_group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            **modulation,
        },
    }


def validate(config: dict):
    return TypeAdapter(SphericalInputNoiseUnion).validate_python(config)


def test_plain_fcn3_noise_still_validates() -> None:
    config = {"_target_": "anemoi.models.layers.ensemble.SphericalInputNoise", "grid": "n320", "noise": {}}
    assert isinstance(validate(config), SphericalInputNoiseSchema)


def test_conditioned_noise_validates_with_defaults() -> None:
    schema = validate(conditioned(area_weight=True, source="eda_stdev", clip=[0.25, 4.0]))

    assert isinstance(schema, SphericalInputConditionedNoiseSchema)
    assert schema.modulation.variable_prefix == "std_"
    assert schema.modulation.band_filter.quantile == 0.99
    assert schema.modulation.band_filter.taper == 0.25


def test_undefined_channel_group_is_rejected() -> None:
    with pytest.raises(ValidationError, match="undefined groups"):
        validate(conditioned(channel_group=["A"] * 11 + ["D"]))


def test_channel_group_must_cover_every_channel() -> None:
    with pytest.raises(ValidationError, match="n_channels=12"):
        validate(conditioned(channel_group=["A"] * 8))


def test_disabled_modulation_does_not_need_matching_channels() -> None:
    validate(conditioned(enabled=False, channel_group=["A"]))


@pytest.mark.parametrize("band_filter", [{"quantile": 0.0}, {"quantile": 1.2}, {"taper": -0.5}])
def test_invalid_band_filter_is_rejected(band_filter) -> None:
    with pytest.raises(ValidationError):
        validate(conditioned(band_filter=band_filter))


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        validate(conditioned(smooth_to_chanel=True))
