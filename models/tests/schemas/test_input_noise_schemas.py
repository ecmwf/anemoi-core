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

SPECTRUM = {"degree": [1.0, 4.0, 16.0, 64.0], "sigma2": [1.0, 0.8, 0.1, 1e-4]}


def conditioned(channels: dict | None = None, **modulation) -> dict:
    return {
        "_target_": "anemoi.models.layers.ensemble.SphericalInputConditionedNoise",
        "grid": "n320",
        "dataset": "era5",
        "noise": {"type": "diffusion", "sigma": 1.0, "lmax": 640},
        "channels": channels
        or {
            "wind_850": {"spectrum": SPECTRUM, "spread": ["u_850", "v_850"]},
            "t_850": {"spectrum": SPECTRUM, "spread": ["t_850"], "smoothing_km": 200},
            "fcn3_1600km": {"kT": 3.1545e-2},
        },
        "modulation": {"normalizer": "std", **modulation},
    }


def validate(config: dict):
    return TypeAdapter(SphericalInputNoiseUnion).validate_python(config)


def test_plain_fcn3_noise_still_validates() -> None:
    config = {"_target_": "anemoi.models.layers.ensemble.SphericalInputNoise", "grid": "n320", "noise": {}}
    schema = validate(config)
    assert isinstance(schema, SphericalInputNoiseSchema)
    assert schema.n_channels is None and schema.channels is None


def test_plain_noise_accepts_named_channels_without_spread() -> None:
    config = {
        "_target_": "anemoi.models.layers.ensemble.SphericalInputNoise",
        "grid": "n320",
        "noise": {"type": "diffusion"},
        "channels": {"small": {"kT": 1.2322e-4}, "measured": {"spectrum": SPECTRUM}},
    }
    validate(config)
    config["channels"]["small"]["spread"] = ["t"]
    with pytest.raises(ValidationError, match="SphericalInputConditionedNoise"):
        validate(config)


def test_conditioned_noise_validates_with_defaults() -> None:
    schema = validate(conditioned())

    assert isinstance(schema, SphericalInputConditionedNoiseSchema)
    modulation = schema.modulation
    assert (modulation.reference, modulation.variable_prefix) == ("climatology", "std_")
    assert modulation.clip == (0.05, 10.0)
    assert modulation.smoothing_km == 100.0
    assert (modulation.band_filter.quantile, modulation.band_filter.taper) == (0.99, 0.25)


@pytest.mark.parametrize(
    "modulation",
    [{"rescale": "climatology"}, {"smooth_to_channel": True}, {"band_filter": {"preserve_variance": True}}],
)
def test_unknown_modulation_keys_are_rejected(modulation) -> None:
    with pytest.raises(ValidationError):
        validate(conditioned(**modulation))


def test_climatology_needs_a_normalizer() -> None:
    config = conditioned()
    del config["modulation"]["normalizer"]
    with pytest.raises(ValidationError, match="normalizer"):
        validate(config)
    validate(conditioned(normalizer="none"))
    with pytest.raises(ValidationError):
        validate(conditioned(normalizer="mean-std"))


@pytest.mark.parametrize(
    "channel, match",
    [
        ({"spread": ["t"]}, "exactly one"),
        ({"kT": 0.01, "spectrum": SPECTRUM}, "exactly one"),
        ({"spectrum": {"degree": [4.0, 2.0], "sigma2": [1.0, 1.0]}}, "strictly increasing"),
        ({"spectrum": {"degree": [1.0, 2.0, 3.0], "sigma2": [1.0, 1.0]}}, "values"),
    ],
)
def test_invalid_channels_are_rejected(channel, match) -> None:
    with pytest.raises(ValidationError, match=match):
        validate(conditioned({"a": channel, "b": {"kT": 0.01, "spread": ["q"]}}))


def test_scaling_needs_a_channel_with_spread() -> None:
    with pytest.raises(ValidationError, match="names 'spread'"):
        validate(conditioned({"a": {"kT": 0.01}}))
    validate(conditioned({"a": {"kT": 0.01}}, enabled=False))


def test_channels_exclude_noise_kt_and_mismatched_counts() -> None:
    config = conditioned()
    config["noise"]["kT"] = [0.01]
    with pytest.raises(ValidationError, match="per channel"):
        validate(config)
    config = conditioned()
    config["n_channels"] = 4
    with pytest.raises(ValidationError, match="n_channels=4"):
        validate(config)


@pytest.mark.parametrize("band_filter", [{"quantile": 0.0}, {"quantile": 1.2}, {"taper": -0.5}])
def test_invalid_band_filter_is_rejected(band_filter) -> None:
    with pytest.raises(ValidationError):
        validate(conditioned(band_filter=band_filter))


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        validate(conditioned(smooth_to_chanel=True))
