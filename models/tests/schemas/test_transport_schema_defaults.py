# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.schemas.models import TransportConfig
from anemoi.models.schemas.models import TransportSourceConfig
from anemoi.models.transport.settings import EdmSettings
from anemoi.models.transport.settings import NoiseConditioningSettings
from anemoi.models.transport.settings import StochasticInterpolantSettings
from anemoi.models.transport.settings import TransportSourceSettings


def test_schema_defaults_match_transport_settings() -> None:
    """The schema defaults are restated literals, so pin them to the dataclasses."""
    source = TransportSourceConfig()
    assert source.kind == TransportSourceSettings.kind
    assert source.scale == TransportSourceSettings.scale
    assert source.noise_scale == TransportSourceSettings.noise_scale

    transport = TransportConfig()
    assert transport.sigma_data == EdmSettings.sigma_data
    assert transport.sigma_max == EdmSettings.sigma_max
    assert transport.sigma_min == EdmSettings.sigma_min
    assert transport.rho == EdmSettings.rho
    assert transport.noise_channels == NoiseConditioningSettings.channels
    assert transport.noise_cond_dim == NoiseConditioningSettings.cond_dim
    assert transport.si_alpha_schedule == StochasticInterpolantSettings.alpha_schedule
    assert transport.si_beta_schedule == StochasticInterpolantSettings.beta_schedule
    assert transport.si_sigma_schedule == StochasticInterpolantSettings.sigma_schedule
    assert transport.si_noise_scale == StochasticInterpolantSettings.noise_scale
