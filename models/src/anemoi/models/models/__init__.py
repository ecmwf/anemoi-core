# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .base import AnemoiModel
from .encoder_processor_decoder import AnemoiModelEncProcDec
from .ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from .hierarchical import AnemoiModelEncProcDecHierarchical
from .interpolator import AnemoiModelEncProcDecInterpolator
from .multi_model import AnemoiMultiModel

print("todo fix import")
# order matters: downscaling imports AnemoiMultiModel and must be after it
from .downscaling import AnemoiDownscalingModel  # noqa(F401)

__all__ = [
    "AnemoiModelEncProcDec",
    "AnemoiMultiModel",
    "AnemoiEnsModelEncProcDec",
    "AnemoiModelEncProcDecHierarchical",
    "AnemoiModelEncProcDecInterpolator",
    "AnemoiDownscalingModel",
]
