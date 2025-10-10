# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .base import BaseGraphModel
from .encoder_processor_decoder import AnemoiModelEncProcDec
from .ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from .hierarchical import AnemoiModelEncProcDecHierarchical
from .interpolator import AnemoiModelEncProcDecInterpolator
from .autoencoder import AnemoiModelAutoEncoder # isort: skip
from .autoencoder import AnemoiModelHierarchicalAutoEncoder # isort: skip
from .diffusion_encoder_processor_decoder import AnemoiDiffusionModelEncProcDec # isort: skip
from .diffusion_encoder_processor_decoder import AnemoiDiffusionTendModelEncProcDec # isort: skip

__all__ = [
    "BaseGraphModel",
    "AnemoiModelEncProcDec",
    "AnemoiEnsModelEncProcDec",
    "AnemoiDiffusionModelEncProcDec",
    "AnemoiDiffusionTendModelEncProcDec",
    "AnemoiModelEncProcDecHierarchical",
    "AnemoiModelEncProcDecInterpolator",
    "AnemoiEnsModelEncProcDec",
    "AnemoiModelAutoEncoder",
    "AnemoiModelHierarchicalAutoEncoder",
]
