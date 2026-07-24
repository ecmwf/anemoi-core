# (C) Copyright 2024-2026 Anemoi contributors.
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

from .autoencoder import AnemoiModelAutoEncoder  # isort: skip
from .hierarchical_autoencoder import AnemoiModelHierarchicalAutoEncoder  # isort: skip
from .transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec  # isort: skip
from .transport_encoder_processor_decoder import AnemoiTransportSpatialDownscalerModelEncProcDec  # isort: skip
from .transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec  # isort: skip

__all__ = [
    "BaseGraphModel",
    "AnemoiModelEncProcDec",
    "AnemoiEnsModelEncProcDec",
    "AnemoiTransportModelEncProcDec",
    "AnemoiTransportSpatialDownscalerModelEncProcDec",
    "AnemoiTransportTendModelEncProcDec",
    "AnemoiModelEncProcDecHierarchical",
    "AnemoiEnsModelEncProcDec",
    "AnemoiModelAutoEncoder",
    "AnemoiModelHierarchicalAutoEncoder",
]
