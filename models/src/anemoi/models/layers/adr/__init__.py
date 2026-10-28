# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .advection import GraphAdvection
from .advection import LatLonAdvection
from .diffusion import GraphDiffusion
from .diffusion import LatLonDiffusion
from .reaction import GraphReaction
from .reaction import LatLonReaction

__all__ = [
    "LatLonAdvection",
    "LatLonDiffusion",
    "LatLonReaction",
    "GraphAdvection",
    "GraphDiffusion",
    "GraphReaction",
]
