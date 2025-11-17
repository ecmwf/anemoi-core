# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class CosLatitude(BaseNodeAttribute):
    """Computes the cosine of node latitudes. Node coordinates are expected to be in radians."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        lat = nodes.x[:, 0]
        return torch.cos(lat)


class CosLongitude(BaseNodeAttribute):
    """Computes the cosine of node longitudes. Node coordinates are expected to be in radians."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        lon = nodes.x[:, 1]
        return torch.cos(lon)


class SinLatitude(BaseNodeAttribute):
    """Computes the sine of node latitudes. Node coordinates are expected to be in radians."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        lat = nodes.x[:, 0]
        return torch.sin(lat)


class SinLongitude(BaseNodeAttribute):
    """Computes the sine of node longitudes. Node coordinates are expected to be in radians."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        lon = nodes.x[:, 1]
        return torch.sin(lon)
