# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch

from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder


class BaseICONNodeBuilder(BaseNodeBuilder):
    """Base ICON Node Builder."""

    icon_node_class: type[ICONCellDataGrid] | type[ICONMultiMesh]

    def __init__(self, name: str, grid_filename: str, max_level: int) -> None:
        self.icon_nodes = self.icon_node_class(icon_grid_filename=grid_filename, max_level=max_level)
        super().__init__(name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"icon_nodes"}

    def get_coordinates(self) -> torch.Tensor:
        return torch.from_numpy(self.icon_nodes.nodeset.gc_vertices.astype(np.float32)).fliplr()


class ICONMultimeshNodes(BaseICONNodeBuilder):
    """Processor mesh based on an ICON grid."""

    icon_node_class = ICONMultiMesh


class ICONCellGridNodes(BaseICONNodeBuilder):
    """Data mesh based on an ICON grid."""

    icon_node_class = ICONCellDataGrid
