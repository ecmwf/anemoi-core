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
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.utils.config import DotDict


class ICONNodes(BaseNodeBuilder):
    """ICON grid (cell and vertex locations)."""

    def __init__(self, name: str, grid_filename: str, max_level_multimesh: int, max_level_dataset: int) -> None:
        self.grid_filename = grid_filename

        self.multi_mesh = ICONMultiMesh(self.grid_filename, max_level=max_level_multimesh)
        self.cell_grid = ICONCellDataGrid(self.grid_filename, self.multi_mesh, max_level=max_level_dataset)

        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        return torch.from_numpy(self.multi_mesh.nodeset.gc_vertices.astype(np.float32)).fliplr()

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        graph[self.name]["_grid_filename"] = self.grid_filename
        graph[self.name]["_multi_mesh"] = self.multi_mesh
        graph[self.name]["_cell_grid"] = self.cell_grid
        return super().register_attributes(graph, config)


class ICONTopologicalBaseNodeBuilder(BaseNodeBuilder):
    """Base class for data mesh or processor mesh based on an ICON grid.

    Parameters
    ----------
    name : str
        key for the nodes in the HeteroData graph object.
    icon_mesh : str
        key corresponding to the ICON mesh (cells and vertices).
    """

    def __init__(self, name: str, icon_mesh: str) -> None:
        self.icon_mesh = icon_mesh
        super().__init__(name)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        """Update the graph with new nodes."""
        self.icon_sub_graph = graph[self.icon_mesh][self.sub_graph_address]
        return super().update_graph(graph, attrs_config)


class ICONMultimeshNodes(ICONTopologicalBaseNodeBuilder):
    """Processor mesh based on an ICON grid."""

    def __init__(self, name: str, icon_mesh: str) -> None:
        self.sub_graph_address = "_multi_mesh"
        super().__init__(name, icon_mesh)

    def get_coordinates(self) -> torch.Tensor:
        return torch.from_numpy(self.icon_sub_graph.nodeset.gc_vertices.astype(np.float32)).fliplr()


class ICONCellGridNodes(ICONTopologicalBaseNodeBuilder):
    """Data mesh based on an ICON grid."""

    def __init__(self, name: str, icon_mesh: str) -> None:
        self.sub_graph_address = "_cell_grid"
        super().__init__(name, icon_mesh)

    def get_coordinates(self) -> torch.Tensor:
        return torch.from_numpy(self.icon_sub_graph.nodeset[0].gc_vertices.astype(np.float32)).fliplr()
