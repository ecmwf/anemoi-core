# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import uuid
from dataclasses import dataclass
from functools import cached_property

import netCDF4
import numpy as np
import scipy
from typeguard import typechecked
from typing_extensions import Self

from anemoi.graphs.generate.utils import convert_adjacency_matrix_to_list
from anemoi.graphs.generate.utils import convert_list_to_adjacency_matrix
from anemoi.graphs.generate.utils import selection_matrix

LOGGER = logging.getLogger(__name__)


@typechecked
class NodeSet:
    """Stores nodes on the unit sphere."""

    id_iter: int = itertools.count()  # unique ID for each object
    gc_vertices: np.ndarray  # geographical (lat/lon) coordinates [rad], shape [:,2]

    def __init__(self, lon: np.ndarray, lat: np.ndarray):
        self.gc_vertices = np.column_stack((lon, lat))
        self.id = uuid.uuid4()

    @property
    def num_vertices(self) -> int:
        return self.gc_vertices.shape[0]

    @cached_property
    def cc_vertices(self):
        """Cartesian coordinates [rad], shape [:,3]."""
        return self._gc_to_cartesian()

    def __add__(self, other: Self) -> Self:
        """concatenates two node sets."""
        gc_vertices = np.concatenate((self.gc_vertices, other.gc_vertices))
        return NodeSet(gc_vertices[:, 0], gc_vertices[:, 1])

    def __eq__(self, other: Self) -> bool:
        """Compares two node sets."""
        return self.id == other.id

    def _gc_to_cartesian(self, radius: float = 1.0) -> np.ndarray:
        """Returns Cartesian coordinates of the node set, shape [:,3]."""
        xyz = (
            radius * np.cos(lat_rad := self.gc_vertices[:, 1]) * np.cos(lon_rad := self.gc_vertices[:, 0]),
            radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        )
        return np.stack(xyz, axis=-1)


@typechecked
@dataclass
class EdgeID:
    """Stores additional categorical data for each edge (IDs for heterogeneous input)."""

    edge_id: np.ndarray
    num_classes: int

    def __add__(self, other: Self):
        """Concatenates two edge ID datasets."""
        assert self.num_classes == other.num_classes
        return EdgeID(
            edge_id=np.concatenate((self.edge_id, other.edge_id)),
            num_classes=self.num_classes,
        )


@typechecked
class GeneralGraph:
    """Stores edges for a given node set."""

    nodeset: NodeSet  # graph nodes
    edge_vertices: np.ndarray  # vertex indices for each edge, shape [:,2]

    def __init__(self, nodeset: NodeSet, bidirectional: bool, edge_vertices: np.ndarray):
        self.nodeset = nodeset
        # (optional) duplicate edges (bi-directional):
        if bidirectional:
            self.edge_vertices = np.concatenate([edge_vertices, np.fliplr(edge_vertices)])
        else:
            self.edge_vertices = edge_vertices

    @property
    def num_vertices(self) -> int:
        return self.nodeset.num_vertices

    @property
    def num_edges(self) -> int:
        return self.edge_vertices.shape[0]


@typechecked
class BipartiteGraph:
    """Graph defined on a pair of NodeSets."""

    nodeset: tuple[NodeSet, NodeSet]  # source and target node set
    edge_vertices: np.ndarray  # vertex indices for each edge, shape [:,2]
    edge_id: np.ndarray  # additional ID for each edge (markers for heterogeneous input)

    def __init__(
        self,
        nodeset: tuple[NodeSet, NodeSet],
        edge_vertices: np.ndarray,
        edge_id: EdgeID | None = None,
    ):
        self.nodeset = nodeset
        self.edge_vertices = edge_vertices
        self.edge_id = edge_id

    @property
    def num_edges(self) -> int:
        return self.edge_vertices.shape[0]

    def __add__(self, other: "BipartiteGraph"):
        """Concatenates two bipartite graphs that share a common target node set.
        Shifts the node indices of the second bipartite graph.
        """

        if not self.nodeset[1] == other.nodeset[1]:
            raise ValueError("Only bipartite graphs with common target node set can be merged.")
        shifted_edge_vertices = other.edge_vertices
        shifted_edge_vertices[:, 0] += self.nodeset[0].num_vertices
        # (Optional:) merge one-hot-encoded categorical data (`edge_id`)
        edge_id = None if None in (self.edge_id, other.edge_id) else self.edge_id + other.edge_id

        return BipartiteGraph(
            nodeset=(self.nodeset[0] + other.nodeset[0], self.nodeset[1]),
            edge_vertices=np.concatenate((self.edge_vertices, shifted_edge_vertices)),
            edge_id=edge_id,
        )


@typechecked
class ICONMultiMesh(GeneralGraph):
    """Reads vertices and topology from an ICON grid file; creates multi-mesh."""

    uuidOfHGrid: str
    max_level: int
    nodeset: NodeSet  # set of ICON grid vertices
    cell_vertices: np.ndarray

    def __init__(self, icon_grid_filename: str, max_level: int | None = None):

        # open file, representing the finest level
        LOGGER.debug(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:
            # read vertex coordinates
            vlon = read_coordinate_array(ncfile, "vlon", "vertex")
            vlat = read_coordinate_array(ncfile, "vlat", "vertex")

            edge_vertices_fine = np.asarray(ncfile.variables["edge_vertices"][:] - 1, dtype=np.int64).transpose()
            assert ncfile.variables["edge_vertices"].dimensions == ("nc", "edge")

            cell_vertices_fine = np.asarray(ncfile.variables["vertex_of_cell"][:] - 1, dtype=np.int64).transpose()
            assert ncfile.variables["vertex_of_cell"].dimensions == ("nv", "cell")

            reflvl_vertex = ncfile.variables["refinement_level_v"][:]
            assert ncfile.variables["refinement_level_v"].dimensions == ("vertex",)

            self.uuidOfHGrid = ncfile.uuidOfHGrid

        self.max_level = max_level if max_level is not None else reflvl_vertex.max()

        # generate edge-vertex relations for coarser levels:
        (edge_vertices, cell_vertices) = self._get_hierarchy_of_icon_edge_graphs(
            edge_vertices_fine=edge_vertices_fine,
            cell_vertices_fine=cell_vertices_fine,
            reflvl_vertex=reflvl_vertex,
        )
        # restrict edge-vertex list to multi_mesh level "max_level":
        if self.max_level < len(edge_vertices):
            (self.edge_vertices, self.cell_vertices, vlon, vlat) = self._restrict_multi_mesh_level(
                edge_vertices,
                cell_vertices,
                reflvl_vertex=reflvl_vertex,
                vlon=vlon,
                vlat=vlat,
            )
        # store vertices as a `NodeSet`:
        self.nodeset = NodeSet(vlon, vlat)
        # concatenate edge-vertex lists (= edges of the multi-level mesh):
        multi_mesh_edges = np.concatenate([edges for edges in self.edge_vertices], axis=0)
        # generate multi-mesh graph data structure:
        super().__init__(nodeset=self.nodeset, bidirectional=True, edge_vertices=multi_mesh_edges)

    def _restrict_multi_mesh_level(
        self,
        edge_vertices: list[np.ndarray],
        cell_vertices: np.ndarray,
        reflvl_vertex: np.ndarray,
        vlon: np.ndarray,
        vlat: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Creates a new mesh with only the vertices at the desired level."""

        num_vertices = reflvl_vertex.shape[0]
        vertex_mask = reflvl_vertex <= self.max_level
        vertex_glb2loc = np.zeros(num_vertices, dtype=int)
        vertex_glb2loc[vertex_mask] = np.arange(vertex_mask.sum())
        return (
            [vertex_glb2loc[vertices] for vertices in edge_vertices[: self.max_level + 1]],
            # cell_vertices: preserve negative indices (incomplete cells)
            np.where(cell_vertices >= 0, vertex_glb2loc[cell_vertices], cell_vertices),
            vlon[vertex_mask],
            vlat[vertex_mask],
        )

    def _get_hierarchy_of_icon_edge_graphs(
        self,
        edge_vertices_fine: np.ndarray,
        cell_vertices_fine: np.ndarray,
        reflvl_vertex: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Returns a list of edge-vertex relations (coarsest to finest level)."""

        edge_vertices = [edge_vertices_fine]  # list of edge-vertex relations (coarsest to finest level).

        num_vertices = reflvl_vertex.shape[0]
        # edge-to-vertex adjacency matrix with 2 non-zero entries per row:
        edge2vertex_matrix = convert_list_to_adjacency_matrix(edge_vertices_fine, num_vertices)
        # cell-to-vertex adjacency matrix with 3 non-zero entries per row:
        cell2vertex_matrix = convert_list_to_adjacency_matrix(cell_vertices_fine, num_vertices)
        vertex2vertex_matrix = edge2vertex_matrix.transpose() * edge2vertex_matrix
        vertex2vertex_matrix.setdiag(np.ones(num_vertices))  # vertices are self-connected

        selected_vertex_coarse = scipy.sparse.diags(np.ones(num_vertices), dtype=bool)

        # coarsen edge-vertex list from level `ilevel -> ilevel - 1`:
        for ilevel in reversed(range(1, reflvl_vertex.max() + 1)):
            LOGGER.debug(f"  edges[{ilevel}] = {edge_vertices[0].shape[0] : >9}")

            # define edge selection matrix (selecting only edges of which have
            # exactly one coarse vertex):
            #
            # get a boolean mask, matching all edges where one of its vertices
            # has refinement level index `ilevel`:
            ref_level_mask = reflvl_vertex[edge_vertices[0]] == ilevel
            edges_coarse = np.logical_xor(ref_level_mask[:, 0], ref_level_mask[:, 1])  # = bisected coarse edges
            idx_edge2edge = np.argwhere(edges_coarse).flatten()
            selected_edges = selection_matrix(idx_edge2edge, edges_coarse.shape[0])

            # define vertex selection matrix selecting only vertices of
            # level `ilevel`:
            idx_v_fine = np.argwhere(reflvl_vertex == ilevel).flatten()
            selected_vertex_fine = selection_matrix(idx_v_fine, num_vertices)
            # define vertex selection matrix, selecting only vertices of
            # level < `ilevel`, by successively removing `s_fine` from an identity matrix.
            selected_vertex_coarse.data[0][idx_v_fine] = False

            # create an adjacency matrix which links each fine level
            # vertex to its two coarser neighbor vertices:
            vertex2vertex_fine2coarse = selected_vertex_fine * vertex2vertex_matrix * selected_vertex_coarse
            # remove rows that have only one non-zero entry
            # (corresponding to incomplete parent triangles in LAM grids):
            csum = vertex2vertex_fine2coarse * np.ones((vertex2vertex_fine2coarse.shape[1], 1))
            selected_vertex2vertex = selection_matrix(
                np.argwhere(csum == 2).flatten(), vertex2vertex_fine2coarse.shape[0]
            )
            vertex2vertex_fine2coarse = selected_vertex2vertex * vertex2vertex_fine2coarse

            # then construct the edges-to-parent-vertex adjacency matrix:
            parent_edge_vertices = selected_edges * edge2vertex_matrix * vertex2vertex_fine2coarse
            # again, we have might have selected edges within
            # `selected_edges` which are part of an incomplete parent edge
            # (LAM case). We filter these here:
            csum = parent_edge_vertices * np.ones((parent_edge_vertices.shape[1], 1))
            selected_edge2edge = selection_matrix(np.argwhere(csum == 2).flatten(), parent_edge_vertices.shape[0])
            parent_edge_vertices = selected_edge2edge * parent_edge_vertices

            # note: the edges-vertex adjacency matrix still has duplicate
            # rows, since two child edges have the same parent edge.
            edge_vertices_coarse = convert_adjacency_matrix_to_list(parent_edge_vertices, ncols_per_row=2)
            edge_vertices.insert(0, edge_vertices_coarse)

            # store cell-to-vert adjacency matrix
            if ilevel > self.max_level:
                cell2vertex_matrix = cell2vertex_matrix * vertex2vertex_fine2coarse
                # similar to the treatment above, we need to handle
                # coarse LAM cells which are incomplete.
                csum = cell2vertex_matrix * np.ones((cell2vertex_matrix.shape[1], 1))
                selected_cell2cell = selection_matrix(np.argwhere(csum == 3).flatten(), cell2vertex_matrix.shape[0])
                cell2vertex_matrix = selected_cell2cell * cell2vertex_matrix

            # replace edge-to-vertex and vert-to-vert adjacency matrices (for next level):
            if ilevel > 1:
                vertex2vertex_matrix = selected_vertex_coarse * vertex2vertex_matrix * vertex2vertex_fine2coarse
                edge2vertex_matrix = convert_list_to_adjacency_matrix(edge_vertices_coarse, num_vertices)

        # Fine-level cells outside of multi-mesh (LAM boundary)
        # correspond to empty rows in the adjacency matrix. We
        # substitute these by three additional, non-existent vertices:
        csum = 3 - cell2vertex_matrix * np.ones((cell2vertex_matrix.shape[1], 1))
        nvmax = cell2vertex_matrix.shape[1]
        cell2vertex_matrix = scipy.sparse.csr_matrix(scipy.sparse.hstack((cell2vertex_matrix, csum, csum, csum)))

        # build a list of cell-vertices [1..num_cells,1..3] for all
        # fine-level cells:
        cell_vertices = convert_adjacency_matrix_to_list(cell2vertex_matrix, remove_duplicates=False, ncols_per_row=3)

        # finally, translate non-existent vertices into "-1" indices:
        cell_vertices = np.where(
            cell_vertices >= nvmax,
            -np.ones(cell_vertices.shape, dtype=np.int32),
            cell_vertices,
        )

        return (edge_vertices, cell_vertices)


@typechecked
class ICONCellDataGrid(BipartiteGraph):
    """Reads cell locations from an ICON grid file; builds grid-to-mesh edges based on ICON topology."""

    uuidOfHGrid: str
    nodeset: NodeSet  # set of ICON cell circumcenters
    max_level: int
    select_c: np.ndarray

    def __init__(
        self,
        icon_grid_filename: str,
        multi_mesh: ICONMultiMesh | None = None,
        max_level: int | None = None,
    ):
        # open file, representing the finest level
        LOGGER.debug(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:
            # read cell circumcenter coordinates
            clon = read_coordinate_array(ncfile, "clon", "cell")
            clat = read_coordinate_array(ncfile, "clat", "cell")

            reflvl_cell = ncfile.variables["refinement_level_c"][:]
            assert ncfile.variables["refinement_level_c"].dimensions == ("cell",)

            self.uuidOfHGrid = ncfile.uuidOfHGrid

        if max_level is not None:
            self.max_level = max_level
        else:
            self.max_level = reflvl_cell.max()

        # restrict to level `max_level`:
        self.select_c = np.argwhere(reflvl_cell <= self.max_level)
        # generate source grid node set:
        self.nodeset = NodeSet(clon[self.select_c], clat[self.select_c])

        if multi_mesh is not None:
            # generate edges between source grid nodes and multi-mesh nodes:
            edge_vertices = self._get_grid2mesh_edges(self.select_c, multi_mesh=multi_mesh)
            super().__init__((self.nodeset, multi_mesh.nodeset), edge_vertices)

    def _get_grid2mesh_edges(self, select_c: np.ndarray, multi_mesh: ICONMultiMesh) -> np.ndarray:
        """Create "grid-to-mesh" edges, ie. edges from (clat,clon) to the
        vertices of the multi-mesh.
        """

        num_cells = select_c.shape[0]
        num_vertices_per_cell = multi_mesh.cell_vertices.shape[1]
        src_list = np.kron(np.arange(num_cells), np.ones((1, num_vertices_per_cell), dtype=np.int64)).flatten()
        dst_list = multi_mesh.cell_vertices[select_c[:, 0], :].flatten()
        edge_vertices = np.stack((src_list, dst_list), axis=1, dtype=np.int64)
        return edge_vertices


# -------------------------------------------------------------


@typechecked
def read_coordinate_array(ncfile, arrname: str, dimname: str) -> np.ndarray:
    """Auxiliary routine, reading a coordinate array, checking consistency."""
    arr = ncfile.variables[arrname][:]
    assert ncfile.variables[arrname].dimensions == (dimname,)
    assert ncfile.variables[arrname].units == "radian"
    # netCDF4 returns all variables as numpy.ma.core.MaskedArray.
    # -> convert to regular arrays
    assert not arr.mask.any(), f"There are missing values in {arrname}"
    return arr.data
