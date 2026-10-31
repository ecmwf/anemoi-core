# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The hidden nodes the semi-Lagrangian processors run on.

A geometry answers everything about the hidden nodes that does not change between forward passes:
where they are, how they are laid out over the GPUs, and which spatial mixer, learned bias field
and departure-point sampler fit them. ``ADRProcessor`` and ``FlowersProcessor`` build one from the
graph and share it with every block, so the blocks and the steps work the same way on a regular
latitude-longitude grid (``RegularLatLonGeometry``) and on any mesh (``MeshGeometry``).
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data.storage import NodeStorage
from torch_geometric.typing import Adj

from anemoi.models.distributed.halo import HaloInfoCache
from anemoi.models.distributed.khop_edges import ensure_edges_are_dst_sorted
from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.adr.layers import GraphLowRankBias
from anemoi.models.layers.adr.layers import GraphSpatialMixer
from anemoi.models.layers.adr.layers import LatLonLowRankBias
from anemoi.models.layers.adr.layers import LatLonSpatialMixer
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.layout import Layout
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.models.layers.layout import latlon_grid_layout
from anemoi.models.layers.layout import latlon_to_unit_vector
from anemoi.models.layers.semi_lagrangian import GridSampler
from anemoi.models.layers.semi_lagrangian import KNNSampler
from anemoi.models.layers.semi_lagrangian import NearestNodeSearch
from anemoi.models.layers.semi_lagrangian import SemiLagrangianWarp
from anemoi.models.layers.semi_lagrangian import nearest_node_distance
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class LatentGeometry(nn.Module, ABC):
    """The hidden nodes of a semi-Lagrangian processor, and the pieces that depend on them.

    The processor builds one geometry and shares it with every block, so the node coordinates are
    held once rather than once per layer. It is the factory for the three geometry-specific pieces
    of the steps, so the steps themselves need not know which geometry they run on:

    - ``spatial_mixer``: looks at the neighbours of each point and mixes the channels;
    - ``position_bias``: a learned field over the globe, added at each point;
    - ``sampler``: reads a value at a departure point.

    ``bind`` opens one forward pass and returns the ``Layout`` that the blocks are given. A
    geometry holds no parameters, only non-persistent buffers, so sharing one instance across the
    blocks adds nothing to the state dict.
    """

    num_nodes: int

    @abstractmethod
    def bind(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        **runtime,
    ) -> Layout:
        """Open a forward pass on the state x, of shape (batch * local nodes, channels)."""

    @abstractmethod
    def spatial_mixer(self, in_channels: int, out_channels: int, layer_kernels: DotDict) -> nn.Module:
        """A stencil over the neighbours of each point, then a linear mix of the channels."""

    @abstractmethod
    def position_bias(self, num_channels: int, layer_kernels: DotDict) -> nn.Module:
        """A learned field over the globe, added to the state at each point."""

    @abstractmethod
    def sampler(self, num_heads: int, cartesian_displacement: bool = False) -> nn.Module:
        """Reads the value of each group of channels at its departure point."""

    def warp(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        layer_kernels: DotDict,
        cartesian_displacement: bool = False,
    ) -> SemiLagrangianWarp:
        """The FLOWERS warp on this geometry."""
        return SemiLagrangianWarp(
            sampler=self.sampler(num_heads, cartesian_displacement=cartesian_displacement),
            in_channels=in_channels,
            out_channels=out_channels,
            layer_kernels=layer_kernels,
        )


class RegularLatLonGeometry(LatentGeometry):
    """Hidden nodes on a regular latitude-longitude grid, as in PARADIS.

    The grid has ``nlat x nlon`` points at cell centres. Latitudes run from north to south,
    ``90 - (i + 0.5) * 180 / nlat`` degrees, and longitudes run east from zero,
    ``j * 360 / nlon`` degrees, so no point lies on a pole. Build such nodes with
    ``anemoi.graphs.nodes.RectilinearNodes``, without ``include_poles``, and give them a
    ``anemoi.graphs.nodes.attributes.RectilinearLayoutIndex`` attribute, which holds the row and
    column of each node and lets the processor work whatever order the nodes are stored in.

    The mixer is a square stencil per channel that continues across the poles, the bias field is
    built from latitude and longitude profiles, and values are read at departure points with
    ``F.grid_sample``.
    """

    def __init__(
        self,
        *,
        hidden_nodes: NodeStorage,
        layout_node_attribute_name: str,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs  # edge_dim, which a grid does not use
        nlat, nlon, grid_index = latlon_grid_layout(hidden_nodes, layout_node_attribute_name)
        if nlon % 2 != 0:
            raise ValueError(f"nlon must be even so that the grid continues across the poles, got {nlon}.")

        self.nlat = nlat
        self.nlon = nlon
        self.num_nodes = nlat * nlon
        self.kernel_size = kernel_size
        self.interpolation = interpolation
        self.bias_rank = bias_rank
        self.bias_base_maps = bias_base_maps

        # Flat grid position of each node, when the nodes are not stored row by row.
        self.register_buffer("grid_index", grid_index, persistent=False)
        lat, lon = latlon_cell_centres(nlat, nlon)
        lat, lon = lat.view(nlat, nlon), lon.view(nlat, nlon)
        self.register_buffer("lat", lat.float(), persistent=False)
        self.register_buffer("lon", lon.float(), persistent=False)
        self.register_buffer("position", latlon_to_unit_vector(lat, lon).float(), persistent=False)

    def bind(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        **runtime,
    ) -> LatLonLayout:
        del x, runtime  # a grid needs no edges
        return LatLonLayout(self, batch_size, shard_info.nodes, model_comm_group)

    def spatial_mixer(self, in_channels: int, out_channels: int, layer_kernels: DotDict) -> nn.Module:
        return LatLonSpatialMixer(in_channels, out_channels, layer_kernels, self.kernel_size)

    def position_bias(self, num_channels: int, layer_kernels: DotDict) -> nn.Module:
        return LatLonLowRankBias(
            num_channels,
            self.nlat,
            self.nlon,
            layer_kernels,
            rank=self.bias_rank,
            num_base_maps=self.bias_base_maps,
        )

    def sampler(self, num_heads: int, cartesian_displacement: bool = False) -> GridSampler:
        return GridSampler(
            num_heads=num_heads,
            interpolation=self.interpolation,
            cartesian_displacement=cartesian_displacement,
        )


class MeshGeometry(LatentGeometry, HaloInfoCache):
    """Hidden nodes anywhere on the sphere, with the hidden-to-hidden edges of the graph.

    For any hidden mesh: icosahedral, HEALPix, a reduced Gaussian grid, a limited area. Only the
    node coordinates and the edges are used, so nothing has to be laid out in rows and columns.

    The mixer is a block over the edges (``mixer_block``, see ``GraphSpatialMixer``), the bias
    field is built from profiles evaluated at the node coordinates, and values are read at
    departure points from the ``num_neighbours`` nearest nodes. Those are found with
    ``knn_backend``: ``"lookup"`` indexes the nodes once in the exact lookup table of anemoi-graphs
    and reads each departure point's few candidates from it (see ``NearestNodeSearch``), falling
    back to brute force on a mesh that does not cover the globe; ``"brute_force"`` compares every
    departure point with every node. ``bind`` prepares the edges once per forward: sorted by
    destination node, split across GPUs, and with the halo exchange plan when the mixer is the
    graph transformer.
    """

    def __init__(
        self,
        *,
        hidden_nodes: NodeStorage,
        edge_dim: int,
        mixer_block: str = "edge_conv",
        mixer_kwargs: Optional[dict] = None,
        num_neighbours: int = 4,
        smoothing: float = 0.01,
        knn_backend: str = "lookup",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        bias_lat_degree: int = 32,
        bias_lon_wavenumber: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs  # layout_node_attribute_name, which a mesh does not use
        coords = hidden_nodes["x"].detach().to(device="cpu", dtype=torch.float64)
        lat, lon = coords[:, 0], coords[:, 1]
        self.num_nodes = lat.shape[0]
        self.register_buffer("lat", lat.float(), persistent=False)
        self.register_buffer("lon", lon.float(), persistent=False)
        self.register_buffer("position", latlon_to_unit_vector(lat, lon).float(), persistent=False)
        self.register_buffer("spacing", nearest_node_distance(self.position), persistent=False)

        self.edge_dim = edge_dim
        self.mixer_block = mixer_block
        self.mixer_kwargs = dict(mixer_kwargs or {})
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.nearest_node_search = NearestNodeSearch(knn_backend)
        self.bias_rank = bias_rank
        self.bias_base_maps = bias_base_maps
        self.bias_lat_degree = bias_lat_degree
        self.bias_lon_wavenumber = bias_lon_wavenumber

        # Only the graph transformer reads a halo; the other mixers gather what they need.
        self.shard_strategy = "edges" if mixer_block == "transformer" else "none"
        self._cached_halo_info = None
        self._cached_halo_cache_specs = None

    def bind(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        *,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        edges_are_dst_sorted: bool = True,
        **runtime,
    ) -> GraphLayout:
        del runtime
        if edge_attr is None or edge_index is None:
            raise ValueError(
                f"{type(self).__name__} needs the hidden-to-hidden edges. Give the processor a graph with "
                "hidden-to-hidden edges, or use RegularLatLonGeometry on a latitude-longitude hidden grid."
            )

        size = sum(shard_info.nodes) if shard_info.nodes_are_sharded() else x.size(0)
        edge_attr, edge_index = ensure_edges_are_dst_sorted(
            edge_attr,
            edge_index,
            num_dst=size,
            edges_are_sharded=shard_info.edges_are_sharded(),
            model_comm_group=model_comm_group,
            edges_are_dst_sorted=edges_are_dst_sorted,
        )
        if not shard_info.edges_are_sharded():
            edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(
                edge_attr, edge_index, size, size, model_comm_group, edges_are_dst_sorted=edges_are_dst_sorted
            )
            shard_info = GraphShardInfo(nodes=shard_info.nodes, edges=edge_shard_sizes)

        halo_info = self._get_or_build_cached_halo_info(x, edge_index, shard_info, batch_size, model_comm_group)
        return GraphLayout(self, batch_size, shard_info, edge_attr, edge_index, model_comm_group, halo_info)

    def spatial_mixer(self, in_channels: int, out_channels: int, layer_kernels: DotDict) -> nn.Module:
        return GraphSpatialMixer(
            in_channels,
            out_channels,
            self.edge_dim,
            layer_kernels,
            block=self.mixer_block,
            **self.mixer_kwargs,
        )

    def position_bias(self, num_channels: int, layer_kernels: DotDict) -> nn.Module:
        return GraphLowRankBias(
            num_channels,
            layer_kernels,
            rank=self.bias_rank,
            num_base_maps=self.bias_base_maps,
            lat_degree=self.bias_lat_degree,
            lon_wavenumber=self.bias_lon_wavenumber,
        )

    def sampler(self, num_heads: int, cartesian_displacement: bool = False) -> KNNSampler:
        return KNNSampler(
            num_heads=num_heads,
            num_neighbours=self.num_neighbours,
            smoothing=self.smoothing,
            cartesian_displacement=cartesian_displacement,
        )

    def nearest_nodes(self, queries: Tensor, k: int) -> Tensor:
        """Indices, of shape (queries, k), of the k nodes nearest to each query, nearest first.

        queries (q, 3) are 3D unit vectors on the device of the geometry; the nodes are indexed
        on that device the first time they are searched there.
        """
        return self.nearest_node_search(self.position.to(queries.dtype), queries, k)
