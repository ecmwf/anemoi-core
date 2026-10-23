# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on the refined icosahedron: every point attends to the points within ``radius`` hops.

The grid family ``icosahedral`` of :data:`anemoi.models.layers.neighbourhood_attention.GRID_FAMILIES`.
The lookup tables and exception lists that find each point's neighbours are in
:mod:`anemoi.models.layers.icosahedral_grid`, the kernels in
:mod:`anemoi.models.triton.icosahedral_attention`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.graphs.generate.icosahedral_diamonds import IcosahedralDiamonds
from anemoi.models.distributed.graph import halo_exchange
from anemoi.models.distributed.halo import HaloInfo
from anemoi.models.layers.icosahedral_grid import IcosahedralNeighbourhood
from anemoi.models.layers.icosahedral_grid import IcosahedralShard
from anemoi.models.layers.neighbourhood_attention import GpuWork
from anemoi.models.layers.neighbourhood_attention import LayerSplit
from anemoi.models.layers.neighbourhood_attention import Neighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttention
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodBand
from anemoi.models.layers.neighbourhood_attention import check_settings


@dataclass(frozen=True, eq=False)
class IcosahedralGridNeighbourhood(Neighbourhood):
    """Neighbourhood attention on the refined icosahedron: every point attends to the points within ``radius`` hops.

    Self attention only, with the nodes stored in diamond order (anemoi-graphs ``TriNodes`` with
    ``ordering: diamonds``); see :mod:`anemoi.models.layers.icosahedral_grid`. Configured as::

        attention_implementation: neighbourhood
        neighbourhood:
          grid: icosahedral
          radius: 5
          backend: triton
    """

    tables: IcosahedralNeighbourhood
    backend: str
    family: str = "icosahedral"

    @classmethod
    def from_config(
        cls, config: dict, key_coords: Optional[Tensor], query_coords: Optional[Tensor] = None
    ) -> IcosahedralGridNeighbourhood:
        """Set up icosahedral neighbourhood attention from its configuration and the node coordinates.

        Parameters
        ----------
        config : dict
            ``grid: icosahedral``, ``radius`` (hops) and optionally ``backend`` (``"triton"``,
            the default, or ``"sdpa"``).
        key_coords : Tensor
            Latitude and longitude of the nodes in radians, shape ``(num_nodes, 2)``.
        query_coords : Tensor, optional
            Must be left out: the icosahedral kernels only do self attention.

        Returns
        -------
        IcosahedralGridNeighbourhood
            The lookup tables of the mesh and the backend.
        """
        _, backend = check_settings(config, {"grid", "radius", "backend"}, key_coords, query_coords)
        if "radius" not in config:
            raise ValueError("Icosahedral neighbourhood attention needs a 'radius', the number of hops to attend over.")
        radius = int(config["radius"])
        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}.")
        layout = IcosahedralDiamonds.from_num_points(key_coords.shape[0])
        check_diamond_order(key_coords, layout)
        return cls(IcosahedralNeighbourhood(layout.resolution, radius), backend)

    @cached_property
    def whole_shard(self) -> IcosahedralShard:
        """The tables and lists for all points on one GPU."""
        return self.tables.shard()

    def attention_module(self) -> IcosahedralAttentionWrapper:
        return IcosahedralAttentionWrapper(self, self.whole_shard)

    def layer_split(self, cpu_offload: bool) -> Optional[IcosahedralShards]:
        """GPU shards with a halo, when the points are split across GPUs; one GPU runs all points at once."""
        return None if cpu_offload else IcosahedralShards(self)


def check_diamond_order(coords: Tensor, layout: IcosahedralDiamonds) -> None:
    """Check that the nodes at ``coords`` (radians) are the refined icosahedron in diamond order.

    In diamond order the points joined by the lattice are neighbours on the sphere, so all lattice
    edges are about equally long; in any other order some of them span the globe.
    """
    lat, lon = coords[:, 0].double(), coords[:, 1].double()
    xyz = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=1)
    triangles = torch.as_tensor(layout.triangles(), device=coords.device)
    ends = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    lengths = torch.acos((xyz[ends[:, 0]] * xyz[ends[:, 1]]).sum(-1).clamp(-1.0, 1.0))
    if not bool(lengths.max() < 2 * lengths.min()):
        raise ValueError(
            f"The {coords.shape[0]} nodes are not a refined icosahedron in diamond order; build them with "
            "anemoi-graphs TriNodes and 'ordering: diamonds'."
        )


class IcosahedralAttentionWrapper(NeighbourhoodAttention):
    """Neighbourhood attention on the refined icosahedron for the queries and keys of one shard.

    Takes ``(batch, heads, points, head_dim)`` queries of the shard's own points and keys and
    values of its key points (see :class:`anemoi.models.layers.icosahedral_grid.IcosahedralShard`).

    ``"triton"``
        The kernels of :mod:`anemoi.models.triton.icosahedral_attention`. GPU only; the head
        dimension must be a power of two of at least 16. No dropout.
    ``"sdpa"``
        ``scaled_dot_product_attention`` with the full ``queries x keys`` boolean mask, for small
        grids and tests.
    """

    def __init__(self, neighbourhood: IcosahedralGridNeighbourhood, shard: IcosahedralShard) -> None:
        super().__init__(neighbourhood)
        self.shard = shard
        self._kernel = None
        self._masks: dict[torch.device, Tensor] = {}

    def __getstate__(self) -> dict:
        # The masks are built again on first use, so they are left out when the model is saved.
        state = self.__dict__.copy()
        state["_masks"] = {}
        state["_kernel"] = None
        return state

    def _mask_on(self, device: torch.device) -> Tensor:
        if device not in self._masks:
            shard = self.shard
            queries = np.arange(shard.own.start, shard.own.stop)
            reach = self.neighbourhood.tables.within_radius(queries)
            keys = np.concatenate([queries, shard.halo])
            self._masks[device] = torch.as_tensor(reach[:, keys].toarray(), device=device)
        return self._masks[device]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
    ) -> Tensor:
        if query.shape[-2] != self.shard.num_queries or key.shape[-2] != self.shard.num_keys:
            raise ValueError(
                f"Icosahedral attention is set up for {self.shard.num_queries} queries and {self.shard.num_keys} "
                f"keys, got {query.shape[-2]} and {key.shape[-2]}."
            )
        if self.backend == "sdpa":
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=self._mask_on(query.device), dropout_p=dropout_p
            )
        if self._kernel is None:
            self._kernel = self.neighbourhood.grid_family.load_self_attention()
        return self._kernel(query, key, value, self.shard)


@dataclass(frozen=True, eq=False)
class IcosahedralShardPlan:
    """The work of one GPU in icosahedral attention: its own points as one band, and the halo it fetches."""

    band: NeighbourhoodBand
    halo_info: HaloInfo


@dataclass(eq=False)
class HaloWork(GpuWork):
    """The work of one GPU in a layer: one band of its own points, whose keys are those points and the halo."""

    plan: IcosahedralShardPlan
    model_comm_group: ProcessGroup

    @property
    def bands(self) -> list[NeighbourhoodBand]:
        return [self.plan.band]

    @property
    def num_own_queries(self) -> int:
        return self.plan.halo_info.num_local_dst_nodes

    def gather_keys(self, x: Tensor) -> Tensor:
        return halo_exchange(x, self.plan.halo_info, self.model_comm_group)[None]

    def keys_of(self, keys: Tensor, band: NeighbourhoodBand) -> Tensor:
        return keys

    def keep(self, out: Optional[Tensor], band_out: Tensor, band: NeighbourhoodBand) -> Tensor:
        return band_out


class IcosahedralShards(LayerSplit):
    """The plans of icosahedral attention for each way of splitting the points across GPUs.

    Each GPU owns a run of consecutive points in diamond order. Before each layer it fetches the
    points of other GPUs that its queries reach (its halo); attention itself needs no
    communication. Every GPU works out the halos of all GPUs, so they agree on what is sent where
    without talking to each other. The plans are worked out once for each way of splitting.
    """

    def __init__(self, neighbourhood: IcosahedralGridNeighbourhood) -> None:
        self.neighbourhood = neighbourhood
        self._plans: dict[tuple, IcosahedralShardPlan] = {}

    def __getstate__(self) -> dict:
        # The plans hold tensors on a device; they are worked out again where the model is loaded.
        state = self.__dict__.copy()
        state["_plans"] = {}
        return state

    def used(self, sharded: bool) -> bool:
        return sharded

    def work(
        self,
        batch_size: int,
        device: torch.device,
        model_comm_group: Optional[ProcessGroup] = None,
        query_shard_sizes: Optional[list[int]] = None,
        key_shard_sizes: Optional[list[int]] = None,
    ) -> HaloWork:
        if model_comm_group is None or query_shard_sizes is None:
            raise ValueError("Icosahedral attention works in shards only when the points are split across GPUs.")
        if batch_size != 1:
            raise ValueError("Only batch size 1 is supported when the points are split across GPUs.")
        return HaloWork(self.plan(query_shard_sizes, model_comm_group.rank(), device), model_comm_group)

    def plan(self, shard_sizes: list[int], rank: int, device: torch.device) -> IcosahedralShardPlan:
        """The plan of GPU ``rank`` when GPU ``r`` owns ``shard_sizes[r]`` points, in rank order."""
        cache_key = (tuple(shard_sizes), rank, device)
        if cache_key not in self._plans:
            self._plans[cache_key] = self._make_plan(list(shard_sizes), rank, device)
        return self._plans[cache_key]

    def _make_plan(self, shard_sizes: list[int], rank: int, device: torch.device) -> IcosahedralShardPlan:
        tables = self.neighbourhood.tables
        if sum(shard_sizes) != tables.num_points:
            raise ValueError(f"The shard sizes {shard_sizes} do not add up to the {tables.num_points} points.")
        starts = [0, *itertools.accumulate(shard_sizes)]
        runs = [slice(start, stop) for start, stop in zip(starts[:-1], starts[1:])]
        shards = [tables.shard(run) for run in runs]
        own, shard = runs[rank], shards[rank]

        # The halo is in increasing order, so the points from each GPU come in rank order.
        def from_run(points: np.ndarray, run: slice) -> np.ndarray:
            return points[(points >= run.start) & (points < run.stop)]

        send_indices = tuple(
            torch.as_tensor(from_run(other.halo, own) - own.start, dtype=torch.long, device=device) for other in shards
        )
        recv_counts = tuple(len(from_run(shard.halo, run)) for run in runs)
        halo_info = HaloInfo(
            num_local_src_nodes=shard.num_queries,
            num_local_dst_nodes=shard.num_queries,
            num_halo_nodes=len(shard.halo),
            send_indices=send_indices,
            recv_counts=recv_counts,
            edge_index_local=torch.empty((2, 0), dtype=torch.long, device=device),
        )
        key_points = np.concatenate([np.arange(own.start, own.stop), shard.halo])
        band = NeighbourhoodBand(
            query_points=own,
            key_points=torch.as_tensor(key_points, dtype=torch.long, device=device),
            queries_in_keys=slice(0, shard.num_queries),
            attention=IcosahedralAttentionWrapper(self.neighbourhood, shard),
        )
        return IcosahedralShardPlan(band=band, halo_info=halo_info)
