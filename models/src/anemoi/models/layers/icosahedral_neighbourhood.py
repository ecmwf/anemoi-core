# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on the refined icosahedron (grid family ``icosahedral``).

Every point attends to the points within ``radius`` hops. The lookup tables are in
:mod:`anemoi.models.layers.icosahedral_grid`, the kernels in :mod:`anemoi.models.triton.icosahedral_attention`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Callable
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
    """Neighbourhood of the refined icosahedron: the points within ``radius`` hops.

    Self attention only, nodes in diamond order (anemoi-graphs ``TriNodes`` with
    ``ordering: diamonds``). Configured as::

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
        """Build the neighbourhood from its config and the node coordinates.

        Parameters
        ----------
        config : dict
            ``grid: icosahedral``, ``radius`` (hops) and optional ``backend`` (``"triton"``, default,
            or ``"sdpa"``).
        key_coords : Tensor
            Latitude and longitude of the nodes in radians, ``(num_nodes, 2)``.
        query_coords : Tensor, optional
            Must be None: self attention only.

        Returns
        -------
        IcosahedralGridNeighbourhood
            The lookup tables and the backend.
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
        """Tables for all points on one GPU."""
        return self.tables.shard()

    def attention_module(self) -> IcosahedralAttentionWrapper:
        return IcosahedralAttentionWrapper(self, self.whole_shard)

    def layer_split(self, cpu_offload: bool) -> Optional[IcosahedralShards]:
        """Shards with a halo when the points are split across GPUs; one GPU runs all points at once."""
        return None if cpu_offload else IcosahedralShards(self)


def check_diamond_order(coords: Tensor, layout: IcosahedralDiamonds) -> None:
    """Check that the nodes are the refined icosahedron in diamond order.

    In diamond order all lattice edges are about equally long; in any other order some span the globe.
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
    """Neighbourhood attention on the refined icosahedron for the queries and keys of one shard."""

    def __init__(self, neighbourhood: IcosahedralGridNeighbourhood, shard: IcosahedralShard) -> None:
        super().__init__(neighbourhood)
        self.shard = shard

    def build_mask(self, device: torch.device) -> Tensor:
        queries = np.arange(self.shard.own.start, self.shard.own.stop)
        reach = self.neighbourhood.tables.within_radius(queries)
        keys = np.concatenate([queries, self.shard.halo])
        return torch.as_tensor(reach[:, keys].toarray(), device=device)

    def kernel_path(self) -> str:
        return self.neighbourhood.grid_family.self_attention

    def run_kernel(self, kernel: Callable, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return kernel(query, key, value, self.shard)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
    ) -> Tensor:
        self.check_lengths(query, key, self.shard.num_queries, self.shard.num_keys)
        return self.attend(query, key, value, dropout_p)


@dataclass(frozen=True, eq=False)
class IcosahedralShardPlan:
    """Plan of one GPU: its own points as one band, and the halo it fetches."""

    band: NeighbourhoodBand
    halo_info: HaloInfo


@dataclass(eq=False)
class HaloWork(GpuWork):
    """Work of one GPU in a layer: one band of its own points, with those points and the halo as keys."""

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
    """Icosahedral attention split across GPUs.

    Each GPU owns a run of points in diamond order and fetches its halo before each layer. Every GPU
    works out all halos, so they agree on the exchange without communicating. Plans are cached per split.
    """

    def __init__(self, neighbourhood: IcosahedralGridNeighbourhood) -> None:
        self.neighbourhood = neighbourhood
        self._plans: dict[tuple, IcosahedralShardPlan] = {}

    def __getstate__(self) -> dict:
        # Plans hold device tensors; they are rebuilt after loading.
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
        """Plan of GPU ``rank``, with ``shard_sizes`` points per GPU in rank order."""
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
