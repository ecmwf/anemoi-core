# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on reduced grids (grid families ``octahedral`` and ``healpix``).

A query attends to ``kernel_size[0]`` latitude rows centred on its own and, in each row, to the
``kernel_size[1]`` points nearest in longitude (see :mod:`anemoi.models.layers.reduced_grid`).
Queries and keys can be on the same grid (processor) or on two grids of one family (encoder,
decoder).

With ``num_bands > 1`` the queries are processed in bands of whole rows to save memory. When the
points are split across GPUs, each GPU processes the rows holding its own points and first fetches
the neighbouring points those rows need (see :class:`ShardPlan`).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.point_ranges import PointRangeExchange
from anemoi.models.distributed.point_ranges import PointRun
from anemoi.models.distributed.point_ranges import build_point_range_exchange
from anemoi.models.distributed.point_ranges import fetch_point_range
from anemoi.models.layers.neighbourhood_attention import GpuWork
from anemoi.models.layers.neighbourhood_attention import LayerSplit
from anemoi.models.layers.neighbourhood_attention import Neighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttention
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodBand
from anemoi.models.layers.neighbourhood_attention import check_settings
from anemoi.models.layers.neighbourhood_attention import grid_family
from anemoi.models.layers.neighbourhood_attention import load
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.reduced_grid import ReducedGridCrossNeighbourhoodMask
from anemoi.models.layers.reduced_grid import ReducedGridNeighbourhoodMask

LOGGER = logging.getLogger(__name__)


def is_octahedral(grid: ReducedGrid) -> bool:
    n = grid.num_rows // 2
    north = tuple(4 * i + 20 for i in range(n))
    return grid.num_rows == 2 * n and grid.row_lengths == north + north[::-1] and not grid.is_shifted


def is_healpix(grid: ReducedGrid) -> bool:
    nside = (grid.num_rows + 1) // 4
    return nside > 0 and grid.num_rows == 4 * nside - 1 and grid == ReducedGrid.healpix(nside)


def grid_from_coords(family: str, coords: Tensor) -> tuple[ReducedGrid, Optional[Tensor]]:
    """Find the grid formed by the nodes and check that it belongs to ``family``.

    Parameters
    ----------
    family : str
        A reduced-grid family.
    coords : Tensor
        Latitude and longitude of the nodes in radians, ``(num_nodes, 2)``.

    Returns
    -------
    tuple[ReducedGrid, Tensor | None]
        The grid, and the node order that sorts the nodes into it (node ``order[i]`` is grid point
        ``i``), or None if the nodes are already in grid order.
    """
    if grid_family(family).is_member is None:
        raise ValueError(f"'{family}' is not a family of reduced grids.")
    is_member = load(grid_family(family).is_member)
    grid, order = ReducedGrid.from_unordered_coords(coords)
    if not is_member(grid):
        raise ValueError(
            f"The {coords.shape[0]} nodes, in {grid.num_rows} latitude rows, do not form one of the "
            f"{grid_family(family).description}."
        )
    if torch.equal(order, torch.arange(order.numel(), device=order.device)):
        return grid, None
    return grid, order


def check_every_key_attended(query_grid: ReducedGrid, key_grid: ReducedGrid, kernel_size: tuple[int, int]) -> None:
    """Check that every key is read by some query in cross attention, and log the counts.

    If the key grid is much finer than the query grid, keys between two queries can be missed by
    both. In an encoder that data would never reach the processor.

    Raises
    ------
    ValueError
        If a key is read by no query. The message suggests a kernel size that reads them all.
    """
    rule = ReducedGridCrossNeighbourhoodMask(query_grid, key_grid, kernel_size)
    counts = rule.times_attended()
    keys = rule.keys_per_query()
    unread = int((counts == 0).sum())
    pair = (
        f"neighbourhood cross attention from {key_grid.num_points} key points ({key_grid.num_rows} rows) "
        f"to {query_grid.num_points} queries ({query_grid.num_rows} rows), kernel_size {list(kernel_size)}"
    )
    LOGGER.info(
        "%s: each query reads %d to %d keys, %.1f on average; each key point is read by %d to %d queries, "
        "%.1f on average.",
        pair,
        int(keys.min()),
        int(keys.max()),
        float(keys.float().mean()),
        int(counts.min()),
        int(counts.max()),
        float(counts.float().mean()),
    )
    if unread == 0:
        return
    suggestion = ""
    for side in range(1, key_grid.num_rows + 1, 2):
        larger = (max(kernel_size[0], side), max(kernel_size[1], side))
        if larger[0] > key_grid.num_rows:
            break
        if bool((ReducedGridCrossNeighbourhoodMask(query_grid, key_grid, larger).times_attended() > 0).all()):
            suggestion = f" kernel_size {list(larger)} reads every key point."
            break
    raise ValueError(
        f"{pair}: {unread} of {key_grid.num_points} key points ({100 * unread / key_grid.num_points:.1f}%) "
        f"are attended to by no query, so their values are never used.{suggestion}"
    )


@dataclass(frozen=True, eq=False)
class ReducedGridNeighbourhood(Neighbourhood):
    """Query and key grids of neighbourhood attention on reduced grids."""

    family: str
    kernel_size: tuple[int, int]
    backend: str
    query_grid: ReducedGrid
    key_grid: ReducedGrid
    query_order: Optional[Tensor] = None
    key_order: Optional[Tensor] = None
    is_self_attention: bool = False
    num_bands: int = 1

    @classmethod
    def from_config(
        cls,
        config: Optional[dict],
        key_coords: Optional[Tensor],
        query_coords: Optional[Tensor] = None,
    ) -> ReducedGridNeighbourhood:
        """Build the neighbourhood from its config and the node coordinates.

        Parameters
        ----------
        config : dict
            ``grid``, ``kernel_size`` (two odd numbers: latitude rows and points per row),
            optional ``backend`` (``"triton"`` or ``"sdpa"``, default ``"triton"``) and optional
            ``num_bands`` (bands of query rows processed one at a time, default 1).
        key_coords : Tensor
            Coordinates of the key nodes in radians, ``(num_keys, 2)``.
        query_coords : Tensor, optional
            Same for the query nodes; None for self attention.

        Returns
        -------
        ReducedGridNeighbourhood
            The grids, node orders and settings.
        """
        family, backend = check_settings(
            config, {"grid", "kernel_size", "backend", "num_bands"}, key_coords, query_coords
        )
        if "kernel_size" not in config:
            raise ValueError(f"Neighbourhood attention on {family} grids needs a kernel_size.")
        kernel_size = tuple(int(k) for k in config["kernel_size"])
        if len(kernel_size) != 2 or any(k <= 0 or k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be two positive odd numbers, got {config['kernel_size']}.")
        num_bands = int(config.get("num_bands", 1))
        if num_bands < 1:
            raise ValueError(f"num_bands must be at least 1, got {num_bands}.")

        key_grid, key_order = grid_from_coords(family, key_coords)
        if query_coords is None:
            query_grid, query_order, is_self_attention = key_grid, key_order, True
        else:
            query_grid, query_order = grid_from_coords(family, query_coords)
            check_every_key_attended(query_grid, key_grid, kernel_size)
            is_self_attention = False
        if num_bands > 1 and (query_order is not None or key_order is not None):
            raise ValueError(
                "Working through bands of rows (num_bands > 1) needs the nodes stored in grid order, row by "
                "row from north to south; build the nodes in that order, e.g. HEALPix in ring ordering."
            )
        return cls(
            family,
            kernel_size,
            backend,
            query_grid,
            key_grid,
            query_order,
            key_order,
            is_self_attention=is_self_attention,
            num_bands=num_bands,
        )

    @property
    def in_grid_order(self) -> bool:
        """Whether the nodes are stored in grid order, row by row from north to south."""
        return self.query_order is None and self.key_order is None

    def attention_module(self) -> ReducedGridAttentionWrapper:
        return ReducedGridAttentionWrapper(self)

    def layer_split(self, cpu_offload: bool) -> Optional[ReducedGridBands]:
        """Bands of whole rows, used when num_bands > 1 or the points are split across GPUs.

        Needs the nodes in grid order.
        """
        if not self.in_grid_order:
            return None
        if cpu_offload:
            if self.num_bands > 1:
                raise ValueError("Neighbourhood attention in bands (num_bands > 1) does not work with cpu_offload.")
            return None
        return ReducedGridBands(self)


def _gather_points(x: Tensor, order: Tensor) -> Tensor:
    """Reorder the points of ``(batch, heads, points, head_dim)`` ``x``.

    Gathers along the point dimension of the point-major layout, so whole rows of memory are moved.
    """
    return x.transpose(-3, -2).index_select(-3, order).transpose(-3, -2)


class _ReorderPoints(torch.autograd.Function):
    """Reorder the points; the backward pass gathers with the inverse permutation.

    A gather is much faster than the scatter-add autograd would use.
    """

    @staticmethod
    def forward(ctx, x: Tensor, order: Tensor, inverse: Tensor) -> Tensor:
        ctx.save_for_backward(inverse)
        return _gather_points(x, order)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        return _gather_points(grad, inverse), None, None


class ReducedGridAttentionWrapper(NeighbourhoodAttention):
    """Neighbourhood attention on reduced grids.

    Takes points in node order, reorders them to grid order if needed, and returns the output in
    node order.
    """

    def __init__(self, neighbourhood: ReducedGridNeighbourhood) -> None:
        super().__init__(neighbourhood)

        # Buffers so they follow the module to the GPU; not saved with the weights.
        self._reorder_queries = neighbourhood.query_order is not None
        self._reorder_keys = neighbourhood.key_order is not None
        if self._reorder_queries:
            self.register_buffer("query_order", neighbourhood.query_order, persistent=False)
            self.register_buffer("query_inverse", torch.argsort(neighbourhood.query_order), persistent=False)
        if self._reorder_keys:
            self.register_buffer("key_order", neighbourhood.key_order, persistent=False)
            self.register_buffer("key_inverse", torch.argsort(neighbourhood.key_order), persistent=False)

        if neighbourhood.is_self_attention:
            self.mask_rule = ReducedGridNeighbourhoodMask(neighbourhood.key_grid, neighbourhood.kernel_size)
        else:
            self.mask_rule = ReducedGridCrossNeighbourhoodMask(
                neighbourhood.query_grid, neighbourhood.key_grid, neighbourhood.kernel_size
            )

    def build_mask(self, device: torch.device) -> Tensor:
        q_idx = torch.arange(self.neighbourhood.query_grid.num_points, device=device)
        k_idx = torch.arange(self.neighbourhood.key_grid.num_points, device=device)
        return self.mask_rule.to(device)(q_idx[:, None], k_idx[None, :])

    def kernel_path(self) -> str:
        family = self.neighbourhood.grid_family
        return family.self_attention if self.neighbourhood.is_self_attention else family.cross_attention

    def run_kernel(self, kernel: Callable, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        nb = self.neighbourhood
        if nb.is_self_attention:
            return kernel(query, key, value, nb.key_grid, nb.kernel_size)
        return kernel(query, key, value, nb.query_grid, nb.key_grid, nb.kernel_size)

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
        self.check_lengths(query, key, self.neighbourhood.query_grid.num_points, self.neighbourhood.key_grid.num_points)

        if self._reorder_queries:
            query = _ReorderPoints.apply(query, self.query_order, self.query_inverse)
        if self._reorder_keys:
            key = _ReorderPoints.apply(key, self.key_order, self.key_inverse)
            value = _ReorderPoints.apply(value, self.key_order, self.key_inverse)

        out = self.attend(query, key, value, dropout_p)

        if self._reorder_queries:
            out = _ReorderPoints.apply(out, self.query_inverse, self.query_order)
        return out


def centre_rows(neighbourhood: ReducedGridNeighbourhood) -> Tensor:
    """Key row each query row is centred on: its own row in self attention, else the nearest in latitude."""
    query_grid, key_grid = neighbourhood.query_grid, neighbourhood.key_grid
    if neighbourhood.is_self_attention:
        return torch.arange(query_grid.num_rows, device=query_grid.row_starts.device)
    return query_grid.nearest_rows(key_grid)


def key_rows_reached(neighbourhood: ReducedGridNeighbourhood, first_row: int, end_row: int) -> tuple[int, int]:
    """Key rows ``(first, end)`` reached by the query rows ``first_row`` to ``end_row - 1``."""
    centre = centre_rows(neighbourhood)[first_row:end_row]
    radius = neighbourhood.kernel_size[0] // 2
    return max(int(centre.min()) - radius, 0), min(int(centre.max()) + radius + 1, neighbourhood.key_grid.num_rows)


def rows_holding(grid: ReducedGrid, points: slice) -> tuple[int, int]:
    """Rows ``(first, end)`` of ``grid`` holding any of ``points``."""
    first = int(torch.searchsorted(grid.row_starts, points.start, right=True)) - 1
    end = int(torch.searchsorted(grid.row_starts, points.stop - 1, right=True))
    return first, end


def points_of_rows(grid: ReducedGrid, first_row: int, end_row: int) -> slice:
    """Points of the rows ``first_row`` to ``end_row - 1``."""
    return slice(int(grid.row_starts[first_row]), int(grid.row_starts[end_row]))


def split_into_bands(
    neighbourhood: ReducedGridNeighbourhood, num_bands: int, rows: Optional[tuple[int, int]] = None
) -> list[NeighbourhoodBand]:
    """Split query rows into bands of whole rows with about the same number of points.

    Each band gets the key rows its queries reach. These form a small grid of their own, on which
    every query sees exactly the keys it sees on the whole grid.

    Parameters
    ----------
    neighbourhood : ReducedGridNeighbourhood
        Neighbourhood of the whole grids, nodes in grid order.
    num_bands : int
        Number of bands to aim for; fewer if there are fewer rows.
    rows : tuple[int, int], optional
        Query rows ``(first, end)`` to split; all rows by default.

    Returns
    -------
    list[NeighbourhoodBand]
        Bands from north to south, with point ranges on the whole grids.
    """
    if not neighbourhood.in_grid_order:
        raise ValueError("Bands of rows need the query and key nodes stored in grid order.")
    query_grid, key_grid = neighbourhood.query_grid, neighbourhood.key_grid
    first, end = rows if rows is not None else (0, query_grid.num_rows)

    # Cut the rows where the running number of points passes each equal share.
    row_starts = query_grid.row_starts.to(torch.float64)
    shares = torch.linspace(
        float(row_starts[first]), float(row_starts[end]), num_bands + 1, dtype=torch.float64, device=row_starts.device
    )
    cuts = torch.searchsorted(row_starts, shares).clamp(first, end)
    cuts = torch.unique(torch.cat([torch.tensor([first, end], device=cuts.device), cuts])).tolist()

    bands = []
    for first_row, end_row in zip(cuts[:-1], cuts[1:]):
        first_key_row, end_key_row = key_rows_reached(neighbourhood, first_row, end_row)
        band_neighbourhood = replace(
            neighbourhood,
            query_grid=query_grid.rows(first_row, end_row),
            key_grid=key_grid.rows(first_key_row, end_key_row),
            is_self_attention=False,
            num_bands=1,
        )
        query_points = points_of_rows(query_grid, first_row, end_row)
        key_points = points_of_rows(key_grid, first_key_row, end_key_row)
        first = query_points.start - key_points.start
        bands.append(
            NeighbourhoodBand(
                query_points=query_points,
                key_points=key_points,
                queries_in_keys=(
                    slice(first, first + query_points.stop - query_points.start)
                    if neighbourhood.is_self_attention
                    else None
                ),
                attention=ReducedGridAttentionWrapper(band_neighbourhood),
            )
        )
    return bands


@dataclass(frozen=True, eq=False)
class ShardPlan:
    """The work of one GPU.

    Each GPU owns a run of consecutive query points. A run can start or end inside a row, so the GPU
    computes every row holding one of its points and keeps only its own outputs. It fetches the
    inputs it lacks from the neighbouring GPUs. Point ranges are on the whole grids.
    """

    own_points: slice
    "The query points whose outputs this GPU returns."
    query_points: slice
    "Points of every query row holding one of the own points."
    key_points: slice
    "Key points those rows reach."
    bands: list[NeighbourhoodBand]
    "Bands covering ``query_points``, north to south."


def plan_shard(neighbourhood: ReducedGridNeighbourhood, num_bands: int, own_points: slice) -> ShardPlan:
    """Plan the work of the GPU that owns ``own_points``."""
    if own_points.stop <= own_points.start:
        raise ValueError(f"Every GPU needs at least one query point, got the points {own_points}.")
    first_row, end_row = rows_holding(neighbourhood.query_grid, own_points)
    first_key_row, end_key_row = key_rows_reached(neighbourhood, first_row, end_row)
    return ShardPlan(
        own_points=own_points,
        query_points=points_of_rows(neighbourhood.query_grid, first_row, end_row),
        key_points=points_of_rows(neighbourhood.key_grid, first_key_row, end_key_row),
        bands=split_into_bands(neighbourhood, num_bands, rows=(first_row, end_row)),
    )


class ReducedGridBands(LayerSplit):
    """Bands of whole rows for one model component.

    One plan on a single GPU, one per GPU when the points are split. Plans are cached per split.
    """

    def __init__(self, neighbourhood: ReducedGridNeighbourhood) -> None:
        self.neighbourhood = neighbourhood
        self.num_bands = neighbourhood.num_bands
        self._plans: dict[tuple[int, ...], list[ShardPlan]] = {}
        self._exchanges: dict[tuple, PointRangeExchange] = {}

    def __getstate__(self) -> dict:
        # Exchanges hold device tensors; they are rebuilt after loading.
        state = self.__dict__.copy()
        state["_exchanges"] = {}
        return state

    def used(self, sharded: bool) -> bool:
        return self.num_bands > 1 or sharded

    def work(
        self,
        batch_size: int,
        device: torch.device,
        model_comm_group: Optional[ProcessGroup] = None,
        query_shard_sizes: Optional[list[int]] = None,
        key_shard_sizes: Optional[list[int]] = None,
    ) -> BandWork:
        if model_comm_group is None:
            return BandWork(self.plans()[0], batch_size, self.num_key_points, self.num_query_points)
        if query_shard_sizes is None:
            raise ValueError("Neighbourhood attention split across GPUs needs its input split as well.")
        key_shard_sizes = query_shard_sizes if key_shard_sizes is None else key_shard_sizes
        rank = model_comm_group.rank()
        return BandWork(
            self.plans(query_shard_sizes)[rank],
            batch_size,
            self.num_key_points,
            self.num_query_points,
            model_comm_group=model_comm_group,
            key_exchange=self.key_exchange(query_shard_sizes, key_shard_sizes, rank, device),
            query_exchange=(
                None if self.neighbourhood.is_self_attention else self.query_exchange(query_shard_sizes, rank, device)
            ),
        )

    @property
    def num_query_points(self) -> int:
        return self.neighbourhood.query_grid.num_points

    @property
    def num_key_points(self) -> int:
        return self.neighbourhood.key_grid.num_points

    def plans(self, shard_sizes: Optional[list[int]] = None) -> list[ShardPlan]:
        """One plan per GPU in rank order; a single plan if ``shard_sizes`` is None."""
        sizes = tuple(shard_sizes) if shard_sizes is not None else (self.num_query_points,)
        if sum(sizes) != self.num_query_points:
            raise ValueError(
                f"The shard sizes {list(sizes)} do not add up to the {self.num_query_points} query points."
            )
        if sizes not in self._plans:
            starts = [0, *itertools.accumulate(sizes)]
            self._plans[sizes] = [
                plan_shard(self.neighbourhood, self.num_bands, slice(start, stop))
                for start, stop in zip(starts[:-1], starts[1:])
            ]
        return self._plans[sizes]

    def key_exchange(
        self, query_shard_sizes: list[int], key_shard_sizes: list[int], rank: int, device: torch.device
    ) -> PointRangeExchange:
        """Exchange that brings each GPU the key points its rows reach."""
        return self._exchange("key", query_shard_sizes, key_shard_sizes, rank, device)

    def query_exchange(self, query_shard_sizes: list[int], rank: int, device: torch.device) -> PointRangeExchange:
        """Exchange that brings each GPU the query points of the rows holding its own points."""
        return self._exchange("query", query_shard_sizes, query_shard_sizes, rank, device)

    def _exchange(
        self, side: str, query_shard_sizes: list[int], shard_sizes: list[int], rank: int, device: torch.device
    ) -> PointRangeExchange:
        cache_key = (side, tuple(query_shard_sizes), tuple(shard_sizes), rank, device)
        if cache_key not in self._exchanges:
            plans = self.plans(query_shard_sizes)
            wanted = [plan.key_points if side == "key" else plan.query_points for plan in plans]
            self._exchanges[cache_key] = build_point_range_exchange(list(shard_sizes), wanted, rank, device)
        return self._exchanges[cache_key]


@dataclass(eq=False)
class BandWork(GpuWork):
    """Bands of one GPU on reduced grids.

    On one GPU the inputs are only reshaped. Across GPUs, the missing runs of key points (and in
    cross attention query points) are fetched from the GPUs that own them.
    """

    plan: ShardPlan
    batch_size: int
    num_key_points: int
    num_query_points: int
    model_comm_group: Optional[ProcessGroup] = None
    key_exchange: Optional[PointRangeExchange] = None
    query_exchange: Optional[PointRangeExchange] = None

    @property
    def bands(self) -> list[NeighbourhoodBand]:
        return self.plan.bands

    @property
    def num_own_queries(self) -> int:
        return self.plan.own_points.stop - self.plan.own_points.start

    def _gather(self, x: Tensor, exchange: Optional[PointRangeExchange], num_points: int) -> PointRun:
        if exchange is None:
            return PointRun.whole(x.view(self.batch_size, num_points, -1))
        return fetch_point_range(x, exchange, self.model_comm_group)

    def gather_keys(self, x: Tensor) -> PointRun:
        return self._gather(x, self.key_exchange, self.num_key_points)

    def keys_of(self, keys: PointRun, band: NeighbourhoodBand) -> Tensor:
        return keys.take(band.key_points)

    def gather_queries(self, x: Tensor) -> PointRun:
        return self._gather(x, self.query_exchange, self.num_query_points)

    def queries_of(self, queries: PointRun, band: NeighbourhoodBand) -> Tensor:
        return queries.take(band.query_points)

    def keep(self, out: Optional[Tensor], band_out: Tensor, band: NeighbourhoodBand) -> Tensor:
        if out is None:
            out = band_out.new_empty(self.batch_size, self.num_own_queries, band_out.shape[-1])
        # Copy the outputs of the band's queries that this GPU owns.
        own = self.plan.own_points
        first, stop = max(band.query_points.start, own.start), min(band.query_points.stop, own.stop)
        if stop > first:
            start = band.query_points.start
            out[:, first - own.start : stop - own.start] = band_out[:, first - start : stop - start]
        return out
