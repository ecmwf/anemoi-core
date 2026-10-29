# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on reduced grids: the grid families ``octahedral`` and ``healpix``.

The grid families of :data:`anemoi.models.layers.neighbourhood_attention.GRID_FAMILIES` whose
points lie in latitude rows. A query attends to ``kernel_size[0]`` latitude rows centred on its
own and, in each row, the ``kernel_size[1]`` points nearest in longitude, going round the globe,
with whole rings of the rows it reaches over a pole (see
:class:`anemoi.models.layers.reduced_grid.ReducedGridCrossNeighbourhoodMask`). Queries and keys
may sit on the same grid (self attention, as in the processor) or on two grids of the same family
(cross attention, as in the encoder and decoder).

With ``num_bands`` above one a model component works through its queries in bands of whole
latitude rows, each with only the key rows its queries reach (see :func:`split_into_bands`). When
the points are split across GPUs, each GPU works through bands of the rows that hold its own
points and fetches before each layer the points just outside its share that those rows reach
(see :class:`ShardPlan`).

The grids are described in :mod:`anemoi.models.layers.reduced_grid`, the kernels are in
:mod:`anemoi.models.triton.reduced_grid_attention` and
:mod:`anemoi.models.triton.reduced_grid_cross_attention`.
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
    """Recognise the grid formed by the nodes at ``coords`` and check that it belongs to ``family``.

    Parameters
    ----------
    family : str
        A reduced-grid family of :data:`anemoi.models.layers.neighbourhood_attention.GRID_FAMILIES`.
    coords : Tensor
        Latitude and longitude of each node in radians, shape ``(num_nodes, 2)``.

    Returns
    -------
    tuple[ReducedGrid, Tensor | None]
        The grid, and the order that sorts the nodes into it (node ``order[i]`` is point ``i`` of
        the grid), or None when the nodes are already in grid order.
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
    """Check that cross attention reads every key point, and log how the queries and keys are connected.

    The log gives how many keys each query reads and how many queries read each key point.

    Every query gets its keys, but when the key grid is much finer than the query grid, the keys
    between two queries can be left out by both; their values then never reach the queries (in an
    encoder: data that never reaches the processor).

    Raises
    ------
    ValueError
        If some key point is attended to by no query. The message names the smallest kernel of
        at least ``kernel_size`` in both directions, with equal sides where possible, that reads
        every key point.
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
    """The query and key grids of neighbourhood attention on a family of reduced grids.

    Built once per model component and shared by all of its layers.
    """

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
        """Set up neighbourhood attention from its configuration and the node coordinates.

        Parameters
        ----------
        config : dict
            ``grid`` (a reduced-grid family of
            :data:`anemoi.models.layers.neighbourhood_attention.GRID_FAMILIES`), ``kernel_size`` (two odd
            numbers: latitude rows and points per row), optionally ``backend`` (``"triton"``,
            the default, or ``"sdpa"``) and optionally ``num_bands`` (the number of
            bands of query rows to work through one at a time, 1 by default).
        key_coords : Tensor
            Coordinates of the key nodes in radians, shape ``(num_keys, 2)``.
        query_coords : Tensor, optional
            Coordinates of the query nodes. Leave out for self attention.

        Returns
        -------
        ReducedGridNeighbourhood
            The query and key grids, the point orders and the neighbourhood settings.
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
        """Whether the query and key nodes are stored in grid order, row by row from north to south."""
        return self.query_order is None and self.key_order is None

    def attention_module(self) -> ReducedGridAttentionWrapper:
        return ReducedGridAttentionWrapper(self)

    def layer_split(self, cpu_offload: bool) -> Optional[ReducedGridBands]:
        """Bands of whole rows, for nodes in grid order: on one GPU when num_bands > 1, and whenever
        the points are split across GPUs, with num_bands bands of each GPU's own rows.
        """
        if not self.in_grid_order:
            return None
        if cpu_offload:
            if self.num_bands > 1:
                raise ValueError("Neighbourhood attention in bands (num_bands > 1) does not work with cpu_offload.")
            return None
        return ReducedGridBands(self)


def _gather_points(x: Tensor, order: Tensor) -> Tensor:
    """The points (second-to-last dimension) of ``(batch, heads, points, head_dim)`` ``x`` in the order ``order``.

    The result is stored point by point, all heads of a point together, the layout of the model's
    ``(batch * points, heads * head_dim)`` features; the gather then moves whole rows of them.
    """
    return x.transpose(-3, -2).index_select(-3, order).transpose(-3, -2)


class _ReorderPoints(torch.autograd.Function):
    """Puts the points (second-to-last dimension) in a new order.

    ``order`` is a permutation and ``inverse`` undoes it. The gradient is put back with
    ``inverse``, a plain gather, which is much faster than the scattered additions autograd
    would otherwise use.
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
    """Neighbourhood attention on the grids described by a :class:`ReducedGridNeighbourhood`.

    Takes ``(batch, heads, points, head_dim)`` tensors with the points in node order, puts them in
    grid order when the two differ, and puts the output back in node order.
    """

    def __init__(self, neighbourhood: ReducedGridNeighbourhood) -> None:
        super().__init__(neighbourhood)

        # The point orders move with the module to the GPU but are not part of the saved weights.
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
    """The key row each query row's window is centred on: its own row for self attention, else the nearest in latitude."""
    query_grid, key_grid = neighbourhood.query_grid, neighbourhood.key_grid
    if neighbourhood.is_self_attention:
        return torch.arange(query_grid.num_rows, device=query_grid.row_starts.device)
    return query_grid.nearest_rows(key_grid)


def key_rows_reached(neighbourhood: ReducedGridNeighbourhood, first_row: int, end_row: int) -> tuple[int, int]:
    """The key rows ``(first, end)`` that the query rows ``first_row`` to ``end_row - 1`` attend to."""
    centre = centre_rows(neighbourhood)[first_row:end_row]
    radius = neighbourhood.kernel_size[0] // 2
    return max(int(centre.min()) - radius, 0), min(int(centre.max()) + radius + 1, neighbourhood.key_grid.num_rows)


def rows_holding(grid: ReducedGrid, points: slice) -> tuple[int, int]:
    """The rows ``(first, end)`` of ``grid`` that hold any of the points ``points.start`` to ``points.stop - 1``."""
    first = int(torch.searchsorted(grid.row_starts, points.start, right=True)) - 1
    end = int(torch.searchsorted(grid.row_starts, points.stop - 1, right=True))
    return first, end


def points_of_rows(grid: ReducedGrid, first_row: int, end_row: int) -> slice:
    """The points of the rows ``first_row`` to ``end_row - 1`` of ``grid``."""
    return slice(int(grid.row_starts[first_row]), int(grid.row_starts[end_row]))


def split_into_bands(
    neighbourhood: ReducedGridNeighbourhood, num_bands: int, rows: Optional[tuple[int, int]] = None
) -> list[NeighbourhoodBand]:
    """Split query rows into bands of whole rows with about the same number of points.

    Each band attends to the rows of the key grid its queries reach, which themselves form a
    small grid. A query finds its key rows from its own latitude and its points from the lengths
    and shifts of the rows, so within its band it sees exactly the keys it sees on the whole grid.
    For self attention the band's key rows are its own rows and a margin of ``kernel_size[0] // 2``
    rows on either side (fewer at the poles).

    Parameters
    ----------
    neighbourhood : ReducedGridNeighbourhood
        The neighbourhood of the whole grids, with the nodes in grid order.
    num_bands : int
        Number of bands to aim for; there are fewer when there are fewer rows.
    rows : tuple[int, int], optional
        The query rows ``(first, end)`` to split; all rows by default.

    Returns
    -------
    list[NeighbourhoodBand]
        The bands from north to south. Their point ranges count from the first point of the whole grids.
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
    """The work of one GPU: the outputs of its own query points, computed band by band.

    When the points are split across GPUs, each GPU owns a run of consecutive query points (its
    shard) and returns their outputs. A shard can start and end inside a row, but attention works
    on whole rows, so the GPU computes every row that holds one of its points and keeps only its
    own outputs. For that it needs the inputs of a run of points from its neighbours as well:
    the query points of those whole rows and the key points their rows reach.

    All point ranges count from the first point of the whole grids.
    """

    own_points: slice
    "The GPU's own query points, whose outputs it returns."
    query_points: slice
    "The points of every query row that holds one of the GPU's own points."
    key_points: slice
    "The key points those query rows attend to."
    bands: list[NeighbourhoodBand]
    "Bands covering ``query_points``, from north to south."


def plan_shard(neighbourhood: ReducedGridNeighbourhood, num_bands: int, own_points: slice) -> ShardPlan:
    """The work of the GPU that owns the query points ``own_points`` (see :class:`ShardPlan`)."""
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
    """The bands of one model component, for the query points each GPU owns.

    On a single GPU there is one plan, owning every query point. When the query points are split
    across GPUs in runs of consecutive points (``shard_sizes`` points each, in rank order), there
    is one plan per GPU (see :class:`ShardPlan`). The plans for each way of splitting are worked
    out once and kept.
    """

    def __init__(self, neighbourhood: ReducedGridNeighbourhood) -> None:
        self.neighbourhood = neighbourhood
        self.num_bands = neighbourhood.num_bands
        self._plans: dict[tuple[int, ...], list[ShardPlan]] = {}
        self._exchanges: dict[tuple, PointRangeExchange] = {}

    def __getstate__(self) -> dict:
        # The exchanges hold index tensors on a device; they are planned again where the model is loaded.
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
        """One plan per GPU, in rank order; a single plan for all query points if ``shard_sizes`` is None."""
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
        """The exchange that brings every GPU the key points its query rows reach."""
        return self._exchange("key", query_shard_sizes, key_shard_sizes, rank, device)

    def query_exchange(self, query_shard_sizes: list[int], rank: int, device: torch.device) -> PointRangeExchange:
        """The exchange that brings every GPU the query points of the whole rows holding its own points."""
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
    """The bands of one GPU on reduced grids (see :class:`ShardPlan`).

    On one GPU the inputs are those of all points and are only reshaped. Split across GPUs, the
    GPU fetches the runs of key points (and in cross attention of query points) its whole rows
    reach from the GPUs that own them, and reads its own points in place.
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
        # Copy the outputs of the band's query points that this GPU owns.
        own = self.plan.own_points
        first, stop = max(band.query_points.start, own.start), min(band.query_points.stop, own.stop)
        if stop > first:
            start = band.query_points.start
            out[:, first - own.start : stop - own.start] = band_out[:, first - start : stop - start]
        return out
