# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on global grids, with a registry of the grids that have kernels.

Each query attends only to the keys around it on the sphere: ``kernel_size[0]`` latitude rows
around its own and, in each row, the ``kernel_size[1]`` points nearest in longitude, going round
the globe (see :class:`anemoi.models.layers.reduced_grid.ReducedGridCrossNeighbourhoodMask`).
Queries and keys may sit on the same grid (self attention, as in the processor) or on two grids
of the same family (cross attention, as in the encoder and decoder).

A model component names the grid family in its configuration, for example::

    attention_implementation: neighbourhood
    neighbourhood:
      grid: octahedral
      kernel_size: [7, 13]
      backend: triton

and the node coordinates of the graph fix the resolution and the order of the points.

With ``num_bands`` above one, a model component works through its queries in bands of whole
latitude rows, each band with only the key rows its queries reach (see :func:`split_into_bands`).
Each query attends to exactly the same keys, and only one band is held in memory at a time.

When the points are split across GPUs, each GPU works through bands of the rows that hold its own
points, with ``num_bands`` bands per GPU. Before each layer it fetches from its neighbours the
points just outside its share that those rows reach (see :class:`ShardPlan`); attention itself
needs no communication.

The kernels compare queries and keys by content only; rotary position embeddings
(:mod:`anemoi.models.layers.spherical_rotary`) add where each key lies relative to its query.
"""

from __future__ import annotations

import importlib
import itertools
import logging
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.distributed.point_ranges import PointRangeExchange
from anemoi.models.distributed.point_ranges import build_point_range_exchange
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.reduced_grid import ReducedGridCrossNeighbourhoodMask
from anemoi.models.layers.reduced_grid import ReducedGridNeighbourhoodMask

LOGGER = logging.getLogger(__name__)

BACKENDS = ("triton", "flex", "sdpa")


def _is_octahedral(grid: ReducedGrid) -> bool:
    n = grid.num_rows // 2
    north = tuple(4 * i + 20 for i in range(n))
    return grid.num_rows == 2 * n and grid.row_lengths == north + north[::-1] and not grid.is_shifted


def _is_healpix(grid: ReducedGrid) -> bool:
    nside = (grid.num_rows + 1) // 4
    return nside > 0 and grid.num_rows == 4 * nside - 1 and grid == ReducedGrid.healpix(nside)


@dataclass(frozen=True)
class GridKernels:
    """The grids of one family and the Triton kernels that compute neighbourhood attention on them.

    The kernels are named by import path, ``"module:function"``, and imported on first use, so
    the registry can be read on machines without Triton.
    """

    description: str
    is_member: Callable[[ReducedGrid], bool]
    self_attention: str
    cross_attention: str

    @staticmethod
    def _load(path: str) -> Callable:
        module, name = path.split(":")
        return getattr(importlib.import_module(module), name)

    def load_self_attention(self) -> Callable:
        return self._load(self.self_attention)

    def load_cross_attention(self) -> Callable:
        return self._load(self.cross_attention)


GRID_KERNELS: dict[str, GridKernels] = {
    "octahedral": GridKernels(
        description="octahedral reduced Gaussian grids O<N>: rows of 20, 24, 28, ... points from each pole",
        is_member=_is_octahedral,
        self_attention="anemoi.models.triton.reduced_grid_attention:reduced_grid_attention",
        cross_attention="anemoi.models.triton.reduced_grid_cross_attention:reduced_grid_cross_attention",
    ),
    "healpix": GridKernels(
        description="HEALPix grids in ring or nested ordering: 12 * nside**2 points",
        is_member=_is_healpix,
        self_attention="anemoi.models.triton.reduced_grid_attention:reduced_grid_attention",
        cross_attention="anemoi.models.triton.reduced_grid_cross_attention:reduced_grid_cross_attention",
    ),
}
"""Grid families with neighbourhood attention kernels, by the name used in the configuration."""


def grid_from_coords(family: str, coords: Tensor) -> tuple[ReducedGrid, Optional[Tensor]]:
    """Recognise the grid formed by the nodes at ``coords`` and check that it belongs to ``family``.

    Parameters
    ----------
    family : str
        A key of :data:`GRID_KERNELS`.
    coords : Tensor
        Latitude and longitude of each node in radians, shape ``(num_nodes, 2)``.

    Returns
    -------
    tuple[ReducedGrid, Tensor | None]
        The grid, and the order that sorts the nodes into it (node ``order[i]`` is point ``i`` of
        the grid), or None when the nodes are already in grid order.
    """
    if family not in GRID_KERNELS:
        raise ValueError(f"No neighbourhood attention kernels for grid '{family}'. Known grids: {list(GRID_KERNELS)}.")
    grid, order = ReducedGrid.from_unordered_coords(coords)
    if not GRID_KERNELS[family].is_member(grid):
        raise ValueError(
            f"The {coords.shape[0]} nodes, in {grid.num_rows} latitude rows, do not form one of the "
            f"{GRID_KERNELS[family].description}."
        )
    if torch.equal(order, torch.arange(order.numel())):
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
class GridNeighbourhood:
    """Everything a neighbourhood attention layer needs to know about its query and key grids.

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
    ) -> GridNeighbourhood:
        """Set up neighbourhood attention from its configuration and the node coordinates.

        Parameters
        ----------
        config : dict
            ``grid`` (a key of :data:`GRID_KERNELS`), ``kernel_size`` (two odd numbers: latitude
            rows and points per row), optionally ``backend`` (``"triton"``, the default,
            ``"flex"`` or ``"sdpa"``) and optionally ``num_bands`` (the number of bands of query
            rows to work through one at a time, 1 by default).
        key_coords : Tensor
            Coordinates of the key nodes in radians, shape ``(num_keys, 2)``.
        query_coords : Tensor, optional
            Coordinates of the query nodes. Leave out for self attention.

        Returns
        -------
        GridNeighbourhood
            The query and key grids, the point orders and the neighbourhood settings.
        """
        if config is None:
            raise ValueError("attention_implementation 'neighbourhood' needs a 'neighbourhood' configuration.")
        if key_coords is None:
            raise ValueError("Neighbourhood attention needs the coordinates of the graph nodes.")
        unknown = set(config) - {"grid", "kernel_size", "backend", "num_bands"}
        if unknown:
            raise ValueError(
                f"Unknown neighbourhood settings {sorted(unknown)}; use grid, kernel_size, backend and num_bands."
            )
        family = config["grid"]
        kernel_size = tuple(int(k) for k in config["kernel_size"])
        backend = config.get("backend", "triton")
        if len(kernel_size) != 2 or any(k <= 0 or k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be two positive odd numbers, got {config['kernel_size']}.")
        if backend not in BACKENDS:
            raise ValueError(f"Neighbourhood attention backend must be one of {BACKENDS}, got '{backend}'.")
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
    def kernels(self) -> GridKernels:
        return GRID_KERNELS[self.family]

    @property
    def in_grid_order(self) -> bool:
        """Whether the query and key nodes are stored in grid order, row by row from north to south."""
        return self.query_order is None and self.key_order is None


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


class NeighbourhoodAttentionWrapper(nn.Module):
    """Neighbourhood attention on the grids described by a :class:`GridNeighbourhood`.

    Takes ``(batch, heads, points, head_dim)`` tensors with the points in node order, puts them in
    grid order when the two differ, and puts the output back in node order.

    Three backends compute the same thing:

    ``"triton"``
        The kernels registered for the grid family in :data:`GRID_KERNELS`, which only visit the
        keys each tile of queries can reach. GPU only; the head dimension must be a power of two
        of at least 16. No dropout.
    ``"flex"``
        Flex attention with a block mask, so blocks of keys outside every neighbourhood are
        skipped. Compiled on the first call on GPU. No dropout.
    ``"sdpa"``
        ``scaled_dot_product_attention`` with the full ``queries x keys`` boolean mask. Runs
        anywhere; the mask takes one byte per query-key pair, so it suits small grids and tests.
    """

    def __init__(self, neighbourhood: GridNeighbourhood) -> None:
        super().__init__()
        self.neighbourhood = neighbourhood
        self.backend = neighbourhood.backend

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
            self.mask_mod = ReducedGridNeighbourhoodMask(neighbourhood.key_grid, neighbourhood.kernel_size)
        else:
            self.mask_mod = ReducedGridCrossNeighbourhoodMask(
                neighbourhood.query_grid, neighbourhood.key_grid, neighbourhood.kernel_size
            )

        # Masks depend only on the grids and device, so they are built once per device.
        self._masks: dict[torch.device, object] = {}
        self._flex_attention = None
        self._kernel = None

    def __getstate__(self) -> dict:
        # The masks, the compiled attention and the kernel are set up again on first use, so they
        # are left out when the model is saved.
        state = self.__dict__.copy()
        state["_masks"] = {}
        state["_flex_attention"] = None
        state["_kernel"] = None
        return state

    def _mask_on(self, device: torch.device):
        if device not in self._masks:
            mask_mod = self.mask_mod.to(device)
            n_q = self.neighbourhood.query_grid.num_points
            n_k = self.neighbourhood.key_grid.num_points
            if self.backend == "sdpa":
                q_idx = torch.arange(n_q, device=device)
                k_idx = torch.arange(n_k, device=device)
                self._masks[device] = mask_mod(None, None, q_idx[:, None], k_idx[None, :])
            else:
                from torch.nn.attention.flex_attention import create_block_mask

                self._masks[device] = create_block_mask(mask_mod, None, None, n_q, n_k, device=device)
        return self._masks[device]

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, dropout_p: float) -> Tensor:
        if self.backend == "sdpa":
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=self._mask_on(query.device), dropout_p=dropout_p
            )
        if dropout_p > 0.0:
            raise NotImplementedError(
                f"Dropout is not supported by the {self.backend} backend of neighbourhood attention."
            )

        nb = self.neighbourhood
        if self.backend == "triton":
            if self._kernel is None:
                kernels = nb.kernels
                self._kernel = kernels.load_self_attention() if nb.is_self_attention else kernels.load_cross_attention()
            if nb.is_self_attention:
                return self._kernel(query, key, value, nb.key_grid, nb.kernel_size)
            return self._kernel(query, key, value, nb.query_grid, nb.key_grid, nb.kernel_size)

        if self._flex_attention is None:
            from torch.nn.attention.flex_attention import flex_attention

            # Compiling produces the fused kernel on GPU; on CPU the uncompiled version is used.
            self._flex_attention = torch.compile(flex_attention, dynamic=False) if query.is_cuda else flex_attention
        return self._flex_attention(query, key, value, block_mask=self._mask_on(query.device))

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
        if causal or window_size is not None:
            raise ValueError("Neighbourhood attention sets its own mask; causal and window_size must not be used.")
        if softcap is not None and softcap > 0:
            raise NotImplementedError("Softcap is not supported by neighbourhood attention.")
        n_q = self.neighbourhood.query_grid.num_points
        n_k = self.neighbourhood.key_grid.num_points
        if query.shape[-2] != n_q or key.shape[-2] != n_k:
            raise ValueError(
                f"Neighbourhood attention is set up for {n_q} queries and {n_k} keys, "
                f"got {query.shape[-2]} and {key.shape[-2]}."
            )

        if self._reorder_queries:
            query = _ReorderPoints.apply(query, self.query_order, self.query_inverse)
        if self._reorder_keys:
            key = _ReorderPoints.apply(key, self.key_order, self.key_inverse)
            value = _ReorderPoints.apply(value, self.key_order, self.key_inverse)

        out = self._attend(query, key, value, dropout_p)

        if self._reorder_queries:
            out = _ReorderPoints.apply(out, self.query_inverse, self.query_order)
        return out


@dataclass(frozen=True, eq=False)
class NeighbourhoodBand:
    """A band of whole query rows and the key rows its queries reach.

    The points of a band are contiguous in grid order, which is also the node order; the key
    points include every key any query of the band attends to.
    """

    query_points: slice
    key_points: slice
    attention: NeighbourhoodAttentionWrapper


def window_start_rows(neighbourhood: GridNeighbourhood) -> Tensor:
    """The first key row of each query row's window, kept inside the key grid as the attention rule does."""
    query_grid, key_grid = neighbourhood.query_grid, neighbourhood.key_grid
    kernel_h = neighbourhood.kernel_size[0]
    if neighbourhood.is_self_attention:
        row_map = torch.arange(query_grid.num_rows)
    else:
        row_map = query_grid.nearest_rows(key_grid)
    return (row_map - kernel_h // 2).clamp(0, key_grid.num_rows - kernel_h)


def key_rows_reached(neighbourhood: GridNeighbourhood, first_row: int, end_row: int) -> tuple[int, int]:
    """The key rows ``(first, end)`` that the query rows ``first_row`` to ``end_row - 1`` attend to."""
    window_start = window_start_rows(neighbourhood)[first_row:end_row]
    return int(window_start.min()), int(window_start.max()) + neighbourhood.kernel_size[0]


def rows_holding(grid: ReducedGrid, points: slice) -> tuple[int, int]:
    """The rows ``(first, end)`` of ``grid`` that hold any of the points ``points.start`` to ``points.stop - 1``."""
    first = int(torch.searchsorted(grid.row_starts, points.start, right=True)) - 1
    end = int(torch.searchsorted(grid.row_starts, points.stop - 1, right=True))
    return first, end


def points_of_rows(grid: ReducedGrid, first_row: int, end_row: int) -> slice:
    """The points of the rows ``first_row`` to ``end_row - 1`` of ``grid``."""
    return slice(int(grid.row_starts[first_row]), int(grid.row_starts[end_row]))


def split_into_bands(
    neighbourhood: GridNeighbourhood, num_bands: int, rows: Optional[tuple[int, int]] = None
) -> list[NeighbourhoodBand]:
    """Split query rows into bands of whole rows with about the same number of points.

    Each band attends to the rows of the key grid its queries reach, which themselves form a
    small grid. A query finds its key rows from its own latitude and its points from the lengths
    and shifts of the rows, so within its band it sees exactly the keys it sees on the whole grid.
    For self attention the band's key rows are its own rows and a margin of ``kernel_size[0] // 2``
    rows on either side (fewer at the poles).

    Parameters
    ----------
    neighbourhood : GridNeighbourhood
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
    shares = torch.linspace(float(row_starts[first]), float(row_starts[end]), num_bands + 1, dtype=torch.float64)
    cuts = torch.searchsorted(row_starts, shares).clamp(first, end)
    cuts = torch.unique(torch.cat([torch.tensor([first, end]), cuts])).tolist()

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
        bands.append(
            NeighbourhoodBand(
                query_points=points_of_rows(query_grid, first_row, end_row),
                key_points=points_of_rows(key_grid, first_key_row, end_key_row),
                attention=NeighbourhoodAttentionWrapper(band_neighbourhood),
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


def plan_shard(neighbourhood: GridNeighbourhood, num_bands: int, own_points: slice) -> ShardPlan:
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


class NeighbourhoodBands:
    """The bands of one model component, for the query points each GPU owns.

    On a single GPU there is one plan, owning every query point. When the query points are split
    across GPUs in runs of consecutive points (``shard_sizes`` points each, in rank order), there
    is one plan per GPU (see :class:`ShardPlan`). The plans for each way of splitting are worked
    out once and kept.
    """

    def __init__(self, neighbourhood: GridNeighbourhood) -> None:
        self.neighbourhood = neighbourhood
        self.num_bands = neighbourhood.num_bands
        self._plans: dict[tuple[int, ...], list[ShardPlan]] = {}
        self._exchanges: dict[tuple, PointRangeExchange] = {}

    def __getstate__(self) -> dict:
        # The exchanges hold index tensors on a device; they are planned again where the model is loaded.
        state = self.__dict__.copy()
        state["_exchanges"] = {}
        return state

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


def write_own_points(out: Tensor, band_out: Tensor, band: NeighbourhoodBand, plan: ShardPlan) -> None:
    """Copy the outputs of the band's query points that the GPU owns into ``out``.

    ``band_out`` holds the outputs of all of the band's query points and ``out`` those of the GPU's
    own points, both laid out ``(batch, points, channels)``.
    """
    first = max(band.query_points.start, plan.own_points.start)
    stop = min(band.query_points.stop, plan.own_points.stop)
    if stop > first:
        own_start, band_start = plan.own_points.start, band.query_points.start
        out[:, first - own_start : stop - own_start] = band_out[:, first - band_start : stop - band_start]


def build_grid_neighbourhood(
    attention_implementation: str,
    config: Optional[dict],
    key_coords: Optional[Tensor],
    query_coords: Optional[Tensor] = None,
) -> Optional[GridNeighbourhood]:
    """The :class:`GridNeighbourhood` of a model component, or None if it does not use neighbourhood attention."""
    if attention_implementation != "neighbourhood":
        if config is not None:
            raise ValueError(
                "'neighbourhood' settings are only used with attention_implementation 'neighbourhood', "
                f"got '{attention_implementation}'."
            )
        return None
    return GridNeighbourhood.from_config(config, key_coords, query_coords)
