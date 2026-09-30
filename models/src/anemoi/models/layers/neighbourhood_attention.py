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

The kernels compare queries and keys by content only; rotary position embeddings
(:mod:`anemoi.models.layers.spherical_rotary`) add where each key lies relative to its query.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch import nn

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
            rows and points per row) and optionally ``backend`` (``"triton"``, the default,
            ``"flex"`` or ``"sdpa"``).
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
        unknown = set(config) - {"grid", "kernel_size", "backend"}
        if unknown:
            raise ValueError(f"Unknown neighbourhood settings {sorted(unknown)}; use grid, kernel_size and backend.")
        family = config["grid"]
        kernel_size = tuple(int(k) for k in config["kernel_size"])
        backend = config.get("backend", "triton")
        if len(kernel_size) != 2 or any(k <= 0 or k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be two positive odd numbers, got {config['kernel_size']}.")
        if backend not in BACKENDS:
            raise ValueError(f"Neighbourhood attention backend must be one of {BACKENDS}, got '{backend}'.")

        key_grid, key_order = grid_from_coords(family, key_coords)
        if query_coords is None:
            return cls(family, kernel_size, backend, key_grid, key_grid, key_order, key_order, is_self_attention=True)
        query_grid, query_order = grid_from_coords(family, query_coords)
        check_every_key_attended(query_grid, key_grid, kernel_size)
        return cls(family, kernel_size, backend, query_grid, key_grid, query_order, key_order)

    @property
    def kernels(self) -> GridKernels:
        return GRID_KERNELS[self.family]


class _ReorderPoints(torch.autograd.Function):
    """Puts the points (second-to-last dimension) in a new order.

    ``order`` is a permutation and ``inverse`` undoes it. The gradient is put back with
    ``inverse``, a plain gather, which is much faster than the scattered additions autograd
    would otherwise use.
    """

    @staticmethod
    def forward(ctx, x: Tensor, order: Tensor, inverse: Tensor) -> Tensor:
        ctx.save_for_backward(inverse)
        return x.index_select(-2, order)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        return grad.index_select(-2, inverse), None, None


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
