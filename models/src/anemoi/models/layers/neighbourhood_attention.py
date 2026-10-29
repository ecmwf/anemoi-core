# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on global grids, with a registry of the grid families that have kernels.

Each query attends only to the keys around it on the sphere. A model component names the grid
family of its nodes in its configuration, for example::

    attention_implementation: neighbourhood
    neighbourhood:
      grid: octahedral
      kernel_size: [7, 13]
      backend: triton

and the node coordinates of the graph fix the resolution and the order of the points.

:data:`GRID_FAMILIES` lists every grid family with its settings, its Triton kernels and whether it
does cross attention; :data:`NEIGHBOURHOOD_BACKENDS` lists the backends every family has (the
Triton kernels and PyTorch attention with a dense mask, which compute the same thing).
:func:`build_grid_neighbourhood` checks a configuration against it and builds the family's
:class:`Neighbourhood` when the model is built: it checks that the nodes form a grid of the family,
and the attention layer then checks that the backend works with its options and head size. A
neighbourhood gives the attention module of its grids and says how a model component splits its
queries into bands, on one GPU or across several (:class:`LayerSplit`).

This module holds what all grid families share. Each family lives in a module of its own:

- Reduced grids (``octahedral``, ``healpix``), in
  :mod:`anemoi.models.layers.reduced_grid_neighbourhood`: a query attends to ``kernel_size[0]``
  latitude rows centred on its own and, in each row, the ``kernel_size[1]`` points nearest in
  longitude, in self or cross attention, optionally in bands of whole rows.
- The refined icosahedron (``icosahedral``, in
  :mod:`anemoi.models.layers.icosahedral_neighbourhood`): each point attends to the points
  within ``radius`` hops, in self attention.

When the points are split across GPUs, each GPU fetches before each layer the points of other
GPUs its queries reach.

Attention itself needs no communication. The kernels compare queries and keys by content only;
rotary position embeddings (:mod:`anemoi.models.layers.spherical_rotary`) add where each key lies
relative to its query.
"""

from __future__ import annotations

import importlib
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.layers.attention_wrapper import AttentionWrapper
from anemoi.models.triton.utils import is_triton_available


def load(path: str):
    """The object named by an import path ``"module:name"``."""
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class NeighbourhoodBackend:
    """One way of computing neighbourhood attention, and what it cannot do."""

    unsupported: frozenset[str]
    "Options of the attention layer it cannot honour, named as in the model config."
    needs_triton: bool = False
    "Whether it runs Triton kernels, which need a GPU and heads of a power of two of at least 16 channels."


# Neighbourhood attention sets its own mask, so no backend takes a sliding window.
NEIGHBOURHOOD_BACKENDS = {
    # The family's Triton kernels, which only visit the keys each tile of queries can reach. GPU only.
    "triton": NeighbourhoodBackend(unsupported=frozenset({"dropout_p", "window_size"}), needs_triton=True),
    # scaled_dot_product_attention with the full queries x keys mask, one byte per pair; for CPUs,
    # small grids and tests.
    "sdpa": NeighbourhoodBackend(unsupported=frozenset({"window_size"})),
}
"""The backends of every grid family, by the name used in the configuration."""


@dataclass(frozen=True, kw_only=True)
class GridFamily:
    """A family of grids with neighbourhood attention: how it is set up and its Triton kernels.

    The neighbourhood class, the kernels and the membership test are named by import path, ``"module:name"``, and
    imported on first use (:func:`load`), so the registry can be read on machines without Triton.
    """

    description: str
    neighbourhood: str
    "The :class:`Neighbourhood` class that checks the settings and nodes and builds the neighbourhood."
    self_attention: str
    "The Triton kernel of self attention."
    cross_attention: Optional[str] = None
    "The Triton kernel of cross attention, or None if the family only does self attention."
    is_member: Optional[str] = None
    "For reduced grids: the function that says whether a reduced grid belongs to the family."


GRID_FAMILIES: dict[str, GridFamily] = {
    "octahedral": GridFamily(
        description="octahedral reduced Gaussian grids O<N>: rows of 20, 24, 28, ... points from each pole",
        neighbourhood="anemoi.models.layers.reduced_grid_neighbourhood:ReducedGridNeighbourhood",
        self_attention="anemoi.models.triton.reduced_grid_attention:reduced_grid_attention",
        cross_attention="anemoi.models.triton.reduced_grid_cross_attention:reduced_grid_cross_attention",
        is_member="anemoi.models.layers.reduced_grid_neighbourhood:is_octahedral",
    ),
    "healpix": GridFamily(
        description="HEALPix grids in ring or nested ordering: 12 * nside**2 points",
        neighbourhood="anemoi.models.layers.reduced_grid_neighbourhood:ReducedGridNeighbourhood",
        self_attention="anemoi.models.triton.reduced_grid_attention:reduced_grid_attention",
        cross_attention="anemoi.models.triton.reduced_grid_cross_attention:reduced_grid_cross_attention",
        is_member="anemoi.models.layers.reduced_grid_neighbourhood:is_healpix",
    ),
    "icosahedral": GridFamily(
        description="refined icosahedra in diamond order (anemoi-graphs TriNodes, ordering: diamonds): 10 * 4**r + 2 points",
        neighbourhood="anemoi.models.layers.icosahedral_neighbourhood:IcosahedralGridNeighbourhood",
        self_attention="anemoi.models.triton.icosahedral_attention:icosahedral_attention",
    ),
}
"""Grid families with neighbourhood attention, by the name used in the configuration."""


def grid_family(name: Optional[str]) -> GridFamily:
    """The registered grid family ``name``."""
    if name not in GRID_FAMILIES:
        raise ValueError(f"No neighbourhood attention kernels for grid '{name}'. Known grids: {list(GRID_FAMILIES)}.")
    return GRID_FAMILIES[name]


def check_settings(
    config: dict, settings: set[str], key_coords: Optional[Tensor], query_coords: Optional[Tensor]
) -> tuple[str, str]:
    """Check what every grid family's configuration has in common, and give its family and backend.

    Parameters
    ----------
    config : dict
        The ``neighbourhood`` section of a model component, checked to be present by
        :func:`build_grid_neighbourhood`.
    settings : set[str]
        The settings the family takes, ``grid`` and ``backend`` among them.
    key_coords, query_coords : Tensor, optional
        Coordinates of the key and query nodes; no query coordinates for self attention.

    Returns
    -------
    tuple[str, str]
        The name of the grid family and of the backend, ``"triton"`` by default.
    """
    if key_coords is None:
        raise ValueError("Neighbourhood attention needs the coordinates of the graph nodes.")
    name = config.get("grid")
    family = grid_family(name)
    unknown = set(config) - settings
    if unknown:
        raise ValueError(
            f"Unknown neighbourhood settings {sorted(unknown)} for grid '{name}'; use {', '.join(sorted(settings))}."
        )
    backend = config.get("backend", "triton")
    if backend not in NEIGHBOURHOOD_BACKENDS:
        raise ValueError(
            f"Neighbourhood attention backend must be one of {list(NEIGHBOURHOOD_BACKENDS)}, got '{backend}'."
        )
    if query_coords is not None and family.cross_attention is None:
        raise ValueError(f"Neighbourhood attention on {name} grids only does self attention, as in the processor.")
    return name, backend


class Neighbourhood(ABC):
    """What a model component needs to know about the grids of its neighbourhood attention.

    Each grid family has its own subclass, built once per model component by
    :func:`build_grid_neighbourhood` and shared by all of its layers.
    """

    family: str
    "The name of the grid family in :data:`GRID_FAMILIES`."
    backend: str
    "The name of the backend in :data:`NEIGHBOURHOOD_BACKENDS`."

    @property
    def grid_family(self) -> GridFamily:
        return GRID_FAMILIES[self.family]

    @property
    def backend_spec(self) -> NeighbourhoodBackend:
        return NEIGHBOURHOOD_BACKENDS[self.backend]

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: dict, key_coords: Optional[Tensor], query_coords: Optional[Tensor] = None
    ) -> Neighbourhood:
        """Check the configuration and the nodes, and build the neighbourhood."""

    @abstractmethod
    def attention_module(self) -> NeighbourhoodAttention:
        """The attention module on all points of the grids."""

    @abstractmethod
    def layer_split(self, cpu_offload: bool) -> Optional[LayerSplit]:
        """How a model component splits its queries into bands, or None if it always runs on all points at once."""


class NeighbourhoodAttention(AttentionWrapper, ABC):
    """Base class of the attention modules of neighbourhoods.

    Takes ``(batch, heads, points, head_dim)`` tensors, like the other attention backends, and runs
    them through one of :data:`NEIGHBOURHOOD_BACKENDS`: the family's Triton kernel, or
    ``scaled_dot_product_attention`` with the dense mask of the neighbourhood. Subclasses say how
    the mask is built and how the kernel is called.
    """

    def __init__(self, neighbourhood: Neighbourhood) -> None:
        super().__init__()
        self.neighbourhood = neighbourhood
        self.backend = neighbourhood.backend
        self.unsupported = neighbourhood.backend_spec.unsupported
        if neighbourhood.backend_spec.needs_triton and not is_triton_available():
            raise ImportError(
                f"Neighbourhood attention backend '{self.backend}' needs Triton and a GPU; use the 'sdpa' backend "
                "on machines without them."
            )
        # Masks depend only on the grids and the device, so each is built once per device.
        self._masks: dict[torch.device, Tensor] = {}
        self._kernel: Optional[Callable] = None

    def __getstate__(self) -> dict:
        # The masks and the kernel are set up again on first use, so they are left out when the
        # model is saved.
        state = self.__dict__.copy()
        state["_masks"] = {}
        state["_kernel"] = None
        return state

    @abstractmethod
    def build_mask(self, device: torch.device) -> Tensor:
        """The ``(queries, keys)`` boolean mask of the neighbourhood, True where a query attends to a key."""

    @abstractmethod
    def kernel_path(self) -> str:
        """The import path of the Triton kernel, ``"module:name"``."""

    @abstractmethod
    def run_kernel(self, kernel: Callable, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Neighbourhood attention with the Triton kernel ``kernel``."""

    def check_lengths(self, query: Tensor, key: Tensor, num_queries: int, num_keys: int) -> None:
        """Raises if ``query`` and ``key`` do not hold the ``num_queries`` and ``num_keys`` points the module is set up for."""
        if query.shape[-2] != num_queries or key.shape[-2] != num_keys:
            raise ValueError(
                f"{self.describe()} is set up for {num_queries} queries and {num_keys} keys, "
                f"got {query.shape[-2]} and {key.shape[-2]}."
            )

    def attend(self, query: Tensor, key: Tensor, value: Tensor, dropout_p: float) -> Tensor:
        """Neighbourhood attention with the backend, on queries and keys in grid order."""
        if self.backend == "sdpa":
            if query.device not in self._masks:
                self._masks[query.device] = self.build_mask(query.device)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=self._masks[query.device], dropout_p=dropout_p
            )
        if self._kernel is None:
            self._kernel = load(self.kernel_path())
        return self.run_kernel(self._kernel, query, key, value)

    def describe(self) -> str:
        return f"Neighbourhood attention on {self.neighbourhood.family} grids with the '{self.backend}' backend"

    def check_head_dim(self, head_dim: int) -> None:
        if self.neighbourhood.backend_spec.needs_triton and (head_dim < 16 or head_dim & (head_dim - 1)):
            raise ValueError(
                f"{self.describe()} needs heads of a power of two of at least 16 channels, got {head_dim}."
            )


@dataclass(frozen=True, eq=False)
class NeighbourhoodBand:
    """A band of query points and the key points its queries reach.

    On reduced grids a band is a run of whole query rows, contiguous in grid order, which is also
    the node order, and its keys are the run of key rows they reach. On the icosahedron a band is
    the run of points one GPU owns, and its keys are those points followed by the points of
    other GPUs they reach. The key points include every key any query of the band attends to,
    and include the query points themselves in self attention.
    """

    query_points: slice
    key_points: slice | Tensor
    "The key points, as a run or as the numbers of the points in the order they are held."
    queries_in_keys: Optional[slice]
    "Where the query points sit among the key points, in self attention."
    attention: NeighbourhoodAttention


class GpuWork(ABC):
    """The bands one GPU works through in each layer, and how it gathers their inputs and keeps their outputs.

    A model component gathers the inputs of all key points of its bands once per layer, then runs
    each band on its share of them. With the points split across GPUs, gathering fetches the
    points of other GPUs; on one GPU it only reshapes.
    """

    bands: list[NeighbourhoodBand]
    num_own_queries: int
    "The query points whose outputs this GPU returns."

    @abstractmethod
    def gather_keys(self, x: Tensor) -> object:
        """The inputs of every key point the bands reach, from this GPU's ``(batch * points, channels)`` key inputs."""

    @abstractmethod
    def keys_of(self, keys: object, band: NeighbourhoodBand) -> Tensor:
        """The band's key inputs, ``(batch, band key points, channels)``, out of :meth:`gather_keys`."""

    def gather_queries(self, x: Tensor) -> object:
        """The inputs of every query point of the bands, in cross attention."""
        raise NotImplementedError(f"{type(self).__name__} only does self attention.")

    def queries_of(self, queries: object, band: NeighbourhoodBand) -> Tensor:
        """The band's query inputs, ``(batch, band query points, channels)``, out of :meth:`gather_queries`."""
        raise NotImplementedError(f"{type(self).__name__} only does self attention.")

    @abstractmethod
    def keep(self, out: Optional[Tensor], band_out: Tensor, band: NeighbourhoodBand) -> Tensor:
        """Add the outputs of the band's query points this GPU owns to ``out``, ``(batch, own queries, channels)``.

        ``out`` is None for the first band, and the result is passed in for the next one.
        """


class LayerSplit(ABC):
    """How a model component works through its queries in bands, on one GPU or split across GPUs."""

    @abstractmethod
    def used(self, sharded: bool) -> bool:
        """Whether the component works band by band, rather than on all points at once."""

    @abstractmethod
    def work(
        self,
        batch_size: int,
        device: torch.device,
        model_comm_group: Optional[ProcessGroup] = None,
        query_shard_sizes: Optional[list[int]] = None,
        key_shard_sizes: Optional[list[int]] = None,
    ) -> GpuWork:
        """The work of this GPU.

        Parameters
        ----------
        batch_size : int
            Batch size.
        device : torch.device
            Device of the inputs.
        model_comm_group : ProcessGroup, optional
            The GPUs the points are split across, or None on one GPU.
        query_shard_sizes, key_shard_sizes : list[int], optional
            Number of query and key points of each GPU, in rank order, when the points are split
            across GPUs. The key shard sizes are those of the queries in self attention.

        Returns
        -------
        GpuWork
            The bands of this GPU and how it gathers their inputs.
        """


def build_grid_neighbourhood(
    attention_implementation: str,
    config: Optional[dict],
    key_coords: Optional[Tensor],
    query_coords: Optional[Tensor] = None,
) -> Optional[Neighbourhood]:
    """The neighbourhood of a model component, or None if it does not use neighbourhood attention.

    Parameters
    ----------
    attention_implementation : str
        The component's attention implementation.
    config : dict, optional
        Its ``neighbourhood`` section: ``grid`` (a key of :data:`GRID_FAMILIES`), optionally
        ``backend`` (a backend of that family, ``"triton"`` by default) and the family's own
        settings.
    key_coords : Tensor, optional
        Latitude and longitude of the key nodes in radians, shape ``(num_keys, 2)``.
    query_coords : Tensor, optional
        Coordinates of the query nodes. Leave out for self attention.

    Returns
    -------
    Neighbourhood or None
        The neighbourhood of the grid family named in ``config``.
    """
    if attention_implementation != "neighbourhood":
        if config is not None:
            raise ValueError(
                "'neighbourhood' settings are only used with attention_implementation 'neighbourhood', "
                f"got '{attention_implementation}'."
            )
        return None
    if config is None:
        raise ValueError("attention_implementation 'neighbourhood' needs a 'neighbourhood' configuration.")
    neighbourhood_class = load(grid_family(config.get("grid")).neighbourhood)
    return neighbourhood_class.from_config(config, key_coords, query_coords)
