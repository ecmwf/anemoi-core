# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on global grids: each query attends only to the keys around it.

A model component picks a grid family and its settings in the config, e.g.::

    attention_implementation: neighbourhood
    neighbourhood:
      grid: octahedral
      kernel_size: [7, 13]
      backend: triton

The node coordinates then fix the resolution and the order of the points.

This module holds what the grid families share: the registry :data:`GRID_FAMILIES`, the backends
:data:`NEIGHBOURHOOD_BACKENDS` and the base classes each family implements. The families live in
their own modules:

- ``octahedral`` and ``healpix``: :mod:`anemoi.models.layers.reduced_grid_neighbourhood`
- ``icosahedral``: :mod:`anemoi.models.layers.icosahedral_neighbourhood`

The kernels compare queries and keys by content only; positions come from rotary embeddings
(:mod:`anemoi.models.layers.spherical_rotary`).
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
    """Import the object at ``"module:name"``."""
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class NeighbourhoodBackend:
    """One way of computing neighbourhood attention."""

    unsupported: frozenset[str]
    "Attention options it does not support, named as in the model config."
    needs_triton: bool = False
    "Whether it runs Triton kernels (GPU only, head size a power of two of at least 16)."


# Neighbourhood attention sets its own mask, so no backend takes a sliding window.
NEIGHBOURHOOD_BACKENDS = {
    # The family's Triton kernels; they only visit the keys each tile of queries can reach.
    "triton": NeighbourhoodBackend(unsupported=frozenset({"dropout_p", "window_size"}), needs_triton=True),
    # PyTorch attention with a dense queries x keys mask, for CPUs, small grids and tests.
    "sdpa": NeighbourhoodBackend(unsupported=frozenset({"window_size"})),
}
"""Backends shared by all grid families, by config name."""


@dataclass(frozen=True, kw_only=True)
class GridFamily:
    """A grid family with neighbourhood attention.

    Classes and functions are given as ``"module:name"`` and imported on first use, so the registry
    can be read without Triton.
    """

    description: str
    neighbourhood: str
    "The family's :class:`Neighbourhood` subclass."
    self_attention: str
    "Triton kernel for self attention."
    cross_attention: Optional[str] = None
    "Triton kernel for cross attention; None if the family only does self attention."
    is_member: Optional[str] = None
    "Reduced grids only: test whether a grid belongs to the family."


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
"""Grid families with neighbourhood attention, by config name."""


def grid_family(name: Optional[str]) -> GridFamily:
    """The registered grid family ``name``."""
    if name not in GRID_FAMILIES:
        raise ValueError(f"No neighbourhood attention kernels for grid '{name}'. Known grids: {list(GRID_FAMILIES)}.")
    return GRID_FAMILIES[name]


def check_settings(
    config: dict, settings: set[str], key_coords: Optional[Tensor], query_coords: Optional[Tensor]
) -> tuple[str, str]:
    """Check the settings all grid families share.

    Parameters
    ----------
    config : dict
        The ``neighbourhood`` section of the config.
    settings : set[str]
        The settings the family accepts.
    key_coords, query_coords : Tensor, optional
        Node coordinates; ``query_coords`` is None for self attention.

    Returns
    -------
    tuple[str, str]
        Family name and backend name (default ``"triton"``).
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
    """Grids and settings of the neighbourhood attention of one model component.

    Each grid family subclasses it. One instance is shared by all layers of a component.
    """

    family: str
    "Name in :data:`GRID_FAMILIES`."
    backend: str
    "Name in :data:`NEIGHBOURHOOD_BACKENDS`."

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
        """Check the config and the nodes, and build the neighbourhood."""

    @abstractmethod
    def attention_module(self) -> NeighbourhoodAttention:
        """Attention module over all points."""

    @abstractmethod
    def layer_split(self, cpu_offload: bool) -> Optional[LayerSplit]:
        """How a component splits its queries into bands, or None to always run all points at once."""


class NeighbourhoodAttention(AttentionWrapper, ABC):
    """Base class of the neighbourhood attention modules.

    Takes ``(batch, heads, points, head_dim)`` tensors and runs either the family's Triton kernel or
    ``scaled_dot_product_attention`` with a dense mask. Subclasses build the mask and call the kernel.
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
        # One mask per device, built on first use.
        self._masks: dict[torch.device, Tensor] = {}
        self._kernel: Optional[Callable] = None

    def __getstate__(self) -> dict:
        # Masks and kernel are rebuilt on first use, so they are not saved with the model.
        state = self.__dict__.copy()
        state["_masks"] = {}
        state["_kernel"] = None
        return state

    @abstractmethod
    def build_mask(self, device: torch.device) -> Tensor:
        """Boolean ``(queries, keys)`` mask, True where a query attends to a key."""

    @abstractmethod
    def kernel_path(self) -> str:
        """Import path of the Triton kernel."""

    @abstractmethod
    def run_kernel(self, kernel: Callable, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Run the Triton kernel ``kernel``."""

    def check_lengths(self, query: Tensor, key: Tensor, num_queries: int, num_keys: int) -> None:
        """Raise if ``query`` and ``key`` do not have the numbers of points the module was built for."""
        if query.shape[-2] != num_queries or key.shape[-2] != num_keys:
            raise ValueError(
                f"{self.describe()} is set up for {num_queries} queries and {num_keys} keys, "
                f"got {query.shape[-2]} and {key.shape[-2]}."
            )

    def attend(self, query: Tensor, key: Tensor, value: Tensor, dropout_p: float) -> Tensor:
        """Run the backend on queries and keys in grid order."""
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
    """A set of query points and the key points they reach.

    On reduced grids a band is a run of whole query rows. On the icosahedron it is the points one
    GPU owns, and its keys are those points followed by the ones fetched from other GPUs. In self
    attention the keys include the queries.
    """

    query_points: slice
    key_points: slice | Tensor
    "A run of points, or point indices in the order they are held."
    queries_in_keys: Optional[slice]
    "Position of the queries among the keys, in self attention."
    attention: NeighbourhoodAttention


class GpuWork(ABC):
    """The bands one GPU runs in a layer, how their inputs are gathered and their outputs kept.

    The inputs of all key points are gathered once per layer and each band takes its part. Across
    GPUs, gathering fetches points from other GPUs; on one GPU it only reshapes.
    """

    bands: list[NeighbourhoodBand]
    num_own_queries: int
    "Number of query points whose outputs this GPU returns."

    @abstractmethod
    def gather_keys(self, x: Tensor) -> object:
        """Inputs of all key points the bands reach, from this GPU's ``(batch * points, channels)`` inputs."""

    @abstractmethod
    def keys_of(self, keys: object, band: NeighbourhoodBand) -> Tensor:
        """The band's key inputs, ``(batch, band key points, channels)``."""

    def gather_queries(self, x: Tensor) -> object:
        """Inputs of all query points of the bands, in cross attention."""
        raise NotImplementedError(f"{type(self).__name__} only does self attention.")

    def queries_of(self, queries: object, band: NeighbourhoodBand) -> Tensor:
        """The band's query inputs, ``(batch, band query points, channels)``."""
        raise NotImplementedError(f"{type(self).__name__} only does self attention.")

    @abstractmethod
    def keep(self, out: Optional[Tensor], band_out: Tensor, band: NeighbourhoodBand) -> Tensor:
        """Write the outputs of the band's query points this GPU owns into ``out``, ``(batch, own queries, channels)``.

        ``out`` is None for the first band.
        """


class LayerSplit(ABC):
    """How a model component splits its queries into bands, on one GPU or across GPUs."""

    @abstractmethod
    def used(self, sharded: bool) -> bool:
        """Whether the component runs band by band rather than on all points at once."""

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
            GPUs the points are split across; None on one GPU.
        query_shard_sizes, key_shard_sizes : list[int], optional
            Query and key points per GPU, in rank order. The key sizes default to the query sizes.

        Returns
        -------
        GpuWork
            The bands of this GPU and how to gather their inputs.
        """


def build_grid_neighbourhood(
    attention_implementation: str,
    config: Optional[dict],
    key_coords: Optional[Tensor],
    query_coords: Optional[Tensor] = None,
) -> Optional[Neighbourhood]:
    """Build the neighbourhood of a model component; None if it does not use neighbourhood attention.

    Parameters
    ----------
    attention_implementation : str
        The component's attention implementation.
    config : dict, optional
        The ``neighbourhood`` section: ``grid`` (a key of :data:`GRID_FAMILIES`), optional
        ``backend`` (default ``"triton"``) and the family's own settings.
    key_coords : Tensor, optional
        Latitude and longitude of the key nodes in radians, ``(num_keys, 2)``.
    query_coords : Tensor, optional
        Same for the query nodes; None for self attention.

    Returns
    -------
    Neighbourhood or None
        The neighbourhood of the configured grid family.
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
