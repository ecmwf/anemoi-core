# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A grid family for tests: the nodes lie on a line, and each query attends to the keys near it.

Query ``i`` of ``num_queries`` sits at key position ``i * num_keys // num_queries`` and attends to
the keys at most ``radius`` positions from there. It fills in every hook of
:mod:`anemoi.models.layers.neighbourhood_attention` with a few lines of PyTorch, so the tests can
run a model component through neighbourhood attention, in one go or band by band, without the
kernels of a real grid. Register it with :func:`register_window_family`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.layers import neighbourhood_attention
from anemoi.models.layers.neighbourhood_attention import GpuWork
from anemoi.models.layers.neighbourhood_attention import GridFamily
from anemoi.models.layers.neighbourhood_attention import LayerSplit
from anemoi.models.layers.neighbourhood_attention import Neighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttention
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodBand
from anemoi.models.layers.neighbourhood_attention import check_settings

WINDOW_FAMILY = GridFamily(
    description="points on a line, for tests",
    neighbourhood="tests.layers.window_neighbourhood:WindowNeighbourhood",
    self_attention="tests.layers.window_neighbourhood:window_kernel",
    cross_attention="tests.layers.window_neighbourhood:window_kernel",
)

KERNEL_CALLS: list[tuple[int, int]] = []
"Numbers of queries and keys of each call of :func:`window_kernel`."


def register_window_family(monkeypatch) -> None:
    """Add the family to the registry under the name ``"window"`` for one test."""
    monkeypatch.setitem(neighbourhood_attention.GRID_FAMILIES, "window", WINDOW_FAMILY)


def window_kernel(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
    """Stands in for a Triton kernel: PyTorch attention with the mask, and a note of the call."""
    KERNEL_CALLS.append((query.shape[-2], key.shape[-2]))
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


@dataclass(frozen=True, eq=False)
class WindowNeighbourhood(Neighbourhood):
    """A run of queries and the run of keys they see, on the line of all queries and all keys."""

    family: str
    backend: str
    radius: int
    total_queries: int
    total_keys: int
    is_self_attention: bool
    num_bands: int = 1
    queries: slice = slice(0, 0)
    keys: slice = slice(0, 0)

    @classmethod
    def from_config(
        cls, config: dict, key_coords: Optional[Tensor], query_coords: Optional[Tensor] = None
    ) -> WindowNeighbourhood:
        family, backend = check_settings(config, {"grid", "backend", "radius", "num_bands"}, key_coords, query_coords)
        if "radius" not in config:
            raise ValueError("The window family needs a radius.")
        num_keys = key_coords.shape[0]
        num_queries = num_keys if query_coords is None else query_coords.shape[0]
        return cls(
            family,
            backend,
            int(config["radius"]),
            num_queries,
            num_keys,
            is_self_attention=query_coords is None,
            num_bands=int(config.get("num_bands", 1)),
            queries=slice(0, num_queries),
            keys=slice(0, num_keys),
        )

    def centre(self, queries: Tensor) -> Tensor:
        """Key position of each query, both numbered on the whole line."""
        return queries * self.total_keys // self.total_queries

    def keys_reached(self, queries: slice) -> slice:
        """The run of keys that the run of queries ``queries`` sees."""
        ends = self.centre(torch.tensor([queries.start, queries.stop - 1]))
        return slice(max(0, int(ends[0]) - self.radius), min(self.total_keys, int(ends[1]) + self.radius + 1))

    def mask(self, device: torch.device) -> Tensor:
        query_idx = torch.arange(self.queries.start, self.queries.stop, device=device)
        key_idx = torch.arange(self.keys.start, self.keys.stop, device=device)
        return (self.centre(query_idx)[:, None] - key_idx[None, :]).abs() <= self.radius

    def attention_module(self) -> WindowAttention:
        return WindowAttention(self)

    def layer_split(self, cpu_offload: bool) -> Optional[WindowBands]:
        return None if cpu_offload else WindowBands(self)

    def bands(self) -> list[NeighbourhoodBand]:
        """Split the queries into ``num_bands`` runs of about the same length."""
        cuts = torch.linspace(0, self.total_queries, self.num_bands + 1).round().long().tolist()
        bands = []
        for first, stop in zip(cuts[:-1], cuts[1:]):
            queries = slice(first, stop)
            keys = self.keys_reached(queries)
            band = replace(self, is_self_attention=False, num_bands=1, queries=queries, keys=keys)
            bands.append(
                NeighbourhoodBand(
                    query_points=queries,
                    key_points=keys,
                    queries_in_keys=(slice(first - keys.start, stop - keys.start) if self.is_self_attention else None),
                    attention=WindowAttention(band),
                )
            )
        return bands


class WindowAttention(NeighbourhoodAttention):
    def build_mask(self, device: torch.device) -> Tensor:
        return self.neighbourhood.mask(device)

    def kernel_path(self) -> str:
        family = self.neighbourhood.grid_family
        return family.self_attention if self.neighbourhood.is_self_attention else family.cross_attention

    def run_kernel(self, kernel: Callable, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return kernel(query, key, value, self.neighbourhood.mask(query.device))

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
        nb = self.neighbourhood
        self.check_lengths(query, key, nb.queries.stop - nb.queries.start, nb.keys.stop - nb.keys.start)
        return self.attend(query, key, value, dropout_p)


class WindowBands(LayerSplit):
    """Bands on one GPU, used when there are several."""

    def __init__(self, neighbourhood: WindowNeighbourhood) -> None:
        self.neighbourhood = neighbourhood

    def used(self, sharded: bool) -> bool:
        return self.neighbourhood.num_bands > 1

    def work(
        self,
        batch_size: int,
        device: torch.device,
        model_comm_group: Optional[ProcessGroup] = None,
        query_shard_sizes: Optional[list[int]] = None,
        key_shard_sizes: Optional[list[int]] = None,
    ) -> WindowWork:
        if model_comm_group is not None:
            raise NotImplementedError("The window family runs on one GPU.")
        nb = self.neighbourhood
        return WindowWork(nb.bands(), nb.total_queries, nb.total_keys, batch_size)


@dataclass(eq=False)
class WindowWork(GpuWork):
    bands: list[NeighbourhoodBand]
    num_own_queries: int
    num_keys: int
    batch_size: int

    def gather_keys(self, x: Tensor) -> Tensor:
        return x.view(self.batch_size, self.num_keys, -1)

    def keys_of(self, keys: Tensor, band: NeighbourhoodBand) -> Tensor:
        return keys[:, band.key_points]

    def gather_queries(self, x: Tensor) -> Tensor:
        return x.view(self.batch_size, self.num_own_queries, -1)

    def queries_of(self, queries: Tensor, band: NeighbourhoodBand) -> Tensor:
        return queries[:, band.query_points]

    def keep(self, out: Optional[Tensor], band_out: Tensor, band: NeighbourhoodBand) -> Tensor:
        if out is None:
            out = band_out.new_empty(self.batch_size, self.num_own_queries, band_out.shape[-1])
        out[:, band.query_points] = band_out
        return out
