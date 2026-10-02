# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fetch a run of consecutive points from the GPUs that own them.

The points of a grid are split across the GPUs of a model group in runs of consecutive points,
one run per GPU in rank order (its shard). Some layers need, on each GPU, a run of points that
overlaps its own shard and those of its neighbours, for example the rows around its own rows.
:func:`fetch_point_range` gathers that run, and its gradient goes back to the GPUs that own the
points, where the contributions of all GPUs are added up.
"""

import itertools
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import halo_exchange


@dataclass(frozen=True)
class PointRangeExchange:
    """What one GPU sends and receives so that it ends up with the run of points it asked for.

    ``num_local_nodes``, ``send_indices`` and ``recv_counts`` are what :func:`halo_exchange`
    reads: it returns this GPU's own points followed by the points received from each other GPU,
    in rank order. ``order`` then picks the run that was asked for out of that, in point order.
    """

    num_local_nodes: int
    "Number of points this GPU owns."
    send_indices: tuple[Tensor, ...]
    "For each GPU, the positions within this GPU's shard of the points to send it."
    recv_counts: tuple[int, ...]
    "For each GPU, how many points to receive from it."
    order: Tensor
    "Positions in the halo exchange output of the points asked for, in point order."


def build_point_range_exchange(
    shard_sizes: list[int], wanted: list[slice], rank: int, device: torch.device
) -> PointRangeExchange:
    """Plan the exchange that gives every GPU the run of points it wants.

    Every GPU calls this with the same ``shard_sizes`` and ``wanted``, so they all agree on what
    is sent where without talking to each other.

    Parameters
    ----------
    shard_sizes : list[int]
        Number of points each GPU owns, in rank order; GPU ``r`` owns the run after those of GPUs ``0`` to ``r - 1``.
    wanted : list[slice]
        The run of points each GPU wants, in rank order.
    rank : int
        The rank of this GPU.
    device : torch.device
        Where to put the index tensors.

    Returns
    -------
    PointRangeExchange
        The exchange as seen by this GPU.
    """
    if len(wanted) != len(shard_sizes):
        raise ValueError(f"Got {len(wanted)} wanted runs of points for {len(shard_sizes)} GPUs.")
    starts = [0, *itertools.accumulate(shard_sizes)]
    shards = [slice(start, stop) for start, stop in zip(starts[:-1], starts[1:])]
    own = shards[rank]
    for want in wanted:
        if want.start < 0 or want.stop > starts[-1] or want.stop < want.start:
            raise ValueError(f"The wanted points {want} lie outside the {starts[-1]} points.")

    def overlap(a: slice, b: slice) -> slice:
        return slice(max(a.start, b.start), max(min(a.stop, b.stop), max(a.start, b.start)))

    # This GPU sends every other GPU the part of its own shard that the other GPU wants.
    send_indices = []
    for other, want in enumerate(wanted):
        part = overlap(own, want) if other != rank else slice(own.start, own.start)
        send_indices.append(torch.arange(part.start - own.start, part.stop - own.start, device=device))

    # It receives from every other GPU the part of that GPU's shard that it wants itself.
    parts = [overlap(shard, wanted[rank]) for shard in shards]
    recv_counts = [part.stop - part.start if other != rank else 0 for other, part in enumerate(parts)]

    # The halo exchange output is the own shard, then what came from GPU 0, 1, ... in turn. The
    # shards follow each other in rank order, so going through the GPUs in rank order gives the
    # wanted points in point order.
    recv_offsets = [own.stop - own.start, *(own.stop - own.start + c for c in itertools.accumulate(recv_counts))]
    order = []
    for other, part in enumerate(parts):
        if part.stop == part.start:
            continue
        if other == rank:
            order.append(torch.arange(part.start - own.start, part.stop - own.start, device=device))
        else:
            order.append(torch.arange(recv_offsets[other], recv_offsets[other] + part.stop - part.start, device=device))
    return PointRangeExchange(
        num_local_nodes=own.stop - own.start,
        send_indices=tuple(send_indices),
        recv_counts=tuple(recv_counts),
        order=torch.cat(order),
    )


def fetch_point_range(x: Tensor, exchange: PointRangeExchange, model_comm_group: Optional[ProcessGroup]) -> Tensor:
    """The wanted run of points, gathered from the GPUs that own them.

    Parameters
    ----------
    x : Tensor
        This GPU's own points, shape ``(points in the shard, ...)``.
    exchange : PointRangeExchange
        The exchange planned by :func:`build_point_range_exchange`.
    model_comm_group : ProcessGroup, optional
        The GPUs the points are split across.

    Returns
    -------
    Tensor
        The wanted points in point order, shape ``(wanted points, ...)``.
    """
    return halo_exchange(x, exchange, model_comm_group)[exchange.order]
