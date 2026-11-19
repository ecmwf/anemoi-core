# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fetch a run of consecutive points that overlaps the shards of neighbouring GPUs.

Each GPU owns a run of consecutive points (its shard), in rank order. Some layers need a longer run,
e.g. the rows around their own rows. Such a run is held in three parts (:class:`PointRun`): points
of lower ranks, the own shard read in place, and points of higher ranks. Only the outer parts are
sent; their gradients go back to the owning GPUs.
"""

import itertools
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.primitives import _alltoallwrapper


@dataclass(frozen=True)
class PointRangeExchange:
    """What one GPU sends and receives to get the run of points it wants."""

    send_indices: tuple[Tensor, ...]
    "Per GPU, positions in this GPU's shard of the points to send it."
    recv_counts: tuple[int, ...]
    "Per GPU, number of points to receive from it."
    wanted: slice
    "The run this GPU wants, in global point numbers."
    own_shard: slice
    "This GPU's shard, in global point numbers."

    @property
    def num_before(self) -> int:
        """Number of wanted points owned by lower ranks."""
        return max(0, min(self.wanted.stop, self.own_shard.start) - self.wanted.start)

    @property
    def own_part(self) -> slice:
        """Wanted points this GPU owns, as positions in its shard."""
        first = max(self.wanted.start, self.own_shard.start)
        stop = max(min(self.wanted.stop, self.own_shard.stop), first)
        return slice(first - self.own_shard.start, stop - self.own_shard.start)


def build_point_range_exchange(
    shard_sizes: list[int], wanted: list[slice], rank: int, device: torch.device
) -> PointRangeExchange:
    """Plan the exchange that gives every GPU the points it wants.

    All GPUs call this with the same arguments, so they agree on the plan without communicating.

    Parameters
    ----------
    shard_sizes : list[int]
        Number of points each GPU owns, in rank order.
    wanted : list[slice]
        Run of points each GPU wants, in rank order.
    rank : int
        Rank of this GPU.
    device : torch.device
        Where to put the index tensors.

    Returns
    -------
    PointRangeExchange
        The exchange seen from this GPU.
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
        first = max(a.start, b.start)
        return slice(first, max(min(a.stop, b.stop), first))

    # Send each GPU the part of this shard it wants.
    send_indices = []
    for other, want in enumerate(wanted):
        part = overlap(own, want) if other != rank else slice(own.start, own.start)
        send_indices.append(torch.arange(part.start - own.start, part.stop - own.start, device=device))

    # Receive from each GPU the part of its shard this GPU wants.
    recv_counts = []
    for other, shard in enumerate(shards):
        part = overlap(shard, wanted[rank])
        recv_counts.append(part.stop - part.start if other != rank else 0)

    return PointRangeExchange(
        send_indices=tuple(send_indices),
        recv_counts=tuple(recv_counts),
        wanted=wanted[rank],
        own_shard=own,
    )


class _FetchPoints(torch.autograd.Function):
    """Send every GPU the points it wants from this shard; the backward sends the gradients back.

    Returns the own wanted points (a view of the shard) and the received points in rank order. The
    own points pass through here too, so the backward runs after everything that reads them, at the
    same place on every GPU, which keeps the exchanges of all GPUs in step.
    """

    @staticmethod
    def forward(ctx, x: Tensor, send_indices: tuple[Tensor, ...], recv_counts: tuple[int, ...], own_part: slice, group):
        ctx.send_indices, ctx.recv_counts, ctx.own_part, ctx.group = send_indices, recv_counts, own_part, group
        ctx.shape = x.shape
        send = [x[idx].contiguous() for idx in send_indices]
        recv = [x.new_empty((count, *x.shape[1:])) for count in recv_counts]
        _alltoallwrapper(recv, send, group=group)
        return x[own_part], torch.cat(recv)

    @staticmethod
    def backward(ctx, grad_own: Tensor, grad_received: Tensor) -> tuple[Tensor, None, None, None, None]:
        send = [g.contiguous() for g in torch.split(grad_received, list(ctx.recv_counts))]
        recv = [grad_received.new_empty((len(idx), *grad_received.shape[1:])) for idx in ctx.send_indices]
        _alltoallwrapper(recv, send, group=ctx.group)
        if ctx.own_part == slice(0, ctx.shape[0]):
            grad_x = grad_own.contiguous()
        else:
            grad_x = grad_own.new_zeros(ctx.shape)
            grad_x[ctx.own_part] = grad_own
        grad_x.index_add_(0, torch.cat(list(ctx.send_indices)), torch.cat(recv))
        return grad_x, None, None, None, None


@dataclass(frozen=True)
class PointRun:
    """A run of consecutive points in three parts, ``(batch, points, channels)`` each.

    ``before`` and ``after`` hold points of lower and higher ranks, ``own`` a view of this GPU's
    points. On one GPU the whole run is ``own``.
    """

    start: int
    "Global number of the run's first point."
    before: Tensor
    own: Tensor
    after: Tensor

    @classmethod
    def whole(cls, x: Tensor) -> "PointRun":
        """A run whose own part is all of ``x``, ``(batch, points, channels)``."""
        return cls(start=0, before=x[:, :0], own=x, after=x[:, :0])

    @property
    def stop(self) -> int:
        return self.start + self.before.shape[1] + self.own.shape[1] + self.after.shape[1]

    def take(self, points: slice) -> Tensor:
        """The points ``points`` (global numbers) of the run.

        A view if they lie in one part, else the pieces joined. Always reads the own part, even if
        empty, so the exchange's backward runs last (see :class:`_FetchPoints`).
        """
        if points.start < self.start or points.stop > self.stop:
            raise ValueError(f"The points {points} lie outside the run {self.start}..{self.stop}.")
        pieces, reads_own = [], False
        offset = self.start
        for part in (self.before, self.own, self.after):
            first, stop = max(points.start, offset), min(points.stop, offset + part.shape[1])
            if stop > first:
                pieces.append(part[:, first - offset : stop - offset])
                reads_own = reads_own or part is self.own
            offset += part.shape[1]
        if not reads_own:
            pieces.append(self.own[:, :0])
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def assemble_point_run(own: Tensor, received: Tensor, exchange: PointRangeExchange) -> PointRun:
    """Assemble the wanted run from the own points ``(points, channels)`` and the ``received`` ones."""
    num_before = exchange.num_before
    return PointRun(
        start=exchange.wanted.start,
        before=received[:num_before][None],
        own=own[None],
        after=received[num_before:][None],
    )


def fetch_point_range(x: Tensor, exchange: PointRangeExchange, model_comm_group: Optional[ProcessGroup]) -> PointRun:
    """Fetch the wanted run of points: own ones read in place, the others from their GPUs.

    Parameters
    ----------
    x : Tensor
        This GPU's points, ``(shard points, channels)``.
    exchange : PointRangeExchange
        From :func:`build_point_range_exchange`.
    model_comm_group : ProcessGroup, optional
        GPUs the points are split across.

    Returns
    -------
    PointRun
        The wanted points, batch dimension 1.
    """
    own, received = _FetchPoints.apply(
        x, exchange.send_indices, exchange.recv_counts, exchange.own_part, model_comm_group
    )
    return assemble_point_run(own, received, exchange)
