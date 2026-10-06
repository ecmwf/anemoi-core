# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Hold a run of consecutive points, part of it owned by other GPUs.

The points of a grid are split across the GPUs of a model group in runs of consecutive points,
one run per GPU in rank order (its shard). Some layers need, on each GPU, a run of points that
overlaps its own shard and those of its neighbours, for example the rows around its own rows.

Such a run is held in three parts (:class:`PointRun`): the points owned by GPUs of lower rank,
the GPU's own points, read from its shard in place, and the points owned by GPUs of higher rank.
Only the first and last part are sent between GPUs (:func:`fetch_point_range`); their gradient
goes back to the GPUs that own the points, where the contributions of all GPUs are added up.
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
    """What one GPU sends and receives so that it can hold the run of points it asked for."""

    send_indices: tuple[Tensor, ...]
    "For each GPU, the positions within this GPU's shard of the points to send it."
    recv_counts: tuple[int, ...]
    "For each GPU, how many points to receive from it."
    wanted: slice
    "The run of points this GPU asked for, numbered over the whole grid."
    own_shard: slice
    "This GPU's shard, numbered over the whole grid."

    @property
    def num_before(self) -> int:
        """How many of the wanted points belong to GPUs of lower rank, and so come before the own ones."""
        return max(0, min(self.wanted.stop, self.own_shard.start) - self.wanted.start)

    @property
    def own_part(self) -> slice:
        """The wanted points that this GPU owns, as positions within its shard."""
        first = max(self.wanted.start, self.own_shard.start)
        stop = max(min(self.wanted.stop, self.own_shard.stop), first)
        return slice(first - self.own_shard.start, stop - self.own_shard.start)


def build_point_range_exchange(
    shard_sizes: list[int], wanted: list[slice], rank: int, device: torch.device
) -> PointRangeExchange:
    """Plan the exchange that gives every GPU the points it wants from the other GPUs.

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
        first = max(a.start, b.start)
        return slice(first, max(min(a.stop, b.stop), first))

    # This GPU sends every other GPU the part of its own shard that the other GPU wants.
    send_indices = []
    for other, want in enumerate(wanted):
        part = overlap(own, want) if other != rank else slice(own.start, own.start)
        send_indices.append(torch.arange(part.start - own.start, part.stop - own.start, device=device))

    # It receives from every other GPU the part of that GPU's shard that it wants itself.
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
    """Sends every GPU the points it asked for from this GPU's shard.

    Returns this GPU's own wanted points, a view of its shard, and the points received from the
    other GPUs in rank order, which is also point order, since the shards follow each other in
    rank order. The backward pass sends the gradients of the received points back to the GPUs
    that own them and adds what comes back for the points this GPU sent at their places in its
    shard.

    The own points go through this function as well, so that everything reading them depends on
    it. Its backward then runs after all of them, at the same point on every GPU, which keeps the
    exchanges of all GPUs in the same order.
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
    """A run of consecutive points held in three parts, laid out ``(batch, points, channels)``.

    ``before`` holds the points owned by GPUs of lower rank, ``own`` a view of this GPU's own points
    and ``after`` the points owned by GPUs of higher rank. On a single GPU the whole run is ``own``.
    """

    start: int
    "Number of the run's first point, counted over the whole grid."
    before: Tensor
    own: Tensor
    after: Tensor

    @classmethod
    def whole(cls, x: Tensor) -> "PointRun":
        """All points of ``x``, laid out ``(batch, points, channels)``, as the run's own part."""
        return cls(start=0, before=x[:, :0], own=x, after=x[:, :0])

    @property
    def stop(self) -> int:
        return self.start + self.before.shape[1] + self.own.shape[1] + self.after.shape[1]

    def take(self, points: slice) -> Tensor:
        """The points ``points`` (numbered over the whole grid) of the run.

        A view of one part when they all lie in it; otherwise the overlapping pieces joined. The
        result always reads the own part, if need be none of its points, so that the exchange's
        backward runs only after everything taken from the run (see :class:`_FetchPoints`).
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
    """The wanted run from this GPU's own wanted points and the points ``received`` from the other GPUs.

    ``own`` is laid out ``(points, channels)`` and ``received`` holds the received points in rank
    order, as :class:`_FetchPoints` returns them.
    """
    num_before = exchange.num_before
    return PointRun(
        start=exchange.wanted.start,
        before=received[:num_before][None],
        own=own[None],
        after=received[num_before:][None],
    )


def fetch_point_range(x: Tensor, exchange: PointRangeExchange, model_comm_group: Optional[ProcessGroup]) -> PointRun:
    """The wanted run of points: the own ones read from ``x`` in place, the others fetched from their GPUs.

    Parameters
    ----------
    x : Tensor
        This GPU's own points, shape ``(points in the shard, channels)``.
    exchange : PointRangeExchange
        The exchange planned by :func:`build_point_range_exchange`.
    model_comm_group : ProcessGroup, optional
        The GPUs the points are split across.

    Returns
    -------
    PointRun
        The wanted points, with a batch dimension of 1.
    """
    own, received = _FetchPoints.apply(
        x, exchange.send_indices, exchange.recv_counts, exchange.own_part, model_comm_group
    )
    return assemble_point_run(own, received, exchange)
