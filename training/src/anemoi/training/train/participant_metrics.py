# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-participant metric accumulation for multi-domain training.

Every batch holds a single participant (see :mod:`anemoi.training.data.batch_meta`),
so per-participant metrics can be derived from the ordinary per-batch metrics.
They cannot be logged with ``LightningModule.log(..., sync_dist=True)`` though:
Lightning requires every rank to log the same keys in the same order, while the
participants seen by a rank in a step (or even an epoch) can differ. This module
accumulates weighted sums locally and reduces them explicitly at epoch end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping


class ParticipantMetrics:
    """Accumulate ``{participant: {metric: mean}}`` over an epoch across all ranks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop all accumulated values."""
        # participant -> metric -> tensor([weighted sum, total weight]) in float64
        self._sums: dict[str, dict[str, torch.Tensor]] = {}

    def update(self, participant: str | None, metrics: Mapping[str, torch.Tensor], batch_size: int) -> None:
        """Add the metrics of one batch of ``participant``; a no-op when ``participant`` is None."""
        if participant is None:
            return
        accumulated = self._sums.setdefault(participant, {})
        for name, value in metrics.items():
            weight = torch.tensor(batch_size, dtype=torch.float64, device=value.device)
            entry = torch.stack([value.detach().double().reshape(()) * weight, weight])
            accumulated[name] = accumulated[name] + entry if name in accumulated else entry

    def compute(self, device: torch.device | str | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """Return the per-participant means, reduced over all ranks.

        This is a collective call: every rank must call it (even ranks that saw no
        participant), so the set of keys is agreed on first and every rank then
        takes part in one all-reduce per key, in the same order.
        """
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        local_keys = sorted((p, metric) for p, metrics in self._sums.items() for metric in metrics)
        if distributed:
            gathered: list[list[tuple[str, str]] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local_keys)
            keys = sorted(set().union(*gathered))
        else:
            keys = local_keys

        results: dict[str, dict[str, torch.Tensor]] = {}
        for participant, metric in keys:
            entry = self._sums.get(participant, {}).get(metric)
            if entry is None:
                entry = torch.zeros(2, dtype=torch.float64)
            entry = entry.to(device) if device is not None else entry
            if distributed:
                torch.distributed.all_reduce(entry)
            if entry[1] > 0:
                results.setdefault(participant, {})[metric] = (entry[0] / entry[1]).float()
        return results
