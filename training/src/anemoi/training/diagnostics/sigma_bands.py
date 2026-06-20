# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-sigma-band diffusion-loss tracking.

The diffusion training loss is only a proxy, but the way that proxy is split
across the noise level ``sigma`` is the single most informative thing we can log
cheaply: with the Karras EDM (velocity) formulation the network solves a
genuinely different problem at low vs high ``sigma`` (relative to
``sigma_data``), so a loss that is fine in aggregate can still be dominated by a
particular band. This module accumulates the loss into ``sigma`` bins over an
epoch and reduces once across ranks, so the overhead is a handful of scalars per
epoch rather than anything per step.

All metrics are emitted under a single ``sigma_bands/`` prefix so they cluster
together in the MLflow UI.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

# MLflow groups metrics by the path before the first "/"
SIGMA_BAND_PREFIX = "sigma_bands"

# Bin edges in units of sigma (multiplied by sigma_data at construction time).
# Centred on sigma_data == 1: 
# Training keeps fine resolution; validation collapses to low / mid / high. 
DEFAULT_TRAIN_SIGMA_EDGES: tuple[float, ...] = (0.05, 0.15, 0.4, 1.0, 2.5, 7.0, 20.0)  # -> 8 bands
DEFAULT_VAL_SIGMA_EDGES: tuple[float, ...] = (0.4, 2.5)  # -> 3 bands: low / ~sigma_data / high


def _format_edge(value: float) -> str:
    return f"{value:g}"


def make_band_labels(edges: tuple[float, ...]) -> list[str]:
    """Human-readable, sortable label per band.

    e.g. ``("0_lt0.05", "1_0.05-0.15", ..., "7_ge20")``. The leading index keeps
    the bands ordered in the MLflow UI.
    """
    n_bands = len(edges) + 1
    labels: list[str] = []
    for i in range(n_bands):
        if i == 0:
            labels.append(f"{i}_lt{_format_edge(edges[0])}")
        elif i == n_bands - 1:
            labels.append(f"{i}_ge{_format_edge(edges[-1])}")
        else:
            labels.append(f"{i}_{_format_edge(edges[i - 1])}-{_format_edge(edges[i])}")
    return labels


class SigmaBandLossTracker:
    """Accumulate a scalar loss into ``sigma`` bins over one epoch.

    One instance per stage (train / val). The contract is:

    * :meth:`reset` at the start of every epoch (lazily allocates on ``device``),
    * :meth:`update` once per step with that step's scalar loss and ``sigma``,
    * :meth:`compute` once at epoch end -> a single cross-rank all-reduce.

    Each band's value is a sample-weighted mean (sum of loss / count), so it
    stays correct even when GPUs see different numbers of samples in a band.
    (Grid sharding duplicates a sample across its GPU group, but that cancels in
    the sum/count ratio, so summing over all GPUs is still unbiased.)
    """

    def __init__(self, edges: tuple[float, ...], *, sigma_data: float = 1.0, name: str = "train") -> None:
        self.edges = tuple(float(e) * float(sigma_data) for e in edges)
        self.labels = make_band_labels(self.edges)
        self.name = name
        self.n_bands = len(self.edges) + 1
        self._sum: torch.Tensor | None = None
        self._count: torch.Tensor | None = None

    @property
    def active(self) -> bool:
        return self._sum is not None

    def reset(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self._sum = torch.zeros(self.n_bands, device=device, dtype=dtype)
        self._count = torch.zeros(self.n_bands, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, loss: torch.Tensor, sigma: torch.Tensor) -> None:
        """Add ``loss`` to the band(s) selected by ``sigma``.

        ``loss`` is the (batch-reduced) scalar for the step; ``sigma`` holds one
        value per batch element. With batch size 1 this is exact; for larger
        batches the step's mean loss is attributed to each sample's band, which
        is still a faithful per-band average over the epoch.
        """
        if self._sum is None:
            return
        dtype = self._sum.dtype
        edges = torch.as_tensor(self.edges, device=sigma.device, dtype=sigma.dtype)
        bands = torch.bucketize(sigma.reshape(-1), edges)  # 0 .. n_bands-1
        ones = torch.ones_like(bands, dtype=dtype)
        self._sum.scatter_add_(0, bands, ones * loss.detach().to(dtype))
        self._count.scatter_add_(0, bands, ones)

    def compute(self) -> dict[str, tuple[float, float, float]]:
        """Reduce across ranks and return ``{label: (mean_loss, fraction, count)}``."""
        s, c = self._sum, self._count
        if s is None:
            return {}
        if dist.is_available() and dist.is_initialized():
            packed = torch.stack([s, c])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            s, c = packed[0], packed[1]
        total = c.sum().clamp(min=1.0)
        means = s / c.clamp(min=1.0)
        fracs = c / total
        return {
            self.labels[i]: (means[i].item(), fracs[i].item(), c[i].item())
            for i in range(self.n_bands)
        }
