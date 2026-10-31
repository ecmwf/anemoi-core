# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import torch
from torch_geometric.data import HeteroData


class Timedeltas:
    """Encode runtime per-node time offsets.

    Input values are signed seconds. The first output feature is the offset in
    scaled units, followed by sine/cosine pairs for each configured period.
    """

    def __init__(
        self,
        scale_seconds: float = 3600.0,
        periods: list[float] | None = None,
        dtype: str = "float32",
    ) -> None:
        if not math.isfinite(scale_seconds) or scale_seconds <= 0:
            raise ValueError(f"scale_seconds must be positive and finite, got {scale_seconds}.")

        self.periods = [] if periods is None else list(periods)
        if any(not math.isfinite(period) or period <= 0 for period in self.periods):
            raise ValueError(f"periods must contain only positive finite values, got {self.periods}.")

        torch_dtype = getattr(torch, dtype, None)
        if not isinstance(torch_dtype, torch.dtype) or not torch_dtype.is_floating_point:
            raise ValueError(f"dtype must name a floating-point torch dtype, got {dtype!r}.")

        self.scale_seconds = float(scale_seconds)
        self.dtype = torch_dtype

    @property
    def ndim(self) -> int:
        """Number of output features per node."""
        return 1 + 2 * len(self.periods)

    def compute(self, timedeltas: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Encode signed timedelta seconds into scalar and Fourier features."""
        if isinstance(timedeltas, HeteroData):
            raise RuntimeError(
                "Timedeltas is a runtime-only node attribute and cannot be computed by GraphCreator; "
                "call compute(timedeltas) with a per-node tensor at model runtime."
            )
        if args or kwargs:
            raise TypeError("Timedeltas.compute accepts only a per-node timedeltas tensor.")
        if not isinstance(timedeltas, torch.Tensor):
            raise TypeError(f"timedeltas must be a torch.Tensor, got {type(timedeltas).__name__}.")
        if timedeltas.ndim == 1:
            timedeltas = timedeltas.unsqueeze(-1)
        elif timedeltas.ndim != 2 or timedeltas.shape[1] != 1:
            raise ValueError(
                f"timedeltas must have shape (num_nodes,) or (num_nodes, 1), got {tuple(timedeltas.shape)}."
            )

        scaled = timedeltas.to(dtype=self.dtype) / self.scale_seconds
        features = [scaled]
        for period in self.periods:
            phase = 2 * torch.pi * scaled / period
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=-1)
