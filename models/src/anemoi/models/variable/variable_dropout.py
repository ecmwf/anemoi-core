# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch import nn


class VariableDropout(nn.Module):
    """Drop one or more prognostic input fields while retaining at least one."""

    def __init__(
        self,
        dropout_rate: float,
        multi_variable_dropout: bool = False,
        max_drop: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1")
        if not isinstance(max_drop, int) or max_drop < 1:
            raise ValueError("max_drop must be a positive integer")

        self.dropout_rate = dropout_rate
        self.multi_variable_dropout = multi_variable_dropout
        self.max_drop = max_drop

    def forward(
        self,
        x: torch.Tensor,
        names: list[str],
        prognostic_indices: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Return the retained fields, their names, and original dropped indices."""
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        if not self.training or self.dropout_rate == 0.0:
            return x, names, empty
        if torch.rand((), device=x.device) >= self.dropout_rate:
            return x, names, empty

        prognostic = torch.as_tensor(prognostic_indices, dtype=torch.long, device=x.device)
        if prognostic.numel() <= 1:
            return x, names, empty

        max_drop = min(self.max_drop if self.multi_variable_dropout else 1, prognostic.numel() - 1)
        num_drop = int(torch.randint(1, max_drop + 1, (), device=x.device)) if self.multi_variable_dropout else 1
        dropped = prognostic[torch.randperm(prognostic.numel(), device=x.device)[:num_drop]]

        keep = torch.ones(len(names), dtype=torch.bool, device=x.device)
        keep[dropped] = False
        retained_names = [name for name, retain in zip(names, keep.tolist(), strict=True) if retain]
        return x[..., keep], retained_names, dropped
