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
    def __init__(
        self,
        dropout_rate: float,
        multi_variable_dropout: bool = False,
        max_drop: int = 1,
    ) -> None:
        """Initializes the VariableDropout layer.

        Parameters
        ----------
        dropout_rate : float
            The rate at which to drop variables.
        multi_variable_dropout : bool, optional
            Whether to apply multi-variable dropout, by default False
        max_drop : int, optional
            The maximum number of variables to drop, by default 1
        """

        super().__init__()

        assert 0.0 <= dropout_rate <= 1.0, "Dropout rate must be between 0 and 1."
        assert isinstance(max_drop, int) and max_drop > 0, "max_drop must be a positive integer."

        self.dropout_rate = dropout_rate
        self.multi_variable_dropout = multi_variable_dropout
        self.max_drop = max_drop

    def forward(
        self, x: torch.Tensor, names: list[str], prognostic_indices: list[int]
    ) -> tuple[torch.Tensor, list[str]]:
        """Forward pass for the VariableDropout layer.
        args:
        x : torch.Tensor
            The input tensor of shape (batch_size x time, num_ens, gridpoints,num_variables).
        names : list[str]
            The list of variable names corresponding to the last dimension of x.
        prognostic_indices : list[int]
            The indices of the prognostic variables in the names list.
        return:
        tuple[torch.Tensor, list[str]]
            The output tensor after dropout and the updated list of variable names.
        """
        if not self.training or self.dropout_rate == 0.0:
            return x, names

        if torch.rand((), device=x.device) >= self.dropout_rate:
            # no dropout applied, return original tensor and names
            return x, names

        # keep = torch.ones(len(names), dtype=torch.bool, device=x.device)
        prognostic_indices = torch.as_tensor(prognostic_indices, dtype=torch.long, device=x.device)

        num_prognostic = len(prognostic_indices)

        if num_prognostic <= 1:
            return x, names

        if not self.multi_variable_dropout:
            num_drop = 1
        else:
            max_drop = min(self.max_drop, num_prognostic - 1)
            num_drop = torch.randint(1, max_drop + 1, size=(1,), device=x.device).item()

        perm = torch.randperm(num_prognostic, device=x.device)
        drop_indices = prognostic_indices[perm[:num_drop]]

        keep = torch.ones(len(names), dtype=torch.bool, device=x.device)
        keep[drop_indices] = False
        x = x[..., keep]

        keep_cpu = keep.tolist()
        names = [name for name, keep_variable in zip(names, keep_cpu) if keep_variable]

        return x, names
