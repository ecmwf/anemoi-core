# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


class SparseProjector(torch.nn.Module):
    """Applies sparse projection matrix to input tensors.

    Stateless projector that receives matrix as input to forward().
    """

    def __init__(self, autocast: bool = False) -> None:
        """Initialize SparseProjector.

        Parameters
        ----------
        autocast : bool
            Use automatic mixed precision
        """
        super().__init__()
        self.autocast = autocast

    def forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Apply sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on correct device)

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        out = []
        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(projection_matrix, x[i, ...]))
        return torch.stack(out)
