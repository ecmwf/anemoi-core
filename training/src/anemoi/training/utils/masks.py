# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection


class BaseMask:
    """Base class for masking model output."""

    @property
    def supporting_arrays(self) -> dict:
        return {}

    @abstractmethod
    def apply(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        error_message = "Method `apply` must be implemented in subclass."
        raise NotImplementedError(error_message)

    @abstractmethod
    def rollout_boundary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        error_message = "Method `rollout_boundary` must be implemented in subclass."
        raise NotImplementedError(error_message)


class Boolean1DMask(BaseMask):
    """1D Boolean mask."""

    def __init__(self, values: torch.Tensor) -> None:
        self.mask = values.bool().squeeze()

    @property
    def supporting_arrays(self) -> dict:
        return {"output_mask": self.mask.numpy()}

    def broadcast_like(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert x.shape[dim] == len(
            self.mask,
        ), f"Dimension mismatch: dimension {dim} has size {x.shape[dim]}, but mask length is {len(self.mask)}."
        target_shape = [1 for _ in range(x.ndim)]
        target_shape[dim] = len(self.mask)
        mask = self.mask.reshape(target_shape)
        return mask.to(x.device)

    def masked_select(self, input: torch.Tensor, dim: int, negate: bool = False) -> torch.Tensor:
        mask = self.mask
        if negate:
            mask = ~mask
        assert len(mask) == input.shape[dim], "The mask should fit the size of the chosen dimension"
        args = []
        for i, size in enumerate(input.shape):
            if i != dim:
                args.append(np.arange(size))
            else:
                args.append(mask)
        return input[np.ix_(*args)]

    @staticmethod
    def _fill_masked_tensor(x: torch.Tensor, mask: torch.Tensor, fill_value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(fill_value, torch.Tensor):
            return x.masked_scatter(mask, fill_value)
        return x.masked_fill(mask, fill_value)

    def apply(
        self, x: torch.Tensor, dim: int, fill_value: float | torch.Tensor = np.nan, dim_sel: int = 0,
    ) -> torch.Tensor:
        """Apply the mask to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be masked.
        dim : int
            The dimension along which to apply the mask to x.
        fill_value : float | torch.Tensor, optional
            The value to fill in the masked positions, by default np.nan.
        dim_sel : int, optional
            The dimension along which to select values out of fill_value using the mask

        Returns
        -------
        torch.Tensor
            The masked tensor with fill_value in the positions where the mask is False.
        """
        mask = self.broadcast_like(x, dim)
        fill_selection = fill_value
        if isinstance(fill_value, torch.Tensor):
            fill_selection = self.masked_select(fill_value, dim_sel, negate=True)
        return Boolean1DMask._fill_masked_tensor(x, ~mask, fill_selection)

    def rollout_boundary(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        data_indices: IndexCollection,
    ) -> torch.Tensor:
        """Rollout the boundary forcing.

        Parameters
        ----------
        pred_state : torch.Tensor
            The predicted state tensor of shape (bs, ens, latlon, nvar)
        true_state : torch.Tensor
            The true state tensor of shape (bs, ens, latlon, nvar)
        data_indices : IndexCollection
            Collection of data indices.

        Returns
        -------
        torch.Tensor
            The updated predicted state tensor with boundary forcing applied.
        """
        pred_state[..., data_indices.model.input.prognostic] = self.apply(
            pred_state[..., data_indices.model.input.prognostic],
            dim=2,
            fill_value=true_state[..., data_indices.data.output.prognostic],
            dim_sel=2,
        )

        return pred_state


class NoOutputMask(BaseMask):
    """No output mask."""

    def apply(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x

    def rollout_boundary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x
