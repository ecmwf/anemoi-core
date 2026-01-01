# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import abstractmethod

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection


class BaseMask:
    """Base class for masking model output."""

    def __init__(self, *_args, **_kwargs) -> None:
        """Initialize base mask."""

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


class Boolean1DMask(torch.nn.Module, BaseMask):
    """1D Boolean mask."""

    def __init__(self, graph_data: HeteroData, nodes_name: str, attribute_name: str) -> None:
        super().__init__()

        mask = graph_data[nodes_name][attribute_name].bool().squeeze()
        self.register_buffer("mask", mask)

    @property
    def supporting_arrays(self) -> dict:
        return {"output_mask": self.mask.numpy()}

    def broadcast_like(self, x: torch.Tensor, dim: int, grid_shard_slice: slice | None = None) -> torch.Tensor:
        assert x.shape[dim] == len(
            self.mask,
        ), f"Dimension mismatch: dimension {dim} has size {x.shape[dim]}, but mask length is {len(self.mask)}."
        target_shape = [1 for _ in range(x.ndim)]
        target_shape[dim] = len(self.mask)
        mask = self.mask[grid_shard_slice] if grid_shard_slice is not None else self.mask
        return mask.reshape(target_shape)

    @staticmethod
    def _fill_tensor_with_tensor(
        x: torch.Tensor,
        indices: torch.Tensor,
        fill_value: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        assert fill_value.ndim == 4, "fill_value has to be shape (bs, ens, latlon, nvar)"
        fill_value = torch.index_select(fill_value, dim, indices)  # The mask is applied over the latlon dim
        return x.index_copy_(dim, indices, fill_value)

    @staticmethod
    def _fill_tensor_with_float(x: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
        return x.masked_fill(mask, fill_value)

    def apply(
        self,
        x: torch.Tensor,
        dim: int,
        fill_value: float | torch.Tensor = np.nan,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Apply the mask to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be masked.
        dim : int
            The dimension along which to apply the mask.
        fill_value : float | torch.Tensor, optional
            The value to fill in the masked positions, by default np.nan.

        Returns
        -------
        torch.Tensor
            The masked tensor with fill_value in the positions where the mask is False.
        """
        mask = self.mask[grid_shard_slice] if grid_shard_slice is not None else self.mask

        if isinstance(fill_value, torch.Tensor):
            indices = (~mask).nonzero(as_tuple=True)[0]
            return Boolean1DMask._fill_tensor_with_tensor(x, indices, fill_value, dim)

        mask = self.broadcast_like(x, dim, grid_shard_slice).cpu()
        return Boolean1DMask._fill_tensor_with_float(x, ~mask, fill_value)

    def rollout_boundary(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        data_indices: IndexCollection,
        grid_shard_slice: slice | None = None,
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
            grid_shard_slice=grid_shard_slice,
        )

        return pred_state


class NoOutputMask(BaseMask):
    """No output mask."""

    def apply(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x

    def rollout_boundary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x
