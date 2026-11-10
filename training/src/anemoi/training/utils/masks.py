# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from hydra.utils import instantiate
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.models.data_indices.collection import IndexCollection


class BaseMask:
    """Base class for masking model output."""

    def __init__(self, *_args, **_kwargs) -> None:
        """Initialize base mask."""

    @property
    def supporting_arrays(self) -> dict:
        return {}

    @abstractmethod
    def as_tuple(self) -> tuple:
        """Return the range of contiguous True values in the mask as a tuple (start, end)."""
        error_message = "Method `as_tuple` must be implemented in subclass."
        raise NotImplementedError(error_message)

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

    def __init__(self, nodes: NodeStorage, attribute_name: str) -> None:
        super().__init__()
        assert attribute_name in nodes, f"{self.__class__.__name__} cannot find attribute '{attribute_name}' in nodes."
        mask = nodes[attribute_name].bool().squeeze()
        self.register_buffer("mask", mask)

    @property
    def supporting_arrays(self) -> dict:
        return {"output_mask": self.mask.numpy()}

    def as_tuple(self) -> tuple:
        n = int(self.mask.sum())
        first = int(self.mask.int().argmax())
        assert bool(
            self.mask[first : first + n].all(),
        ), "Currently only output_masks with a contiguous block of True values are supported."
        return (first, first + n)

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
            indices = (~mask).nonzero(as_tuple=True)[0].to(x.device)
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

    def as_tuple(self) -> tuple:
        return (None, None)

    def apply(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x

    def rollout_boundary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return x


def build_output_masks(output_mask_configs: dict, graph_data: HeteroData) -> dict[str, BaseMask]:
    """Build output masks for each dataset.

    Parameters
    ----------
    output_mask_configs : dict[str, dict]
        Dictionary of output mask configurations for each dataset.
    graph_data : HeteroData
        Dictionary of graph data for each dataset.

    Returns
    -------
    dict[str, BaseMask]
        Dictionary of output masks for each dataset.
    """
    output_masks = defaultdict(lambda: NoOutputMask())
    for dataset_name, output_mask_config in output_mask_configs.items():
        if output_mask_config is not None:
            assert dataset_name in graph_data.node_types, f"Dataset '{dataset_name}' not found in graph_data."
            output_masks[dataset_name] = instantiate(output_mask_config, nodes=graph_data[dataset_name])

    return output_masks
