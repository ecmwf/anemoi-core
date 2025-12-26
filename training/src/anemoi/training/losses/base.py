# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseLoss(nn.Module, ABC):
    """Base loss."""

    scaler: ScaleTensor

    def __init__(self, ignore_nans: bool = False) -> None:
        """Node- and feature_weighted Loss.

        Exposes:
        - self.avg_function: torch.nanmean or torch.mean
        - self.sum_function: torch.nansum or torch.sum
        depending on the value of `ignore_nans`

        Registers:
        - self.scaler: ScaleTensor modified with `add_scaler` and `update_scaler`

        These losses are designed for use within the context of
        the anemoi-training configuration, where scalars are added
        after initialisation. If being used outside of this
        context, call `add_scalar` and `update_scalar` to add or
        update the scale tensors.

        Parameters
        ----------
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.add_module("scaler", ScaleTensor())

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.supports_sharding = True

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        self.scaler.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        self.scaler.update_scaler(name=name, scaler=scaler, override=override)

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Hook to set the data indices for the loss."""

    def scale(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...] | None = None,
        *,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Scale a tensor by the variable_scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs)
        subset_indices: tuple[int,...], optional
            Indices to subset the calculated scaler and `x` tensor with, by default None.
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        if subset_indices is None:
            subset_indices = [Ellipsis]

        if len(self.scaler) == 0:
            return x[subset_indices]

        if TensorDim.GRID not in self.scaler:
            error_msg = (
                "Scaler tensor must be at least applied to the GRID dimension. "
                "Please add a scaler here, use `UniformWeights` for simple uniform scaling.",
            )
            raise RuntimeError(error_msg)

        scale_tensor = self.scaler
        if without_scalers is not None and len(without_scalers) > 0:
            if isinstance(without_scalers[0], str):
                scale_tensor = self.scaler.without(without_scalers)
            else:
                scale_tensor = self.scaler.without_by_dim(without_scalers)

        return scale_tensor.scale_iteratively(
            x,
            subset_indices=subset_indices,
            grid_shard_slice=grid_shard_slice,
        )

    def reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: str = "avg", # Originally avg
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Reduce the out of the loss.

        If `squash` is True, the last dimension is averaged.

        Irrespective of `squash`, the output is reduced over the
        batch, ensemble and grid dimensions.

        Parameters
        ----------
        out : torch.Tensor
            Difference tensor, of shape TensorDim
        squash : bool, optional
            Whether to squash the variable dimension, by default True
        squash_mode : str, optional
            Mode to use for squashing the variable dimension, by default "avg"
            If "avg", the last dimension is averaged.
            If "sum", the last dimension is summed.

        Returns
        -------
        torch.Tensor
            Reduced output tensor

        Raises
        ------
        ValueError
            If squash_mode is not one of ['avg', 'sum']
        """
        #[B, Ens, Grid, Variable] 
        
        if squash:
            if squash_mode == "avg":
                out = self.avg_function(out, dim=TensorDim.VARIABLE)
            elif squash_mode == "sum":
                out = self.sum_function(out, dim=TensorDim.VARIABLE)
            else:
                msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
                raise ValueError(msg)

        # Monte: commented out to see what happens with avg all dimensions. 
        # here the grid dimension is summed because the normalisation is handled in the node weighting
        grid_summed = self.sum_function(out, dim=(TensorDim.GRID))

        out = self.avg_function(
            grid_summed,
            #out,
            dim=(
                TensorDim.BATCH_SIZE,
                TensorDim.ENSEMBLE_DIM,
                #TensorDim.GRID Add this in in not grid summing
            ),
        )

        return out if group is None else reduce_tensor(out, group)

    @property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return self.__class__.__name__.lower()

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """


class FunctionalLoss(BaseLoss):
    """Loss which a user can subclass and provide `calculate_difference`.

    `calculate_difference` should calculate the difference between the prediction and target.
    All scaling and weighting is handled by the parent class.

    Example:
    --------
    ```python
    class MyLoss(FunctionalLoss):
        def calculate_difference(self, pred, target):
            return pred - target
    ```
    """

    @abstractmethod
    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate difference between prediction and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None
        out = self.calculate_difference(pred, target)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(out, squash, group=group if is_sharded else None)
    
    
class GraphCastBaseLoss(FunctionalLoss):
    """
    A minimal override that makes Anemoi's loss match GraphCast semantics:

    - squashing over variable dimension happens at the end
    - Grid reduction uses MEAN, not SUM
    - Final loss = mean over batch of the sum of per-variable MSEs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize these as None so they exist on all instances
        self.variable_groups = None
        self.group_slices = None
        self.group_sizes = None
        self._per_variable_initialized = False
    
    def build_variable_groups_from_data_indices(self):
        """
        Produces a dict: basename -> list of variable indices (int)
        Example: {"qv": [10,11,12,...], "theta": [...], "t2m":[3], ...}
        Also builds contiguous slices for efficient indexing.
        """
        # Only want the prognostic variables. 
        prognostic_indices = self.data_indices.data.output.prognostic # list of indices 
        
        name_to_idx = self.data_indices.data.output.name_to_index
        var_names = list(name_to_idx.keys())

        groups = {}
        for name, idx in name_to_idx.items():
             # Skip if not a prognostic variable
            if idx not in prognostic_indices:
                continue
            
            # Split on last underscore: qv_7 -> ('qv', '7')
            if "_" in name and name.split("_")[-1].isdigit():
                base = "_".join(name.split("_")[:-1])
            else:
                base = name  # single-level variable

            idx = name_to_idx[name]
            groups.setdefault(base, []).append(idx)

        # Sort indices within each group to ensure they're in order
        for base in groups:
            groups[base].sort()

        LOGGER.info(f"Prognostic variable groups for loss: {groups}")    
            
        return groups
    
    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Build level-groups based on real Anemoi variable indices."""
        self.data_indices = data_indices

        # auto-build groups: basename -> [indices]
        self.variable_groups = self.build_variable_groups_from_data_indices()

        # Build contiguous slices for efficient indexing
        self.group_slices = []
        for basename, idxs in self.variable_groups.items():
            start_idx = min(idxs)
            end_idx = max(idxs) + 1
            
            # Verify contiguity
            expected = list(range(start_idx, end_idx))
            if idxs != expected:
                error_msg = (
                    f"Variable group '{basename}' has non-contiguous indices: {idxs}. "
                    f"Expected contiguous range: {expected}. "
                    "This loss function requires variables to be stored in contiguous order."
                )
                raise ValueError(error_msg)
            
            self.group_slices.append((start_idx, end_idx))
        
        # Store group sizes as tensor for potential use
        self.group_sizes = torch.tensor(
            [len(idxs) for idxs in self.variable_groups.values()],
            dtype=torch.float,
        )
        
        LOGGER.info(
            f"GraphCastLoss initialized with {len(self.variable_groups)} variable groups: "
            f"{list(self.variable_groups.keys())}"
            f"{self.group_slices=}"
            f"{self.group_sizes}"
        )
    
    def _per_variable_loss(self, out: torch.Tensor) -> torch.Tensor:
        """
        out: (B, E, G, Vflat) raw per-variable loss contributions.
        Returns: (B, n_groups) after:
             1. vertical-level aggregation (mean over levels within each group)
             2. spatial + ensemble averaging
        
        Optimized for contiguous variable storage using direct slicing.
        """
        B, E, G, Vflat = out.shape
        
        # Process each variable group using contiguous slicing
        per_group_means = []
        for start_idx, end_idx in self.group_slices:
            # Extract contiguous slice for this variable group
            group_data = out[..., start_idx:end_idx]  # (B, E, G, group_size)
            
            # Mean over vertical levels (last dim) within this group
            group_mean = group_data.mean(dim=-1)  # (B, E, G)
            per_group_means.append(group_mean)
        
        # Stack into (B, E, G, n_groups)
        per_group = torch.stack(per_group_means, dim=-1)
        
        # Average over ensemble + grid dims → (B, n_groups)
        return per_group.mean(dim=(1, 2))
    
    def reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: str = "avg",
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Reduce the out of the loss.

        Following the procedure in Lam et al. (GraphCast)
        
        - Compute batch- and variable-preserving mean
            -- variables are grouped by vertical level and averaged
               before spatial or ensemble averaging
        - Sum over the variable dimension 
        - Compute batch-wise mean 
        
        If `squash` is False, then loss returned per-variable. 
        
        Irrespective of `squash`, the output is reduced over the
        batch, ensemble and grid dimensions.

        Parameters
        ----------
        out : torch.Tensor
            Difference tensor, of shape TensorDim
        squash : bool, optional
            Whether to squash the variable dimension, by default True
        squash_mode : str, optional
            Ignored. 

        Returns
        -------
        torch.Tensor
            Reduced output tensor

        Raises
        ------
        ValueError
            If squash_mode is not one of ['avg', 'sum']
        """
        # Compute batch- and variable group-preserving mean. 
        # (B, ens, grid, var) -> (B, var)
        out = self._per_variable_loss(out)

        LOGGER.info(f"After _per_variable_loss: {out=}")
        
        if squash:
            out = self.sum_function(out, dim=-1) # (B, var) -> (B,)
            out = self.avg_function(out) #(B) -> scalar
            
            LOGGER.info(f"After squash: {out=}")
            
            return out if group is None else reduce_tensor(out, group) 
                
        # Average over the batch dimension for the per-variable losses
        # (for plotting and logging)
        out = self.avg_function(out, dim=TensorDim.BATCH_SIZE)  # (var,)

        return out if group is None else reduce_tensor(out, group)
          