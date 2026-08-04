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
from collections.abc import Iterator
from enum import StrEnum
from typing import Any
from typing import ClassVar
from typing import Literal

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


Squash_mode = Literal["avg", "sum"]


class LossFactoryContextKey(StrEnum):
    """Named constructor-context inputs that selected loss classes can request."""

    AVAILABLE_SCALERS = "available_scalers"
    DATA_INDICES = "data_indices"


class BaseLoss(nn.Module, ABC):
    """Base loss."""

    # Most losses are built from config alone. Subclasses can list any
    # extra inputs they need from get_loss_function() here.
    factory_context_keys: ClassVar[frozenset[LossFactoryContextKey | str]] = frozenset()
    scaler: ScaleTensor
    needs_graph_data: bool = False

    def __init__(
        self,
        ignore_nans: bool = False,
    ) -> None:
        """Node- and feature_weighted Loss.

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

        self.ignore_nans = ignore_nans

        self.supports_sharding = True
        self.num_scales = 1

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        self.scaler.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        self.scaler.update_scaler(name=name, scaler=scaler, override=override)

    @functools.wraps(ScaleTensor.has_scaler_for_dim)
    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return self.scaler.has_scaler_for_dim(dim=dim)

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
            if len(self.scaler) == 0:
                return x
        elif not isinstance(subset_indices, tuple):
            msg = "subset_indices must be a tuple of per-dimension indexers, e.g. (..., indices)"
            raise TypeError(msg)

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

    def mask_nans(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fraction of ignored nan-values in the target and masked  prediction and target tensors.

        Parameters
        ----------
        pred : torch.Tensor,
            Prediction tensor
        target : torch.Tensor,
            Target tensor

        Returns
        -------
        torch.Tensor, torch.Tensor]
            * 0-masked copy of ``pred`` if ``self.ignore_nans``, else ``pred``
            * 0-masked copy of ``target`` if ``self.ignore_nans``, else ``target``
        """
        if self.ignore_nans:
            nan_mask = torch.isnan(target + pred)
            target = target.masked_fill(nan_mask, 0.0)
            pred = pred.masked_fill(nan_mask, 0.0)

            return pred, target

        return pred, target

    def reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: Squash_mode = "avg",
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
        squash_mode : {"avg", "sum"} , optional
            Mode to use for squashing the variable dimension, by default "avg"
            If "avg", the last dimension is averaged.
            If "sum", the last dimension is summed.
        group : ProcessGroup | None, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Reduced output tensor

        Raises
        ------
        ValueError
            If squash_mode is not one of ['avg', 'sum']
        """
        if squash:
            if squash_mode == "avg":
                out = torch.mean(out, dim=TensorDim.VARIABLE, keepdim=True)
            elif squash_mode == "sum":
                out = torch.sum(out, dim=TensorDim.VARIABLE, keepdim=True)
            else:
                msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
                raise ValueError(msg)

        # here the grid and time dimension are summed because
        # 1. the normalisation over grid points is handled in the node weighting
        # 2. the normalization over output steps is handled by the time_step scaler
        space_time_summed = torch.sum(
            out,
            dim=(
                TensorDim.TIME,
                TensorDim.GRID,
            ),
            keepdim=True,
        )
        out = torch.mean(
            space_time_summed,
            dim=(
                TensorDim.BATCH_SIZE,
                TensorDim.TIME,
                TensorDim.ENSEMBLE_DIM,
            ),
        )
        # Remove the grid dimension while retaining one value per variable
        # when squash=False.
        out = out.squeeze(0)
        if squash:
            out = out.squeeze(-1)

        return out if group is None else reduce_tensor(out, group)

    def iter_leaf_losses(self) -> Iterator["BaseLoss"]:
        """Yield all leaf loss modules.

        For simple losses, yields self. For composite losses (e.g. CombinedLoss),
        recursively yields the underlying leaf losses.
        """
        yield self

    @property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return self.__class__.__name__.lower()

    @property
    def needs_shard_layout_info(self) -> bool:
        """Whether the loss needs explicit shard-layout metadata beyond grid_shard_slice/group."""
        return False

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
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)
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
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Weighted loss
        """


class BaseLossWrapper(BaseLoss):
    """Transparent wrapper around a single inner loss.

    By default, all scaler and metadata methods are delegated to the
    wrapped loss so that the wrapper behaves as if it *were* the inner
    loss from the perspective of ``CombinedLoss`` and the scaler
    machinery.  Subclasses only need to override ``forward``.
    """

    def __init__(self, loss: BaseLoss, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(loss, BaseLoss):
            msg = f"Invalid loss type provided: {type(loss)}. Expected BaseLoss."
            raise TypeError(msg)
        self.ignore_nans = loss.ignore_nans or self.ignore_nans
        if self.ignore_nans and not loss.ignore_nans:
            msg = "BaseLossWrapper.ignore_nans and BaseLoss.ignore_nans missmatch."
            msg += f" {self.ignore_nans} != {loss.ignore_nans}"
            raise ValueError(msg)
        self.loss = loss
        # Share the inner loss's scaler so that scaler additions/updates
        # applied to this wrapper are visible to the actual loss computation.
        self.scaler = self.loss.scaler
        self.supports_sharding = getattr(self.loss, "supports_sharding", True)

    # -- scaler delegation --------------------------------------------------

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        self.loss.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        self.loss.update_scaler(name=name, scaler=scaler, override=override)

    @functools.wraps(ScaleTensor.has_scaler_for_dim)
    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return self.loss.has_scaler_for_dim(dim=dim)

    # -- metadata delegation ------------------------------------------------

    @property
    def needs_shard_layout_info(self) -> bool:
        """Delegate to the wrapped loss."""
        return getattr(self.loss, "needs_shard_layout_info", False)

    def iter_leaf_losses(self) -> Iterator["BaseLoss"]:
        """Yield leaf losses from the wrapped loss."""
        yield from self.loss.iter_leaf_losses()


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
        squash_mode: Squash_mode = "avg",
        **_kwargs,
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
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None
        pred, target = self.mask_nans(pred, target)

        out = self.calculate_difference(pred, target)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if is_sharded else None, squash_mode=squash_mode)
