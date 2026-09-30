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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import TensorLayout
from anemoi.models.data.utils import apply_pairwise
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from anemoi.models.data import Source

LOGGER = logging.getLogger(__name__)


Squash_mode = Literal["avg", "sum"]


def _tabular_shard_group(target: "Source", group: ProcessGroup | None) -> ProcessGroup | None:
    """Return ``group`` when the tabular ``target`` is still split across it, else ``None``.

    Observation readers set shard sizes even for a single rank; only a split across several
    ranks needs the loss to be reduced over the group.
    """
    if target.shard_sizes is None or not model_is_distributed(group):
        return None
    if all(len(window) <= 1 for sample in target.shard_sizes for window in sample):
        return None
    return group


def _tabular_valid_counts(
    target: "Source",
    *,
    ignore_nans: bool,
    group: ProcessGroup | None,
) -> list[torch.Tensor]:
    """Per sample, the number of observations of each variable that enter the loss.

    With ``ignore_nans``, missing (NaN) targets are left out, so the loss is a mean over the
    observations that exist. When the target is split across ``group`` the counts are summed over
    it, so every rank divides its partial sum by the same total.
    """
    layout = target.layout
    counts = []
    for sample in target.data:
        valid = ~torch.isnan(sample) if ignore_nans else torch.ones_like(sample, dtype=torch.bool)
        ensemble = layout.axis("ensemble", ndim=sample.ndim) if layout.has_axis("ensemble") else None
        if ensemble is not None:
            valid = valid.all(dim=ensemble, keepdim=True)  # masked when any member's target is missing
        dims = tuple(dim for dim in range(sample.ndim) if dim != layout.axis("variables", ndim=sample.ndim))
        counts.append(valid.sum(dim=dims).to(torch.float32))

    if group is not None and counts:
        stacked = torch.stack(counts)
        torch.distributed.all_reduce(stacked, group=group)
        counts = list(stacked.unbind(0))
    return counts


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
    needs_data_node_name: bool = False

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
        layout: TensorLayout = None,
        *,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Scale a tensor by the variable_scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs).
        subset_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler and `x` tensor with, by default None.
        layout: TensorLayout, optional
            Layout describing the logical axes of x.
        without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None.

        Returns
        -------
        torch.Tensor
            Scaled error tensor.
        """
        if subset_indices is None:
            if len(self.scaler) == 0:
                return x
        elif not isinstance(subset_indices, tuple):
            msg = "subset_indices must be a tuple of per-dimension indexers, e.g. (..., indices)"
            raise TypeError(msg)

        if len(self.scaler) == 0:
            return x[subset_indices]

        scale_tensor = self.scaler
        if without_scalers is not None and len(without_scalers) > 0:
            if isinstance(without_scalers[0], str):
                scale_tensor = self.scaler.without(without_scalers)
            else:
                scale_tensor = self.scaler.without_by_dim(without_scalers)

        return scale_tensor.scale_iteratively(
            x,
            subset_indices=subset_indices,
            layout=layout,
            grid_shard_slice=grid_shard_slice,
        )

    def mask_nans(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask NaN targets and corresponding predictions.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor.
        target : torch.Tensor
            Target tensor.

        Returns
        -------
        torch.Tensor, torch.Tensor
            * 0-masked copy of ``pred`` if ``self.ignore_nans``, else ``pred``.
            * 0-masked copy of ``target`` if ``self.ignore_nans``, else ``target``.
        """
        if self.ignore_nans:
            nan_mask = torch.isnan(target)
            target = target.masked_fill(nan_mask, 0.0)
            pred = pred.masked_fill(nan_mask, 0.0)

            return pred, target

        return pred, target

    def reduce(
        self,
        out: torch.Tensor,
        layout: TensorLayout,
        squash: bool = True,
        squash_mode: Squash_mode = "avg",
        group: ProcessGroup | None = None,
        valid_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce the out of the loss.

        If `squash` is True, the variable dimension is reduced (averaged or summed).

        The spatial dimension is then reduced: for gridded fields the grid (and time)
        dimensions are *summed* (grid normalisation is handled by node weighting, time by
        the time-step scaler), while for tabular observations (layouts without a time axis;
        :func:`apply_pairwise` calls the loss once per sample) the grid dimension is
        *averaged*, since there is no node weighting and the number of observations varies
        per sample. The batch and ensemble dimensions are averaged.

        Parameters
        ----------
        out : torch.Tensor
            Difference tensor, of shape TensorDim.
        layout : TensorLayout
            Layout describing the logical axes of out.
        squash : bool, optional
            Whether to squash the variable dimension, by default True.
        squash_mode : {"avg", "sum"}, optional
            Mode to use for squashing the variable dimension, by default "avg".
            If "avg", the last dimension is averaged.
            If "sum", the last dimension is summed.
        group : ProcessGroup | None, optional
            Distributed group to reduce over, by default None.
        valid_counts : torch.Tensor | None, optional
            Tabular observations only: per-variable number of observations entering the loss.
            Defaults to the number of grid points.

        Returns
        -------
        torch.Tensor
            Reduced output tensor.

        Raises
        ------
        ValueError
            If squash_mode is not one of ['avg', 'sum'].
        """
        if squash_mode not in ("avg", "sum"):
            msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
            raise ValueError(msg)

        if not layout.has_axis("time"):
            # Sparse observations: we average over the spatial dimension, per variable. Unlike
            # gridded fields there is no node weighting that normalises over grid points,
            # and the number of observations varies per sample and per variable.
            grid_sum = torch.sum(out, dim=layout.grid, keepdim=True)
            if valid_counts is None:
                space_time_reduced = grid_sum / max(out.shape[layout.grid], 1)
            else:
                space_time_reduced = grid_sum / valid_counts.clamp(min=1).to(grid_sum.dtype)
            if squash:
                reduce_variables = torch.mean if squash_mode == "avg" else torch.sum
                space_time_reduced = reduce_variables(space_time_reduced, dim=layout.variables, keepdim=True)
        else:
            if squash:
                if squash_mode == "avg":
                    out = torch.mean(out, dim=layout.variables, keepdim=True)
                else:
                    out = torch.sum(out, dim=layout.variables, keepdim=True)
            # Gridded fields: the grid and time dimensions are summed because
            # 1. the normalisation over grid points is handled in the node weighting
            # 2. the normalization over output steps is handled by the time_step scaler
            dims = [layout.time, layout.grid]
            space_time_reduced = torch.sum(
                out,
                dim=tuple([f for f in dims if f is not None]),
                keepdim=True,
            )

        dims = tuple(f for f in (layout.batch, layout.time, layout.ensemble) if f is not None)
        out = torch.mean(space_time_reduced, dim=dims, keepdim=True) if dims else space_time_reduced
        # Return a scalar or one loss per variable, preserving a single-variable vector.
        if squash:
            out = out.squeeze()

        return out if group is None else reduce_tensor(out, group)

    def _evaluate_loss_tensor(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the numerical loss in at least float32, including under autocast."""
        dtype = torch.promote_types(torch.promote_types(pred.dtype, target.dtype), torch.float32)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return self._forward_impl(pred.to(dtype), target.to(dtype), **kwargs)

    def _apply_pairwise(
        self,
        pred: "Source",
        target: "Source",
        func: Any,
        *,
        group: ProcessGroup | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply ``func`` to aligned ``pred``/``target`` sources (see :func:`apply_pairwise`).

        Tabular observations are averaged over the observations that enter the loss, per variable
        (``valid_counts``). When the target is split across ``group``, the counts are totals over it,
        each rank's result is a partial sum, and the partial sums are reduced over the group - as for
        sharded gridded fields.
        """
        if not target.is_tabular:
            return apply_pairwise(pred, target, func, group=group, **kwargs)

        shard_group = _tabular_shard_group(target, group)
        valid_counts = _tabular_valid_counts(target, ignore_nans=self.ignore_nans, group=shard_group)
        per_sample_kwargs = {**(kwargs.pop("per_sample_kwargs", None) or {}), "valid_counts": valid_counts}
        loss = apply_pairwise(pred, target, func, group=group, per_sample_kwargs=per_sample_kwargs, **kwargs)
        return loss if shard_group is None else reduce_tensor(loss, shard_group)

    @staticmethod
    def _counts_like(
        valid_counts: torch.Tensor | None,
        out: torch.Tensor,
        layout: TensorLayout,
        subset_indices: tuple | None,
    ) -> torch.Tensor | None:
        """Shape per-variable ``valid_counts`` to broadcast against ``out``, subset like it."""
        if valid_counts is None:
            return None
        shape = [1] * out.ndim
        shape[layout.axis("variables", ndim=out.ndim)] = -1
        counts = valid_counts.to(out.device).view(shape)
        return counts if subset_indices is None else counts[subset_indices]

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
            Prediction tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs).
        target : torch.Tensor
            Target tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs).
        squash : bool, optional
            Average last dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None.
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        **_kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Weighted loss.
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
    """Loss which a user can subclass and provide ``calculate_difference``.

    ``calculate_difference`` should calculate the difference between the prediction and target.
    All scaling and weighting is handled by the parent class.

    Example
    -------
    .. code-block:: python

        class MyLoss(FunctionalLoss):
            def calculate_difference(self, pred, target):
                return pred - target
    """

    @abstractmethod
    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate difference between prediction and target."""

    def _forward_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        layout: TensorLayout,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        valid_counts: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        is_sharded = grid_shard_slice is not None

        pred, target = self.mask_nans(pred, target)

        out = self.calculate_difference(pred, target)

        out = self.scale(
            out,
            scaler_indices,
            layout=layout,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        return self.reduce(
            out,
            layout=layout,
            squash=squash,
            group=group if is_sharded else None,
            squash_mode=squash_mode,
            valid_counts=self._counts_like(valid_counts, pred, layout, scaler_indices),
        )

    def forward(
        self,
        pred: "Source",
        target: "Source",
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Dispatches to the tensor-level _forward_impl via the source view's layout.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs).
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs).
        squash : bool, optional
            Average last dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None.
        group : ProcessGroup, optional
            Distributed group, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        return self._apply_pairwise(
            pred,
            target,
            self._evaluate_loss_tensor,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            squash_mode=squash_mode,
            **kwargs,
        )
