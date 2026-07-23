# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scaler_tensor import ScalerDomain
from anemoi.training.losses.scaler_tensor import ScalerSpec
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim


class CombinedLoss(BaseLoss):
    """Combined Loss function."""

    needs_graph_data: bool = True
    # CombinedLoss builds child losses itself, so it needs the full scaler
    # set and data indices during construction.
    factory_context_keys = frozenset(
        {LossFactoryContextKey.AVAILABLE_SCALERS, LossFactoryContextKey.DATA_INDICES},
    )
    _initial_set_scaler: bool = False

    def __init__(
        self,
        *extra_losses: dict[str, Any] | Callable | BaseLoss,
        loss_weights: tuple[int, ...] | None = None,
        losses: tuple[dict[str, Any] | Callable | BaseLoss] | None = None,
        available_scalers: dict[str, ScalerSpec] | None = None,
        data_indices: IndexCollection | None = None,
        **kwargs,
    ):
        """Combined loss function.

        Allows multiple losses to be combined into a single loss function,
        and the components weighted.

        Each child loss controls its own scalers via its `scalers` config key.
        All available scalers are passed through to child losses unconditionally.

        Parameters
        ----------
        losses: tuple[dict[str, Any] | Callable | BaseLoss],
            if a `tuple[dict]`:
                Tuple of losses to initialise with `get_loss_function`.
                Allows for kwargs to be passed, and weighings controlled.
                Each child loss specifies its own `scalers` to control which
                scalers it receives.
            if a `tuple[Callable]`:
                Will be called with `kwargs`, and all scalers added to this class added.
            if a `tuple[BaseLoss]`:
                Added to the loss function, and no scalers passed through.
        *extra_losses: dict[str, Any]  | Callable | BaseLoss],
            Additional arg form of losses to include in the combined loss.
        loss_weights : optional, tuple[int, ...] | None
            Weights of each loss function in the combined loss.
            Must be the same length as the number of losses.
            If None, all losses are weighted equally.
            by default None.
        available_scalers : dict[str, ScalerSpec] | None, optional
            All scaler tensors available. Passed through to child losses.
        data_indices : IndexCollection | None, optional
            Training data indices needed by child losses that perform variable mapping.
        kwargs: Any
            Additional arguments to pass to the loss functions, if not Loss.

        Examples
        --------
        >>> CombinedLoss(
                {"__target__": "anemoi.training.losses.MSELoss"},
                loss_weights=(1.0,),
            )
            CombinedLoss.add_scaler(name = 'scaler_1', ...)
        --------
        >>> CombinedLoss(
                losses = [anemoi.training.losses.MSELoss],
                loss_weights=(1.0,),
            )
        Or from the config,

        ```
        training_loss:
            _target_: anemoi.training.losses.combined.CombinedLoss
            losses:
                - _target_: anemoi.training.losses.MSELoss
                  scalers: ['variable', 'node_weights']
                - _target_: anemoi.training.losses.MAELoss
                  scalers: ['loss_weights_mask']
            loss_weights: [1.0, 0.6]
            # Each child loss specifies its own scalers
        ```
        """
        super().__init__()

        self.losses: list[type[BaseLoss]] = []

        losses = (*(losses or []), *extra_losses)
        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        for i, loss in enumerate(losses):
            if isinstance(loss, DictConfig | dict):
                loss_config = dict(loss)
                self.losses.append(
                    get_loss_function(
                        DictConfig(loss_config),
                        scalers=available_scalers,
                        data_indices=data_indices,
                        graph_data=kwargs.get("graph_data"),
                        data_node_name=kwargs.get("data_node_name"),
                    ),
                )
            elif isinstance(loss, type):
                self.losses.append(loss(**kwargs))
            else:
                assert isinstance(loss, BaseLoss)
                self.losses.append(loss)

            self.add_module(str(i), self.losses[-1])
        self.loss_weights = loss_weights
        del self.scaler  # Remove scaler property from parent class, as it is not used here

    @property
    def needs_shard_layout_info(self) -> bool:
        """Whether any wrapped loss requires explicit shard-layout metadata."""
        return any(getattr(loss, "needs_shard_layout_info", False) for loss in self.losses)

    @staticmethod
    def _forward_kwargs_for_loss(loss_fn: BaseLoss, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter shard-layout kwargs for child losses that do not require them."""
        if getattr(loss_fn, "needs_shard_layout_info", False):
            return kwargs

        filtered_kwargs = dict(kwargs)
        filtered_kwargs.pop("grid_dim", None)
        filtered_kwargs.pop("grid_shard_sizes", None)
        return filtered_kwargs

    def iter_leaf_losses(self) -> Iterator["BaseLoss"]:
        """Recursively yield leaf losses from all sub-losses."""
        for sub_loss in self.losses:
            yield from sub_loss.iter_leaf_losses()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the combined loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        kwargs: Any
            Additional arguments to pass to the loss functions
            Will be passed to all loss functions

        Returns
        -------
        torch.Tensor
            Combined loss
        """
        loss = None
        for i, loss_fn in enumerate(self.losses):
            loss_kwargs = self._forward_kwargs_for_loss(loss_fn, kwargs)
            if loss is not None:
                loss += self.loss_weights[i] * loss_fn(pred, target, **loss_kwargs)
            else:
                loss = self.loss_weights[i] * loss_fn(pred, target, **loss_kwargs)
        return loss

    @functools.wraps(ScaleTensor.add_scaler, assigned=("__doc__", "__annotations__"))
    def add_scaler(
        self,
        dimension: int | tuple[int],
        scaler: torch.Tensor,
        *,
        grid_domain: ScalerDomain | None = None,
        name: str | None = None,
    ) -> None:
        for loss in self.losses:
            loss.add_scaler(dimension=dimension, scaler=scaler, grid_domain=grid_domain, name=name)

    @functools.wraps(ScaleTensor.update_scaler, assigned=("__doc__", "__annotations__"))
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        for loss in self.losses:
            loss_scaler = getattr(loss, "scaler", None)
            if loss_scaler is not None and name not in loss_scaler:
                continue
            loss.update_scaler(name=name, scaler=scaler, override=override)

    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return any(loss.has_scaler_for_dim(dim=dim) for loss in self.losses)
