# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from omegaconf import DictConfig

from anemoi.training.losses.utils import ScaleTensor
from anemoi.training.losses.weightedloss import BaseWeightedLoss

if TYPE_CHECKING:
    import torch


class CombinedLoss(BaseWeightedLoss):
    """Combined Loss function."""

    def __init__(
        self,
        *extra_losses: dict[str, Any] | Callable,
        losses: tuple[dict[str, Any] | Callable] | None = None,
        loss_weights: tuple[int, ...],
        **kwargs,
    ):
        """Combined loss function.

        Allows multiple losses to be combined into a single loss function,
        and the components weighted.

        If a sub loss function requires additional weightings or code created tensors,
        that must be `included_` for this function, and then controlled by the underlying
        loss function configuration.

        Parameters
        ----------
        losses: tuple[dict[str, Any]| Callable]
            Tuple of losses to initialise with `GraphForecaster.get_loss_function`.
            Allows for kwargs to be passed, and weighings controlled.
        *extra_losses: dict[str, Any] | Callable
            Additional arg form of losses to include in the combined loss.
        loss_weights : tuple[int, ...]
            Weights of each loss function in the combined loss.
        kwargs: Any
            Additional arguments to pass to the loss functions

        Examples
        --------
        >>> CombinedLoss(
                {"_target_": "anemoi.training.losses.mse.WeightedMSELoss"},
                loss_weights=(1.0,),
                node_weights=node_weights
            )
        --------
        >>> CombinedLoss(
                losses = [anemoi.training.losses.mse.WeightedMSELoss],
                loss_weights=(1.0,),
                node_weights=node_weights
            )
        Or from the config,

        ```
        training_loss:
            _target_: anemoi.training.losses.combined.CombinedLoss
            losses:
                - _target_: anemoi.training.losses.mse.WeightedMSELoss
                - _target_: anemoi.training.losses.mae.WeightedMAELoss
            scalars: ['*']
            loss_weights: [1.0, 0.6]
        ```

        ```
        training_loss:
            _target_: anemoi.training.losses.combined.CombinedLoss
            losses:
                - _target_: anemoi.training.losses.mse.WeightedMSELoss
                  scalars: ['ocean']
                - _target_: anemoi.training.losses.mae.WeightedMAELoss
                  scalars: ['atmosphere']
            scalars: ['*']
            loss_weights: [1.0, 1.0]
        ```
        """
        self.losses: list[BaseWeightedLoss] = []
        super().__init__(node_weights=None)

        losses = (*(losses or []), *extra_losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        from anemoi.training.train.forecaster import GraphForecaster

        for i, loss in enumerate(losses):
            self.losses.append(
                (
                    GraphForecaster.get_loss_function(loss, **kwargs)
                    if isinstance(loss, (DictConfig, dict))
                    else loss(**kwargs)
                ),
            )
            self.add_module(self.losses[-1].name + str(i), self.losses[-1])
        self.loss_weights = loss_weights

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
            if loss is not None:
                loss += self.loss_weights[i] * loss_fn(pred, target, **kwargs)
            else:
                loss = self.loss_weights[i] * loss_fn(pred, target, **kwargs)
        return loss

    @property
    def name(self) -> str:
        return "combined_" + "_".join(getattr(loss, "name", loss.__class__.__name__.lower()) for loss in self.losses)

    @property
    def scalar(self) -> ScaleTensor:
        """Get union of underlying scalars."""
        scalars = {}
        for loss in self.losses:
            scalars.update(loss.scalar.tensors)
        return ScaleTensor(scalars)

    @scalar.setter
    def scalar(self, value: Any) -> None:
        """Set underlying loss scalars."""
        for loss_fn in self.losses:
            loss_fn.scalar = value

    def wrap_around_losses(func: Callable) -> Callable:  # noqa: N805
        """Wrap function to return result from underlying losses."""

        def wrapper(self: CombinedLoss, *args, **kwargs) -> Any:
            return self.__getattribute_from_losses__(func.__name__)(*args, **kwargs)

        return wrapper

    @functools.wraps(ScaleTensor.add_scalar, assigned=("__doc__", "__annotations__"))
    @wrap_around_losses
    def add_scalar(self, dimension: int | tuple[int], scalar: torch.Tensor, *, name: str | None = None) -> None:
        pass

    @functools.wraps(ScaleTensor.update_scalar, assigned=("__doc__", "__annotations__"))
    @wrap_around_losses
    def update_scalar(self, name: str, scalar: torch.Tensor, *, override: bool = False) -> None:
        pass

    def __getattribute_from_losses__(self, name: str) -> Callable:
        """Allow access to underlying attributes of the loss functions."""
        if not all(hasattr(loss, name) for loss in self.losses):
            error_msg = f"Attribute {name} not found in all loss functions"
            raise AttributeError(error_msg)

        @functools.wraps(getattr(self.losses[0], name))
        def hidden_func(*args, **kwargs) -> list[Any]:
            return [getattr(loss, name)(*args, **kwargs) for loss in self.losses]

        return hidden_func
