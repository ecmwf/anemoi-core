# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scaler_tensor import ScaleTensor


class CombinedLoss(BaseLoss):
    """Combined Loss function."""

    _initial_set_scaler: bool = False

    def __init__(
        self,
        *extra_losses: dict[str, Any] | Callable | BaseLoss,
        loss_weights: tuple[int, ...] | None = None,
        losses: tuple[dict[str, Any] | Callable | BaseLoss] | None = None,
        **kwargs,
    ):
        """Combined loss function.

        Allows multiple losses to be combined into a single loss function,
        and the components weighted.

        As the losses are designed for use within the context of the
        anemoi-training configuration, `losses` work best as a dictionary.

        If `losses` is a `tuple[dict]`, the `scalers` key will be extracted
        before being passed to `get_loss_function`, and the `scalers` defined
        in each loss only applied to the respective loss. Thereby `scalers`
        added to this class will be routed correctly.
        If `losses` is a `tuple[Callable]`, all `scalers` added to this class
        will be added to all underlying losses.
        And if `losses` is a `tuple[BaseLoss]`, no scalers added to
        this class will be added to the underlying losses, as it is
        assumed that will be done by the parent function.

        Parameters
        ----------
        losses: tuple[dict[str, Any] | Callable | BaseLoss],
            if a `tuple[dict]`:
                Tuple of losses to initialise with `get_loss_function`.
                Allows for kwargs to be passed, and weighings controlled.
                If a loss should only have some of the scalers, set `scalers` in the loss config.
                If no scalers are set, all scalers added to this class will be included.
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
        kwargs: Any
            Additional arguments to pass to the loss functions, if not Loss.

        Examples
        --------
        >>> CombinedLoss(
                {"__target__": "anemoi.training.losses.MSELoss"},
                loss_weights=(1.0,),
            )
            CombinedLoss.add_scaler(name = 'scaler_1', ...)
            # Only added to the `MSELoss` if specified in it's `scalers`.
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
                - _target_: anemoi.training.losses.MAELoss
            scalers: ['*']
            loss_weights: [1.0, 0.6]
            # All scalers passed to this class will be added to each underlying loss
        ```

        ```
        training_loss:
            _target_: anemoi.training.losses.combined.CombinedLoss
            losses:
                - _target_: anemoi.training.losses.MSELoss
                  scalers: ['variable']
                - _target_: anemoi.training.losses.MAELoss
                  scalers: ['loss_weights_mask']
            scalers: ['*']
            # Only the specified scalers will be added to each loss
        ```
        """
        super().__init__()

        self.losses: list[type[BaseLoss]] = []
        self._loss_scaler_specification: dict[int, list[str]] = {}

        losses = (*(losses or []), *extra_losses)
        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        for i, loss in enumerate(losses):
            if isinstance(loss, DictConfig | dict):
                self._loss_scaler_specification[i] = loss.pop("scalers", ["*"])
                self.losses.append(get_loss_function(loss, scalers={}, **dict(kwargs)))
            elif isinstance(loss, type):
                self._loss_scaler_specification[i] = ["*"]
                self.losses.append(loss(**kwargs))
            else:
                assert isinstance(loss, BaseLoss)
                self._loss_scaler_specification[i] = loss.scaler
                self.losses.append(loss)

            self.add_module(str(i), self.losses[-1])  # (self.losses[-1].name + str(i), self.losses[-1])
        self.loss_weights = loss_weights
        del self.scaler  # Remove scaler property from parent class, as it is not used here

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

    @functools.wraps(ScaleTensor.add_scaler, assigned=("__doc__", "__annotations__"))
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        for i, spec in self._loss_scaler_specification.items():
            if "*" in spec or name in spec:
                self.losses[i].scaler.add_scaler(dimension, scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler, assigned=("__doc__", "__annotations__"))
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        for i, spec in self._loss_scaler_specification.items():
            if "*" in spec or name in spec:
                self.losses[i].scaler.update_scaler(name, scaler=scaler, override=override)
