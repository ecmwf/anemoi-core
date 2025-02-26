# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from omegaconf import DictConfig

from anemoi.training.losses.utils import ScaleTensor
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.train.forecaster import GraphForecaster

if TYPE_CHECKING:
    import torch


class CombinedLoss(BaseWeightedLoss):
    """Combined Loss function."""

    _initial_set_scalar: bool = False

    def __init__(
        self,
        *extra_losses: dict[str, Any] | Callable | BaseWeightedLoss,
        loss_weights: tuple[int, ...] | None = None,
        losses: tuple[dict[str, Any] | Callable | BaseWeightedLoss] | None = None,
        **kwargs,
    ):
        """Combined loss function.

        Allows multiple losses to be combined into a single loss function,
        and the components weighted.

        As the losses are designed for use within the context of the
        anemoi-training configuration, `losses` work best as a dictionary.

        If `losses` is a `tuple[dict]`, the `scalars` key will be extracted
        before being passed to `get_loss_function`, and the `scalars` defined
        in each loss only applied to the respective loss. Thereby `scalars`
        added to this class will be routed correctly.
        If `losses` is a `tuple[Callable]`, all `scalars` added to this class
        will be added to all underlying losses.
        And if `losses` is a `tuple[WeightedLoss]`, no scalars added to
        this class will be added to the underlying losses, as it is
        assumed that will be done by the parent function.

        Parameters
        ----------
        losses: tuple[dict[str, Any] | Callable | BaseLoss],
            if a `tuple[dict]`:
                Tuple of losses to initialise with `get_loss_function`.
                Allows for kwargs to be passed, and weighings controlled.
                If a loss should only have some of the scalars, set `scalars` in the loss config.
                If no scalars are set, all scalars added to this class will be included.
            if a `tuple[Callable]`:
                Will be called with `kwargs`, and all scalars added to this class added.
            if a `tuple[BaseLoss]`:
                Added to the loss function, and no scalars passed through.
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
            CombinedLoss.add_scaler(name = 'scalar_1', ...)
            # Only added to the `MSELoss` if specified in it's `scalars`.
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
                - _target_: anemoi.training.losses.mse.WeightedMSELoss
                - _target_: anemoi.training.losses.mae.WeightedMAELoss
            scalars: ['*']
            loss_weights: [1.0, 0.6]
            # All scalars passed to this class will be added to each underlying loss
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
        super().__init__(node_weights=None)

        self.losses: list[BaseWeightedLoss] = []
        self._loss_scalar_specification: dict[int, list[str]] = {}

        losses = (*(losses or []), *extra_losses)
        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        for i, loss in enumerate(losses):

            if isinstance(loss, (DictConfig, dict)):
                self._loss_scalar_specification[i] = loss.pop("scalars", ["*"])
                self.losses.append(GraphForecaster.get_loss_function(loss, scalars={}, **dict(kwargs)))
            elif isinstance(loss, Callable):
                self._loss_scalar_specification[i] = ["*"]
                self.losses.append(loss(**kwargs))
            else:
                self._loss_scalar_specification[i] = []
                self.losses.append(loss)

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
    def scalar(self, _: Any) -> None:
        """Set underlying loss scalars."""
        if not self._initial_set_scalar:  # Allow parent class to 'initialise' the scalar
            self._initial_set_scalar = True
            return
        excep_msg = "Cannot set `CombinedLoss` scalar directly, use `add_scalar` or `update_scalar`."
        raise AttributeError(excep_msg)

    @functools.wraps(ScaleTensor.add_scalar, assigned=("__doc__", "__annotations__"))
    def add_scalar(self, dimension: int | tuple[int], scalar: torch.Tensor, *, name: str | None = None) -> None:
        for i, spec in self._loss_scalar_specification.items():
            if "*" in spec or name in spec:
                self.losses[i].scalar.add_scalar(dimension, scalar, name=name)

    @functools.wraps(ScaleTensor.update_scalar, assigned=("__doc__", "__annotations__"))
    def update_scalar(self, name: str, scalar: torch.Tensor, *, override: bool = False) -> None:
        for i, spec in self._loss_scalar_specification.items():
            if "*" in spec or name in spec:
                self.losses[i].scalar.update_scalar(name, scalar=scalar, override=override)
