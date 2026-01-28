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

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scaler_tensor import ScaleTensor


class CombinedLoss(BaseLoss):
    """Combined Loss function."""

    _initial_set_scaler: bool = False

    def __init__(
        self,
        *extra_losses: BaseLoss | Callable[..., BaseLoss] | tuple[BaseLoss | Callable[..., BaseLoss], list[str] | None],
        loss_weights: tuple[int, ...] | None = None,
        losses: (
            tuple[BaseLoss | Callable[..., BaseLoss] | tuple[BaseLoss | Callable[..., BaseLoss], list[str] | None]]
            | None
        ) = None,
        **kwargs,
    ):
        """Combined loss function.

        Allows multiple losses to be combined into a single loss function,
        and the components weighted.

        If `losses` entries are provided as `(loss, scalers)`, the `scalers`
        list specifies which scalers to apply to that loss when this combined
        loss receives scalers. If `scalers` is None, all scalers are applied.
        If `losses` entries are `Callable`, all scalers added to this class
        will be added to those losses. If `losses` entries are `BaseLoss`,
        no scalers added to this class will be added to the underlying losses,
        as it is assumed that will be done externally.

        Parameters
        ----------
        losses: tuple[BaseLoss | Callable | tuple[BaseLoss | Callable, list[str] | None]],
            If a `tuple[(loss, scalers)]`, the scalers list controls which
            scalers are forwarded to that loss. If a `Callable`, it will be
            called with `kwargs`, and all scalers will be forwarded.
            If a `BaseLoss`, no scalers are forwarded.
        *extra_losses: BaseLoss | Callable | tuple[BaseLoss | Callable, list[str] | None],
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
                (anemoi.training.losses.MSELoss, ["*"]),
                loss_weights=(1.0,),
            )
            CombinedLoss.add_scaler(name = 'scaler_1', ...)
            # Only added to the `MSELoss` if specified in the scalers list.
        --------
        >>> CombinedLoss(
                losses = [anemoi.training.losses.MSELoss],
                loss_weights=(1.0,),
            )
        """
        super().__init__()

        self.losses: list[BaseLoss] = []
        self._loss_scaler_specification: dict[int, list[str] | ScaleTensor] = {}

        losses = (*(losses or []), *extra_losses)
        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        if len(losses) != len(loss_weights):
            msg = "Number of losses and weights must match"
            raise ValueError(msg)
        if len(losses) == 0:
            msg = "At least one loss must be provided"
            raise ValueError(msg)

        for i, loss_entry in enumerate(losses):
            loss_obj, scaler_spec = self._resolve_loss_entry(loss_entry, **kwargs)
            self._loss_scaler_specification[i] = scaler_spec
            self.losses.append(loss_obj)

            self.add_module(str(i), self.losses[-1])  # (self.losses[-1].name + str(i), self.losses[-1])
        self.loss_weights = loss_weights
        del self.scaler  # Remove scaler property from parent class, as it is not used here

    @staticmethod
    def _resolve_loss_entry(
        loss_entry: BaseLoss | Callable[..., BaseLoss] | tuple[BaseLoss | Callable[..., BaseLoss], list[str] | None],
        **kwargs,
    ) -> tuple[BaseLoss, list[str] | ScaleTensor]:
        scaler_spec: list[str] | None = None
        loss_obj: BaseLoss | Callable[..., BaseLoss] = loss_entry

        if isinstance(loss_entry, tuple | list):
            if len(loss_entry) != 2:
                msg = "Loss tuple entries must be of the form (loss, scalers)"
                raise TypeError(msg)
            loss_obj, scaler_spec = loss_entry
            if scaler_spec is None:
                scaler_spec = ["*"]
            elif isinstance(scaler_spec, tuple):
                scaler_spec = list(scaler_spec)
            elif not isinstance(scaler_spec, list):
                msg = "Scaler specification must be a list or tuple of strings"
                raise TypeError(msg)

        if isinstance(loss_obj, BaseLoss):
            resolved = loss_obj
            spec = scaler_spec if scaler_spec is not None else loss_obj.scaler
        elif callable(loss_obj):
            resolved = loss_obj(**kwargs)
            spec = scaler_spec if scaler_spec is not None else ["*"]
        else:
            msg = f"Invalid loss type provided: {type(loss_obj)}"
            raise TypeError(msg)

        if not isinstance(resolved, BaseLoss):
            msg = f"Loss must be a subclass of 'BaseLoss', not {type(resolved)}"
            raise TypeError(msg)

        return resolved, spec

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
