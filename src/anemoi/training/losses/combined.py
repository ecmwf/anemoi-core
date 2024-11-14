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
from typing import Any
from typing import Callable

import torch

from anemoi.training.train.forecaster import GraphForecaster


class CombinedLoss(torch.nn.Module):
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
                {"__target__": "anemoi.training.losses.mse.WeightedMSELoss"},
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
            __target__: anemoi.training.losses.combined.CombinedLoss
            losses:
                - __target__: anemoi.training.losses.mse.WeightedMSELoss
                - __target__: anemoi.training.losses.mae.WeightedMAELoss
            scalars: ['variable']
            loss_weights: [1.0,0.5]
        ```
        """
        super().__init__()

        losses = (*(losses or []), *extra_losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        self.losses = [
            GraphForecaster.get_loss_function(loss, **kwargs) if isinstance(loss, dict) else loss(**kwargs)
            for loss in losses
        ]
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

    def __getattr__(self, name: str) -> Callable:
        """Allow access to underlying attributes of the loss functions."""
        if not all(hasattr(loss, name) for loss in self.losses):
            error_msg = f"Attribute {name} not found in all loss functions"
            raise AttributeError(error_msg)

        @functools.wraps(getattr(self.losses[0], name))
        def hidden_func(*args, **kwargs) -> list[Any]:
            return [getattr(loss, name)(*args, **kwargs) for loss in self.losses]

        return hidden_func
