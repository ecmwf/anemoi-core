# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections.abc import Sequence
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
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
        checkpoint_losses: bool | Sequence[bool] = False,
        offload_saved_tensors: bool | Sequence[bool] = False,
        offload_pin_memory: bool = False,
        available_scalers: dict[str, TENSOR_SPEC] | None = None,
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
        checkpoint_losses : bool | Sequence[bool], optional
            Whether to checkpoint child losses and recompute them during backward.
            A single bool applies to all children; a sequence configures each child separately.
        offload_saved_tensors : bool | Sequence[bool], optional
            Whether to offload tensors saved for backward by child losses to CPU.
            A single bool applies to all children; a sequence configures each child separately.
        offload_pin_memory : bool, optional
            Whether CPU-offloaded saved tensors use pinned memory.
        available_scalers : dict[str, TENSOR_SPEC] | None, optional
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

        self.checkpoint_losses = self._expand_bool_option(checkpoint_losses, len(losses), "checkpoint_losses")
        self.offload_saved_tensors = self._expand_bool_option(
            offload_saved_tensors,
            len(losses),
            "offload_saved_tensors",
        )
        self.offload_pin_memory = offload_pin_memory

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

    @staticmethod
    def _expand_bool_option(value: bool | Sequence[bool], length: int, name: str) -> tuple[bool, ...]:
        """Expand a bool or per-loss bool sequence to one flag per child loss."""
        if isinstance(value, bool):
            return (value,) * length

        result = tuple(bool(item) for item in value)
        if len(result) != length:
            msg = f"{name} must be a bool or have one entry per loss."
            raise ValueError(msg)
        return result

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

    def _call_loss(
        self,
        loss_fn: BaseLoss,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_kwargs: dict[str, Any],
        *,
        checkpoint_loss: bool,
        offload_saved_tensors: bool,
    ) -> torch.Tensor:
        """Call a child loss, optionally checkpointing or offloading saved tensors."""

        def run_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            context = (
                torch.autograd.graph.save_on_cpu(pin_memory=self.offload_pin_memory)
                if offload_saved_tensors and torch.is_grad_enabled()
                else nullcontext()
            )
            with context:
                return loss_fn(pred, target, **loss_kwargs)

        if checkpoint_loss and torch.is_grad_enabled():
            return checkpoint(run_loss, pred, target, use_reentrant=False)

        return run_loss(pred, target)

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
            loss_value = self._call_loss(
                loss_fn,
                pred,
                target,
                loss_kwargs,
                checkpoint_loss=self.checkpoint_losses[i],
                offload_saved_tensors=self.offload_saved_tensors[i],
            )
            weighted_loss = self.loss_weights[i] * loss_value
            loss = weighted_loss if loss is None else loss + weighted_loss
        return loss

    @functools.wraps(ScaleTensor.add_scaler, assigned=("__doc__", "__annotations__"))
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        for loss in self.losses:
            loss.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler, assigned=("__doc__", "__annotations__"))
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        for loss in self.losses:
            loss.update_scaler(name=name, scaler=scaler, override=override)

    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return any(loss.has_scaler_for_dim(dim=dim) for loss in self.losses)
