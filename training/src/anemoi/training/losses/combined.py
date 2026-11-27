# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy
import functools
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

REMAINING_VARIABLES = "remaining"
"""Value of ``predicted_variables`` selecting all variables not claimed by sibling losses."""

_NESTED_LOSS_KEYS = ("per_scale_loss", "loss_fn")


def _variable_holder(loss_config: dict[str, Any]) -> dict[str, Any]:
    """Return the (possibly nested) config holding ``predicted_variables`` for a child loss.

    Wrappers such as ``MultiscaleLossWrapper`` and ``TimeAggregateLossWrapper`` take their
    variable selection from the inner loss (``per_scale_loss`` / ``loss_fn``).
    """
    for key in _NESTED_LOSS_KEYS:
        inner = loss_config.get(key)
        if isinstance(inner, dict):
            return _variable_holder(inner)
    return loss_config


def _is_remaining(value: Any) -> bool:
    return isinstance(value, str) and value == REMAINING_VARIABLES


def _resolve_remaining_variables(
    losses: tuple[dict[str, Any] | Callable | BaseLoss, ...],
    data_indices: IndexCollection | None,
) -> tuple[dict[str, Any] | Callable | BaseLoss, ...]:
    """Replace ``predicted_variables: remaining`` by the variables not claimed by sibling losses.

    The remaining variables are all model output variables, in model output order,
    that are not listed in the ``predicted_variables`` of the other child losses.

    Parameters
    ----------
    losses : tuple
        Child loss specifications of a ``CombinedLoss``.
    data_indices : IndexCollection | None
        Training data indices, used for the model output variable names.

    Returns
    -------
    tuple
        The child loss specifications, with ``remaining`` resolved to an explicit list.
        Returned unchanged if no child uses ``remaining``.

    Raises
    ------
    ValueError
        If more than one child uses ``remaining``, if a sibling does not list its
        ``predicted_variables`` (so it already covers all variables), if ``data_indices``
        is missing, or if no variables remain.
    """
    configs = [OmegaConf.to_container(loss, resolve=True) if isinstance(loss, DictConfig) else loss for loss in losses]
    remaining_idx = [
        i
        for i, cfg in enumerate(configs)
        if isinstance(cfg, dict) and _is_remaining(_variable_holder(cfg).get("predicted_variables"))
    ]
    if not remaining_idx:
        return losses

    if len(remaining_idx) > 1:
        msg = (
            f"Only one child loss of a CombinedLoss can use `predicted_variables: {REMAINING_VARIABLES}`, "
            f"found {len(remaining_idx)} (children {remaining_idx})."
        )
        raise ValueError(msg)
    idx = remaining_idx[0]

    if data_indices is None:
        msg = f"`predicted_variables: {REMAINING_VARIABLES}` requires data indices to know the model output variables."
        raise ValueError(msg)

    claimed: set[str] = set()
    for i, cfg in enumerate(configs):
        if i == idx:
            continue
        explicit = _variable_holder(cfg).get("predicted_variables") if isinstance(cfg, dict) else None
        if explicit is None:
            msg = (
                f"Child loss {i} of a CombinedLoss has no `predicted_variables`, so it covers all variables, "
                f"which leaves nothing for the child {idx} using `predicted_variables: {REMAINING_VARIABLES}`. "
                f"List the variables of child {i} explicitly."
            )
            raise ValueError(msg)
        claimed.update(explicit)

    remaining = [name for name in data_indices.model.output.ordered_names if name not in claimed]
    if not remaining:
        msg = (
            f"`predicted_variables: {REMAINING_VARIABLES}` on child loss {idx} of a CombinedLoss resolves to no "
            "variables: the other child losses already cover all model output variables."
        )
        raise ValueError(msg)

    resolved = list(losses)
    config = copy.deepcopy(configs[idx])
    _variable_holder(config)["predicted_variables"] = remaining
    resolved[idx] = config
    return tuple(resolved)


class CombinedLoss(BaseLoss):
    """Combined Loss function."""

    needs_graph_data: bool = True
    needs_data_node_name: bool = True
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

        One child may set `predicted_variables: remaining` to cover all model output
        variables not listed by its siblings, in model output order. All siblings must
        then list their `predicted_variables` explicitly.
        """
        super().__init__()

        self.losses: list[type[BaseLoss]] = []

        losses = (*(losses or []), *extra_losses)
        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        assert len(losses) == len(loss_weights), "Number of losses and weights must match"
        assert len(losses) > 0, "At least one loss must be provided"

        losses = _resolve_remaining_variables(losses, data_indices)

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

    @property
    def scores_across_time(self) -> bool:
        """Whether any wrapped loss scores across output steps."""
        return any(loss.scores_across_time for loss in self.losses)

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
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        for loss in self.losses:
            loss.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler, assigned=("__doc__", "__annotations__"))
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        for loss in self.losses:
            loss_scaler = getattr(loss, "scaler", None)
            if loss_scaler is not None and name not in loss_scaler:
                continue
            loss.update_scaler(name=name, scaler=scaler, override=override)

    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return any(loss.has_scaler_for_dim(dim=dim) for loss in self.losses)
