# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.builders.components import build_component
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function

if TYPE_CHECKING:
    from anemoi.training.losses.scaler_tensor import TENSOR_SPEC

_MULTISCALE_TARGETS = {"anemoi.training.losses.MultiscaleLossWrapper"}
_COMBINED_TARGETS = {
    "anemoi.training.losses.combined.CombinedLoss",
    "anemoi.training.losses.CombinedLoss",
}
_FILTERING_TARGETS = {
    "anemoi.training.losses.filtering.FilteringLossWrapper",
    "anemoi.training.losses.FilteringLossWrapper",
}


def build_loss_from_config(
    config: Mapping[str, Any],
    *,
    scalers: dict[str, TENSOR_SPEC] | None = None,
    data_indices: dict | None = None,
    **kwargs: Any,
) -> BaseLoss:
    """Instantiate a loss from config and attach scalers."""
    return _build_loss_from_config(
        config,
        scalers=scalers,
        data_indices=data_indices,
        apply_scalers=True,
        **kwargs,
    )


def _build_loss_from_config(
    config: Mapping[str, Any],
    *,
    scalers: dict[str, TENSOR_SPEC] | None,
    data_indices: dict | None,
    apply_scalers: bool,
    **kwargs: Any,
) -> BaseLoss:
    loss_config = _normalize_loss_config(config)
    scalers_to_include = loss_config.pop("scalers", [])
    target = loss_config.get("_target_")

    if target in _MULTISCALE_TARGETS:
        per_scale_loss_config = loss_config.pop("per_scale_loss")
        per_scale_loss = _build_loss_from_config(
            per_scale_loss_config,
            scalers=scalers,
            data_indices=data_indices,
            apply_scalers=apply_scalers,
        )
        loss = build_component(loss_config, per_scale_loss=per_scale_loss, **kwargs)
        _ensure_base_loss(loss, target)
        return loss

    if target in _COMBINED_TARGETS:
        losses_config = loss_config.get("losses", [])
        built_losses = []
        for loss_entry in losses_config:
            entry_config = _normalize_loss_config(loss_entry)
            entry_scalers = entry_config.pop("scalers", ["*"])
            inner_loss = _build_loss_from_config(
                entry_config,
                scalers=scalers,
                data_indices=data_indices,
                apply_scalers=False,
                **kwargs,
            )
            built_losses.append((inner_loss, entry_scalers))
        loss_config["losses"] = built_losses
        loss = build_component(loss_config, **kwargs)
        _ensure_base_loss(loss, target)
        if apply_scalers:
            return get_loss_function(
                loss,
                scalers=scalers,
                data_indices=data_indices,
                scalers_to_include=scalers_to_include,
            )
        return loss

    if target in _FILTERING_TARGETS:
        inner_loss_config = loss_config.pop("loss")
        inner_loss = _build_loss_from_config(
            inner_loss_config,
            scalers=scalers,
            data_indices=data_indices,
            apply_scalers=False,
            **kwargs,
        )
        loss = build_component(loss_config, loss=inner_loss, **kwargs)
        _ensure_base_loss(loss, target)
        if apply_scalers:
            return get_loss_function(
                loss,
                scalers=scalers,
                data_indices=data_indices,
                scalers_to_include=scalers_to_include,
            )
        return loss

    loss = build_component(loss_config, **kwargs)
    _ensure_base_loss(loss, target)
    if apply_scalers:
        return get_loss_function(
            loss,
            scalers=scalers,
            data_indices=data_indices,
            scalers_to_include=scalers_to_include,
        )
    return loss


def _normalize_loss_config(config: Mapping[str, Any]) -> dict:
    if hasattr(config, "model_dump"):
        loss_config = config.model_dump(by_alias=True)
    elif isinstance(config, Mapping):
        loss_config = copy.deepcopy(config)
    else:
        error_msg = f"Loss config must be a mapping, not {type(config)}"
        raise TypeError(error_msg)
    if not isinstance(loss_config, dict):
        loss_config = dict(loss_config)
    return loss_config


def _ensure_base_loss(loss: Any, target: str | None) -> None:
    if not isinstance(loss, BaseLoss):
        error_msg = f"Loss {target!r} must be a subclass of 'BaseLoss', not {type(loss)}"
        raise TypeError(error_msg)
