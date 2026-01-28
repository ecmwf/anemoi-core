# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.training.builders.components import build_component
from anemoi.training.diagnostics.callbacks import get_callbacks

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import Callback

    from anemoi.training.config_types import ConfigBase


def build_config_callbacks_from_config(config: ConfigBase) -> list[Callback]:
    callbacks_cfg = config.diagnostics.callbacks
    if not callbacks_cfg:
        return []
    return [build_component(callback, config=config) for callback in callbacks_cfg]


def build_plot_callbacks_from_config(config: ConfigBase) -> list[Callback]:
    callbacks_cfg = config.diagnostics.plot.callbacks
    if not callbacks_cfg:
        return []
    return [build_component(callback, config=config) for callback in callbacks_cfg]


def build_progress_bar_from_config(config: ConfigBase) -> Callback | None:
    if not config.diagnostics.enable_progress_bar:
        return None
    return build_component(config.diagnostics.progress_bar)


def build_callbacks_from_config(config: ConfigBase) -> list[Callback]:
    callbacks = build_config_callbacks_from_config(config)
    plot_callbacks = build_plot_callbacks_from_config(config)
    progress_bar = build_progress_bar_from_config(config)
    return get_callbacks(
        config,
        callbacks=callbacks,
        plot_callbacks=plot_callbacks,
        progress_bar=progress_bar,
    )


__all__ = [
    "build_callbacks_from_config",
    "build_config_callbacks_from_config",
    "build_plot_callbacks_from_config",
    "build_progress_bar_from_config",
]
