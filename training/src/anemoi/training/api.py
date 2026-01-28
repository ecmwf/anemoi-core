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
from typing import Any

from anemoi.training.builders.graphs import build_graphs_from_config
from anemoi.training.utils.config import as_settings
from anemoi.training.utils.config import postprocess_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anemoi.training.config_types import Settings
    from anemoi.training.train.train import AnemoiTrainer


def normalize_config(
    config: Settings | Mapping[str, Any] | Any,
    *,
    validate: bool = True,
) -> Settings:
    """Normalize config into typed Settings.

    Parameters
    ----------
    config : Settings | Mapping[str, Any] | Any
        Configuration in Pydantic, Hydra, or plain mapping form.
    validate : bool, optional
        Whether to apply lightweight validation before returning it.
    """
    settings = as_settings(config)
    return postprocess_config(settings, validate=validate)


def build_trainer(
    config: Settings | Mapping[str, Any] | Any,
    *,
    wandb_logger: Any | None = None,
    mlflow_logger: Any | None = None,
    tensorboard_logger: Any | None = None,
    callbacks: list[Any] | None = None,
) -> AnemoiTrainer:
    """Create a trainer from a typed Settings config or mapping."""
    from anemoi.training.train.train import AnemoiTrainer

    return AnemoiTrainer(
        normalize_config(config),
        wandb_logger=wandb_logger,
        mlflow_logger=mlflow_logger,
        tensorboard_logger=tensorboard_logger,
        callbacks=callbacks,
    )


def train(
    config: Settings | Mapping[str, Any] | Any,
    *,
    wandb_logger: Any | None = None,
    mlflow_logger: Any | None = None,
    tensorboard_logger: Any | None = None,
    callbacks: list[Any] | None = None,
) -> None:
    """Run training using a typed Settings config or mapping."""
    build_trainer(
        config,
        wandb_logger=wandb_logger,
        mlflow_logger=mlflow_logger,
        tensorboard_logger=tensorboard_logger,
        callbacks=callbacks,
    ).train()


def build_graphs(
    config: Settings | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Build graph data for each dataset from a typed Settings config or mapping."""
    return build_graphs_from_config(normalize_config(config))


__all__ = ["build_graphs", "build_trainer", "normalize_config", "train"]
